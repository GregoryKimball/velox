/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/faiss/FaissOperators.h"

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "folly/ScopeGuard.h"

#include <cuda_runtime_api.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace facebook::velox::faiss {
namespace {

void checkCuda(cudaError_t error, const char* operation) {
  VELOX_USER_CHECK_EQ(
      static_cast<int>(error),
      static_cast<int>(cudaSuccess),
      "{} failed: {}",
      operation,
      cudaGetErrorString(error));
}

void copyInt64Column(
    cudf::column_view column,
    std::vector<int64_t>& output,
    rmm::cuda_stream_view stream,
    const char* name) {
  VELOX_USER_CHECK(
      column.type().id() == cudf::type_id::INT64, "{} must be BIGINT", name);
  VELOX_USER_CHECK(!column.has_nulls(), "{} contains nulls", name);
  output.resize(column.size());
  if (!output.empty()) {
    checkCuda(
        cudaMemcpyAsync(
            output.data(),
            column.data<int64_t>(),
            output.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream.value()),
        name);
  }
}

class GpuAssignClustersOperator final
    : public cudf_velox::CudfOperatorBase {
 public:
  GpuAssignClustersOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const AssignClustersNode> planNode)
      : CudfOperatorBase(
            operatorId,
            driverCtx,
            planNode->outputType(),
            planNode->id(),
            "FaissGpuAssignClusters",
            nvtx3::rgb{147, 112, 219},
            cudf_velox::NvtxMethodFlag::kAddInput |
                cudf_velox::NvtxMethodFlag::kGetOutput),
        planNode_(std::move(planNode)) {}

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 protected:
  void doAddInput(RowVectorPtr input) override {
    VELOX_CHECK_NULL(input_);
    input_ = std::move(input);
  }

  RowVectorPtr doGetOutput() override {
    if (!input_) {
      return nullptr;
    }
    auto cudfInput =
        std::dynamic_pointer_cast<cudf_velox::CudfVector>(input_);
    VELOX_USER_CHECK_NOT_NULL(
        cudfInput, "GPU FAISS cluster assignment requires CudfVector input");

    const auto stream = cudfInput->stream();
    const auto table = cudfInput->getTableView();
    const auto embedding =
        table.column(planNode_->sources()[0]->outputType()->getChildIdx(
            planNode_->embeddingColumn()));
    VELOX_USER_CHECK(
        embedding.type().id() == cudf::type_id::LIST,
        "FAISS embedding column must be ARRAY<REAL>");
    VELOX_USER_CHECK(
        !embedding.has_nulls(), "FAISS embedding column contains nulls");
    cudf::lists_column_view lists(embedding);
    const auto values = lists.get_sliced_child(stream);
    VELOX_USER_CHECK(
        values.type().id() == cudf::type_id::FLOAT32,
        "FAISS embedding child must be REAL");
    VELOX_USER_CHECK(
        !values.has_nulls(), "FAISS embedding child contains nulls");
    VELOX_USER_CHECK_EQ(
        values.size(),
        input_->size() * planNode_->dimension(),
        "FAISS embedding dimension mismatch");

    std::vector<cudf::size_type> offsets(input_->size() + 1);
    if (!offsets.empty()) {
      checkCuda(
          cudaMemcpyAsync(
              offsets.data(),
              lists.offsets_begin(),
              offsets.size() * sizeof(cudf::size_type),
              cudaMemcpyDeviceToHost,
              stream.value()),
          "copying FAISS list offsets");
      stream.synchronize();
    }
    for (vector_size_t row = 0; row < input_->size(); ++row) {
      VELOX_USER_CHECK_EQ(
          offsets[row + 1] - offsets[row],
          planNode_->dimension(),
          "FAISS embedding dimension mismatch at row {}",
          row);
    }

    const auto centroidCount =
        planNode_->centroids().size() / planNode_->dimension();
    rmm::device_uvector<float> centroids(
        planNode_->centroids().size(), stream);
    checkCuda(
        cudaMemcpyAsync(
            centroids.data(),
            planNode_->centroids().data(),
            planNode_->centroids().size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream.value()),
        "copying FAISS centroids");

    rmm::device_uvector<int64_t> clusterIds(input_->size(), stream);
    rmm::device_uvector<float> distances(input_->size(), stream);
    raft::device_resources resources;
    raft::resource::set_cuda_stream(resources, stream);
    cuvs::neighbors::brute_force::index_params indexParams;
    indexParams.metric = planNode_->metric() == FaissMetric::kL2
        ? cuvs::distance::DistanceType::L2Expanded
        : cuvs::distance::DistanceType::InnerProduct;
    const auto index = cuvs::neighbors::brute_force::build(
        resources,
        indexParams,
        raft::make_device_matrix_view<const float, int64_t>(
            centroids.data(),
            static_cast<int64_t>(centroidCount),
            planNode_->dimension()));
    cuvs::neighbors::brute_force::search_params searchParams;
    cuvs::neighbors::brute_force::search(
        resources,
        searchParams,
        index,
        raft::make_device_matrix_view<const float, int64_t>(
            values.data<float>(), input_->size(), planNode_->dimension()),
        raft::make_device_matrix_view<int64_t, int64_t>(
            clusterIds.data(), input_->size(), 1),
        raft::make_device_matrix_view<float, int64_t>(
            distances.data(), input_->size(), 1));

    auto clusterColumn = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        input_->size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        cudf_velox::get_output_mr());
    if (input_->size() > 0) {
      checkCuda(
          cudaMemcpyAsync(
              clusterColumn->mutable_view().data<int64_t>(),
              clusterIds.data(),
              input_->size() * sizeof(int64_t),
              cudaMemcpyDeviceToDevice,
              stream.value()),
          "copying FAISS cluster IDs");
    }

    auto pool = cudfInput->pool();
    const auto size = cudfInput->size();
    auto columns = cudfInput->release()->release();
    columns.push_back(std::move(clusterColumn));
    input_.reset();
    return std::make_shared<cudf_velox::CudfVector>(
        pool,
        outputType_,
        size,
        std::make_unique<cudf::table>(std::move(columns)),
        stream);
  }

 private:
  std::shared_ptr<const AssignClustersNode> planNode_;
};

} // namespace

std::unique_ptr<exec::Operator> makeFaissGpuAssignClusters(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const AssignClustersNode> planNode) {
  return std::make_unique<GpuAssignClustersOperator>(
      operatorId, driverCtx, std::move(planNode));
}

std::optional<FaissGpuQueryInput> extractFaissGpuQueryInput(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& queryIdColumn,
    const std::string& embeddingColumn,
    const std::optional<std::string>& clusterColumn,
    int32_t dimension,
    bool copyEmbeddingsToHost) {
  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  if (!cudfInput) {
    return std::nullopt;
  }

  const auto table = cudfInput->getTableView();
  const auto stream = cudfInput->stream();
  const auto embedding = table.column(type->getChildIdx(embeddingColumn));
  VELOX_USER_CHECK(
      embedding.type().id() == cudf::type_id::LIST,
      "FAISS embedding column must be ARRAY<REAL>");
  VELOX_USER_CHECK(
      !embedding.has_nulls(), "FAISS embedding column contains nulls");

  cudf::lists_column_view lists(embedding);
  const auto child = lists.get_sliced_child(stream);
  VELOX_USER_CHECK(
      child.type().id() == cudf::type_id::FLOAT32,
      "FAISS embedding child must be REAL");
  VELOX_USER_CHECK(!child.has_nulls(), "FAISS embedding child contains nulls");
  VELOX_USER_CHECK_EQ(
      child.size(),
      input->size() * dimension,
      "FAISS embedding dimension mismatch");

  std::vector<cudf::size_type> offsets(input->size() + 1);
  if (!offsets.empty()) {
    checkCuda(
        cudaMemcpyAsync(
            offsets.data(),
            lists.offsets_begin(),
            offsets.size() * sizeof(cudf::size_type),
            cudaMemcpyDeviceToHost,
            stream.value()),
        "copying FAISS list offsets");
  }

  FaissGpuQueryInput result;
  result.embeddings = child.data<float>();
  result.stream = reinterpret_cast<uintptr_t>(stream.value());
  copyInt64Column(
      table.column(type->getChildIdx(queryIdColumn)),
      result.queryIds,
      stream,
      "FAISS query ID column");
  if (clusterColumn) {
    result.clusters.emplace();
    copyInt64Column(
        table.column(type->getChildIdx(*clusterColumn)),
        *result.clusters,
        stream,
        "FAISS cluster ID column");
  }
  stream.synchronize();

  for (vector_size_t row = 0; row < input->size(); ++row) {
    VELOX_USER_CHECK_EQ(
        offsets[row + 1] - offsets[row],
        dimension,
        "FAISS embedding dimension mismatch at row {}",
        row);
  }
  if (copyEmbeddingsToHost && child.size() > 0) {
    result.ownedEmbeddings.resize(child.size());
    checkCuda(
        cudaMemcpyAsync(
            result.ownedEmbeddings.data(),
            result.embeddings,
            result.ownedEmbeddings.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream.value()),
        "copying FAISS query embeddings");
    stream.synchronize();
    result.embeddings = result.ownedEmbeddings.data();
    result.stream = 0;
  }
  return result;
}

std::optional<FaissGpuBuildInput> extractFaissGpuBuildInput(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& idColumn,
    const std::string& embeddingColumn,
    const std::optional<std::string>& clusterColumn,
    int32_t dimension) {
  auto deviceInput = extractFaissGpuQueryInput(
      input, type, idColumn, embeddingColumn, clusterColumn, dimension);
  if (!deviceInput) {
    return std::nullopt;
  }

  FaissGpuBuildInput result;
  result.embeddings = deviceInput->embeddings;
  result.rowCount = input->size();
  result.documentIds = std::move(deviceInput->queryIds);
  result.clusters = std::move(deviceInput->clusters);
  result.stream = deviceInput->stream;
  result.owner = input;
  return result;
}

void searchFaissGpuRows(
    FaissIndexState& state,
    FaissClusterIndex& cluster,
    const FaissGpuQueryInput& input,
    const std::vector<vector_size_t>& rows,
    int32_t dimension,
    int32_t topK,
    std::vector<float>& distances,
    std::vector<::faiss::idx_t>& labels) {
  VELOX_CHECK(!rows.empty());
  distances.resize(rows.size() * topK);
  labels.resize(rows.size() * topK);

  vector_size_t offset = 0;
  const bool contiguous = std::all_of(
      rows.begin(), rows.end(), [&](vector_size_t row) {
        return row == rows.front() + offset++;
      });
  const float* queries = input.embeddings + rows.front() * dimension;
  float* gathered = nullptr;
  auto freeGathered = folly::makeGuard([&] {
    if (gathered) {
      cudaFree(gathered);
    }
  });
  if (!contiguous) {
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&gathered),
            rows.size() * dimension * sizeof(float)),
        "allocating routed FAISS queries");
    const auto stream = reinterpret_cast<cudaStream_t>(input.stream);
    for (size_t index = 0; index < rows.size(); ++index) {
      checkCuda(
          cudaMemcpyAsync(
              gathered + index * dimension,
              input.embeddings + rows[index] * dimension,
              dimension * sizeof(float),
              cudaMemcpyDeviceToDevice,
              stream),
          "gathering routed FAISS queries");
    }
    queries = gathered;
  }
  searchFaissIndex(
      state,
      cluster,
      rows.size(),
      queries,
      topK,
      distances.data(),
      labels.data(),
      input.stream);
}

} // namespace facebook::velox::faiss
