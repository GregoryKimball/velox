/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/faiss/FaissOperators.h"

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cuda_runtime_api.h>
#include <cudf/lists/lists_column_view.hpp>

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

} // namespace

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

std::optional<FaissGpuBuildInput> copyFaissGpuBuildInput(
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
  result.documentIds = std::move(deviceInput->queryIds);
  result.clusters = std::move(deviceInput->clusters);
  result.embeddings.resize(input->size() * dimension);
  if (!result.embeddings.empty()) {
    const auto stream = reinterpret_cast<cudaStream_t>(deviceInput->stream);
    checkCuda(
        cudaMemcpyAsync(
            result.embeddings.data(),
            deviceInput->embeddings,
            result.embeddings.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
        "copying FAISS build embeddings");
    checkCuda(
        cudaStreamSynchronize(stream), "synchronizing FAISS build embeddings");
  }
  return result;
}

} // namespace facebook::velox::faiss
