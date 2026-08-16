/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#pragma once

#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/faiss/FaissIndex.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::faiss {

struct FaissGpuQueryInput {
  const float* embeddings{nullptr};
  std::vector<float> ownedEmbeddings;
  std::vector<int64_t> queryIds;
  std::optional<std::vector<int64_t>> clusters;
  uintptr_t stream{0};
};

struct FaissGpuBuildInput {
  const float* embeddings{nullptr};
  vector_size_t rowCount{0};
  std::vector<int64_t> documentIds;
  std::optional<std::vector<int64_t>> clusters;
  uintptr_t stream{0};
  // Retains the CudfVector and its device buffers until the index build is
  // complete.
  RowVectorPtr owner;
};

/// Validates and returns a flat ARRAY<REAL> embedding column.
const ArrayVector* validateFaissEmbeddings(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& name,
    int32_t dimension);

#if defined(VELOX_ENABLE_FAISS_GPU)
std::unique_ptr<exec::Operator> makeFaissGpuAssignClusters(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const AssignClustersNode> planNode);

std::optional<FaissGpuQueryInput> extractFaissGpuQueryInput(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& queryIdColumn,
    const std::string& embeddingColumn,
    const std::optional<std::string>& clusterColumn,
    int32_t dimension,
    bool copyEmbeddingsToHost = false);

std::optional<FaissGpuBuildInput> extractFaissGpuBuildInput(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& idColumn,
    const std::string& embeddingColumn,
    const std::optional<std::string>& clusterColumn,
    int32_t dimension);

void searchFaissGpuRows(
    FaissIndexState& state,
    FaissClusterIndex& cluster,
    const FaissGpuQueryInput& input,
    const std::vector<vector_size_t>& rows,
    int32_t dimension,
    int32_t topK,
    std::vector<float>& distances,
    std::vector<::faiss::idx_t>& labels);
#endif

class FaissIndexBridge final : public exec::JoinBridge {
 public:
  void setState(std::shared_ptr<FaissIndexState> state);
  std::shared_ptr<FaissIndexState> stateOrFuture(ContinueFuture* future);

 private:
  std::shared_ptr<FaissIndexState> state_;
};

class FaissPlanNodeTranslator final
    : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;
  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override;
  exec::OperatorSupplier toOperatorSupplier(
      const core::PlanNodePtr& node) override;
  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override;
};

void registerFaiss();

} // namespace facebook::velox::faiss
