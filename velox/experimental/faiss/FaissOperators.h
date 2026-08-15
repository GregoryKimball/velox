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
  std::vector<float> embeddings;
  std::vector<int64_t> documentIds;
  std::optional<std::vector<int64_t>> clusters;
};

/// Validates and returns a flat ARRAY<REAL> embedding column.
const ArrayVector* validateFaissEmbeddings(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& name,
    int32_t dimension);

#if defined(VELOX_ENABLE_FAISS_GPU)
std::optional<FaissGpuQueryInput> extractFaissGpuQueryInput(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& queryIdColumn,
    const std::string& embeddingColumn,
    const std::optional<std::string>& clusterColumn,
    int32_t dimension,
    bool copyEmbeddingsToHost = false);

std::optional<FaissGpuBuildInput> copyFaissGpuBuildInput(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& idColumn,
    const std::string& embeddingColumn,
    const std::optional<std::string>& clusterColumn,
    int32_t dimension);
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
