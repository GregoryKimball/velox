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

/// Validates and returns a flat ARRAY<REAL> embedding column.
const ArrayVector* validateFaissEmbeddings(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& name,
    int32_t dimension);

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
  std::optional<uint32_t> maxDrivers(
      const core::PlanNodePtr& node) override;
};

void registerFaiss();

} // namespace facebook::velox::faiss
