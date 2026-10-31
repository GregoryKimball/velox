/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <memory>

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/supervelox/queries/Query.h"

namespace facebook::velox::supervelox {

/// Process-wide GPU state shared by fused operators: the pinned column store,
/// the HBM cache and the stream queries run on.
class Engine {
 public:
  struct Run {
    double gpuMs{0};
    DeviceCache::Stats cache;
  };

  Engine(const std::string& dataDirectory, size_t cacheBytes, size_t stageBytes);
  ~Engine();

  static void setInstance(Engine* engine);
  static Engine& instance();

  Result run(const QuerySpec& spec);

  const Run& lastRun() const {
    return lastRun_;
  }

  ColumnStore& store() {
    return store_;
  }

  DeviceCache& cache() {
    return cache_;
  }

 private:
  ColumnStore store_;
  DeviceCache cache_;
  cudaStream_t stream_;
  Run lastRun_;
};

RowTypePtr resultRowType(const std::vector<ResultField>& schema);

/// Leaf plan node standing for the fused GPU part of a TPC-H query. Velox
/// operators above it finish the query on its small result.
class FusedQueryNode : public core::PlanNode {
 public:
  FusedQueryNode(const core::PlanNodeId& id, int32_t query);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    static const std::vector<core::PlanNodePtr> kNoSources;
    return kNoSources;
  }

  std::string_view name() const override {
    return "FusedQuery";
  }

  const QuerySpec& spec() const {
    return *spec_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const QuerySpec* spec_;
  const RowTypePtr outputType_;
};

class FusedQueryOperator : public exec::SourceOperator {
 public:
  FusedQueryOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const FusedQueryNode>& node);

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 private:
  const QuerySpec& spec_;
  bool finished_{false};
};

class FusedQueryTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node) override;

  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override;
};

/// Registers the translator for FusedQueryNode.
void registerFusedQuery();

} // namespace facebook::velox::supervelox
