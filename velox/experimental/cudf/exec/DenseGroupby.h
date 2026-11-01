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

#include "velox/core/PlanNode.h"
#include "velox/experimental/cudf/exec/DenseGroupbyDevice.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream_ref>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A single step aggregation grouped by one integer key whose values all lie
/// in [low, high]. Every driver of the pipeline adds its input into one set of
/// accumulators indexed by key - low, and the last driver to finish emits the
/// groups, so the input needs no partitioning by key. Only CudfGroupby runs
/// it; a CPU aggregation over the same node would see each driver's share of
/// every group.
class DenseAggregationNode : public core::AggregationNode {
 public:
  DenseAggregationNode(
      const core::PlanNodeId& id,
      const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
      const std::vector<std::string>& aggregateNames,
      const std::vector<Aggregate>& aggregates,
      core::PlanNodePtr source,
      int64_t low,
      int64_t high)
      : AggregationNode(
            id,
            Step::kSingle,
            groupingKeys,
            {},
            aggregateNames,
            aggregates,
            /*ignoreNullKeys=*/false,
            /*noGroupsSpanBatches=*/false,
            std::nullopt,
            std::move(source)),
        low_(low),
        high_(high) {}

  int64_t low() const {
    return low_;
  }

  int64_t high() const {
    return high_;
  }

 private:
  const int64_t low_;
  const int64_t high_;
};

/// Whether an aggregate can be accumulated densely: sum, avg, count, min or
/// max of an integer, decimal of up to 18 digits or floating point input,
/// without mask, distinct or ordering.
bool isDenseAggregate(const core::AggregationNode::Aggregate& aggregate);

/// The accumulators of one dense aggregation, shared by the drivers running
/// it.
class DenseGroupby {
 public:
  using Kind = dense::Kind;

  struct Aggregate {
    Kind kind;
    /// Input column, or -1 for count(*).
    int32_t channel;
    /// Null for count(*).
    TypePtr inputType;
    TypePtr resultType;
  };

  /// The aggregates of 'node' over input of 'inputType'.
  static std::vector<Aggregate> aggregates(
      const core::AggregationNode& node,
      const RowTypePtr& inputType);

  DenseGroupby(
      int64_t low,
      int64_t high,
      TypePtr keyType,
      std::vector<Aggregate> aggregates,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Adds the rows of 'input' to the accumulators. 'keyChannel' and the
  /// aggregates' channels index the columns of 'input'.
  void add(
      const std::vector<cudf::column_view>& input,
      int32_t keyChannel,
      cuda::stream_ref stream);

  /// The key and aggregate columns of the groups with rows.
  std::vector<std::unique_ptr<cudf::column>> release(
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// The instance for 'node' in 'task', created by the first caller.
  static std::shared_ptr<DenseGroupby> shared(
      const void* task,
      const core::PlanNodeId& node,
      const std::function<std::shared_ptr<DenseGroupby>()>& make);

  /// Forgets the instance for 'node' in 'task'.
  static void forget(const void* task, const core::PlanNodeId& node);

 private:
  struct State {
    Aggregate aggregate;
    /// Per slot: int128 sums of integers, doubles, or int64/double extremes.
    rmm::device_buffer values;
    /// Per slot count of non-null inputs.
    rmm::device_buffer counts;
    /// Whether 'values' holds doubles rather than integers.
    bool floating;
  };

  const int64_t low_;
  const int64_t slots_;
  const TypePtr keyType_;
  rmm::device_buffer rows_;
  rmm::device_buffer outOfRange_;
  std::vector<State> states_;
};

} // namespace facebook::velox::cudf_velox
