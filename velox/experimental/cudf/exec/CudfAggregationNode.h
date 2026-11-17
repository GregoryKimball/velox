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

namespace facebook::velox::cudf_velox {

/// An aggregation whose marked aggregates exchange a GPU-only intermediate
/// state.
///
/// SUM over a short decimal column normally carries a 32-byte VARBINARY state
/// for each group (sum, count, overflow) plus a 4-byte offset. Here the state
/// of a marked aggregate is one DECIMAL(38, scale) column: 16 bytes for each
/// group and no offsets. The state is lossless. A sum of fewer than 2^63
/// values below 10^18 in magnitude is below 9.3 * 10^36, which fits the
/// 128-bit integer and the DECIMAL(38, scale) result type, so no overflow word
/// is needed. A NULL state row stands for a group without non-NULL input, so
/// no count is needed.
///
/// The node has the shape of a core::AggregationNode but must not run as a
/// CPU aggregation: the CPU sum function reads and writes the VARBINARY state.
/// The cuDF driver adapter fails the query instead of falling back.
class CudfAggregationNode final : public core::AggregationNode {
 public:
  /// 'node' already has the calls and the source of the compact plan. Each
  /// entry of 'compactIntermediateTypes' belongs to the aggregate at the same
  /// position; nullptr keeps that aggregate's ordinary intermediate type.
  CudfAggregationNode(
      const core::AggregationNode& node,
      std::vector<TypePtr> compactIntermediateTypes);

  /// Returns true if the aggregate at 'index' exchanges the compact state.
  bool usesCompactDecimalSum(size_t index) const {
    return compactIntermediateTypes_.at(index) != nullptr;
  }

  /// Returns the compact state type of the aggregate at 'index', or nullptr if
  /// the aggregate keeps its ordinary intermediate type.
  const TypePtr& intermediateType(size_t index) const {
    return compactIntermediateTypes_.at(index);
  }

  /// Returns DECIMAL(38, scale of the input) if 'aggregate' is a plain SUM
  /// over one short decimal input column, and nullptr otherwise. A DISTINCT,
  /// masked or ordered aggregate and one over an expression are not eligible.
  static TypePtr compactIntermediateType(const Aggregate& aggregate);

  /// Rewrites the grouped FINAL <- PARTIAL chain that ends at
  /// 'finalAggregation', with or without one LocalPartition in between, to
  /// exchange the compact state for its eligible decimal SUMs. Every other
  /// aggregate, the grouping keys, the node ids and the FINAL output type stay
  /// as they are.
  ///
  /// Returns 'finalAggregation' itself if cuDF is not registered or disabled,
  /// if the chain does not have exactly that shape, or if it has no eligible
  /// aggregate.
  static core::PlanNodePtr rewriteFinalAggregation(
      const core::PlanNodePtr& finalAggregation);

  /// Applies rewriteFinalAggregation() to every aggregation of 'plan'. Nodes
  /// above a rewritten chain are copied to read it: a rewritten chain has the
  /// output type of the one it replaces, so nothing else about them changes. A
  /// chain below a node this function cannot copy is left as it is.
  ///
  /// Returns 'plan' itself if nothing was rewritten.
  static core::PlanNodePtr rewritePlan(const core::PlanNodePtr& plan);

  /// Returns false: the CPU spill path serializes the ordinary state.
  bool canSpill(const core::QueryConfig& queryConfig) const override;

  /// Throws: a deserialized core::AggregationNode would drop the compact
  /// state contract.
  folly::dynamic serialize() const override;

 private:
  // One entry for each aggregate. See intermediateType().
  const std::vector<TypePtr> compactIntermediateTypes_;
};

} // namespace facebook::velox::cudf_velox
