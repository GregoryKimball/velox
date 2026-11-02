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

#include "velox/experimental/cudf/exec/DenseGroupby.h"
#include "velox/type/Filter.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/udf.hpp>

#include <cuda/stream_ref>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A filter on one input column, as a table scan applies it.
struct ColumnFilter {
  std::string column;
  std::shared_ptr<const common::Filter> filter;
};

/// An input column passed as an integer narrower than the values it holds:
/// days of a date, the unscaled value of a decimal, or of a double with
/// 'scale' digits after the point. Expressions read it as 'type'.
struct Encoding {
  TypePtr type;
  int32_t scale{0};
};

/// Encodings of input columns by name.
using Encodings = std::unordered_map<std::string, Encoding>;

/// A dense aggregation that evaluates its filter and its aggregates' inputs
/// from the columns of its source in the same kernel that accumulates them,
/// rather than over the batches that filter and project nodes would
/// materialize. Keys and aggregates follow DenseAggregationNode; with no
/// keys it is a global aggregation, emitting one row even without input.
class FusedAggregationNode : public DenseAggregationNode {
 public:
  /// 'groupingKeys' are source columns. 'inputs' are the aggregates' input
  /// expressions over the source, null for count(*); the base node's
  /// aggregates take constants in their place. A row is aggregated when
  /// every one of 'filters' and 'columnFilters' passes. Source columns in
  /// 'encodings' arrive encoded.
  FusedAggregationNode(
      const core::PlanNodeId& id,
      const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
      const std::vector<std::string>& aggregateNames,
      const std::vector<Aggregate>& aggregates,
      core::PlanNodePtr source,
      std::vector<DenseKey> keys,
      RowTypePtr outputType,
      std::vector<core::TypedExprPtr> inputs,
      std::vector<core::TypedExprPtr> filters,
      std::vector<ColumnFilter> columnFilters,
      Encodings encodings = {});

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::TypedExprPtr>& inputs() const {
    return inputs_;
  }

  const std::vector<core::TypedExprPtr>& filters() const {
    return filters_;
  }

  const std::vector<ColumnFilter>& columnFilters() const {
    return columnFilters_;
  }

  const Encodings& encodings() const {
    return encodings_;
  }

 private:
  const RowTypePtr outputType_;
  const std::vector<core::TypedExprPtr> inputs_;
  const std::vector<core::TypedExprPtr> filters_;
  const std::vector<ColumnFilter> columnFilters_;
  const Encodings encodings_;
};

/// Why 'expr' over 'input' cannot be compiled into a fused aggregation, or
/// nullopt if it can.
std::optional<std::string> unfusable(
    const core::TypedExprPtr& expr,
    const RowTypePtr& input,
    const Encodings& encodings = {});

/// Why 'filter' on 'type' cannot be compiled, or nullopt if it can.
std::optional<std::string> unfusable(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<Encoding>& encoding = std::nullopt);

/// Why 'aggregate' of an input of 'type' cannot be accumulated by a fused
/// aggregation, or nullopt if it can.
std::optional<std::string> unfusable(
    const core::AggregationNode::Aggregate& aggregate,
    const TypePtr& type);

/// The bytes of partial state per input row block of a fused aggregation
/// with 'slots' groups and these aggregates.
int64_t fusedStateBytes(
    int64_t slots,
    const std::vector<core::AggregationNode::Aggregate>& aggregates,
    const std::vector<core::TypedExprPtr>& inputs);

/// The compiled reduction of one FusedAggregationNode.
class FusedReduction {
 public:
  FusedReduction(const FusedAggregationNode& node, const RowTypePtr& input);

  /// The aggregates the DenseGroupby that 'add' merges into must have.
  std::vector<DenseGroupby::Aggregate> aggregates() const {
    return aggregates_;
  }

  /// Reduces 'input', the columns of the node's source, and merges the
  /// partial state into 'into'.
  void add(
      const std::vector<cudf::column_view>& input,
      DenseGroupby& into,
      cuda::stream_ref stream);

  /// The generated CUDA source.
  const std::string& source() const {
    return source_;
  }

 private:
  /// A partial state value of one slot.
  struct Value {
    dense::MergeOp op;
    /// State field, two for kAddInt128.
    int32_t field;
  };

  std::vector<DenseGroupby::Aggregate> aggregates_;
  /// Input channels in callback parameter order.
  std::vector<int32_t> channels_;
  int64_t slots_;
  /// Fields per slot: the row count, then the values.
  int32_t fieldsPerSlot_;
  /// Per aggregate, its value, or nullopt for counts.
  std::vector<std::optional<Value>> values_;
  std::vector<cudf::data_type> stateTypes_;
  bool filtered_;
  std::string source_;
  const cudf::compiled_fragment* fragment_;
};

/// Two generated kernels over the columns of 'input', some of them encoded:
/// one computing which rows pass every one of 'filters' and
/// 'columnFilters', and one computing 'projections' of each row, decoding
/// what they read.
class FusedTransform {
 public:
  /// Why these cannot be generated, or nullopt if they can.
  static std::optional<std::string> unfusable(
      const RowTypePtr& input,
      const Encodings& encodings,
      const std::vector<core::TypedExprPtr>& filters,
      const std::vector<ColumnFilter>& columnFilters,
      const std::vector<core::TypedExprPtr>& projections);

  FusedTransform(
      const RowTypePtr& input,
      const Encodings& encodings,
      const std::vector<core::TypedExprPtr>& filters,
      const std::vector<ColumnFilter>& columnFilters,
      const std::vector<core::TypedExprPtr>& projections);

  bool filters() const {
    return maskFragment_ != nullptr;
  }

  /// BOOL8, true for the rows of 'input', the columns of 'input', that pass.
  std::unique_ptr<cudf::column> mask(
      const std::vector<cudf::column_view>& input,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  /// The input channels the projections read.
  const std::vector<int32_t>& projectChannels() const {
    return projectChannels_;
  }

  /// The projections of 'rows' rows of 'input', the columns of 'input' of
  /// which only those of projectChannels() are read, in order.
  std::vector<std::unique_ptr<cudf::column>> project(
      const std::vector<cudf::column_view>& input,
      cudf::size_type rows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  const std::string& source() const {
    return projectSource_;
  }

 private:
  std::vector<int32_t> maskChannels_;
  const cudf::compiled_fragment* maskFragment_{nullptr};
  std::vector<int32_t> projectChannels_;
  std::vector<cudf::data_type> outputTypes_;
  const cudf::compiled_fragment* projectFragment_{nullptr};
  std::string maskSource_;
  std::string projectSource_;
};

} // namespace facebook::velox::cudf_velox
