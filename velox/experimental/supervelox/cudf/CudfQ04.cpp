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
#include <limits>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"o_orderpriority", ResultType::kVarchar},
    {"order_count", ResultType::kBigint},
};

} // namespace

/// The orders of the date range are contiguous, and so are their lines.
/// Each line's lateness, then ANY over each order's lines as a segmented
/// reduction, then one reduction counting late orders by priority.
Result q4(QueryContext& ctx) {
  const auto stream = ctx.stream();
  const auto [begin, end] = sortedRange(
      ctx,
      "orders",
      "o_orderdate",
      ctx.param("orderdate_min", days(1993, 7, 1)),
      ctx.param("orderdate_max", days(1993, 10, 1) - 1) + 1);
  const auto& names = ctx.meta("orders", "o_orderpriority").dictionary;
  Result result(kSchema);
  if (end <= begin) {
    return result;
  }
  std::vector<int64_t> counts(names.size(), 0);
  for (const auto& [first, last] : orderChunks(ctx, begin, end, std::numeric_limits<int64_t>::max())) {
    std::vector<int64_t> part;
    if (useJit()) {
      // One reduction over the orders, each scanning its lines until one is
      // late.
      GroupedSums sums;
      sums.inputs = "int8_t priority";
      sums.arrays = withLines(
          orderBatch(ctx, first, last), {{"lineitem", "l_commitdate", "commit"}, {"lineitem", "l_receiptdate", "receipt"}});
      sums.prelude =
          "  bool late = false;\n"
          "  for (int64_t l = starts[row]; l < starts[row + 1] && !late; ++l) late = commit[l] < receipt[l];\n";
      sums.filter = "late";
      sums.group = "priority";
      sums.groups = static_cast<int32_t>(names.size());
      sums.values = {"1"};
      part = sumGroups(ctx, sums, {fixedView(ctx, "orders", "o_orderpriority", {first, last})});
    } else {
      const OrderLines lines = orderLines(ctx, first, last);
      auto late = computeColumn(
          ctx,
          "int16_t commit, int16_t receipt",
          "commit < receipt",
          cudf::type_id::BOOL8,
          {fixedView(ctx, "lineitem", "l_commitdate", lines.lines),
           fixedView(ctx, "lineitem", "l_receiptdate", lines.lines)});
      auto anyLate = cudf::segmented_reduce(
          late->view(),
          cudf::device_span<cudf::size_type const>(lines.offsets->view().data<cudf::size_type>(), lines.offsets->size()),
          *cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>(),
          cudf::data_type{cudf::type_id::BOOL8},
          cudf::null_policy::EXCLUDE,
          stream);
      GroupedSums sums;
      sums.inputs = "int8_t priority, bool late";
      sums.filter = "late";
      sums.group = "priority";
      sums.groups = static_cast<int32_t>(names.size());
      sums.values = {"1"};
      part = sumGroups(ctx, sums, {fixedView(ctx, "orders", "o_orderpriority", {first, last}), anyLate->view()});
    }
    for (size_t i = 0; i < names.size(); ++i) {
      counts[i] += part[i];
    }
  }
  for (size_t i = 0; i < names.size(); ++i) {
    if (counts[i]) {
      result.strings[0].push_back(names[i]);
      result.ints[1].push_back(counts[i]);
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
