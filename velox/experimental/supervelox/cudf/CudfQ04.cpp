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
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

constexpr const char* kLate = R"(
#include <cuda/std/cstdint>

using namespace cuda::std;

extern "C" __device__ int transform(bool* late, int16_t commit, int16_t receipt) {
  *late = commit < receipt;
  return 0;
}
)";

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
  const uint32_t* starts = orderLineStarts(ctx);
  const auto bounds = ctx.download(starts + begin, end - begin + 1);
  const RowRange lines{bounds.front(), bounds.back()};

  const cudf::transform_input lateInputs[] = {
      fixedView(ctx, "lineitem", "l_commitdate", lines), fixedView(ctx, "lineitem", "l_receiptdate", lines)};
  const cudf::transform_output lateOutputs[] = {
      {cudf::data_type{cudf::type_id::BOOL8}, cudf::output_nullability::ALL_VALID}};
  auto late = cudf::transform_lto(
      compileFragment(kLate).view(),
      cudf::lto_binary_type::LTO_IR,
      cudf::null_aware::NO,
      std::nullopt,
      lateInputs,
      lateOutputs,
      {},
      std::nullopt,
      stream);

  const cudf::numeric_scalar<int32_t> base(static_cast<int32_t>(lines.begin), true, stream);
  auto offsets = cudf::binary_operation(
      deviceView(cudf::type_id::INT32, starts + begin, end - begin + 1),
      base,
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      stream);
  auto anyLate = cudf::segmented_reduce(
      late->get_column(0).view(),
      cudf::device_span<cudf::size_type const>(offsets->view().data<cudf::size_type>(), offsets->size()),
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
  const auto counts = sumGroups(
      ctx, sums, {fixedView(ctx, "orders", "o_orderpriority", {begin, end}), anyLate->view()});
  for (size_t i = 0; i < names.size(); ++i) {
    if (counts[i]) {
      result.strings[0].push_back(names[i]);
      result.ints[1].push_back(counts[i]);
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
