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
#include <algorithm>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"s_suppkey", ResultType::kBigint},
    {"s_name", ResultType::kVarchar},
    {"s_address", ResultType::kVarchar},
    {"s_phone", ResultType::kVarchar},
    {"total_revenue", ResultType::kDecimal, 38, 4},
};

} // namespace

/// Revenue per supplier over the shipdate zones of the quarter, grouped by
/// supplier key; the suppliers at the maximum are read on the host.
Result q15(QueryContext& ctx) {
  const int32_t low = days(1996, 1, 1);
  const int32_t high = days(1996, 4, 1);
  const RowRange rows = splitZones(ctx, "lineitem", "l_shipdate", low, high - 1).some;
  Result result(kSchema);
  if (rows.size() == 0) {
    return result;
  }
  // The lines of `batch` shipped in the quarter, with their revenue.
  auto shippedLines = [&](RowRange batch) {
    auto revenue = computeColumn(
        ctx,
        "int16_t ship, int32_t price, int8_t discount",
        "ship >= " + std::to_string(low) + " && ship < " + std::to_string(high) +
            " ? int64_t{price} * (100 - discount) : 0",
        cudf::type_id::INT64,
        {fixedView(ctx, "lineitem", "l_shipdate", batch),
         fixedView(ctx, "lineitem", "l_extendedprice", batch),
         fixedView(ctx, "lineitem", "l_discount", batch)});
    const cudf::numeric_scalar<int64_t> zero(0, true, ctx.stream());
    auto shipped = cudf::binary_operation(
        revenue->view(), zero, cudf::binary_operator::NOT_EQUAL, cudf::data_type{cudf::type_id::BOOL8}, ctx.stream());
    return selectRows(
        ctx, cudf::table_view({fixedView(ctx, "lineitem", "l_suppkey", batch), revenue->view()}), shipped->view());
  };
  Grouped grouped;
  if (batched()) {
    // Each batch's revenue per supplier, added to that so far in a
    // streaming groupby sized to the suppliers.
    StreamingGroups groups({Aggregate::kSum}, ctx.rows("supplier"));
    forEachBatch(ctx, rows, rows.size(), [&](RowRange batch) {
      auto kept = shippedLines(batch);
      auto part = groupBy(
          ctx, cudf::table_view({kept->get_column(0).view()}), {{kept->get_column(1).view(), Aggregate::kSum}});
      groups.add(ctx, cudf::table_view({part.keys->get_column(0).view(), part.values[0]->view()}));
    });
    grouped = groups.finish(ctx);
    if (!grouped.keys) {
      return result;
    }
  } else {
    // Only the lines shipped in the quarter go to groupby.
    auto kept = shippedLines(rows);
    grouped = groupBy(
        ctx, cudf::table_view({kept->get_column(0).view()}), {{kept->get_column(1).view(), Aggregate::kSum}});
  }
  const cudf::column_view sums = grouped.values[0]->view();
  auto maximum = cudf::reduce(
      sums, *cudf::make_max_aggregation<cudf::reduce_aggregation>(), sums.type(), ctx.stream());
  const int64_t best = static_cast<cudf::numeric_scalar<int64_t>&>(*maximum).value(ctx.stream());
  if (best <= 0) {
    return result;
  }
  auto atBest = cudf::binary_operation(
      sums, *maximum, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8}, ctx.stream());
  auto top = selectRows(ctx, grouped.keys->view(), atBest->view());
  auto winners = downloadInts(ctx, top->get_column(0).view());
  std::sort(winners.begin(), winners.end());
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  for (const int64_t supplier : winners) {
    const int64_t row = supplier - 1;
    result.ints[0].push_back(supplier);
    result.strings[1].push_back(name.string(row));
    result.strings[2].push_back(address.string(row));
    result.strings[3].push_back(phone.string(row));
    result.decimals[4].push_back(best);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
