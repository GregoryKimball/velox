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
#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"l_orderkey", ResultType::kBigint},
    {"revenue", ResultType::kDecimal, 38, 4},
    {"o_orderdate", ResultType::kDate},
    {"o_shippriority", ResultType::kInteger},
};

} // namespace

/// The orders before the date whose lines can ship after it, each order's
/// revenue as a segmented sum over its lines, the market segment gathered by
/// customer key, then the ten best by sorting.
Result q3(QueryContext& ctx) {
  const int32_t date = days(1995, 3, 15);
  const auto lag = lineDateLag(ctx, "l_shipdate");
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", date - lag.second + 1, date);
  Result result(kSchema);
  if (end <= begin) {
    return result;
  }
  const OrderLines lines = orderLines(ctx, begin, end);
  auto revenue = computeColumn(
      ctx,
      "int16_t ship, int32_t price, int8_t discount",
      "ship > " + std::to_string(date) + " ? int64_t{price} * (100 - discount) : 0",
      cudf::type_id::INT64,
      {fixedView(ctx, "lineitem", "l_shipdate", lines.lines),
       fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
       fixedView(ctx, "lineitem", "l_discount", lines.lines)});
  auto perOrder = sumPerOrder(ctx, revenue->view(), lines);
  revenue.reset();
  auto customers = keyRows(ctx, fixedView(ctx, "orders", "o_custkey", {begin, end}));
  auto segment = gatherColumns(ctx, "customer", {"c_mktsegment"}, customers->view());
  auto score = computeColumn(
      ctx,
      "int64_t revenue, int8_t segment",
      "segment == " + std::to_string(ctx.meta("customer", "c_mktsegment").code("BUILDING")) + " ? revenue : 0",
      cudf::type_id::INT64,
      {perOrder->view(), segment->get_column(0).view()});
  const auto top = firstSorted(
      ctx,
      cudf::table_view({score->view(), fixedView(ctx, "orders", "o_orderdate", {begin, end})}),
      {cudf::order::DESCENDING, cudf::order::ASCENDING},
      10);
  const auto scores = download<int64_t>(ctx, score->view());
  const Column& orderkey = ctx.meta("orders", "o_orderkey");
  const Column& orderdate = ctx.meta("orders", "o_orderdate");
  const Column& shippriority = ctx.meta("orders", "o_shippriority");
  for (const int32_t index : top) {
    if (scores[index] == 0) {
      break;
    }
    const int64_t row = begin + index;
    result.ints[0].push_back(orderkey.value(row));
    result.decimals[1].push_back(scores[index]);
    result.ints[2].push_back(orderdate.value(row));
    result.ints[3].push_back(shippriority.value(row));
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
