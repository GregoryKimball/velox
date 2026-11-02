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
#include <functional>
#include <limits>
#include <tuple>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <fmt/format.h>

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
  const Column& orderkey = ctx.meta("orders", "o_orderkey");
  const Column& orderdate = ctx.meta("orders", "o_orderdate");
  const Column& shippriority = ctx.meta("orders", "o_shippriority");
  const int32_t firstDate = date - lag.second + 1;
  const int32_t building = ctx.meta("customer", "c_mktsegment").code("BUILDING");
  // The ten best of each batch, as (score, -date, row), merged into the ten
  // best overall.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> best;
  for (const auto& [first, last] : orderChunks(ctx, begin, end, std::numeric_limits<int64_t>::max())) {
    const cudf::column_view customer = fixedView(ctx, "orders", "o_custkey", {first, last});
    const cudf::column_view dates = fixedView(ctx, "orders", "o_orderdate", {first, last});
    std::unique_ptr<cudf::column> score;
    if (useJit()) {
      score = std::move(perOrderColumns(
                            ctx,
                            first,
                            last,
                            std::string(deviceType(customer.type().id())) + " customer",
                            {customer},
                            {{"lineitem", "l_shipdate", "ship"},
                             {"lineitem", "l_extendedprice", "price"},
                             {"lineitem", "l_discount", "discount"},
                             {"customer", "c_mktsegment", "segment"}},
                            {{cudf::type_id::INT64, "score"}},
                            "  int64_t revenue = 0;",
                            "    revenue += ship[l] > " + std::to_string(date) +
                                " ? int64_t{price[l]} * (100 - discount[l]) : 0;",
                            "  *score = segment[customer - 1] == " + std::to_string(building) + " ? revenue : 0;")
                            ->release()[0]);
    } else {
      const OrderLines lines = orderLines(ctx, first, last);
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
      auto customers = keyRows(ctx, customer);
      auto segment = gatherColumns(ctx, "customer", {"c_mktsegment"}, customers->view());
      score = computeColumn(
          ctx,
          "int64_t revenue, int8_t segment",
          "segment == " + std::to_string(building) + " ? revenue : 0",
          cudf::type_id::INT64,
          {perOrder->view(), segment->get_column(0).view()});
    }
    if (useJit() && date - firstDate < 4096) {
      // The order is (score desc, date asc): one key, the score above the
      // reversed date, whose top ten come from top_k.
      auto keyed = computeColumn(
          ctx,
          "int64_t score, int16_t date",
          fmt::format("score * 4096 + (4095 - (date - {}))", firstDate),
          cudf::type_id::INT64,
          {score->view(), dates});
      const auto count = static_cast<cudf::size_type>(std::min<int64_t>(10, last - first));
      auto indices = cudf::top_k_order(keyed->view(), count, cudf::order::DESCENDING, ctx.stream());
      auto keys = cudf::gather(
          cudf::table_view({keyed->view()}), indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
      const auto values = download<int64_t>(ctx, keys->get_column(0).view());
      const auto rows = download<int32_t>(ctx, indices->view());
      for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] >> 12 != 0) {
          best.emplace_back(values[i] >> 12, (values[i] & 4095) - 4095 - firstDate, first + rows[i]);
        }
      }
    } else {
      const auto top = firstSorted(
          ctx, cudf::table_view({score->view(), dates}), {cudf::order::DESCENDING, cudf::order::ASCENDING}, 10);
      const auto scores = download<int64_t>(ctx, score->view());
      for (const int32_t index : top) {
        if (scores[index] != 0) {
          best.emplace_back(scores[index], -orderdate.value(first + index), first + index);
        }
      }
    }
    std::sort(best.begin(), best.end(), std::greater<>());
    best.resize(std::min<size_t>(best.size(), 10));
  }
  for (const auto& [value, negativeDate, row] : best) {
    result.ints[0].push_back(orderkey.value(row));
    result.decimals[1].push_back(value);
    result.ints[2].push_back(-negativeDate);
    result.ints[3].push_back(shippriority.value(row));
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
