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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"c_custkey", ResultType::kBigint},
    {"c_name", ResultType::kVarchar},
    {"revenue", ResultType::kDecimal, 38, 4},
    {"c_acctbal", ResultType::kDecimal, 15, 2},
    {"n_name", ResultType::kVarchar},
    {"c_address", ResultType::kVarchar},
    {"c_phone", ResultType::kVarchar},
    {"c_comment", ResultType::kVarchar},
};

} // namespace

/// Each order's revenue from returned lines as a segmented sum, grouped by
/// customer, then the twenty best by sorting.
Result q10(QueryContext& ctx) {
  constexpr int32_t kLimit = 20;
  constexpr int32_t kCandidates = 64;
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1993, 10, 1), days(1994, 1, 1));
  Result result(kSchema);
  if (end <= begin) {
    return result;
  }
  auto revenueOf = [&](int64_t first, int64_t last) -> std::unique_ptr<cudf::column> {
    if (useJit()) {
      return std::move(perOrderColumns(
                           ctx,
                           first,
                           last,
                           "",
                           {},
                           {{"lineitem", "l_returnflag", "flag"},
                            {"lineitem", "l_extendedprice", "price"},
                            {"lineitem", "l_discount", "discount"}},
                           {{cudf::type_id::INT64, "revenue"}},
                           "  int64_t total = 0;",
                           "    total += flag[l] == " + std::to_string(ctx.meta("lineitem", "l_returnflag").code("R")) +
                               " ? int64_t{price[l]} * (100 - discount[l]) : 0;",
                           "  *revenue = total;")
                           ->release()[0]);
    }
    const OrderLines lines = orderLines(ctx, first, last);
    auto revenue = computeColumn(
        ctx,
        "int8_t flag, int32_t price, int8_t discount",
        "flag == " + std::to_string(ctx.meta("lineitem", "l_returnflag").code("R")) +
            " ? int64_t{price} * (100 - discount) : 0",
        cudf::type_id::INT64,
        {fixedView(ctx, "lineitem", "l_returnflag", lines.lines),
         fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
         fixedView(ctx, "lineitem", "l_discount", lines.lines)});
    return sumPerOrder(ctx, revenue->view(), lines);
  };
  Grouped grouped;
  if (batched()) {
    // Batch by batch, each customer's revenue so far in a streaming groupby
    // sized to the customers.
    StreamingGroups groups({Aggregate::kSum}, ctx.rows("customer"));
    for (const auto& [first, last] : orderChunks(ctx, begin, end)) {
      auto perOrder = revenueOf(first, last);
      groups.add(ctx, cudf::table_view({fixedView(ctx, "orders", "o_custkey", {first, last}), perOrder->view()}));
    }
    grouped = groups.finish(ctx);
  } else {
    auto perOrder = revenueOf(begin, end);
    grouped = groupBy(
        ctx,
        cudf::table_view({fixedView(ctx, "orders", "o_custkey", {begin, end})}),
        {{perOrder->view(), Aggregate::kSum}});
  }
  const cudf::column_view customer = grouped.keys->get_column(0).view();
  const cudf::column_view sum = grouped.values[0]->view();
  std::vector<int64_t> customers;
  std::vector<int64_t> sums;
  if (useJit() && sum.size() > kCandidates) {
    // The best sums from top_k, ordered on the host; exact unless the last
    // candidate ties the twentieth.
    auto indices = cudf::top_k_order(sum, kCandidates, cudf::order::DESCENDING, ctx.stream());
    auto candidates =
        cudf::gather(cudf::table_view({customer, sum}), indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
    const auto keys = downloadInts(ctx, candidates->get_column(0).view());
    const auto values = download<int64_t>(ctx, candidates->get_column(1).view());
    std::vector<std::pair<int64_t, int64_t>> best;
    for (size_t i = 0; i < keys.size(); ++i) {
      best.emplace_back(-values[i], keys[i]);
    }
    std::sort(best.begin(), best.end());
    if (best[kLimit - 1].first != best.back().first) {
      for (int32_t i = 0; i < kLimit; ++i) {
        customers.push_back(best[i].second);
        sums.push_back(-best[i].first);
      }
    }
  }
  if (customers.empty()) {
    auto top = firstRows(
        ctx,
        cudf::table_view({customer, sum}),
        cudf::table_view({sum, customer}),
        {cudf::order::DESCENDING, cudf::order::ASCENDING},
        kLimit);
    customers = downloadInts(ctx, top->get_column(0).view());
    sums = download<int64_t>(ctx, top->get_column(1).view());
  }
  const Column& name = ctx.meta("customer", "c_name");
  const Column& acctbal = ctx.meta("customer", "c_acctbal");
  const Column& nation = ctx.meta("customer", "c_nationkey");
  const Column& address = ctx.meta("customer", "c_address");
  const Column& phone = ctx.meta("customer", "c_phone");
  const Column& comment = ctx.meta("customer", "c_comment");
  for (size_t i = 0; i < customers.size() && sums[i] > 0; ++i) {
    const int64_t row = customers[i] - 1;
    result.ints[0].push_back(customers[i]);
    result.strings[1].push_back(name.string(row));
    result.decimals[2].push_back(sums[i]);
    result.decimals[3].push_back(acctbal.value(row));
    result.strings[4].push_back(nationName(ctx, nation.value(row)));
    result.strings[5].push_back(address.string(row));
    result.strings[6].push_back(phone.string(row));
    result.strings[7].push_back(comment.string(row));
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
