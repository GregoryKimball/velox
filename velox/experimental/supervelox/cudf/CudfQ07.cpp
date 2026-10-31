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
#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"supp_nation", ResultType::kVarchar},
    {"cust_nation", ResultType::kVarchar},
    {"l_year", ResultType::kBigint},
    {"revenue", ResultType::kDecimal, 38, 4},
};

} // namespace

/// Lines of the orders that can ship in 1995 or 1996, the customer's nation
/// repeated over them and the supplier's gathered, then one reduction into
/// (supplier nation, year) bins.
Result q7(QueryContext& ctx) {
  const int32_t dateLow = days(1995, 1, 1);
  const int32_t dateHigh = days(1996, 12, 31);
  const int32_t secondYear = days(1996, 1, 1);
  const auto lag = lineDateLag(ctx, "l_shipdate");
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", dateLow - lag.second, dateHigh - lag.first + 1);
  const std::string names[2] = {"FRANCE", "GERMANY"};
  const int32_t first = nationKey(ctx, names[0]);
  const int32_t second = nationKey(ctx, names[1]);
  GroupedSums sums;
  sums.inputs = "int8_t customer, int8_t supplier, int16_t date, int32_t price, int8_t discount";
  sums.filter = fmt::format(
      "date >= {0} && date <= {1} && ((customer == {2} && supplier == {3}) || (customer == {3} && supplier == {2}))",
      dateLow,
      dateHigh,
      first,
      second);
  sums.group = fmt::format("(supplier == {} ? 0 : 2) + (date >= {} ? 1 : 0)", first, secondYear);
  sums.groups = 4;
  sums.values = {"int64_t{price} * (100 - discount)"};
  std::vector<int64_t> totals(4, 0);
  for (const auto& [chunkBegin, chunkEnd] : orderChunks(ctx, begin, end)) {
    const OrderLines lines = orderLines(ctx, chunkBegin, chunkEnd);
    auto customers = keyRows(ctx, fixedView(ctx, "orders", "o_custkey", {chunkBegin, chunkEnd}));
    auto customerNation = gatherColumns(ctx, "customer", {"c_nationkey"}, customers->view());
    auto lineCustomer = perLine(ctx, lines, {customerNation->get_column(0).view()});
    auto suppliers = keyRows(ctx, fixedView(ctx, "lineitem", "l_suppkey", lines.lines));
    auto supplierNation = gatherColumns(ctx, "supplier", {"s_nationkey"}, suppliers->view());
    const auto part = sumGroups(
        ctx,
        sums,
        {lineCustomer->get_column(0).view(),
         supplierNation->get_column(0).view(),
         fixedView(ctx, "lineitem", "l_shipdate", lines.lines),
         fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
         fixedView(ctx, "lineitem", "l_discount", lines.lines)});
    for (int32_t bin = 0; bin < 4; ++bin) {
      totals[bin] += part[bin];
    }
  }
  Result result(kSchema);
  for (int32_t bin = 0; bin < 4; ++bin) {
    if (totals[bin] == 0) {
      continue;
    }
    const int32_t supplier = bin / 2;
    result.strings[0].push_back(names[supplier]);
    result.strings[1].push_back(names[1 - supplier]);
    result.ints[2].push_back(1995 + bin % 2);
    result.decimals[3].push_back(totals[bin]);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
