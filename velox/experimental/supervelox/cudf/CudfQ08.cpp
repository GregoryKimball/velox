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
    {"o_year", ResultType::kBigint},
    {"mkt_share", ResultType::kDouble},
};

} // namespace

/// Lines of the 1995 and 1996 orders, the customer's nation and the order
/// date repeated over them, the part type and supplier nation gathered, then
/// one reduction of all and Brazilian volume by year.
Result q8(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1995, 1, 1), days(1996, 12, 31) + 1);
  GroupedSums sums;
  sums.inputs = "int8_t customer, int16_t date, int16_t type, int8_t supplier, int32_t price, int8_t discount";
  sums.prelude = "  const int64_t volume = int64_t{price} * (100 - discount);\n";
  sums.filter = fmt::format(
      "(({}u >> customer) & 1) && type == {}",
      nationsInRegion(ctx, "AMERICA"),
      ctx.meta("part", "p_type").code("ECONOMY ANODIZED STEEL"));
  sums.group = fmt::format("date >= {} ? 1 : 0", days(1996, 1, 1));
  sums.groups = 2;
  sums.values = {"volume", fmt::format("supplier == {} ? volume : 0", nationKey(ctx, "BRAZIL"))};
  std::vector<int64_t> totals(4, 0);
  for (const auto& [first, last] : orderChunks(ctx, begin, end)) {
    const OrderLines lines = orderLines(ctx, first, last);
    auto customers = keyRows(ctx, fixedView(ctx, "orders", "o_custkey", {first, last}));
    auto customerNation = gatherColumns(ctx, "customer", {"c_nationkey"}, customers->view());
    auto perOrder = perLine(
        ctx, lines, {customerNation->get_column(0).view(), fixedView(ctx, "orders", "o_orderdate", {first, last})});
    auto parts = keyRows(ctx, fixedView(ctx, "lineitem", "l_partkey", lines.lines));
    auto partType = gatherColumns(ctx, "part", {"p_type"}, parts->view());
    parts.reset();
    auto suppliers = keyRows(ctx, fixedView(ctx, "lineitem", "l_suppkey", lines.lines));
    auto supplierNation = gatherColumns(ctx, "supplier", {"s_nationkey"}, suppliers->view());
    const auto part = sumGroups(
        ctx,
        sums,
        {perOrder->get_column(0).view(),
         perOrder->get_column(1).view(),
         partType->get_column(0).view(),
         supplierNation->get_column(0).view(),
         fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
         fixedView(ctx, "lineitem", "l_discount", lines.lines)});
    for (size_t i = 0; i < totals.size(); ++i) {
      totals[i] += part[i];
    }
  }
  Result result(kSchema);
  for (int32_t year = 0; year < 2; ++year) {
    if (totals[year * 2]) {
      result.ints[0].push_back(1995 + year);
      result.doubles[1].push_back(static_cast<double>(totals[year * 2 + 1]) / static_cast<double>(totals[year * 2]));
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
