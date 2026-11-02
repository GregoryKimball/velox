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

#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

constexpr int32_t kNations = 25;

const std::vector<ResultField> kSchema = {
    {"n_name", ResultType::kVarchar},
    {"revenue", ResultType::kDecimal, 38, 4},
};

} // namespace

/// The customer's nation gathered by key and repeated over the order's
/// lines, the supplier's gathered by key, then one reduction by nation.
Result q5(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1994, 1, 1), days(1995, 1, 1));
  const uint32_t nations = nationsInRegion(ctx, "ASIA");
  GroupedSums sums;
  sums.inputs = "int8_t customer, int8_t supplier, int32_t price, int8_t discount";
  sums.filter = "((" + std::to_string(nations) + "u >> customer) & 1) && supplier == customer";
  sums.group = "customer";
  sums.groups = kNations;
  sums.values = {"int64_t{price} * (100 - discount)"};
  std::vector<int64_t> totals(kNations, 0);
  const int64_t maxLines = useJit() ? std::numeric_limits<int64_t>::max() : int64_t{1} << 28;
  for (const auto& [first, last] : orderChunks(ctx, begin, end, maxLines)) {
    if (useJit()) {
      // One reduction over the orders, each summing its lines from suppliers
      // of the customer's nation.
      const cudf::column_view customer = fixedView(ctx, "orders", "o_custkey", {first, last});
      GroupedSums perOrder;
      perOrder.inputs = fmt::format("{} customer", deviceType(customer.type().id()));
      perOrder.arrays = withLines(
          orderBatch(ctx, first, last),
          {{"customer", "c_nationkey", "customerNation"},
           {"lineitem", "l_suppkey", "suppkey"},
           {"lineitem", "l_extendedprice", "price"},
           {"lineitem", "l_discount", "discount"},
           {"supplier", "s_nationkey", "supplierNation"}});
      perOrder.prelude = fmt::format(
          "  const int nation = customerNation[customer - 1];\n"
          "  int64_t revenue = 0;\n"
          "  if (({}u >> nation) & 1) {{\n"
          "    for (int64_t l = starts[row]; l < starts[row + 1]; ++l) {{\n"
          "      revenue += supplierNation[suppkey[l] - 1] == nation ? int64_t{{price[l]}} * (100 - discount[l]) : 0;\n"
          "    }}\n"
          "  }}\n",
          nations);
      perOrder.filter = "true";
      perOrder.group = "nation";
      perOrder.groups = kNations;
      perOrder.values = {"revenue"};
      const auto part = sumGroups(ctx, perOrder, {customer});
      for (int32_t n = 0; n < kNations; ++n) {
        totals[n] += part[n];
      }
      continue;
    }
    const OrderLines lines = orderLines(ctx, first, last);
    auto customers = keyRows(ctx, fixedView(ctx, "orders", "o_custkey", {first, last}));
    auto customerNation = gatherColumns(ctx, "customer", {"c_nationkey"}, customers->view());
    auto lineCustomer = perLine(ctx, lines, {customerNation->get_column(0).view()});
    auto suppliers = keyRows(ctx, fixedView(ctx, "lineitem", "l_suppkey", lines.lines));
    auto supplierNation = gatherColumns(ctx, "supplier", {"s_nationkey"}, suppliers->view());
    const auto part = sumGroups(
        ctx,
        sums,
        {lineCustomer->get_column(0).view(),
         supplierNation->get_column(0).view(),
         fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
         fixedView(ctx, "lineitem", "l_discount", lines.lines)});
    for (int32_t n = 0; n < kNations; ++n) {
      totals[n] += part[n];
    }
  }
  Result result(kSchema);
  for (int32_t n = 0; n < kNations; ++n) {
    if ((nations >> n) & 1) {
      result.strings[0].push_back(nationName(ctx, n));
      result.decimals[1].push_back(totals[n]);
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
