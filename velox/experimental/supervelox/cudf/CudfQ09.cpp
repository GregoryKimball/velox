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
#include <stdexcept>

#include <cudf/copying.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

constexpr int kNations = 25;
constexpr int kFirstYear = 1992;
constexpr int kYears = 8;

const std::vector<ResultField> kSchema = {
    {"nation", ResultType::kVarchar},
    {"o_year", ResultType::kBigint},
    {"sum_profit", ResultType::kDecimal, 38, 4},
};

} // namespace

/// The lines of green parts, kept chunk by chunk through a gathered LIKE
/// mask; the partsupp row found among the part's four, and the profit grouped
/// by supplier nation and order year.
Result q9(QueryContext& ctx) {
  const int64_t parts = ctx.rows("part");
  auto green = likeColumn(ctx, "part", "p_name", "%green%", {0, parts});
  std::string year = "  int year = 0;\n";
  for (int y = kFirstYear + 1; y < kFirstYear + kYears; ++y) {
    year += "  year += date >= " + std::to_string(days(y, 1, 1)) + ";\n";
  }
  std::vector<int64_t> profit(kNations * kYears, 0);
  for (const auto& [first, last] : orderChunks(ctx, 0, ctx.rows("orders"))) {
    const OrderLines lines = orderLines(ctx, first, last);
    const cudf::column_view partkey = fixedView(ctx, "lineitem", "l_partkey", lines.lines);
    auto partRows = keyRows(ctx, partkey);
    auto keep = cudf::gather(
        cudf::table_view({green->view()}), partRows->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
    partRows.reset();
    auto date = perLine(ctx, lines, {fixedView(ctx, "orders", "o_orderdate", {first, last})});
    auto kept = selectRows(
        ctx,
        cudf::table_view(
            {partkey,
             fixedView(ctx, "lineitem", "l_suppkey", lines.lines),
             fixedView(ctx, "lineitem", "l_quantity", lines.lines),
             fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
             fixedView(ctx, "lineitem", "l_discount", lines.lines),
             date->get_column(0).view()}),
        keep->get_column(0).view());
    keep.reset();
    date.reset();
    if (kept->num_rows() == 0) {
      continue;
    }
    const cudf::column_view supplier = kept->get_column(1).view();
    auto rows = partsuppRows(ctx, kept->get_column(0).view(), supplier);
    auto cost = gatherColumns(ctx, "partsupp", {"ps_supplycost"}, rows->view());
    auto supplierRows = keyRows(ctx, supplier);
    auto nation = gatherColumns(ctx, "supplier", {"s_nationkey"}, supplierRows->view());
    const std::vector<cudf::column_view> inputs = {
        rows->view(),
        kept->get_column(2).view(),
        kept->get_column(3).view(),
        kept->get_column(4).view(),
        kept->get_column(5).view(),
        cost->get_column(0).view(),
        nation->get_column(0).view()};
    const std::string declared = std::string("int32_t row, int16_t quantity, int32_t price, int8_t discount, ") +
        deviceType(inputs[4].type().id()) + " date, int32_t cost, int8_t nation";
    auto bin = computeColumn(
        ctx, declared, "row < 0 ? -1 : nation * " + std::to_string(kYears) + " + year", cudf::type_id::INT32, inputs, year);
    auto amount = computeColumn(
        ctx,
        declared,
        "int64_t{price} * (100 - discount) - int64_t{cost} * quantity",
        cudf::type_id::INT64,
        inputs);
    auto grouped = groupBy(ctx, cudf::table_view({bin->view()}), {{amount->view(), Aggregate::kSum}});
    const auto bins = download<int32_t>(ctx, grouped.keys->get_column(0).view());
    const auto sums = download<int64_t>(ctx, grouped.values[0]->view());
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i] < 0) {
        throw std::runtime_error("q9: lineitem (part, supplier) missing from partsupp layout");
      }
      profit[bins[i]] += sums[i];
    }
  }
  std::vector<std::pair<std::string, int>> nations;
  for (int n = 0; n < kNations; ++n) {
    nations.emplace_back(nationName(ctx, n), n);
  }
  std::sort(nations.begin(), nations.end());
  Result result(kSchema);
  for (const auto& [name, n] : nations) {
    for (int y = kYears - 1; y >= 0; --y) {
      const int64_t value = profit[n * kYears + y];
      if (value) {
        result.strings[0].push_back(name);
        result.ints[1].push_back(kFirstYear + y);
        result.decimals[2].push_back(value);
      }
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
