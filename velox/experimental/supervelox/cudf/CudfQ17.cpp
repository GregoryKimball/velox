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
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {{"avg_yearly", ResultType::kDouble}};

constexpr int64_t kChunkRows = int64_t{1} << 28;

} // namespace

/// The lines of the brand and container's parts, kept chunk by chunk through
/// a gathered part mask; per-part quantity sums and counts from groupby,
/// joined back to the lines.
Result q17(QueryContext& ctx) {
  const int64_t parts = ctx.rows("part");
  auto wanted = computeColumn(
      ctx,
      "int8_t brand, int8_t container",
      "brand == " + std::to_string(ctx.meta("part", "p_brand").code("Brand#23")) +
          " && container == " + std::to_string(ctx.meta("part", "p_container").code("MED BOX")),
      cudf::type_id::BOOL8,
      {fixedView(ctx, "part", "p_brand", {0, parts}), fixedView(ctx, "part", "p_container", {0, parts})});
  const int64_t rows = ctx.rows("lineitem");
  std::unique_ptr<cudf::table> lines;
  if (useJit()) {
    lines = linesOfParts(ctx, wanted->view(), {0, rows}, {}, "", {"l_partkey", "l_quantity", "l_extendedprice"});
  } else {
    std::vector<std::unique_ptr<cudf::table>> pieces;
    forEachBatch(ctx, {0, rows}, kChunkRows, [&](RowRange chunk) {
      const cudf::column_view partkey = fixedView(ctx, "lineitem", "l_partkey", chunk);
      auto partRows = keyRows(ctx, partkey);
      auto keep = cudf::gather(
          cudf::table_view({wanted->view()}), partRows->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
      partRows.reset();
      pieces.push_back(selectRows(
          ctx,
          cudf::table_view(
              {partkey,
               fixedView(ctx, "lineitem", "l_quantity", chunk),
               fixedView(ctx, "lineitem", "l_extendedprice", chunk)}),
          keep->get_column(0).view()));
    });
    std::vector<cudf::table_view> views;
    for (const auto& piece : pieces) {
      views.push_back(piece->view());
    }
    lines = cudf::concatenate(views, ctx.stream());
  }
  Result result(kSchema);
  if (lines->num_rows() == 0) {
    result.doubles[0].push_back(0);
    return result;
  }
  const cudf::table_view partkey({lines->get_column(0).view()});
  auto grouped = groupBy(
      ctx,
      partkey,
      {{lines->get_column(1).view(), Aggregate::kSum}, {lines->get_column(1).view(), Aggregate::kCount}});
  auto [lineMap, groupMap] = cudf::inner_join(partkey, grouped.keys->view(), cudf::null_equality::EQUAL, ctx.stream());
  const auto lineRows = deviceView(cudf::type_id::INT32, lineMap->data(), lineMap->size());
  const auto groupRows = deviceView(cudf::type_id::INT32, groupMap->data(), groupMap->size());
  auto line = cudf::gather(
      cudf::table_view({lines->get_column(1).view(), lines->get_column(2).view()}),
      lineRows,
      cudf::out_of_bounds_policy::DONT_CHECK,
      ctx.stream());
  auto group = cudf::gather(
      cudf::table_view({grouped.values[0]->view(), grouped.values[1]->view()}),
      groupRows,
      cudf::out_of_bounds_policy::DONT_CHECK,
      ctx.stream());
  // quantity < 0.2 * avg(quantity), kept in integers.
  auto small = computeColumn(
      ctx,
      "int16_t quantity, int32_t price, int64_t sum, int32_t count",
      "5 * int64_t{quantity} * count < sum ? int64_t{price} : 0",
      cudf::type_id::INT64,
      {line->get_column(0).view(), line->get_column(1).view(), group->get_column(0).view(), group->get_column(1).view()});
  auto total = cudf::reduce(
      small->view(), *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), small->type(), ctx.stream());
  result.doubles[0].push_back(
      static_cast<cudf::numeric_scalar<int64_t>&>(*total).value(ctx.stream()) / 100.0 / 7.0);
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
