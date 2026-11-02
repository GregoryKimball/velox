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
#include <cudf/stream_compaction.hpp>

#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

constexpr int64_t kChunkRows = int64_t{1} << 28;

const std::vector<ResultField> kSchema = {
    {"s_name", ResultType::kVarchar},
    {"s_address", ResultType::kVarchar},
};

} // namespace

/// The year's lines of forest parts, chunk by chunk over the shipdate zones
/// through a gathered LIKE mask; shipped quantity per partsupp row from
/// groupby; the rows with stock above half of it whose supplier is in the
/// nation, made distinct by supplier.
Result q20(QueryContext& ctx) {
  const int32_t low = days(1994, 1, 1);
  const int32_t high = days(1995, 1, 1);
  const int64_t parts = ctx.rows("part");
  auto forest = likeColumn(ctx, "part", "p_name", "forest%", {0, parts});
  const RowRange zones = splitZones(ctx, "lineitem", "l_shipdate", low, high - 1).some;
  Result result(kSchema);
  std::unique_ptr<cudf::table> lines;
  if (useJit()) {
    lines = linesOfParts(
        ctx,
        forest->view(),
        zones,
        {{"lineitem", "l_shipdate", "ship"}},
        fmt::format("ship[l] >= {} && ship[l] < {}", low, high),
        {"l_partkey", "l_suppkey", "l_quantity"});
  } else {
    std::vector<std::unique_ptr<cudf::table>> pieces;
    forEachBatch(ctx, zones, kChunkRows, [&](RowRange chunk) {
      const cudf::column_view partkey = fixedView(ctx, "lineitem", "l_partkey", chunk);
      auto partRows = keyRows(ctx, partkey);
      auto isForest = cudf::gather(
          cudf::table_view({forest->view()}), partRows->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
      partRows.reset();
      auto keep = computeColumn(
          ctx,
          "int16_t ship, bool forest",
          "forest && ship >= " + std::to_string(low) + " && ship < " + std::to_string(high),
          cudf::type_id::BOOL8,
          {fixedView(ctx, "lineitem", "l_shipdate", chunk), isForest->get_column(0).view()});
      pieces.push_back(selectRows(
          ctx,
          cudf::table_view(
              {partkey, fixedView(ctx, "lineitem", "l_suppkey", chunk), fixedView(ctx, "lineitem", "l_quantity", chunk)}),
          keep->view()));
    });
    if (pieces.empty()) {
      return result;
    }
    lines = concatenateTables(ctx, pieces);
  }
  if (lines->num_rows() == 0) {
    return result;
  }
  auto rows = partsuppRows(ctx, lines->get_column(0).view(), lines->get_column(1).view());
  auto shipped = groupBy(ctx, cudf::table_view({rows->view()}), {{lines->get_column(2).view(), Aggregate::kSum}});
  const cudf::column_view row = shipped.keys->get_column(0).view();
  auto stock = gatherColumns(ctx, "partsupp", {"ps_availqty", "ps_suppkey"}, row);
  const cudf::column_view supplier = stock->get_column(1).view();
  auto supplierRows = keyRows(ctx, supplier);
  auto nation = gatherColumns(ctx, "supplier", {"s_nationkey"}, supplierRows->view());
  // availqty > 0.5 * sum(l_quantity); quantities carry two decimals.
  auto excess = computeColumn(
      ctx,
      "int32_t row, int16_t available, int64_t sum, int8_t nation",
      "row >= 0 && 200 * int64_t{available} > sum && nation == " + std::to_string(nationKey(ctx, "CANADA")),
      cudf::type_id::BOOL8,
      {row, stock->get_column(0).view(), shipped.values[0]->view(), nation->get_column(0).view()});
  auto chosen = selectRows(ctx, cudf::table_view({supplier}), excess->view());
  auto distinct = cudf::distinct(
      chosen->view(), {0}, cudf::duplicate_keep_option::KEEP_ANY, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, ctx.stream());
  // Supplier names sort as their keys.
  auto suppliers = downloadInts(ctx, distinct->get_column(0).view());
  std::sort(suppliers.begin(), suppliers.end());
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& address = ctx.meta("supplier", "s_address");
  result.strings[0].reserve(suppliers.size());
  result.strings[1].reserve(suppliers.size());
  for (const int64_t key : suppliers) {
    result.strings[0].push_back(name.string(key - 1));
    result.strings[1].push_back(address.string(key - 1));
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
