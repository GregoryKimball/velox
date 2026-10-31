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
#include <tuple>

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"s_acctbal", ResultType::kDecimal, 15, 2},
    {"s_name", ResultType::kVarchar},
    {"n_name", ResultType::kVarchar},
    {"p_partkey", ResultType::kBigint},
    {"p_mfgr", ResultType::kVarchar},
    {"s_address", ResultType::kVarchar},
    {"s_phone", ResultType::kVarchar},
    {"s_comment", ResultType::kVarchar},
};

} // namespace

/// The partsupp rows of the wanted parts, through the part mask repeated over
/// each part's four rows; those in the region; the minimum cost per part from
/// groupby, joined back on (part, cost); the order is made on the host.
Result q2(QueryContext& ctx) {
  const int64_t parts = ctx.rows("part");
  const int64_t rows = ctx.rows("partsupp");
  auto wanted = computeColumn(
      ctx,
      "int8_t size, int16_t type",
      "size == 15 && " + codeTest("type", dictionaryCodes(ctx, "part", "p_type", Match::kSuffix, "BRASS")),
      cudf::type_id::BOOL8,
      {fixedView(ctx, "part", "p_size", {0, parts}), fixedView(ctx, "part", "p_type", {0, parts})});
  auto perRow = cudf::repeat(cudf::table_view({wanted->view()}), 4, ctx.stream());
  auto index = sequence(ctx, rows);
  auto candidates = selectRows(
      ctx,
      cudf::table_view({index->view(), fixedView(ctx, "partsupp", "ps_suppkey", {0, rows}), fixedView(ctx, "partsupp", "ps_supplycost", {0, rows})}),
      perRow->get_column(0).view());
  perRow.reset();
  index.reset();
  Result result(kSchema);
  if (candidates->num_rows() == 0) {
    return result;
  }
  auto supplierRows = keyRows(ctx, candidates->get_column(1).view());
  auto nation = gatherColumns(ctx, "supplier", {"s_nationkey"}, supplierRows->view());
  auto inRegion = computeColumn(
      ctx,
      "int8_t nation",
      "(" + std::to_string(nationsInRegion(ctx, "EUROPE")) + "u >> nation) & 1",
      cudf::type_id::BOOL8,
      {nation->get_column(0).view()});
  auto offers = selectRows(ctx, candidates->view(), inRegion->view());
  if (offers->num_rows() == 0) {
    return result;
  }
  auto part = computeColumn(ctx, "int32_t row", "row / 4", cudf::type_id::INT32, {offers->get_column(0).view()});
  auto cheapest = groupBy(ctx, cudf::table_view({part->view()}), {{offers->get_column(2).view(), Aggregate::kMin}});
  auto [offerMap, cheapestMap] = cudf::inner_join(
      cudf::table_view({part->view(), offers->get_column(2).view()}),
      cudf::table_view({cheapest.keys->get_column(0).view(), cheapest.values[0]->view()}),
      cudf::null_equality::EQUAL,
      ctx.stream());
  auto best = cudf::gather(
      cudf::table_view({offers->get_column(0).view()}),
      deviceView(cudf::type_id::INT32, offerMap->data(), offerMap->size()),
      cudf::out_of_bounds_policy::DONT_CHECK,
      ctx.stream());
  const auto bestRows = download<int32_t>(ctx, best->get_column(0).view());

  const Column& suppkey = ctx.meta("partsupp", "ps_suppkey");
  const Column& partkey = ctx.meta("partsupp", "ps_partkey");
  const Column& acctbal = ctx.meta("supplier", "s_acctbal");
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& nationOf = ctx.meta("supplier", "s_nationkey");
  struct Row {
    int64_t acctbal;
    std::string nation;
    std::string name;
    int64_t partkey;
    int64_t supplier;
  };
  std::vector<Row> sorted;
  for (const int32_t row : bestRows) {
    const int64_t supplier = suppkey.value(row) - 1;
    sorted.push_back(
        {acctbal.value(supplier), nationName(ctx, nationOf.value(supplier)), name.string(supplier), partkey.value(row), supplier});
  }
  std::sort(sorted.begin(), sorted.end(), [](const Row& x, const Row& y) {
    return std::make_tuple(-x.acctbal, std::cref(x.nation), std::cref(x.name), x.partkey) <
        std::make_tuple(-y.acctbal, std::cref(y.nation), std::cref(y.name), y.partkey);
  });
  sorted.resize(std::min<size_t>(sorted.size(), 100));
  const Column& mfgr = ctx.meta("part", "p_mfgr");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  const Column& comment = ctx.meta("supplier", "s_comment");
  for (const auto& row : sorted) {
    result.decimals[0].push_back(row.acctbal);
    result.strings[1].push_back(row.name);
    result.strings[2].push_back(row.nation);
    result.ints[3].push_back(row.partkey);
    result.strings[4].push_back(mfgr.string(row.partkey - 1));
    result.strings[5].push_back(address.string(row.supplier));
    result.strings[6].push_back(phone.string(row.supplier));
    result.strings[7].push_back(comment.string(row.supplier));
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
