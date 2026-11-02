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
#include <cudf/binaryop.hpp>
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

/// Q2 with transform_lto: per wanted part, a mask of its partsupp rows in
/// the region at the part's minimum cost, then the first 100 offers ordered
/// on the host by number, names only for those. Supplier names sort like
/// their keys.
Result cheapestOffers(QueryContext& ctx, int64_t parts) {
  Result result(kSchema);
  const std::string body = "  *best = 0;\n  if (!(size == 15 && " +
      codeTest("type", dictionaryCodes(ctx, "part", "p_type", Match::kSuffix, "BRASS")) +
      ")) {\n    return 0;\n  }\n  int64_t low = INT64_MAX;\n  int32_t mask = 0;\n"
      "  for (int k = 0; k < 4; ++k) {\n    const int64_t r = int64_t{row} * 4 + k;\n"
      "    if ((" + std::to_string(nationsInRegion(ctx, "EUROPE")) + "u >> nation[supplier[r] - 1]) & 1) {\n"
      "      const int64_t c = cost[r];\n      mask = c < low ? 1 << k : c == low ? mask | 1 << k : mask;\n"
      "      low = c < low ? c : low;\n    }\n  }\n  *best = mask;";
  const cudf::numeric_scalar<int32_t> none(0, true, ctx.stream());
  std::vector<int32_t> part;
  std::vector<int32_t> mask;
  forEachBatch(
      ctx,
      {0, parts},
      parts,
      [&](RowRange batch) {
        auto masks = transformRows(
            ctx,
            batch.size(),
            "int8_t size, int16_t type",
            {fixedView(ctx, "part", "p_size", batch), fixedView(ctx, "part", "p_type", batch)},
            {{"partsupp", "ps_suppkey", "supplier", {}, batch.begin * 4},
             {"partsupp", "ps_supplycost", "cost", {}, batch.begin * 4},
             {"supplier", "s_nationkey", "nation"}},
            {{cudf::type_id::INT32, "best"}},
            body);
        auto any = cudf::binary_operation(
            masks->get_column(0).view(),
            none,
            cudf::binary_operator::NOT_EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            ctx.stream());
        auto index = sequence(ctx, batch.size(), static_cast<int32_t>(batch.begin));
        auto kept = selectRows(ctx, cudf::table_view({index->view(), masks->get_column(0).view()}), any->view());
        const auto batchPart = download<int32_t>(ctx, kept->get_column(0).view());
        const auto batchMask = download<int32_t>(ctx, kept->get_column(1).view());
        part.insert(part.end(), batchPart.begin(), batchPart.end());
        mask.insert(mask.end(), batchMask.begin(), batchMask.end());
      },
      4);

  std::vector<std::string> names;
  for (int n = 0; n < 25; ++n) {
    names.push_back(nationName(ctx, n));
  }
  std::vector<int> rank(names.size());
  {
    std::vector<int> order(names.size());
    for (size_t n = 0; n < order.size(); ++n) {
      order[n] = static_cast<int>(n);
    }
    std::sort(order.begin(), order.end(), [&](int x, int y) { return names[x] < names[y]; });
    for (size_t i = 0; i < order.size(); ++i) {
      rank[order[i]] = static_cast<int>(i);
    }
  }
  const Column& suppkey = ctx.meta("partsupp", "ps_suppkey");
  const Column& partkey = ctx.meta("partsupp", "ps_partkey");
  const Column& acctbal = ctx.meta("supplier", "s_acctbal");
  const Column& nationOf = ctx.meta("supplier", "s_nationkey");
  // (-acctbal, nation name rank, supplier, partkey).
  std::vector<std::tuple<int64_t, int, int64_t, int64_t>> offers;
  for (size_t i = 0; i < part.size(); ++i) {
    for (int k = 0; k < 4; ++k) {
      if (mask[i] >> k & 1) {
        const int64_t row = int64_t{part[i]} * 4 + k;
        const int64_t supplier = suppkey.value(row) - 1;
        offers.emplace_back(-acctbal.value(supplier), rank[nationOf.value(supplier)], supplier, partkey.value(row));
      }
    }
  }
  const size_t limit = std::min<size_t>(offers.size(), 100);
  std::partial_sort(offers.begin(), offers.begin() + limit, offers.end());
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& mfgr = ctx.meta("part", "p_mfgr");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  const Column& comment = ctx.meta("supplier", "s_comment");
  for (size_t i = 0; i < limit; ++i) {
    const auto& [negative, nationRank, supplier, key] = offers[i];
    result.decimals[0].push_back(-negative);
    result.strings[1].push_back(name.string(supplier));
    result.strings[2].push_back(nationName(ctx, nationOf.value(supplier)));
    result.ints[3].push_back(key);
    result.strings[4].push_back(mfgr.string(key - 1));
    result.strings[5].push_back(address.string(supplier));
    result.strings[6].push_back(phone.string(supplier));
    result.strings[7].push_back(comment.string(supplier));
  }
  return result;
}

} // namespace

/// The partsupp rows of the wanted parts, through the part mask repeated over
/// each part's four rows; those in the region; the minimum cost per part from
/// groupby, joined back on (part, cost); the order is made on the host.
Result q2(QueryContext& ctx) {
  const int64_t parts = ctx.rows("part");
  if (useJit()) {
    return cheapestOffers(ctx, parts);
  }
  Result result(kSchema);
  // A part's four rows are in one batch, so each batch has its parts'
  // cheapest offers.
  std::vector<int32_t> bestRows;
  forEachBatch(
      ctx,
      {0, parts},
      parts,
      [&](RowRange batch) {
        const RowRange slots{batch.begin * 4, batch.end * 4};
        auto wanted = computeColumn(
            ctx,
            "int8_t size, int16_t type",
            "size == 15 && " + codeTest("type", dictionaryCodes(ctx, "part", "p_type", Match::kSuffix, "BRASS")),
            cudf::type_id::BOOL8,
            {fixedView(ctx, "part", "p_size", batch), fixedView(ctx, "part", "p_type", batch)});
        auto perRow = cudf::repeat(cudf::table_view({wanted->view()}), 4, ctx.stream());
        auto index = sequence(ctx, slots.size(), static_cast<int32_t>(slots.begin));
        auto candidates = selectRows(
            ctx,
            cudf::table_view(
                {index->view(),
                 fixedView(ctx, "partsupp", "ps_suppkey", slots),
                 fixedView(ctx, "partsupp", "ps_supplycost", slots)}),
            perRow->get_column(0).view());
        perRow.reset();
        index.reset();
        if (candidates->num_rows() == 0) {
          return;
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
          return;
        }
        auto part = computeColumn(ctx, "int32_t row", "row / 4", cudf::type_id::INT32, {offers->get_column(0).view()});
        auto cheapest =
            groupBy(ctx, cudf::table_view({part->view()}), {{offers->get_column(2).view(), Aggregate::kMin}});
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
        const auto rows = download<int32_t>(ctx, best->get_column(0).view());
        bestRows.insert(bestRows.end(), rows.begin(), rows.end());
      },
      4);

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
