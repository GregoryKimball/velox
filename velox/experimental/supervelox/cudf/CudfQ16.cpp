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
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"p_brand", ResultType::kVarchar},
    {"p_type", ResultType::kVarchar},
    {"p_size", ResultType::kInteger},
    {"supplier_cnt", ResultType::kBigint},
};

} // namespace

/// The partsupp rows of the wanted parts, through the part mask repeated over
/// each part's four rows, without suppliers whose comment is LIKE the
/// complaint; distinct suppliers per (brand, type, size) from groupby, sorted.
/// Dictionary codes sort like their strings.
Result q16(QueryContext& ctx) {
  const int64_t parts = ctx.rows("part");
  const int64_t rows = ctx.rows("partsupp");
  CodeSet sizes;
  for (const int size : {49, 14, 23, 45, 19, 3, 36, 9}) {
    sizes.add(size);
  }
  if (useJit()) {
    // Per partsupp row, its part's (brand, type, size) packed into one key
    // that sorts like them, above the supplier, or -1; then the distinct
    // pairs, counted per key.
    // Batch by batch of parts, the distinct pairs of the batch; then the
    // distinct pairs of all.
    auto complaints = likeColumn(ctx, "supplier", "s_comment", "%Customer%Complaints%", {0, ctx.rows("supplier")});
    const std::string pairOf = "  const int64_t part = row / 4;\n  const int32_t brand = brands[part], type = types[part], size = sizes[part];\n"
        "  *pair = brand != " + std::to_string(ctx.meta("part", "p_brand").code("Brand#45")) + " && !" +
        codeTest("type", dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "MEDIUM POLISHED")) +
        " && size >= 0 && " + codeTest("size", sizes) +
        " && !complained[supplier - 1] ? int64_t{brand << 24 | type << 8 | size} << 32 | supplier : -1;";
    const cudf::numeric_scalar<int64_t> none(-1, true, ctx.stream());
    std::vector<std::unique_ptr<cudf::table>> pieces;
    forEachBatch(
        ctx,
        {0, parts},
        parts,
        [&](RowRange batch) {
          const cudf::column_view supplier = fixedView(ctx, "partsupp", "ps_suppkey", {batch.begin * 4, batch.end * 4});
          auto keyed = transformRows(
              ctx,
              supplier.size(),
              std::string(deviceType(supplier.type().id())) + " supplier",
              {supplier},
              {{"part", "p_brand", "brands", {}, batch.begin},
               {"part", "p_type", "types", {}, batch.begin},
               {"part", "p_size", "sizes", {}, batch.begin},
               {.name = "complained", .view = complaints->view()}},
              {{cudf::type_id::INT64, "pair"}},
              pairOf);
          auto wanted = cudf::binary_operation(
              keyed->get_column(0).view(),
              none,
              cudf::binary_operator::NOT_EQUAL,
              cudf::data_type{cudf::type_id::BOOL8},
              ctx.stream());
          auto kept = selectRows(ctx, cudf::table_view({keyed->get_column(0).view()}), wanted->view());
          keyed.reset();
          wanted.reset();
          pieces.push_back(cudf::distinct(
              kept->view(),
              {0},
              cudf::duplicate_keep_option::KEEP_ANY,
              cudf::null_equality::EQUAL,
              cudf::nan_equality::ALL_EQUAL,
              ctx.stream()));
        },
        4);
    Result result(kSchema);
    const auto& brands = ctx.meta("part", "p_brand").dictionary;
    const auto& types = ctx.meta("part", "p_type").dictionary;
    result.dictionaries[0] = &brands;
    result.dictionaries[1] = &types;
    auto pairs = pieces.size() == 1 ? std::move(pieces[0]) : concatenateTables(ctx, pieces);
    pieces.clear();
    if (pairs->num_rows() == 0) {
      return result;
    }
    if (batched()) {
      pairs = cudf::distinct(
          pairs->view(),
          {0},
          cudf::duplicate_keep_option::KEEP_ANY,
          cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL,
          ctx.stream());
    }
    const cudf::numeric_scalar<int32_t> shift(32, true, ctx.stream());
    auto pairKey = cudf::binary_operation(
        pairs->get_column(0).view(), shift, cudf::binary_operator::SHIFT_RIGHT, cudf::data_type{cudf::type_id::INT32}, ctx.stream());
    pairs.reset();
    auto grouped = groupBy(ctx, cudf::table_view({pairKey->view()}), {{pairKey->view(), Aggregate::kCount}});
    const cudf::column_view key = grouped.keys->get_column(0).view();
    const cudf::column_view count = grouped.values[0]->view();
    auto sorted = firstRows(
        ctx,
        cudf::table_view({key, count}),
        cudf::table_view({count, key}),
        {cudf::order::DESCENDING, cudf::order::ASCENDING},
        key.size());
    const auto keys = downloadInts(ctx, sorted->get_column(0).view());
    const auto suppliers = downloadInts(ctx, sorted->get_column(1).view());
    for (size_t i = 0; i < keys.size(); ++i) {
      result.codes[0].push_back(keys[i] >> 24);
      result.codes[1].push_back(keys[i] >> 8 & 0xffff);
      result.ints[2].push_back(keys[i] & 0xff);
      result.ints[3].push_back(suppliers[i]);
    }
    return result;
  }
  auto complaints = likeColumn(ctx, "supplier", "s_comment", "%Customer%Complaints%", {0, ctx.rows("supplier")});
  std::vector<std::unique_ptr<cudf::table>> pieces;
  forEachBatch(
      ctx,
      {0, parts},
      parts,
      [&](RowRange batch) {
        auto wanted = computeColumn(
            ctx,
            "int8_t brand, int16_t type, int8_t size",
            "brand != " + std::to_string(ctx.meta("part", "p_brand").code("Brand#45")) + " && !" +
                codeTest("type", dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "MEDIUM POLISHED")) +
                " && size >= 0 && " + codeTest("size", sizes),
            cudf::type_id::BOOL8,
            {fixedView(ctx, "part", "p_brand", batch),
             fixedView(ctx, "part", "p_type", batch),
             fixedView(ctx, "part", "p_size", batch)});
        auto partIndex = sequence(ctx, batch.size(), static_cast<int32_t>(batch.begin));
        auto perRow = cudf::repeat(cudf::table_view({wanted->view(), partIndex->view()}), 4, ctx.stream());
        partIndex.reset();
        auto offers = selectRows(
            ctx,
            cudf::table_view(
                {perRow->get_column(1).view(),
                 fixedView(ctx, "partsupp", "ps_suppkey", {batch.begin * 4, batch.end * 4})}),
            perRow->get_column(0).view());
        perRow.reset();
        auto supplierRows = keyRows(ctx, offers->get_column(1).view());
        auto complained = cudf::gather(
            cudf::table_view({complaints->view()}),
            supplierRows->view(),
            cudf::out_of_bounds_policy::DONT_CHECK,
            ctx.stream());
        auto clean =
            computeColumn(ctx, "bool complained", "!complained", cudf::type_id::BOOL8, {complained->get_column(0).view()});
        pieces.push_back(selectRows(ctx, offers->view(), clean->view()));
      },
      4);
  Result result(kSchema);
  const auto& brands = ctx.meta("part", "p_brand").dictionary;
  const auto& types = ctx.meta("part", "p_type").dictionary;
  result.dictionaries[0] = &brands;
  result.dictionaries[1] = &types;
  auto kept = pieces.size() == 1 ? std::move(pieces[0]) : concatenateTables(ctx, pieces);
  pieces.clear();
  if (kept->num_rows() == 0) {
    return result;
  }
  auto attributes = gatherColumns(ctx, "part", {"p_brand", "p_type", "p_size"}, kept->get_column(0).view());
  auto grouped = groupBy(ctx, attributes->view(), {{kept->get_column(1).view(), Aggregate::kDistinct}});
  const cudf::table_view keys = grouped.keys->view();
  const cudf::column_view count = grouped.values[0]->view();
  auto sorted = firstRows(
      ctx,
      cudf::table_view({keys.column(0), keys.column(1), keys.column(2), count}),
      cudf::table_view({count, keys.column(0), keys.column(1), keys.column(2)}),
      {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
      keys.num_rows());
  const auto brand = downloadInts(ctx, sorted->get_column(0).view());
  const auto type = downloadInts(ctx, sorted->get_column(1).view());
  const auto size = downloadInts(ctx, sorted->get_column(2).view());
  const auto suppliers = downloadInts(ctx, sorted->get_column(3).view());
  for (size_t i = 0; i < brand.size(); ++i) {
    result.codes[0].push_back(brand[i]);
    result.codes[1].push_back(type[i]);
    result.ints[2].push_back(size[i]);
    result.ints[3].push_back(suppliers[i]);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
