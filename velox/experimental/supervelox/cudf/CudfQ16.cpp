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
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>

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
  auto wanted = computeColumn(
      ctx,
      "int8_t brand, int16_t type, int8_t size",
      "brand != " + std::to_string(ctx.meta("part", "p_brand").code("Brand#45")) + " && !" +
          codeTest("type", dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "MEDIUM POLISHED")) + " && size >= 0 && " +
          codeTest("size", sizes),
      cudf::type_id::BOOL8,
      {fixedView(ctx, "part", "p_brand", {0, parts}),
       fixedView(ctx, "part", "p_type", {0, parts}),
       fixedView(ctx, "part", "p_size", {0, parts})});
  auto partIndex = sequence(ctx, parts);
  auto perRow = cudf::repeat(cudf::table_view({wanted->view(), partIndex->view()}), 4, ctx.stream());
  partIndex.reset();
  auto offers = selectRows(
      ctx,
      cudf::table_view({perRow->get_column(1).view(), fixedView(ctx, "partsupp", "ps_suppkey", {0, rows})}),
      perRow->get_column(0).view());
  perRow.reset();
  Result result(kSchema);
  const auto& brands = ctx.meta("part", "p_brand").dictionary;
  const auto& types = ctx.meta("part", "p_type").dictionary;
  result.dictionaries[0] = &brands;
  result.dictionaries[1] = &types;
  if (offers->num_rows() == 0) {
    return result;
  }
  auto complaints = likeColumn(ctx, "supplier", "s_comment", "%Customer%Complaints%", {0, ctx.rows("supplier")});
  auto supplierRows = keyRows(ctx, offers->get_column(1).view());
  auto complained = cudf::gather(
      cudf::table_view({complaints->view()}), supplierRows->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
  auto clean = computeColumn(ctx, "bool complained", "!complained", cudf::type_id::BOOL8, {complained->get_column(0).view()});
  auto kept = selectRows(ctx, offers->view(), clean->view());
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
