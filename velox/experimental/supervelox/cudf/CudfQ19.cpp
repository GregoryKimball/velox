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
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {{"revenue", ResultType::kDecimal, 38, 4}};

} // namespace

/// The lines that pass the lineitem-only conjuncts, their parts' brand,
/// container and size gathered by part key, then one reduction over the
/// three branches.
Result q19(QueryContext& ctx) {
  const auto stream = ctx.stream();
  const int64_t instruct = ctx.param("shipinstruct", ctx.meta("lineitem", "l_shipinstruct").code("DELIVER IN PERSON"));
  const CodeSet modes =
      codesParam(ctx, "shipmodes", dictionaryCodes(ctx, "lineitem", "l_shipmode", {"AIR", "AIR REG"}));
  const RowRange lines{0, ctx.rows("lineitem")};

  auto keep = computeColumn(
      ctx,
      "int8_t instruct, int8_t mode",
      "instruct == " + std::to_string(instruct) + " && " + codeTest("mode", modes),
      cudf::type_id::BOOL8,
      {fixedView(ctx, "lineitem", "l_shipinstruct", lines), fixedView(ctx, "lineitem", "l_shipmode", lines)});
  auto kept = cudf::apply_retention_mask(
      cudf::table_view(
          {fixedView(ctx, "lineitem", "l_partkey", lines),
           fixedView(ctx, "lineitem", "l_quantity", lines),
           fixedView(ctx, "lineitem", "l_extendedprice", lines),
           fixedView(ctx, "lineitem", "l_discount", lines)}),
      keep->view(),
      stream);
  keep.reset();

  const cudf::numeric_scalar<int32_t> one(1, true, stream);
  auto partRows = cudf::binary_operation(
      kept->get_column(0).view(), one, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::INT32}, stream);
  const RowRange parts{0, ctx.rows("part")};
  auto part = cudf::gather(
      cudf::table_view(
          {fixedView(ctx, "part", "p_brand", parts),
           fixedView(ctx, "part", "p_container", parts),
           fixedView(ctx, "part", "p_size", parts)}),
      partRows->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream);

  const Column& brand = ctx.meta("part", "p_brand");
  const char* defaultBrands[3] = {"Brand#12", "Brand#23", "Brand#34"};
  const std::vector<std::string> defaultContainers[3] = {
      {"SM CASE", "SM BOX", "SM PACK", "SM PKG"},
      {"MED BAG", "MED BOX", "MED PKG", "MED PACK"},
      {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}};
  const int64_t defaultQuantity[3][2] = {{100, 1100}, {1000, 2000}, {2000, 3000}};
  const int64_t defaultSize[3] = {5, 10, 15};
  std::string filter;
  for (int b = 0; b < 3; ++b) {
    const std::string n = std::to_string(b);
    const CodeSet containers =
        codesParam(ctx, "containers" + n, dictionaryCodes(ctx, "part", "p_container", defaultContainers[b]));
    filter += fmt::format(
        "{}(brand == {} && {} && quantity >= {} && quantity <= {} && size >= {} && size <= {})",
        b ? " || " : "",
        ctx.param("brand" + n, brand.code(defaultBrands[b])),
        codeTest("container", containers),
        ctx.param("quantity_min" + n, defaultQuantity[b][0]),
        ctx.param("quantity_max" + n, defaultQuantity[b][1]),
        ctx.param("size_min" + n, 1),
        ctx.param("size_max" + n, defaultSize[b]));
  }
  GroupedSums sums;
  sums.inputs = "int16_t quantity, int32_t price, int8_t discount, int8_t brand, int8_t container, int8_t size";
  sums.filter = filter;
  sums.values = {"int64_t{price} * (100 - discount)"};
  const auto totals = sumGroups(
      ctx,
      sums,
      {kept->get_column(1).view(),
       kept->get_column(2).view(),
       kept->get_column(3).view(),
       part->get_column(0).view(),
       part->get_column(1).view(),
       part->get_column(2).view()});
  Result result(kSchema);
  result.decimals[0].push_back(totals[0]);
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
