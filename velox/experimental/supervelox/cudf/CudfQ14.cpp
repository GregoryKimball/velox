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
#include <cudf/table/table.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {{"promo_revenue", ResultType::kDouble}};

} // namespace

/// The part type of each line in the ship date zones, gathered by part key,
/// then one reduction to the revenue in all and of promoted parts.
Result q14(QueryContext& ctx) {
  const auto stream = ctx.stream();
  const int64_t dateLow = ctx.param("shipdate_min", days(1995, 9, 1));
  const int64_t dateHigh = ctx.param("shipdate_max", days(1995, 10, 1) - 1);
  const CodeSet promo =
      codesParam(ctx, "promo_types", dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "PROMO"));

  Result result(kSchema);
  const RowRange rows = splitZones(ctx, "lineitem", "l_shipdate", dateLow, dateHigh).some;
  if (rows.size() <= 0) {
    result.doubles[0].push_back(0.0 / 0.0);
    return result;
  }
  const cudf::numeric_scalar<int32_t> one(1, true, stream);
  auto partRows = cudf::binary_operation(
      fixedView(ctx, "lineitem", "l_partkey", rows),
      one,
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      stream);
  auto partType = cudf::gather(
      cudf::table_view({fixedView(ctx, "part", "p_type", {0, ctx.rows("part")})}),
      partRows->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream);

  GroupedSums sums;
  sums.inputs = "int16_t shipdate, int32_t price, int8_t discount, int16_t type";
  sums.prelude = "  const int64_t volume = int64_t{price} * (100 - discount);\n";
  sums.filter = "shipdate >= " + std::to_string(dateLow) + " && shipdate <= " + std::to_string(dateHigh);
  sums.values = {"volume", codeTest("type", promo) + " ? volume : 0"};
  const auto totals = sumGroups(
      ctx,
      sums,
      {fixedView(ctx, "lineitem", "l_shipdate", rows),
       fixedView(ctx, "lineitem", "l_extendedprice", rows),
       fixedView(ctx, "lineitem", "l_discount", rows),
       partType->get_column(0).view()});
  result.doubles[0].push_back(100.0 * static_cast<double>(totals[1]) / static_cast<double>(totals[0]));
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
