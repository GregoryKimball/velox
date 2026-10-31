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
#include <cudf/filling.hpp>
#include <cudf/table/table.hpp>

#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"l_shipmode", ResultType::kVarchar},
    {"high_line_count", ResultType::kBigint},
    {"low_line_count", ResultType::kBigint},
};

} // namespace

/// The orders that can have lines received in the date range are
/// contiguous, and so are their lines. Each order's priority repeated over
/// its lines, then one reduction counting high and low priority lines by
/// ship mode.
Result q12(QueryContext& ctx) {
  const auto stream = ctx.stream();
  const int64_t dateLow = ctx.param("receiptdate_min", days(1994, 1, 1));
  const int64_t dateHigh = ctx.param("receiptdate_max", days(1995, 1, 1) - 1);
  const CodeSet modes =
      codesParam(ctx, "shipmodes", dictionaryCodes(ctx, "lineitem", "l_shipmode", {"MAIL", "SHIP"}));
  const CodeSet high = codesParam(
      ctx, "high_priorities", dictionaryCodes(ctx, "orders", "o_orderpriority", {"1-URGENT", "2-HIGH"}));
  const auto lag = lineDateLag(ctx, "l_receiptdate");
  const auto [begin, end] =
      sortedRange(ctx, "orders", "o_orderdate", dateLow - lag.second, dateHigh + 1 - lag.first);
  const auto& names = ctx.meta("lineitem", "l_shipmode").dictionary;
  Result result(kSchema);
  if (end <= begin) {
    return result;
  }
  const uint32_t* starts = orderLineStarts(ctx);
  const auto bounds = ctx.download(starts + begin, end - begin + 1);
  const RowRange lines{bounds.front(), bounds.back()};

  auto counts = cudf::binary_operation(
      deviceView(cudf::type_id::INT32, starts + begin + 1, end - begin),
      deviceView(cudf::type_id::INT32, starts + begin, end - begin),
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      stream);
  auto priority = cudf::repeat(
      cudf::table_view({fixedView(ctx, "orders", "o_orderpriority", {begin, end})}), counts->view(), stream);

  GroupedSums sums;
  sums.inputs = "int8_t mode, int16_t ship, int16_t commit, int16_t receipt, int8_t priority";
  sums.filter = fmt::format(
      "{} && receipt >= {} && receipt <= {} && commit < receipt && ship < commit",
      codeTest("mode", modes),
      dateLow,
      dateHigh);
  sums.group = "mode * 2 + (" + codeTest("priority", high) + " ? 0 : 1)";
  sums.groups = static_cast<int32_t>(names.size()) * 2;
  sums.values = {"1"};
  const auto totals = sumGroups(
      ctx,
      sums,
      {fixedView(ctx, "lineitem", "l_shipmode", lines),
       fixedView(ctx, "lineitem", "l_shipdate", lines),
       fixedView(ctx, "lineitem", "l_commitdate", lines),
       fixedView(ctx, "lineitem", "l_receiptdate", lines),
       priority->get_column(0).view()});
  for (size_t mode = 0; mode < names.size(); ++mode) {
    if (modes.contains(mode) && totals[mode * 2] + totals[mode * 2 + 1]) {
      result.strings[0].push_back(names[mode]);
      result.ints[1].push_back(totals[mode * 2]);
      result.ints[2].push_back(totals[mode * 2 + 1]);
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
