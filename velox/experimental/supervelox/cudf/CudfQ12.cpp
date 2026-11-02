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
#include <limits>

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
  // Per (mode, priority), high priority first.
  const size_t width = names.size();
  std::vector<int64_t> totals(width * 2, 0);
  for (const auto& [first, last] : orderChunks(ctx, begin, end, std::numeric_limits<int64_t>::max())) {
    if (useJit()) {
      // One reduction over the orders, grouped by high or low priority, each
      // counting its qualifying lines by ship mode.
      std::string counts = "  int64_t counts[" + std::to_string(std::max<size_t>(1, width)) + "] = {};\n";
      std::string tally;
      for (size_t mode = 0; mode < width; ++mode) {
        if (modes.contains(mode)) {
          tally += fmt::format("      counts[{0}] += mode[l] == {0};\n", mode);
        }
      }
      GroupedSums sums;
      sums.inputs = "int8_t priority";
      sums.arrays = withLines(
          orderBatch(ctx, first, last),
          {{"lineitem", "l_shipmode", "mode"},
           {"lineitem", "l_shipdate", "ship"},
           {"lineitem", "l_commitdate", "commit"},
           {"lineitem", "l_receiptdate", "receipt"}});
      sums.prelude = counts +
          fmt::format(
                         "  for (int64_t l = starts[row]; l < starts[row + 1]; ++l) {{\n"
                         "    const int r = receipt[l], c = commit[l];\n"
                         "    if (r < {} || r > {} || c >= r || ship[l] >= c) continue;\n",
                         dateLow,
                         dateHigh) +
          tally + "  }\n";
      sums.filter = "true";
      sums.group = "(" + codeTest("priority", high) + ") ? 0 : 1";
      sums.groups = 2;
      for (size_t mode = 0; mode < width; ++mode) {
        sums.values.push_back("counts[" + std::to_string(mode) + "]");
      }
      const auto part = sumGroups(ctx, sums, {fixedView(ctx, "orders", "o_orderpriority", {first, last})});
      for (size_t mode = 0; mode < width; ++mode) {
        totals[mode * 2] += part[mode];
        totals[mode * 2 + 1] += part[width + mode];
      }
      continue;
    }
    const OrderLines lines = orderLines(ctx, first, last);
    auto priority = cudf::repeat(
        cudf::table_view({fixedView(ctx, "orders", "o_orderpriority", {first, last})}), lines.counts->view(), stream);
    GroupedSums sums;
    sums.inputs = "int8_t mode, int16_t ship, int16_t commit, int16_t receipt, int8_t priority";
    sums.filter = fmt::format(
        "{} && receipt >= {} && receipt <= {} && commit < receipt && ship < commit",
        codeTest("mode", modes),
        dateLow,
        dateHigh);
    sums.group = "mode * 2 + (" + codeTest("priority", high) + " ? 0 : 1)";
    sums.groups = static_cast<int32_t>(width) * 2;
    sums.values = {"1"};
    const auto part = sumGroups(
        ctx,
        sums,
        {fixedView(ctx, "lineitem", "l_shipmode", lines.lines),
         fixedView(ctx, "lineitem", "l_shipdate", lines.lines),
         fixedView(ctx, "lineitem", "l_commitdate", lines.lines),
         fixedView(ctx, "lineitem", "l_receiptdate", lines.lines),
         priority->get_column(0).view()});
    for (size_t i = 0; i < totals.size(); ++i) {
      totals[i] += part[i];
    }
  }
  for (size_t mode = 0; mode < width; ++mode) {
    if (modes.contains(mode) && totals[mode * 2] + totals[mode * 2 + 1]) {
      result.strings[0].push_back(names[mode]);
      result.ints[1].push_back(totals[mode * 2]);
      result.ints[2].push_back(totals[mode * 2 + 1]);
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
