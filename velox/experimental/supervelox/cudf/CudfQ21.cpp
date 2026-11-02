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
#include <cudf/scalar/scalar.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <map>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"s_name", ResultType::kVarchar},
    {"numwait", ResultType::kBigint},
};

} // namespace

/// Per order, segmented minima and maxima of the supplier over all lines and
/// over late lines, and the late line count. An order counts for supplier s
/// when every late line is s's and another supplier has a line: then each of
/// s's late lines has an EXISTS partner and no NOT EXISTS one.
Result q21(QueryContext& ctx) {
  const int32_t status = ctx.meta("orders", "o_orderstatus").code("F");
  const int32_t saudi = nationKey(ctx, "SAUDI ARABIA");
  std::map<int64_t, int64_t> counts;
  if (useJit()) {
    // Per order in one pass over its lines: the supplier waiting on the
    // order, if any, and its late lines; then their sum per supplier and the
    // first 100, all on the GPU. Supplier names sort like their keys.
    const cudf::numeric_scalar<int32_t> zero(0, true, ctx.stream());
    std::vector<std::unique_ptr<cudf::table>> pieces;
    for (const auto& [first, last] : orderChunks(ctx, 0, ctx.rows("orders"))) {
      auto waiting = perOrderColumns(
          ctx,
          first,
          last,
          "int8_t status",
          {fixedView(ctx, "orders", "o_orderstatus", {first, last})},
          {{"lineitem", "l_suppkey", "supplier"},
           {"lineitem", "l_commitdate", "commit"},
           {"lineitem", "l_receiptdate", "receipt"},
           {"supplier", "s_nationkey", "nation"}},
          {{cudf::type_id::INT32, "waiter"}, {cudf::type_id::INT64, "lates"}},
          "  int64_t low = INT64_MAX, high = 0, lateLow = INT64_MAX, lateHigh = 0;\n  int64_t count = 0;",
          "    const int64_t s = supplier[l];\n"
          "    low = s < low ? s : low;\n    high = s > high ? s : high;\n"
          "    if (receipt[l] > commit[l]) {\n"
          "      lateLow = s < lateLow ? s : lateLow;\n      lateHigh = s > lateHigh ? s : lateHigh;\n      ++count;\n"
          "    }",
          fmt::format(
              "  const bool waits = status == {} && count > 0 && lateLow == lateHigh && low != high && "
              "nation[lateLow - 1] == {};\n  *waiter = waits ? int32_t(lateLow) : 0;\n  *lates = waits ? count : 0;",
              status,
              saudi));
      auto waits = cudf::binary_operation(
          waiting->get_column(0).view(),
          zero,
          cudf::binary_operator::NOT_EQUAL,
          cudf::data_type{cudf::type_id::BOOL8},
          ctx.stream());
      pieces.push_back(selectRows(ctx, waiting->view(), waits->view()));
    }
    Result result(kSchema);
    auto ours = concatenateTables(ctx, pieces);
    if (ours->num_rows() == 0) {
      return result;
    }
    pieces.clear();
    auto grouped =
        groupBy(ctx, cudf::table_view({ours->get_column(0).view()}), {{ours->get_column(1).view(), Aggregate::kSum}});
    const cudf::column_view supplier = grouped.keys->get_column(0).view();
    const cudf::column_view count = grouped.values[0]->view();
    auto top = firstRows(
        ctx,
        cudf::table_view({supplier, count}),
        cudf::table_view({count, supplier}),
        {cudf::order::DESCENDING, cudf::order::ASCENDING},
        100);
    const auto suppliers = downloadInts(ctx, top->get_column(0).view());
    const auto sums = downloadInts(ctx, top->get_column(1).view());
    const Column& name = ctx.meta("supplier", "s_name");
    for (size_t i = 0; i < suppliers.size(); ++i) {
      result.strings[0].push_back(name.string(suppliers[i] - 1));
      result.ints[1].push_back(sums[i]);
    }
    return result;
  }
  for (const auto& [first, last] : orderChunks(ctx, 0, ctx.rows("orders"))) {
    const OrderLines lines = orderLines(ctx, first, last);
    const cudf::column_view supplier = fixedView(ctx, "lineitem", "l_suppkey", lines.lines);
    const std::string type = deviceType(supplier.type().id());
    const std::vector<cudf::column_view> lineColumns = {
        supplier,
        fixedView(ctx, "lineitem", "l_commitdate", lines.lines),
        fixedView(ctx, "lineitem", "l_receiptdate", lines.lines)};
    const std::string declared = type + " supplier, int16_t commit, int16_t receipt";
    const std::string none = supplier.type().id() == cudf::type_id::INT16 ? "32767" : "2147483647";
    auto lateLow = computeColumn(
        ctx, declared, "receipt > commit ? supplier : " + type + "(" + none + ")", supplier.type().id(), lineColumns);
    auto lateHigh = computeColumn(ctx, declared, "receipt > commit ? supplier : " + type + "(0)", supplier.type().id(), lineColumns);
    auto late = computeColumn(ctx, declared, "receipt > commit ? int64_t{1} : int64_t{0}", cudf::type_id::INT64, lineColumns);
    auto low = extremePerOrder(ctx, supplier, lines, false);
    auto high = extremePerOrder(ctx, supplier, lines, true);
    auto lateMin = extremePerOrder(ctx, lateLow->view(), lines, false);
    auto lateMax = extremePerOrder(ctx, lateHigh->view(), lines, true);
    auto lateCount = sumPerOrder(ctx, late->view(), lines);
    lateLow.reset();
    lateHigh.reset();
    late.reset();
    auto keep = computeColumn(
        ctx,
        "int8_t status, " + type + " low, " + type + " high, " + type + " lateMin, " + type + " lateMax, int64_t count",
        "status == " + std::to_string(status) + " && count > 0 && lateMin == lateMax && low != high",
        cudf::type_id::BOOL8,
        {fixedView(ctx, "orders", "o_orderstatus", {first, last}),
         low->view(),
         high->view(),
         lateMin->view(),
         lateMax->view(),
         lateCount->view()});
    auto waiting = selectRows(ctx, cudf::table_view({lateMin->view(), lateCount->view()}), keep->view());
    if (waiting->num_rows() == 0) {
      continue;
    }
    auto supplierRows = keyRows(ctx, waiting->get_column(0).view());
    auto nation = gatherColumns(ctx, "supplier", {"s_nationkey"}, supplierRows->view());
    auto inNation = computeColumn(
        ctx, "int8_t nation", "nation == " + std::to_string(saudi), cudf::type_id::BOOL8, {nation->get_column(0).view()});
    auto ours = selectRows(ctx, waiting->view(), inNation->view());
    auto grouped =
        groupBy(ctx, cudf::table_view({ours->get_column(0).view()}), {{ours->get_column(1).view(), Aggregate::kSum}});
    const auto keys = downloadInts(ctx, grouped.keys->get_column(0).view());
    const auto sums = downloadInts(ctx, grouped.values[0]->view());
    for (size_t i = 0; i < keys.size(); ++i) {
      counts[keys[i]] += sums[i];
    }
  }
  const Column& name = ctx.meta("supplier", "s_name");
  std::vector<std::pair<int64_t, std::string>> rows;
  for (const auto& [supplier, count] : counts) {
    rows.emplace_back(count, name.string(supplier - 1));
  }
  std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) {
    return x.first != y.first ? x.first > y.first : x.second < y.second;
  });
  rows.resize(std::min<size_t>(rows.size(), 100));
  Result result(kSchema);
  for (auto& [count, supplierName] : rows) {
    result.strings[0].push_back(std::move(supplierName));
    result.ints[1].push_back(count);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
