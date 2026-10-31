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
#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"c_custkey", ResultType::kBigint},
    {"c_name", ResultType::kVarchar},
    {"revenue", ResultType::kDecimal, 38, 4},
    {"c_acctbal", ResultType::kDecimal, 15, 2},
    {"n_name", ResultType::kVarchar},
    {"c_address", ResultType::kVarchar},
    {"c_phone", ResultType::kVarchar},
    {"c_comment", ResultType::kVarchar},
};

} // namespace

/// Each order's revenue from returned lines as a segmented sum, grouped by
/// customer, then the twenty best by sorting.
Result q10(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1993, 10, 1), days(1994, 1, 1));
  Result result(kSchema);
  if (end <= begin) {
    return result;
  }
  const OrderLines lines = orderLines(ctx, begin, end);
  auto revenue = computeColumn(
      ctx,
      "int8_t flag, int32_t price, int8_t discount",
      "flag == " + std::to_string(ctx.meta("lineitem", "l_returnflag").code("R")) +
          " ? int64_t{price} * (100 - discount) : 0",
      cudf::type_id::INT64,
      {fixedView(ctx, "lineitem", "l_returnflag", lines.lines),
       fixedView(ctx, "lineitem", "l_extendedprice", lines.lines),
       fixedView(ctx, "lineitem", "l_discount", lines.lines)});
  auto perOrder = sumPerOrder(ctx, revenue->view(), lines);
  revenue.reset();
  auto grouped = groupBy(
      ctx,
      cudf::table_view({fixedView(ctx, "orders", "o_custkey", {begin, end})}),
      {{perOrder->view(), Aggregate::kSum}});
  const cudf::column_view customer = grouped.keys->get_column(0).view();
  const cudf::column_view sum = grouped.values[0]->view();
  auto top = firstRows(
      ctx,
      cudf::table_view({customer, sum}),
      cudf::table_view({sum, customer}),
      {cudf::order::DESCENDING, cudf::order::ASCENDING},
      20);
  const auto customers = downloadInts(ctx, top->get_column(0).view());
  const auto sums = download<int64_t>(ctx, top->get_column(1).view());
  const Column& name = ctx.meta("customer", "c_name");
  const Column& acctbal = ctx.meta("customer", "c_acctbal");
  const Column& nation = ctx.meta("customer", "c_nationkey");
  const Column& address = ctx.meta("customer", "c_address");
  const Column& phone = ctx.meta("customer", "c_phone");
  const Column& comment = ctx.meta("customer", "c_comment");
  for (size_t i = 0; i < customers.size() && sums[i] > 0; ++i) {
    const int64_t row = customers[i] - 1;
    result.ints[0].push_back(customers[i]);
    result.strings[1].push_back(name.string(row));
    result.decimals[2].push_back(sums[i]);
    result.decimals[3].push_back(acctbal.value(row));
    result.strings[4].push_back(nationName(ctx, nation.value(row)));
    result.strings[5].push_back(address.string(row));
    result.strings[6].push_back(phone.string(row));
    result.strings[7].push_back(comment.string(row));
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
