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

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"c_name", ResultType::kVarchar},
    {"c_custkey", ResultType::kBigint},
    {"o_orderkey", ResultType::kBigint},
    {"o_orderdate", ResultType::kDate},
    {"o_totalprice", ResultType::kDecimal, 15, 2},
    {"sum(l_quantity)", ResultType::kDecimal, 38, 2},
};

} // namespace

/// Each order's quantity as a segmented sum over its lines, the large ones
/// kept by a mask, then ordered on the host.
Result q18(QueryContext& ctx) {
  const cudf::numeric_scalar<int64_t> limit(30000, true, ctx.stream());
  std::vector<std::pair<int64_t, int64_t>> found;
  for (const auto& [first, last] : orderChunks(ctx, 0, ctx.rows("orders"))) {
    std::unique_ptr<cudf::column> quantity;
    if (useJit()) {
      quantity = std::move(perOrderColumns(
                               ctx,
                               first,
                               last,
                               "",
                               {},
                               {{"lineitem", "l_quantity", "quantity"}},
                               {{cudf::type_id::INT64, "sum"}},
                               "  int64_t total = 0;",
                               "    total += quantity[l];",
                               "  *sum = total;")
                               ->release()[0]);
    } else {
      const OrderLines lines = orderLines(ctx, first, last);
      quantity = sumPerOrder(ctx, fixedView(ctx, "lineitem", "l_quantity", lines.lines), lines);
    }
    auto large = cudf::binary_operation(
        quantity->view(), limit, cudf::binary_operator::GREATER, cudf::data_type{cudf::type_id::BOOL8}, ctx.stream());
    auto rows = sequence(ctx, last - first);
    auto kept = selectRows(ctx, cudf::table_view({rows->view(), quantity->view()}), large->view());
    const auto index = download<int32_t>(ctx, kept->get_column(0).view());
    const auto sum = download<int64_t>(ctx, kept->get_column(1).view());
    for (size_t i = 0; i < index.size(); ++i) {
      found.emplace_back(first + index[i], sum[i]);
    }
  }
  const Column& totalprice = ctx.meta("orders", "o_totalprice");
  const Column& orderdate = ctx.meta("orders", "o_orderdate");
  std::sort(found.begin(), found.end(), [&](const auto& x, const auto& y) {
    const int64_t px = totalprice.value(x.first);
    const int64_t py = totalprice.value(y.first);
    if (px != py) {
      return px > py;
    }
    return orderdate.value(x.first) < orderdate.value(y.first);
  });
  found.resize(std::min<size_t>(found.size(), 100));
  const Column& custkey = ctx.meta("orders", "o_custkey");
  const Column& orderkey = ctx.meta("orders", "o_orderkey");
  const Column& name = ctx.meta("customer", "c_name");
  Result result(kSchema);
  for (const auto& [row, sum] : found) {
    const int64_t customer = custkey.value(row);
    result.strings[0].push_back(name.string(customer - 1));
    result.ints[1].push_back(customer);
    result.ints[2].push_back(orderkey.value(row));
    result.ints[3].push_back(orderdate.value(row));
    result.decimals[4].push_back(totalprice.value(row));
    result.decimals[5].push_back(sum);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
