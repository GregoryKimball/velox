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

#include <cudf/unary.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

constexpr int64_t kChunkRows = int64_t{1} << 28;

const std::vector<ResultField> kSchema = {
    {"c_count", ResultType::kBigint},
    {"custdist", ResultType::kBigint},
};

} // namespace

/// Orders whose comment is not LIKE the pattern, chunk by chunk; their count
/// per customer from groupby, merged over the chunks, then a second groupby
/// of the counts.
Result q13(QueryContext& ctx) {
  const int64_t orders = ctx.rows("orders");
  std::vector<std::unique_ptr<cudf::table>> pieces;
  for (int64_t begin = 0; begin < orders; begin += kChunkRows) {
    const RowRange chunk{begin, std::min(orders, begin + kChunkRows)};
    auto special = likeColumn(ctx, "orders", "o_comment", "%special%requests%", chunk);
    auto plain = cudf::unary_operation(special->view(), cudf::unary_operator::NOT, ctx.stream());
    special.reset();
    auto kept = selectRows(ctx, cudf::table_view({fixedView(ctx, "orders", "o_custkey", chunk)}), plain->view());
    plain.reset();
    auto counted = groupBy(ctx, kept->view(), {{kept->get_column(0).view(), Aggregate::kCount}});
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(counted.keys->release()[0]));
    columns.push_back(std::move(counted.values[0]));
    pieces.push_back(std::make_unique<cudf::table>(std::move(columns)));
  }
  auto partial = concatenateTables(ctx, pieces);
  pieces.clear();
  auto perCustomer =
      groupBy(ctx, cudf::table_view({partial->get_column(0).view()}), {{partial->get_column(1).view(), Aggregate::kSum}});
  partial.reset();
  auto histogram = groupBy(
      ctx, cudf::table_view({perCustomer.values[0]->view()}), {{perCustomer.values[0]->view(), Aggregate::kCount}});
  const auto counts = downloadInts(ctx, histogram.keys->get_column(0).view());
  const auto customers = downloadInts(ctx, histogram.values[0]->view());
  std::vector<std::pair<int64_t, int64_t>> rows;
  int64_t withOrders = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    rows.emplace_back(customers[i], counts[i]);
    withOrders += customers[i];
  }
  if (ctx.rows("customer") > withOrders) {
    rows.emplace_back(ctx.rows("customer") - withOrders, 0);
  }
  std::sort(rows.rbegin(), rows.rend());
  Result result(kSchema);
  for (const auto& [customersWithCount, count] : rows) {
    result.ints[0].push_back(count);
    result.ints[1].push_back(customersWithCount);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
