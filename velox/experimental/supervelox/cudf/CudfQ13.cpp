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
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
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
  // The customers of the orders of `chunk` kept.
  auto keptCustomers = [&](RowRange chunk) {
    auto special = likeColumn(ctx, "orders", "o_comment", "%special%requests%", chunk);
    auto plain = cudf::unary_operation(special->view(), cudf::unary_operator::NOT, ctx.stream());
    special.reset();
    return selectRows(ctx, cudf::table_view({fixedView(ctx, "orders", "o_custkey", chunk)}), plain->view());
  };
  auto counted = [&](RowRange chunk) {
    auto kept = keptCustomers(chunk);
    return groupBy(ctx, kept->view(), {{kept->get_column(0).view(), Aggregate::kCount}});
  };
  Grouped perCustomer;
  if (batched()) {
    // The counts so far in a column of one row per customer, at key - 1:
    // each batch's counts added at its customers.
    const cudf::numeric_scalar<int32_t> zero(0, true, ctx.stream());
    const cudf::numeric_scalar<int32_t> one(1, true, ctx.stream());
    auto dense = std::make_unique<cudf::table>(
        cudf::table_view({cudf::make_column_from_scalar(zero, ctx.rows("customer"), ctx.stream())->view()}),
        ctx.stream());
    forEachBatch(
        ctx,
        {0, orders},
        orders,
        [&](RowRange chunk) {
          auto part = counted(chunk);
          auto rows = cudf::binary_operation(
              part.keys->get_column(0).view(),
              one,
              cudf::binary_operator::SUB,
              cudf::data_type{cudf::type_id::INT32},
              ctx.stream());
          auto before = cudf::gather(dense->view(), rows->view(), cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
          auto after = cudf::binary_operation(
              before->get_column(0).view(),
              part.values[0]->view(),
              cudf::binary_operator::ADD,
              cudf::data_type{cudf::type_id::INT32},
              ctx.stream());
          dense = cudf::scatter(cudf::table_view({after->view()}), rows->view(), dense->view(), ctx.stream());
        },
        4);
    // Customers without orders count 0 here, as below.
    auto counts = std::move(dense->release()[0]);
    const cudf::numeric_scalar<int32_t> none(0, true, ctx.stream());
    auto has = cudf::binary_operation(
        counts->view(), none, cudf::binary_operator::NOT_EQUAL, cudf::data_type{cudf::type_id::BOOL8}, ctx.stream());
    auto withOrders = selectRows(ctx, cudf::table_view({counts->view()}), has->view());
    perCustomer.values.push_back(std::move(withOrders->release()[0]));
  } else {
    std::vector<std::unique_ptr<cudf::table>> pieces;
    for (int64_t begin = 0; begin < orders; begin += kChunkRows) {
      auto part = counted({begin, std::min(orders, begin + kChunkRows)});
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(std::move(part.keys->release()[0]));
      columns.push_back(std::move(part.values[0]));
      pieces.push_back(std::make_unique<cudf::table>(std::move(columns)));
    }
    auto partial = concatenateTables(ctx, pieces);
    pieces.clear();
    perCustomer = groupBy(
        ctx, cudf::table_view({partial->get_column(0).view()}), {{partial->get_column(1).view(), Aggregate::kSum}});
  }
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
