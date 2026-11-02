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
#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

const std::vector<ResultField> kSchema = {
    {"ps_partkey", ResultType::kBigint},
    {"value", ResultType::kDecimal, 38, 2},
};

} // namespace

/// The value of each partsupp row whose supplier is in the nation, summed
/// over each part's four rows by a segmented reduction, kept above the
/// fraction of the total and sorted by value.
Result q11(QueryContext& ctx) {
  const int64_t rows = ctx.rows("partsupp");
  const int64_t parts = rows / 4;
  Result result(kSchema);
  int64_t threshold = 0;
  std::vector<std::unique_ptr<cudf::column>> batches;
  forEachBatch(
      ctx,
      {0, parts},
      parts,
      [&](RowRange batch) {
        const RowRange slots{batch.begin * 4, batch.end * 4};
        std::unique_ptr<cudf::column> value;
        if (useJit()) {
          // Per part, over its four partsupp rows.
          value = std::move(transformRows(
                                ctx,
                                batch.size(),
                                "",
                                {},
                                {{"partsupp", "ps_suppkey", "supplier", {}, slots.begin},
                                 {"partsupp", "ps_supplycost", "cost", {}, slots.begin},
                                 {"partsupp", "ps_availqty", "available", {}, slots.begin},
                                 {"supplier", "s_nationkey", "nation"}},
                                {{cudf::type_id::INT64, "value"}},
                                "  int64_t sum = 0;\n  for (int64_t r = int64_t{row} * 4; r < int64_t{row} * 4 + 4; ++r) {\n"
                                "    sum += nation[supplier[r] - 1] == " + std::to_string(nationKey(ctx, "GERMANY")) +
                                    " ? int64_t{cost[r]} * available[r] : 0;\n  }\n  *value = sum;")
                                ->release()[0]);
        } else {
          auto suppliers = keyRows(ctx, fixedView(ctx, "partsupp", "ps_suppkey", slots));
          auto nation = gatherColumns(ctx, "supplier", {"s_nationkey"}, suppliers->view());
          suppliers.reset();
          auto slotValue = computeColumn(
              ctx,
              "int8_t nation, int32_t cost, int16_t available",
              "nation == " + std::to_string(nationKey(ctx, "GERMANY")) + " ? int64_t{cost} * available : 0",
              cudf::type_id::INT64,
              {nation->get_column(0).view(),
               fixedView(ctx, "partsupp", "ps_supplycost", slots),
               fixedView(ctx, "partsupp", "ps_availqty", slots)});
          nation.reset();
          auto offsets = sequence(ctx, batch.size() + 1, 0, 4);
          value = cudf::segmented_reduce(
              slotValue->view(),
              cudf::device_span<cudf::size_type const>(offsets->view().data<cudf::size_type>(), offsets->size()),
              *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
              cudf::data_type{cudf::type_id::INT64},
              cudf::null_policy::EXCLUDE,
              ctx.stream());
        }
        auto total = cudf::reduce(
            value->view(), *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), value->type(), ctx.stream());
        threshold += static_cast<cudf::numeric_scalar<int64_t>&>(*total).value(ctx.stream());
        batches.push_back(std::move(value));
      },
      4);
  std::unique_ptr<cudf::column> perPart;
  if (batches.size() == 1) {
    perPart = std::move(batches[0]);
  } else {
    std::vector<cudf::column_view> views;
    for (const auto& batch : batches) {
      views.push_back(batch->view());
    }
    perPart = cudf::concatenate(views, ctx.stream());
  }
  batches.clear();
  // value > total * 0.0001 / SF with SF = suppliers / 10000.
  auto above = computeColumn(
      ctx,
      "int64_t value",
      "value * " + std::to_string(ctx.rows("supplier")) + " > " + std::to_string(threshold),
      cudf::type_id::BOOL8,
      {perPart->view()});
  auto index = sequence(ctx, parts);
  auto kept = selectRows(ctx, cudf::table_view({index->view(), perPart->view()}), above->view());
  auto sorted = cudf::stable_sort_by_key(
      kept->view(), cudf::table_view({kept->get_column(1).view()}), {cudf::order::DESCENDING}, {}, ctx.stream());
  const auto part = download<int32_t>(ctx, sorted->get_column(0).view());
  const auto values = download<int64_t>(ctx, sorted->get_column(1).view());
  const Column& partkey = ctx.meta("partsupp", "ps_partkey");
  for (size_t i = 0; i < part.size(); ++i) {
    result.ints[0].push_back(partkey.value(int64_t{part[i]} * 4));
    result.decimals[1].push_back(values[i]);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
