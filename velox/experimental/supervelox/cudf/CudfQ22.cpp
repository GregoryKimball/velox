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
#include <functional>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

constexpr int kCodes = 7;
constexpr int kCountryCodes[kCodes] = {13, 31, 23, 29, 30, 18, 17};

const std::vector<ResultField> kSchema = {
    {"cntrycode", ResultType::kVarchar},
    {"numcust", ResultType::kBigint},
    {"totacctbal", ResultType::kDecimal, 38, 2},
};

} // namespace

/// Country codes from the first two characters of the phone, customers with
/// orders marked by a scatter of the orders' customer keys, then two
/// reductions: the average positive balance, and the bins above it.
Result q22(QueryContext& ctx) {
  const int64_t customers = ctx.rows("customer");
  const int64_t orders = ctx.rows("orders");
  const cudf::numeric_scalar<bool> no(false, true, ctx.stream());
  const cudf::numeric_scalar<bool> yes(true, true, ctx.stream());
  // Scattered by key, batch by batch of orders, into one more row than
  // customers, so that row r of the customers is row r + 1.
  std::vector<std::unique_ptr<cudf::column>> none;
  none.push_back(cudf::make_column_from_scalar(no, static_cast<cudf::size_type>(customers + 1), ctx.stream()));
  auto hasOrders = std::make_unique<cudf::table>(std::move(none));
  forEachBatch(
      ctx,
      {0, orders},
      orders,
      [&](RowRange batch) {
        hasOrders = cudf::scatter(
            std::vector<std::reference_wrapper<const cudf::scalar>>{yes},
            fixedView(ctx, "orders", "o_custkey", batch),
            hasOrders->view(),
            ctx.stream());
      },
      4);

  std::string prelude = "  const int bin = ";
  for (int i = 0; i < kCodes; ++i) {
    prelude += "code == " + std::to_string(kCountryCodes[i]) + " ? " + std::to_string(i) + " : ";
  }
  prelude += "-1;\n";
  // The sums of `sums` over the customers, batch by batch.
  auto sumCustomers = [&](const GroupedSums& sums) {
    std::vector<int64_t> totals;
    forEachBatch(ctx, {0, customers}, customers, [&](RowRange batch) {
      auto prefix = cudf::strings::slice_strings(
          cudf::strings_column_view(stringView(ctx, "customer", "c_phone", batch)), 0, 2, std::nullopt, ctx.stream());
      auto code = cudf::strings::to_integers(
          cudf::strings_column_view(prefix->view()), cudf::data_type{cudf::type_id::INT32}, ctx.stream());
      prefix.reset();
      const cudf::column_view has = cudf::slice(
          hasOrders->get_column(0).view(),
          {static_cast<cudf::size_type>(batch.begin + 1), static_cast<cudf::size_type>(batch.end + 1)})[0];
      const auto part =
          sumGroups(ctx, sums, {code->view(), fixedView(ctx, "customer", "c_acctbal", batch), has});
      totals.resize(part.size(), 0);
      for (size_t i = 0; i < part.size(); ++i) {
        totals[i] += part[i];
      }
    });
    return totals;
  };
  GroupedSums positive;
  positive.inputs = "int32_t code, int32_t balance, bool has";
  positive.prelude = prelude;
  positive.filter = "balance > 0 && bin >= 0";
  positive.values = {"int64_t{balance}", "1"};
  const auto average = sumCustomers(positive);

  GroupedSums bins = positive;
  // balance > avg, kept in integers.
  bins.filter = "balance > 0 && int64_t{balance} * " + std::to_string(average[1]) + " > " +
      std::to_string(average[0]) + " && !has && bin >= 0";
  bins.group = "bin";
  bins.groups = kCodes;
  bins.values = {"1", "int64_t{balance}"};
  const auto totals = sumCustomers(bins);

  std::vector<std::pair<int, int>> sorted;
  for (int i = 0; i < kCodes; ++i) {
    sorted.emplace_back(kCountryCodes[i], i);
  }
  std::sort(sorted.begin(), sorted.end());
  Result result(kSchema);
  for (const auto& [country, index] : sorted) {
    if (totals[index * 2]) {
      result.strings[0].push_back(std::to_string(country));
      result.ints[1].push_back(totals[index * 2]);
      result.decimals[2].push_back(totals[index * 2 + 1]);
    }
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
