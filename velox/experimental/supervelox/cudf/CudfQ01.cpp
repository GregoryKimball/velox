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
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cstdlib>
#include <limits>
#include <map>

#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

/// Per row, the discounted price at scale 4 and the charge at scale 6. The
/// charge needs 128 bits once summed, so it is a DECIMAL128 column.
constexpr const char* kTransform = R"(
#include <cudf/fixed_point/fixed_point.hpp>
#include <cuda/std/cstdint>

using namespace cuda::std;

extern "C" __device__ int transform(
    int64_t* discounted, numeric::decimal128* charge, int32_t price, int8_t discount, int8_t tax) {
  const int64_t value = int64_t{price} * (100 - discount);
  *discounted = value;
  *charge = numeric::decimal128{numeric::scaled_integer<__int128_t>{
      __int128_t{value} * (100 + tax), numeric::scale_type{-6}}};
  return 0;
}
)";

/// Rows per batch: the derived columns of a batch take 24 bytes a row.
constexpr int64_t kBatchRows = 128 << 20;

const std::vector<ResultField> kSchema = {
    {"l_returnflag", ResultType::kVarchar},
    {"l_linestatus", ResultType::kVarchar},
    {"sum_qty", ResultType::kDecimal, 38, 2},
    {"sum_base_price", ResultType::kDecimal, 38, 2},
    {"sum_disc_price", ResultType::kDecimal, 38, 4},
    {"sum_charge", ResultType::kDecimal, 38, 6},
    {"avg_qty", ResultType::kDouble},
    {"avg_price", ResultType::kDouble},
    {"avg_disc", ResultType::kDouble},
    {"count_order", ResultType::kBigint},
};

/// Sums of quantity, price, discounted price, charge and discount, and the
/// row count, of one (returnflag, linestatus) group.
struct Totals {
  __int128 quantity{0};
  __int128 price{0};
  __int128 discounted{0};
  __int128 charge{0};
  __int128 discount{0};
  int64_t count{0};
};

template <typename T>
std::vector<T> download(QueryContext& ctx, const cudf::column_view& column) {
  std::vector<T> host(column.size());
  if (!host.empty()) {
    cudaMemcpyAsync(
        host.data(), column.head<T>() + column.offset(), host.size() * sizeof(T), cudaMemcpyDeviceToHost, ctx.stream());
  }
  return host;
}

/// Groups rows `rows` of lineitem, keeping only those set in `mask` if
/// given, and adds the sums to `groups`.
void aggregate(
    QueryContext& ctx,
    RowRange rows,
    const cudf::bitmask_type* mask,
    cudf::size_type nullCount,
    std::map<std::pair<int8_t, int8_t>, Totals>& groups) {
  const auto stream = ctx.stream();
  const cudf::column_view price = fixedView(ctx, "lineitem", "l_extendedprice", rows);
  const cudf::column_view discount = fixedView(ctx, "lineitem", "l_discount", rows);
  const cudf::column_view tax = fixedView(ctx, "lineitem", "l_tax", rows);

  // Without JIT, the charge is 100 * discounted + discounted * tax, summed
  // as two int64 columns.
  std::vector<std::unique_ptr<cudf::column>> derived;
  if (useJit()) {
    const cudf::transform_input inputs[] = {price, discount, tax};
    const cudf::transform_output outputs[] = {
        {cudf::data_type{cudf::type_id::INT64}, cudf::output_nullability::ALL_VALID},
        {cudf::data_type{cudf::type_id::DECIMAL128, numeric::scale_type{-6}}, cudf::output_nullability::ALL_VALID},
    };
    derived = transformColumns(ctx, kTransform, inputs, outputs)->release();
  } else {
    const std::string declared = "int32_t price, int8_t discount, int8_t tax";
    const std::string discounted = "int64_t{price} * (100 - discount)";
    derived.push_back(computeColumn(ctx, declared, discounted, cudf::type_id::INT64, {price, discount, tax}));
    derived.push_back(
        computeColumn(ctx, declared, "(" + discounted + ") * tax", cudf::type_id::INT64, {price, discount, tax}));
  }

  const cudf::column_view flag = fixedView(ctx, "lineitem", "l_returnflag", rows);
  const cudf::column_view keys[] = {
      cudf::column_view(flag.type(), flag.size(), flag.head(), mask, nullCount),
      fixedView(ctx, "lineitem", "l_linestatus", rows),
  };
  const cudf::column_view values[] = {
      fixedView(ctx, "lineitem", "l_quantity", rows),
      price,
      derived[0]->view(),
      derived[1]->view(),
      discount,
  };
  std::vector<cudf::groupby::aggregation_request> requests(std::size(values));
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i].values = values[i];
    requests[i].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  }
  requests[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::groupby groupby(cudf::table_view({keys[0], keys[1]}));
  auto [groupKeys, results] = groupby.aggregate(requests, stream);

  const auto flags = download<int8_t>(ctx, groupKeys->get_column(0).view());
  const auto statuses = download<int8_t>(ctx, groupKeys->get_column(1).view());
  const auto quantity = download<int64_t>(ctx, results[0].results[0]->view());
  const auto count = download<int32_t>(ctx, results[0].results[1]->view());
  const auto priceSum = download<int64_t>(ctx, results[1].results[0]->view());
  const auto discounted = download<int64_t>(ctx, results[2].results[0]->view());
  const auto charge = useJit() ? download<__int128>(ctx, results[3].results[0]->view()) : std::vector<__int128>{};
  const auto taxed = useJit() ? std::vector<int64_t>{} : download<int64_t>(ctx, results[3].results[0]->view());
  const auto discountSum = download<int64_t>(ctx, results[4].results[0]->view());
  cudaStreamSynchronize(stream);
  for (size_t g = 0; g < flags.size(); ++g) {
    Totals& totals = groups[{flags[g], statuses[g]}];
    totals.quantity += quantity[g];
    totals.price += priceSum[g];
    totals.discounted += discounted[g];
    totals.charge += useJit() ? charge[g] : static_cast<__int128>(discounted[g]) * 100 + taxed[g];
    totals.discount += discountSum[g];
    totals.count += count[g];
  }
}

/// Rows of `rows` whose shipdate is at most `limit`, as a validity mask.
std::pair<std::unique_ptr<cuda::device_buffer<cudf::bitmask_type>>, cudf::size_type>
shipdateMask(QueryContext& ctx, RowRange rows, int64_t limit) {
  const auto stream = ctx.stream();
  cudf::numeric_scalar<int16_t> bound(static_cast<int16_t>(limit), true, stream);
  cudf::ast::tree tree;
  const auto& shipdate = tree.push(cudf::ast::column_reference(0));
  const auto& literal = tree.push(cudf::ast::literal(bound));
  const auto& keep = tree.push(cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, shipdate, literal));
  const cudf::table_view table({fixedView(ctx, "lineitem", "l_shipdate", rows)});
  auto bools = useJit() ? cudf::compute_column_jit(table, keep, stream) : cudf::compute_column(table, keep, stream);
  return cudf::bools_to_mask(bools->view(), stream);
}

/// Sums kept per group by the reduction: quantity, price, discounted price,
/// discounted price times tax, discount and rows. The charge is
/// 100 * discounted + taxed, each of which fits 64 bits at SF300.
constexpr int32_t kSums = 6;

/// The most groups a reduction keeps in its state rather than handing the
/// rows to groupby: the state is kSums 64-bit fields a group.
constexpr int64_t kMostReducedGroups = 8;

/// A reduction whose state is the sums of every (returnflag, linestatus)
/// group, which the dictionaries make a small dense domain.
std::string reductionSource(int32_t statuses, int32_t groups, int64_t dateLimit) {
  const int32_t fields = groups * kSums;
  auto list = [&](const std::string& type, const std::string& prefix) {
    std::string text;
    for (int32_t i = 0; i < fields; ++i) {
      text += (i ? ", " : "") + type + " " + prefix + std::to_string(i);
    }
    return text;
  };
  const std::string inputs =
      "int8_t flag, int8_t status, int16_t quantity, int32_t price, int8_t discount, int8_t tax, int16_t shipdate";
  std::string source = R"(
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cuda/std/cstdint>

using cudf::errc;
using cudf::size_type;
using namespace cuda::std;

extern "C" {
)";
  source += "__device__ errc q1_filter(size_type, bool* keep, " + inputs + ") {\n  *keep = shipdate <= " +
      std::to_string(dateLimit) + ";\n  return errc::SUCCESS;\n}\n";
  source += "__device__ errc q1_init(" + list("int64_t*", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *s" + std::to_string(i) + " = 0;\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc q1_update(size_type, " + list("int64_t*", "s") + ", " + inputs + ") {\n";
  source += "  const int group = flag * " + std::to_string(statuses) + " + status;\n";
  source += "  const int64_t discounted = int64_t{price} * (100 - discount);\n";
  source += "  const int64_t taxed = discounted * tax;\n";
  const char* values[kSums] = {"quantity", "price", "discounted", "taxed", "discount", "1"};
  for (int32_t g = 0; g < groups; ++g) {
    source += "  if (group == " + std::to_string(g) + ") {\n";
    for (int32_t k = 0; k < kSums; ++k) {
      source += "    *s" + std::to_string(g * kSums + k) + " += " + values[k] + ";\n";
    }
    source += "  }\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc q1_merge(" + list("int64_t*", "d") + ", " + list("int64_t", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *d" + std::to_string(i) + " += s" + std::to_string(i) + ";\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc q1_finalize(" + list("int64_t*", "o") + ", " + list("int64_t", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *o" + std::to_string(i) + " = s" + std::to_string(i) + ";\n";
  }
  source += "  return errc::SUCCESS;\n}\n}\n";
  return source;
}

void reduceGroups(
    QueryContext& ctx,
    RowRange rows,
    int64_t dateLimit,
    int32_t statuses,
    int32_t groups,
    std::map<std::pair<int8_t, int8_t>, Totals>& totals) {
  if (rows.size() <= 0) {
    return;
  }
  const auto& fragment = compileFragment(reductionSource(statuses, groups, dateLimit));
  const cudf::reduce_input inputs[] = {
      fixedView(ctx, "lineitem", "l_returnflag", rows),
      fixedView(ctx, "lineitem", "l_linestatus", rows),
      fixedView(ctx, "lineitem", "l_quantity", rows),
      fixedView(ctx, "lineitem", "l_extendedprice", rows),
      fixedView(ctx, "lineitem", "l_discount", rows),
      fixedView(ctx, "lineitem", "l_tax", rows),
      fixedView(ctx, "lineitem", "l_shipdate", rows),
  };
  const cudf::lto_fragment fragments[] = {{cudf::lto_binary_type::LTO_IR, fragment.view()}};
  const cudf::reduce_udf udf{fragments, "q1_init", "q1_update", "q1_merge", "q1_finalize", "q1_filter"};
  const std::vector<cudf::data_type> types(groups * kSums, cudf::data_type{cudf::type_id::INT64});
  auto columns = cudf::reduce_lto(udf, cudf::null_aware::NO, inputs, types, types, ctx.stream());
  std::vector<int64_t> sums(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    cudaMemcpyAsync(&sums[i], columns[i]->view().head(), sizeof(int64_t), cudaMemcpyDeviceToHost, ctx.stream());
  }
  cudaStreamSynchronize(ctx.stream());
  for (int32_t g = 0; g < groups; ++g) {
    const int64_t* s = &sums[g * kSums];
    if (s[5] == 0) {
      continue;
    }
    Totals& t = totals[{static_cast<int8_t>(g / statuses), static_cast<int8_t>(g % statuses)}];
    t.quantity += s[0];
    t.price += s[1];
    t.discounted += s[2];
    t.charge += static_cast<__int128>(s[2]) * 100 + s[3];
    t.discount += s[4];
    t.count += s[5];
  }
}

/// Most threads of the strided pass, each summing every threads-th row.
constexpr int64_t kStrideThreads = 1 << 18;

/// Rows each thread of the strided pass loads before summing any.
constexpr int32_t kStrideUnroll = 4;

/// The sums of every group, per thread, of one transform whose threads
/// stride over `rows` keeping the state in registers, starting from the
/// threads' sums `carried` of earlier rows if any.
std::unique_ptr<cudf::table> strideGroups(
    QueryContext& ctx,
    RowRange rows,
    int64_t dateLimit,
    int32_t statuses,
    int32_t groups,
    std::unique_ptr<cudf::table> carried) {
  const int32_t fields = groups * kSums;
  const int64_t threads = carried ? carried->num_rows() : std::min(kStrideThreads, rows.size());
  const char* values[kSums] = {"q[k]", "p[k]", "discounted", "taxed", "d[k]", "1"};
  std::string body;
  std::string inputs;
  std::vector<cudf::column_view> columns;
  for (int32_t i = 0; i < fields; ++i) {
    body += "  int64_t s" + std::to_string(i) + " = " + (carried ? "c" + std::to_string(i) : "0") + ";\n";
    if (carried) {
      inputs += (i ? ", int64_t c" : "int64_t c") + std::to_string(i);
      columns.push_back(carried->get_column(i).view());
    }
  }
  body += fmt::format(
      "  for (int64_t first = {0} + row; first < {1}; first += {2} * {3}) {{\n"
      "    int8_t f[{3}], st[{3}], d[{3}], tx[{3}];\n"
      "    int16_t q[{3}], sh[{3}];\n"
      "    int32_t p[{3}];\n"
      "#pragma unroll\n"
      "    for (int k = 0; k < {3}; ++k) {{\n"
      "      const int64_t l = first + k * int64_t{{{2}}};\n"
      "      const bool inside = l < {1};\n"
      "      sh[k] = inside ? shipdate[l] : int16_t{{32767}};\n"
      "      f[k] = inside ? flag[l] : int8_t{{0}};\n"
      "      st[k] = inside ? status[l] : int8_t{{0}};\n"
      "      q[k] = inside ? quantity[l] : int16_t{{0}};\n"
      "      p[k] = inside ? price[l] : 0;\n"
      "      d[k] = inside ? discount[l] : int8_t{{0}};\n"
      "      tx[k] = inside ? tax[l] : int8_t{{0}};\n"
      "    }}\n"
      "#pragma unroll\n"
      "    for (int k = 0; k < {3}; ++k) {{\n"
      "      if (sh[k] > {4}) continue;\n"
      "      const int group = f[k] * {5} + st[k];\n"
      "      const int64_t discounted = int64_t{{p[k]}} * (100 - d[k]);\n"
      "      const int64_t taxed = discounted * tx[k];\n",
      0,
      rows.size(),
      threads,
      kStrideUnroll,
      dateLimit,
      statuses);
  for (int32_t g = 0; g < groups; ++g) {
    body += "      if (group == " + std::to_string(g) + ") {\n";
    for (int32_t k = 0; k < kSums; ++k) {
      body += "        s" + std::to_string(g * kSums + k) + " += " + values[k] + ";\n";
    }
    body += "      }\n";
  }
  body += "    }\n";
  body += "  }\n";
  std::vector<std::pair<cudf::type_id, std::string>> outputs;
  for (int32_t i = 0; i < fields; ++i) {
    outputs.emplace_back(cudf::type_id::INT64, "o" + std::to_string(i));
    body += "  *o" + std::to_string(i) + " = s" + std::to_string(i) + ";\n";
  }
  std::vector<ArrayInput> arrays = {
      {"lineitem", "l_returnflag", "flag"},
      {"lineitem", "l_linestatus", "status"},
      {"lineitem", "l_quantity", "quantity"},
      {"lineitem", "l_extendedprice", "price"},
      {"lineitem", "l_discount", "discount"},
      {"lineitem", "l_tax", "tax"},
      {"lineitem", "l_shipdate", "shipdate"}};
  for (auto& array : arrays) {
    array.first = rows.begin;
  }
  return transformRows(ctx, threads, inputs, columns, arrays, outputs, body);
}

/// Adds the threads' sums `partial` of strideGroups to `totals`, in one
/// reduction.
void addStrideSums(
    QueryContext& ctx,
    const cudf::table_view& partial,
    int32_t statuses,
    int32_t groups,
    std::map<std::pair<int8_t, int8_t>, Totals>& totals) {
  GroupedSums total;
  total.filter = "true";
  std::vector<cudf::column_view> columns;
  for (int32_t i = 0; i < partial.num_columns(); ++i) {
    total.inputs += (i ? ", int64_t o" : "int64_t o") + std::to_string(i);
    total.values.push_back("o" + std::to_string(i));
    columns.push_back(partial.column(i));
  }
  const std::vector<int64_t> sums = sumGroups(ctx, total, columns);
  for (int32_t g = 0; g < groups; ++g) {
    const int64_t* s = &sums[g * kSums];
    if (s[5] == 0) {
      continue;
    }
    Totals& t = totals[{static_cast<int8_t>(g / statuses), static_cast<int8_t>(g % statuses)}];
    t.quantity += s[0];
    t.price += s[1];
    t.discounted += s[2];
    t.charge += static_cast<__int128>(s[2]) * 100 + s[3];
    t.discount += s[4];
    t.count += s[5];
  }
}

} // namespace

Result q1(QueryContext& ctx) {
  const int64_t dateLimit = ctx.param("shipdate_max", days(1998, 9, 2));
  const ZoneSplit split = splitZones(ctx, "lineitem", "l_shipdate", std::numeric_limits<int32_t>::min(), dateLimit);

  const auto& flags = ctx.meta("lineitem", "l_returnflag").dictionary;
  const auto& statuses = ctx.meta("lineitem", "l_linestatus").dictionary;
  std::map<std::pair<int8_t, int8_t>, Totals> groups;
  const int64_t domain = static_cast<int64_t>(flags.size() * statuses.size());
  const auto statusCount = static_cast<int32_t>(statuses.size());
  if (domain <= kMostReducedGroups && useJit() && !std::getenv("SV_CUDF_GROUPBY") && !std::getenv("SV_CUDF_REDUCE")) {
    // Each batch's threads start from the last batch's sums.
    std::unique_ptr<cudf::table> partial;
    forEachBatch(ctx, split.some, split.some.size(), [&](RowRange batch) {
      partial = strideGroups(ctx, batch, dateLimit, statusCount, static_cast<int32_t>(domain), std::move(partial));
    });
    if (partial) {
      addStrideSums(ctx, partial->view(), statusCount, static_cast<int32_t>(domain), groups);
    }
  } else if (domain <= kMostReducedGroups && useJit() && !std::getenv("SV_CUDF_GROUPBY")) {
    forEachBatch(ctx, split.some, split.some.size(), [&](RowRange batch) {
      reduceGroups(ctx, batch, dateLimit, statusCount, static_cast<int32_t>(domain), groups);
    });
  } else {
    auto forBatches = [&](RowRange rows, auto&& body) { forEachBatch(ctx, rows, kBatchRows, body); };
    forBatches(split.all, [&](RowRange batch) { aggregate(ctx, batch, nullptr, 0, groups); });
    for (RowRange rows : {RowRange{split.some.begin, split.all.begin}, RowRange{split.all.end, split.some.end}}) {
      forBatches(rows, [&](RowRange batch) {
        auto [mask, nullCount] = shipdateMask(ctx, batch, dateLimit);
        aggregate(ctx, batch, mask->data(), nullCount, groups);
      });
    }
  }

  Result result(kSchema);
  for (const auto& [key, t] : groups) {
    if (t.count == 0) {
      continue;
    }
    const double count = t.count;
    result.strings[0].push_back(flags.at(key.first));
    result.strings[1].push_back(statuses.at(key.second));
    result.decimals[2].push_back(t.quantity);
    result.decimals[3].push_back(t.price);
    result.decimals[4].push_back(t.discounted);
    result.decimals[5].push_back(t.charge);
    result.doubles[6].push_back(static_cast<double>(t.quantity) / 100.0 / count);
    result.doubles[7].push_back(static_cast<double>(t.price) / 100.0 / count);
    result.doubles[8].push_back(static_cast<double>(t.discount) / 100.0 / count);
    result.ints[9].push_back(t.count);
  }
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
