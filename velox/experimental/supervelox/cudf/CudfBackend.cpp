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

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#include "velox/experimental/supervelox/common/Cuda.h"

namespace facebook::velox::supervelox::cudf_backend {

Result q1(QueryContext& ctx);
Result q2(QueryContext& ctx);
Result q3(QueryContext& ctx);
Result q4(QueryContext& ctx);
Result q5(QueryContext& ctx);
Result q6(QueryContext& ctx);
Result q7(QueryContext& ctx);
Result q8(QueryContext& ctx);
Result q9(QueryContext& ctx);
Result q10(QueryContext& ctx);
Result q11(QueryContext& ctx);
Result q12(QueryContext& ctx);
Result q13(QueryContext& ctx);
Result q14(QueryContext& ctx);
Result q15(QueryContext& ctx);
Result q16(QueryContext& ctx);
Result q17(QueryContext& ctx);
Result q18(QueryContext& ctx);
Result q19(QueryContext& ctx);
Result q20(QueryContext& ctx);
Result q21(QueryContext& ctx);
Result q22(QueryContext& ctx);

namespace {

cudf::type_id typeId(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return cudf::type_id::INT8;
    case DType::kInt16:
      return cudf::type_id::INT16;
    case DType::kInt32:
      return cudf::type_id::INT32;
    case DType::kInt64:
      return cudf::type_id::INT64;
  }
  throw std::logic_error("unknown dtype");
}

const void* storedColumn(QueryContext& ctx, const std::string& table, const std::string& name, DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return ctx.col<int8_t>(table, name);
    case DType::kInt16:
      return ctx.col<int16_t>(table, name);
    case DType::kInt32:
      return ctx.col<int32_t>(table, name);
    case DType::kInt64:
      return ctx.col<int64_t>(table, name);
  }
  throw std::logic_error("unknown dtype");
}

/// Host copies of zone maps, by device address and zone count.
const std::vector<int2>& hostZones(QueryContext& ctx, const std::string& table, const std::string& name) {
  static std::mutex mutex;
  static std::map<std::pair<const int2*, int64_t>, std::vector<int2>> copies;
  const int2* device = ctx.zones(table, name);
  const int64_t count = (ctx.rows(table) + QueryContext::kZoneRows - 1) / QueryContext::kZoneRows;
  std::lock_guard<std::mutex> lock(mutex);
  auto& copy = copies[{device, count}];
  if (copy.empty() && count > 0) {
    copy = ctx.download(device, count);
  }
  return copy;
}

} // namespace

cudf::column_view fixedView(QueryContext& ctx, const std::string& table, const std::string& name, RowRange rows) {
  const Column& column = ctx.meta(table, name);
  if (column.kind == Kind::kString) {
    throw std::invalid_argument(table + "." + name + " is not fixed-width");
  }
  if (rows.end > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("cuDF columns hold at most 2^31 - 1 rows");
  }
  const auto* data = static_cast<const char*>(storedColumn(ctx, table, name, column.dtype));
  return cudf::column_view(
      cudf::data_type{typeId(column.dtype)},
      static_cast<cudf::size_type>(rows.size()),
      data + rows.begin * dtypeSize(column.dtype),
      nullptr,
      0);
}

ZoneSplit
splitZones(QueryContext& ctx, const std::string& table, const std::string& name, int64_t low, int64_t high) {
  const auto& zones = hostZones(ctx, table, name);
  const int64_t rows = ctx.rows(table);
  const int64_t zoneRows = QueryContext::kZoneRows;
  int64_t first = -1;
  int64_t last = -1;
  int64_t runStart = -1;
  int64_t bestStart = 0;
  int64_t bestEnd = 0;
  for (int64_t zone = 0; zone < static_cast<int64_t>(zones.size()); ++zone) {
    const int2 range = zones[zone];
    const bool any = range.y >= low && range.x <= high;
    const bool all = range.x >= low && range.y <= high;
    if (any) {
      first = first < 0 ? zone : first;
      last = zone;
    }
    if (all) {
      runStart = runStart < 0 ? zone : runStart;
      if (zone + 1 - runStart > bestEnd - bestStart) {
        bestStart = runStart;
        bestEnd = zone + 1;
      }
    } else {
      runStart = -1;
    }
  }
  ZoneSplit split;
  if (first < 0) {
    return split;
  }
  auto rowOf = [&](int64_t zone) { return std::min(rows, zone * zoneRows); };
  split.some = {rowOf(first), rowOf(last + 1)};
  split.all = {rowOf(bestStart), rowOf(bestEnd)};
  return split;
}

cudf::column_view deviceView(cudf::type_id type, const void* data, int64_t rows) {
  if (rows > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("cuDF columns hold at most 2^31 - 1 rows");
  }
  return cudf::column_view(cudf::data_type{type}, static_cast<cudf::size_type>(rows), data, nullptr, 0);
}

std::string codeTest(const std::string& code, const CodeSet& codes) {
  std::string test;
  for (int32_t c = 0; c < 256; ++c) {
    if (codes.contains(c)) {
      test += (test.empty() ? "" : " || ") + code + " == " + std::to_string(c);
    }
  }
  return test.empty() ? "false" : "(" + test + ")";
}

std::vector<int64_t>
sumGroups(QueryContext& ctx, const GroupedSums& sums, const std::vector<cudf::column_view>& columns) {
  const int32_t fields = sums.groups * static_cast<int32_t>(sums.values.size());
  auto list = [&](const std::string& type, const std::string& prefix) {
    std::string text;
    for (int32_t i = 0; i < fields; ++i) {
      text += (i ? ", " : "") + type + " " + prefix + std::to_string(i);
    }
    return text;
  };
  std::string source = R"(
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cuda/std/cstdint>

using cudf::errc;
using cudf::size_type;
using namespace cuda::std;

extern "C" {
)";
  source += "__device__ errc sums_filter(size_type, bool* keep, " + sums.inputs + ") {\n" + sums.prelude +
      "  *keep = " + sums.filter + ";\n  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_init(" + list("int64_t*", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *s" + std::to_string(i) + " = 0;\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_update(size_type, " + list("int64_t*", "s") + ", " + sums.inputs + ") {\n" +
      sums.prelude + "  const int group = " + sums.group + ";\n";
  for (size_t k = 0; k < sums.values.size(); ++k) {
    source += "  const int64_t v" + std::to_string(k) + " = " + sums.values[k] + ";\n";
  }
  for (int32_t g = 0; g < sums.groups; ++g) {
    source += sums.groups == 1 ? "  {\n" : "  if (group == " + std::to_string(g) + ") {\n";
    for (size_t k = 0; k < sums.values.size(); ++k) {
      source += "    *s" + std::to_string(g * sums.values.size() + k) + " += v" + std::to_string(k) + ";\n";
    }
    source += "  }\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_merge(" + list("int64_t*", "d") + ", " + list("int64_t", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *d" + std::to_string(i) + " += s" + std::to_string(i) + ";\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_finalize(" + list("int64_t*", "o") + ", " + list("int64_t", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *o" + std::to_string(i) + " = s" + std::to_string(i) + ";\n";
  }
  source += "  return errc::SUCCESS;\n}\n}\n";

  std::vector<int64_t> totals(fields, 0);
  if (columns.empty() || columns[0].size() == 0) {
    return totals;
  }
  const auto& fragment = compileFragment(source);
  const std::vector<cudf::reduce_input> inputs(columns.begin(), columns.end());
  const cudf::lto_fragment fragments[] = {{cudf::lto_binary_type::LTO_IR, fragment.view()}};
  const cudf::reduce_udf udf{fragments, "sums_init", "sums_update", "sums_merge", "sums_finalize", "sums_filter"};
  const std::vector<cudf::data_type> types(fields, cudf::data_type{cudf::type_id::INT64});
  auto results = cudf::reduce_lto(udf, cudf::null_aware::NO, inputs, types, types, ctx.stream());
  for (int32_t i = 0; i < fields; ++i) {
    SV_CUDA(cudaMemcpyAsync(
        &totals[i], results[i]->view().head(), sizeof(int64_t), cudaMemcpyDeviceToHost, ctx.stream()));
  }
  SV_CUDA(cudaStreamSynchronize(ctx.stream()));
  return totals;
}

const cudf::compiled_fragment& compileFragment(const std::string& source) {
  static std::mutex mutex;
  static std::map<std::string, cudf::compiled_fragment> fragments;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = fragments.find(source);
  if (it == fragments.end()) {
    it = fragments.emplace(source, cudf::compile_cuda_to_lto_ir(source.c_str(), true, {}, {}, {})).first;
  }
  return it->second;
}

const char* deviceType(cudf::type_id type) {
  switch (type) {
    case cudf::type_id::BOOL8:
      return "bool";
    case cudf::type_id::INT8:
      return "int8_t";
    case cudf::type_id::INT16:
      return "int16_t";
    case cudf::type_id::INT32:
      return "int32_t";
    case cudf::type_id::INT64:
      return "int64_t";
    case cudf::type_id::FLOAT64:
      return "double";
    default:
      throw std::invalid_argument("no device type for this cuDF type");
  }
}

std::unique_ptr<cudf::column> computeColumn(
    QueryContext& ctx,
    const std::string& inputs,
    const std::string& expression,
    cudf::type_id type,
    const std::vector<cudf::column_view>& columns,
    const std::string& prelude) {
  const std::string source = std::string(R"(
#include <cuda/std/cstdint>

using namespace cuda::std;

extern "C" __device__ int transform()") +
      deviceType(type) + "* out, " + inputs + ") {\n" + prelude + "  *out = " + expression + ";\n  return 0;\n}\n";
  const std::vector<cudf::transform_input> in(columns.begin(), columns.end());
  const cudf::transform_output out[] = {{cudf::data_type{type}, cudf::output_nullability::ALL_VALID}};
  auto table = cudf::transform_lto(
      compileFragment(source).view(),
      cudf::lto_binary_type::LTO_IR,
      cudf::null_aware::NO,
      std::nullopt,
      in,
      out,
      {},
      std::nullopt,
      ctx.stream());
  return std::move(table->release()[0]);
}

std::unique_ptr<cudf::column> keyRows(QueryContext& ctx, const cudf::column_view& keys) {
  const cudf::numeric_scalar<int32_t> one(1, true, ctx.stream());
  return cudf::binary_operation(keys, one, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::INT32}, ctx.stream());
}

std::unique_ptr<cudf::table> gatherColumns(
    QueryContext& ctx,
    const std::string& table,
    const std::vector<std::string>& names,
    const cudf::column_view& rows) {
  std::vector<cudf::column_view> columns;
  for (const auto& name : names) {
    columns.push_back(fixedView(ctx, table, name, {0, ctx.rows(table)}));
  }
  return cudf::gather(cudf::table_view(columns), rows, cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
}

OrderLines orderLines(QueryContext& ctx, int64_t begin, int64_t end) {
  OrderLines lines;
  lines.begin = begin;
  lines.end = end;
  if (end <= begin) {
    return lines;
  }
  const uint32_t* starts = orderLineStarts(ctx);
  const auto bounds = ctx.download(starts + begin, 1);
  const auto last = ctx.download(starts + end, 1);
  lines.lines = {bounds[0], last[0]};
  lines.counts = cudf::binary_operation(
      deviceView(cudf::type_id::INT32, starts + begin + 1, end - begin),
      deviceView(cudf::type_id::INT32, starts + begin, end - begin),
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      ctx.stream());
  const cudf::numeric_scalar<int32_t> base(static_cast<int32_t>(lines.lines.begin), true, ctx.stream());
  lines.offsets = cudf::binary_operation(
      deviceView(cudf::type_id::INT32, starts + begin, end - begin + 1),
      base,
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      ctx.stream());
  lines.groups = std::make_unique<cudf::groupby::groupby>(
      cudf::table_view({fixedView(ctx, "lineitem", "l_orderkey", lines.lines)}),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES);
  return lines;
}

std::vector<std::pair<int64_t, int64_t>>
orderChunks(QueryContext& ctx, int64_t begin, int64_t end, int64_t maxLines) {
  std::vector<std::pair<int64_t, int64_t>> chunks;
  if (end <= begin) {
    return chunks;
  }
  // Orders hold at most seven lines, so a run of maxLines / 8 orders fits.
  const int64_t orders = std::max<int64_t>(1, maxLines / 8);
  for (int64_t first = begin; first < end; first += orders) {
    chunks.emplace_back(first, std::min(end, first + orders));
  }
  return chunks;
}

std::unique_ptr<cudf::table>
perLine(QueryContext& ctx, const OrderLines& lines, const std::vector<cudf::column_view>& columns) {
  return cudf::repeat(cudf::table_view(columns), lines.counts->view(), ctx.stream());
}

namespace {

/// 'aggregation' of 'values' over each order's lines. A sorted groupby over
/// l_orderkey finds the orders' boundaries once per OrderLines; segmented
/// reductions spend a thread block on each order's few lines.
std::unique_ptr<cudf::column> perOrder(
    QueryContext& ctx,
    const cudf::column_view& values,
    const OrderLines& lines,
    std::unique_ptr<cudf::groupby_aggregation> aggregation) {
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(std::move(aggregation));
  auto [keys, results] = lines.groups->aggregate(requests, ctx.stream());
  if (keys->num_rows() != lines.end - lines.begin) {
    throw std::runtime_error("orders without lines");
  }
  return std::move(results[0].results[0]);
}

} // namespace

std::unique_ptr<cudf::column> sumPerOrder(QueryContext& ctx, const cudf::column_view& values, const OrderLines& lines) {
  auto sums = perOrder(ctx, values, lines, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  return sums->type().id() == cudf::type_id::INT64 ? std::move(sums)
                                                     : cudf::cast(sums->view(), cudf::data_type{cudf::type_id::INT64}, ctx.stream());
}

std::unique_ptr<cudf::column>
extremePerOrder(QueryContext& ctx, const cudf::column_view& values, const OrderLines& lines, bool maximum) {
  return perOrder(
      ctx,
      values,
      lines,
      maximum ? cudf::make_max_aggregation<cudf::groupby_aggregation>()
              : cudf::make_min_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<cudf::column>
partsuppRows(QueryContext& ctx, const cudf::column_view& partkeys, const cudf::column_view& suppkeys) {
  const std::string part = std::string(deviceType(partkeys.type().id())) + " part";
  std::vector<std::unique_ptr<cudf::table>> slots;
  for (int slot = 0; slot < 4; ++slot) {
    auto rows = computeColumn(
        ctx, part, "(int32_t{part} - 1) * 4 + " + std::to_string(slot), cudf::type_id::INT32, {partkeys});
    slots.push_back(gatherColumns(ctx, "partsupp", {"ps_suppkey"}, rows->view()));
  }
  const std::string supplier = deviceType(slots[0]->get_column(0).type().id());
  auto rows = computeColumn(
      ctx,
      part + ", " + deviceType(suppkeys.type().id()) + " supplier, " + supplier + " s0, " + supplier + " s1, " + supplier +
          " s2, " + supplier + " s3",
      "supplier == s0 ? base : supplier == s1 ? base + 1 : supplier == s2 ? base + 2 : supplier == s3 ? base + 3 : -1",
      cudf::type_id::INT32,
      {partkeys,
       suppkeys,
       slots[0]->get_column(0).view(),
       slots[1]->get_column(0).view(),
       slots[2]->get_column(0).view(),
       slots[3]->get_column(0).view()},
      "  const int32_t base = (int32_t{part} - 1) * 4;\n");
  if (rows->size() > 0) {
    auto lowest =
        cudf::reduce(rows->view(), *cudf::make_min_aggregation<cudf::reduce_aggregation>(), rows->type(), ctx.stream());
    if (static_cast<cudf::numeric_scalar<int32_t>&>(*lowest).value(ctx.stream()) < 0) {
      throw std::runtime_error("lineitem (part, supplier) missing from the partsupp layout");
    }
  }
  return rows;
}

std::unique_ptr<cudf::table> concatenateTables(QueryContext& ctx, const std::vector<std::unique_ptr<cudf::table>>& pieces) {
  std::vector<cudf::table_view> views;
  for (const auto& piece : pieces) {
    views.push_back(piece->view());
  }
  return cudf::concatenate(views, ctx.stream());
}

std::unique_ptr<cudf::column>
likeColumn(QueryContext& ctx, const std::string& table, const std::string& column, const std::string& pattern, RowRange rows) {
  return cudf::strings::like(
      cudf::strings_column_view(stringView(ctx, table, column, rows)), pattern, "", ctx.stream());
}

cudf::column_view stringView(QueryContext& ctx, const std::string& table, const std::string& name, RowRange rows) {
  if (rows.size() > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("cuDF columns hold at most 2^31 - 1 rows");
  }
  const int64_t* offsets = ctx.offsets(table, name);
  const cudf::column_view offsetView(
      cudf::data_type{cudf::type_id::INT64}, static_cast<cudf::size_type>(rows.size() + 1), offsets + rows.begin, nullptr, 0);
  return cudf::column_view(
      cudf::data_type{cudf::type_id::STRING},
      static_cast<cudf::size_type>(rows.size()),
      ctx.chars(table, name),
      nullptr,
      0,
      0,
      {offsetView});
}

std::vector<int32_t> firstSorted(
    QueryContext& ctx,
    const cudf::table_view& keys,
    const std::vector<cudf::order>& orders,
    int64_t limit) {
  auto order = cudf::sorted_order(keys, orders, {}, ctx.stream());
  const auto count = std::min<int64_t>(limit, order->size());
  return download<int32_t>(ctx, cudf::slice(order->view(), {0, static_cast<cudf::size_type>(count)})[0]);
}

std::unique_ptr<cudf::table> firstRows(
    QueryContext& ctx,
    const cudf::table_view& values,
    const cudf::table_view& keys,
    const std::vector<cudf::order>& orders,
    int64_t limit) {
  auto order = cudf::sorted_order(keys, orders, {}, ctx.stream());
  const auto count = static_cast<cudf::size_type>(std::min<int64_t>(limit, order->size()));
  return cudf::gather(values, cudf::slice(order->view(), {0, count})[0], cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
}

std::unique_ptr<cudf::column> sequence(QueryContext& ctx, int64_t count, int32_t first, int32_t step) {
  const cudf::numeric_scalar<int32_t> init(first, true, ctx.stream());
  const cudf::numeric_scalar<int32_t> stride(step, true, ctx.stream());
  return cudf::sequence(static_cast<cudf::size_type>(count), init, stride, ctx.stream());
}

std::unique_ptr<cudf::table> selectRows(QueryContext& ctx, const cudf::table_view& table, const cudf::column_view& mask) {
  return cudf::apply_boolean_mask(table, mask, ctx.stream());
}

std::vector<int64_t> downloadInts(QueryContext& ctx, const cudf::column_view& column) {
  if (column.type().id() == cudf::type_id::INT64) {
    return download<int64_t>(ctx, column);
  }
  auto wide = cudf::cast(column, cudf::data_type{cudf::type_id::INT64}, ctx.stream());
  return download<int64_t>(ctx, wide->view());
}

Grouped groupBy(
    QueryContext& ctx,
    const cudf::table_view& keys,
    const std::vector<std::pair<cudf::column_view, Aggregate>>& aggregates) {
  std::vector<cudf::groupby::aggregation_request> requests(aggregates.size());
  for (size_t i = 0; i < aggregates.size(); ++i) {
    requests[i].values = aggregates[i].first;
    switch (aggregates[i].second) {
      case Aggregate::kSum:
        requests[i].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kMin:
        requests[i].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kMax:
        requests[i].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kCount:
        requests[i].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kDistinct:
        requests[i].aggregations.push_back(cudf::make_nunique_aggregation<cudf::groupby_aggregation>());
        break;
    }
  }
  cudf::groupby::groupby grouping(keys);
  auto [groupKeys, results] = grouping.aggregate(requests, ctx.stream());
  Grouped grouped;
  grouped.keys = std::move(groupKeys);
  for (auto& result : results) {
    grouped.values.push_back(std::move(result.results[0]));
  }
  return grouped;
}

void registerCudfBackend() {
  cudaMemPool_t pool;
  SV_CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
  rmm::mr::set_current_device_resource(rmm::mr::cuda_async_view_memory_resource(pool));
  registerImplementation(Backend::kCudf, 1, q1);
  registerImplementation(Backend::kCudf, 2, q2);
  registerImplementation(Backend::kCudf, 3, q3);
  registerImplementation(Backend::kCudf, 4, q4);
  registerImplementation(Backend::kCudf, 5, q5);
  registerImplementation(Backend::kCudf, 6, q6);
  registerImplementation(Backend::kCudf, 7, q7);
  registerImplementation(Backend::kCudf, 8, q8);
  registerImplementation(Backend::kCudf, 9, q9);
  registerImplementation(Backend::kCudf, 10, q10);
  registerImplementation(Backend::kCudf, 11, q11);
  registerImplementation(Backend::kCudf, 12, q12);
  registerImplementation(Backend::kCudf, 13, q13);
  registerImplementation(Backend::kCudf, 14, q14);
  registerImplementation(Backend::kCudf, 15, q15);
  registerImplementation(Backend::kCudf, 16, q16);
  registerImplementation(Backend::kCudf, 17, q17);
  registerImplementation(Backend::kCudf, 18, q18);
  registerImplementation(Backend::kCudf, 19, q19);
  registerImplementation(Backend::kCudf, 20, q20);
  registerImplementation(Backend::kCudf, 21, q21);
  registerImplementation(Backend::kCudf, 22, q22);
}

} // namespace facebook::velox::supervelox::cudf_backend
