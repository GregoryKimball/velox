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

#include <cudf/reduction.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#include "velox/experimental/supervelox/common/Cuda.h"

namespace facebook::velox::supervelox::cudf_backend {

Result q1(QueryContext& ctx);
Result q4(QueryContext& ctx);
Result q6(QueryContext& ctx);
Result q12(QueryContext& ctx);
Result q14(QueryContext& ctx);
Result q19(QueryContext& ctx);

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

void registerCudfBackend() {
  cudaMemPool_t pool;
  SV_CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
  rmm::mr::set_current_device_resource(rmm::mr::cuda_async_view_memory_resource(pool));
  registerImplementation(Backend::kCudf, 1, q1);
  registerImplementation(Backend::kCudf, 4, q4);
  registerImplementation(Backend::kCudf, 6, q6);
  registerImplementation(Backend::kCudf, 12, q12);
  registerImplementation(Backend::kCudf, 14, q14);
  registerImplementation(Backend::kCudf, 19, q19);
}

} // namespace facebook::velox::supervelox::cudf_backend
