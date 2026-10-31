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
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {

size_t Result::rows() const {
  if (fields.empty()) {
    return 0;
  }
  switch (fields[0].type) {
    case ResultType::kDecimal:
      return decimals[0].size();
    case ResultType::kDouble:
      return doubles[0].size();
    case ResultType::kVarchar:
      return dictionaries[0] ? codes[0].size() : strings[0].size();
    default:
      return ints[0].size();
  }
}

namespace {

template <typename From, typename To>
__global__ void widenKernel(const From* in, To* out, int64_t rows) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < rows;
       i += (int64_t)gridDim.x * blockDim.x) {
    out[i] = in[i];
  }
}

template <typename To>
void widen(const Column& column, const void* in, To* out, cudaStream_t stream) {
  const int blocks = 4096;
  switch (column.dtype) {
    case DType::kInt8:
      widenKernel<<<blocks, 256, 0, stream>>>(static_cast<const int8_t*>(in), out, column.rows);
      break;
    case DType::kInt16:
      widenKernel<<<blocks, 256, 0, stream>>>(static_cast<const int16_t*>(in), out, column.rows);
      break;
    case DType::kInt32:
      widenKernel<<<blocks, 256, 0, stream>>>(static_cast<const int32_t*>(in), out, column.rows);
      break;
    default:
      throw std::runtime_error("cannot widen " + column.name);
  }
  SV_CUDA(cudaGetLastError());
}

template <typename T>
__global__ void zoneKernel(const T* in, int64_t rows, int2* zones) {
  const int64_t begin = blockIdx.x * (int64_t)QueryContext::kZoneRows;
  const int64_t end = min(rows, begin + QueryContext::kZoneRows);
  int lo = INT_MAX;
  int hi = INT_MIN;
  for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    int v = in[i];
    lo = min(lo, v);
    hi = max(hi, v);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    lo = min(lo, __shfl_down_sync(0xffffffff, lo, offset));
    hi = max(hi, __shfl_down_sync(0xffffffff, hi, offset));
  }
  __shared__ int los[32];
  __shared__ int his[32];
  if (threadIdx.x % 32 == 0) {
    los[threadIdx.x / 32] = lo;
    his[threadIdx.x / 32] = hi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < blockDim.x / 32; ++w) {
      lo = min(lo, los[w]);
      hi = max(hi, his[w]);
    }
    zones[blockIdx.x] = make_int2(lo, hi);
  }
}

} // namespace

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t batchRowsSetting = 0;

} // namespace

void setBatchRows(int64_t rows) {
  batchRowsSetting = rows;
}

int64_t batchRows() {
  return batchRowsSetting;
}

QueryContext::QueryContext(
    ColumnStore& store,
    DeviceCache& cache,
    cudaStream_t stream,
    const QueryParams& params)
    : store_(store),
      cache_(cache),
      stream_(stream),
      params_(params),
      trace_(getenv("SV_TRACE") != nullptr),
      lastMark_(nowMs()) {}

void QueryContext::mark(const char* phase) {
  if (!trace_) {
    return;
  }
  cudaStreamSynchronize(stream_);
  const double now = nowMs();
  fprintf(stderr, "  %-24s %8.3f ms\n", phase, now - lastMark_);
  lastMark_ = now;
}

QueryContext::~QueryContext() {
  for (void* buffer : scratch_) {
    cudaFreeAsync(buffer, stream_);
  }
  cudaStreamSynchronize(stream_);
}

const void* QueryContext::column(const std::string& table, const std::string& name, size_t width) {
  const Column& column = store_.column(table, name);
  const size_t stored = dtypeSize(column.dtype);
  const std::string key = table + "." + name;
  if (stored == width) {
    return cache_.get(key, column.data, column.bytes, stream_);
  }
  if (stored > width) {
    throw std::runtime_error(key + " is wider than requested");
  }
  return cache_.derived(
      key + "@" + std::to_string(width),
      column.rows * width,
      [&](void* out) {
        switch (width) {
          case 2:
            widen(column, column.data, static_cast<int16_t*>(out), stream_);
            break;
          case 4:
            widen(column, column.data, static_cast<int32_t*>(out), stream_);
            break;
          default:
            widen(column, column.data, static_cast<int64_t*>(out), stream_);
        }
      },
      stream_);
}

const int64_t* QueryContext::offsets(const std::string& table, const std::string& name) {
  const Column& column = store_.column(table, name);
  return static_cast<const int64_t*>(cache_.get(
      table + "." + name + ".offsets", column.offsets, (column.rows + 1) * sizeof(int64_t), stream_));
}

const char* QueryContext::chars(const std::string& table, const std::string& name) {
  const Column& column = store_.column(table, name);
  return static_cast<const char*>(cache_.get(table + "." + name, column.data, column.bytes, stream_));
}

const int2* QueryContext::zones(const std::string& table, const std::string& name) {
  const Column& column = store_.column(table, name);
  const int64_t count = (column.rows + kZoneRows - 1) / kZoneRows;
  return static_cast<const int2*>(cache_.derived(
      table + "." + name + "#zones",
      count * sizeof(int2),
      [&](void* out) {
        auto* zones = static_cast<int2*>(out);
        const void* in = column.data;
        switch (column.dtype) {
          case DType::kInt8:
            zoneKernel<<<count, 256, 0, stream_>>>(static_cast<const int8_t*>(in), column.rows, zones);
            break;
          case DType::kInt16:
            zoneKernel<<<count, 256, 0, stream_>>>(static_cast<const int16_t*>(in), column.rows, zones);
            break;
          case DType::kInt32:
            zoneKernel<<<count, 256, 0, stream_>>>(static_cast<const int32_t*>(in), column.rows, zones);
            break;
          default:
            throw std::runtime_error("no zones for int64 " + name);
        }
        SV_CUDA(cudaGetLastError());
      },
      stream_));
}

void* QueryContext::scratchBytes(size_t bytes) {
  void* buffer = nullptr;
  SV_CUDA(cudaMallocAsync(&buffer, std::max<size_t>(bytes, 16), stream_));
  scratch_.push_back(buffer);
  return buffer;
}

} // namespace facebook::velox::supervelox
