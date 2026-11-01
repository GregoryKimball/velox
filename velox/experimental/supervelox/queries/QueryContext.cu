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
#include <cstring>

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
void widen(const Column& column, const void* in, To* out, cudaStream_t stream, int64_t rows = -1) {
  const int blocks = 4096;
  if (rows < 0) {
    rows = column.rows;
  }
  switch (column.dtype) {
    case DType::kInt8:
      widenKernel<<<blocks, 256, 0, stream>>>(static_cast<const int8_t*>(in), out, rows);
      break;
    case DType::kInt16:
      widenKernel<<<blocks, 256, 0, stream>>>(static_cast<const int16_t*>(in), out, rows);
      break;
    case DType::kInt32:
      widenKernel<<<blocks, 256, 0, stream>>>(static_cast<const int32_t*>(in), out, rows);
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
  endBatch();
  for (void* buffer : scratch_) {
    cudaFreeAsync(buffer, stream_);
  }
  cudaStreamSynchronize(stream_);
  if (staging_ != nullptr) {
    cudaFreeHost(staging_);
  }
}

void* QueryContext::batchBytes(size_t bytes) {
  void* buffer = nullptr;
  SV_CUDA(cudaMallocAsync(&buffer, std::max<size_t>(bytes, 16), stream_));
  batch_.push_back(buffer);
  return buffer;
}

void QueryContext::endBatch() {
  for (void* buffer : batch_) {
    cudaFreeAsync(buffer, stream_);
  }
  batch_.clear();
}

void* QueryContext::staging(size_t bytes) {
  if (bytes > stagingBytes_) {
    if (staging_ != nullptr) {
      SV_CUDA(cudaStreamSynchronize(stream_));
      cudaFreeHost(staging_);
    }
    stagingBytes_ = std::max(bytes, stagingBytes_ * 2);
    SV_CUDA(cudaHostAlloc(&staging_, stagingBytes_, cudaHostAllocDefault));
  }
  return staging_;
}

void QueryContext::upload(void* to, const void* from, size_t bytes, bool pinned) {
  if (!pinned) {
    void* stage = staging(bytes);
    std::memcpy(stage, from, bytes);
    from = stage;
  }
  SV_CUDA(cudaMemcpyAsync(to, from, bytes, cudaMemcpyHostToDevice, stream_));
  // The staging buffer is reused by the next upload.
  SV_CUDA(cudaStreamSynchronize(stream_));
}

const void* QueryContext::range(
    const std::string& table,
    const std::string& name,
    int64_t first,
    int64_t last,
    size_t width) {
  if (batchRows() == 0) {
    return column(table, name, width);
  }
  const Column& meta = store_.metadata(table, name);
  const size_t stored = dtypeSize(meta.dtype);
  if (stored > width) {
    throw std::runtime_error(table + "." + name + " is wider than requested");
  }
  const int64_t count = std::max<int64_t>(last - first, 0);
  char* out = static_cast<char*>(batchBytes(count * width));
  char* in = stored == width ? out : static_cast<char*>(batchBytes(count * stored));
  if (meta.data != nullptr) {
    upload(in, static_cast<const char*>(meta.data) + first * stored, count * stored, true);
  } else {
    void* stage = staging(count * stored);
    store_.read(table, name, first, count, stage);
    upload(in, stage, count * stored, true);
  }
  if (stored != width) {
    switch (width) {
      case 2:
        widen(meta, in, reinterpret_cast<int16_t*>(out), stream_, count);
        break;
      case 4:
        widen(meta, in, reinterpret_cast<int32_t*>(out), stream_, count);
        break;
      default:
        widen(meta, in, reinterpret_cast<int64_t*>(out), stream_, count);
    }
  }
  return out - first * static_cast<int64_t>(width);
}

QueryContext::StringRows
QueryContext::strings(const std::string& table, const std::string& name, int64_t first, int64_t last) {
  if (batchRows() == 0) {
    return {offsets(table, name), chars(table, name)};
  }
  const int64_t count = std::max<int64_t>(last - first, 0);
  auto* offsetRows = static_cast<int64_t*>(batchBytes((count + 1) * sizeof(int64_t)));
  const Column& meta = store_.metadata(table, name);
  if (meta.data != nullptr) {
    const int64_t begin = meta.offsets[first];
    const int64_t end = meta.offsets[first + count];
    upload(offsetRows, meta.offsets + first, (count + 1) * sizeof(int64_t), true);
    char* chars = static_cast<char*>(batchBytes(end - begin));
    upload(chars, static_cast<const char*>(meta.data) + begin, end - begin, true);
    return {offsetRows - first, chars - begin};
  }
  auto* hostOffsets = static_cast<int64_t*>(staging((count + 1) * sizeof(int64_t)));
  store_.read(table, name, first, count, hostOffsets);
  const int64_t begin = hostOffsets[0];
  const int64_t end = hostOffsets[count];
  upload(offsetRows, hostOffsets, (count + 1) * sizeof(int64_t), true);
  char* chars = static_cast<char*>(batchBytes(end - begin));
  void* stage = staging(end - begin);
  store_.readChars(table, name, begin, end, stage);
  upload(chars, stage, end - begin, true);
  return {offsetRows - first, chars - begin};
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
  const Column& column = batchRows() > 0 ? store_.metadata(table, name) : store_.column(table, name);
  const int64_t count = (column.rows + kZoneRows - 1) / kZoneRows;
  return static_cast<const int2*>(cache_.derived(
      table + "." + name + "#zones",
      count * sizeof(int2),
      [&](void* out) {
        // Batch by batch, the column streams through a chunk buffer as a file
        // writer computing its statistics would.
        const int64_t chunkRows = batchRows() > 0 ? int64_t{1} << 26 : column.rows;
        const size_t width = dtypeSize(column.dtype);
        void* chunk = batchRows() > 0 ? scratchBytes(chunkRows * width) : nullptr;
        for (int64_t first = 0; first < column.rows; first += chunkRows) {
          const int64_t rows = std::min(chunkRows, column.rows - first);
          const void* in = column.data;
          if (chunk != nullptr) {
            void* stage = staging(rows * width);
            store_.read(table, name, first, rows, stage);
            upload(chunk, stage, rows * width, true);
            in = chunk;
          }
          auto* zones = static_cast<int2*>(out) + first / kZoneRows;
          const int64_t chunkZones = (rows + kZoneRows - 1) / kZoneRows;
          switch (column.dtype) {
            case DType::kInt8:
              zoneKernel<<<chunkZones, 256, 0, stream_>>>(static_cast<const int8_t*>(in), rows, zones);
              break;
            case DType::kInt16:
              zoneKernel<<<chunkZones, 256, 0, stream_>>>(static_cast<const int16_t*>(in), rows, zones);
              break;
            case DType::kInt32:
              zoneKernel<<<chunkZones, 256, 0, stream_>>>(static_cast<const int32_t*>(in), rows, zones);
              break;
            default:
              throw std::runtime_error("no zones for int64 " + name);
          }
          SV_CUDA(cudaGetLastError());
          SV_CUDA(cudaStreamSynchronize(stream_));
        }
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
