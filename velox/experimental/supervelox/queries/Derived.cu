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
#include "velox/experimental/supervelox/queries/Derived.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <map>
#include <mutex>

#include <cub/cub.cuh>

#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

template <typename Key>
struct OrderStart {
  const Key* keys;
  int64_t base;

  __device__ bool operator()(uint32_t row) const {
    return row == 0 || keys[base + row] != keys[base + row - 1];
  }
};

/// l_orderkey of lines [first, first + count), widened to int64.
void readKeys(QueryContext& ctx, int64_t first, int64_t count, int64_t* out) {
  const Column& meta = ctx.store().metadata("lineitem", "l_orderkey");
  std::vector<char> raw(count * dtypeSize(meta.dtype));
  ctx.store().read("lineitem", "l_orderkey", first, count, raw.data());
  for (int64_t i = 0; i < count; ++i) {
    switch (meta.dtype) {
      case DType::kInt8:
        out[i] = reinterpret_cast<const int8_t*>(raw.data())[i];
        break;
      case DType::kInt16:
        out[i] = reinterpret_cast<const int16_t*>(raw.data())[i];
        break;
      case DType::kInt32:
        out[i] = reinterpret_cast<const int32_t*>(raw.data())[i];
        break;
      case DType::kInt64:
        out[i] = reinterpret_cast<const int64_t*>(raw.data())[i];
        break;
    }
  }
}

/// First line of every kStride-th order row, from one pass over l_orderkey.
constexpr int64_t kStride = 4096;

const std::vector<int64_t>& lineIndex(QueryContext& ctx) {
  static std::mutex mutex;
  static std::map<const ColumnStore*, std::vector<int64_t>> indexes;
  std::lock_guard<std::mutex> lock(mutex);
  auto& index = indexes[&ctx.store()];
  if (!index.empty()) {
    return index;
  }
  const int64_t lines = ctx.rows("lineitem");
  constexpr int64_t kChunk = int64_t{1} << 24;
  std::vector<int64_t> keys(kChunk);
  int64_t order = 0;
  int64_t previous = 0;
  for (int64_t first = 0; first < lines; first += kChunk) {
    const int64_t count = std::min(kChunk, lines - first);
    readKeys(ctx, first, count, keys.data());
    for (int64_t i = 0; i < count; ++i) {
      if (first + i == 0 || keys[i] != previous) {
        if (order % kStride == 0) {
          index.push_back(first + i);
        }
        ++order;
      }
      previous = keys[i];
    }
  }
  index.push_back(lines);
  return index;
}

} // namespace

int64_t lineStart(QueryContext& ctx, int64_t order) {
  const auto& index = lineIndex(ctx);
  const int64_t orders = ctx.rows("orders");
  if (order >= orders) {
    return ctx.rows("lineitem");
  }
  int64_t line = index[order / kStride];
  int64_t skip = order % kStride;
  const int64_t lines = ctx.rows("lineitem");
  constexpr int64_t kChunk = 1 << 16;
  std::vector<int64_t> keys(kChunk + 1);
  while (skip > 0) {
    const int64_t count = std::min(kChunk + 1, lines - line);
    readKeys(ctx, line, count, keys.data());
    for (int64_t i = 1; i < count; ++i) {
      if (keys[i] != keys[i - 1] && --skip == 0) {
        return line + i;
      }
    }
    line += count - 1;
  }
  return line;
}

namespace {

constexpr int kMaxPattern = 32;

struct Pattern {
  char first[kMaxPattern];
  int firstLength;
  char second[kMaxPattern];
  int secondLength;
  Match match;
};

/// Returns the end of the first occurrence of 'needle' in [begin, end), or -1.
__device__ __forceinline__ int64_t
find(const char* chars, int64_t begin, int64_t end, const char* needle, int length) {
  for (int64_t i = begin; i + length <= end; ++i) {
    int j = 0;
    while (j < length && chars[i + j] == needle[j]) {
      ++j;
    }
    if (j == length) {
      return i + length;
    }
  }
  return -1;
}

__device__ __forceinline__ bool
matches(const char* chars, int64_t begin, int64_t end, const Pattern& p) {
  switch (p.match) {
    case Match::kPrefix:
      return end - begin >= p.firstLength && find(chars, begin, begin + p.firstLength, p.first, p.firstLength) >= 0;
    case Match::kSuffix:
      return end - begin >= p.firstLength && find(chars, end - p.firstLength, end, p.first, p.firstLength) >= 0;
    case Match::kContains: {
      const int64_t after = find(chars, begin, end, p.first, p.firstLength);
      if (after < 0) {
        return false;
      }
      return p.secondLength == 0 || find(chars, after, end, p.second, p.secondLength) >= 0;
    }
  }
  return false;
}

__global__ void likeKernel(const int64_t* offsets, const char* chars, int64_t rows, Pattern pattern, uint32_t* bits) {
  const int64_t padded = (rows + 31) / 32 * 32;
  for (int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; row < padded;
       row += (int64_t)gridDim.x * blockDim.x) {
    bool hit = row < rows && matches(chars, offsets[row], offsets[row + 1], pattern);
    const unsigned word = __ballot_sync(0xffffffff, hit);
    if (threadIdx.x % 32 == 0) {
      bits[row >> 5] = word;
    }
  }
}

bool hostMatches(const std::string& value, Match match, const std::string& pattern) {
  switch (match) {
    case Match::kPrefix:
      return value.compare(0, pattern.size(), pattern) == 0;
    case Match::kSuffix:
      return value.size() >= pattern.size() &&
          value.compare(value.size() - pattern.size(), pattern.size(), pattern) == 0;
    case Match::kContains:
      return value.find(pattern) != std::string::npos;
  }
  return false;
}

constexpr int kBuckets = 4096;

__device__ __forceinline__ int bucketOf(int64_t value, int64_t floor, double scale) {
  return min(kBuckets - 1, (int)((double)(value - floor) * scale));
}

__global__ void histogramKernel(const int64_t* values, int64_t count, int64_t floor, double scale, unsigned* histogram) {
  __shared__ unsigned local[kBuckets];
  for (int i = threadIdx.x; i < kBuckets; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t v = values[i];
    if (v > floor) {
      atomicAdd(&local[bucketOf(v, floor, scale)], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kBuckets; i += blockDim.x) {
    if (local[i]) {
      atomicAdd(&histogram[i], local[i]);
    }
  }
}

__global__ void compactKernel(
    const int64_t* values,
    int64_t count,
    int64_t floor,
    double scale,
    int threshold,
    Candidate* out,
    unsigned long long* outCount,
    unsigned long long capacity) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t v = values[i];
    if (v > floor && bucketOf(v, floor, scale) >= threshold) {
      const unsigned long long slot = atomicAdd(outCount, 1ULL);
      if (slot < capacity) {
        out[slot] = Candidate{i, v};
      }
    }
  }
}

__global__ void lagKernel(
    const uint32_t* starts,
    int64_t base,
    const int16_t* orderDates,
    const int16_t* lineDates,
    int64_t firstOrder,
    int64_t orders,
    int* lag) {
  int low = INT_MAX;
  int high = INT_MIN;
  for (int64_t o = firstOrder + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < orders;
       o += (int64_t)gridDim.x * blockDim.x) {
    const int date = orderDates[o];
    for (int64_t line = base + starts[o]; line < base + starts[o + 1]; ++line) {
      const int d = lineDates[line] - date;
      low = min(low, d);
      high = max(high, d);
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    low = min(low, __shfl_down_sync(0xffffffff, low, offset));
    high = max(high, __shfl_down_sync(0xffffffff, high, offset));
  }
  if (threadIdx.x % 32 == 0) {
    atomicMin(lag, low);
    atomicMax(lag + 1, high);
  }
}

} // namespace

uint32_t nationsInRegion(QueryContext& ctx, const std::string& region) {
  const Column& names = ctx.meta("region", "r_name");
  const Column& regionKeys = ctx.meta("region", "r_regionkey");
  int64_t regionKey = -1;
  for (int64_t row = 0; row < names.rows; ++row) {
    if (names.string(row) == region) {
      regionKey = regionKeys.value(row);
    }
  }
  const Column& nationKeys = ctx.meta("nation", "n_nationkey");
  const Column& nationRegions = ctx.meta("nation", "n_regionkey");
  uint32_t mask = 0;
  for (int64_t row = 0; row < nationKeys.rows; ++row) {
    if (nationRegions.value(row) == regionKey) {
      mask |= 1u << nationKeys.value(row);
    }
  }
  return mask;
}

int32_t nationKey(QueryContext& ctx, const std::string& nation) {
  const Column& names = ctx.meta("nation", "n_name");
  const Column& keys = ctx.meta("nation", "n_nationkey");
  for (int64_t row = 0; row < names.rows; ++row) {
    if (names.string(row) == nation) {
      return keys.value(row);
    }
  }
  throw std::runtime_error("no nation " + nation);
}

std::string nationName(QueryContext& ctx, int32_t key) {
  const Column& names = ctx.meta("nation", "n_name");
  const Column& keys = ctx.meta("nation", "n_nationkey");
  for (int64_t row = 0; row < names.rows; ++row) {
    if (keys.value(row) == key) {
      return names.string(row);
    }
  }
  throw std::runtime_error("no nation key " + std::to_string(key));
}

std::pair<int32_t, int32_t> lineDateLag(QueryContext& ctx, const std::string& column) {
  const int* lag = static_cast<const int*>(ctx.cache().derived(
      "lineitem." + column + "$lag",
      2 * sizeof(int),
      [&](void* buffer) {
        const int initial[2] = {INT_MAX, INT_MIN};
        SV_CUDA(cudaMemcpyAsync(buffer, initial, sizeof(initial), cudaMemcpyHostToDevice, ctx.stream()));
        forBatches(
            ctx,
            0,
            ctx.rows("orders"),
            [&](int64_t first, int64_t last) {
              const LineRange lines = lineRange(ctx, first, last);
              lagKernel<<<2048, 256, 0, ctx.stream()>>>(
                  lines.starts,
                  lines.base,
                  ctx.rows<int16_t>("orders", "o_orderdate", first, last),
                  ctx.rows<int16_t>("lineitem", column, lines.firstLine, lines.lastLine),
                  first,
                  last,
                  static_cast<int*>(buffer));
              SV_CUDA(cudaGetLastError());
            },
            4);
      },
      ctx.stream()));
  auto host = ctx.download(lag, 2);
  return {host[0], host[1]};
}

const uint32_t* orderLineStarts(QueryContext& ctx) {
  const int64_t orders = ctx.rows("orders");
  return static_cast<const uint32_t*>(ctx.cache().derived(
      "orders.$linestarts",
      (orders + 1) * sizeof(uint32_t),
      [&](void* buffer) {
        auto* starts = static_cast<uint32_t*>(buffer);
        const int64_t lines = ctx.rows("lineitem");
        const int32_t* keys = static_cast<const int32_t*>(ctx.meta("lineitem", "l_orderkey").data);
        auto* selected = ctx.scratch<int64_t>(1);
        size_t tempBytes = 0;
        cub::CountingInputIterator<uint32_t> rows(0);
        const OrderStart<int32_t> isStart{keys, 0};
        SV_CUDA(cub::DeviceSelect::If(nullptr, tempBytes, rows, starts, selected, lines, isStart, ctx.stream()));
        void* temp = ctx.scratchBytes(tempBytes);
        SV_CUDA(cub::DeviceSelect::If(temp, tempBytes, rows, starts, selected, lines, isStart, ctx.stream()));
        const int64_t found = ctx.download(selected, 1)[0];
        if (found != orders) {
          throw std::runtime_error(
              "lineitem has " + std::to_string(found) + " orders, orders has " + std::to_string(orders));
        }
        const uint32_t end = static_cast<uint32_t>(lines);
        SV_CUDA(cudaMemcpyAsync(starts + orders, &end, sizeof(end), cudaMemcpyHostToDevice, ctx.stream()));
        SV_CUDA(cudaStreamSynchronize(ctx.stream()));
      },
      ctx.stream()));
}

LineRange lineRange(QueryContext& ctx, int64_t first, int64_t last) {
  if (batchRows() == 0) {
    return {orderLineStarts(ctx), 0, 0, ctx.rows("lineitem")};
  }
  const int64_t firstLine = lineStart(ctx, first);
  const int64_t lastLine = lineStart(ctx, last);
  const int64_t orders = last - first;
  auto* starts = ctx.batchScratch<uint32_t>(orders + 1);
  if (orders > 0) {
    const int64_t* keys = ctx.rows<int64_t>("lineitem", "l_orderkey", firstLine, lastLine);
    auto* selected = ctx.batchScratch<int64_t>(1);
    size_t tempBytes = 0;
    cub::CountingInputIterator<uint32_t> rows(0);
    const OrderStart<int64_t> isStart{keys, firstLine};
    const int64_t lines = lastLine - firstLine;
    SV_CUDA(cub::DeviceSelect::If(nullptr, tempBytes, rows, starts, selected, lines, isStart, ctx.stream()));
    void* temp = ctx.batchBytes(tempBytes);
    SV_CUDA(cub::DeviceSelect::If(temp, tempBytes, rows, starts, selected, lines, isStart, ctx.stream()));
    const int64_t found = ctx.download(selected, 1)[0];
    if (found != orders) {
      throw std::runtime_error(
          "lines " + std::to_string(firstLine) + ".." + std::to_string(lastLine) + " hold " + std::to_string(found) +
          " orders, not " + std::to_string(orders));
    }
  }
  const uint32_t end = static_cast<uint32_t>(lastLine - firstLine);
  SV_CUDA(cudaMemcpyAsync(starts + orders, &end, sizeof(end), cudaMemcpyHostToDevice, ctx.stream()));
  SV_CUDA(cudaStreamSynchronize(ctx.stream()));
  return {starts - first, firstLine, firstLine, lastLine};
}

std::pair<int64_t, int64_t>
sortedRange(QueryContext& ctx, const std::string& table, const std::string& name, int64_t low, int64_t high) {
  const int64_t rows = ctx.rows(table);
  auto lowerBound = [&](int64_t value) {
    int64_t begin = 0;
    int64_t end = rows;
    while (begin < end) {
      const int64_t mid = (begin + end) / 2;
      if (ctx.value(table, name, mid) < value) {
        begin = mid + 1;
      } else {
        end = mid;
      }
    }
    return begin;
  };
  return {lowerBound(low), lowerBound(high)};
}

Pattern makePattern(Match match, const std::string& pattern, const std::string& second) {
  Pattern p{};
  if (pattern.size() > kMaxPattern || second.size() > kMaxPattern) {
    throw std::runtime_error("pattern too long");
  }
  memcpy(p.first, pattern.data(), pattern.size());
  p.firstLength = pattern.size();
  memcpy(p.second, second.data(), second.size());
  p.secondLength = second.size();
  p.match = match;
  return p;
}

const uint32_t* likeBitmap(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    Match match,
    const std::string& pattern,
    const std::string& second) {
  const int64_t rows = ctx.rows(table);
  auto* bits = ctx.scratch<uint32_t>((rows + 31) / 32);
  likeKernel<<<4096, 256, 0, ctx.stream()>>>(
      ctx.offsets(table, column), ctx.chars(table, column), rows, makePattern(match, pattern, second), bits);
  SV_CUDA(cudaGetLastError());
  return bits;
}

void likeRows(
    QueryContext& ctx,
    QueryContext::StringRows strings,
    int64_t first,
    int64_t last,
    Match match,
    const std::string& pattern,
    const std::string& second,
    uint32_t* bits) {
  likeKernel<<<4096, 256, 0, ctx.stream()>>>(
      strings.offsets + first, strings.chars, last - first, makePattern(match, pattern, second), bits);
  SV_CUDA(cudaGetLastError());
}

struct PopCount {
  const uint32_t* bits;

  __device__ uint32_t operator()(int64_t word) const {
    return __popc(bits[word]);
  }
};

BitmapRanks rankBitmap(QueryContext& ctx, const uint32_t* bits, int64_t rows) {
  const int64_t words = (rows + 31) / 32;
  auto* ranks = ctx.scratch<uint32_t>(words);
  cub::TransformInputIterator<uint32_t, PopCount, cub::CountingInputIterator<int64_t>> counts(
      cub::CountingInputIterator<int64_t>(0), PopCount{bits});
  size_t tempBytes = 0;
  SV_CUDA(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, counts, ranks, words, ctx.stream()));
  void* temp = ctx.scratchBytes(tempBytes);
  SV_CUDA(cub::DeviceScan::ExclusiveSum(temp, tempBytes, counts, ranks, words, ctx.stream()));
  const uint32_t last = ctx.download(ranks + words - 1, 1)[0];
  uint32_t lastWord = 0;
  SV_CUDA(cudaMemcpyAsync(&lastWord, bits + words - 1, sizeof(lastWord), cudaMemcpyDeviceToHost, ctx.stream()));
  SV_CUDA(cudaStreamSynchronize(ctx.stream()));
  return BitmapRanks{bits, ranks, static_cast<int64_t>(last) + __builtin_popcount(lastWord)};
}

CodeSet dictionaryCodes(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    Match match,
    const std::string& pattern) {
  const auto& dictionary = ctx.info(table, column).dictionary;
  CodeSet codes;
  for (size_t i = 0; i < dictionary.size(); ++i) {
    if (hostMatches(dictionary[i], match, pattern)) {
      codes.add(i);
    }
  }
  return codes;
}

CodeSet dictionaryCodes(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    const std::vector<std::string>& values) {
  const Column& meta = ctx.info(table, column);
  CodeSet codes;
  for (const auto& value : values) {
    const int32_t code = meta.code(value);
    if (code >= 0) {
      codes.add(code);
    }
  }
  return codes;
}

std::vector<Candidate>
topCandidates(QueryContext& ctx, const int64_t* values, int64_t count, int32_t k, int64_t floor) {
  auto* maximum = ctx.scratch<int64_t>(1);
  size_t tempBytes = 0;
  SV_CUDA(cub::DeviceReduce::Max(nullptr, tempBytes, values, maximum, count, ctx.stream()));
  void* temp = ctx.scratchBytes(tempBytes);
  SV_CUDA(cub::DeviceReduce::Max(temp, tempBytes, values, maximum, count, ctx.stream()));
  const int64_t top = ctx.download(maximum, 1)[0];
  if (top <= floor) {
    return {};
  }
  const double scale = kBuckets / ((double)(top - floor) + 1);
  auto* histogram = ctx.zeroed<unsigned>(kBuckets);
  histogramKernel<<<1024, 256, 0, ctx.stream()>>>(values, count, floor, scale, histogram);
  SV_CUDA(cudaGetLastError());
  auto hostHistogram = ctx.download(histogram, kBuckets);
  int threshold = kBuckets - 1;
  int64_t total = 0;
  for (; threshold > 0; --threshold) {
    total += hostHistogram[threshold];
    if (total >= k) {
      break;
    }
  }
  if (threshold == 0) {
    total += hostHistogram[0];
  }
  auto* out = ctx.scratch<Candidate>(total);
  auto* outCount = ctx.zeroed<unsigned long long>(1);
  compactKernel<<<1024, 256, 0, ctx.stream()>>>(values, count, floor, scale, threshold, out, outCount, total);
  SV_CUDA(cudaGetLastError());
  return ctx.download(out, total);
}

} // namespace facebook::velox::supervelox
