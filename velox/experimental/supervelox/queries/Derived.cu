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

#include <cub/cub.cuh>

#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

struct OrderStart {
  const int32_t* keys;

  __device__ bool operator()(uint32_t row) const {
    return row == 0 || keys[row] != keys[row - 1];
  }
};

template <typename T>
int64_t lowerBound(const T* values, int64_t rows, int64_t value) {
  int64_t low = 0;
  int64_t high = rows;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    if (values[mid] < value) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

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
    const int16_t* orderDates,
    const int16_t* lineDates,
    int64_t orders,
    int* lag) {
  int low = INT_MAX;
  int high = INT_MIN;
  for (int64_t o = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < orders; o += (int64_t)gridDim.x * blockDim.x) {
    const int date = orderDates[o];
    for (uint32_t line = starts[o]; line < starts[o + 1]; ++line) {
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
        lagKernel<<<2048, 256, 0, ctx.stream()>>>(
            orderLineStarts(ctx),
            ctx.col<int16_t>("orders", "o_orderdate"),
            ctx.col<int16_t>("lineitem", column),
            ctx.rows("orders"),
            static_cast<int*>(buffer));
        SV_CUDA(cudaGetLastError());
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
        SV_CUDA(cub::DeviceSelect::If(
            nullptr, tempBytes, rows, starts, selected, lines, OrderStart{keys}, ctx.stream()));
        void* temp = ctx.scratchBytes(tempBytes);
        SV_CUDA(cub::DeviceSelect::If(
            temp, tempBytes, rows, starts, selected, lines, OrderStart{keys}, ctx.stream()));
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

std::pair<int64_t, int64_t>
sortedRange(QueryContext& ctx, const std::string& table, const std::string& name, int64_t low, int64_t high) {
  const Column& column = ctx.meta(table, name);
  auto bounds = [&](auto* values) {
    return std::make_pair(lowerBound(values, column.rows, low), lowerBound(values, column.rows, high));
  };
  switch (column.dtype) {
    case DType::kInt8:
      return bounds(static_cast<const int8_t*>(column.data));
    case DType::kInt16:
      return bounds(static_cast<const int16_t*>(column.data));
    case DType::kInt32:
      return bounds(static_cast<const int32_t*>(column.data));
    case DType::kInt64:
      return bounds(static_cast<const int64_t*>(column.data));
  }
  return {0, 0};
}

const uint32_t* likeBitmap(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    Match match,
    const std::string& pattern,
    const std::string& second) {
  Pattern p{};
  if (pattern.size() > kMaxPattern || second.size() > kMaxPattern) {
    throw std::runtime_error("pattern too long");
  }
  memcpy(p.first, pattern.data(), pattern.size());
  p.firstLength = pattern.size();
  memcpy(p.second, second.data(), second.size());
  p.secondLength = second.size();
  p.match = match;
  const int64_t rows = ctx.rows(table);
  auto* bits = ctx.scratch<uint32_t>((rows + 31) / 32);
  likeKernel<<<4096, 256, 0, ctx.stream()>>>(
      ctx.offsets(table, column), ctx.chars(table, column), rows, p, bits);
  SV_CUDA(cudaGetLastError());
  return bits;
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
  const auto& dictionary = ctx.meta(table, column).dictionary;
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
  const Column& meta = ctx.meta(table, column);
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
