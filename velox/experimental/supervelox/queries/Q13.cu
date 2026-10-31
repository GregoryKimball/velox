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

#include "velox/experimental/supervelox/queries/Derived.h"
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

constexpr int kMaxCount = 256;
constexpr int kBlockBytes = 4096;
constexpr int kChunk = 16;

/// For each kBlockBytes block of the characters, the row whose string holds
/// the block's first byte.
__global__ void blockRowsKernel(const int64_t* offsets, int64_t rows, uint32_t* blockRows) {
  for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < rows; r += (int64_t)gridDim.x * blockDim.x) {
    const int64_t end = offsets[r + 1];
    for (int64_t b = (offsets[r] + kBlockBytes - 1) / kBlockBytes; b * kBlockBytes < end; ++b) {
      blockRows[b] = r;
    }
  }
}

__device__ __forceinline__ bool
matchAt(const char* chars, int64_t position, const char* pattern, int length) {
  for (int i = 0; i < length; ++i) {
    if (chars[position + i] != pattern[i]) {
      return false;
    }
  }
  return true;
}

/// Sets bit o of 'excluded' when o_comment of order row o matches
/// '%special%requests%'. Each thread scans kChunk bytes for "special".
__global__ void commentKernel(
    const char* chars,
    int64_t bytes,
    const int64_t* offsets,
    int64_t rows,
    const uint32_t* blockRows,
    int64_t firstChunk,
    int64_t lastChunk,
    uint32_t* excluded) {
  constexpr char kFirst[] = "special";
  constexpr char kSecond[] = "requests";
  constexpr int kFirstLength = 7;
  constexpr int kSecondLength = 8;
  for (int64_t chunk = firstChunk + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; chunk < lastChunk;
       chunk += (int64_t)gridDim.x * blockDim.x) {
    const int64_t begin = chunk * kChunk;
    uint4 data;
    if (begin + kChunk <= bytes) {
      data = __ldg(reinterpret_cast<const uint4*>(chars + begin));
    } else {
      data = make_uint4(0, 0, 0, 0);
      for (int64_t i = begin; i < bytes; ++i) {
        reinterpret_cast<char*>(&data)[i - begin] = chars[i];
      }
    }
    const uint32_t words[4] = {data.x, data.y, data.z, data.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      // Bytes equal to 's' in this word.
      uint32_t hits = __vcmpeq4(words[w], 0x73737373u);
      while (hits) {
        const int byte = (__ffs(hits) - 1) / 8;
        hits &= ~(0xffu << (byte * 8));
        const int64_t position = begin + w * 4 + byte;
        if (position + kFirstLength + kSecondLength > bytes || !matchAt(chars, position + 1, kFirst + 1, kFirstLength - 1)) {
          continue;
        }
        int64_t low = blockRows[position / kBlockBytes];
        int64_t high = position / kBlockBytes + 1 < (bytes + kBlockBytes - 1) / kBlockBytes
            ? blockRows[position / kBlockBytes + 1] + 1
            : rows;
        // Last row with offsets[row] <= position.
        while (high - low > 1) {
          const int64_t mid = (low + high) / 2;
          if (offsets[mid] <= position) {
            low = mid;
          } else {
            high = mid;
          }
        }
        const int64_t end = offsets[low + 1];
        for (int64_t q = position + kFirstLength; q + kSecondLength <= end; ++q) {
          if (chars[q] == 'r' && matchAt(chars, q, kSecond, kSecondLength)) {
            atomicOr(&excluded[low >> 5], 1u << (low & 31));
            break;
          }
        }
      }
    }
  }
}

/// Customers' order counts are bytes packed four to a word.
__global__ void q13OrdersKernel(const int32_t* custkey, const uint32_t* excluded, int64_t begin, int64_t end, uint32_t* counts) {
  for (int64_t o = begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < end; o += (int64_t)gridDim.x * blockDim.x) {
    if (!testBit(excluded, o)) {
      const int c = custkey[o] - 1;
      atomicAdd(&counts[c >> 2], 1u << ((c & 3) * 8));
    }
  }
}

__global__ void q13HistogramKernel(const uint32_t* counts, int64_t customers, unsigned long long* histogram) {
  __shared__ unsigned local[kMaxCount];
  local[threadIdx.x] = 0;
  __syncthreads();
  for (int64_t c = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; c < customers; c += (int64_t)gridDim.x * blockDim.x) {
    atomicAdd(&local[(counts[c >> 2] >> ((c & 3) * 8)) & 0xff], 1u);
  }
  __syncthreads();
  if (local[threadIdx.x]) {
    atomicAdd(&histogram[threadIdx.x], (unsigned long long)local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"c_count", ResultType::kBigint},
    {"custdist", ResultType::kBigint},
};

Result q13(QueryContext& ctx) {
  const int64_t orders = ctx.rows("orders");
  const int64_t customers = ctx.rows("customer");
  const Column& comment = ctx.meta("orders", "o_comment");
  const int64_t* offsets = ctx.offsets("orders", "o_comment");
  const int64_t blocks = (comment.bytes + kBlockBytes - 1) / kBlockBytes;
  const auto* blockRows = static_cast<const uint32_t*>(ctx.cache().derived(
      "orders.o_comment$blockrows",
      blocks * sizeof(uint32_t),
      [&](void* buffer) {
        blockRowsKernel<<<4096, 256, 0, ctx.stream()>>>(offsets, orders, static_cast<uint32_t*>(buffer));
        SV_CUDA(cudaGetLastError());
      },
      ctx.stream()));
  auto* excluded = ctx.zeroed<uint32_t>((orders + 31) / 32);
  const char* chars = ctx.chars("orders", "o_comment");
  forBatches(ctx, 0, (comment.bytes + kChunk - 1) / kChunk, [&](int64_t first, int64_t last) {
    commentKernel<<<persistentBlocks(commentKernel, 256), 256, 0, ctx.stream()>>>(
        chars, comment.bytes, offsets, orders, blockRows, first, last, excluded);
  });
  SV_CUDA(cudaGetLastError());
  auto* counts = ctx.zeroed<uint32_t>((customers + 3) / 4);
  const int32_t* custkey = ctx.col<int32_t>("orders", "o_custkey");
  forBatches(
      ctx,
      0,
      orders,
      [&](int64_t first, int64_t last) {
        q13OrdersKernel<<<4096, 256, 0, ctx.stream()>>>(custkey, excluded, first, last, counts);
      },
      4);
  SV_CUDA(cudaGetLastError());
  auto* histogram = ctx.zeroed<unsigned long long>(kMaxCount);
  q13HistogramKernel<<<1024, kMaxCount, 0, ctx.stream()>>>(counts, customers, histogram);
  SV_CUDA(cudaGetLastError());
  auto host = ctx.download(histogram, kMaxCount);
  if (host[kMaxCount - 1]) {
    throw std::runtime_error("q13: a customer may have 255 or more orders");
  }
  std::vector<std::pair<int64_t, int64_t>> rows;
  for (int count = 0; count < kMaxCount; ++count) {
    if (host[count]) {
      rows.emplace_back(host[count], count);
    }
  }
  std::sort(rows.rbegin(), rows.rend());
  Result result(kSchema);
  for (const auto& [customersWithCount, count] : rows) {
    result.ints[0].push_back(count);
    result.ints[1].push_back(customersWithCount);
  }
  return result;
}

} // namespace

QuerySpec q13Spec() {
  return {13, q13, kSchema};
}

} // namespace facebook::velox::supervelox
