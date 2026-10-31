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
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerThread = 8;

__global__ void partKernel(const int8_t* brand, const int8_t* container, int64_t parts, int brandCode, int containerCode, uint32_t* bits) {
  const int64_t padded = (parts + 31) / 32 * 32;
  for (int64_t p = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; p < padded; p += (int64_t)gridDim.x * blockDim.x) {
    const bool hit = p < parts && brand[p] == brandCode && container[p] == containerCode;
    const unsigned word = __ballot_sync(0xffffffff, hit);
    if (threadIdx.x % 32 == 0) {
      bits[p >> 5] = word;
    }
  }
}

struct Q17Args {
  const int32_t* partkey;
  const int16_t* quantity;
  BitmapRanks parts;
  int64_t firstRow;
  int64_t rows;
  unsigned* quantitySum;
  unsigned* lineCount;
  uint32_t* lines;
  unsigned long long* lineTotal;
  unsigned long long capacity;
};

__device__ __forceinline__ void q17Line(const Q17Args& a, int64_t row, int part) {
  const int64_t slot = bitRank(a.parts, part - 1);
  atomicAdd(&a.quantitySum[slot], (unsigned)a.quantity[row]);
  atomicAdd(&a.lineCount[slot], 1u);
  const unsigned long long index = atomicAdd(a.lineTotal, 1ULL);
  if (index < a.capacity) {
    a.lines[index] = row;
  }
}

__global__ void __launch_bounds__(kThreads) q17Kernel(Q17Args a) {
  const int64_t stride = (int64_t)gridDim.x * kThreads * kRowsPerThread;
  for (int64_t row = a.firstRow + (blockIdx.x * (int64_t)kThreads + threadIdx.x) * kRowsPerThread; row < a.rows; row += stride) {
    if (row + kRowsPerThread <= a.rows) {
      int32_t part[kRowsPerThread];
      loadVector(a.partkey + row, part);
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        if (testBit(a.parts.bits, part[i] - 1)) {
          q17Line(a, row + i, part[i]);
        }
      }
    } else {
      for (int64_t r = row; r < a.rows; ++r) {
        if (testBit(a.parts.bits, a.partkey[r] - 1)) {
          q17Line(a, r, a.partkey[r]);
        }
      }
    }
  }
}

__global__ void q17SumKernel(Q17Args a, const int32_t* price, int64_t lineCount, unsigned long long* total) {
  unsigned long long sum = 0;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < lineCount; i += (int64_t)gridDim.x * blockDim.x) {
    const uint32_t row = a.lines[i];
    const int64_t slot = bitRank(a.parts, a.partkey[row] - 1);
    // quantity < 0.2 * avg(quantity), kept in integers.
    if (5ULL * a.quantity[row] * a.lineCount[slot] < a.quantitySum[slot]) {
      sum += price[row];
    }
  }
  sum = warpSum(sum);
  if (threadIdx.x % 32 == 0 && sum) {
    atomicAdd(total, sum);
  }
}

const std::vector<ResultField> kSchema = {{"avg_yearly", ResultType::kDouble}};

Result q17(QueryContext& ctx) {
  const int64_t parts = ctx.rows("part");
  auto* bits = ctx.scratch<uint32_t>((parts + 31) / 32);
  partKernel<<<2048, 256, 0, ctx.stream()>>>(
      ctx.col<int8_t>("part", "p_brand"),
      ctx.col<int8_t>("part", "p_container"),
      parts,
      ctx.meta("part", "p_brand").code("Brand#23"),
      ctx.meta("part", "p_container").code("MED BOX"),
      bits);
  SV_CUDA(cudaGetLastError());
  Q17Args a;
  a.parts = rankBitmap(ctx, bits, parts);
  a.partkey = ctx.col<int32_t>("lineitem", "l_partkey");
  a.quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  a.rows = ctx.rows("lineitem");
  a.quantitySum = ctx.zeroed<unsigned>(a.parts.count);
  a.lineCount = ctx.zeroed<unsigned>(a.parts.count);
  a.lineTotal = ctx.zeroed<unsigned long long>(1);
  // Lines per part are about rows / parts; leave room for skew.
  a.capacity = a.parts.count * (a.rows / parts + 1) * 2 + 1024;
  a.lines = ctx.scratch<uint32_t>(a.capacity);
  forBatches(
      ctx,
      0,
      ctx.rows("lineitem"),
      [&](int64_t first, int64_t last) {
        a.firstRow = first;
        a.rows = last;
        q17Kernel<<<persistentBlocks(q17Kernel, kThreads), kThreads, 0, ctx.stream()>>>(a);
      },
      1,
      QueryContext::kZoneRows);
  SV_CUDA(cudaGetLastError());
  const unsigned long long lineCount = ctx.download(a.lineTotal, 1)[0];
  if (lineCount > a.capacity) {
    throw std::runtime_error("q17: line buffer overflow");
  }
  auto* total = ctx.zeroed<unsigned long long>(1);
  q17SumKernel<<<1024, 256, 0, ctx.stream()>>>(a, ctx.col<int32_t>("lineitem", "l_extendedprice"), lineCount, total);
  SV_CUDA(cudaGetLastError());
  Result result(kSchema);
  result.doubles[0].push_back(ctx.download(total, 1)[0] / 100.0 / 7.0);
  return result;
}

} // namespace

QuerySpec q17Spec() {
  return {17, q17, kSchema};
}

} // namespace facebook::velox::supervelox
