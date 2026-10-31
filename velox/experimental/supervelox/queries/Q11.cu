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
#include <cub/cub.cuh>

#include "velox/experimental/supervelox/queries/Derived.h"
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

struct PartValue {
  const int32_t* suppkey;
  const int32_t* supplycost;
  const int16_t* availqty;
  const int8_t* supplierNation;
  int32_t nation;

  __device__ __forceinline__ long long operator()(int64_t part) const {
    long long value = 0;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      const int64_t row = part * 4 + slot;
      if (supplierNation[suppkey[row] - 1] == nation) {
        value += (long long)supplycost[row] * availqty[row];
      }
    }
    return value;
  }
};

__global__ void q11TotalKernel(PartValue value, int64_t begin, int64_t end, unsigned long long* total) {
  unsigned long long sum = 0;
  for (int64_t part = begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; part < end; part += (int64_t)gridDim.x * blockDim.x) {
    sum += value(part);
  }
  sum = warpSum(sum);
  if (threadIdx.x % 32 == 0 && sum) {
    atomicAdd(total, sum);
  }
}

struct AboveThreshold {
  PartValue value;
  __int128 total;
  int64_t suppliers;

  /// value > total * 0.0001 / SF with SF = suppliers / 10000.
  __device__ bool operator()(int64_t part) const {
    return (__int128)value(part) * suppliers > total;
  }
};

__global__ void q11ValueKernel(PartValue value, const int64_t* parts, int64_t count, long long* values) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
    values[i] = value(parts[i]);
  }
}

const std::vector<ResultField> kSchema = {
    {"ps_partkey", ResultType::kBigint},
    {"value", ResultType::kDecimal, 38, 2},
};

Result q11(QueryContext& ctx) {
  PartValue value;
  value.suppkey = ctx.col<int32_t>("partsupp", "ps_suppkey");
  value.supplycost = ctx.col<int32_t>("partsupp", "ps_supplycost");
  value.availqty = ctx.col<int16_t>("partsupp", "ps_availqty");
  value.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  value.nation = nationKey(ctx, "GERMANY");
  const int64_t parts = ctx.rows("partsupp") / 4;
  auto* total = ctx.zeroed<unsigned long long>(1);
  forBatches(
      ctx,
      0,
      parts,
      [&](int64_t first, int64_t last) { q11TotalKernel<<<2048, 256, 0, ctx.stream()>>>(value, first, last, total); },
      4);
  SV_CUDA(cudaGetLastError());
  AboveThreshold above{value, (__int128)ctx.download(total, 1)[0], ctx.rows("supplier")};

  auto* selected = ctx.scratch<int64_t>(parts);
  auto* selectedCount = ctx.scratch<int64_t>(1);
  cub::CountingInputIterator<int64_t> rows(0);
  size_t tempBytes = 0;
  SV_CUDA(cub::DeviceSelect::If(nullptr, tempBytes, rows, selected, selectedCount, parts, above, ctx.stream()));
  void* temp = ctx.scratchBytes(tempBytes);
  SV_CUDA(cub::DeviceSelect::If(temp, tempBytes, rows, selected, selectedCount, parts, above, ctx.stream()));
  const int64_t count = ctx.download(selectedCount, 1)[0];

  auto* values = ctx.scratch<long long>(count);
  q11ValueKernel<<<256, 256, 0, ctx.stream()>>>(value, selected, count, values);
  SV_CUDA(cudaGetLastError());
  auto* sortedValues = ctx.scratch<long long>(count);
  auto* sortedParts = ctx.scratch<int64_t>(count);
  tempBytes = 0;
  SV_CUDA(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, tempBytes, values, sortedValues, selected, sortedParts, count, 0, 64, ctx.stream()));
  temp = ctx.scratchBytes(tempBytes);
  SV_CUDA(cub::DeviceRadixSort::SortPairsDescending(
      temp, tempBytes, values, sortedValues, selected, sortedParts, count, 0, 64, ctx.stream()));
  auto hostValues = ctx.download(sortedValues, count);
  auto hostParts = ctx.download(sortedParts, count);
  const Column& partkey = ctx.meta("partsupp", "ps_partkey");
  Result result(kSchema);
  result.ints[0].reserve(count);
  result.decimals[1].reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    result.ints[0].push_back(partkey.value(hostParts[i] * 4));
    result.decimals[1].push_back(hostValues[i]);
  }
  return result;
}

} // namespace

QuerySpec q11Spec() {
  return {11, q11, kSchema};
}

} // namespace facebook::velox::supervelox
