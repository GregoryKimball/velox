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
#include <tuple>

#include <cub/cub.cuh>

#include "velox/experimental/supervelox/queries/Derived.h"
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

constexpr int kSizes = 8;
constexpr int kMaxSize = 64;
constexpr int kHashSlots = 4096;

struct Q16Args {
  const int8_t* brand;
  const int16_t* type;
  const int8_t* size;
  const int32_t* suppkey;
  const uint32_t* complaints;
  int64_t firstPart;
  int64_t parts;
  int32_t excludedBrand;
  CodeSet excludedTypes;
  int32_t typeCount;
  int8_t sizeIndex[kMaxSize];
  unsigned* groupCount;
  const unsigned* groupOffset;
  unsigned* groupCursor;
  int32_t* members;
};

__device__ __forceinline__ int q16Group(const Q16Args& a, int64_t part) {
  const int brand = a.brand[part];
  const int type = a.type[part];
  const int size = a.size[part];
  if (brand == a.excludedBrand || a.excludedTypes.contains(type) || size < 0 || size >= kMaxSize) {
    return -1;
  }
  const int index = a.sizeIndex[size];
  if (index < 0) {
    return -1;
  }
  return (brand * a.typeCount + type) * kSizes + index;
}

template <bool kScatter>
__global__ void q16PartKernel(Q16Args a) {
  for (int64_t part = a.firstPart + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; part < a.parts; part += (int64_t)gridDim.x * blockDim.x) {
    const int group = q16Group(a, part);
    if (group < 0) {
      continue;
    }
    int supplier[4];
    int valid = 0;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      const int s = a.suppkey[part * 4 + slot];
      if (!testBit(a.complaints, s - 1)) {
        supplier[valid++] = s;
      }
    }
    if (valid == 0) {
      continue;
    }
    if constexpr (kScatter) {
      const unsigned position = a.groupOffset[group] + atomicAdd(&a.groupCursor[group], (unsigned)valid);
      for (int i = 0; i < valid; ++i) {
        a.members[position + i] = supplier[i];
      }
    } else {
      atomicAdd(&a.groupCount[group], (unsigned)valid);
    }
  }
}

/// One block per group: counts distinct suppliers with a shared hash set.
__global__ void q16DistinctKernel(const unsigned* offsets, const unsigned* counts, const int32_t* members, unsigned* distinct, unsigned* overflow) {
  __shared__ int32_t slots[kHashSlots];
  __shared__ unsigned found;
  const int group = blockIdx.x;
  const unsigned count = counts[group];
  if (count == 0) {
    return;
  }
  if (count > kHashSlots / 2) {
    if (threadIdx.x == 0) {
      atomicAdd(overflow, 1u);
    }
    return;
  }
  for (int i = threadIdx.x; i < kHashSlots; i += blockDim.x) {
    slots[i] = 0;
  }
  if (threadIdx.x == 0) {
    found = 0;
  }
  __syncthreads();
  const int32_t* groupMembers = members + offsets[group];
  unsigned local = 0;
  for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
    const int32_t key = groupMembers[i];
    unsigned slot = ((unsigned)key * 2654435761u) & (kHashSlots - 1);
    while (true) {
      const int32_t previous = atomicCAS(&slots[slot], 0, key);
      if (previous == 0) {
        ++local;
        break;
      }
      if (previous == key) {
        break;
      }
      slot = (slot + 1) & (kHashSlots - 1);
    }
  }
  atomicAdd(&found, local);
  __syncthreads();
  if (threadIdx.x == 0) {
    distinct[group] = found;
  }
}

/// Sort key of a result row: supplier count descending, then brand, type and
/// size ascending. Dictionary codes sort like their strings.
__global__ void q16KeyKernel(const unsigned* distinct, int64_t groups, int typeCount, unsigned long long* keys, unsigned long long* count) {
  constexpr int kSizeValues[kSizes] = {49, 14, 23, 45, 19, 3, 36, 9};
  const int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (g >= groups || distinct[g] == 0) {
    return;
  }
  const unsigned long long size = kSizeValues[g % kSizes];
  const unsigned long long type = (g / kSizes) % typeCount;
  const unsigned long long brand = g / kSizes / typeCount;
  keys[atomicAdd(count, 1ULL)] = ((0xffffffffULL - distinct[g]) << 32) | (brand << 16) | (type << 6) | size;
}

const std::vector<ResultField> kSchema = {
    {"p_brand", ResultType::kVarchar},
    {"p_type", ResultType::kVarchar},
    {"p_size", ResultType::kInteger},
    {"supplier_cnt", ResultType::kBigint},
};

Result q16(QueryContext& ctx) {
  Q16Args a;
  a.parts = ctx.rows("part");
  a.brand = ctx.col<int8_t>("part", "p_brand");
  a.type = ctx.col<int16_t>("part", "p_type");
  a.size = ctx.col<int8_t>("part", "p_size");
  a.suppkey = ctx.col<int32_t>("partsupp", "ps_suppkey");
  a.complaints = likeBitmap(ctx, "supplier", "s_comment", Match::kContains, "Customer", "Complaints");
  a.excludedBrand = ctx.meta("part", "p_brand").code("Brand#45");
  a.excludedTypes = dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "MEDIUM POLISHED");
  const auto& brands = ctx.meta("part", "p_brand").dictionary;
  const auto& types = ctx.meta("part", "p_type").dictionary;
  a.typeCount = types.size();
  constexpr int sizes[kSizes] = {49, 14, 23, 45, 19, 3, 36, 9};
  std::fill(a.sizeIndex, a.sizeIndex + kMaxSize, -1);
  for (int i = 0; i < kSizes; ++i) {
    a.sizeIndex[sizes[i]] = i;
  }
  const int64_t groups = (int64_t)brands.size() * types.size() * kSizes;
  a.groupCount = ctx.zeroed<unsigned>(groups);
  a.groupCursor = ctx.zeroed<unsigned>(groups);
  ctx.mark("q16 setup");
  forBatches(
      ctx,
      0,
      ctx.rows("part"),
      [&](int64_t first, int64_t last) {
        a.firstPart = first;
        a.parts = last;
        q16PartKernel<false><<<2048, 256, 0, ctx.stream()>>>(a);
      },
      4);
  SV_CUDA(cudaGetLastError());
  ctx.mark("q16 count");
  auto* offsets = ctx.scratch<unsigned>(groups);
  size_t tempBytes = 0;
  SV_CUDA(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, a.groupCount, offsets, groups, ctx.stream()));
  void* temp = ctx.scratchBytes(tempBytes);
  SV_CUDA(cub::DeviceScan::ExclusiveSum(temp, tempBytes, a.groupCount, offsets, groups, ctx.stream()));
  a.groupOffset = offsets;
  const unsigned members = ctx.download(offsets + groups - 1, 1)[0] + ctx.download(a.groupCount + groups - 1, 1)[0];
  a.members = ctx.scratch<int32_t>(members);
  ctx.mark("q16 scan");
  forBatches(
      ctx,
      0,
      ctx.rows("part"),
      [&](int64_t first, int64_t last) {
        a.firstPart = first;
        a.parts = last;
        q16PartKernel<true><<<2048, 256, 0, ctx.stream()>>>(a);
      },
      4);
  SV_CUDA(cudaGetLastError());
  ctx.mark("q16 scatter");
  auto* distinct = ctx.zeroed<unsigned>(groups);
  auto* overflow = ctx.zeroed<unsigned>(1);
  q16DistinctKernel<<<groups, 256, 0, ctx.stream()>>>(offsets, a.groupCount, a.members, distinct, overflow);
  SV_CUDA(cudaGetLastError());
  if (ctx.download(overflow, 1)[0]) {
    throw std::runtime_error("q16: group larger than the shared hash set");
  }
  auto* keys = ctx.scratch<unsigned long long>(groups);
  auto* keyCount = ctx.zeroed<unsigned long long>(1);
  q16KeyKernel<<<(groups + 255) / 256, 256, 0, ctx.stream()>>>(distinct, groups, a.typeCount, keys, keyCount);
  SV_CUDA(cudaGetLastError());
  const int64_t rowCount = ctx.download(keyCount, 1)[0];
  auto* sorted = ctx.scratch<unsigned long long>(rowCount);
  tempBytes = 0;
  SV_CUDA(cub::DeviceRadixSort::SortKeys(nullptr, tempBytes, keys, sorted, rowCount, 0, 64, ctx.stream()));
  temp = ctx.scratchBytes(tempBytes);
  SV_CUDA(cub::DeviceRadixSort::SortKeys(temp, tempBytes, keys, sorted, rowCount, 0, 64, ctx.stream()));
  auto hostKeys = ctx.download(sorted, rowCount);
  ctx.mark("q16 sort");
  Result result(kSchema);
  result.dictionaries[0] = &brands;
  result.dictionaries[1] = &types;
  result.codes[0].reserve(rowCount);
  result.codes[1].reserve(rowCount);
  result.ints[2].reserve(rowCount);
  result.ints[3].reserve(rowCount);
  for (const unsigned long long key : hostKeys) {
    result.codes[0].push_back((key >> 16) & 0xff);
    result.codes[1].push_back((key >> 6) & 0x3ff);
    result.ints[2].push_back(key & 0x3f);
    result.ints[3].push_back(0xffffffffULL - (key >> 32));
  }
  return result;
}

} // namespace

QuerySpec q16Spec() {
  return {16, q16, kSchema};
}

} // namespace facebook::velox::supervelox
