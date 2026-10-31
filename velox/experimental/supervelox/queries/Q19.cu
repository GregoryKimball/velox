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

/// One of the three OR'ed conjuncts.
struct Branch {
  int32_t brand;
  CodeSet containers;
  int32_t quantityLow;
  int32_t quantityHigh;
  int32_t sizeLow;
  int32_t sizeHigh;
};

struct Q19Args {
  const int8_t* shipinstruct;
  const int8_t* shipmode;
  const int32_t* partkey;
  const int16_t* quantity;
  const int32_t* price;
  const int8_t* discount;
  const int8_t* brand;
  const int8_t* container;
  const int8_t* size;
  int64_t firstRow;
  int64_t rows;
  int32_t instruct;
  CodeSet modes;
  Branch branches[3];
  unsigned long long* revenue;
};

__device__ __forceinline__ long long q19Row(const Q19Args& a, int64_t row) {
  const int part = a.partkey[row] - 1;
  const int quantity = a.quantity[row];
  const int brand = a.brand[part];
  const int container = a.container[part];
  const int size = a.size[part];
  bool pass = false;
#pragma unroll
  for (int b = 0; b < 3; ++b) {
    const Branch& branch = a.branches[b];
    pass |= brand == branch.brand && branch.containers.contains(container) && quantity >= branch.quantityLow &&
        quantity <= branch.quantityHigh && size >= branch.sizeLow && size <= branch.sizeHigh;
  }
  return pass ? (long long)a.price[row] * (100 - a.discount[row]) : 0;
}

__global__ void __launch_bounds__(kThreads) q19Kernel(Q19Args a) {
  long long sum = 0;
  const int64_t stride = (int64_t)gridDim.x * kThreads * kRowsPerThread;
  for (int64_t row = a.firstRow + (blockIdx.x * (int64_t)kThreads + threadIdx.x) * kRowsPerThread; row < a.rows; row += stride) {
    if (row + kRowsPerThread <= a.rows) {
      int8_t instruct[kRowsPerThread];
      int8_t mode[kRowsPerThread];
      loadVector(a.shipinstruct + row, instruct);
      loadVector(a.shipmode + row, mode);
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        if (instruct[i] == a.instruct && a.modes.contains(mode[i])) {
          sum += q19Row(a, row + i);
        }
      }
    } else {
      for (int64_t r = row; r < a.rows; ++r) {
        if (a.shipinstruct[r] == a.instruct && a.modes.contains(a.shipmode[r])) {
          sum += q19Row(a, r);
        }
      }
    }
  }
  sum = warpSum(sum);
  if (threadIdx.x % 32 == 0 && sum) {
    atomicAdd(a.revenue, (unsigned long long)sum);
  }
}

const std::vector<ResultField> kSchema = {{"revenue", ResultType::kDecimal, 38, 4}};

Result q19(QueryContext& ctx) {
  Q19Args a;
  a.shipinstruct = ctx.col<int8_t>("lineitem", "l_shipinstruct");
  a.shipmode = ctx.col<int8_t>("lineitem", "l_shipmode");
  a.partkey = ctx.col<int32_t>("lineitem", "l_partkey");
  a.quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.brand = ctx.col<int8_t>("part", "p_brand");
  a.container = ctx.col<int8_t>("part", "p_container");
  a.size = ctx.col<int8_t>("part", "p_size");
  a.rows = ctx.rows("lineitem");
  a.instruct = ctx.param("shipinstruct", ctx.meta("lineitem", "l_shipinstruct").code("DELIVER IN PERSON"));
  a.modes = codesParam(ctx, "shipmodes", dictionaryCodes(ctx, "lineitem", "l_shipmode", {"AIR", "AIR REG"}));
  const Column& brand = ctx.meta("part", "p_brand");
  const Branch defaults[3] = {
      {brand.code("Brand#12"),
       dictionaryCodes(ctx, "part", "p_container", {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}),
       100,
       1100,
       1,
       5},
      {brand.code("Brand#23"),
       dictionaryCodes(ctx, "part", "p_container", {"MED BAG", "MED BOX", "MED PKG", "MED PACK"}),
       1000,
       2000,
       1,
       10},
      {brand.code("Brand#34"),
       dictionaryCodes(ctx, "part", "p_container", {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}),
       2000,
       3000,
       1,
       15}};
  for (int b = 0; b < 3; ++b) {
    const std::string n = std::to_string(b);
    const Branch& d = defaults[b];
    a.branches[b] = {
        static_cast<int32_t>(ctx.param("brand" + n, d.brand)),
        codesParam(ctx, "containers" + n, d.containers),
        static_cast<int32_t>(ctx.param("quantity_min" + n, d.quantityLow)),
        static_cast<int32_t>(ctx.param("quantity_max" + n, d.quantityHigh)),
        static_cast<int32_t>(ctx.param("size_min" + n, d.sizeLow)),
        static_cast<int32_t>(ctx.param("size_max" + n, d.sizeHigh))};
  }
  a.revenue = ctx.zeroed<unsigned long long>(1);
  forBatches(
      ctx,
      0,
      ctx.rows("lineitem"),
      [&](int64_t first, int64_t last) {
        a.firstRow = first;
        a.rows = last;
        q19Kernel<<<persistentBlocks(q19Kernel, kThreads), kThreads, 0, ctx.stream()>>>(a);
      },
      1,
      QueryContext::kZoneRows);
  SV_CUDA(cudaGetLastError());
  Result result(kSchema);
  result.decimals[0].push_back(ctx.download(a.revenue, 1)[0]);
  return result;
}

} // namespace

QuerySpec q19Spec() {
  return {19, q19, kSchema};
}

} // namespace facebook::velox::supervelox
