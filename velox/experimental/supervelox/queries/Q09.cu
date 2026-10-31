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

constexpr int kNations = 25;
constexpr int kFirstYear = 1992;
constexpr int kYears = 8;
constexpr int kBins = kNations * kYears;

struct Q9Args {
  const uint32_t* starts;
  const int16_t* orderdate;
  const int32_t* partkey;
  const int32_t* suppkey;
  const int16_t* quantity;
  const int32_t* price;
  const int8_t* discount;
  const int32_t* partsuppSuppkey;
  const int32_t* supplycost;
  const int8_t* supplierNation;
  const uint32_t* parts;
  int64_t firstOrder;
  int64_t orders;
  unsigned long long* profit;
  unsigned long long* missing;
};

__global__ void q9Kernel(Q9Args a) {
  __shared__ unsigned long long local[kBins];
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();
  for (int64_t o = a.firstOrder + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.orders; o += (int64_t)gridDim.x * blockDim.x) {
    const int year = yearOf(a.orderdate[o]) - kFirstYear;
    for (uint32_t line = a.starts[o]; line < a.starts[o + 1]; ++line) {
      const int part = a.partkey[line];
      if (!testBit(a.parts, part - 1)) {
        continue;
      }
      const int supplier = a.suppkey[line];
      const int64_t base = (int64_t)(part - 1) * 4;
      int slot = 0;
      while (slot < 4 && a.partsuppSuppkey[base + slot] != supplier) {
        ++slot;
      }
      if (slot == 4) {
        atomicAdd(a.missing, 1ULL);
        continue;
      }
      const long long amount = (long long)a.price[line] * (100 - a.discount[line]) -
          (long long)a.supplycost[base + slot] * a.quantity[line];
      atomicAdd(&local[a.supplierNation[supplier - 1] * kYears + year], (unsigned long long)amount);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    if (local[i]) {
      atomicAdd(&a.profit[i], local[i]);
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"nation", ResultType::kVarchar},
    {"o_year", ResultType::kBigint},
    {"sum_profit", ResultType::kDecimal, 38, 4},
};

Result q9(QueryContext& ctx) {
  Q9Args a;
  a.parts = likeBitmap(ctx, "part", "p_name", Match::kContains, "green");
  a.starts = orderLineStarts(ctx);
  a.orderdate = ctx.col<int16_t>("orders", "o_orderdate");
  a.partkey = ctx.col<int32_t>("lineitem", "l_partkey");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.partsuppSuppkey = ctx.col<int32_t>("partsupp", "ps_suppkey");
  a.supplycost = ctx.col<int32_t>("partsupp", "ps_supplycost");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.profit = ctx.zeroed<unsigned long long>(kBins);
  a.missing = ctx.zeroed<unsigned long long>(1);
  forBatches(
      ctx,
      0,
      ctx.rows("orders"),
      [&](int64_t first, int64_t last) {
        a.firstOrder = first;
        a.orders = last;
        q9Kernel<<<4096, 256, 0, ctx.stream()>>>(a);
      },
      4);
  SV_CUDA(cudaGetLastError());
  auto profit = ctx.download(a.profit, kBins);
  if (ctx.download(a.missing, 1)[0]) {
    throw std::runtime_error("q9: lineitem (part, supplier) missing from partsupp layout");
  }
  std::vector<std::pair<std::string, int>> nations;
  for (int n = 0; n < kNations; ++n) {
    nations.emplace_back(nationName(ctx, n), n);
  }
  std::sort(nations.begin(), nations.end());
  Result result(kSchema);
  for (const auto& [name, n] : nations) {
    for (int year = kYears - 1; year >= 0; --year) {
      const auto value = (long long)profit[n * kYears + year];
      if (value) {
        result.strings[0].push_back(name);
        result.ints[1].push_back(kFirstYear + year);
        result.decimals[2].push_back(value);
      }
    }
  }
  return result;
}

} // namespace

QuerySpec q9Spec() {
  return {9, q9, kSchema};
}

} // namespace facebook::velox::supervelox
