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

constexpr int kNations = 32;

struct Q5Args {
  const uint32_t* starts;
  const int32_t* custkey;
  const int8_t* customerNation;
  const int32_t* suppkey;
  const int8_t* supplierNation;
  const int32_t* price;
  const int8_t* discount;
  int64_t begin;
  int64_t end;
  uint32_t nations;
  unsigned long long* revenue;
};

__global__ void q5Kernel(Q5Args a) {
  __shared__ unsigned long long local[kNations];
  if (threadIdx.x < kNations) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int64_t o = a.begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.end;
       o += (int64_t)gridDim.x * blockDim.x) {
    const int nation = a.customerNation[a.custkey[o] - 1];
    if (!((a.nations >> nation) & 1)) {
      continue;
    }
    long long revenue = 0;
    for (uint32_t line = a.starts[o]; line < a.starts[o + 1]; ++line) {
      if (a.supplierNation[a.suppkey[line] - 1] == nation) {
        revenue += (long long)a.price[line] * (100 - a.discount[line]);
      }
    }
    if (revenue) {
      atomicAdd(&local[nation], (unsigned long long)revenue);
    }
  }
  __syncthreads();
  if (threadIdx.x < kNations && local[threadIdx.x]) {
    atomicAdd(&a.revenue[threadIdx.x], local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"n_name", ResultType::kVarchar},
    {"revenue", ResultType::kDecimal, 38, 4},
};

Result q5(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1994, 1, 1), days(1995, 1, 1));
  Q5Args a;
  a.starts = orderLineStarts(ctx);
  a.custkey = ctx.col<int32_t>("orders", "o_custkey");
  a.customerNation = ctx.col<int8_t>("customer", "c_nationkey");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.begin = begin;
  a.end = end;
  a.nations = nationsInRegion(ctx, "ASIA");
  a.revenue = ctx.zeroed<unsigned long long>(kNations);
  q5Kernel<<<2048, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  auto revenue = ctx.download(a.revenue, kNations);
  Result result(kSchema);
  for (int n = 0; n < kNations; ++n) {
    if ((a.nations >> n) & 1) {
      result.strings[0].push_back(nationName(ctx, n));
      result.decimals[1].push_back(revenue[n]);
    }
  }
  return result;
}

} // namespace

QuerySpec q5Spec() {
  return {5, q5, kSchema};
}

} // namespace facebook::velox::supervelox
