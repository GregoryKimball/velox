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

/// Per year 1995 and 1996: total volume, then the nation's volume.
constexpr int kBins = 4;

struct Q8Args {
  const uint32_t* starts;
  const int32_t* custkey;
  const int16_t* orderdate;
  const int8_t* customerNation;
  const int32_t* partkey;
  const int16_t* partType;
  const int32_t* suppkey;
  const int8_t* supplierNation;
  const int32_t* price;
  const int8_t* discount;
  int64_t begin;
  int64_t end;
  uint32_t nations;
  int32_t typeCode;
  int32_t nation;
  int32_t secondYear;
  unsigned long long* volume;
};

__global__ void q8Kernel(Q8Args a) {
  __shared__ unsigned long long local[kBins];
  if (threadIdx.x < kBins) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int64_t o = a.begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.end;
       o += (int64_t)gridDim.x * blockDim.x) {
    if (!((a.nations >> a.customerNation[a.custkey[o] - 1]) & 1)) {
      continue;
    }
    const int year = a.orderdate[o] >= a.secondYear ? 1 : 0;
    for (uint32_t line = a.starts[o]; line < a.starts[o + 1]; ++line) {
      if (a.partType[a.partkey[line] - 1] != a.typeCode) {
        continue;
      }
      const unsigned long long volume = (unsigned long long)a.price[line] * (100 - a.discount[line]);
      atomicAdd(&local[year * 2], volume);
      if (a.supplierNation[a.suppkey[line] - 1] == a.nation) {
        atomicAdd(&local[year * 2 + 1], volume);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < kBins && local[threadIdx.x]) {
    atomicAdd(&a.volume[threadIdx.x], local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"o_year", ResultType::kBigint},
    {"mkt_share", ResultType::kDouble},
};

Result q8(QueryContext& ctx) {
  Q8Args a;
  const auto range = sortedRange(ctx, "orders", "o_orderdate", days(1995, 1, 1), days(1996, 12, 31) + 1);
  a.begin = range.first;
  a.end = range.second;
  a.secondYear = days(1996, 1, 1);
  a.starts = orderLineStarts(ctx);
  a.custkey = ctx.col<int32_t>("orders", "o_custkey");
  a.orderdate = ctx.col<int16_t>("orders", "o_orderdate");
  a.customerNation = ctx.col<int8_t>("customer", "c_nationkey");
  a.partkey = ctx.col<int32_t>("lineitem", "l_partkey");
  a.partType = ctx.col<int16_t>("part", "p_type");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.nations = nationsInRegion(ctx, "AMERICA");
  a.typeCode = ctx.meta("part", "p_type").code("ECONOMY ANODIZED STEEL");
  a.nation = nationKey(ctx, "BRAZIL");
  a.volume = ctx.zeroed<unsigned long long>(kBins);
  q8Kernel<<<2048, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  auto volume = ctx.download(a.volume, kBins);
  Result result(kSchema);
  for (int year = 0; year < 2; ++year) {
    if (volume[year * 2]) {
      result.ints[0].push_back(1995 + year);
      result.doubles[1].push_back((double)volume[year * 2 + 1] / (double)volume[year * 2]);
    }
  }
  return result;
}

} // namespace

QuerySpec q8Spec() {
  return {8, q8, kSchema};
}

} // namespace facebook::velox::supervelox
