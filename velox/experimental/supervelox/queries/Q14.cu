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

constexpr int kThreads = 256;

struct Q14Args {
  const int16_t* shipdate;
  const int32_t* partkey;
  const int32_t* price;
  const int8_t* discount;
  const int16_t* partType;
  const int2* zones;
  int64_t rows;
  int64_t firstZone;
  int64_t lastZone;
  int32_t dateLow;
  int32_t dateHigh;
  CodeSet promo;
  unsigned long long* sums;
};

__global__ void __launch_bounds__(kThreads) q14Kernel(Q14Args a) {
  constexpr int64_t kZone = QueryContext::kZoneRows;
  unsigned long long total = 0;
  unsigned long long promo = 0;
  for (int64_t zone = a.firstZone + blockIdx.x; zone < a.lastZone; zone += gridDim.x) {
    const int2 range = a.zones[zone];
    if (range.y < a.dateLow || range.x >= a.dateHigh) {
      continue;
    }
    const int64_t end = min(a.rows, (zone + 1) * kZone);
    for (int64_t row = zone * kZone + threadIdx.x; row < end; row += kThreads) {
      const int date = a.shipdate[row];
      if (date < a.dateLow || date >= a.dateHigh) {
        continue;
      }
      const unsigned long long volume = (unsigned long long)a.price[row] * (100 - a.discount[row]);
      total += volume;
      if (a.promo.contains(a.partType[a.partkey[row] - 1])) {
        promo += volume;
      }
    }
  }
  total = warpSum(total);
  promo = warpSum(promo);
  if (threadIdx.x % 32 == 0 && total) {
    atomicAdd(&a.sums[0], total);
    atomicAdd(&a.sums[1], promo);
  }
}

const std::vector<ResultField> kSchema = {{"promo_revenue", ResultType::kDouble}};

Result q14(QueryContext& ctx) {
  Q14Args a;
  a.partType = ctx.col<int16_t>("part", "p_type");
  a.zones = ctx.zones("lineitem", "l_shipdate");
  a.rows = ctx.rows("lineitem");
  a.dateLow = ctx.param("shipdate_min", days(1995, 9, 1));
  a.dateHigh = ctx.param("shipdate_max", days(1995, 10, 1) - 1) + 1;
  a.promo = codesParam(ctx, "promo_types", dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "PROMO"));
  a.sums = ctx.zeroed<unsigned long long>(2);
  constexpr int64_t kZone = QueryContext::kZoneRows;
  forBatches(
      ctx,
      0,
      (a.rows + kZone - 1) / kZone,
      [&](int64_t first, int64_t last) {
        a.firstZone = first;
        a.lastZone = last;
        a.shipdate = ctx.rows<int16_t>("lineitem", "l_shipdate", first * kZone, std::min(a.rows, last * kZone));
        a.partkey = ctx.rows<int32_t>("lineitem", "l_partkey", first * kZone, std::min(a.rows, last * kZone));
        a.price = ctx.rows<int32_t>("lineitem", "l_extendedprice", first * kZone, std::min(a.rows, last * kZone));
        a.discount = ctx.rows<int8_t>("lineitem", "l_discount", first * kZone, std::min(a.rows, last * kZone));
        q14Kernel<<<persistentBlocks(q14Kernel, kThreads), kThreads, 0, ctx.stream()>>>(a);
      },
      kZone);
  SV_CUDA(cudaGetLastError());
  auto sums = ctx.download(a.sums, 2);
  Result result(kSchema);
  result.doubles[0].push_back(100.0 * (double)sums[1] / (double)sums[0]);
  return result;
}

} // namespace

QuerySpec q14Spec() {
  return {14, q14, kSchema};
}

} // namespace facebook::velox::supervelox
