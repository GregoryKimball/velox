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

struct Q15Args {
  const int16_t* shipdate;
  const int32_t* suppkey;
  const int32_t* price;
  const int8_t* discount;
  const int2* zones;
  int64_t rows;
  int64_t firstZone;
  int64_t lastZone;
  int32_t dateLow;
  int32_t dateHigh;
  unsigned long long* revenue;
};

__global__ void __launch_bounds__(kThreads) q15Kernel(Q15Args a) {
  constexpr int64_t kZone = QueryContext::kZoneRows;
  for (int64_t zone = a.firstZone + blockIdx.x; zone < a.lastZone; zone += gridDim.x) {
    const int2 range = a.zones[zone];
    if (range.y < a.dateLow || range.x >= a.dateHigh) {
      continue;
    }
    const int64_t end = min(a.rows, (zone + 1) * kZone);
    for (int64_t row = zone * kZone + threadIdx.x; row < end; row += kThreads) {
      const int date = a.shipdate[row];
      if (date >= a.dateLow && date < a.dateHigh) {
        atomicAdd(&a.revenue[a.suppkey[row] - 1], (unsigned long long)a.price[row] * (100 - a.discount[row]));
      }
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"s_suppkey", ResultType::kBigint},
    {"s_name", ResultType::kVarchar},
    {"s_address", ResultType::kVarchar},
    {"s_phone", ResultType::kVarchar},
    {"total_revenue", ResultType::kDecimal, 38, 4},
};

Result q15(QueryContext& ctx) {
  Q15Args a;
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.zones = ctx.zones("lineitem", "l_shipdate");
  a.rows = ctx.rows("lineitem");
  a.dateLow = days(1996, 1, 1);
  a.dateHigh = days(1996, 4, 1);
  const int64_t suppliers = ctx.rows("supplier");
  a.revenue = ctx.zeroed<unsigned long long>(suppliers);
  constexpr int64_t kZone = QueryContext::kZoneRows;
  forBatches(
      ctx,
      0,
      (a.rows + kZone - 1) / kZone,
      [&](int64_t first, int64_t last) {
        a.firstZone = first;
        a.lastZone = last;
        q15Kernel<<<persistentBlocks(q15Kernel, kThreads), kThreads, 0, ctx.stream()>>>(a);
      },
      kZone);
  SV_CUDA(cudaGetLastError());
  auto candidates = topCandidates(ctx, reinterpret_cast<const int64_t*>(a.revenue), suppliers, 1);
  int64_t best = 0;
  for (const auto& c : candidates) {
    best = std::max(best, c.value);
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& x, const Candidate& y) {
    return x.index < y.index;
  });
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  Result result(kSchema);
  for (const auto& c : candidates) {
    if (c.value != best) {
      continue;
    }
    result.ints[0].push_back(c.index + 1);
    result.strings[1].push_back(name.string(c.index));
    result.strings[2].push_back(address.string(c.index));
    result.strings[3].push_back(phone.string(c.index));
    result.decimals[4].push_back(c.value);
  }
  return result;
}

} // namespace

QuerySpec q15Spec() {
  return {15, q15, kSchema};
}

} // namespace facebook::velox::supervelox
