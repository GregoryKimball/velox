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
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerThread = 8;

struct Q6Args {
  const int16_t* shipdate;
  const int8_t* discount;
  const int16_t* quantity;
  const int32_t* price;
  const int2* zones;
  int64_t rows;
  int32_t dateLow;
  int32_t dateHigh;
  int32_t discountLow;
  int32_t discountHigh;
  int32_t quantityLimit;
};

template <bool kCheckDate>
__device__ __forceinline__ long long q6Rows(const Q6Args& a, int64_t row, int64_t end) {
  long long sum = 0;
  if (row + kRowsPerThread <= end) {
    int8_t discount[kRowsPerThread];
    int16_t quantity[kRowsPerThread];
    loadVector(a.discount + row, discount);
    loadVector(a.quantity + row, quantity);
    int16_t shipdate[kRowsPerThread];
    if constexpr (kCheckDate) {
      loadVector(a.shipdate + row, shipdate);
    }
    bool any = false;
    bool pass[kRowsPerThread];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
      pass[i] = discount[i] >= a.discountLow && discount[i] <= a.discountHigh &&
          quantity[i] < a.quantityLimit;
      if constexpr (kCheckDate) {
        pass[i] = pass[i] && shipdate[i] >= a.dateLow && shipdate[i] < a.dateHigh;
      }
      any |= pass[i];
    }
    if (any) {
      int32_t price[kRowsPerThread];
      loadVector(a.price + row, price);
#pragma unroll
      for (int i = 0; i < kRowsPerThread; ++i) {
        sum += pass[i] ? (long long)price[i] * discount[i] : 0;
      }
    }
    return sum;
  }
  for (; row < end; ++row) {
    bool pass = a.discount[row] >= a.discountLow && a.discount[row] <= a.discountHigh &&
        a.quantity[row] < a.quantityLimit && a.shipdate[row] >= a.dateLow &&
        a.shipdate[row] < a.dateHigh;
    sum += pass ? (long long)a.price[row] * a.discount[row] : 0;
  }
  return sum;
}

__global__ void __launch_bounds__(kThreads) q6Kernel(Q6Args a, unsigned long long* out) {
  constexpr int64_t kZone = QueryContext::kZoneRows;
  const int64_t zoneCount = (a.rows + kZone - 1) / kZone;
  long long sum = 0;
  for (int64_t zone = blockIdx.x; zone < zoneCount; zone += gridDim.x) {
    const int2 range = a.zones[zone];
    if (range.y < a.dateLow || range.x >= a.dateHigh) {
      continue;
    }
    const bool full = range.x >= a.dateLow && range.y < a.dateHigh;
    const int64_t end = min(a.rows, (zone + 1) * kZone);
    for (int64_t row = zone * kZone + threadIdx.x * kRowsPerThread; row < end;
         row += kThreads * kRowsPerThread) {
      sum += full ? q6Rows<false>(a, row, end) : q6Rows<true>(a, row, end);
    }
  }
  sum = warpSum(sum);
  if (threadIdx.x % 32 == 0 && sum != 0) {
    atomicAdd(out, (unsigned long long)sum);
  }
}

const std::vector<ResultField> kSchema = {{"revenue", ResultType::kDecimal, 38, 4}};

Result q6(QueryContext& ctx) {
  Q6Args a;
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.zones = ctx.zones("lineitem", "l_shipdate");
  a.rows = ctx.rows("lineitem");
  a.dateLow = ctx.param("shipdate_min", days(1994, 1, 1));
  a.dateHigh = ctx.param("shipdate_max", days(1994, 12, 31)) + 1;
  a.discountLow = ctx.param("discount_min", 5);
  a.discountHigh = ctx.param("discount_max", 7);
  a.quantityLimit = ctx.param("quantity_max", 2399) + 1;
  auto* sum = ctx.zeroed<unsigned long long>(1);
  q6Kernel<<<persistentBlocks(q6Kernel, kThreads), kThreads, 0, ctx.stream()>>>(a, sum);
  SV_CUDA(cudaGetLastError());
  Result result(kSchema);
  result.decimals[0].push_back(ctx.download(sum, 1)[0]);
  return result;
}

} // namespace

QuerySpec q6Spec() {
  return {6, q6, kSchema};
}

} // namespace facebook::velox::supervelox
