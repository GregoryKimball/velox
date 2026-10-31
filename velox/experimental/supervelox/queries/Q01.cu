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
/// returnflag (3 values) x linestatus (2 values).
constexpr int kGroups = 6;

struct Q1Args {
  const int8_t* returnflag;
  const int8_t* linestatus;
  const int16_t* quantity;
  const int32_t* price;
  const int8_t* discount;
  const int8_t* tax;
  const int16_t* shipdate;
  const int2* zones;
  int64_t rows;
  int32_t dateLimit;
};

/// Per group: quantity, base price, discounted price, charge, discount, count.
struct Totals {
  unsigned long long quantity;
  unsigned long long price;
  unsigned long long discounted;
  unsigned long long charge[2];
  unsigned long long discount;
  unsigned long long count;
};

struct Accumulator {
  long long quantity[kGroups] = {};
  long long price[kGroups] = {};
  long long discounted[kGroups] = {};
  long long charge[kGroups] = {};
  unsigned discount[kGroups] = {};
  unsigned count[kGroups] = {};

  __device__ __forceinline__ void
  add(bool pass, int group, int quantityValue, int priceValue, int discountValue, int taxValue) {
    const long long discountedValue = (long long)priceValue * (100 - discountValue);
    const long long chargeValue = discountedValue * (100 + taxValue);
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      const bool hit = pass && g == group;
      quantity[g] += hit ? quantityValue : 0;
      price[g] += hit ? priceValue : 0;
      discounted[g] += hit ? discountedValue : 0;
      charge[g] += hit ? chargeValue : 0;
      discount[g] += hit ? discountValue : 0;
      count[g] += hit;
    }
  }
};

template <bool kCheckDate>
__device__ __forceinline__ void q1Rows(const Q1Args& a, int64_t row, int64_t end, Accumulator& acc) {
  if (row + kRowsPerThread <= end) {
    int8_t flag[kRowsPerThread];
    int8_t status[kRowsPerThread];
    int16_t quantity[kRowsPerThread];
    int32_t price[kRowsPerThread];
    int8_t discount[kRowsPerThread];
    int8_t tax[kRowsPerThread];
    int16_t shipdate[kRowsPerThread];
    loadVector(a.returnflag + row, flag);
    loadVector(a.linestatus + row, status);
    loadVector(a.quantity + row, quantity);
    loadVector(a.price + row, price);
    loadVector(a.discount + row, discount);
    loadVector(a.tax + row, tax);
    if constexpr (kCheckDate) {
      loadVector(a.shipdate + row, shipdate);
    }
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
      bool pass = true;
      if constexpr (kCheckDate) {
        pass = shipdate[i] <= a.dateLimit;
      }
      acc.add(pass, flag[i] * 2 + status[i], quantity[i], price[i], discount[i], tax[i]);
    }
    return;
  }
  for (; row < end; ++row) {
    acc.add(
        a.shipdate[row] <= a.dateLimit,
        a.returnflag[row] * 2 + a.linestatus[row],
        a.quantity[row],
        a.price[row],
        a.discount[row],
        a.tax[row]);
  }
}

__global__ void __launch_bounds__(kThreads) q1Kernel(Q1Args a, Totals* out) {
  constexpr int64_t kZone = QueryContext::kZoneRows;
  const int64_t zoneCount = (a.rows + kZone - 1) / kZone;
  Accumulator acc;
  for (int64_t zone = blockIdx.x; zone < zoneCount; zone += gridDim.x) {
    const int2 range = a.zones[zone];
    if (range.x > a.dateLimit) {
      continue;
    }
    const bool full = range.y <= a.dateLimit;
    const int64_t end = min(a.rows, (zone + 1) * kZone);
    for (int64_t row = zone * kZone + threadIdx.x * kRowsPerThread; row < end;
         row += kThreads * kRowsPerThread) {
      if (full) {
        q1Rows<false>(a, row, end, acc);
      } else {
        q1Rows<true>(a, row, end, acc);
      }
    }
  }
  __shared__ Totals shared[kGroups];
  if (threadIdx.x < kGroups * sizeof(Totals) / 8) {
    reinterpret_cast<unsigned long long*>(shared)[threadIdx.x] = 0;
  }
  __syncthreads();
  const bool leader = threadIdx.x % 32 == 0;
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    long long quantity = warpSum(acc.quantity[g]);
    long long price = warpSum(acc.price[g]);
    long long discounted = warpSum(acc.discounted[g]);
    long long charge = warpSum(acc.charge[g]);
    unsigned long long discount = warpSum((unsigned long long)acc.discount[g]);
    unsigned long long count = warpSum((unsigned long long)acc.count[g]);
    if (leader && count) {
      atomicAdd(&shared[g].quantity, (unsigned long long)quantity);
      atomicAdd(&shared[g].price, (unsigned long long)price);
      atomicAdd(&shared[g].discounted, (unsigned long long)discounted);
      atomicAdd128(shared[g].charge, (unsigned long long)charge);
      atomicAdd(&shared[g].discount, discount);
      atomicAdd(&shared[g].count, count);
    }
  }
  __syncthreads();
  if (threadIdx.x < kGroups) {
    const Totals& t = shared[threadIdx.x];
    Totals& o = out[threadIdx.x];
    if (t.count) {
      atomicAdd(&o.quantity, t.quantity);
      atomicAdd(&o.price, t.price);
      atomicAdd(&o.discounted, t.discounted);
      atomicAdd128(o.charge, t.charge[0]);
      if (t.charge[1]) {
        atomicAdd(&o.charge[1], t.charge[1]);
      }
      atomicAdd(&o.discount, t.discount);
      atomicAdd(&o.count, t.count);
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"l_returnflag", ResultType::kVarchar},
    {"l_linestatus", ResultType::kVarchar},
    {"sum_qty", ResultType::kDecimal, 38, 2},
    {"sum_base_price", ResultType::kDecimal, 38, 2},
    {"sum_disc_price", ResultType::kDecimal, 38, 4},
    {"sum_charge", ResultType::kDecimal, 38, 6},
    {"avg_qty", ResultType::kDouble},
    {"avg_price", ResultType::kDouble},
    {"avg_disc", ResultType::kDouble},
    {"count_order", ResultType::kBigint},
};

Result q1(QueryContext& ctx) {
  Q1Args a;
  a.returnflag = ctx.col<int8_t>("lineitem", "l_returnflag");
  a.linestatus = ctx.col<int8_t>("lineitem", "l_linestatus");
  a.quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.tax = ctx.col<int8_t>("lineitem", "l_tax");
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.zones = ctx.zones("lineitem", "l_shipdate");
  a.rows = ctx.rows("lineitem");
  a.dateLimit = days(1998, 9, 2);
  auto* totals = ctx.zeroed<Totals>(kGroups);
  q1Kernel<<<persistentBlocks(q1Kernel, kThreads), kThreads, 0, ctx.stream()>>>(a, totals);
  SV_CUDA(cudaGetLastError());
  auto host = ctx.download(totals, kGroups);

  const auto& flags = ctx.meta("lineitem", "l_returnflag").dictionary;
  const auto& statuses = ctx.meta("lineitem", "l_linestatus").dictionary;
  Result result(kSchema);
  for (int g = 0; g < kGroups; ++g) {
    const Totals& t = host[g];
    if (t.count == 0) {
      continue;
    }
    const double count = t.count;
    result.strings[0].push_back(flags.at(g / 2));
    result.strings[1].push_back(statuses.at(g % 2));
    result.decimals[2].push_back(t.quantity);
    result.decimals[3].push_back(t.price);
    result.decimals[4].push_back(t.discounted);
    result.decimals[5].push_back(toInt128(t.charge));
    result.doubles[6].push_back(t.quantity / 100.0 / count);
    result.doubles[7].push_back(t.price / 100.0 / count);
    result.doubles[8].push_back(t.discount / 100.0 / count);
    result.ints[9].push_back(t.count);
  }
  return result;
}

} // namespace

QuerySpec q1Spec() {
  return {1, q1, kSchema};
}

} // namespace facebook::velox::supervelox
