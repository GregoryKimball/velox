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

/// (supplier nation is the first nation) x (year 1995 or 1996).
constexpr int kBins = 4;

struct Q7Args {
  const uint32_t* starts;
  const int32_t* custkey;
  const int8_t* customerNation;
  const int32_t* suppkey;
  const int8_t* supplierNation;
  const int16_t* shipdate;
  const int32_t* price;
  const int8_t* discount;
  int64_t begin;
  int64_t end;
  int32_t first;
  int32_t second;
  int32_t dateLow;
  int32_t dateHigh;
  int32_t secondYear;
  unsigned long long* revenue;
};

__global__ void q7Kernel(Q7Args a) {
  __shared__ unsigned long long local[kBins];
  if (threadIdx.x < kBins) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int64_t o = a.begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.end;
       o += (int64_t)gridDim.x * blockDim.x) {
    const int customer = a.customerNation[a.custkey[o] - 1];
    if (customer != a.first && customer != a.second) {
      continue;
    }
    const int wanted = customer == a.first ? a.second : a.first;
    for (uint32_t line = a.starts[o]; line < a.starts[o + 1]; ++line) {
      const int date = a.shipdate[line];
      if (date < a.dateLow || date > a.dateHigh || a.supplierNation[a.suppkey[line] - 1] != wanted) {
        continue;
      }
      const int bin = (wanted == a.first ? 0 : 2) + (date >= a.secondYear ? 1 : 0);
      atomicAdd(&local[bin], (unsigned long long)a.price[line] * (100 - a.discount[line]));
    }
  }
  __syncthreads();
  if (threadIdx.x < kBins && local[threadIdx.x]) {
    atomicAdd(&a.revenue[threadIdx.x], local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"supp_nation", ResultType::kVarchar},
    {"cust_nation", ResultType::kVarchar},
    {"l_year", ResultType::kBigint},
    {"revenue", ResultType::kDecimal, 38, 4},
};

Result q7(QueryContext& ctx) {
  Q7Args a;
  a.dateLow = days(1995, 1, 1);
  a.dateHigh = days(1996, 12, 31);
  a.secondYear = days(1996, 1, 1);
  const auto lag = lineDateLag(ctx, "l_shipdate");
  const auto range = sortedRange(ctx, "orders", "o_orderdate", a.dateLow - lag.second, a.dateHigh - lag.first + 1);
  a.begin = range.first;
  a.end = range.second;
  a.starts = orderLineStarts(ctx);
  a.custkey = ctx.col<int32_t>("orders", "o_custkey");
  a.customerNation = ctx.col<int8_t>("customer", "c_nationkey");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  const std::string names[2] = {"FRANCE", "GERMANY"};
  a.first = nationKey(ctx, names[0]);
  a.second = nationKey(ctx, names[1]);
  a.revenue = ctx.zeroed<unsigned long long>(kBins);
  const std::pair<int64_t, int64_t> all{a.begin, a.end};
  forBatches(
      ctx,
      all.first,
      all.second,
      [&](int64_t first, int64_t last) {
        a.begin = first;
        a.end = last;
        q7Kernel<<<2048, 256, 0, ctx.stream()>>>(a);
      },
      4);
  SV_CUDA(cudaGetLastError());
  auto revenue = ctx.download(a.revenue, kBins);
  Result result(kSchema);
  for (int bin = 0; bin < kBins; ++bin) {
    if (revenue[bin] == 0) {
      continue;
    }
    const int supplier = bin / 2;
    result.strings[0].push_back(names[supplier]);
    result.strings[1].push_back(names[1 - supplier]);
    result.ints[2].push_back(1995 + bin % 2);
    result.decimals[3].push_back(revenue[bin]);
  }
  return result;
}

} // namespace

QuerySpec q7Spec() {
  return {7, q7, kSchema};
}

} // namespace facebook::velox::supervelox
