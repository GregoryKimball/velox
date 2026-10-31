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

constexpr int kModes = 8;

struct Q12Args {
  const uint32_t* starts;
  const int8_t* priority;
  const int8_t* shipmode;
  const int16_t* shipdate;
  const int16_t* commitdate;
  const int16_t* receiptdate;
  int64_t begin;
  int64_t end;
  CodeSet modes;
  CodeSet high;
  int32_t dateLow;
  int32_t dateHigh;
  unsigned long long* counts;
};

__global__ void q12Kernel(Q12Args a) {
  __shared__ unsigned local[kModes * 2];
  if (threadIdx.x < kModes * 2) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int64_t o = a.begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.end;
       o += (int64_t)gridDim.x * blockDim.x) {
    const int isHigh = a.high.contains(a.priority[o]) ? 0 : 1;
    for (uint32_t line = a.starts[o]; line < a.starts[o + 1]; ++line) {
      const int mode = a.shipmode[line];
      if (!a.modes.contains(mode)) {
        continue;
      }
      const int receipt = a.receiptdate[line];
      const int commit = a.commitdate[line];
      if (receipt >= a.dateLow && receipt < a.dateHigh && commit < receipt && a.shipdate[line] < commit) {
        atomicAdd(&local[(mode & (kModes - 1)) * 2 + isHigh], 1u);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < kModes * 2 && local[threadIdx.x]) {
    atomicAdd(&a.counts[threadIdx.x], (unsigned long long)local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"l_shipmode", ResultType::kVarchar},
    {"high_line_count", ResultType::kBigint},
    {"low_line_count", ResultType::kBigint},
};

Result q12(QueryContext& ctx) {
  Q12Args a;
  a.dateLow = days(1994, 1, 1);
  a.dateHigh = days(1995, 1, 1);
  const auto lag = lineDateLag(ctx, "l_receiptdate");
  const auto range = sortedRange(ctx, "orders", "o_orderdate", a.dateLow - lag.second, a.dateHigh - lag.first);
  a.begin = range.first;
  a.end = range.second;
  a.starts = orderLineStarts(ctx);
  a.priority = ctx.col<int8_t>("orders", "o_orderpriority");
  a.shipmode = ctx.col<int8_t>("lineitem", "l_shipmode");
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.commitdate = ctx.col<int16_t>("lineitem", "l_commitdate");
  a.receiptdate = ctx.col<int16_t>("lineitem", "l_receiptdate");
  a.modes = dictionaryCodes(ctx, "lineitem", "l_shipmode", {"MAIL", "SHIP"});
  a.high = dictionaryCodes(ctx, "orders", "o_orderpriority", {"1-URGENT", "2-HIGH"});
  a.counts = ctx.zeroed<unsigned long long>(kModes * 2);
  q12Kernel<<<2048, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  auto counts = ctx.download(a.counts, kModes * 2);
  const auto& names = ctx.meta("lineitem", "l_shipmode").dictionary;
  if (names.size() > kModes) {
    throw std::runtime_error("too many ship modes");
  }
  Result result(kSchema);
  for (size_t mode = 0; mode < names.size(); ++mode) {
    if (a.modes.contains(mode)) {
      result.strings[0].push_back(names[mode]);
      result.ints[1].push_back(counts[mode * 2]);
      result.ints[2].push_back(counts[mode * 2 + 1]);
    }
  }
  return result;
}

} // namespace

QuerySpec q12Spec() {
  return {12, q12, kSchema};
}

} // namespace facebook::velox::supervelox
