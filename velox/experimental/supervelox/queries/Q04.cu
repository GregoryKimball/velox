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

constexpr int kPriorities = 8;

__global__ void q4Kernel(
    const uint32_t* starts,
    const int8_t* priority,
    const int16_t* commitdate,
    const int16_t* receiptdate,
    int64_t begin,
    int64_t end,
    unsigned long long* counts) {
  __shared__ unsigned local[kPriorities];
  if (threadIdx.x < kPriorities) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int64_t o = begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < end;
       o += (int64_t)gridDim.x * blockDim.x) {
    bool late = false;
    for (uint32_t line = starts[o]; line < starts[o + 1] && !late; ++line) {
      late = commitdate[line] < receiptdate[line];
    }
    if (late) {
      atomicAdd(&local[priority[o]], 1u);
    }
  }
  __syncthreads();
  if (threadIdx.x < kPriorities && local[threadIdx.x]) {
    atomicAdd(&counts[threadIdx.x], (unsigned long long)local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"o_orderpriority", ResultType::kVarchar},
    {"order_count", ResultType::kBigint},
};

Result q4(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1993, 7, 1), days(1993, 10, 1));
  auto* counts = ctx.zeroed<unsigned long long>(kPriorities);
  q4Kernel<<<1024, 256, 0, ctx.stream()>>>(
      orderLineStarts(ctx),
      ctx.col<int8_t>("orders", "o_orderpriority"),
      ctx.col<int16_t>("lineitem", "l_commitdate"),
      ctx.col<int16_t>("lineitem", "l_receiptdate"),
      begin,
      end,
      counts);
  SV_CUDA(cudaGetLastError());
  auto host = ctx.download(counts, kPriorities);
  const auto& names = ctx.meta("orders", "o_orderpriority").dictionary;
  Result result(kSchema);
  for (size_t i = 0; i < names.size(); ++i) {
    if (host[i]) {
      result.strings[0].push_back(names[i]);
      result.ints[1].push_back(host[i]);
    }
  }
  return result;
}

} // namespace

QuerySpec q4Spec() {
  return {4, q4, kSchema};
}

} // namespace facebook::velox::supervelox
