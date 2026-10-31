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

__global__ void q18Kernel(
    const uint32_t* starts,
    const int16_t* quantity,
    int64_t begin,
    int64_t end,
    int32_t limit,
    Candidate* out,
    unsigned long long* count,
    unsigned long long capacity) {
  for (int64_t o = begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < end; o += (int64_t)gridDim.x * blockDim.x) {
    int sum = 0;
    for (uint32_t line = starts[o]; line < starts[o + 1]; ++line) {
      sum += quantity[line];
    }
    if (sum > limit) {
      const unsigned long long slot = atomicAdd(count, 1ULL);
      if (slot < capacity) {
        out[slot] = Candidate{o, sum};
      }
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"c_name", ResultType::kVarchar},
    {"c_custkey", ResultType::kBigint},
    {"o_orderkey", ResultType::kBigint},
    {"o_orderdate", ResultType::kDate},
    {"o_totalprice", ResultType::kDecimal, 15, 2},
    {"sum(l_quantity)", ResultType::kDecimal, 38, 2},
};

Result q18(QueryContext& ctx) {
  const int64_t orders = ctx.rows("orders");
  const uint32_t* starts = orderLineStarts(ctx);
  const int16_t* quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  std::vector<Candidate> candidates;
  for (unsigned long long capacity = 1 << 16;; capacity *= 16) {
    auto* out = ctx.scratch<Candidate>(capacity);
    auto* count = ctx.zeroed<unsigned long long>(1);
    forBatches(
        ctx,
        0,
        orders,
        [&](int64_t first, int64_t last) {
          q18Kernel<<<4096, 256, 0, ctx.stream()>>>(starts, quantity, first, last, 30000, out, count, capacity);
        },
        4);
    SV_CUDA(cudaGetLastError());
    const unsigned long long found = ctx.download(count, 1)[0];
    if (found <= capacity) {
      candidates = ctx.download(out, found);
      break;
    }
  }
  const Column& totalprice = ctx.meta("orders", "o_totalprice");
  const Column& orderdate = ctx.meta("orders", "o_orderdate");
  std::sort(candidates.begin(), candidates.end(), [&](const Candidate& x, const Candidate& y) {
    const int64_t px = totalprice.value(x.index);
    const int64_t py = totalprice.value(y.index);
    if (px != py) {
      return px > py;
    }
    return orderdate.value(x.index) < orderdate.value(y.index);
  });
  candidates.resize(std::min<size_t>(candidates.size(), 100));
  const Column& custkey = ctx.meta("orders", "o_custkey");
  const Column& orderkey = ctx.meta("orders", "o_orderkey");
  const Column& name = ctx.meta("customer", "c_name");
  Result result(kSchema);
  for (const auto& c : candidates) {
    const int64_t customer = custkey.value(c.index);
    result.strings[0].push_back(name.string(customer - 1));
    result.ints[1].push_back(customer);
    result.ints[2].push_back(orderkey.value(c.index));
    result.ints[3].push_back(orderdate.value(c.index));
    result.decimals[4].push_back(totalprice.value(c.index));
    result.decimals[5].push_back(c.value);
  }
  return result;
}

} // namespace

QuerySpec q18Spec() {
  return {18, q18, kSchema};
}

} // namespace facebook::velox::supervelox
