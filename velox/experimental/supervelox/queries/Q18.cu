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
    int64_t lineBase,
    const int16_t* quantity,
    int64_t begin,
    int64_t end,
    int32_t limit,
    Candidate* out,
    unsigned long long* count,
    unsigned long long capacity) {
  for (int64_t o = begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < end; o += (int64_t)gridDim.x * blockDim.x) {
    int sum = 0;
    for (int64_t line = lineBase + starts[o]; line < lineBase + starts[o + 1]; ++line) {
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
  std::vector<Candidate> candidates;
  auto* count = ctx.scratch<unsigned long long>(1);
  forBatches(
      ctx,
      0,
      orders,
      [&](int64_t first, int64_t last) {
        const LineRange lines = lineRange(ctx, first, last);
        const int16_t* quantity = ctx.rows<int16_t>("lineitem", "l_quantity", lines.firstLine, lines.lastLine);
        // Few orders qualify; if more than the guess, rerun with room for all.
        for (unsigned long long capacity = std::min<int64_t>(1 << 16, last - first);; capacity = last - first) {
          auto* out = ctx.batchScratch<Candidate>(std::max<unsigned long long>(capacity, 1));
          SV_CUDA(cudaMemsetAsync(count, 0, sizeof(*count), ctx.stream()));
          q18Kernel<<<4096, 256, 0, ctx.stream()>>>(
              lines.starts, lines.base, quantity, first, last, 30000, out, count, capacity);
          SV_CUDA(cudaGetLastError());
          const unsigned long long found = ctx.download(count, 1)[0];
          if (found <= capacity) {
            const auto batch = ctx.download(out, found);
            candidates.insert(candidates.end(), batch.begin(), batch.end());
            break;
          }
        }
      },
      4);
  auto totalprice = [&](int64_t row) { return ctx.value("orders", "o_totalprice", row); };
  auto orderdate = [&](int64_t row) { return ctx.value("orders", "o_orderdate", row); };
  struct Ranked {
    int64_t price;
    int64_t date;
    Candidate candidate;
  };
  std::vector<Ranked> ranked;
  ranked.reserve(candidates.size());
  for (const auto& c : candidates) {
    ranked.push_back({totalprice(c.index), orderdate(c.index), c});
  }
  std::sort(ranked.begin(), ranked.end(), [](const Ranked& x, const Ranked& y) {
    return x.price != y.price ? x.price > y.price : x.date < y.date;
  });
  ranked.resize(std::min<size_t>(ranked.size(), 100));
  candidates.clear();
  for (const auto& r : ranked) {
    candidates.push_back(r.candidate);
  }
  const Column& name = ctx.meta("customer", "c_name");
  Result result(kSchema);
  for (const auto& c : candidates) {
    const int64_t customer = ctx.value("orders", "o_custkey", c.index);
    result.strings[0].push_back(name.string(customer - 1));
    result.ints[1].push_back(customer);
    result.ints[2].push_back(ctx.value("orders", "o_orderkey", c.index));
    result.ints[3].push_back(orderdate(c.index));
    result.decimals[4].push_back(totalprice(c.index));
    result.decimals[5].push_back(c.value);
  }
  return result;
}

} // namespace

QuerySpec q18Spec() {
  return {18, q18, kSchema};
}

} // namespace facebook::velox::supervelox
