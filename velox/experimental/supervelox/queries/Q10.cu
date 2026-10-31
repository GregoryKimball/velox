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

__global__ void q10Kernel(
    const uint32_t* starts,
    const int32_t* custkey,
    const int8_t* returnflag,
    const int32_t* price,
    const int8_t* discount,
    int64_t begin,
    int64_t end,
    int32_t flag,
    unsigned long long* revenue) {
  for (int64_t o = begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < end;
       o += (int64_t)gridDim.x * blockDim.x) {
    long long sum = 0;
    for (uint32_t line = starts[o]; line < starts[o + 1]; ++line) {
      if (returnflag[line] == flag) {
        sum += (long long)price[line] * (100 - discount[line]);
      }
    }
    if (sum) {
      atomicAdd(&revenue[custkey[o] - 1], (unsigned long long)sum);
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"c_custkey", ResultType::kBigint},
    {"c_name", ResultType::kVarchar},
    {"revenue", ResultType::kDecimal, 38, 4},
    {"c_acctbal", ResultType::kDecimal, 15, 2},
    {"n_name", ResultType::kVarchar},
    {"c_address", ResultType::kVarchar},
    {"c_phone", ResultType::kVarchar},
    {"c_comment", ResultType::kVarchar},
};

Result q10(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1993, 10, 1), days(1994, 1, 1));
  const int64_t customers = ctx.rows("customer");
  auto* revenue = ctx.zeroed<unsigned long long>(customers);
  const uint32_t* starts = orderLineStarts(ctx);
  const int32_t* custkey = ctx.col<int32_t>("orders", "o_custkey");
  const int8_t* returnflag = ctx.col<int8_t>("lineitem", "l_returnflag");
  const int32_t* price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  const int8_t* discount = ctx.col<int8_t>("lineitem", "l_discount");
  const int32_t returned = ctx.meta("lineitem", "l_returnflag").code("R");
  forBatches(
      ctx,
      begin,
      end,
      [&](int64_t first, int64_t last) {
        q10Kernel<<<2048, 256, 0, ctx.stream()>>>(
            starts, custkey, returnflag, price, discount, first, last, returned, revenue);
      },
      4);
  SV_CUDA(cudaGetLastError());
  auto candidates = topCandidates(ctx, reinterpret_cast<const int64_t*>(revenue), customers, 20);
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& x, const Candidate& y) {
    return x.value != y.value ? x.value > y.value : x.index < y.index;
  });
  candidates.resize(std::min<size_t>(candidates.size(), 20));
  const Column& name = ctx.meta("customer", "c_name");
  const Column& acctbal = ctx.meta("customer", "c_acctbal");
  const Column& nation = ctx.meta("customer", "c_nationkey");
  const Column& address = ctx.meta("customer", "c_address");
  const Column& phone = ctx.meta("customer", "c_phone");
  const Column& comment = ctx.meta("customer", "c_comment");
  Result result(kSchema);
  for (const auto& c : candidates) {
    result.ints[0].push_back(c.index + 1);
    result.strings[1].push_back(name.string(c.index));
    result.decimals[2].push_back(c.value);
    result.decimals[3].push_back(acctbal.value(c.index));
    result.strings[4].push_back(nationName(ctx, nation.value(c.index)));
    result.strings[5].push_back(address.string(c.index));
    result.strings[6].push_back(phone.string(c.index));
    result.strings[7].push_back(comment.string(c.index));
  }
  return result;
}

} // namespace

QuerySpec q10Spec() {
  return {10, q10, kSchema};
}

} // namespace facebook::velox::supervelox
