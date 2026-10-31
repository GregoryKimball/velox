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

struct Q3Args {
  const uint32_t* starts;
  const int32_t* custkey;
  const int8_t* segment;
  const int16_t* shipdate;
  const int32_t* price;
  const int8_t* discount;
  int64_t begin;
  int64_t end;
  int32_t segmentCode;
  int32_t date;
  int64_t* revenue;
};

__global__ void q3Kernel(Q3Args a) {
  for (int64_t o = a.begin + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.end;
       o += (int64_t)gridDim.x * blockDim.x) {
    long long revenue = 0;
    if (a.segment[a.custkey[o] - 1] == a.segmentCode) {
      for (uint32_t line = a.starts[o]; line < a.starts[o + 1]; ++line) {
        if (a.shipdate[line] > a.date) {
          revenue += (long long)a.price[line] * (100 - a.discount[line]);
        }
      }
    }
    a.revenue[o - a.begin] = revenue;
  }
}

const std::vector<ResultField> kSchema = {
    {"l_orderkey", ResultType::kBigint},
    {"revenue", ResultType::kDecimal, 38, 4},
    {"o_orderdate", ResultType::kDate},
    {"o_shippriority", ResultType::kInteger},
};

Result q3(QueryContext& ctx) {
  const int32_t date = days(1995, 3, 15);
  const auto lag = lineDateLag(ctx, "l_shipdate");
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", date - lag.second + 1, date);
  Q3Args a;
  a.starts = orderLineStarts(ctx);
  a.custkey = ctx.col<int32_t>("orders", "o_custkey");
  a.segment = ctx.col<int8_t>("customer", "c_mktsegment");
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.price = ctx.col<int32_t>("lineitem", "l_extendedprice");
  a.discount = ctx.col<int8_t>("lineitem", "l_discount");
  a.begin = begin;
  a.end = end;
  a.segmentCode = ctx.meta("customer", "c_mktsegment").code("BUILDING");
  a.date = date;
  a.revenue = ctx.scratch<int64_t>(std::max<int64_t>(end - begin, 1));
  const std::pair<int64_t, int64_t> all{a.begin, a.end};
  auto* revenue = a.revenue;
  forBatches(
      ctx,
      all.first,
      all.second,
      [&](int64_t first, int64_t last) {
        a.begin = first;
        a.end = last;
        a.revenue = revenue + (first - all.first);
        q3Kernel<<<2048, 256, 0, ctx.stream()>>>(a);
      },
      4);
  a.revenue = revenue;
  SV_CUDA(cudaGetLastError());

  auto candidates = topCandidates(ctx, a.revenue, end - begin, 10);
  const Column& orderdate = ctx.meta("orders", "o_orderdate");
  std::sort(candidates.begin(), candidates.end(), [&](const Candidate& x, const Candidate& y) {
    if (x.value != y.value) {
      return x.value > y.value;
    }
    return orderdate.value(begin + x.index) < orderdate.value(begin + y.index);
  });
  candidates.resize(std::min<size_t>(candidates.size(), 10));
  const Column& orderkey = ctx.meta("orders", "o_orderkey");
  const Column& shippriority = ctx.meta("orders", "o_shippriority");
  Result result(kSchema);
  for (const auto& c : candidates) {
    const int64_t row = begin + c.index;
    result.ints[0].push_back(orderkey.value(row));
    result.decimals[1].push_back(c.value);
    result.ints[2].push_back(orderdate.value(row));
    result.ints[3].push_back(shippriority.value(row));
  }
  return result;
}

} // namespace

QuerySpec q3Spec() {
  return {3, q3, kSchema};
}

} // namespace facebook::velox::supervelox
