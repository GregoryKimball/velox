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
  int64_t lineBase;
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
      for (int64_t line = a.lineBase + a.starts[o]; line < a.lineBase + a.starts[o + 1]; ++line) {
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
  a.segment = ctx.col<int8_t>("customer", "c_mktsegment");
  a.segmentCode = ctx.info("customer", "c_mktsegment").code("BUILDING");
  a.date = date;
  std::vector<Candidate> candidates;
  forBatches(
      ctx,
      begin,
      end,
      [&](int64_t first, int64_t last) {
        const LineRange lines = lineRange(ctx, first, last);
        a.starts = lines.starts;
        a.lineBase = lines.base;
        a.custkey = ctx.rows<int32_t>("orders", "o_custkey", first, last);
        a.shipdate = ctx.rows<int16_t>("lineitem", "l_shipdate", lines.firstLine, lines.lastLine);
        a.price = ctx.rows<int32_t>("lineitem", "l_extendedprice", lines.firstLine, lines.lastLine);
        a.discount = ctx.rows<int8_t>("lineitem", "l_discount", lines.firstLine, lines.lastLine);
        a.begin = first;
        a.end = last;
        a.revenue = ctx.batchScratch<int64_t>(std::max<int64_t>(last - first, 1));
        q3Kernel<<<2048, 256, 0, ctx.stream()>>>(a);
        SV_CUDA(cudaGetLastError());
        for (Candidate c : topCandidates(ctx, a.revenue, last - first, 10)) {
          c.index += first;
          candidates.push_back(c);
        }
      },
      4);

  auto orderdate = [&](int64_t row) { return ctx.value("orders", "o_orderdate", row); };
  std::sort(candidates.begin(), candidates.end(), [&](const Candidate& x, const Candidate& y) {
    if (x.value != y.value) {
      return x.value > y.value;
    }
    return orderdate(x.index) < orderdate(y.index);
  });
  candidates.resize(std::min<size_t>(candidates.size(), 10));
  Result result(kSchema);
  for (const auto& c : candidates) {
    const int64_t row = c.index;
    result.ints[0].push_back(ctx.value("orders", "o_orderkey", row));
    result.decimals[1].push_back(c.value);
    result.ints[2].push_back(orderdate(row));
    result.ints[3].push_back(ctx.value("orders", "o_shippriority", row));
  }
  return result;
}

} // namespace

QuerySpec q3Spec() {
  return {3, q3, kSchema};
}

} // namespace facebook::velox::supervelox
