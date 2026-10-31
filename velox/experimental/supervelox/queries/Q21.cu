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

constexpr int kMaxLines = 8;

struct Q21Args {
  const uint32_t* starts;
  const int8_t* status;
  const int32_t* suppkey;
  const int16_t* commitdate;
  const int16_t* receiptdate;
  const int8_t* supplierNation;
  int64_t firstOrder;
  int64_t orders;
  int32_t statusCode;
  int32_t nation;
  unsigned long long* counts;
};

__global__ void q21Kernel(Q21Args a) {
  for (int64_t o = a.firstOrder + blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.orders; o += (int64_t)gridDim.x * blockDim.x) {
    if (a.status[o] != a.statusCode) {
      continue;
    }
    const uint32_t begin = a.starts[o];
    const int n = a.starts[o + 1] - begin;
    if (n > kMaxLines) {
      // TPC-H orders have at most 7 lines.
      __trap();
    }
    int supp[kMaxLines];
    bool late[kMaxLines];
    bool anyLate = false;
#pragma unroll
    for (int i = 0; i < kMaxLines; ++i) {
      if (i < n) {
        supp[i] = a.suppkey[begin + i];
        late[i] = a.receiptdate[begin + i] > a.commitdate[begin + i];
        anyLate |= late[i];
      }
    }
    if (!anyLate) {
      continue;
    }
#pragma unroll
    for (int i = 0; i < kMaxLines; ++i) {
      if (i >= n || !late[i]) {
        continue;
      }
      bool other = false;
      bool otherLate = false;
#pragma unroll
      for (int j = 0; j < kMaxLines; ++j) {
        if (j < n && supp[j] != supp[i]) {
          other = true;
          otherLate |= late[j];
        }
      }
      if (other && !otherLate && a.supplierNation[supp[i] - 1] == a.nation) {
        atomicAdd(&a.counts[supp[i] - 1], 1ULL);
      }
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"s_name", ResultType::kVarchar},
    {"numwait", ResultType::kBigint},
};

Result q21(QueryContext& ctx) {
  Q21Args a;
  a.starts = orderLineStarts(ctx);
  a.status = ctx.col<int8_t>("orders", "o_orderstatus");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.commitdate = ctx.col<int16_t>("lineitem", "l_commitdate");
  a.receiptdate = ctx.col<int16_t>("lineitem", "l_receiptdate");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.statusCode = ctx.meta("orders", "o_orderstatus").code("F");
  a.nation = nationKey(ctx, "SAUDI ARABIA");
  const int64_t suppliers = ctx.rows("supplier");
  a.counts = ctx.zeroed<unsigned long long>(suppliers);
  forBatches(
      ctx,
      0,
      ctx.rows("orders"),
      [&](int64_t first, int64_t last) {
        a.firstOrder = first;
        a.orders = last;
        q21Kernel<<<4096, 256, 0, ctx.stream()>>>(a);
      },
      4);
  SV_CUDA(cudaGetLastError());
  auto candidates = topCandidates(ctx, reinterpret_cast<const int64_t*>(a.counts), suppliers, 100);
  const Column& name = ctx.meta("supplier", "s_name");
  std::vector<std::pair<Candidate, std::string>> rows;
  for (const auto& c : candidates) {
    rows.emplace_back(c, name.string(c.index));
  }
  std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) {
    return x.first.value != y.first.value ? x.first.value > y.first.value : x.second < y.second;
  });
  rows.resize(std::min<size_t>(rows.size(), 100));
  Result result(kSchema);
  for (const auto& [c, n] : rows) {
    result.strings[0].push_back(n);
    result.ints[1].push_back(c.value);
  }
  return result;
}

} // namespace

QuerySpec q21Spec() {
  return {21, q21, kSchema};
}

} // namespace facebook::velox::supervelox
