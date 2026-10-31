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
#include <climits>
#include <tuple>

#include "velox/experimental/supervelox/queries/Derived.h"
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

/// Makes account balances (two decimals, above -1000) positive for topCandidates.
constexpr int64_t kBalanceOffset = 100001;

struct Q2Args {
  const int8_t* size;
  const int16_t* type;
  const int32_t* suppkey;
  const int32_t* supplycost;
  const int8_t* supplierNation;
  const int32_t* acctbal;
  int64_t parts;
  int32_t sizeValue;
  CodeSet types;
  uint32_t nations;
  int64_t* balances;
  int64_t* partsuppRows;
  unsigned long long* count;
};

template <bool kEmit>
__global__ void q2Kernel(Q2Args a) {
  for (int64_t part = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; part < a.parts; part += (int64_t)gridDim.x * blockDim.x) {
    if (a.size[part] != a.sizeValue || !a.types.contains(a.type[part])) {
      continue;
    }
    int cost[4];
    bool inRegion[4];
    int best = INT_MAX;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      const int64_t row = part * 4 + slot;
      inRegion[slot] = (a.nations >> a.supplierNation[a.suppkey[row] - 1]) & 1;
      cost[slot] = a.supplycost[row];
      if (inRegion[slot]) {
        best = min(best, cost[slot]);
      }
    }
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (inRegion[slot] && cost[slot] == best) {
        const int64_t row = part * 4 + slot;
        const unsigned long long index = atomicAdd(a.count, 1ULL);
        if constexpr (kEmit) {
          a.balances[index] = a.acctbal[a.suppkey[row] - 1] + kBalanceOffset;
          a.partsuppRows[index] = row;
        }
      }
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"s_acctbal", ResultType::kDecimal, 15, 2},
    {"s_name", ResultType::kVarchar},
    {"n_name", ResultType::kVarchar},
    {"p_partkey", ResultType::kBigint},
    {"p_mfgr", ResultType::kVarchar},
    {"s_address", ResultType::kVarchar},
    {"s_phone", ResultType::kVarchar},
    {"s_comment", ResultType::kVarchar},
};

Result q2(QueryContext& ctx) {
  Q2Args a;
  a.parts = ctx.rows("part");
  a.size = ctx.col<int8_t>("part", "p_size");
  a.type = ctx.col<int16_t>("part", "p_type");
  a.suppkey = ctx.col<int32_t>("partsupp", "ps_suppkey");
  a.supplycost = ctx.col<int32_t>("partsupp", "ps_supplycost");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.acctbal = ctx.col<int32_t>("supplier", "s_acctbal");
  a.sizeValue = 15;
  a.types = dictionaryCodes(ctx, "part", "p_type", Match::kSuffix, "BRASS");
  a.nations = nationsInRegion(ctx, "EUROPE");
  a.count = ctx.zeroed<unsigned long long>(1);
  q2Kernel<false><<<2048, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  const int64_t count = ctx.download(a.count, 1)[0];
  a.balances = ctx.scratch<int64_t>(count);
  a.partsuppRows = ctx.scratch<int64_t>(count);
  SV_CUDA(cudaMemsetAsync(a.count, 0, sizeof(unsigned long long), ctx.stream()));
  q2Kernel<true><<<2048, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  auto candidates = topCandidates(ctx, a.balances, count, 100);
  const auto partsuppRows = ctx.download(a.partsuppRows, count);
  std::vector<int64_t> rows;
  for (const auto& c : candidates) {
    rows.push_back(partsuppRows[c.index]);
  }

  const Column& suppkey = ctx.meta("partsupp", "ps_suppkey");
  const Column& partkey = ctx.meta("partsupp", "ps_partkey");
  const Column& acctbal = ctx.meta("supplier", "s_acctbal");
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& nation = ctx.meta("supplier", "s_nationkey");
  struct Row {
    int64_t acctbal;
    std::string nation;
    std::string name;
    int64_t partkey;
    int64_t supplier;
  };
  std::vector<Row> sorted;
  for (int64_t row : rows) {
    const int64_t supplier = suppkey.value(row) - 1;
    sorted.push_back(
        {acctbal.value(supplier), nationName(ctx, nation.value(supplier)), name.string(supplier), partkey.value(row), supplier});
  }
  std::sort(sorted.begin(), sorted.end(), [](const Row& x, const Row& y) {
    return std::make_tuple(-x.acctbal, std::cref(x.nation), std::cref(x.name), x.partkey) <
        std::make_tuple(-y.acctbal, std::cref(y.nation), std::cref(y.name), y.partkey);
  });
  sorted.resize(std::min<size_t>(sorted.size(), 100));
  const Column& mfgr = ctx.meta("part", "p_mfgr");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  const Column& comment = ctx.meta("supplier", "s_comment");
  Result result(kSchema);
  for (const auto& row : sorted) {
    result.decimals[0].push_back(row.acctbal);
    result.strings[1].push_back(row.name);
    result.strings[2].push_back(row.nation);
    result.ints[3].push_back(row.partkey);
    result.strings[4].push_back(mfgr.string(row.partkey - 1));
    result.strings[5].push_back(address.string(row.supplier));
    result.strings[6].push_back(phone.string(row.supplier));
    result.strings[7].push_back(comment.string(row.supplier));
  }
  return result;
}

} // namespace

QuerySpec q2Spec() {
  return {2, q2, kSchema};
}

} // namespace facebook::velox::supervelox
