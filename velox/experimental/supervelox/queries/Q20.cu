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

constexpr int kThreads = 256;

struct Q20Args {
  const int16_t* shipdate;
  const int32_t* partkey;
  const int32_t* suppkey;
  const int16_t* quantity;
  const int32_t* partsuppSuppkey;
  const int16_t* availqty;
  const int8_t* supplierNation;
  const int2* zones;
  BitmapRanks parts;
  int64_t rows;
  int64_t firstZone;
  int64_t lastZone;
  int64_t partCount;
  int32_t dateLow;
  int32_t dateHigh;
  int32_t nation;
  unsigned* quantitySum;
  uint32_t* suppliers;
  unsigned long long* missing;
};

__global__ void __launch_bounds__(kThreads) q20LineKernel(Q20Args a) {
  constexpr int64_t kZone = QueryContext::kZoneRows;
  for (int64_t zone = a.firstZone + blockIdx.x; zone < a.lastZone; zone += gridDim.x) {
    const int2 range = a.zones[zone];
    if (range.y < a.dateLow || range.x >= a.dateHigh) {
      continue;
    }
    const int64_t end = min(a.rows, (zone + 1) * kZone);
    for (int64_t row = zone * kZone + threadIdx.x; row < end; row += kThreads) {
      const int date = a.shipdate[row];
      if (date < a.dateLow || date >= a.dateHigh) {
        continue;
      }
      const int part = a.partkey[row] - 1;
      if (!testBit(a.parts.bits, part)) {
        continue;
      }
      const int supplier = a.suppkey[row];
      const int64_t base = (int64_t)part * 4;
      int slot = 0;
      while (slot < 4 && a.partsuppSuppkey[base + slot] != supplier) {
        ++slot;
      }
      if (slot == 4) {
        atomicAdd(a.missing, 1ULL);
        continue;
      }
      atomicAdd(&a.quantitySum[bitRank(a.parts, part) * 4 + slot], (unsigned)a.quantity[row]);
    }
  }
}

__global__ void q20PartKernel(Q20Args a) {
  for (int64_t part = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; part < a.partCount;
       part += (int64_t)gridDim.x * blockDim.x) {
    if (!testBit(a.parts.bits, part)) {
      continue;
    }
    const int64_t rank = bitRank(a.parts, part);
    for (int slot = 0; slot < 4; ++slot) {
      const unsigned sum = a.quantitySum[rank * 4 + slot];
      const int supplier = a.partsuppSuppkey[part * 4 + slot] - 1;
      // availqty > 0.5 * sum(l_quantity); quantities carry two decimals. No lines means a NULL sum.
      if (sum > 0 && 200ULL * a.availqty[part * 4 + slot] > sum && a.supplierNation[supplier] == a.nation) {
        atomicOr(&a.suppliers[supplier >> 5], 1u << (supplier & 31));
      }
    }
  }
}

const std::vector<ResultField> kSchema = {
    {"s_name", ResultType::kVarchar},
    {"s_address", ResultType::kVarchar},
};

Result q20(QueryContext& ctx) {
  Q20Args a;
  a.partCount = ctx.rows("part");
  a.parts = rankBitmap(ctx, likeBitmap(ctx, "part", "p_name", Match::kPrefix, "forest"), a.partCount);
  a.shipdate = ctx.col<int16_t>("lineitem", "l_shipdate");
  a.partkey = ctx.col<int32_t>("lineitem", "l_partkey");
  a.suppkey = ctx.col<int32_t>("lineitem", "l_suppkey");
  a.quantity = ctx.col<int16_t>("lineitem", "l_quantity");
  a.partsuppSuppkey = ctx.col<int32_t>("partsupp", "ps_suppkey");
  a.availqty = ctx.col<int16_t>("partsupp", "ps_availqty");
  a.supplierNation = ctx.col<int8_t>("supplier", "s_nationkey");
  a.zones = ctx.zones("lineitem", "l_shipdate");
  a.rows = ctx.rows("lineitem");
  a.dateLow = days(1994, 1, 1);
  a.dateHigh = days(1995, 1, 1);
  a.nation = nationKey(ctx, "CANADA");
  a.quantitySum = ctx.zeroed<unsigned>(a.parts.count * 4);
  const int64_t supplierCount = ctx.rows("supplier");
  const int64_t supplierWords = (supplierCount + 31) / 32;
  a.suppliers = ctx.zeroed<uint32_t>(supplierWords);
  a.missing = ctx.zeroed<unsigned long long>(1);
  constexpr int64_t kZone = QueryContext::kZoneRows;
  forBatches(
      ctx,
      0,
      (a.rows + kZone - 1) / kZone,
      [&](int64_t first, int64_t last) {
        a.firstZone = first;
        a.lastZone = last;
        q20LineKernel<<<persistentBlocks(q20LineKernel, kThreads), kThreads, 0, ctx.stream()>>>(a);
      },
      kZone);
  SV_CUDA(cudaGetLastError());
  q20PartKernel<<<2048, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  if (ctx.download(a.missing, 1)[0]) {
    throw std::runtime_error("q20: lineitem (part, supplier) missing from partsupp layout");
  }
  auto suppliers = ctx.download(a.suppliers, supplierWords);
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& address = ctx.meta("supplier", "s_address");
  std::vector<std::pair<std::string, int64_t>> rows;
  for (int64_t word = 0; word < supplierWords; ++word) {
    for (uint32_t bits = suppliers[word]; bits; bits &= bits - 1) {
      const int64_t row = word * 32 + __builtin_ctz(bits);
      rows.emplace_back(name.string(row), row);
    }
  }
  if (!std::is_sorted(rows.begin(), rows.end())) {
    std::sort(rows.begin(), rows.end());
  }
  Result result(kSchema);
  for (auto& [supplierName, row] : rows) {
    result.strings[0].push_back(std::move(supplierName));
    result.strings[1].push_back(address.string(row));
  }
  return result;
}

} // namespace

QuerySpec q20Spec() {
  return {20, q20, kSchema};
}

} // namespace facebook::velox::supervelox
