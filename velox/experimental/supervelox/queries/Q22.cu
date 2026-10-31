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

#include <algorithm>

#include "velox/experimental/supervelox/queries/Derived.h"
#include "velox/experimental/supervelox/queries/Kernels.cuh"

namespace facebook::velox::supervelox {
namespace {

constexpr int kCodes = 7;
constexpr int kCountryCodes[kCodes] = {13, 31, 23, 29, 30, 18, 17};

struct Q22Args {
  const int32_t* custkey;
  const int64_t* phoneOffsets;
  const char* phoneChars;
  const int32_t* acctbal;
  int64_t orders;
  int64_t customers;
  int8_t codeIndex[100];
  uint32_t* hasOrders;
  unsigned long long* positive;
  unsigned long long* bins;
};

__global__ void q22OrdersKernel(Q22Args a) {
  for (int64_t o = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; o < a.orders; o += (int64_t)gridDim.x * blockDim.x) {
    const int c = a.custkey[o] - 1;
    const uint32_t bit = 1u << (c & 31);
    if (!(a.hasOrders[c >> 5] & bit)) {
      atomicOr(&a.hasOrders[c >> 5], bit);
    }
  }
}

__device__ __forceinline__ int q22Code(const Q22Args& a, int64_t c) {
  const char* phone = a.phoneChars + a.phoneOffsets[c];
  const int code = (phone[0] - '0') * 10 + (phone[1] - '0');
  return code >= 0 && code < 100 ? a.codeIndex[code] : -1;
}

__global__ void q22AverageKernel(Q22Args a) {
  unsigned long long sum = 0;
  unsigned long long count = 0;
  for (int64_t c = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; c < a.customers; c += (int64_t)gridDim.x * blockDim.x) {
    const int balance = a.acctbal[c];
    if (balance > 0 && q22Code(a, c) >= 0) {
      sum += balance;
      ++count;
    }
  }
  sum = warpSum(sum);
  count = warpSum(count);
  if (threadIdx.x % 32 == 0 && count) {
    atomicAdd(&a.positive[0], sum);
    atomicAdd(&a.positive[1], count);
  }
}

__global__ void q22Kernel(Q22Args a) {
  __shared__ unsigned long long local[kCodes * 2];
  if (threadIdx.x < kCodes * 2) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  const unsigned long long sum = a.positive[0];
  const unsigned long long count = a.positive[1];
  for (int64_t c = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; c < a.customers; c += (int64_t)gridDim.x * blockDim.x) {
    const int balance = a.acctbal[c];
    // balance > avg, kept in integers.
    if (balance <= 0 || (unsigned long long)balance * count <= sum || ((a.hasOrders[c >> 5] >> (c & 31)) & 1)) {
      continue;
    }
    const int code = q22Code(a, c);
    if (code >= 0) {
      atomicAdd(&local[code * 2], 1ULL);
      atomicAdd(&local[code * 2 + 1], (unsigned long long)balance);
    }
  }
  __syncthreads();
  if (threadIdx.x < kCodes * 2 && local[threadIdx.x]) {
    atomicAdd(&a.bins[threadIdx.x], local[threadIdx.x]);
  }
}

const std::vector<ResultField> kSchema = {
    {"cntrycode", ResultType::kVarchar},
    {"numcust", ResultType::kBigint},
    {"totacctbal", ResultType::kDecimal, 38, 2},
};

Result q22(QueryContext& ctx) {
  Q22Args a;
  a.custkey = ctx.col<int32_t>("orders", "o_custkey");
  a.phoneOffsets = ctx.offsets("customer", "c_phone");
  a.phoneChars = ctx.chars("customer", "c_phone");
  a.acctbal = ctx.col<int32_t>("customer", "c_acctbal");
  a.orders = ctx.rows("orders");
  a.customers = ctx.rows("customer");
  std::fill(a.codeIndex, a.codeIndex + 100, -1);
  for (int i = 0; i < kCodes; ++i) {
    a.codeIndex[kCountryCodes[i]] = i;
  }
  a.hasOrders = ctx.zeroed<uint32_t>((a.customers + 31) / 32);
  a.positive = ctx.zeroed<unsigned long long>(2);
  a.bins = ctx.zeroed<unsigned long long>(kCodes * 2);
  q22OrdersKernel<<<4096, 256, 0, ctx.stream()>>>(a);
  q22AverageKernel<<<1024, 256, 0, ctx.stream()>>>(a);
  q22Kernel<<<1024, 256, 0, ctx.stream()>>>(a);
  SV_CUDA(cudaGetLastError());
  auto bins = ctx.download(a.bins, kCodes * 2);
  std::vector<std::pair<int, int>> codes;
  for (int i = 0; i < kCodes; ++i) {
    codes.emplace_back(kCountryCodes[i], i);
  }
  std::sort(codes.begin(), codes.end());
  Result result(kSchema);
  for (const auto& [code, index] : codes) {
    if (bins[index * 2]) {
      result.strings[0].push_back(std::to_string(code));
      result.ints[1].push_back(bins[index * 2]);
      result.decimals[2].push_back(bins[index * 2 + 1]);
    }
  }
  return result;
}

} // namespace

QuerySpec q22Spec() {
  return {22, q22, kSchema};
}

} // namespace facebook::velox::supervelox
