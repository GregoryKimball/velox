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
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "velox/experimental/supervelox/common/Cuda.h"
#include "velox/experimental/supervelox/queries/Query.h"

namespace facebook::velox::supervelox {

/// Days since 1970-01-01 for a proleptic Gregorian date.
constexpr int32_t days(int32_t year, int32_t month, int32_t day) {
  year -= month <= 2;
  const int32_t era = (year >= 0 ? year : year - 399) / 400;
  const int32_t yoe = year - era * 400;
  const int32_t doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}

/// Year of a day count since 1970-01-01.
__host__ __device__ constexpr int32_t yearOf(int32_t z) {
  z += 719468;
  const int32_t era = (z >= 0 ? z : z - 146096) / 146097;
  const int32_t doe = z - era * 146097;
  const int32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int32_t mp = (5 * doy + 2) / 153;
  const int32_t month = mp + (mp < 10 ? 3 : -9);
  return yoe + era * 400 + (month <= 2);
}

/// Loads N consecutive values starting at 'p', which must be aligned to N * sizeof(T).
template <int N, typename T>
__device__ __forceinline__ void loadVector(const T* p, T (&out)[N]) {
  constexpr int kBytes = N * sizeof(T);
  static_assert(kBytes % 4 == 0 || kBytes < 4);
  if constexpr (kBytes % 16 == 0) {
    const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int i = 0; i < kBytes / 16; ++i) {
      reinterpret_cast<uint4*>(out)[i] = __ldg(q + i);
    }
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint2*>(out) = __ldg(reinterpret_cast<const uint2*>(p));
  } else if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t*>(out) = __ldg(reinterpret_cast<const unsigned int*>(p));
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out[i] = p[i];
    }
  }
}

/// Adds a non-negative 64-bit value to a 128-bit counter stored as {low, high}.
__device__ __forceinline__ void atomicAdd128(unsigned long long* counter, unsigned long long value) {
  unsigned long long old = atomicAdd(counter, value);
  if (old + value < old) {
    atomicAdd(counter + 1, 1ULL);
  }
}

/// Adds a signed 64-bit value to a 128-bit two's complement counter {low, high}.
__device__ __forceinline__ void atomicAddSigned128(unsigned long long* counter, long long value) {
  unsigned long long v = static_cast<unsigned long long>(value);
  unsigned long long old = atomicAdd(counter, v);
  unsigned long long carry = (old + v < old) ? 1ULL : 0ULL;
  unsigned long long high = carry + (value < 0 ? ~0ULL : 0ULL);
  if (high) {
    atomicAdd(counter + 1, high);
  }
}

template <typename T>
__device__ __forceinline__ T warpSum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

inline __int128 toInt128(const unsigned long long* counter) {
  return static_cast<__int128>((static_cast<unsigned __int128>(counter[1]) << 64) | counter[0]);
}

/// Number of resident CTAs of 'threads' threads for a persistent kernel.
template <typename Kernel>
int persistentBlocks(Kernel kernel, int threads, size_t sharedBytes = 0) {
  static int smCount = [] {
    int device = 0;
    int count = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    return count;
  }();
  int perSm = 0;
  SV_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, threads, sharedBytes));
  return smCount * (perSm > 0 ? perSm : 1);
}

} // namespace facebook::velox::supervelox
