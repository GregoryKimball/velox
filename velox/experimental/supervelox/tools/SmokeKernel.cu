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

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void sumKernel(int64_t const* values, int64_t n, unsigned long long* out) {
  unsigned long long local = 0;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    local += values[i];
  }
  atomicAdd(out, local);
}

} // namespace

int64_t smokeDeviceSum(int64_t n) {
  int64_t* values;
  unsigned long long* out;
  cudaMallocManaged(&values, n * sizeof(int64_t));
  cudaMallocManaged(&out, sizeof(unsigned long long));
  for (int64_t i = 0; i < n; ++i) {
    values[i] = i;
  }
  *out = 0;
  sumKernel<<<132, 256>>>(values, n, out);
  cudaDeviceSynchronize();
  auto result = static_cast<int64_t>(*out);
  cudaFree(values);
  cudaFree(out);
  return result;
}
