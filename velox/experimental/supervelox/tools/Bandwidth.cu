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

// Measures host-to-device copy and kernel read bandwidth for pinned host memory.
// Usage: sv_bandwidth [GB]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "velox/experimental/supervelox/common/Cuda.h"

__global__ void readKernel(const uint4* data, size_t count, unsigned long long* out) {
  unsigned long long sum = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    sum += data[i].x;
  }
  if (sum == 42) {
    *out = sum;
  }
}

double seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char** argv) {
  const size_t bytes = (argc > 1 ? atof(argv[1]) : 4.0) * 1e9;
  void* host;
  void* device;
  unsigned long long* out;
  auto start = std::chrono::steady_clock::now();
  SV_CUDA(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  printf("cudaHostAlloc %.2f s\n", seconds(start));
  start = std::chrono::steady_clock::now();
  memset(host, 1, bytes);
  printf("first touch %.1f GB/s\n", bytes / seconds(start) / 1e9);
  start = std::chrono::steady_clock::now();
  SV_CUDA(cudaMalloc(&device, bytes));
  printf("cudaMalloc %.3f s\n", seconds(start));
  SV_CUDA(cudaMalloc(&out, 8));
  void* second;
  start = std::chrono::steady_clock::now();
  SV_CUDA(cudaMalloc(&second, bytes));
  SV_CUDA(cudaMemcpy(second, host, bytes, cudaMemcpyHostToDevice));
  printf("cudaMalloc + first memcpy %.3f s\n", seconds(start));
  SV_CUDA(cudaFree(second));
  for (int run = 0; run < 3; ++run) {
    start = std::chrono::steady_clock::now();
    SV_CUDA(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    printf("memcpy htod %.1f GB/s\n", bytes / seconds(start) / 1e9);
  }
  for (int run = 0; run < 3; ++run) {
    start = std::chrono::steady_clock::now();
    readKernel<<<132 * 8, 512>>>(static_cast<const uint4*>(host), bytes / 16, out);
    SV_CUDA(cudaDeviceSynchronize());
    printf("kernel read pinned %.1f GB/s\n", bytes / seconds(start) / 1e9);
  }
  start = std::chrono::steady_clock::now();
  readKernel<<<132 * 8, 512>>>(static_cast<const uint4*>(device), bytes / 16, out);
  SV_CUDA(cudaDeviceSynchronize());
  printf("kernel read hbm %.1f GB/s\n", bytes / seconds(start) / 1e9);
  return 0;
}
