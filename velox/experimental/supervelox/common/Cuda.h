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

#include <stdexcept>
#include <string>

namespace facebook::velox::supervelox {

inline void cudaCheck(cudaError_t error, const char* expression, const char* file, int line) {
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + ": " + expression + ": " +
        cudaGetErrorString(error));
  }
}

} // namespace facebook::velox::supervelox

#define SV_CUDA(expression) \
  ::facebook::velox::supervelox::cudaCheck((expression), #expression, __FILE__, __LINE__)
