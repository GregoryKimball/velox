# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(sv_smoke tools/Smoke.cpp tools/SmokeKernel.cu)

target_link_libraries(sv_smoke velox_exec_test_lib velox_exec CUDA::cudart gflags::gflags)

file(GLOB SUPERVELOX_QUERIES CONFIGURE_DEPENDS queries/Q*.cu)

add_library(
  velox_supervelox
  store/ColumnStore.cpp
  store/DeviceCache.cpp
  queries/QueryContext.cu
  queries/Derived.cu
  queries/Queries.cpp
  ${SUPERVELOX_QUERIES}
  exec/FusedQuery.cpp
)

target_link_libraries(velox_supervelox velox_exec CUDA::cudart Folly::folly)
target_compile_options(
  velox_supervelox
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

add_executable(sv_bench tools/SvBench.cpp)

target_link_libraries(sv_bench velox_supervelox velox_exec_test_lib velox_exec gflags::gflags)

add_executable(sv_bandwidth tools/Bandwidth.cu)

target_link_libraries(sv_bandwidth CUDA::cudart)
