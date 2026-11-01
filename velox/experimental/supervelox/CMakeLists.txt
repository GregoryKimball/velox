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
  exec/PlanMatcher.cpp
)

target_link_libraries(velox_supervelox velox_exec CUDA::cudart Folly::folly)
target_compile_options(
  velox_supervelox
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

add_executable(sv_bench tools/SvBench.cpp)

target_link_libraries(
  sv_bench
  velox_supervelox
  velox_exec_test_lib
  velox_exec
  velox_aggregates
  velox_dwio_parquet_reader
  gflags::gflags
)

set(SUPERVELOX_MELD_INCLUDE_DIR
    ""
    CACHE PATH "Directory holding meldvelox/MeldVelox.h; empty builds no Meld backend")
if(SUPERVELOX_MELD_INCLUDE_DIR)
  add_library(velox_supervelox_meld meld/MeldBackend.cpp)
  target_include_directories(velox_supervelox_meld PRIVATE ${SUPERVELOX_MELD_INCLUDE_DIR})
  target_link_libraries(velox_supervelox_meld velox_supervelox CUDA::cudart ${CMAKE_DL_LIBS})
  target_link_libraries(sv_bench velox_supervelox_meld)
  target_compile_definitions(sv_bench PRIVATE SUPERVELOX_MELD)
endif()

if(VELOX_ENABLE_CUDF)
  add_library(
    velox_supervelox_cudf
    cudf/CacheConnector.cpp
    cudf/AggregationRules.cpp
    cudf/CudfBackend.cpp
    cudf/CudfQ01.cpp
    cudf/CudfQ02.cpp
    cudf/CudfQ03.cpp
    cudf/CudfQ04.cpp
    cudf/CudfQ05.cpp
    cudf/CudfQ06.cpp
    cudf/CudfQ07.cpp
    cudf/CudfQ08.cpp
    cudf/CudfQ09.cpp
    cudf/CudfQ10.cpp
    cudf/CudfQ11.cpp
    cudf/CudfQ12.cpp
    cudf/CudfQ13.cpp
    cudf/CudfQ14.cpp
    cudf/CudfQ15.cpp
    cudf/CudfQ16.cpp
    cudf/CudfQ17.cpp
    cudf/CudfQ18.cpp
    cudf/CudfQ19.cpp
    cudf/CudfQ20.cpp
    cudf/CudfQ21.cpp
    cudf/CudfQ22.cpp
  )
  target_link_libraries(velox_supervelox_cudf velox_supervelox velox_cudf_exec cudf::cudf fmt::fmt)
  target_link_libraries(sv_bench velox_supervelox_cudf)
  target_compile_definitions(sv_bench PRIVATE SUPERVELOX_CUDF)
endif()

add_executable(sv_bandwidth tools/Bandwidth.cu)

target_link_libraries(sv_bandwidth CUDA::cudart)
