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
include_guard(GLOBAL)

set(VELOX_FAISS_BUILD_VERSION 1.14.3)
set(
  VELOX_FAISS_BUILD_SHA256_CHECKSUM
  7f3c4ed9aec3bd7524382862f5fcbd4d8984e2a8979ff3bdb2c0bcea5144149e
)
set(
  VELOX_FAISS_SOURCE_URL
  "https://github.com/facebookresearch/faiss/archive/refs/tags/v${VELOX_FAISS_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FAISS)

# We need these hints for macos to build.
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  message(STATUS "Detected Apple platform")
  execute_process(
    COMMAND brew --prefix libomp
    RESULT_VARIABLE BREW_LIBOMP_RESULT
    OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_LIBOMP_RESULT EQUAL 0 AND EXISTS "${BREW_LIBOMP_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_LIBOMP_PREFIX}")
  endif()

  execute_process(
    COMMAND brew --prefix openblas
    RESULT_VARIABLE BREW_OPENBLAS_RESULT
    OUTPUT_VARIABLE BREW_OPENBLAS_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_OPENBLAS_RESULT EQUAL 0 AND EXISTS "${BREW_OPENBLAS_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_OPENBLAS_PREFIX}")
  endif()
endif()

FetchContent_Declare(
  faiss
  URL ${VELOX_FAISS_SOURCE_URL}
  URL_HASH ${VELOX_FAISS_BUILD_SHA256_CHECKSUM}
  SYSTEM
  EXCLUDE_FROM_ALL
  PATCH_COMMAND
    patch -p1 -i ${CMAKE_CURRENT_LIST_DIR}/faiss/faiss-1.14-cuvs-26.08.patch
)

# Set build options
block()
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(CMAKE_BUILD_TYPE Release)
  set(FAISS_ENABLE_GPU ${VELOX_ENABLE_FAISS_GPU})
  set(FAISS_ENABLE_CUVS ${VELOX_ENABLE_FAISS_GPU})
  set(FAISS_USE_CUDA_TOOLKIT_STATIC ON)
  set(FAISS_ENABLE_PYTHON OFF)
  set(FAISS_ENABLE_GPU_TESTS OFF)
  # Make FAISS available
  FetchContent_MakeAvailable(faiss)
  add_library(FAISS::faiss ALIAS faiss)
  unset(BUILD_TESTING CACHE)
  unset(BUILD_SHARED_LIBS CACHE)
endblock()
