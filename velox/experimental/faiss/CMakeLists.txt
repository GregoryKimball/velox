# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

set(
  VELOX_FAISS_SOURCES
  FaissPlanNode.cpp
  FaissIndex.cpp
  FaissOperators.cpp
)

if(VELOX_ENABLE_FAISS_GPU)
  list(APPEND VELOX_FAISS_SOURCES FaissGpuIndex.cpp FaissGpuOperators.cpp)
endif()

velox_add_library(
  velox_experimental_faiss
  ${VELOX_FAISS_SOURCES}
  HEADERS
  FaissPlanNode.h
  FaissIndex.h
  FaissOperators.h
  FaissGpuIndex.h
)

velox_link_libraries(
  velox_experimental_faiss
  velox_core
  velox_exec
  velox_vector
  FAISS::faiss
)

if(VELOX_ENABLE_FAISS_GPU)
  if(VELOX_MONO_LIBRARY)
    target_compile_definitions(velox PRIVATE VELOX_ENABLE_FAISS_GPU)
    # CudfVector is otherwise kept in a separate static target. The FAISS GPU
    # operators live in the monolithic shared library and require its RTTI and
    # implementation there as well.
    target_sources(velox PRIVATE ../cudf/vector/CudfVector.cpp)
  else()
    target_compile_definitions(
      velox_experimental_faiss
      PRIVATE VELOX_ENABLE_FAISS_GPU
    )
    velox_link_libraries(velox_experimental_faiss velox_cudf_exec)
  endif()
  velox_link_libraries(
    velox_experimental_faiss
    velox_cudf_vector
    cudf::cudf
    cuvs::cuvs
    CUDA::cudart
    nvtx3::nvtx3-cpp
  )
endif()

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(
  VELOX_ENABLE_BENCHMARKS
  AND VELOX_ENABLE_PARQUET
  AND VELOX_ENABLE_HIVE_CONNECTOR
)
  add_subdirectory(benchmarks)
endif()
