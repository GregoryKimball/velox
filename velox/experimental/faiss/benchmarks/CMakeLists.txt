# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

add_executable(
  velox_candidate_retrieval_benchmark
  CandidateRetrievalBenchmark.cpp
)

target_link_libraries(
  velox_candidate_retrieval_benchmark
  PRIVATE
    velox_experimental_faiss
    velox_exec_test_lib
    velox_dwio_parquet_reader
    velox_dwio_parquet_writer
    velox_functions_prestosql
    Folly::folly
    GTest::gtest
    gflags::gflags
)

if(VELOX_ENABLE_FAISS_GPU)
  target_compile_definitions(
    velox_candidate_retrieval_benchmark
    PRIVATE VELOX_ENABLE_FAISS_GPU
  )
  target_link_libraries(
    velox_candidate_retrieval_benchmark
    PRIVATE velox_cudf_exec nvtx3::nvtx3-c
  )
endif()
