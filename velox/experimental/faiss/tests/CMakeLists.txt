# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

add_executable(velox_experimental_faiss_test FaissTest.cpp)
add_test(NAME velox_experimental_faiss_test COMMAND velox_experimental_faiss_test)
target_link_libraries(
  velox_experimental_faiss_test
  PRIVATE
    velox_experimental_faiss
    velox_vector_test_lib
    Folly::folly
    GTest::gtest
)

if(VELOX_ENABLE_FAISS_GPU)
  target_compile_definitions(
    velox_experimental_faiss_test
    PRIVATE VELOX_ENABLE_FAISS_GPU
  )
  target_link_libraries(
    velox_experimental_faiss_test
    PRIVATE CUDA::cudart velox_cudf_exec velox_cudf_vector
  )
else()
  target_link_libraries(
    velox_experimental_faiss_test
    PRIVATE velox_exec_test_lib
  )
endif()
