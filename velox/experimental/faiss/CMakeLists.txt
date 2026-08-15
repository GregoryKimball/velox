# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

velox_add_library(
  velox_experimental_faiss
  FaissPlanNode.cpp
  FaissIndex.cpp
  FaissOperators.cpp
  HEADERS
  FaissPlanNode.h
  FaissIndex.h
  FaissOperators.h
)

velox_link_libraries(
  velox_experimental_faiss
  velox_core
  velox_exec
  velox_vector
  FAISS::faiss
)

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()
