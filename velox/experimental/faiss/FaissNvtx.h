/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#pragma once

#if defined(VELOX_ENABLE_FAISS_GPU)
#include <nvtx3/nvToolsExt.h>
#endif

namespace facebook::velox::faiss {

/// Small, CPU-build-safe NVTX range used by the benchmarked execution path.
class FaissNvtxRange {
 public:
  explicit FaissNvtxRange(const char* name) {
#if defined(VELOX_ENABLE_FAISS_GPU)
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~FaissNvtxRange() {
#if defined(VELOX_ENABLE_FAISS_GPU)
    nvtxRangePop();
#endif
  }

  FaissNvtxRange(const FaissNvtxRange&) = delete;
  FaissNvtxRange& operator=(const FaissNvtxRange&) = delete;
};

} // namespace facebook::velox::faiss
