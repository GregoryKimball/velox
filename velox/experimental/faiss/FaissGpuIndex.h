/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#pragma once

#include "velox/experimental/faiss/FaissIndex.h"

namespace facebook::velox::faiss {

std::shared_ptr<FaissIndexState> buildFaissGpuIndexState(
    const FaissIndexConfig& config,
    const std::map<int64_t, std::vector<float>>& vectors,
    const std::map<int64_t, std::vector<int64_t>>& documentIds);

void promoteLoadedIndexesToGpu(FaissIndexState& state);

} // namespace facebook::velox::faiss
