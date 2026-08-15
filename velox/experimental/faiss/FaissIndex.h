/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#pragma once

#include "velox/experimental/faiss/FaissPlanNode.h"

#include <faiss/Index.h>

#include <map>
#include <memory>

namespace facebook::velox::faiss {

class FaissGpuContext {
 public:
  virtual ~FaissGpuContext() = default;

  virtual void search(
      const ::faiss::Index* index,
      ::faiss::idx_t count,
      const float* queries,
      ::faiss::idx_t topK,
      float* distances,
      ::faiss::idx_t* labels,
      uintptr_t stream = 0) = 0;

  virtual std::unique_ptr<::faiss::Index> toCpu(
      const ::faiss::Index* index) = 0;
};

struct FaissClusterIndex {
  std::unique_ptr<::faiss::Index> index;
  std::vector<int64_t> documentIds;
  bool gpuResident{false};
};

struct FaissIndexState {
  FaissIndexConfig config;
  // Declared before clusters so GPU indexes are destroyed before their
  // task-scoped resources.
  std::shared_ptr<FaissGpuContext> gpuContext;
  std::map<int64_t, FaissClusterIndex> clusters;
  int64_t rowCount{0};
  double cagraCopyToMilliseconds{0};
  double trainMilliseconds{0};
  double addMilliseconds{0};
  double searchMilliseconds{0};
  double loadReadMilliseconds{0};
  double loadDeserializeMilliseconds{0};
  double loadUploadMilliseconds{0};
};

void searchFaissIndex(
    FaissIndexState& state,
    FaissClusterIndex& cluster,
    ::faiss::idx_t count,
    const float* queries,
    ::faiss::idx_t topK,
    float* distances,
    ::faiss::idx_t* labels,
    uintptr_t stream = 0);

std::shared_ptr<FaissIndexState> buildFaissIndexState(
    const FaissIndexConfig& config,
    const std::map<int64_t, std::vector<float>>& vectors,
    const std::map<int64_t, std::vector<int64_t>>& documentIds);

// Artifact package:
//   manifest.json - versioned metadata and checksums
//   cluster_<id>.faiss - native FAISS index
//   cluster_<id>.ids - little-endian signed BIGINT ordinal-to-document-ID map
void writeFaissArtifact(
    const FaissIndexState& state,
    const std::string& directory);

std::shared_ptr<FaissIndexState> loadFaissArtifact(
    const std::string& directory);

/// Validates a CPU-loaded artifact against the requested serving config,
/// applies search parameters and uploads indexes when GPU serving is requested.
void applyFaissLoadTarget(
    FaissIndexState& state,
    const FaissIndexConfig& targetConfig);

} // namespace facebook::velox::faiss
