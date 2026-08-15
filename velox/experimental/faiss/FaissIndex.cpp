/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissIndex.h"

#include "folly/FileUtil.h"
#include "folly/json.h"

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace facebook::velox::faiss {
namespace {

constexpr int32_t kArtifactVersion = 1;

::faiss::MetricType metricType(FaissMetric metric) {
  return metric == FaissMetric::kL2 ? ::faiss::METRIC_L2
                                    : ::faiss::METRIC_INNER_PRODUCT;
}

std::string factoryDescription(const FaissIndexConfig& config) {
  switch (config.algorithm) {
    case FaissAlgorithm::kFlat:
      return "Flat";
    case FaissAlgorithm::kIvfFlat:
      return fmt::format("IVF{},Flat", config.nlist);
    case FaissAlgorithm::kIvfPq:
      return fmt::format(
          "IVF{},PQ{}x{}", config.nlist, config.pqSubquantizers, config.pqBits);
    case FaissAlgorithm::kHnsw:
    case FaissAlgorithm::kHnswCagra:
      // IndexHNSWCagra is the CPU representation produced by a GPU CAGRA
      // conversion. CPU-only builds construct the compatible HNSW form.
      return fmt::format("HNSW{},Flat", config.hnswM);
  }
  VELOX_UNREACHABLE();
}

std::unique_ptr<::faiss::Index> createIndex(const FaissIndexConfig& config) {
  auto index = std::unique_ptr<::faiss::Index>(::faiss::index_factory(
      config.dimension,
      factoryDescription(config).c_str(),
      metricType(config.metric)));
  VELOX_CHECK_NOT_NULL(index);
  if (auto* ivf = dynamic_cast<::faiss::IndexIVF*>(index.get())) {
    ivf->nprobe = config.nprobe;
  }
  if (auto* hnsw = dynamic_cast<::faiss::IndexHNSW*>(index.get())) {
    hnsw->hnsw.efConstruction = config.efConstruction;
    hnsw->hnsw.efSearch = config.efSearch;
  }
  return index;
}

uint64_t checksum(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  VELOX_USER_CHECK(input, "Cannot open artifact file: {}", path);
  uint64_t hash = 1469598103934665603ULL;
  char buffer[8192];
  while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
    for (std::streamsize i = 0; i < input.gcount(); ++i) {
      hash ^= static_cast<uint8_t>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::string checksumString(uint64_t value) {
  return fmt::format("{:016x}", value);
}

std::string clusterStem(int64_t cluster) {
  return fmt::format("cluster_{}", cluster);
}

void writeIds(const std::string& path, const std::vector<int64_t>& ids) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  VELOX_USER_CHECK(output, "Cannot create ID sidecar: {}", path);
  for (uint64_t value : ids) {
    for (int byte = 0; byte < 8; ++byte) {
      output.put(static_cast<char>((value >> (byte * 8)) & 0xff));
    }
  }
  VELOX_USER_CHECK(output.good(), "Failed writing ID sidecar: {}", path);
}

std::vector<int64_t> readIds(const std::string& path, size_t count) {
  std::ifstream input(path, std::ios::binary);
  VELOX_USER_CHECK(input, "Cannot open ID sidecar: {}", path);
  std::vector<int64_t> ids(count);
  for (size_t i = 0; i < count; ++i) {
    uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
      const auto ch = input.get();
      VELOX_USER_CHECK_NE(ch, EOF, "Truncated ID sidecar: {}", path);
      value |= static_cast<uint64_t>(static_cast<uint8_t>(ch)) << (byte * 8);
    }
    ids[i] = static_cast<int64_t>(value);
  }
  VELOX_USER_CHECK_EQ(input.get(), EOF, "ID sidecar has trailing bytes: {}", path);
  return ids;
}

} // namespace

std::shared_ptr<FaissIndexState> buildFaissIndexState(
    const FaissIndexConfig& config,
    const std::map<int64_t, std::vector<float>>& vectors,
    const std::map<int64_t, std::vector<int64_t>>& documentIds) {
  config.validate();
  auto state = std::make_shared<FaissIndexState>();
  state->config = config;
  for (const auto& [cluster, values] : vectors) {
    const auto idIt = documentIds.find(cluster);
    VELOX_USER_CHECK(
        idIt != documentIds.end(), "Missing document IDs for cluster {}", cluster);
    VELOX_USER_CHECK_EQ(
        values.size(),
        idIt->second.size() * static_cast<size_t>(config.dimension),
        "Vector and document-ID counts differ for cluster {}",
        cluster);
    auto index = createIndex(config);
    const auto count = static_cast<::faiss::idx_t>(idIt->second.size());
    if (!index->is_trained && count > 0) {
      index->train(count, values.data());
    }
    if (count > 0) {
      index->add(count, values.data());
    }
    state->rowCount += count;
    state->clusters.emplace(
        cluster, FaissClusterIndex{std::move(index), idIt->second});
  }
  return state;
}

void writeFaissArtifact(
    const FaissIndexState& state,
    const std::string& directory) {
  std::filesystem::create_directories(directory);
  folly::dynamic manifest = folly::dynamic::object;
  manifest["artifactVersion"] = kArtifactVersion;
  manifest["faissVersion"] = "1.14.3";
  manifest["idEncoding"] = "little-endian-int64-ordinal-map";
  manifest["config"] = state.config.serialize();
  manifest["rowCount"] = state.rowCount;
  manifest["clusters"] = folly::dynamic::array;
  for (const auto& [cluster, value] : state.clusters) {
    const auto stem = clusterStem(cluster);
    const auto indexFile = stem + ".faiss";
    const auto idsFile = stem + ".ids";
    const auto indexPath = (std::filesystem::path(directory) / indexFile).string();
    const auto idsPath = (std::filesystem::path(directory) / idsFile).string();
    ::faiss::write_index(value.index.get(), indexPath.c_str());
    writeIds(idsPath, value.documentIds);
    manifest["clusters"].push_back(
        folly::dynamic::object("cluster", cluster)(
            "rows", static_cast<int64_t>(value.documentIds.size()))(
            "indexFile", indexFile)("idsFile", idsFile)(
            "indexChecksum", checksumString(checksum(indexPath)))(
            "idsChecksum", checksumString(checksum(idsPath))));
  }
  const auto manifestPath =
      (std::filesystem::path(directory) / "manifest.json").string();
  VELOX_USER_CHECK(
      folly::writeFile(folly::toPrettyJson(manifest), manifestPath.c_str()),
      "Cannot write FAISS manifest: {}",
      manifestPath);
}

std::shared_ptr<FaissIndexState> loadFaissArtifact(
    const std::string& directory) {
  const auto manifestPath =
      (std::filesystem::path(directory) / "manifest.json").string();
  std::string contents;
  VELOX_USER_CHECK(
      folly::readFile(manifestPath.c_str(), contents),
      "Cannot read FAISS manifest: {}",
      manifestPath);
  const auto manifest = folly::parseJson(contents);
  VELOX_USER_CHECK_EQ(
      manifest["artifactVersion"].asInt(),
      kArtifactVersion,
      "Unsupported FAISS artifact version");
  VELOX_USER_CHECK_EQ(
      manifest["idEncoding"].asString(),
      "little-endian-int64-ordinal-map",
      "Unsupported FAISS document-ID encoding");

  auto state = std::make_shared<FaissIndexState>();
  state->config = FaissIndexConfig::deserialize(manifest["config"]);
  for (const auto& entry : manifest["clusters"]) {
    const auto cluster = entry["cluster"].asInt();
    const auto rows = entry["rows"].asInt();
    const auto indexPath =
        (std::filesystem::path(directory) / entry["indexFile"].asString())
            .string();
    const auto idsPath =
        (std::filesystem::path(directory) / entry["idsFile"].asString()).string();
    VELOX_USER_CHECK_EQ(
        checksumString(checksum(indexPath)),
        entry["indexChecksum"].asString(),
        "FAISS index checksum mismatch");
    VELOX_USER_CHECK_EQ(
        checksumString(checksum(idsPath)),
        entry["idsChecksum"].asString(),
        "FAISS ID sidecar checksum mismatch");
    auto index = ::faiss::read_index_up(indexPath.c_str());
    VELOX_USER_CHECK_EQ(index->d, state->config.dimension);
    VELOX_USER_CHECK_EQ(index->ntotal, rows);
    auto ids = readIds(idsPath, rows);
    state->rowCount += rows;
    state->clusters.emplace(
        cluster, FaissClusterIndex{std::move(index), std::move(ids)});
  }
  VELOX_USER_CHECK_EQ(state->rowCount, manifest["rowCount"].asInt());
  return state;
}

} // namespace facebook::velox::faiss
