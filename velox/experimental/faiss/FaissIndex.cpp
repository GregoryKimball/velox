/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissIndex.h"
#include "velox/experimental/faiss/FaissNvtx.h"

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"

#if defined(VELOX_ENABLE_FAISS_GPU)
#include "velox/experimental/faiss/FaissGpuIndex.h"
#endif

#include "folly/json.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include <chrono>
#include <iomanip>
#include <string_view>

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
    case FaissAlgorithm::kCagra:
      VELOX_UNREACHABLE();
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

std::string artifactPath(
    const std::string& directory,
    const std::string& file) {
  return directory.ends_with('/') ? directory + file
                                    : fmt::format("{}/{}", directory, file);
}

std::string readArtifactFile(const std::string& path) {
  auto file = filesystems::getFileSystem(path, nullptr)->openFileForRead(path);
  return file->pread(0, file->size());
}

void writeArtifactFile(const std::string& path, std::string_view data) {
  auto file = filesystems::getFileSystem(path, nullptr)
                  ->openFileForWrite(
                      path,
                      {.shouldCreateParentDirectories = true,
                       .shouldThrowOnFileAlreadyExists = true});
  file->append(data);
  file->close();
}

uint64_t checksum(std::string_view data) {
  uint64_t hash = 1469598103934665603ULL;
  for (const auto byte : data) {
    hash ^= static_cast<uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string checksumString(uint64_t value) {
  return fmt::format("{:016x}", value);
}

std::string clusterStem(int64_t cluster) {
  return fmt::format("cluster_{}", cluster);
}

std::string encodeIds(const std::vector<int64_t>& ids) {
  std::string output(ids.size() * sizeof(int64_t), '\0');
  for (size_t row = 0; row < ids.size(); ++row) {
    const auto value = static_cast<uint64_t>(ids[row]);
    for (int byte = 0; byte < 8; ++byte) {
      output[row * sizeof(int64_t) + byte] =
          static_cast<char>((value >> (byte * 8)) & 0xff);
    }
  }
  return output;
}

std::vector<int64_t> decodeIds(std::string_view input, size_t count) {
  VELOX_USER_CHECK_EQ(
      input.size(),
      count * sizeof(int64_t),
      "FAISS ID sidecar size mismatch");
  std::vector<int64_t> ids(count);
  for (size_t i = 0; i < count; ++i) {
    uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
      value |= static_cast<uint64_t>(
                   static_cast<uint8_t>(input[i * sizeof(int64_t) + byte]))
          << (byte * 8);
    }
    ids[i] = static_cast<int64_t>(value);
  }
  return ids;
}

void validateArtifactIndex(
    const ::faiss::Index& index,
    FaissAlgorithm algorithm) {
  bool matches = false;
  switch (algorithm) {
    case FaissAlgorithm::kFlat:
      matches = dynamic_cast<const ::faiss::IndexFlat*>(&index) != nullptr;
      break;
    case FaissAlgorithm::kIvfFlat:
      matches = dynamic_cast<const ::faiss::IndexIVFFlat*>(&index) != nullptr;
      break;
    case FaissAlgorithm::kIvfPq:
      matches = dynamic_cast<const ::faiss::IndexIVFPQ*>(&index) != nullptr;
      break;
    case FaissAlgorithm::kCagra:
      matches =
          dynamic_cast<const ::faiss::IndexHNSWCagra*>(&index) != nullptr;
      break;
    case FaissAlgorithm::kHnsw:
    case FaissAlgorithm::kHnswCagra:
      matches = dynamic_cast<const ::faiss::IndexHNSW*>(&index) != nullptr;
      break;
  }
  VELOX_USER_CHECK(matches, "FAISS artifact index strategy mismatch");
}

} // namespace

std::shared_ptr<FaissIndexState> buildFaissIndexState(
    const FaissIndexConfig& config,
    const std::map<int64_t, std::vector<float>>& vectors,
    const std::map<int64_t, std::vector<int64_t>>& documentIds) {
  config.validate();
  if (config.executionDevice == FaissExecutionDevice::kGpu) {
#if defined(VELOX_ENABLE_FAISS_GPU)
    return buildFaissGpuIndexState(config, vectors, documentIds);
#else
    VELOX_USER_FAIL(
        "FAISS GPU execution requested, but VELOX_ENABLE_FAISS_GPU is disabled");
#endif
  }
  auto state = std::make_shared<FaissIndexState>();
  state->config = config;
  for (const auto& [cluster, values] : vectors) {
    const auto idIt = documentIds.find(cluster);
    VELOX_USER_CHECK(
        idIt != documentIds.end(),
        "Missing document IDs for cluster {}",
        cluster);
    VELOX_USER_CHECK_EQ(
        values.size(),
        idIt->second.size() * static_cast<size_t>(config.dimension),
        "Vector and document-ID counts differ for cluster {}",
        cluster);
    auto index = createIndex(config);
    const auto count = static_cast<::faiss::idx_t>(idIt->second.size());
    if (!index->is_trained && count > 0) {
      FaissNvtxRange range("FAISS train");
      const auto start = std::chrono::steady_clock::now();
      index->train(count, values.data());
      state->trainMilliseconds +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
    }
    if (count > 0) {
      FaissNvtxRange range("FAISS add");
      const auto start = std::chrono::steady_clock::now();
      index->add(count, values.data());
      state->addMilliseconds += std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
    }
    state->rowCount += count;
    state->clusters.emplace(
        cluster, FaissClusterIndex{std::move(index), idIt->second});
  }
  return state;
}

void searchFaissIndex(
    FaissIndexState& state,
    FaissClusterIndex& cluster,
    ::faiss::idx_t count,
    const float* queries,
    ::faiss::idx_t topK,
    float* distances,
    ::faiss::idx_t* labels,
    uintptr_t stream) {
  if (cluster.gpuResident) {
    VELOX_CHECK_NOT_NULL(state.gpuContext);
    FaissNvtxRange range("FAISS search");
    const auto start = std::chrono::steady_clock::now();
    state.gpuContext->search(
        cluster.index.get(), count, queries, topK, distances, labels, stream);
    state.searchMilliseconds += std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
    return;
  }
  FaissNvtxRange range("FAISS search");
  const auto start = std::chrono::steady_clock::now();
  cluster.index->search(count, queries, topK, distances, labels);
  state.searchMilliseconds +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
}

void writeFaissArtifact(
    const FaissIndexState& state,
    const std::string& directory) {
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
    const auto indexPath = artifactPath(directory, indexFile);
    const auto idsPath = artifactPath(directory, idsFile);
    std::unique_ptr<::faiss::Index> cpuIndex;
    const ::faiss::Index* persistedIndex = value.index.get();
    if (value.gpuResident) {
      VELOX_CHECK_NOT_NULL(state.gpuContext);
      cpuIndex = state.gpuContext->toCpu(value.index.get());
      persistedIndex = cpuIndex.get();
    }
    ::faiss::VectorIOWriter writer;
    writer.name = indexPath;
    ::faiss::write_index(persistedIndex, &writer);
    const std::string_view indexBytes(
        reinterpret_cast<const char*>(writer.data.data()), writer.data.size());
    writeArtifactFile(indexPath, indexBytes);
    const auto idsBytes = encodeIds(value.documentIds);
    writeArtifactFile(idsPath, idsBytes);
    manifest["clusters"].push_back(
        folly::dynamic::object("cluster", cluster)(
            "rows", static_cast<int64_t>(value.documentIds.size()))(
            "indexFile", indexFile)("idsFile", idsFile)(
            "indexChecksum", checksumString(checksum(indexBytes)))(
            "idsChecksum", checksumString(checksum(idsBytes))));
  }
  writeArtifactFile(
      artifactPath(directory, "manifest.json"),
      folly::toPrettyJson(manifest));
}

std::shared_ptr<FaissIndexState> loadFaissArtifact(
    const std::string& directory) {
  auto readStart = std::chrono::steady_clock::now();
  const auto manifestPath = artifactPath(directory, "manifest.json");
  folly::dynamic manifest;
  {
    FaissNvtxRange readRange("FAISS load read");
    manifest = folly::parseJson(readArtifactFile(manifestPath));
  }
  VELOX_USER_CHECK_EQ(
      manifest["artifactVersion"].asInt(),
      kArtifactVersion,
      "Unsupported FAISS artifact version");
  VELOX_USER_CHECK_EQ(
      manifest["faissVersion"].asString(),
      "1.14.3",
      "Unsupported FAISS library version");
  VELOX_USER_CHECK_EQ(
      manifest["idEncoding"].asString(),
      "little-endian-int64-ordinal-map",
      "Unsupported FAISS document-ID encoding");

  auto state = std::make_shared<FaissIndexState>();
  state->loadReadMilliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - readStart)
          .count();
  state->config = FaissIndexConfig::deserialize(manifest["config"]);
  // Artifacts always contain CPU FAISS indexes. Serving placement is selected
  // explicitly by applyFaissLoadTarget, never by persisted build metadata.
  state->config.executionDevice = FaissExecutionDevice::kCpu;
  for (const auto& entry : manifest["clusters"]) {
    readStart = std::chrono::steady_clock::now();
    const auto cluster = entry["cluster"].asInt();
    const auto rows = entry["rows"].asInt();
    const auto indexPath =
        artifactPath(directory, entry["indexFile"].asString());
    const auto idsPath = artifactPath(directory, entry["idsFile"].asString());
    std::string indexBytes;
    std::string idsBytes;
    {
      FaissNvtxRange readRange("FAISS load read");
      indexBytes = readArtifactFile(indexPath);
      idsBytes = readArtifactFile(idsPath);
      VELOX_USER_CHECK_EQ(
          checksumString(checksum(indexBytes)),
          entry["indexChecksum"].asString(),
          "FAISS index checksum mismatch");
      VELOX_USER_CHECK_EQ(
          checksumString(checksum(idsBytes)),
          entry["idsChecksum"].asString(),
          "FAISS ID sidecar checksum mismatch");
    }
    state->loadReadMilliseconds +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - readStart)
            .count();
    const auto deserializeStart = std::chrono::steady_clock::now();
    FaissNvtxRange deserializeRange("FAISS load deserialize");
    ::faiss::VectorIOReader reader;
    reader.name = indexPath;
    reader.data.assign(indexBytes.begin(), indexBytes.end());
    auto index =
        std::unique_ptr<::faiss::Index>(::faiss::read_index(&reader));
    VELOX_USER_CHECK_EQ(index->d, state->config.dimension);
    VELOX_USER_CHECK_EQ(
        static_cast<int>(index->metric_type),
        static_cast<int>(metricType(state->config.metric)));
    validateArtifactIndex(*index, state->config.algorithm);
    VELOX_USER_CHECK_EQ(index->ntotal, rows);
    auto ids = decodeIds(idsBytes, rows);
    state->loadDeserializeMilliseconds +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - deserializeStart)
            .count();
    state->rowCount += rows;
    state->clusters.emplace(
        cluster, FaissClusterIndex{std::move(index), std::move(ids)});
  }
  VELOX_USER_CHECK_EQ(state->rowCount, manifest["rowCount"].asInt());
  return state;
}

void applyFaissLoadTarget(
    FaissIndexState& state,
    const FaissIndexConfig& targetConfig) {
  targetConfig.validate();
  VELOX_USER_CHECK_EQ(
      state.config.dimension,
      targetConfig.dimension,
      "FAISS artifact dimension does not match target");
  VELOX_USER_CHECK(
      state.config.metric == targetConfig.metric,
      "FAISS artifact metric does not match target");
  VELOX_USER_CHECK(
      state.config.algorithm == targetConfig.algorithm,
      "FAISS artifact algorithm does not match target strategy");

  int64_t rows = 0;
  for (auto& [cluster, value] : state.clusters) {
    VELOX_USER_CHECK(
        value.index != nullptr,
        "FAISS artifact cluster {} has no index",
        cluster);
    VELOX_USER_CHECK_EQ(
        value.index->d,
        targetConfig.dimension,
        "FAISS artifact cluster {} has inconsistent dimension",
        cluster);
    VELOX_USER_CHECK_EQ(
        value.index->ntotal,
        value.documentIds.size(),
        "FAISS artifact cluster {} index and ID counts differ",
        cluster);
    rows += value.documentIds.size();
    if (auto* ivf = dynamic_cast<::faiss::IndexIVF*>(value.index.get())) {
      ivf->nprobe = targetConfig.nprobe;
    }
    if (auto* hnsw = dynamic_cast<::faiss::IndexHNSW*>(value.index.get())) {
      hnsw->hnsw.efConstruction = targetConfig.efConstruction;
      hnsw->hnsw.efSearch = targetConfig.efSearch;
    }
  }
  VELOX_USER_CHECK_EQ(
      rows,
      state.rowCount,
      "FAISS artifact row count is inconsistent with cluster indexes");

  state.config = targetConfig;
  if (targetConfig.executionDevice == FaissExecutionDevice::kGpu &&
      targetConfig.algorithm != FaissAlgorithm::kHnswCagra) {
#if defined(VELOX_ENABLE_FAISS_GPU)
    const auto uploadStart = std::chrono::steady_clock::now();
    FaissNvtxRange uploadRange("FAISS load upload");
    promoteLoadedIndexesToGpu(state);
    state.loadUploadMilliseconds +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - uploadStart)
            .count();
#else
    VELOX_USER_FAIL(
        "FAISS GPU load target requested, but VELOX_ENABLE_FAISS_GPU is "
        "disabled");
#endif
  }
}

} // namespace facebook::velox::faiss
