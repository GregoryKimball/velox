/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissGpuIndex.h"

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_factory.h>

#include <chrono>
#include <mutex>

namespace facebook::velox::faiss {
namespace {

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
      VELOX_UNREACHABLE();
  }
  VELOX_UNREACHABLE();
}

class FaissGpuContextImpl final : public FaissGpuContext {
 public:
  explicit FaissGpuContextImpl(int device)
      : device_(device),
        resources_(std::make_shared<::faiss::gpu::StandardGpuResources>()) {
    resources_->getResources()->initializeForDevice(device_);
  }

  ::faiss::gpu::StandardGpuResources* resources() {
    return resources_.get();
  }

  std::unique_ptr<::faiss::Index> toGpu(const ::faiss::Index* cpuIndex) {
    std::lock_guard<std::mutex> lock(mutex_);
    ::faiss::gpu::GpuClonerOptions options;
    options.use_cuvs = true;
    return std::unique_ptr<::faiss::Index>(::faiss::gpu::index_cpu_to_gpu(
        resources_.get(), device_, cpuIndex, &options));
  }

  void search(
      const ::faiss::Index* index,
      ::faiss::idx_t count,
      const float* queries,
      ::faiss::idx_t topK,
      float* distances,
      ::faiss::idx_t* labels,
      uintptr_t stream) override {
    std::lock_guard<std::mutex> lock(mutex_);
    resources_->setDefaultStream(
        device_, reinterpret_cast<cudaStream_t>(stream));
    index->search(count, queries, topK, distances, labels);
  }

  std::unique_ptr<::faiss::Index> toCpu(const ::faiss::Index* index) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::unique_ptr<::faiss::Index>(
        ::faiss::gpu::index_gpu_to_cpu(index));
  }

 private:
  const int device_;
  // Index operations sharing task resources are serialized because FAISS
  // mutates the resource's default stream before each operation.
  std::mutex mutex_;
  std::shared_ptr<::faiss::gpu::StandardGpuResources> resources_;
};

std::unique_ptr<::faiss::Index> buildCpuStagingIndex(
    const FaissIndexConfig& config,
    ::faiss::idx_t count,
    const float* values) {
  auto index = std::unique_ptr<::faiss::Index>(::faiss::index_factory(
      config.dimension,
      factoryDescription(config).c_str(),
      metricType(config.metric)));
  VELOX_CHECK_NOT_NULL(index);
  if (auto* ivf = dynamic_cast<::faiss::IndexIVF*>(index.get())) {
    ivf->nprobe = config.nprobe;
  }
  if (!index->is_trained && count > 0) {
    index->train(count, values);
  }
  if (count > 0) {
    index->add(count, values);
  }
  return index;
}

std::unique_ptr<::faiss::Index> buildCagraCpuSearchIndex(
    const FaissIndexConfig& config,
    ::faiss::gpu::StandardGpuResources* resources,
    ::faiss::idx_t count,
    const float* values,
    double& copyMilliseconds) {
  auto cpuIndex = std::make_unique<::faiss::IndexHNSWCagra>(
      config.dimension, config.hnswM, metricType(config.metric));
  cpuIndex->base_level_only = true;
  cpuIndex->hnsw.efConstruction = config.efConstruction;
  cpuIndex->hnsw.efSearch = config.efSearch;
  cpuIndex->num_base_level_search_entrypoints = std::max(1, config.efSearch);
  if (count == 0) {
    return cpuIndex;
  }
  VELOX_USER_CHECK_GT(
      count,
      static_cast<::faiss::idx_t>(config.hnswM) * 2,
      "CAGRA requires more vectors than its graph degree ({})",
      config.hnswM * 2);

  ::faiss::gpu::GpuIndexCagraConfig gpuConfig;
  gpuConfig.device = config.gpuDevice;
  gpuConfig.use_cuvs = true;
  gpuConfig.graph_degree = static_cast<size_t>(config.hnswM) * 2;
  gpuConfig.intermediate_graph_degree =
      std::max<size_t>(128, gpuConfig.graph_degree * 2);
  gpuConfig.build_algo = ::faiss::gpu::graph_build_algo::NN_DESCENT;
  ::faiss::gpu::GpuIndexCagra gpuIndex(
      resources, config.dimension, metricType(config.metric), gpuConfig);
  gpuIndex.train(count, values);

  const auto start = std::chrono::steady_clock::now();
  gpuIndex.copyTo(cpuIndex.get());
  resources->getResources()->syncDefaultStream(config.gpuDevice);
  copyMilliseconds += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  return cpuIndex;
}

} // namespace

std::shared_ptr<FaissIndexState> buildFaissGpuIndexState(
    const FaissIndexConfig& config,
    const std::map<int64_t, std::vector<float>>& vectors,
    const std::map<int64_t, std::vector<int64_t>>& documentIds) {
  config.validate();
  VELOX_USER_CHECK(
      config.executionDevice == FaissExecutionDevice::kGpu,
      "GPU index builder requires GPU execution");

  auto state = std::make_shared<FaissIndexState>();
  state->config = config;
  auto context = std::make_shared<FaissGpuContextImpl>(config.gpuDevice);
  state->gpuContext = context;

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
    const auto count = static_cast<::faiss::idx_t>(idIt->second.size());

    std::unique_ptr<::faiss::Index> index;
    bool gpuResident = true;
    if (config.algorithm == FaissAlgorithm::kHnswCagra) {
      index = buildCagraCpuSearchIndex(
          config,
          context->resources(),
          count,
          values.data(),
          state->cagraCopyToMilliseconds);
      gpuResident = false;
    } else {
      auto cpuIndex = buildCpuStagingIndex(config, count, values.data());
      index = context->toGpu(cpuIndex.get());
    }

    state->rowCount += count;
    state->clusters.emplace(
        cluster,
        FaissClusterIndex{std::move(index), idIt->second, gpuResident});
  }
  return state;
}

void promoteLoadedIndexesToGpu(FaissIndexState& state) {
  VELOX_USER_CHECK(state.config.executionDevice == FaissExecutionDevice::kGpu);
  VELOX_USER_CHECK(
      state.config.algorithm != FaissAlgorithm::kHnswCagra,
      "CAGRA artifacts must remain CPU HNSW indexes when loaded");
  auto context = std::make_shared<FaissGpuContextImpl>(state.config.gpuDevice);
  for (auto& [cluster, value] : state.clusters) {
    value.index = context->toGpu(value.index.get());
    value.gpuResident = true;
  }
  state.gpuContext = std::move(context);
}

} // namespace facebook::velox::faiss
