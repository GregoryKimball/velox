/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissGpuIndex.h"
#include "velox/experimental/faiss/FaissNvtx.h"
#include "velox/experimental/faiss/FaissOperators.h"

#include "folly/ScopeGuard.h"

#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_factory.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <mutex>

namespace facebook::velox::faiss {
namespace {

::faiss::MetricType metricType(FaissMetric metric) {
  return metric == FaissMetric::kL2 ? ::faiss::METRIC_L2
                                    : ::faiss::METRIC_INNER_PRODUCT;
}

void checkCuda(cudaError_t error, const char* operation) {
  VELOX_USER_CHECK_EQ(
      static_cast<int>(error),
      static_cast<int>(cudaSuccess),
      "{} failed: {}",
      operation,
      cudaGetErrorString(error));
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

  void setDefaultStream(uintptr_t stream) {
    resources_->setDefaultStream(
        device_, reinterpret_cast<cudaStream_t>(stream));
  }

  void synchronize() {
    resources_->getResources()->syncDefaultStream(device_);
  }

  std::unique_ptr<::faiss::Index> toGpu(const ::faiss::Index* cpuIndex) {
    std::lock_guard<std::mutex> lock(mutex_);
    ::faiss::gpu::GpuClonerOptions options;
    options.use_cuvs = true;
    return std::unique_ptr<::faiss::Index>(::faiss::gpu::index_cpu_to_gpu(
        resources_.get(), device_, cpuIndex, &options));
  }

  std::unique_ptr<::faiss::Index> toGpuCagra(
      const ::faiss::IndexHNSWCagra* cpuIndex,
      const FaissIndexConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    ::faiss::gpu::GpuIndexCagraConfig gpuConfig;
    gpuConfig.device = device_;
    gpuConfig.use_cuvs = true;
    gpuConfig.graph_degree = static_cast<size_t>(config.hnswM) * 2;
    gpuConfig.intermediate_graph_degree =
        std::max<size_t>(128, gpuConfig.graph_degree * 2);
    auto gpuIndex = std::make_unique<::faiss::gpu::GpuIndexCagra>(
        resources_.get(),
        config.dimension,
        metricType(config.metric),
        gpuConfig);
    gpuIndex->copyFrom(cpuIndex);
    return gpuIndex;
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
      if (dynamic_cast<const ::faiss::gpu::GpuIndexCagra*>(index)) {
        ::faiss::gpu::SearchParametersCagra search;
        search.itopk_size =
            std::bit_ceil(std::max<size_t>(64, static_cast<size_t>(topK)));
        index->search(count, queries, topK, distances, labels, &search);
      } else {
        index->search(count, queries, topK, distances, labels);
      }
      // Search outputs are host buffers and are consumed immediately by the
      // operator. Do not let a subsequent call reuse them while GPU work is
      // still pending on the operator stream.
      resources_->getResources()->syncDefaultStream(device_);
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

std::unique_ptr<::faiss::Index> createCpuTemplateIndex(
    const FaissIndexConfig& config) {
  auto index = std::unique_ptr<::faiss::Index>(::faiss::index_factory(
      config.dimension,
      factoryDescription(config).c_str(),
      metricType(config.metric)));
  VELOX_CHECK_NOT_NULL(index);
  if (auto* ivf = dynamic_cast<::faiss::IndexIVF*>(index.get())) {
    ivf->nprobe = config.nprobe;
  }
  return index;
}

std::unique_ptr<::faiss::Index> buildCagraCpuSearchIndex(
    const FaissIndexConfig& config,
    ::faiss::gpu::StandardGpuResources* resources,
    ::faiss::idx_t count,
    const float* values,
    double& copyMilliseconds,
    double& trainMilliseconds) {
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
  FaissNvtxRange trainRange("FAISS train");
  const auto trainStart = std::chrono::steady_clock::now();
  gpuIndex.train(count, values);
  trainMilliseconds += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - trainStart)
                           .count();

  const auto start = std::chrono::steady_clock::now();
  FaissNvtxRange copyRange("CAGRA-to-HNSW");
  gpuIndex.copyTo(cpuIndex.get());
  resources->getResources()->syncDefaultStream(config.gpuDevice);
  copyMilliseconds += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  return cpuIndex;
}

void addGpuCluster(
    FaissIndexState& state,
    const FaissIndexConfig& config,
    FaissGpuContextImpl& context,
    int64_t cluster,
    const float* values,
    const std::vector<int64_t>& documentIds) {
  const auto count = static_cast<::faiss::idx_t>(documentIds.size());
  std::unique_ptr<::faiss::Index> index;
  bool gpuResident = true;
  if (config.algorithm == FaissAlgorithm::kHnswCagra) {
    index = buildCagraCpuSearchIndex(
        config,
        context.resources(),
        count,
        values,
        state.cagraCopyToMilliseconds,
        state.trainMilliseconds);
    gpuResident = false;
  } else if (config.algorithm == FaissAlgorithm::kCagra) {
    ::faiss::gpu::GpuIndexCagraConfig gpuConfig;
    gpuConfig.device = config.gpuDevice;
    gpuConfig.use_cuvs = true;
    gpuConfig.graph_degree = static_cast<size_t>(config.hnswM) * 2;
    gpuConfig.intermediate_graph_degree =
        std::max<size_t>(128, gpuConfig.graph_degree * 2);
    gpuConfig.build_algo = ::faiss::gpu::graph_build_algo::NN_DESCENT;
    auto gpuIndex = std::make_unique<::faiss::gpu::GpuIndexCagra>(
        context.resources(),
        config.dimension,
        metricType(config.metric),
        gpuConfig);
    {
      FaissNvtxRange range("FAISS train");
      const auto start = std::chrono::steady_clock::now();
      gpuIndex->train(count, values);
      state.trainMilliseconds +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
    }
    index = std::move(gpuIndex);
  } else {
    auto cpuIndex = createCpuTemplateIndex(config);
    index = context.toGpu(cpuIndex.get());
    if (!index->is_trained && count > 0) {
      FaissNvtxRange range("FAISS train");
      const auto start = std::chrono::steady_clock::now();
      index->train(count, values);
      state.trainMilliseconds +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
    }
    if (count > 0) {
      FaissNvtxRange range("FAISS add");
      const auto start = std::chrono::steady_clock::now();
      index->add(count, values);
      state.addMilliseconds +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
    }
  }
  context.synchronize();
  state.rowCount += count;
  state.clusters.emplace(
      cluster,
      FaissClusterIndex{std::move(index), documentIds, gpuResident});
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
    addGpuCluster(
        *state, config, *context, cluster, values.data(), idIt->second);
  }
  context->synchronize();
  return state;
}

std::shared_ptr<FaissIndexState> buildFaissGpuIndexState(
    const FaissIndexConfig& config,
    const std::vector<FaissGpuBuildInput>& inputs) {
  config.validate();
  VELOX_USER_CHECK(
      config.executionDevice == FaissExecutionDevice::kGpu,
      "GPU index builder requires GPU execution");
  checkCuda(cudaSetDevice(config.gpuDevice), "selecting FAISS GPU device");
  for (const auto& input : inputs) {
    VELOX_USER_CHECK_EQ(input.documentIds.size(), input.rowCount);
    if (input.clusters) {
      VELOX_USER_CHECK_EQ(input.clusters->size(), input.rowCount);
    }
  }

  auto state = std::make_shared<FaissIndexState>();
  state->config = config;
  auto context = std::make_shared<FaissGpuContextImpl>(config.gpuDevice);
  state->gpuContext = context;
  if (!inputs.empty()) {
    context->setDefaultStream(inputs.front().stream);
  }
  if (inputs.size() == 1) {
    const auto& input = inputs.front();
    const auto cluster =
        input.clusters && !input.clusters->empty() ? input.clusters->front() : 0;
    const auto singleCluster =
        !input.clusters ||
        std::all_of(
            input.clusters->begin(),
            input.clusters->end(),
            [&](int64_t value) { return value == cluster; });
    if (singleCluster) {
      addGpuCluster(
          *state,
          config,
          *context,
          cluster,
          input.embeddings,
          input.documentIds);
          if (config.algorithm == FaissAlgorithm::kCagra) {
            state->retainedGpuBuffers.emplace_back(
                input.owner, input.owner.get());
          }
      context->synchronize();
      return state;
    }
  }

  std::map<int64_t, size_t> rowCounts;
  std::map<int64_t, std::vector<int64_t>> documentIds;
  for (const auto& input : inputs) {
    for (vector_size_t row = 0; row < input.rowCount; ++row) {
      const auto cluster = input.clusters ? input.clusters->at(row) : 0;
      ++rowCounts[cluster];
      documentIds[cluster].push_back(input.documentIds[row]);
    }
  }

  std::map<int64_t, float*> deviceVectors;
  auto freeDeviceVectors = folly::makeGuard([&] {
    for (const auto& [cluster, values] : deviceVectors) {
      static_cast<void>(cluster);
      cudaFree(values);
    }
  });
  for (const auto& [cluster, count] : rowCounts) {
    float* values = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&values),
            count * config.dimension * sizeof(float)),
        "allocating FAISS GPU build input");
    deviceVectors.emplace(cluster, values);
  }

  std::map<int64_t, size_t> offsets;
  for (const auto& input : inputs) {
    const auto stream = reinterpret_cast<cudaStream_t>(input.stream);
    for (vector_size_t row = 0; row < input.rowCount; ++row) {
      const auto cluster = input.clusters ? input.clusters->at(row) : 0;
      auto* destination =
          deviceVectors.at(cluster) + offsets[cluster] * config.dimension;
      checkCuda(
          cudaMemcpyAsync(
              destination,
              input.embeddings + row * config.dimension,
              config.dimension * sizeof(float),
              cudaMemcpyDeviceToDevice,
              stream),
          "gathering FAISS GPU build input");
      ++offsets[cluster];
    }
  }
  for (const auto& input : inputs) {
    checkCuda(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(input.stream)),
        "synchronizing FAISS GPU build input");
  }

  for (const auto& [cluster, values] : deviceVectors) {
    addGpuCluster(
        *state, config, *context, cluster, values, documentIds.at(cluster));
  }
  if (config.algorithm == FaissAlgorithm::kCagra) {
    for (const auto& [cluster, values] : deviceVectors) {
      static_cast<void>(cluster);
      state->retainedGpuBuffers.emplace_back(values, [](void* buffer) {
        cudaFree(buffer);
      });
    }
    deviceVectors.clear();
  }
  // Publishing only after the task-scoped indexes are ready is a conservative
  // cross-stream readiness barrier for all probe drivers.
  context->synchronize();
  return state;
}

void promoteLoadedIndexesToGpu(FaissIndexState& state) {
  VELOX_USER_CHECK(state.config.executionDevice == FaissExecutionDevice::kGpu);
  VELOX_USER_CHECK(
      state.config.algorithm != FaissAlgorithm::kHnswCagra,
      "CAGRA-to-HNSW artifacts must remain CPU HNSW indexes when loaded");
  auto context = std::make_shared<FaissGpuContextImpl>(state.config.gpuDevice);
  for (auto& [cluster, value] : state.clusters) {
    if (state.config.algorithm == FaissAlgorithm::kCagra) {
      auto* cpuCagra =
          dynamic_cast<::faiss::IndexHNSWCagra*>(value.index.get());
      VELOX_USER_CHECK_NOT_NULL(
          cpuCagra,
          "FAISS CAGRA artifact cluster {} is not IndexHNSWCagra",
          cluster);
      value.index = context->toGpuCagra(cpuCagra, state.config);
    } else {
      value.index = context->toGpu(value.index.get());
    }
    value.gpuResident = true;
  }
  state.gpuContext = std::move(context);
}

} // namespace facebook::velox::faiss
