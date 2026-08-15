/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissOperators.h"

#include "velox/exec/Task.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <limits>

namespace facebook::velox::faiss {

const ArrayVector* validateFaissEmbeddings(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& name,
    int32_t dimension) {
  const auto channel = type->getChildIdx(name);
  const auto* arrays = input->childAt(channel)->as<ArrayVector>();
  VELOX_USER_CHECK_NOT_NULL(
      arrays, "FAISS embedding column must be a flat ARRAY<REAL>");
  VELOX_USER_CHECK_EQ(arrays->elements()->typeKind(), TypeKind::REAL);
  const auto* elements = arrays->elements()->as<SimpleVector<float>>();
  VELOX_USER_CHECK_NOT_NULL(elements);
  for (vector_size_t row = 0; row < input->size(); ++row) {
    VELOX_USER_CHECK(
        !arrays->isNullAt(row), "FAISS embedding is null at row {}", row);
    VELOX_USER_CHECK_EQ(
        arrays->sizeAt(row),
        dimension,
        "FAISS embedding dimension mismatch at row {}",
        row);
    const auto offset = arrays->offsetAt(row);
    for (int32_t d = 0; d < dimension; ++d) {
      VELOX_USER_CHECK(
          !elements->isNullAt(offset + d),
          "FAISS embedding element is null at row {}",
          row);
    }
  }
  return arrays;
}

namespace {

std::vector<float> embeddingAt(const ArrayVector* arrays, vector_size_t row) {
  const auto* elements = arrays->elements()->as<SimpleVector<float>>();
  const auto offset = arrays->offsetAt(row);
  std::vector<float> result(arrays->sizeAt(row));
  for (vector_size_t d = 0; d < arrays->sizeAt(row); ++d) {
    result[d] = elements->valueAt(offset + d);
  }
  return result;
}

template <typename T>
std::shared_ptr<FlatVector<T>> makeFlat(
    memory::MemoryPool* pool,
    const TypePtr& type,
    const std::vector<T>& values) {
  auto result = BaseVector::create<FlatVector<T>>(type, values.size(), pool);
  for (vector_size_t i = 0; i < values.size(); ++i) {
    result->set(i, values[i]);
  }
  return result;
}

class AssignClustersOperator final : public exec::Operator {
 public:
  AssignClustersOperator(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const AssignClustersNode> node)
      : Operator(
            ctx,
            node->outputType(),
            id,
            node->id(),
            "FaissAssignClusters"),
        node_(std::move(node)) {}

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }
  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }
  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }
  RowVectorPtr getOutput() override {
    if (!input_) {
      return nullptr;
    }
    const auto* arrays = validateFaissEmbeddings(
        input_,
        node_->sources()[0]->outputType(),
        node_->embeddingColumn(),
        node_->dimension());
    std::vector<int64_t> clusters(input_->size());
    const auto count = node_->centroids().size() / node_->dimension();
    for (vector_size_t row = 0; row < input_->size(); ++row) {
      const auto vector = embeddingAt(arrays, row);
      float best = node_->metric() == FaissMetric::kL2
          ? std::numeric_limits<float>::infinity()
          : -std::numeric_limits<float>::infinity();
      for (size_t cluster = 0; cluster < count; ++cluster) {
        float score = 0;
        for (int32_t d = 0; d < node_->dimension(); ++d) {
          const auto centroid =
              node_->centroids()[cluster * node_->dimension() + d];
          if (node_->metric() == FaissMetric::kL2) {
            const auto delta = vector[d] - centroid;
            score += delta * delta;
          } else {
            score += vector[d] * centroid;
          }
        }
        if ((node_->metric() == FaissMetric::kL2 && score < best) ||
            (node_->metric() == FaissMetric::kInnerProduct && score > best)) {
          best = score;
          clusters[row] = cluster;
        }
      }
    }
    auto children = input_->children();
    children.push_back(makeFlat<int64_t>(pool(), BIGINT(), clusters));
    auto output = std::make_shared<RowVector>(
        pool(),
        node_->outputType(),
        nullptr,
        input_->size(),
        std::move(children));
    input_.reset();
    return output;
  }
  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 private:
  std::shared_ptr<const AssignClustersNode> node_;
};

class ProviderPassThrough final : public exec::Operator {
 public:
  ProviderPassThrough(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const BuildIndexNode> node)
      : Operator(
            ctx,
            node->outputType(),
            id,
            node->id(),
            "FaissBuildIndexProvider") {}
  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }
  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }
  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }
  RowVectorPtr getOutput() override {
    return std::exchange(input_, nullptr);
  }
  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }
};

class LoadProviderSource final : public exec::Operator {
 public:
  LoadProviderSource(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const LoadIndexNode> node)
      : Operator(
            ctx,
            node->outputType(),
            id,
            node->id(),
            "FaissLoadIndexProvider") {}
  bool needsInput() const override {
    return false;
  }
  void addInput(RowVectorPtr /* input */) override {
    VELOX_UNREACHABLE();
  }
  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }
  RowVectorPtr getOutput() override {
    if (finished_) {
      return nullptr;
    }
    finished_ = true;
    return std::make_shared<RowVector>(
        pool(), outputType_, nullptr, 1, std::vector<VectorPtr>{});
  }
  bool isFinished() override {
    return finished_;
  }

 private:
  bool finished_{false};
};

class IndexBuildOperator final : public exec::Operator {
 public:
  IndexBuildOperator(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const SearchIndexNode> node)
      : Operator(ctx, nullptr, id, node->id(), "FaissIndexBuild"),
        node_(std::move(node)),
        buildNode_(
            std::dynamic_pointer_cast<const BuildIndexNode>(
                node_->sources()[1])),
        loadNode_(
            std::dynamic_pointer_cast<const LoadIndexNode>(
                node_->sources()[1])) {}

  bool needsInput() const override {
    return !noMoreInput_;
  }
  void addInput(RowVectorPtr input) override {
    if (!buildNode_ || input->size() == 0) {
      return;
    }
#if defined(VELOX_ENABLE_FAISS_GPU)
    if (buildNode_->config().executionDevice == FaissExecutionDevice::kGpu) {
      auto gpuInput = copyFaissGpuBuildInput(
          input,
          buildNode_->outputType(),
          buildNode_->idColumn(),
          buildNode_->embeddingColumn(),
          buildNode_->clusterColumn(),
          buildNode_->config().dimension);
      if (gpuInput) {
        for (vector_size_t row = 0; row < input->size(); ++row) {
          const auto cluster =
              gpuInput->clusters ? gpuInput->clusters->at(row) : 0;
          const auto begin = gpuInput->embeddings.begin() +
              row * buildNode_->config().dimension;
          vectors_[cluster].insert(
              vectors_[cluster].end(),
              begin,
              begin + buildNode_->config().dimension);
          ids_[cluster].push_back(gpuInput->documentIds[row]);
        }
        return;
      }
    }
#endif
    const auto* arrays = validateFaissEmbeddings(
        input,
        buildNode_->outputType(),
        buildNode_->embeddingColumn(),
        buildNode_->config().dimension);
    const auto* ids = input
                          ->childAt(buildNode_->outputType()->getChildIdx(
                              buildNode_->idColumn()))
                          ->as<SimpleVector<int64_t>>();
    VELOX_USER_CHECK_NOT_NULL(ids);
    const SimpleVector<int64_t>* clusters = nullptr;
    if (buildNode_->clusterColumn()) {
      clusters = input
                     ->childAt(buildNode_->outputType()->getChildIdx(
                         *buildNode_->clusterColumn()))
                     ->as<SimpleVector<int64_t>>();
      VELOX_USER_CHECK_NOT_NULL(clusters);
    }
    for (vector_size_t row = 0; row < input->size(); ++row) {
      VELOX_USER_CHECK(!ids->isNullAt(row), "FAISS document ID is null");
      if (clusters) {
        VELOX_USER_CHECK(!clusters->isNullAt(row), "FAISS cluster ID is null");
      }
      const auto cluster = clusters ? clusters->valueAt(row) : 0;
      auto vector = embeddingAt(arrays, row);
      vectors_[cluster].insert(
          vectors_[cluster].end(), vector.begin(), vector.end());
      ids_[cluster].push_back(ids->valueAt(row));
    }
  }
  RowVectorPtr getOutput() override {
    return nullptr;
  }
  void noMoreInput() override {
    Operator::noMoreInput();
    std::vector<ContinuePromise> promises;
    std::vector<std::shared_ptr<exec::Driver>> peers;
    if (!operatorCtx_->task()->allPeersFinished(
            planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
      return;
    }
    for (auto& peer : peers) {
      auto* build =
          dynamic_cast<IndexBuildOperator*>(peer->findOperator(planNodeId()));
      VELOX_CHECK_NOT_NULL(build);
      for (auto& [cluster, values] : build->vectors_) {
        vectors_[cluster].insert(
            vectors_[cluster].end(), values.begin(), values.end());
      }
      for (auto& [cluster, ids] : build->ids_) {
        ids_[cluster].insert(ids_[cluster].end(), ids.begin(), ids.end());
      }
    }
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }

    std::shared_ptr<FaissIndexState> state;
    if (buildNode_) {
      state = buildFaissIndexState(buildNode_->config(), vectors_, ids_);
      if (buildNode_->artifactDirectory()) {
        writeFaissArtifact(*state, *buildNode_->artifactDirectory());
      }
    } else {
      VELOX_CHECK_NOT_NULL(loadNode_);
      state = loadFaissArtifact(loadNode_->artifactDirectory());
    }
    auto bridge = std::dynamic_pointer_cast<FaissIndexBridge>(
        operatorCtx_->task()->getCustomJoinBridge(
            operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
    VELOX_CHECK_NOT_NULL(bridge);
    bridge->setState(std::move(state));
  }
  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    if (!future_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  bool isFinished() override {
    return noMoreInput_ && !future_.valid();
  }

 private:
  std::shared_ptr<const SearchIndexNode> node_;
  std::shared_ptr<const BuildIndexNode> buildNode_;
  std::shared_ptr<const LoadIndexNode> loadNode_;
  std::map<int64_t, std::vector<float>> vectors_;
  std::map<int64_t, std::vector<int64_t>> ids_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

class IndexSearchOperator final : public exec::Operator {
 public:
  IndexSearchOperator(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const SearchIndexNode> node)
      : Operator(ctx, node->outputType(), id, node->id(), "FaissIndexSearch"),
        node_(std::move(node)) {}

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }
  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }
  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    if (state_) {
      return exec::BlockingReason::kNotBlocked;
    }
    auto bridge = std::dynamic_pointer_cast<FaissIndexBridge>(
        operatorCtx_->task()->getCustomJoinBridge(
            operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
    VELOX_CHECK_NOT_NULL(bridge);
    state_ = bridge->stateOrFuture(future);
    return state_ ? exec::BlockingReason::kNotBlocked
                  : exec::BlockingReason::kWaitForJoinBuild;
  }
  RowVectorPtr getOutput() override {
    if (!input_ || !state_) {
      return nullptr;
    }
    const auto queryType = node_->sources()[0]->outputType();
    const ArrayVector* arrays = nullptr;
    const SimpleVector<int64_t>* queryIds = nullptr;
    const SimpleVector<int64_t>* clusters = nullptr;
    std::optional<FaissGpuQueryInput> gpuInput;
#if defined(VELOX_ENABLE_FAISS_GPU)
    if (state_->config.executionDevice == FaissExecutionDevice::kGpu) {
      gpuInput = extractFaissGpuQueryInput(
          input_,
          queryType,
          node_->queryIdColumn(),
          node_->queryEmbeddingColumn(),
          node_->queryClusterColumn(),
          state_->config.dimension,
          state_->config.algorithm == FaissAlgorithm::kHnswCagra);
    }
#endif
    if (!gpuInput) {
      arrays = validateFaissEmbeddings(
          input_,
          queryType,
          node_->queryEmbeddingColumn(),
          state_->config.dimension);
      queryIds = input_->childAt(queryType->getChildIdx(node_->queryIdColumn()))
                     ->as<SimpleVector<int64_t>>();
      VELOX_USER_CHECK_NOT_NULL(queryIds);
      if (node_->queryClusterColumn()) {
        clusters =
            input_
                ->childAt(queryType->getChildIdx(*node_->queryClusterColumn()))
                ->as<SimpleVector<int64_t>>();
        VELOX_USER_CHECK_NOT_NULL(clusters);
      }
    }
    std::vector<int64_t> outputQueryIds;
    std::vector<int64_t> outputResultIds;
    std::vector<float> outputDistances;
    std::vector<int32_t> outputRanks;
    std::vector<::faiss::idx_t> labels(node_->topK());
    std::vector<float> distances(node_->topK());
    for (vector_size_t row = 0; row < input_->size(); ++row) {
      if (!gpuInput) {
        VELOX_USER_CHECK(!queryIds->isNullAt(row), "FAISS query ID is null");
        if (clusters) {
          VELOX_USER_CHECK(
              !clusters->isNullAt(row), "FAISS cluster ID is null");
        }
      }
      const auto queryId =
          gpuInput ? gpuInput->queryIds[row] : queryIds->valueAt(row);
      const auto cluster = gpuInput
          ? (gpuInput->clusters ? gpuInput->clusters->at(row) : 0)
          : (clusters ? clusters->valueAt(row) : 0);
      const auto it = state_->clusters.find(cluster);
      if (it == state_->clusters.end()) {
        continue;
      }
      std::vector<float> hostVector;
      const float* query;
      uintptr_t stream = 0;
      if (gpuInput) {
        query = gpuInput->embeddings + row * state_->config.dimension;
        stream = gpuInput->stream;
      } else {
        hostVector = embeddingAt(arrays, row);
        query = hostVector.data();
      }
      searchFaissIndex(
          *state_,
          it->second,
          1,
          query,
          node_->topK(),
          distances.data(),
          labels.data(),
          stream);
      for (int32_t rank = 0; rank < node_->topK(); ++rank) {
        const auto ordinal = labels[rank];
        if (ordinal < 0) {
          continue;
        }
        VELOX_CHECK_LT(ordinal, it->second.documentIds.size());
        if (node_->maxDistance() && distances[rank] > *node_->maxDistance()) {
          continue;
        }
        outputQueryIds.push_back(queryId);
        outputResultIds.push_back(it->second.documentIds[ordinal]);
        outputDistances.push_back(distances[rank]);
        outputRanks.push_back(rank + 1);
      }
    }
    input_.reset();
    std::vector<VectorPtr> children{
        makeFlat<int64_t>(pool(), BIGINT(), outputQueryIds),
        makeFlat<int64_t>(pool(), BIGINT(), outputResultIds),
        makeFlat<float>(pool(), REAL(), outputDistances),
        makeFlat<int32_t>(pool(), INTEGER(), outputRanks)};
    return std::make_shared<RowVector>(
        pool(),
        node_->outputType(),
        nullptr,
        outputQueryIds.size(),
        std::move(children));
  }
  bool isFinished() override {
    if (noMoreInput_ && input_ == nullptr) {
      state_.reset();
      return true;
    }
    return false;
  }

 private:
  std::shared_ptr<const SearchIndexNode> node_;
  std::shared_ptr<FaissIndexState> state_;
};

} // namespace

void FaissIndexBridge::setState(std::shared_ptr<FaissIndexState> state) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_NULL(state_);
    state_ = std::move(state);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::shared_ptr<FaissIndexState> FaissIndexBridge::stateOrFuture(
    ContinueFuture* future) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!cancelled_, "Getting FAISS state after build was aborted");
  if (state_) {
    return state_;
  }
  promises_.emplace_back("FaissIndexBridge::stateOrFuture");
  *future = promises_.back().getSemiFuture();
  return nullptr;
}

std::unique_ptr<exec::Operator> FaissPlanNodeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto assign = std::dynamic_pointer_cast<const AssignClustersNode>(node)) {
    return std::make_unique<AssignClustersOperator>(id, ctx, assign);
  }
  if (auto build = std::dynamic_pointer_cast<const BuildIndexNode>(node)) {
    return std::make_unique<ProviderPassThrough>(id, ctx, build);
  }
  if (auto load = std::dynamic_pointer_cast<const LoadIndexNode>(node)) {
    return std::make_unique<LoadProviderSource>(id, ctx, load);
  }
  if (auto search = std::dynamic_pointer_cast<const SearchIndexNode>(node)) {
    return std::make_unique<IndexSearchOperator>(id, ctx, search);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> FaissPlanNodeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const SearchIndexNode>(node)) {
    return std::make_unique<FaissIndexBridge>();
  }
  return nullptr;
}

exec::OperatorSupplier FaissPlanNodeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (auto search = std::dynamic_pointer_cast<const SearchIndexNode>(node)) {
    return [search](int32_t id, exec::DriverCtx* ctx) {
      return std::make_unique<IndexBuildOperator>(id, ctx, search);
    };
  }
  return nullptr;
}

std::optional<uint32_t> FaissPlanNodeTranslator::maxDrivers(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const LoadIndexNode>(node)) {
    return 1;
  }
  return std::nullopt;
}

void registerFaiss() {
  registerFaissPlanNodeSerDe();
  exec::Operator::registerOperator(std::make_unique<FaissPlanNodeTranslator>());
}

} // namespace facebook::velox::faiss
