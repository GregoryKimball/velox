/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissOperators.h"
#if defined(VELOX_ENABLE_FAISS_GPU)
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/faiss/FaissGpuIndex.h"
#endif

#include "velox/exec/Task.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <chrono>
#include <iterator>
#include <limits>

namespace facebook::velox::faiss {
namespace {

#if defined(VELOX_ENABLE_FAISS_GPU)
using cudf_velox::extractClassAndFunction;
using cudf_velox::extractFunctionName;
using cudf_velox::NvtxHelper;
using cudf_velox::VeloxDomain;

class FaissOperatorNvtx : public NvtxHelper {
 public:
  FaissOperatorNvtx(int32_t operatorId, const core::PlanNodeId& nodeId)
      : NvtxHelper(
            nvtx3::rgb{147, 112, 219}, // Medium purple.
            operatorId,
            fmt::format("[{}]", nodeId)) {}
};

#define VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE(name) \
  VELOX_NVTX_OPERATOR_FUNC_RANGE(name)
#else
class FaissOperatorNvtx {
 public:
  FaissOperatorNvtx(int32_t /* operatorId */, const core::PlanNodeId& /* id */) {}
};

#define VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE(name)
#endif

RuntimeCounter milliseconds(double value) {
  return RuntimeCounter(
      static_cast<int64_t>(value * 1'000'000),
      RuntimeCounter::Unit::kNanos);
}

} // namespace

const ArrayVector* validateFaissEmbeddings(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& name,
    int32_t dimension) {
  const auto channel = type->getChildIdx(name);
  auto& embeddings = input->childAt(channel);
  embeddings = BaseVector::loadedVectorShared(embeddings);
  BaseVector::flattenVector(embeddings);
  const auto* arrays = embeddings->as<ArrayVector>();
  VELOX_USER_CHECK_NOT_NULL(
      arrays, "FAISS embedding column must be ARRAY<REAL>");
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
const SimpleVector<T>* materializeSimpleColumn(
    const RowVectorPtr& input,
    const RowTypePtr& type,
    const std::string& name) {
  auto& column = input->childAt(type->getChildIdx(name));
  column = BaseVector::loadedVectorShared(column);
  BaseVector::flattenVector(column);
  const auto* simple = column->as<SimpleVector<T>>();
  VELOX_USER_CHECK_NOT_NULL(simple, "FAISS column {} has an invalid type", name);
  return simple;
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

class AssignClustersOperator final
    : public exec::Operator,
      public FaissOperatorNvtx {
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
        FaissOperatorNvtx(id, node->id()),
        node_(std::move(node)) {}

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }
  void addInput(RowVectorPtr input) override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissAssignClusters");
    input_ = std::move(input);
  }
  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }
  RowVectorPtr getOutput() override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissAssignClusters");
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

class ProviderPassThrough final
    : public exec::Operator,
      public FaissOperatorNvtx {
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
            "FaissBuildIndexProvider"),
        FaissOperatorNvtx(id, node->id()) {}
  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }
  void addInput(RowVectorPtr input) override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissBuildIndexProvider");
    input_ = std::move(input);
  }
  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }
  RowVectorPtr getOutput() override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissBuildIndexProvider");
    return std::exchange(input_, nullptr);
  }
  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }
};

class LoadProviderSource final
    : public exec::Operator,
      public FaissOperatorNvtx {
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
            "FaissLoadIndexProvider"),
        FaissOperatorNvtx(id, node->id()) {}
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
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissLoadIndexProvider");
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

class IndexBuildOperator final
    : public exec::Operator,
      public FaissOperatorNvtx {
 public:
  IndexBuildOperator(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const SearchIndexNode> node)
      : Operator(ctx, nullptr, id, node->id(), "FaissIndexBuild"),
        FaissOperatorNvtx(id, node->id()),
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
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissIndexBuild");
    if (!buildNode_ || input->size() == 0) {
      return;
    }
#if defined(VELOX_ENABLE_FAISS_GPU)
    if (buildNode_->config().executionDevice == FaissExecutionDevice::kGpu) {
      auto gpuInput = extractFaissGpuBuildInput(
          input,
          buildNode_->outputType(),
          buildNode_->idColumn(),
          buildNode_->embeddingColumn(),
          buildNode_->clusterColumn(),
          buildNode_->config().dimension);
      if (gpuInput) {
        gpuInputs_.push_back(std::move(*gpuInput));
        return;
      }
    }
#endif
    const auto* arrays = validateFaissEmbeddings(
        input,
        buildNode_->outputType(),
        buildNode_->embeddingColumn(),
        buildNode_->config().dimension);
    const auto* ids = materializeSimpleColumn<int64_t>(
        input,
        buildNode_->outputType(),
        buildNode_->idColumn());
    const SimpleVector<int64_t>* clusters = nullptr;
    if (buildNode_->clusterColumn()) {
      clusters = materializeSimpleColumn<int64_t>(
          input,
          buildNode_->outputType(),
          *buildNode_->clusterColumn());
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
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissIndexBuild");
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
#if defined(VELOX_ENABLE_FAISS_GPU)
      gpuInputs_.insert(
          gpuInputs_.end(),
          std::make_move_iterator(build->gpuInputs_.begin()),
          std::make_move_iterator(build->gpuInputs_.end()));
#endif
    }
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }

    std::shared_ptr<FaissIndexState> state;
    if (buildNode_) {
#if defined(VELOX_ENABLE_FAISS_GPU)
      if (buildNode_->config().executionDevice ==
              FaissExecutionDevice::kGpu &&
          !gpuInputs_.empty()) {
        VELOX_USER_CHECK(
            vectors_.empty(),
            "FAISS GPU index build cannot mix CudfVector and RowVector input");
        state = buildFaissGpuIndexState(buildNode_->config(), gpuInputs_);
      } else {
        state = buildFaissIndexState(buildNode_->config(), vectors_, ids_);
      }
#else
      state = buildFaissIndexState(buildNode_->config(), vectors_, ids_);
#endif
      if (buildNode_->artifactDirectory()) {
        writeFaissArtifact(*state, *buildNode_->artifactDirectory());
      }
    } else {
      VELOX_CHECK_NOT_NULL(loadNode_);
      state = loadFaissArtifact(loadNode_->artifactDirectory());
      if (loadNode_->targetConfig()) {
        applyFaissLoadTarget(*state, *loadNode_->targetConfig());
      }
    }
    {
      auto stats = stats_.wlock();
      stats->addRuntimeStat(
          "faissTrainWallNanos", milliseconds(state->trainMilliseconds));
      stats->addRuntimeStat(
          "faissAddWallNanos", milliseconds(state->addMilliseconds));
      stats->addRuntimeStat(
          "faissLoadReadWallNanos",
          milliseconds(state->loadReadMilliseconds));
      stats->addRuntimeStat(
          "faissLoadDeserializeWallNanos",
          milliseconds(state->loadDeserializeMilliseconds));
      stats->addRuntimeStat(
          "faissLoadUploadWallNanos",
          milliseconds(state->loadUploadMilliseconds));
      stats->addRuntimeStat(
          "faissCagraToHnswWallNanos",
          milliseconds(state->cagraCopyToMilliseconds));
    }
    auto bridge = std::dynamic_pointer_cast<FaissIndexBridge>(
        operatorCtx_->task()->getCustomJoinBridge(
            operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
    VELOX_CHECK_NOT_NULL(bridge);
    bridge->setState(std::move(state));
  }
  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissIndexBuild");
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
#if defined(VELOX_ENABLE_FAISS_GPU)
  std::vector<FaissGpuBuildInput> gpuInputs_;
#endif
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

class IndexSearchOperator final
    : public exec::Operator,
      public FaissOperatorNvtx {
 public:
  IndexSearchOperator(
      int32_t id,
      exec::DriverCtx* ctx,
      std::shared_ptr<const SearchIndexNode> node)
      : Operator(ctx, node->outputType(), id, node->id(), "FaissIndexSearch"),
        FaissOperatorNvtx(id, node->id()),
        node_(std::move(node)) {}

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }
  void addInput(RowVectorPtr input) override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissIndexSearch");
    input_ = std::move(input);
  }
  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissIndexSearch");
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
    VELOX_FAISS_NVTX_OPERATOR_FUNC_RANGE("FaissIndexSearch");
    if (!input_ || !state_) {
      return nullptr;
    }
    const auto conversionStart = std::chrono::steady_clock::now();
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
      queryIds = materializeSimpleColumn<int64_t>(
          input_, queryType, node_->queryIdColumn());
      if (node_->queryClusterColumn()) {
        clusters = materializeSimpleColumn<int64_t>(
            input_, queryType, *node_->queryClusterColumn());
      }
    }
    conversionMilliseconds_ +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - conversionStart)
            .count();
    std::vector<int64_t> outputQueryIds;
    std::vector<int64_t> outputResultIds;
    std::vector<float> outputDistances;
    std::vector<int32_t> outputRanks;
    outputQueryIds.reserve(input_->size() * node_->topK());
    outputResultIds.reserve(input_->size() * node_->topK());
    outputDistances.reserve(input_->size() * node_->topK());
    outputRanks.reserve(input_->size() * node_->topK());

    std::map<int64_t, std::vector<vector_size_t>> rowsByCluster;
    for (vector_size_t row = 0; row < input_->size(); ++row) {
      if (!gpuInput) {
        VELOX_USER_CHECK(!queryIds->isNullAt(row), "FAISS query ID is null");
        if (clusters) {
          VELOX_USER_CHECK(
              !clusters->isNullAt(row), "FAISS cluster ID is null");
        }
      }
      const auto cluster = gpuInput
          ? (gpuInput->clusters ? gpuInput->clusters->at(row) : 0)
          : (clusters ? clusters->valueAt(row) : 0);
      rowsByCluster[cluster].push_back(row);
    }

    for (const auto& [cluster, rows] : rowsByCluster) {
      const auto it = state_->clusters.find(cluster);
      if (it == state_->clusters.end()) {
        continue;
      }
      std::vector<::faiss::idx_t> labels;
      std::vector<float> distances;
#if defined(VELOX_ENABLE_FAISS_GPU)
      if (gpuInput) {
        searchFaissGpuRows(
            *state_,
            it->second,
            *gpuInput,
            rows,
            state_->config.dimension,
            node_->topK(),
            distances,
            labels);
      } else
#endif
      {
        std::vector<float> queries;
        queries.reserve(rows.size() * state_->config.dimension);
        for (const auto row : rows) {
          const auto vector = embeddingAt(arrays, row);
          queries.insert(queries.end(), vector.begin(), vector.end());
        }
        distances.resize(rows.size() * node_->topK());
        labels.resize(rows.size() * node_->topK());
        searchFaissIndex(
            *state_,
            it->second,
            rows.size(),
            queries.data(),
            node_->topK(),
            distances.data(),
            labels.data());
      }
      const auto gatherStart = std::chrono::steady_clock::now();
      for (size_t batchRow = 0; batchRow < rows.size(); ++batchRow) {
        const auto row = rows[batchRow];
        const auto queryId =
            gpuInput ? gpuInput->queryIds[row] : queryIds->valueAt(row);
        for (int32_t rank = 0; rank < node_->topK(); ++rank) {
          const auto result = batchRow * node_->topK() + rank;
          const auto ordinal = labels[result];
          if (ordinal < 0) {
            continue;
          }
          VELOX_CHECK_LT(ordinal, it->second.documentIds.size());
          if (node_->maxDistance() &&
              distances[result] > *node_->maxDistance()) {
            continue;
          }
          outputQueryIds.push_back(queryId);
          outputResultIds.push_back(it->second.documentIds[ordinal]);
          outputDistances.push_back(distances[result]);
          outputRanks.push_back(rank + 1);
        }
      }
      gatherMilliseconds_ +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - gatherStart)
              .count();
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
      if (state_ && !reportedStats_) {
        auto stats = stats_.wlock();
        stats->addRuntimeStat(
            "faissConversionWallNanos", milliseconds(conversionMilliseconds_));
        stats->addRuntimeStat(
            "faissSearchWallNanos", milliseconds(state_->searchMilliseconds));
        stats->addRuntimeStat(
            "faissIdGatherWallNanos", milliseconds(gatherMilliseconds_));
        reportedStats_ = true;
      }
      state_.reset();
      return true;
    }
    return false;
  }

 private:
  std::shared_ptr<const SearchIndexNode> node_;
  std::shared_ptr<FaissIndexState> state_;
  double conversionMilliseconds_{0};
  double gatherMilliseconds_{0};
  bool reportedStats_{false};
};

#if defined(VELOX_ENABLE_FAISS_GPU)
bool usesGpuIndex(const std::shared_ptr<const SearchIndexNode>& search) {
  if (auto build =
          std::dynamic_pointer_cast<const BuildIndexNode>(search->sources()[1])) {
    return build->config().executionDevice == FaissExecutionDevice::kGpu;
  }
  if (auto load =
          std::dynamic_pointer_cast<const LoadIndexNode>(search->sources()[1])) {
    return load->targetConfig() &&
        load->targetConfig()->executionDevice == FaissExecutionDevice::kGpu;
  }
  return false;
}

bool buildsGpuIndex(const std::shared_ptr<const SearchIndexNode>& search) {
  const auto build =
      std::dynamic_pointer_cast<const BuildIndexNode>(search->sources()[1]);
  return build &&
      build->config().executionDevice == FaissExecutionDevice::kGpu;
}

class AssignClustersAdapter final : public cudf_velox::OperatorAdapter {
 public:
  AssignClustersAdapter() : OperatorAdapter("FaissAssignClusters") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const AssignClustersOperator*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    const auto assign =
        std::dynamic_pointer_cast<const AssignClustersNode>(planNode);
    return assign &&
        assign->executionDevice() == FaissExecutionDevice::kGpu;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto assign = std::dynamic_pointer_cast<const AssignClustersNode>(planNode);
    VELOX_CHECK_NOT_NULL(assign);
    std::vector<std::unique_ptr<exec::Operator>> replacements;
    replacements.push_back(
        makeFaissGpuAssignClusters(operatorId, ctx, std::move(assign)));
    return replacements;
  }
};

class BuildProviderAdapter final : public cudf_velox::OperatorAdapter {
 public:
  BuildProviderAdapter() : OperatorAdapter("FaissBuildIndexProvider") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const ProviderPassThrough*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    const auto build = std::dynamic_pointer_cast<const BuildIndexNode>(planNode);
    return build &&
        build->config().executionDevice == FaissExecutionDevice::kGpu;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

  bool keepOperator() const override {
    return true;
  }
};

class IndexBuildAdapter final : public cudf_velox::OperatorAdapter {
 public:
  IndexBuildAdapter() : OperatorAdapter("FaissIndexBuild") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const IndexBuildOperator*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    const auto search =
        std::dynamic_pointer_cast<const SearchIndexNode>(planNode);
    return search && buildsGpuIndex(search);
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

  bool keepOperator() const override {
    return true;
  }
};

class IndexSearchAdapter final : public cudf_velox::OperatorAdapter {
 public:
  IndexSearchAdapter() : OperatorAdapter("FaissIndexSearch") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const IndexSearchOperator*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    const auto search =
        std::dynamic_pointer_cast<const SearchIndexNode>(planNode);
    return search && usesGpuIndex(search);
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

  bool keepOperator() const override {
    return true;
  }
};
#endif

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
#if defined(VELOX_ENABLE_FAISS_GPU)
  auto& adapters = cudf_velox::OperatorAdapterRegistry::getInstance();
  adapters.registerAdapter(std::make_unique<AssignClustersAdapter>());
  adapters.registerAdapter(std::make_unique<BuildProviderAdapter>());
  adapters.registerAdapter(std::make_unique<IndexBuildAdapter>());
  adapters.registerAdapter(std::make_unique<IndexSearchAdapter>());
#endif
}

} // namespace facebook::velox::faiss
