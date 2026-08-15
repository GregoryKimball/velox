/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissPlanNode.h"

#include "velox/common/serialization/DeserializationRegistry.h"

namespace facebook::velox::faiss {
namespace {

std::string algorithmName(FaissAlgorithm algorithm) {
  switch (algorithm) {
    case FaissAlgorithm::kFlat:
      return "flat";
    case FaissAlgorithm::kIvfFlat:
      return "ivf_flat";
    case FaissAlgorithm::kIvfPq:
      return "ivf_pq";
    case FaissAlgorithm::kCagra:
      return "cagra";
    case FaissAlgorithm::kHnsw:
      return "hnsw";
    case FaissAlgorithm::kHnswCagra:
      return "hnsw_cagra";
  }
  VELOX_UNREACHABLE();
}

FaissAlgorithm parseAlgorithm(const std::string& value) {
  if (value == "flat") {
    return FaissAlgorithm::kFlat;
  }
  if (value == "ivf_flat") {
    return FaissAlgorithm::kIvfFlat;
  }
  if (value == "ivf_pq") {
    return FaissAlgorithm::kIvfPq;
  }
  if (value == "cagra") {
    return FaissAlgorithm::kCagra;
  }
  if (value == "hnsw") {
    return FaissAlgorithm::kHnsw;
  }
  if (value == "hnsw_cagra") {
    return FaissAlgorithm::kHnswCagra;
  }
  VELOX_USER_FAIL("Unknown FAISS algorithm: {}", value);
}

std::string metricName(FaissMetric metric) {
  return metric == FaissMetric::kL2 ? "l2" : "inner_product";
}

FaissMetric parseMetric(const std::string& value) {
  if (value == "l2") {
    return FaissMetric::kL2;
  }
  if (value == "inner_product") {
    return FaissMetric::kInnerProduct;
  }
  VELOX_USER_FAIL("Unknown FAISS metric: {}", value);
}

std::string deviceName(FaissExecutionDevice device) {
  return device == FaissExecutionDevice::kCpu ? "cpu" : "gpu";
}

FaissExecutionDevice parseDevice(const std::string& value) {
  if (value == "cpu") {
    return FaissExecutionDevice::kCpu;
  }
  if (value == "gpu") {
    return FaissExecutionDevice::kGpu;
  }
  VELOX_USER_FAIL("Unknown FAISS execution device: {}", value);
}

core::PlanNodePtr
sourceAt(const folly::dynamic& obj, size_t index, void* context) {
  return ISerializable::deserialize<core::PlanNode>(
      obj["sources"][index], context);
}

std::optional<std::string> optionalString(
    const folly::dynamic& obj,
    const char* key) {
  if (obj.count(key) == 0 || obj[key].isNull()) {
    return std::nullopt;
  }
  return obj[key].asString();
}

folly::dynamic serializeFloats(const std::vector<float>& values) {
  folly::dynamic result = folly::dynamic::array;
  for (auto value : values) {
    result.push_back(static_cast<double>(value));
  }
  return result;
}

std::vector<float> deserializeFloats(const folly::dynamic& values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (const auto& value : values) {
    result.push_back(static_cast<float>(value.asDouble()));
  }
  return result;
}

} // namespace

folly::dynamic FaissIndexConfig::serialize() const {
  return folly::dynamic::object("algorithm", algorithmName(algorithm))(
      "metric", metricName(metric))(
      "executionDevice", deviceName(executionDevice))("gpuDevice", gpuDevice)(
      "dimension", dimension)("nlist", nlist)("nprobe", nprobe)(
      "pqSubquantizers", pqSubquantizers)("pqBits", pqBits)("hnswM", hnswM)(
      "efConstruction", efConstruction)("efSearch", efSearch);
}

FaissIndexConfig FaissIndexConfig::deserialize(const folly::dynamic& obj) {
  FaissIndexConfig config;
  config.algorithm = parseAlgorithm(obj["algorithm"].asString());
  config.metric = parseMetric(obj["metric"].asString());
  // executionDevice was added after the CPU artifact format was introduced.
  // Missing fields therefore intentionally deserialize as CPU.
  if (obj.count("executionDevice")) {
    config.executionDevice = parseDevice(obj["executionDevice"].asString());
  }
  if (obj.count("gpuDevice")) {
    config.gpuDevice = obj["gpuDevice"].asInt();
  }
  config.dimension = obj["dimension"].asInt();
  config.nlist = obj["nlist"].asInt();
  config.nprobe = obj["nprobe"].asInt();
  config.pqSubquantizers = obj["pqSubquantizers"].asInt();
  config.pqBits = obj["pqBits"].asInt();
  config.hnswM = obj["hnswM"].asInt();
  config.efConstruction = obj["efConstruction"].asInt();
  config.efSearch = obj["efSearch"].asInt();
  config.validate();
  return config;
}

void FaissIndexConfig::validate() const {
  VELOX_USER_CHECK_GT(dimension, 0, "FAISS dimension must be positive");
  VELOX_USER_CHECK_GE(gpuDevice, 0, "FAISS GPU device must be non-negative");
  VELOX_USER_CHECK_GT(nlist, 0, "FAISS nlist must be positive");
  VELOX_USER_CHECK_GT(nprobe, 0, "FAISS nprobe must be positive");
  VELOX_USER_CHECK_GT(hnswM, 0, "FAISS HNSW M must be positive");
  VELOX_USER_CHECK(
      executionDevice != FaissExecutionDevice::kGpu ||
          algorithm != FaissAlgorithm::kHnsw,
      "FAISS GPU supports CAGRA, not the CPU HNSW build strategy");
  VELOX_USER_CHECK(
      algorithm != FaissAlgorithm::kCagra ||
          executionDevice == FaissExecutionDevice::kGpu,
      "FAISS CAGRA requires GPU execution");
  if (algorithm == FaissAlgorithm::kIvfPq) {
    VELOX_USER_CHECK_GT(
        pqSubquantizers, 0, "FAISS PQ subquantizers must be positive");
    VELOX_USER_CHECK_EQ(
        dimension % pqSubquantizers,
        0,
        "FAISS dimension must be divisible by PQ subquantizers");
  }
}

AssignClustersNode::AssignClustersNode(
    core::PlanNodeId id,
    core::PlanNodePtr source,
    std::string embeddingColumn,
    std::string clusterColumn,
    int32_t dimension,
    std::vector<float> centroids,
    FaissMetric metric)
    : PlanNode(std::move(id)),
      sources_{std::move(source)},
      embeddingColumn_(std::move(embeddingColumn)),
      clusterColumn_(std::move(clusterColumn)),
      dimension_(dimension),
      centroids_(std::move(centroids)),
      metric_(metric) {
  VELOX_USER_CHECK_GT(dimension_, 0);
  VELOX_USER_CHECK(
      !centroids_.empty() && centroids_.size() % dimension_ == 0,
      "Centroids must contain complete, non-empty vectors");
  auto names = sources_[0]->outputType()->names();
  auto types = sources_[0]->outputType()->children();
  VELOX_USER_CHECK(
      !sources_[0]->outputType()->containsChild(clusterColumn_),
      "Cluster output column already exists: {}",
      clusterColumn_);
  names.push_back(clusterColumn_);
  types.push_back(BIGINT());
  outputType_ = ROW(std::move(names), std::move(types));
}

void AssignClustersNode::addDetails(std::stringstream& stream) const {
  stream << embeddingColumn_ << " -> " << clusterColumn_ << ", "
         << centroids_.size() / dimension_ << " centroids";
}

folly::dynamic AssignClustersNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["embeddingColumn"] = embeddingColumn_;
  obj["clusterColumn"] = clusterColumn_;
  obj["dimension"] = dimension_;
  obj["centroids"] = serializeFloats(centroids_);
  obj["metric"] = metricName(metric_);
  return obj;
}

core::PlanNodePtr AssignClustersNode::create(
    const folly::dynamic& obj,
    void* context) {
  return std::make_shared<AssignClustersNode>(
      obj["id"].asString(),
      sourceAt(obj, 0, context),
      obj["embeddingColumn"].asString(),
      obj["clusterColumn"].asString(),
      obj["dimension"].asInt(),
      deserializeFloats(obj["centroids"]),
      parseMetric(obj["metric"].asString()));
}

BuildIndexNode::BuildIndexNode(
    core::PlanNodeId id,
    core::PlanNodePtr source,
    std::string idColumn,
    std::string embeddingColumn,
    std::optional<std::string> clusterColumn,
    FaissIndexConfig config,
    std::optional<std::string> artifactDirectory)
    : PlanNode(std::move(id)),
      sources_{std::move(source)},
      outputType_(sources_[0]->outputType()),
      idColumn_(std::move(idColumn)),
      embeddingColumn_(std::move(embeddingColumn)),
      clusterColumn_(std::move(clusterColumn)),
      config_(std::move(config)),
      artifactDirectory_(std::move(artifactDirectory)) {
  config_.validate();
  VELOX_USER_CHECK(
      outputType_->childAt(outputType_->getChildIdx(idColumn_))
          ->kindEquals(BIGINT()),
      "FAISS document ID column must be BIGINT");
  VELOX_USER_CHECK(
      outputType_->childAt(outputType_->getChildIdx(embeddingColumn_))
          ->kindEquals(ARRAY(REAL())),
      "FAISS embedding column must be ARRAY<REAL>");
}

void BuildIndexNode::addDetails(std::stringstream& stream) const {
  stream << algorithmName(config_.algorithm) << " on "
         << deviceName(config_.executionDevice) << ", " << embeddingColumn_;
}

folly::dynamic BuildIndexNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["idColumn"] = idColumn_;
  obj["embeddingColumn"] = embeddingColumn_;
  obj["clusterColumn"] = clusterColumn_ ? folly::dynamic(*clusterColumn_)
                                        : folly::dynamic(nullptr);
  obj["config"] = config_.serialize();
  obj["artifactDirectory"] = artifactDirectory_
      ? folly::dynamic(*artifactDirectory_)
      : folly::dynamic(nullptr);
  return obj;
}

core::PlanNodePtr BuildIndexNode::create(
    const folly::dynamic& obj,
    void* context) {
  return std::make_shared<BuildIndexNode>(
      obj["id"].asString(),
      sourceAt(obj, 0, context),
      obj["idColumn"].asString(),
      obj["embeddingColumn"].asString(),
      optionalString(obj, "clusterColumn"),
      FaissIndexConfig::deserialize(obj["config"]),
      optionalString(obj, "artifactDirectory"));
}

LoadIndexNode::LoadIndexNode(
    core::PlanNodeId id,
    std::string artifactDirectory,
    std::optional<FaissIndexConfig> targetConfig)
    : PlanNode(std::move(id)),
      artifactDirectory_(std::move(artifactDirectory)),
      targetConfig_(std::move(targetConfig)) {
  VELOX_USER_CHECK(!artifactDirectory_.empty(), "Artifact directory is empty");
  if (targetConfig_) {
    targetConfig_->validate();
  }
}

const std::vector<core::PlanNodePtr>& LoadIndexNode::sources() const {
  static const std::vector<core::PlanNodePtr> kNoSources;
  return kNoSources;
}

void LoadIndexNode::addDetails(std::stringstream& stream) const {
  stream << artifactDirectory_;
  if (targetConfig_) {
    stream << ", target " << deviceName(targetConfig_->executionDevice);
  }
}

folly::dynamic LoadIndexNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["artifactDirectory"] = artifactDirectory_;
  obj["targetConfig"] =
      targetConfig_ ? targetConfig_->serialize() : folly::dynamic(nullptr);
  return obj;
}

core::PlanNodePtr LoadIndexNode::create(
    const folly::dynamic& obj,
    void* /* context */) {
  std::optional<FaissIndexConfig> targetConfig;
  if (obj.count("targetConfig") && !obj["targetConfig"].isNull()) {
    targetConfig = FaissIndexConfig::deserialize(obj["targetConfig"]);
  }
  return std::make_shared<LoadIndexNode>(
      obj["id"].asString(),
      obj["artifactDirectory"].asString(),
      std::move(targetConfig));
}

SearchIndexNode::SearchIndexNode(
    core::PlanNodeId id,
    core::PlanNodePtr querySource,
    core::PlanNodePtr indexProvider,
    std::string queryIdColumn,
    std::string queryEmbeddingColumn,
    std::optional<std::string> queryClusterColumn,
    int32_t topK,
    std::optional<float> maxDistance)
    : PlanNode(std::move(id)),
      sources_{std::move(querySource), std::move(indexProvider)},
      queryIdColumn_(std::move(queryIdColumn)),
      queryEmbeddingColumn_(std::move(queryEmbeddingColumn)),
      queryClusterColumn_(std::move(queryClusterColumn)),
      topK_(topK),
      maxDistance_(maxDistance) {
  VELOX_USER_CHECK_GT(topK_, 0, "FAISS top-k must be positive");
  const auto& queryType = sources_[0]->outputType();
  VELOX_USER_CHECK(
      queryType->childAt(queryType->getChildIdx(queryIdColumn_))
          ->kindEquals(BIGINT()),
      "FAISS query ID column must be BIGINT");
  VELOX_USER_CHECK(
      queryType->childAt(queryType->getChildIdx(queryEmbeddingColumn_))
          ->kindEquals(ARRAY(REAL())),
      "FAISS query embedding column must be ARRAY<REAL>");
  VELOX_USER_CHECK(
      std::dynamic_pointer_cast<const BuildIndexNode>(sources_[1]) ||
          std::dynamic_pointer_cast<const LoadIndexNode>(sources_[1]),
      "SearchIndex provider must be BuildIndexNode or LoadIndexNode");
}

void SearchIndexNode::addDetails(std::stringstream& stream) const {
  stream << "top " << topK_;
}

folly::dynamic SearchIndexNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["queryIdColumn"] = queryIdColumn_;
  obj["queryEmbeddingColumn"] = queryEmbeddingColumn_;
  obj["queryClusterColumn"] = queryClusterColumn_
      ? folly::dynamic(*queryClusterColumn_)
      : folly::dynamic(nullptr);
  obj["topK"] = topK_;
  obj["maxDistance"] = maxDistance_
      ? folly::dynamic(static_cast<double>(*maxDistance_))
      : folly::dynamic(nullptr);
  return obj;
}

core::PlanNodePtr SearchIndexNode::create(
    const folly::dynamic& obj,
    void* context) {
  std::optional<float> maxDistance;
  if (obj.count("maxDistance") && !obj["maxDistance"].isNull()) {
    maxDistance = static_cast<float>(obj["maxDistance"].asDouble());
  }
  return std::make_shared<SearchIndexNode>(
      obj["id"].asString(),
      sourceAt(obj, 0, context),
      sourceAt(obj, 1, context),
      obj["queryIdColumn"].asString(),
      obj["queryEmbeddingColumn"].asString(),
      optionalString(obj, "queryClusterColumn"),
      obj["topK"].asInt(),
      maxDistance);
}

void registerFaissPlanNodeSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("AssignClustersNode", AssignClustersNode::create);
  registry.Register("BuildIndexNode", BuildIndexNode::create);
  registry.Register("LoadIndexNode", LoadIndexNode::create);
  registry.Register("SearchIndexNode", SearchIndexNode::create);
}

} // namespace facebook::velox::faiss
