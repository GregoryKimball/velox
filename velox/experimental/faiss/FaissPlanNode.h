/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#pragma once

#include "velox/core/PlanNode.h"

#include <optional>

namespace facebook::velox::faiss {

enum class FaissAlgorithm { kFlat, kIvfFlat, kIvfPq, kHnsw, kHnswCagra };
enum class FaissMetric { kL2, kInnerProduct };

struct FaissIndexConfig {
  FaissAlgorithm algorithm{FaissAlgorithm::kFlat};
  FaissMetric metric{FaissMetric::kL2};
  int32_t dimension{0};
  int32_t nlist{1};
  int32_t nprobe{1};
  int32_t pqSubquantizers{1};
  int32_t pqBits{8};
  int32_t hnswM{32};
  int32_t efConstruction{40};
  int32_t efSearch{16};

  folly::dynamic serialize() const;
  static FaissIndexConfig deserialize(const folly::dynamic& obj);
  void validate() const;
};

class AssignClustersNode final : public core::PlanNode {
 public:
  AssignClustersNode(
      core::PlanNodeId id,
      core::PlanNodePtr source,
      std::string embeddingColumn,
      std::string clusterColumn,
      int32_t dimension,
      std::vector<float> centroids,
      FaissMetric metric = FaissMetric::kL2);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }
  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }
  std::string_view name() const override {
    return "AssignClusters";
  }
  const std::string& embeddingColumn() const {
    return embeddingColumn_;
  }
  int32_t dimension() const {
    return dimension_;
  }
  const std::vector<float>& centroids() const {
    return centroids_;
  }
  FaissMetric metric() const {
    return metric_;
  }
  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;
  std::vector<core::PlanNodePtr> sources_;
  RowTypePtr outputType_;
  std::string embeddingColumn_;
  std::string clusterColumn_;
  int32_t dimension_;
  std::vector<float> centroids_;
  FaissMetric metric_;
};

class BuildIndexNode final : public core::PlanNode {
 public:
  BuildIndexNode(
      core::PlanNodeId id,
      core::PlanNodePtr source,
      std::string idColumn,
      std::string embeddingColumn,
      std::optional<std::string> clusterColumn,
      FaissIndexConfig config,
      std::optional<std::string> artifactDirectory = std::nullopt);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }
  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }
  std::string_view name() const override {
    return "BuildIndex";
  }
  const std::string& idColumn() const {
    return idColumn_;
  }
  const std::string& embeddingColumn() const {
    return embeddingColumn_;
  }
  const std::optional<std::string>& clusterColumn() const {
    return clusterColumn_;
  }
  const FaissIndexConfig& config() const {
    return config_;
  }
  const std::optional<std::string>& artifactDirectory() const {
    return artifactDirectory_;
  }
  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;
  std::vector<core::PlanNodePtr> sources_;
  RowTypePtr outputType_;
  std::string idColumn_;
  std::string embeddingColumn_;
  std::optional<std::string> clusterColumn_;
  FaissIndexConfig config_;
  std::optional<std::string> artifactDirectory_;
};

class LoadIndexNode final : public core::PlanNode {
 public:
  LoadIndexNode(core::PlanNodeId id, std::string artifactDirectory);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }
  const std::vector<core::PlanNodePtr>& sources() const override;
  std::string_view name() const override {
    return "LoadIndex";
  }
  const std::string& artifactDirectory() const {
    return artifactDirectory_;
  }
  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;
  RowTypePtr outputType_{ROW({}, {})};
  std::string artifactDirectory_;
};

class SearchIndexNode final : public core::PlanNode {
 public:
  SearchIndexNode(
      core::PlanNodeId id,
      core::PlanNodePtr querySource,
      core::PlanNodePtr indexProvider,
      std::string queryIdColumn,
      std::string queryEmbeddingColumn,
      std::optional<std::string> queryClusterColumn,
      int32_t topK,
      std::optional<float> maxDistance = std::nullopt);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }
  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }
  std::string_view name() const override {
    return "SearchIndex";
  }
  const std::string& queryIdColumn() const {
    return queryIdColumn_;
  }
  const std::string& queryEmbeddingColumn() const {
    return queryEmbeddingColumn_;
  }
  const std::optional<std::string>& queryClusterColumn() const {
    return queryClusterColumn_;
  }
  int32_t topK() const {
    return topK_;
  }
  std::optional<float> maxDistance() const {
    return maxDistance_;
  }
  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;
  std::vector<core::PlanNodePtr> sources_;
  RowTypePtr outputType_{
      ROW({"query_id", "result_id", "distance", "rank"},
          {BIGINT(), BIGINT(), REAL(), INTEGER()})};
  std::string queryIdColumn_;
  std::string queryEmbeddingColumn_;
  std::optional<std::string> queryClusterColumn_;
  int32_t topK_;
  std::optional<float> maxDistance_;
};

void registerFaissPlanNodeSerDe();

} // namespace facebook::velox::faiss
