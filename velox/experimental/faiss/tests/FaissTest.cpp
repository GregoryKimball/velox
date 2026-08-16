/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissOperators.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/LocalFile.h"
#if !defined(VELOX_ENABLE_FAISS_GPU)
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#endif
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <faiss/Index.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#if defined(VELOX_ENABLE_FAISS_GPU)
#include <cuda_runtime_api.h>
#include <faiss/IndexHNSW.h>
#endif

namespace facebook::velox::faiss::test {
namespace {

[[maybe_unused]] const bool kLocalFileSystemRegistered = [] {
  filesystems::registerLocalFileSystem();
  return true;
}();

FaissIndexConfig flatConfig() {
  FaissIndexConfig config;
  config.dimension = 2;
  return config;
}

TEST(FaissIndexTest, exactFlat) {
  auto state = buildFaissIndexState(
      flatConfig(), {{0, {0, 0, 2, 0, 0, 3}}}, {{0, {101, 102, 103}}});
  std::vector<float> distances(2);
  std::vector<::faiss::idx_t> labels(2);
  const float query[] = {1.9, 0};
  state->clusters.at(0).index->search(
      1, query, 2, distances.data(), labels.data());
  EXPECT_EQ(state->clusters.at(0).documentIds.at(labels[0]), 102);
  EXPECT_NEAR(distances[0], 0.01F, 1e-6F);
}

TEST(FaissIndexTest, artifactRoundTripAndBuildLoadParity) {
  auto built = buildFaissIndexState(
      flatConfig(),
      {{7, {0, 0, 2, 0}}, {9, {5, 5}}},
      {{7, {std::numeric_limits<int64_t>::min(), 42}}, {9, {99}}});
  const auto directory = (std::filesystem::temp_directory_path() /
                          "velox_faiss_artifact_roundtrip")
                             .string();
  std::filesystem::remove_all(directory);
  writeFaissArtifact(*built, directory);
  auto loaded = loadFaissArtifact(directory);
  ASSERT_EQ(loaded->rowCount, built->rowCount);
  ASSERT_EQ(loaded->clusters.size(), built->clusters.size());
  for (const auto& [cluster, expected] : built->clusters) {
    const auto& actual = loaded->clusters.at(cluster);
    EXPECT_EQ(actual.documentIds, expected.documentIds);
    const float query[] = {1.9, 0};
    float expectedDistance;
    float actualDistance;
    ::faiss::idx_t expectedLabel;
    ::faiss::idx_t actualLabel;
    expected.index->search(1, query, 1, &expectedDistance, &expectedLabel);
    actual.index->search(1, query, 1, &actualDistance, &actualLabel);
    EXPECT_EQ(actualLabel, expectedLabel);
    EXPECT_FLOAT_EQ(actualDistance, expectedDistance);
  }
  std::filesystem::remove_all(directory);
}

TEST(FaissIndexTest, loadTargetRejectsArtifactMismatch) {
  auto built = buildFaissIndexState(
      flatConfig(), {{0, {0, 0, 2, 0}}}, {{0, {10, 20}}});
  const auto directory = (std::filesystem::temp_directory_path() /
                          "velox_faiss_load_target_mismatch")
                             .string();
  std::filesystem::remove_all(directory);
  writeFaissArtifact(*built, directory);

  auto dimensionMismatch = flatConfig();
  dimensionMismatch.dimension = 4;
  auto loaded = loadFaissArtifact(directory);
  VELOX_ASSERT_THROW(
      applyFaissLoadTarget(*loaded, dimensionMismatch),
      "dimension does not match target");

  auto metricMismatch = flatConfig();
  metricMismatch.metric = FaissMetric::kInnerProduct;
  loaded = loadFaissArtifact(directory);
  VELOX_ASSERT_THROW(
      applyFaissLoadTarget(*loaded, metricMismatch),
      "metric does not match target");

  auto algorithmMismatch = flatConfig();
  algorithmMismatch.algorithm = FaissAlgorithm::kHnsw;
  loaded = loadFaissArtifact(directory);
  VELOX_ASSERT_THROW(
      applyFaissLoadTarget(*loaded, algorithmMismatch),
      "algorithm does not match target strategy");
  std::filesystem::remove_all(directory);
}

TEST(FaissIndexTest, rejectsCorruptArtifactSidecar) {
  auto built = buildFaissIndexState(
      flatConfig(), {{0, {0, 0, 2, 0}}}, {{0, {10, 20}}});
  const auto directory = (std::filesystem::temp_directory_path() /
                          "velox_faiss_corrupt_artifact")
                             .string();
  std::filesystem::remove_all(directory);
  writeFaissArtifact(*built, directory);
  {
    std::fstream output(
        directory + "/cluster_0.ids",
        std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(output);
    output.seekp(0);
    output.put('\xff');
  }
  VELOX_ASSERT_THROW(
      loadFaissArtifact(directory), "FAISS ID sidecar checksum mismatch");
  std::filesystem::remove_all(directory);
}

TEST(FaissIndexTest, bridgePublishesAndCancels) {
  FaissIndexBridge bridge;
  bridge.start();
  ContinueFuture future = ContinueFuture::makeEmpty();
  EXPECT_EQ(bridge.stateOrFuture(&future), nullptr);
  ASSERT_TRUE(future.valid());
  EXPECT_FALSE(future.isReady());

  auto state = buildFaissIndexState(
      flatConfig(), {{0, {0, 0}}}, {{0, {101}}});
  bridge.setState(state);
  EXPECT_TRUE(future.isReady());
  ContinueFuture unused = ContinueFuture::makeEmpty();
  EXPECT_EQ(bridge.stateOrFuture(&unused), state);

  FaissIndexBridge cancelled;
  cancelled.start();
  ContinueFuture cancelledFuture = ContinueFuture::makeEmpty();
  EXPECT_EQ(cancelled.stateOrFuture(&cancelledFuture), nullptr);
  cancelled.cancel();
  EXPECT_TRUE(cancelledFuture.isReady());
  VELOX_ASSERT_THROW(
      cancelled.stateOrFuture(&unused),
      "Getting FAISS state after build was aborted");
}

TEST(FaissIndexTest, invalidDimensions) {
  auto config = flatConfig();
  config.dimension = 0;
  VELOX_ASSERT_THROW(config.validate(), "dimension must be positive");
  config = flatConfig();
  config.algorithm = FaissAlgorithm::kIvfPq;
  config.pqSubquantizers = 3;
  VELOX_ASSERT_THROW(config.validate(), "divisible");
  VELOX_ASSERT_THROW(
      buildFaissIndexState(config, {{0, {1, 2}}}, {{0, {1}}}), "divisible");
}

TEST(FaissIndexTest, executionDeviceSerialization) {
  auto config = flatConfig();
  config.executionDevice = FaissExecutionDevice::kGpu;
  config.gpuDevice = 3;
  const auto restored = FaissIndexConfig::deserialize(config.serialize());
  EXPECT_EQ(restored.executionDevice, FaissExecutionDevice::kGpu);
  EXPECT_EQ(restored.gpuDevice, 3);
}

#if defined(VELOX_ENABLE_FAISS_GPU)
bool hasGpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST(FaissIndexTest, cpuGpuFlatParity) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No CUDA GPU available";
  }
  auto cpuConfig = flatConfig();
  auto gpuConfig = cpuConfig;
  gpuConfig.executionDevice = FaissExecutionDevice::kGpu;
  const std::map<int64_t, std::vector<float>> vectors{
      {0, {0, 0, 2, 0, 0, 3, 4, 4}}};
  const std::map<int64_t, std::vector<int64_t>> ids{{0, {101, 102, 103, 104}}};
  auto cpu = buildFaissIndexState(cpuConfig, vectors, ids);
  auto gpu = buildFaissIndexState(gpuConfig, vectors, ids);

  const float queries[] = {1.9F, 0, 0, 2.8F};
  std::vector<float> cpuDistances(4);
  std::vector<float> gpuDistances(4);
  std::vector<::faiss::idx_t> cpuLabels(4);
  std::vector<::faiss::idx_t> gpuLabels(4);
  searchFaissIndex(
      *cpu,
      cpu->clusters.at(0),
      2,
      queries,
      2,
      cpuDistances.data(),
      cpuLabels.data());
  searchFaissIndex(
      *gpu,
      gpu->clusters.at(0),
      2,
      queries,
      2,
      gpuDistances.data(),
      gpuLabels.data());
  EXPECT_EQ(gpuLabels, cpuLabels);
  for (size_t i = 0; i < cpuDistances.size(); ++i) {
    EXPECT_NEAR(gpuDistances[i], cpuDistances[i], 1e-5);
  }
}

TEST(FaissIndexTest, cpuArtifactLoadsToGpuTarget) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No CUDA GPU available";
  }
  auto cpu = buildFaissIndexState(
      flatConfig(), {{0, {0, 0, 2, 0}}}, {{0, {10, 20}}});
  const auto directory = (std::filesystem::temp_directory_path() /
                          "velox_faiss_cpu_artifact_gpu_target")
                             .string();
  std::filesystem::remove_all(directory);
  writeFaissArtifact(*cpu, directory);

  auto loaded = loadFaissArtifact(directory);
  EXPECT_EQ(loaded->config.executionDevice, FaissExecutionDevice::kCpu);
  auto target = flatConfig();
  target.executionDevice = FaissExecutionDevice::kGpu;
  applyFaissLoadTarget(*loaded, target);
  EXPECT_EQ(loaded->config.executionDevice, FaissExecutionDevice::kGpu);
  EXPECT_TRUE(loaded->clusters.at(0).gpuResident);
  EXPECT_NE(loaded->gpuContext, nullptr);
  std::filesystem::remove_all(directory);
}

TEST(FaissIndexTest, cagraConvertsAndPersistsAsCpuHnsw) {
  if (!hasGpu()) {
    GTEST_SKIP() << "No CUDA GPU available";
  }
  FaissIndexConfig config;
  config.algorithm = FaissAlgorithm::kHnswCagra;
  config.executionDevice = FaissExecutionDevice::kGpu;
  config.dimension = 8;
  config.hnswM = 8;
  config.efSearch = 32;
  std::vector<float> vectors(256 * config.dimension);
  std::vector<int64_t> ids(256);
  for (size_t row = 0; row < ids.size(); ++row) {
    ids[row] = 1000 + row;
    for (int32_t d = 0; d < config.dimension; ++d) {
      vectors[row * config.dimension + d] =
          static_cast<float>((row * 17 + d * 3) % 101) / 101.0F;
    }
  }

  auto built = buildFaissIndexState(config, {{0, vectors}}, {{0, ids}});
  EXPECT_FALSE(built->clusters.at(0).gpuResident);
  EXPECT_NE(
      dynamic_cast<::faiss::IndexHNSWCagra*>(built->clusters.at(0).index.get()),
      nullptr);
  EXPECT_GT(built->cagraCopyToMilliseconds, 0);

  const auto directory = (std::filesystem::temp_directory_path() /
                          "velox_faiss_cagra_hnsw_artifact")
                             .string();
  std::filesystem::remove_all(directory);
  writeFaissArtifact(*built, directory);
  auto loaded = loadFaissArtifact(directory);
  applyFaissLoadTarget(*loaded, config);
  EXPECT_EQ(loaded->config.executionDevice, FaissExecutionDevice::kGpu);
  EXPECT_FALSE(loaded->clusters.at(0).gpuResident);
  EXPECT_NE(
      dynamic_cast<::faiss::IndexHNSWCagra*>(
          loaded->clusters.at(0).index.get()),
      nullptr);
  std::vector<float> distances(3);
  std::vector<::faiss::idx_t> labels(3);
  loaded->clusters.at(0).index->search(
      1, vectors.data(), 3, distances.data(), labels.data());
  ASSERT_GE(labels[0], 0);
  EXPECT_EQ(loaded->clusters.at(0).documentIds.at(labels[0]), ids[0]);
  std::filesystem::remove_all(directory);
}
#endif

class FaissPlanNodeTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    core::PlanNode::registerSerDe();
    registerFaiss();
  }
};

TEST_F(FaissPlanNodeTest, serializationRoundTrip) {
  auto candidates = makeRowVector(
      {"doc_id", "embedding"},
      {makeFlatVector<int64_t>({11, 12}),
       makeArrayVector<float>({{0, 0}, {1, 1}})});
  auto queries = makeRowVector(
      {"query_id", "embedding"},
      {makeFlatVector<int64_t>({21}), makeArrayVector<float>({{0, 0}})});
  auto build = std::make_shared<BuildIndexNode>(
      "build",
      std::make_shared<core::ValuesNode>(
          "candidate_values", std::vector<RowVectorPtr>{candidates}),
      "doc_id",
      "embedding",
      std::nullopt,
      flatConfig());
  auto search = std::make_shared<SearchIndexNode>(
      "search",
      std::make_shared<core::ValuesNode>(
          "query_values", std::vector<RowVectorPtr>{queries}),
      build,
      "query_id",
      "embedding",
      std::nullopt,
      1);
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(search->serialize(), pool());
  const auto restored = std::dynamic_pointer_cast<const SearchIndexNode>(copy);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->topK(), 1);
  ASSERT_NE(
      std::dynamic_pointer_cast<const BuildIndexNode>(
          restored->sources().at(1)),
      nullptr);

  auto target = flatConfig();
  target.nprobe = 7;
  auto load = std::make_shared<LoadIndexNode>("load", "/tmp/index", target);
  const auto loadCopy =
      ISerializable::deserialize<core::PlanNode>(load->serialize(), pool());
  const auto restoredLoad =
      std::dynamic_pointer_cast<const LoadIndexNode>(loadCopy);
  ASSERT_NE(restoredLoad, nullptr);
  ASSERT_TRUE(restoredLoad->targetConfig().has_value());
  EXPECT_EQ(restoredLoad->targetConfig()->nprobe, 7);

  auto legacy = load->serialize();
  legacy.erase("targetConfig");
  const auto legacyCopy =
      ISerializable::deserialize<core::PlanNode>(legacy, pool());
  const auto restoredLegacy =
      std::dynamic_pointer_cast<const LoadIndexNode>(legacyCopy);
  ASSERT_NE(restoredLegacy, nullptr);
  EXPECT_FALSE(restoredLegacy->targetConfig().has_value());

  auto assignment = std::make_shared<AssignClustersNode>(
      "assign",
      std::make_shared<core::ValuesNode>(
          "assignment_values", std::vector<RowVectorPtr>{queries}),
      "embedding",
      "cluster_id",
      2,
      std::vector<float>{0, 0, 1, 1},
      FaissMetric::kL2,
      FaissExecutionDevice::kGpu);
  const auto assignmentCopy = ISerializable::deserialize<core::PlanNode>(
      assignment->serialize(), pool());
  const auto restoredAssignment =
      std::dynamic_pointer_cast<const AssignClustersNode>(assignmentCopy);
  ASSERT_NE(restoredAssignment, nullptr);
  EXPECT_EQ(
      restoredAssignment->executionDevice(), FaissExecutionDevice::kGpu);
  EXPECT_EQ(restoredAssignment->centroids().size(), 4);

  auto legacyAssignment = assignment->serialize();
  legacyAssignment.erase("executionDevice");
  const auto legacyAssignmentCopy = ISerializable::deserialize<core::PlanNode>(
      legacyAssignment, pool());
  EXPECT_EQ(
      std::dynamic_pointer_cast<const AssignClustersNode>(legacyAssignmentCopy)
          ->executionDevice(),
      FaissExecutionDevice::kCpu);
}

TEST_F(FaissPlanNodeTest, rejectsNullEmbeddings) {
  auto nullRow = makeRowVector(
      {"embedding"}, {makeNullableArrayVector<float>({std::nullopt})});
  VELOX_ASSERT_THROW(
      validateFaissEmbeddings(nullRow, nullRow->rowType(), "embedding", 2),
      "embedding is null");

  auto nullElement = makeRowVector(
      {"embedding"}, {makeNullableArrayVector<float>({{1.0F, std::nullopt}})});
  VELOX_ASSERT_THROW(
      validateFaissEmbeddings(
          nullElement, nullElement->rowType(), "embedding", 2),
      "element is null");
}

#if !defined(VELOX_ENABLE_FAISS_GPU)
TEST_F(FaissPlanNodeTest, executesBuildSearchWithMultipleDrivers) {
  auto candidates = makeRowVector(
      {"doc_id", "embedding", "cluster_id"},
      {makeFlatVector<int64_t>({101, 102, 103, 104}),
       makeArrayVector<float>({{0, 0}, {2, 0}, {0, 3}, {4, 4}}),
       makeFlatVector<int64_t>({0, 0, 0, 0})});
  auto queries = makeRowVector(
      {"query_id", "embedding", "cluster_id"},
      {makeFlatVector<int64_t>({201, 202}),
       makeArrayVector<float>({{1.9, 0}, {0, 2.9}}),
       makeFlatVector<int64_t>({0, 0})});
  auto build = std::make_shared<BuildIndexNode>(
      "exec_build",
      std::make_shared<core::ValuesNode>(
          "exec_candidate_values", std::vector<RowVectorPtr>{candidates}),
      "doc_id",
      "embedding",
      "cluster_id",
      flatConfig());
  auto search = std::make_shared<SearchIndexNode>(
      "exec_search",
      std::make_shared<core::ValuesNode>(
          "exec_query_values", std::vector<RowVectorPtr>{queries}),
      build,
      "query_id",
      "embedding",
      "cluster_id",
      1);

  auto result = exec::test::AssertQueryBuilder(search)
                    .maxDrivers(2)
                    .copyResults(pool());
  ASSERT_EQ(result->size(), 2);
  EXPECT_EQ(result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0), 201);
  EXPECT_EQ(result->childAt(1)->as<SimpleVector<int64_t>>()->valueAt(0), 102);
  EXPECT_EQ(result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(1), 202);
  EXPECT_EQ(result->childAt(1)->as<SimpleVector<int64_t>>()->valueAt(1), 103);
}
#endif

} // namespace
} // namespace facebook::velox::faiss::test

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
