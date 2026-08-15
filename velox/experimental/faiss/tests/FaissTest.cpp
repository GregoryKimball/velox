/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#include "velox/experimental/faiss/FaissOperators.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <faiss/Index.h>
#include <gtest/gtest.h>

#include <filesystem>

namespace facebook::velox::faiss::test {
namespace {

FaissIndexConfig flatConfig() {
  FaissIndexConfig config;
  config.dimension = 2;
  return config;
}

TEST(FaissIndexTest, exactFlat) {
  auto state = buildFaissIndexState(
      flatConfig(),
      {{0, {0, 0, 2, 0, 0, 3}}},
      {{0, {101, 102, 103}}});
  std::vector<float> distances(2);
  std::vector<::faiss::idx_t> labels(2);
  const float query[] = {1.9, 0};
  state->clusters.at(0).index->search(
      1, query, 2, distances.data(), labels.data());
  EXPECT_EQ(state->clusters.at(0).documentIds.at(labels[0]), 102);
  EXPECT_FLOAT_EQ(distances[0], 0.01F);
}

TEST(FaissIndexTest, artifactRoundTripAndBuildLoadParity) {
  auto built = buildFaissIndexState(
      flatConfig(),
      {{7, {0, 0, 2, 0}}, {9, {5, 5}}},
      {{7, {std::numeric_limits<int64_t>::min(), 42}}, {9, {99}}});
  const auto directory =
      (std::filesystem::temp_directory_path() /
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

TEST(FaissIndexTest, invalidDimensions) {
  auto config = flatConfig();
  config.dimension = 0;
  VELOX_ASSERT_THROW(config.validate(), "dimension must be positive");
  config = flatConfig();
  config.algorithm = FaissAlgorithm::kIvfPq;
  config.pqSubquantizers = 3;
  VELOX_ASSERT_THROW(config.validate(), "divisible");
  VELOX_ASSERT_THROW(
      buildFaissIndexState(config, {{0, {1, 2}}}, {{0, {1}}}),
      "divisible");
}

class FaissPlanNodeTest : public testing::Test,
                          public velox::test::VectorTestBase {};

TEST_F(FaissPlanNodeTest, serializationRoundTrip) {
  core::PlanNode::registerSerDe();
  registerFaissPlanNodeSerDe();
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
  const auto copy = ISerializable::deserialize<core::PlanNode>(
      search->serialize(), pool());
  const auto restored =
      std::dynamic_pointer_cast<const SearchIndexNode>(copy);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->topK(), 1);
  ASSERT_NE(
      std::dynamic_pointer_cast<const BuildIndexNode>(
          restored->sources().at(1)),
      nullptr);
}

TEST_F(FaissPlanNodeTest, rejectsNullEmbeddings) {
  auto nullRow = makeRowVector(
      {"embedding"},
      {makeNullableArrayVector<float>({std::nullopt})});
  VELOX_ASSERT_THROW(
      validateFaissEmbeddings(
          nullRow, nullRow->rowType(), "embedding", 2),
      "embedding is null");

  auto nullElement = makeRowVector(
      {"embedding"},
      {makeNullableArrayVector<float>({{1.0F, std::nullopt}})});
  VELOX_ASSERT_THROW(
      validateFaissEmbeddings(
          nullElement, nullElement->rowType(), "embedding", 2),
      "element is null");
}

} // namespace
} // namespace facebook::velox::faiss::test
