/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/CudfSpillContext.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

using core::QueryConfig;

// Runs group-bys whose buffered state moves to pinned host memory and back.
// Nothing here depends on debug-only test hooks: a spill is forced by making
// the group-by spill by itself as soon as the query's GPU pool is fuller than
// one percent of a small limit, which every input exceeds.
class GroupbySpillTest : public OperatorTestBase {
 protected:
  // One percent of this is below the smallest reservation of a pool, so a
  // group-by with any buffered state is over its self-spill threshold.
  static constexpr int64_t kQueryGpuBytes = 64LL << 20;
  static constexpr uint64_t kHostSpillBytes = 256ULL << 20;

  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = cudf_velox::CudfConfig::getInstance();
    savedConfig_ = config;
    config.streamingGroupbyEnabled = false;
    config.allowCpuFallback = false;
    config.spillHostBytes = kHostSpillBytes;
    config.groupbySelfSpillPercent = 1;
    config.groupbySelfSpillTargetPercent = 0;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoAggregateFunctions("");
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::unregisterAggregateFunctions();
    cudf_velox::CudfConfig::getInstance() = savedConfig_;
    OperatorTestBase::TearDown();
  }

  // A query with a GPU pool of kQueryGpuBytes, which makes its group-bys
  // spillable.
  std::shared_ptr<core::QueryCtx> makeSpillableQueryCtx(
      const std::string& queryId) {
    auto queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId(queryId)
                        .build();
    cudf_velox::attachCudfMemoryPool(*queryCtx, kQueryGpuBytes);
    EXPECT_NE(
        queryCtx->customPool(std::string{cudf_velox::kCudfMemoryResourceTag}),
        nullptr);
    EXPECT_NE(cudf_velox::cudfSpillContext(*queryCtx), nullptr);
    return queryCtx;
  }

  // 'numBatches' batches of 'rowsPerBatch' rows. Key c0 takes 'numKeys'
  // values, and is null in some rows if 'nullKeys' is set; c1 is nullable.
  std::vector<RowVectorPtr> makeInput(
      int32_t numBatches,
      int32_t rowsPerBatch,
      int64_t numKeys,
      bool nullKeys = true) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t batch = 0; batch < numBatches; ++batch) {
      vectors.push_back(
          makeRowVector({
              makeFlatVector<int64_t>(
                  rowsPerBatch,
                  [&](auto row) {
                    return (static_cast<int64_t>(batch) * rowsPerBatch + row) %
                        numKeys;
                  },
                  [&](auto row) { return nullKeys && row % 97 == 0; }),
              makeFlatVector<int64_t>(
                  rowsPerBatch,
                  [&](auto row) { return batch * 31 + row % 13; },
                  [](auto row) { return row % 11 == 3; }),
          }));
    }
    return vectors;
  }

  static int64_t statSum(
      const std::unordered_map<std::string, RuntimeMetric>& stats,
      std::string_view name) {
    auto it = stats.find(std::string(name));
    return it == stats.end() ? 0 : it->second.sum;
  }

  cudf_velox::CudfConfig savedConfig_;
};

// A final step appends its input to leaves, spills them after every input and
// drains each leaf from host memory.
TEST_F(GroupbySpillTest, finalStepSpillsAndDrainsFromHost) {
  const auto vectors = makeInput(12, 1'000, 5'000);
  createDuckDbTable(vectors);

  core::PlanNodeId finalId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(makeSpillableQueryCtx("groupby-spill-final"))
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .maxDrivers(1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0"}, {"sum(c1)", "count(c1)", "min(c1)", "max(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), count(c1), min(c1), max(c1) "
              "FROM tmp GROUP BY c0");

  const auto stats = toPlanStats(task->taskStats()).at(finalId).customStats;
  EXPECT_GT(statSum(stats, "cudfGroupbySelfSpills"), 0);
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kSpilledBytes), 0);
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kSpilledHostBytes), 0);
  EXPECT_EQ(
      statSum(stats, cudf_velox::CudfGroupby::kSpilledRows),
      statSum(stats, cudf_velox::CudfGroupby::kRestoredRows));
  EXPECT_GT(statSum(stats, "cudfGroupbyBulkDrainPartitions"), 0);
}

// With a small leaf limit the buffered state splits into hash partitions, and
// leaves that outgrew the limit in host memory are partitioned again when
// they are drained.
TEST_F(GroupbySpillTest, finalStepRepartitionsSpilledLeaves) {
  cudf_velox::CudfConfig::getInstance().batchSizeMaxThreshold = 64;
  cudf_velox::CudfConfig::getInstance().spillPartitionFanout = 4;
  const auto vectors = makeInput(32, 32, 20'000);
  createDuckDbTable(vectors);

  core::PlanNodeId finalId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(makeSpillableQueryCtx("groupby-spill-repartition"))
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 32)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .maxDrivers(1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)", "count(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), count(c1) FROM tmp GROUP BY c0");

  const auto stats = toPlanStats(task->taskStats()).at(finalId).customStats;
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kSpilledBytes), 0);
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kRestoredRows), 0);
  EXPECT_GT(statSum(stats, "cudfGroupbyDrainRepartitions"), 0);
}

// Every row has the same key, so hash partitioning cannot split the state
// when it outgrows the leaf limit of 8 rows. The leaf is aggregated in place
// instead, and the result is one group.
TEST_F(GroupbySpillTest, finalStepWithOneKey) {
  cudf_velox::CudfConfig::getInstance().batchSizeMaxThreshold = 8;
  const auto vectors = makeInput(24, 8, 1, false);
  createDuckDbTable(vectors);

  core::PlanNodeId finalId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(makeSpillableQueryCtx("groupby-spill-one-key"))
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 8)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .maxDrivers(1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)", "count(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), count(c1) FROM tmp GROUP BY c0");

  const auto stats = toPlanStats(task->taskStats()).at(finalId).customStats;
  EXPECT_GT(statSum(stats, "cudfGroupbyCompactedLeaves"), 0);
}

// A single step merges each input into its leaf and restores a spilled leaf
// when more input arrives for it.
TEST_F(GroupbySpillTest, singleStepRestoresOnInput) {
  const auto vectors = makeInput(10, 1'000, 3'000);
  createDuckDbTable(vectors);

  core::PlanNodeId aggregationId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(makeSpillableQueryCtx("groupby-spill-single"))
          .maxDrivers(1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"c0"}, {"sum(c1)", "count(c1)", "avg(c1)"})
                  .capturePlanNodeId(aggregationId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), count(c1), avg(c1) FROM tmp GROUP BY c0");

  const auto stats =
      toPlanStats(task->taskStats()).at(aggregationId).customStats;
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kSpilledBytes), 0);
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kRestoredLeaves), 0);
}

// String keys take the general spill path: their columns have children and
// are copied to host memory buffer by buffer.
TEST_F(GroupbySpillTest, finalStepWithStringKeys) {
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 8; ++batch) {
    vectors.push_back(
        makeRowVector({
            makeFlatVector<std::string>(
                1'000,
                [&](auto row) {
                  return fmt::format("key-{}", (batch * 1'000 + row) % 2'500);
                },
                [](auto row) { return row % 53 == 0; }),
            makeFlatVector<int64_t>(1'000, [&](auto row) { return row + batch; }),
        }));
  }
  createDuckDbTable(vectors);

  core::PlanNodeId finalId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .queryCtx(makeSpillableQueryCtx("groupby-spill-strings"))
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .maxDrivers(1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)", "count(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), count(c1) FROM tmp GROUP BY c0");

  const auto stats = toPlanStats(task->taskStats()).at(finalId).customStats;
  EXPECT_GT(statSum(stats, cudf_velox::CudfGroupby::kSpilledBytes), 0);
  EXPECT_EQ(
      statSum(stats, cudf_velox::CudfGroupby::kSpilledRows),
      statSum(stats, cudf_velox::CudfGroupby::kRestoredRows));
}

// A query without a GPU pool buffers one merged result as before and reports
// no spill counters.
TEST_F(GroupbySpillTest, queryWithoutGpuPoolDoesNotSpill) {
  const auto vectors = makeInput(6, 1'000, 2'000);
  createDuckDbTable(vectors);

  core::PlanNodeId finalId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .maxDrivers(1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)", "count(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), count(c1) FROM tmp GROUP BY c0");

  const auto stats = toPlanStats(task->taskStats()).at(finalId).customStats;
  EXPECT_EQ(statSum(stats, cudf_velox::CudfGroupby::kSpilledBytes), 0);
  EXPECT_EQ(statSum(stats, "cudfGroupbySelfSpills"), 0);
}

} // namespace facebook::velox::exec::test
