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
#include "velox/experimental/cudf/exec/CudfJoinBuild.h"
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/CudfSpillContext.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

// Runs hash joins whose build is split into hash partitions that are joined
// one after the other, with the probe rows of the later partitions parked in
// pinned host memory. Nothing here depends on debug-only test hooks: a small
// cudf.spill_join_partition_bytes makes a small build count as too large.
class HashJoinSpillTest : public OperatorTestBase {
 protected:
  static constexpr int64_t kQueryGpuBytes = 256LL << 20;
  static constexpr uint64_t kHostSpillBytes = 256ULL << 20;
  // The build below takes about 320 KB by the build estimate, so this splits
  // it into four partitions.
  static constexpr uint64_t kJoinPartitionBytes = 100'000;
  static constexpr int32_t kNumBuildRows = 10'000;

  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = cudf_velox::CudfConfig::getInstance();
    savedConfig_ = config;
    config.allowCpuFallback = false;
    config.spillHostBytes = kHostSpillBytes;
    config.spillJoinPartitionBytes = kJoinPartitionBytes;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance() = savedConfig_;
    OperatorTestBase::TearDown();
  }

  std::shared_ptr<core::QueryCtx> makeSpillableQueryCtx(
      const std::string& queryId) {
    auto queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId(queryId)
                        .build();
    cudf_velox::attachCudfMemoryPool(*queryCtx, kQueryGpuBytes);
    EXPECT_NE(cudf_velox::cudfSpillContext(*queryCtx), nullptr);
    return queryCtx;
  }

  // Build rows u0 = 0, 3, 6, ... with a payload. Keys are unique.
  std::vector<RowVectorPtr> makeBuildInput() {
    std::vector<RowVectorPtr> vectors;
    const int32_t rowsPerBatch = kNumBuildRows / 4;
    for (int32_t batch = 0; batch < 4; ++batch) {
      vectors.push_back(
          makeRowVector(
              {"u0", "u1"},
              {
                  makeFlatVector<int64_t>(
                      rowsPerBatch,
                      [&](auto row) {
                        return (static_cast<int64_t>(batch) * rowsPerBatch +
                                row) *
                            3;
                      }),
                  makeFlatVector<int64_t>(
                      rowsPerBatch, [&](auto row) { return batch + row * 7; }),
              }));
    }
    return vectors;
  }

  // Probe rows t0 = 0, 1, 2, ..., a third of which have a build match, with
  // some null keys.
  std::vector<RowVectorPtr> makeProbeInput() {
    std::vector<RowVectorPtr> vectors;
    const int32_t rowsPerBatch = 2'000;
    for (int32_t batch = 0; batch < 6; ++batch) {
      vectors.push_back(
          makeRowVector(
              {"t0", "t1"},
              {
                  makeFlatVector<int64_t>(
                      rowsPerBatch,
                      [&](auto row) {
                        return static_cast<int64_t>(batch) * rowsPerBatch +
                            row;
                      },
                      [](auto row) { return row % 101 == 0; }),
                  makeFlatVector<int64_t>(
                      rowsPerBatch, [&](auto row) { return row % 17; }),
              }));
    }
    return vectors;
  }

  // Runs the join of the probe and build inputs and checks its result
  // against DuckDB, and that the build was partitioned and probe rows went
  // through host memory.
  void testPartitionedJoin(
      core::JoinType joinType,
      const std::string& filter,
      const std::vector<std::string>& outputLayout,
      const std::string& referenceSql,
      int32_t numDrivers = 1) {
    const auto probeInput = makeProbeInput();
    const auto buildInput = makeBuildInput();
    // Each driver of a pipeline reads the whole input of its side.
    std::vector<RowVectorPtr> allProbe;
    std::vector<RowVectorPtr> allBuild;
    for (int32_t i = 0; i < numDrivers; ++i) {
      allProbe.insert(allProbe.end(), probeInput.begin(), probeInput.end());
      allBuild.insert(allBuild.end(), buildInput.begin(), buildInput.end());
    }
    createDuckDbTable("t", allProbe);
    createDuckDbTable("u", allBuild);

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId joinId;
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values(probeInput, numDrivers > 1)
                    .hashJoin(
                        {"t0"},
                        {"u0"},
                        PlanBuilder(planNodeIdGenerator)
                            .values(buildInput, numDrivers > 1)
                            .planNode(),
                        filter,
                        outputLayout,
                        joinType)
                    .capturePlanNodeId(joinId)
                    .planNode();

    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .queryCtx(
                        makeSpillableQueryCtx(
                            fmt::format(
                                "join-spill-{}-{}",
                                core::JoinTypeName::toName(joinType),
                                numDrivers)))
                    .maxDrivers(numDrivers)
                    .plan(plan)
                    .assertResults(referenceSql);

    const auto stats = toPlanStats(task->taskStats()).at(joinId).customStats;
    auto statSum = [&](std::string_view name) {
      auto it = stats.find(std::string(name));
      return it == stats.end() ? int64_t{0} : it->second.sum;
    };
    EXPECT_GE(statSum("cudfJoinSpillPartitions"), 2);
    EXPECT_GT(statSum(cudf_velox::CudfJoinBuild::kSpilledBytes), 0);
    EXPECT_GT(statSum(cudf_velox::CudfJoinBuild::kRestoredBytes), 0);
    EXPECT_GT(statSum("cudfJoinProbeSpilledBytes"), 0);
    EXPECT_EQ(
        statSum("cudfJoinProbeSpilledBytes"),
        statSum("cudfJoinProbeRestoredBytes"));
  }

  cudf_velox::CudfConfig savedConfig_;
};

TEST_F(HashJoinSpillTest, innerJoin) {
  testPartitionedJoin(
      core::JoinType::kInner,
      "",
      {"t0", "t1", "u1"},
      "SELECT t0, t1, u1 FROM t, u WHERE t0 = u0");
}

TEST_F(HashJoinSpillTest, innerJoinWithFilter) {
  testPartitionedJoin(
      core::JoinType::kInner,
      "t1 < u1",
      {"t0", "t1", "u1"},
      "SELECT t0, t1, u1 FROM t, u WHERE t0 = u0 AND t1 < u1");
}

TEST_F(HashJoinSpillTest, leftJoin) {
  testPartitionedJoin(
      core::JoinType::kLeft,
      "",
      {"t0", "t1", "u1"},
      "SELECT t0, t1, u1 FROM t LEFT JOIN u ON t0 = u0");
}

TEST_F(HashJoinSpillTest, leftSemiFilterJoin) {
  testPartitionedJoin(
      core::JoinType::kLeftSemiFilter,
      "",
      {"t0", "t1"},
      "SELECT t0, t1 FROM t WHERE EXISTS (SELECT 1 FROM u WHERE t0 = u0)");
}

TEST_F(HashJoinSpillTest, antiJoin) {
  testPartitionedJoin(
      core::JoinType::kAnti,
      "",
      {"t0", "t1"},
      "SELECT t0, t1 FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE t0 = u0)");
}

// Several probe drivers each park their own rows and take part in every
// round. Several build drivers hand their input to the one that partitions.
TEST_F(HashJoinSpillTest, innerJoinWithThreeDrivers) {
  testPartitionedJoin(
      core::JoinType::kInner,
      "",
      {"t0", "t1", "u1"},
      "SELECT t0, t1, u1 FROM t, u WHERE t0 = u0",
      3);
}

// A right join emits build rows, which needs the whole build at once. Its
// build is not partitioned, and the join still returns the right rows.
TEST_F(HashJoinSpillTest, rightJoinIsNotPartitioned) {
  const auto probeInput = makeProbeInput();
  const auto buildInput = makeBuildInput();
  createDuckDbTable("t", probeInput);
  createDuckDbTable("u", buildInput);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId joinId;
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeInput)
                  .hashJoin(
                      {"t0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildInput)
                          .planNode(),
                      "",
                      {"t1", "u0", "u1"},
                      core::JoinType::kRight)
                  .capturePlanNodeId(joinId)
                  .planNode();
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .queryCtx(makeSpillableQueryCtx("join-spill-right"))
                  .maxDrivers(1)
                  .plan(plan)
                  .assertResults(
                      "SELECT t1, u0, u1 FROM t RIGHT JOIN u ON t0 = u0");

  const auto stats = toPlanStats(task->taskStats()).at(joinId).customStats;
  EXPECT_EQ(stats.count("cudfJoinSpillPartitions"), 0);
}

} // namespace facebook::velox::exec::test
