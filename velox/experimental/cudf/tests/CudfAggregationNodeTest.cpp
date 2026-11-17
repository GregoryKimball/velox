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
 */

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAggregationNode.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {
namespace {

// Appends the ids of the nodes of 'plan', sources first.
void collectNodeIds(
    const core::PlanNodePtr& plan,
    std::vector<core::PlanNodeId>& ids) {
  for (const auto& source : plan->sources()) {
    collectNodeIds(source, ids);
  }
  ids.push_back(plan->id());
}

// Appends the aggregations of 'plan' that exchange the compact state, sources
// first.
void collectCompactAggregations(
    const core::PlanNodePtr& plan,
    std::vector<const CudfAggregationNode*>& nodes) {
  for (const auto& source : plan->sources()) {
    collectCompactAggregations(source, nodes);
  }
  if (const auto* node = dynamic_cast<const CudfAggregationNode*>(plan.get())) {
    nodes.push_back(node);
  }
}

// Rewrites the FINAL aggregation 'plan' ends with and fails the test if the
// rewrite did not apply.
core::PlanNodePtr toCompactDecimalSum(const core::PlanNodePtr& plan) {
  auto rewritten = CudfAggregationNode::rewriteFinalAggregation(plan);
  EXPECT_NE(rewritten, plan) << "The compact decimal SUM rewrite did not apply";
  return rewritten;
}

class CudfAggregationNodeTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    savedConfig_ = CudfConfig::getInstance();
    CudfConfig::getInstance().allowCpuFallback = false;
    // Select and initialize a CUDA device before cuDF allocates.
    int numDevices{0};
    const auto status = cudaGetDeviceCount(&numDevices);
    if (status != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << static_cast<int>(status)
                   << " (" << cudaGetErrorString(status) << ")";
    }
    if (numDevices == 0) {
      GTEST_SKIP() << "No CUDA devices visible (check CUDA_VISIBLE_DEVICES)";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    CudfConfig::getInstance() = savedConfig_;
    exec::test::OperatorTestBase::TearDown();
  }

  // Rows of a key 'k', a short decimal 'd' with NULLs, a short decimal 'e' of
  // another scale, and a boolean 'm' to mask with.
  std::vector<RowVectorPtr> makeInput(int32_t numBatches) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t batch = 0; batch < numBatches; ++batch) {
      vectors.push_back(makeRowVector(
          {"k", "d", "e", "m"},
          {
              makeFlatVector<int64_t>({1, 1, 2, 2, 3}),
              makeNullableFlatVector<int64_t>(
                  {12'345, -2'500, 10'000, std::nullopt, std::nullopt},
                  DECIMAL(12, 2)),
              makeNullableFlatVector<int64_t>(
                  {1, std::nullopt, 30'000, -40'000, 55'555}, DECIMAL(15, 4)),
              makeFlatVector<bool>({true, false, true, true, false}),
          }));
    }
    return vectors;
  }

  CudfConfig savedConfig_;
};

TEST_F(CudfAggregationNodeTest, rewriteKeepsPlanShape) {
  auto input = makeRowVector(
      {"k", "d", "w"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>({1, 2}, DECIMAL(30, 2)),
      });
  auto original =
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation(
              {"k"},
              {"sum(d) AS s", "count(d) AS c", "sum(w) AS t", "avg(d) AS a"})
          .localPartition({"k"})
          .finalAggregation()
          .planNode();

  auto rewritten = CudfAggregationNode::rewriteFinalAggregation(original);
  ASSERT_NE(rewritten, original);
  const auto* finalNode =
      dynamic_cast<const CudfAggregationNode*>(rewritten.get());
  ASSERT_NE(finalNode, nullptr);
  // The same node ids and the same result type: only the state column changes.
  EXPECT_EQ(finalNode->id(), original->id());
  EXPECT_TRUE(*finalNode->outputType() == *original->outputType());
  EXPECT_TRUE(finalNode->usesCompactDecimalSum(0));
  // COUNT, the long decimal SUM and the decimal AVG keep their state.
  EXPECT_FALSE(finalNode->usesCompactDecimalSum(1));
  EXPECT_FALSE(finalNode->usesCompactDecimalSum(2));
  EXPECT_FALSE(finalNode->usesCompactDecimalSum(3));

  const auto& local = finalNode->sources()[0];
  EXPECT_EQ(local->id(), original->sources()[0]->id());
  const auto* partialNode =
      dynamic_cast<const CudfAggregationNode*>(local->sources()[0].get());
  ASSERT_NE(partialNode, nullptr);
  EXPECT_EQ(partialNode->id(), original->sources()[0]->sources()[0]->id());
  EXPECT_TRUE(partialNode->usesCompactDecimalSum(0));
  EXPECT_FALSE(partialNode->usesCompactDecimalSum(1));

  const auto& originalState = original->sources()[0]->outputType();
  const auto& compactState = local->outputType();
  EXPECT_EQ(compactState->names(), originalState->names());
  EXPECT_TRUE(*originalState->findChild("s") == *VARBINARY());
  EXPECT_TRUE(*compactState->findChild("s") == *DECIMAL(38, 2));
  for (const auto* name : {"k", "c", "t", "a"}) {
    EXPECT_TRUE(
        *compactState->findChild(name) == *originalState->findChild(name))
        << name;
  }

  // The rewrite does not apply twice.
  EXPECT_EQ(CudfAggregationNode::rewriteFinalAggregation(rewritten), rewritten);
}

TEST_F(CudfAggregationNodeTest, rewriteSkipsIneligiblePlans) {
  auto input = makeRowVector(
      {"k", "d", "w", "m"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>({1, 2}, DECIMAL(30, 2)),
          makeFlatVector<bool>({true, false}),
      });
  const auto expectUnchanged = [](const core::PlanNodePtr& plan) {
    EXPECT_EQ(CudfAggregationNode::rewriteFinalAggregation(plan), plan)
        << plan->toString(true, true);
  };

  // A global aggregation has a single state row.
  expectUnchanged(
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({}, {"sum(d) AS s"})
          .finalAggregation()
          .planNode());
  // A sum of long decimals can overflow 128 bits, which the overflow word of
  // the VARBINARY state records.
  expectUnchanged(
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({"k"}, {"sum(w) AS s"})
          .finalAggregation()
          .planNode());
  // A masked aggregate.
  expectUnchanged(
      exec::test::PlanBuilder()
          .values({input})
          .partialAggregation({"k"}, {"sum(d) AS s"}, {"m"})
          .finalAggregation()
          .planNode());
  // A single-step aggregation exchanges no state.
  expectUnchanged(
      exec::test::PlanBuilder()
          .values({input})
          .singleAggregation({"k"}, {"sum(d) AS s"})
          .planNode());
  // Not an aggregation.
  expectUnchanged(exec::test::PlanBuilder().values({input}).planNode());

  // Nothing keeps a compact aggregation off the CPU when cuDF is off.
  const auto eligible = exec::test::PlanBuilder()
                            .values({input})
                            .partialAggregation({"k"}, {"sum(d) AS s"})
                            .finalAggregation()
                            .planNode();
  CudfConfig::getInstance().enabled = false;
  expectUnchanged(eligible);
  CudfConfig::getInstance().enabled = true;
  unregisterCudf();
  expectUnchanged(eligible);
  registerCudf();
  EXPECT_NE(CudfAggregationNode::rewriteFinalAggregation(eligible), eligible);
}

TEST_F(CudfAggregationNodeTest, constructorRejectsStatesItDoesNotSupport) {
  auto input = makeRowVector(
      {"k", "d", "w"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>({1, 2}, DECIMAL(30, 2)),
      });
  const auto asAggregation = [](const core::PlanNodePtr& plan) {
    return std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  };

  {
    // The call still has the VARBINARY state type.
    const auto partial = asAggregation(
        exec::test::PlanBuilder()
            .values({input})
            .partialAggregation({"k"}, {"sum(d) AS s"})
            .planNode());
    VELOX_ASSERT_USER_THROW(
        CudfAggregationNode(*partial, {DECIMAL(38, 2)}),
        "Compact decimal SUM result must have the type of the state: "
        "VARBINARY");
    VELOX_ASSERT_USER_THROW(
        CudfAggregationNode(*partial, {DECIMAL(38, 4)}),
        "Compact decimal SUM state must be DECIMAL(38, scale of the input): "
        "DECIMAL(38, 4)");
    VELOX_ASSERT_USER_THROW(
        CudfAggregationNode(*partial, {nullptr}),
        "CudfAggregationNode must have an aggregate with the compact state");
  }
  {
    const auto partial = asAggregation(
        exec::test::PlanBuilder()
            .values({input})
            .partialAggregation({"k"}, {"sum(w) AS s"})
            .planNode());
    VELOX_ASSERT_USER_THROW(
        CudfAggregationNode(*partial, {DECIMAL(38, 2)}),
        "Aggregate is not eligible for the compact decimal SUM state");
  }
  {
    const auto single = asAggregation(
        exec::test::PlanBuilder()
            .values({input})
            .singleAggregation({"k"}, {"sum(d) AS s"})
            .planNode());
    VELOX_ASSERT_USER_THROW(
        CudfAggregationNode(*single, {DECIMAL(38, 2)}),
        "A single-step aggregation has no intermediate state to make compact");
  }
}

// rewritePlan() reaches the aggregations below joins, projections and other
// aggregations and leaves every node id and the result type as they were.
TEST_F(CudfAggregationNodeTest, rewritePlanReachesNestedAggregations) {
  const auto vectors = makeInput(3);
  createDuckDbTable(vectors);
  auto names = makeRowVector(
      {"n", "label"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 20}),
      });
  createDuckDbTable("names", {names});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto build =
      exec::test::PlanBuilder(planNodeIdGenerator).values({names}).planNode();
  auto original = exec::test::PlanBuilder(planNodeIdGenerator)
                      .values(vectors)
                      .partialAggregation({"k"}, {"sum(d) AS s"})
                      .localPartition({"k"})
                      .finalAggregation()
                      .filter("k < 100")
                      .project({"k", "s"})
                      .hashJoin({"k"}, {"n"}, build, "", {"label", "s"})
                      .partialAggregation({"label"}, {"sum(s) AS total"})
                      .localPartition(std::vector<std::string>{})
                      .finalAggregation()
                      .orderBy({"label"}, false)
                      .planNode();

  const auto rewritten = CudfAggregationNode::rewritePlan(original);
  ASSERT_NE(rewritten, original);
  std::vector<const CudfAggregationNode*> compactNodes;
  collectCompactAggregations(rewritten, compactNodes);
  // The PARTIAL and the FINAL aggregation over 'd'. The sum over 's', a long
  // decimal, keeps its VARBINARY state.
  ASSERT_EQ(compactNodes.size(), 2);
  EXPECT_EQ(compactNodes[0]->step(), core::AggregationNode::Step::kPartial);
  EXPECT_EQ(compactNodes[1]->step(), core::AggregationNode::Step::kFinal);

  std::vector<core::PlanNodeId> originalIds;
  collectNodeIds(original, originalIds);
  std::vector<core::PlanNodeId> rewrittenIds;
  collectNodeIds(rewritten, rewrittenIds);
  EXPECT_EQ(rewrittenIds, originalIds);
  EXPECT_TRUE(*rewritten->outputType() == *original->outputType());

  // A plan without an eligible aggregation comes back as it is.
  EXPECT_EQ(CudfAggregationNode::rewritePlan(build), build);
  // So does a plan that was rewritten before.
  EXPECT_EQ(CudfAggregationNode::rewritePlan(rewritten), rewritten);

  exec::test::AssertQueryBuilder(rewritten, duckDbQueryRunner_)
      .maxDrivers(3)
      .assertResults(
          "SELECT label, sum(s) FROM ("
          "  SELECT k, sum(d) AS s FROM tmp GROUP BY k"
          ") AS sums, names WHERE k = n GROUP BY label ORDER BY label");
}

TEST_F(CudfAggregationNodeTest, nullsAndWideKeys) {
  // Keys above 2^32. Two of them differ only above bit 32, so a key cut to 32
  // bits would merge their groups.
  const int64_t base{5'000'000'000LL};
  const int64_t high{base + (1LL << 32) + 1};
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 4; ++batch) {
    vectors.push_back(makeRowVector(
        {"k", "d"},
        {
            makeNullableFlatVector<int64_t>(
                {base + 1, base + 1, base + 2, high, base + 3, std::nullopt}),
            makeNullableFlatVector<int64_t>(
                {12'345, -2'500, std::nullopt, 700, std::nullopt, 5},
                DECIMAL(12, 2)),
        }));
  }
  // The groups base + 2 and base + 3 have only NULL input, so their sum is
  // NULL. The NULL key is a group of its own.
  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeNullableFlatVector<int64_t>(
              {base + 1, base + 2, base + 3, high, std::nullopt}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(4 * (12'345 - 2'500)),
               std::nullopt,
               std::nullopt,
               static_cast<int128_t>(4 * 700),
               static_cast<int128_t>(4 * 5)},
              DECIMAL(38, 2)),
      });

  for (const bool withLocalPartition : {false, true}) {
    SCOPED_TRACE(fmt::format("withLocalPartition {}", withLocalPartition));
    exec::test::PlanBuilder builder;
    builder.values(vectors).partialAggregation({"k"}, {"sum(d) AS s"});
    if (withLocalPartition) {
      builder.localPartition({"k"});
    }
    const auto plan =
        toCompactDecimalSum(builder.finalAggregation().planNode());
    // The PARTIAL step emits its states after every input, so the FINAL step
    // merges four compact states for each group.
    auto result =
        exec::test::AssertQueryBuilder(plan)
            .maxDrivers(withLocalPartition ? 3 : 1)
            .config(CudfFromVelox::kGpuBatchSizeRows, 6)
            .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
            .copyResults(pool());
    EXPECT_TRUE(*result->type() == *expected->type());
    exec::test::assertEqualResults({expected}, {result});
  }
}

TEST_F(CudfAggregationNodeTest, accumulatesIn128Bits) {
  // 30 values of 10^18 - 1 add up to about 3 * 10^19, beyond 64 bits. The
  // compact state carries the 128-bit sum through PARTIAL and FINAL.
  const int64_t maxShort{999'999'999'999'999'999LL};
  std::vector<RowVectorPtr> vectors;
  for (int32_t batch = 0; batch < 2; ++batch) {
    std::vector<int64_t> keys;
    std::vector<int64_t> values;
    for (int32_t row = 0; row < 15; ++row) {
      keys.push_back(1);
      values.push_back(maxShort);
      keys.push_back(2);
      values.push_back(-maxShort);
    }
    vectors.push_back(makeRowVector(
        {"k", "d"},
        {
            makeFlatVector<int64_t>(keys),
            makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
        }));
  }
  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int128_t>(
              {static_cast<int128_t>(maxShort) * 30,
               static_cast<int128_t>(-maxShort) * 30},
              DECIMAL(38, 0)),
      });

  const auto plan = toCompactDecimalSum(
      exec::test::PlanBuilder()
          .values(vectors)
          .partialAggregation({"k"}, {"sum(d) AS s"})
          .finalAggregation()
          .planNode());
  auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool());
  exec::test::assertEqualResults({expected}, {result});
}

// Two compact sums of different scale next to a count, a minimum and a
// masked sum, which keep their ordinary states.
TEST_F(CudfAggregationNodeTest, withOtherAggregates) {
  const auto vectors = makeInput(3);
  createDuckDbTable(vectors);

  const auto plan = toCompactDecimalSum(
      exec::test::PlanBuilder()
          .values(vectors)
          .partialAggregation(
              {"k"},
              {"sum(d) AS s",
               "count(d) AS c",
               "sum(e) AS t",
               "min(d) AS a",
               "sum(d) AS f"},
              {"", "", "", "", "m"})
          .finalAggregation()
          .planNode());
  const auto* finalNode = dynamic_cast<const CudfAggregationNode*>(plan.get());
  ASSERT_NE(finalNode, nullptr);
  EXPECT_TRUE(finalNode->usesCompactDecimalSum(0));
  EXPECT_FALSE(finalNode->usesCompactDecimalSum(1));
  EXPECT_TRUE(finalNode->usesCompactDecimalSum(2));
  EXPECT_FALSE(finalNode->usesCompactDecimalSum(3));
  EXPECT_FALSE(finalNode->usesCompactDecimalSum(4));

  exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(CudfFromVelox::kGpuBatchSizeRows, 5)
      .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
      .assertResults(
          "SELECT k, sum(d), count(d), sum(e), min(d), "
          "sum(d) FILTER (WHERE m) FROM tmp GROUP BY k");
}

// A decimal average keeps the VARBINARY state next to the compact sum of the
// same column. DuckDB's average of a DECIMAL is a DOUBLE, so the reference is
// the same plan with ordinary states.
TEST_F(CudfAggregationNodeTest, withDecimalAverage) {
  const auto vectors = makeInput(3);
  const auto original =
      exec::test::PlanBuilder()
          .values(vectors)
          .partialAggregation(
              {"k"},
              {"sum(d) AS s", "avg(d) AS a", "sum(e) AS t", "avg(e) AS b"})
          .finalAggregation()
          .planNode();
  const auto compact = toCompactDecimalSum(original);
  const auto run = [&](const core::PlanNodePtr& plan) {
    return exec::test::AssertQueryBuilder(plan)
        .config(CudfFromVelox::kGpuBatchSizeRows, 5)
        .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
        .copyResults(pool());
  };
  const auto expected = run(original);
  const auto result = run(compact);
  ASSERT_EQ(expected->size(), 3);
  exec::test::assertEqualResults({expected}, {result});
}

TEST_F(CudfAggregationNodeTest, emptyInput) {
  const auto vectors = makeInput(2);
  createDuckDbTable(vectors);

  const auto plan = toCompactDecimalSum(
      exec::test::PlanBuilder()
          .values(vectors)
          .filter("k < 0")
          .partialAggregation({"k"}, {"sum(d) AS s"})
          .finalAggregation()
          .planNode());
  exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) FROM tmp WHERE k < 0 GROUP BY k");
}

// The streaming group-by decodes the VARBINARY state. A FINAL step with the
// compact state takes the other group-by path and gives the same result.
TEST_F(CudfAggregationNodeTest, streamingGroupbyIsNotUsed) {
  CudfConfig::getInstance().streamingGroupbyEnabled = true;
  const auto vectors = makeInput(4);
  createDuckDbTable(vectors);

  const auto plan = toCompactDecimalSum(
      exec::test::PlanBuilder()
          .values(vectors)
          .partialAggregation({"k"}, {"sum(d) AS s", "sum(e) AS t"})
          .finalAggregation()
          .planNode());
  auto task =
      exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(1)
          .config(CudfFromVelox::kGpuBatchSizeRows, 5)
          .config(core::QueryConfig::kMaxPartialAggregationMemory, 1)
          .assertResults("SELECT k, sum(d), sum(e) FROM tmp GROUP BY k");
  const auto stats = exec::toPlanStats(task->taskStats());
  const auto& customStats = stats.at(plan->id()).customStats;
  EXPECT_EQ(customStats.count(std::string{kStreamingGroupbyUsedStat}), 0);
}

// The CPU sum function would write the VARBINARY state into a column the
// plan declares as DECIMAL(38, 2). The query fails instead of falling back.
TEST_F(CudfAggregationNodeTest, rejectsCpuAggregation) {
  const auto plan = toCompactDecimalSum(
      exec::test::PlanBuilder()
          .values(makeInput(1))
          .partialAggregation({"k"}, {"sum(d) AS s"})
          .finalAggregation()
          .planNode());
  VELOX_ASSERT_USER_THROW(
      exec::test::AssertQueryBuilder(plan)
          .config(CudfConfig::kCudfEnabled, "false")
          .config(CudfConfig::kCudfAllowCpuFallback, "true")
          .copyResults(pool()),
      "Compact decimal SUM state requires cuDF to be enabled");
}

} // namespace
} // namespace facebook::velox::cudf_velox
