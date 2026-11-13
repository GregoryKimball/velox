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
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox::cudf_velox {
namespace {

// Plans the compact aggregation state rewrite produces; nothing runs on the
// device.
class CudfPlanRewriterTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    parse::registerTypeResolver();
    aggregate::prestosql::registerAllAggregateFunctions();
    CudfConfig::getInstance().compactAggregationStateEnabled = true;
  }

  void TearDown() override {
    CudfConfig::getInstance().compactAggregationStateEnabled = false;
    exec::test::OperatorTestBase::TearDown();
  }
};

TEST_F(CudfPlanRewriterTest, keepsVarbinaryStateAcrossMaterializedExchange) {
  // A materialized exchange writes PARTIAL state into a table, through a
  // fragment that cannot tell state columns from others, and a later fragment
  // reads it back for the FINAL aggregation. Both ends keep the VARBINARY
  // state: the producer whose output a connector partitions (Hive buckets),
  // and the consumer whose state comes from a table scan. An INTERMEDIATE
  // aggregation has one state type for its input and its output, so at either
  // end it fails instead of choosing one.
  auto input = makeRowVector(
      {"k", "d"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto bucketed = std::make_shared<connector::hive::HivePartitionFunctionSpec>(
      2, std::vector<column_index_t>{0}, std::vector<VectorPtr>{});
  auto toBuckets = [&](const core::PlanNodePtr& source) {
    auto output = exec::test::PlanBuilder(source, ids)
                      .partitionedOutput(
                          {"k"},
                          2,
                          false,
                          {},
                          "Presto",
                          std::string{core::TransportKind::kUcx})
                      .planNode();
    return core::PartitionedOutputNode::Builder(
               *std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
                   output))
        .partitionFunctionSpec(bucketed)
        .build();
  };
  auto partial = exec::test::PlanBuilder(ids)
                     .values({input})
                     .partialAggregation({"k"}, {"sum(d) AS s", "avg(d) AS a"})
                     .planNode();

  // Hash partitioning (an exchange that is not materialized) keeps sending the
  // compact state.
  auto hashed = CudfPlanRewriter::rewrite(
      exec::test::PlanBuilder(partial, ids)
          .partitionedOutput(
              {"k"},
              2,
              false,
              {},
              "Presto",
              std::string{core::TransportKind::kUcx})
          .planNode());
  EXPECT_EQ(*hashed->outputType()->childAt(1), *DECIMAL(38, 2));

  auto producer = toBuckets(partial);
  auto gpuProducer = CudfPlanRewriter::rewrite(producer);
  EXPECT_EQ(*gpuProducer->outputType(), *producer->outputType());
  EXPECT_EQ(*gpuProducer->outputType()->childAt(1), *VARBINARY());

  const auto stateType = asRowType(partial->outputType());
  auto consumer =
      exec::test::PlanBuilder(ids)
          .tableScan(stateType)
          .localPartition({"k"})
          .finalAggregation(
              {"k"},
              {"sum(s) AS s", "avg(a) AS a"},
              {{DECIMAL(12, 2)}, {DECIMAL(12, 2)}})
          .planNode();
  auto gpuConsumer = CudfPlanRewriter::rewrite(consumer);
  EXPECT_EQ(*gpuConsumer->outputType(), *consumer->outputType());
  EXPECT_EQ(
      dynamic_cast<const CudfAggregationNode*>(gpuConsumer.get()), nullptr);
  EXPECT_EQ(
      *gpuConsumer->sources()[0]->outputType()->findChild("s"), *VARBINARY());

  auto intermediate = exec::test::PlanBuilder(partial, ids)
                          .intermediateAggregation()
                          .planNode();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(toBuckets(intermediate)),
      "materialized exchange");
  auto scan = exec::test::PlanBuilder(ids).tableScan(stateType).planNode();
  auto intermediateFromTable =
      core::AggregationNode::Builder(
          *std::dynamic_pointer_cast<const core::AggregationNode>(intermediate))
          .source(scan)
          .build();
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(
          exec::test::PlanBuilder(intermediateFromTable, ids)
              .partitionedOutput(
                  {"k"},
                  2,
                  false,
                  {},
                  "Presto",
                  std::string{core::TransportKind::kUcx})
              .planNode()),
      "materialized exchange");
}

} // namespace
} // namespace facebook::velox::cudf_velox
