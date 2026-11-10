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
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/ucx-exchange/ExchangeStore.h"
#include "velox/experimental/ucx-exchange/ExchangeStoreTables.h"

#include "velox/common/base/Fs.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::exec::test {
namespace {

using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;

// Writes a bucketed Presto temporary table through the TableWriter and reads
// its files back through table scans, with the table kept in the exchange
// store. The parameter is the store's device tier in bytes.
class ExchangeStoreTableTest : public CudfHiveConnectorTestBase,
                               public testing::WithParamInterface<uint64_t> {
 protected:
  static constexpr int kNumBuckets = 4;
  static constexpr vector_size_t kBatchRows = 1'000;
  static constexpr const char* kTable =
      "__presto_temporary_table_PARQUET_test_1";

  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    // The TableWriter has no GPU operator and runs on the CPU, as in Presto.
    CudfConfig::getInstance().allowCpuFallback = true;
    ucx_exchange::registerExchangeStoreTables(GetParam());
  }

  void TearDown() override {
    // The store is process-wide. A failed test must not leave data behind.
    ucx_exchange::ExchangeStore::instance().drop(kTable);
    ucx_exchange::unregisterExchangeStoreTables();
    CudfHiveConnectorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> makeInput() {
    std::vector<std::string> names;
    for (vector_size_t row = 0; row < kBatchRows; ++row) {
      names.push_back(fmt::format("name {}", row % 17));
    }
    std::vector<RowVectorPtr> input;
    for (int batch = 0; batch < 3; ++batch) {
      input.push_back(makeRowVector(
          {"key", "name", "value"},
          {
              makeFlatVector<int64_t>(
                  kBatchRows,
                  [batch](auto row) { return batch * kBatchRows + row; }),
              makeFlatVector<std::string>(names),
              makeFlatVector<double>(
                  kBatchRows, [](auto row) { return row * 0.5; }, nullEvery(7)),
          }));
    }
    return input;
  }

  core::PlanNodePtr scanPlan(const RowTypePtr& outputType, const RowTypePtr& tableType) {
    return PlanBuilder()
        .startTableScan()
        .tableName(kTable)
        .connectorId(kCudfHiveConnectorId)
        .outputType(outputType)
        .dataColumns(tableType)
        .endTableScan()
        .planNode();
  }
};

TEST_P(ExchangeStoreTableTest, readsEachBucketFromItsFile) {
  const auto input = makeInput();
  const auto tableType = asRowType(input[0]->type());
  auto directory = common::testutil::TempDirectoryPath::create();
  const auto tableDirectory = directory->getPath() + "/" + kTable;

  const auto writePlan = PlanBuilder()
                             .values(input)
                             .tableWrite(
                                 tableDirectory,
                                 {},
                                 kNumBuckets,
                                 {"key"},
                                 {},
                                 dwio::common::FileFormat::PARQUET,
                                 {},
                                 kCudfHiveConnectorId)
                             .planNode();
  AssertQueryBuilder(writePlan).copyResults(pool());
  auto& store = ucx_exchange::ExchangeStore::instance();
  // Each file's slices of the three batches are packed as one payload.
  EXPECT_EQ(store.stats().payloads, kNumBuckets);
  if (GetParam() == 0) {
    EXPECT_EQ(store.stats().deviceBytes, 0);
    EXPECT_GT(store.stats().hostBytes, 0);
  }

  // One placeholder file per bucket, as the coordinator expects to list.
  facebook::velox::connector::hive::HivePartitionFunction bucketFunction(
      kNumBuckets, std::vector<column_index_t>{0});
  std::vector<RowVectorPtr> output;
  int numFiles = 0;
  for (const auto& entry : fs::directory_iterator(tableDirectory)) {
    const auto name = entry.path().filename().string();
    const int bucket = std::stoi(name.substr(0, name.find('_')));
    SCOPED_TRACE(name);
    EXPECT_GT(fs::file_size(entry.path()), 0);
    EXPECT_LT(fs::file_size(entry.path()), 100);
    ++numFiles;

    auto split = facebook::velox::connector::hive::HiveConnectorSplitBuilder(entry.path().string())
                     .connectorId(kCudfHiveConnectorId)
                     .fileFormat(dwio::common::FileFormat::PARQUET)
                     .tableBucketNumber(bucket)
                     .build();
    const auto rows = AssertQueryBuilder(scanPlan(tableType, tableType))
                          .split(split)
                          .copyResults(pool());
    ASSERT_GT(rows->size(), 0);
    // Every row of the file belongs to the file's bucket.
    std::vector<uint32_t> buckets;
    bucketFunction.partition(*rows, buckets);
    for (vector_size_t row = 0; row < rows->size(); ++row) {
      ASSERT_EQ(buckets[row], bucket) << "row " << row;
    }
    output.push_back(rows);
  }
  EXPECT_EQ(numFiles, kNumBuckets);
  facebook::velox::exec::test::assertEqualResults(input, output);
  // Each file was read once, which takes its data out of the store.
  EXPECT_EQ(store.stats().payloads, 0);
  EXPECT_EQ(store.stats().deviceBytes + store.stats().hostBytes, 0);
}

TEST_P(ExchangeStoreTableTest, readsTheScanColumnsInTheScanOrder) {
  const auto input = makeInput();
  const auto tableType = asRowType(input[0]->type());
  auto directory = common::testutil::TempDirectoryPath::create();
  const auto tableDirectory = directory->getPath() + "/" + kTable;
  AssertQueryBuilder(PlanBuilder()
                         .values(input)
                         .tableWrite(
                             tableDirectory,
                             {},
                             1,
                             {"key"},
                             {},
                             dwio::common::FileFormat::PARQUET,
                             {},
                             kCudfHiveConnectorId)
                         .planNode())
      .copyResults(pool());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> splits;
  for (const auto& entry : fs::directory_iterator(tableDirectory)) {
    splits.push_back(
        facebook::velox::connector::hive::HiveConnectorSplitBuilder(entry.path().string())
            .connectorId(kCudfHiveConnectorId)
            .fileFormat(dwio::common::FileFormat::PARQUET)
            .tableBucketNumber(0)
            .build());
  }
  ASSERT_EQ(splits.size(), 1);
  const auto outputType = ROW({"value", "key"}, {DOUBLE(), BIGINT()});
  const auto rows = AssertQueryBuilder(scanPlan(outputType, tableType))
                        .splits(splits)
                        .copyResults(pool());
  std::vector<RowVectorPtr> expected;
  for (const auto& batch : input) {
    expected.push_back(
        makeRowVector({"value", "key"}, {batch->childAt(2), batch->childAt(0)}));
  }
  facebook::velox::exec::test::assertEqualResults(expected, {rows});
}

INSTANTIATE_TEST_SUITE_P(
    DeviceAndHostTiers,
    ExchangeStoreTableTest,
    testing::Values(uint64_t{1} << 30, uint64_t{0}),
    [](const testing::TestParamInfo<uint64_t>& info) {
      return info.param == 0 ? std::string{"hostTier"}
                             : std::string{"deviceTier"};
    });

} // namespace
} // namespace facebook::velox::cudf_velox::exec::test
