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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/ucx-exchange/ExchangeStore.h"
#include "velox/experimental/ucx-exchange/ExchangeStoreTables.h"

#include "velox/common/base/Fs.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <optional>
#include <set>
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
    // The store encodes columns for fused FOR, the codec of the best runs.
    CudfConfig::getInstance().exchangeCompression = "fused-for-bitpacked";
    ucx_exchange::registerExchangeStoreTables(GetParam(), 0);
  }

  void TearDown() override {
    // The store is process-wide. A failed test must not leave data behind.
    ucx_exchange::ExchangeStore::instance().drop(kTable);
    ucx_exchange::unregisterExchangeStoreTables();
    CudfConfig::getInstance().exchangeCompression = "none";
    CudfHiveConnectorTestBase::TearDown();
  }

  // Columns the store encodes as the scan cache does: ascending keys as
  // differences, strings and decimals with few distinct values as dictionary
  // indices, and doubles with nulls as they are.
  std::vector<RowVectorPtr> makeInput() {
    std::vector<std::string> names;
    for (vector_size_t row = 0; row < kBatchRows; ++row) {
      names.push_back(fmt::format("name {}", row % 17));
    }
    std::vector<RowVectorPtr> input;
    for (int batch = 0; batch < 3; ++batch) {
      input.push_back(makeRowVector(
          {"key", "name", "quantity", "total", "value"},
          {
              makeFlatVector<int64_t>(
                  kBatchRows,
                  [batch](auto row) {
                    return 1'000'000'000'000 + (batch * kBatchRows + row) * 3;
                  }),
              makeFlatVector<std::string>(names),
              makeFlatVector<int64_t>(
                  kBatchRows,
                  [](auto row) { return (row * 7 % 50 + 1) * 100; },
                  nullptr,
                  DECIMAL(12, 2)),
              makeFlatVector<int128_t>(
                  kBatchRows,
                  [](auto row) {
                    return static_cast<int128_t>(row % 7) *
                        1'000'000'000'000'000'000;
                  },
                  nullptr,
                  DECIMAL(38, 2)),
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
  std::shared_ptr<facebook::velox::exec::Task> writeTask;
  // Batches reach the writer as they are, so that it collects them.
  AssertQueryBuilder(writePlan)
      .config(CudfFromVelox::kGpuBatchSizeRows, std::to_string(kBatchRows))
      .copyResults(pool(), writeTask);
  // The TableWriter kept its GPU input and the sink bucketed it on the GPU.
  const auto writeStats = facebook::velox::exec::toPlanStats(
      writeTask->taskStats());
  const auto& tableWriteStats = writeStats.at(writePlan->id()).customStats;
  ASSERT_EQ(tableWriteStats.count("exchangeStoreGpuBatches"), 1);
  EXPECT_GT(tableWriteStats.at("exchangeStoreGpuBatches").sum, 0);
  // The writer reports what it stored in which tier, and how it encoded the
  // columns: per batch, the key as differences and the name, quantity and
  // total as dictionaries.
  EXPECT_GT(tableWriteStats.at("exchangeStoreLogicalBytes").sum, 0);
  // The writer collected the GPU batches it got and packed them together.
  EXPECT_EQ(
      tableWriteStats.at("exchangeStoreGpuInputBatches").sum, input.size());
  const auto numBatches = tableWriteStats.at("exchangeStoreGpuBatches").sum;
  EXPECT_EQ(numBatches, 1);
  EXPECT_EQ(tableWriteStats.at("exchangeStoreDeltaColumns").sum, numBatches);
  EXPECT_EQ(
      tableWriteStats.at("exchangeStoreDictionaryColumns").sum, 3 * numBatches);
  EXPECT_GT(tableWriteStats.at("exchangeStoreKeysBytes").sum, 0);
  const auto storedIn = [&](const char* name) {
    return tableWriteStats.count(name) ? tableWriteStats.at(name).sum : 0;
  };
  if (GetParam() == 0) {
    EXPECT_EQ(storedIn("exchangeStoreDeviceBytes"), 0);
    EXPECT_GT(storedIn("exchangeStoreHostBytes"), 0);
  } else {
    EXPECT_GT(storedIn("exchangeStoreDeviceBytes"), 0);
    EXPECT_EQ(storedIn("exchangeStoreHostBytes"), 0);
  }
  auto& store = ucx_exchange::ExchangeStore::instance();
  EXPECT_GE(store.stats().payloads, kNumBuckets);
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
    std::shared_ptr<facebook::velox::exec::Task> scanTask;
    const auto scan = scanPlan(tableType, tableType);
    const auto rows =
        AssertQueryBuilder(scan).split(split).copyResults(pool(), scanTask);
    ASSERT_GT(rows->size(), 0);
    // The scan reports what it read from which tier when it closes.
    ASSERT_TRUE(facebook::velox::exec::test::waitForTaskCompletion(
        scanTask.get(), 10'000'000));
    const auto scanPlanStats =
        facebook::velox::exec::toPlanStats(scanTask->taskStats());
    const auto& scanStats = scanPlanStats.at(scan->id()).customStats;
    const auto readFrom = [&](const char* name) {
      return scanStats.count(name) ? scanStats.at(name).sum : 0;
    };
    EXPECT_GT(
        readFrom("exchangeStoreReadDeviceBytes") +
            readFrom("exchangeStoreReadHostBytes"),
        0);
    if (GetParam() == 0) {
      EXPECT_EQ(readFrom("exchangeStoreReadDeviceBytes"), 0);
    }
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

TEST_P(ExchangeStoreTableTest, readsEveryTableBucketOfAReadBucketFromOneSplit) {
  const auto input = makeInput();
  const auto tableType = asRowType(input[0]->type());
  auto directory = common::testutil::TempDirectoryPath::create();
  const auto tableDirectory = directory->getPath() + "/" + kTable;
  AssertQueryBuilder(PlanBuilder()
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
                         .planNode())
      .config(CudfFromVelox::kGpuBatchSizeRows, std::to_string(kBatchRows))
      .copyResults(pool());

  // The coordinator reads the table as 2 buckets, each the union of the table
  // buckets with its remainder, and lists one split per table bucket.
  constexpr int kReadBucketCount = 2;
  const auto split = [&](int bucket) {
    return facebook::velox::connector::hive::HiveConnectorSplitBuilder(
               fmt::format("{}/{:06d}_0_query", tableDirectory, bucket))
        .connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .tableBucketNumber(bucket)
        .customSplitInfo(
            {{std::string{ucx_exchange::kReadBucketCountSplitInfo},
              std::to_string(kReadBucketCount)},
             {std::string{ucx_exchange::kTableBucketCountSplitInfo},
              std::to_string(kNumBuckets)}})
        .build();
  };
  facebook::velox::connector::hive::HivePartitionFunction bucketFunction(
      kNumBuckets, std::vector<column_index_t>{0});
  const auto scan = scanPlan(tableType, tableType);
  std::vector<RowVectorPtr> output;
  for (int readBucket = 0; readBucket < kReadBucketCount; ++readBucket) {
    SCOPED_TRACE(fmt::format("read bucket {}", readBucket));
    // The first split of the read bucket reads all of it.
    const auto rows =
        AssertQueryBuilder(scan).split(split(readBucket)).copyResults(pool());
    std::vector<uint32_t> buckets;
    bucketFunction.partition(*rows, buckets);
    std::set<uint32_t> bucketsRead(buckets.begin(), buckets.end());
    // gmock is not available to this test (gtest comes from the dependencies).
    EXPECT_EQ(
        bucketsRead,
        (std::set<uint32_t>{
            static_cast<uint32_t>(readBucket),
            static_cast<uint32_t>(readBucket + kReadBucketCount)}));
    output.push_back(rows);
    // The read bucket's other split finds its table bucket read already.
    EXPECT_EQ(
        AssertQueryBuilder(scan)
            .split(split(readBucket + kReadBucketCount))
            .copyResults(pool())
            ->size(),
        0);
  }
  facebook::velox::exec::test::assertEqualResults(input, output);
  EXPECT_EQ(ucx_exchange::ExchangeStore::instance().stats().payloads, 0);
}

TEST_P(ExchangeStoreTableTest, namesTheTableAnExchangeWritesUnchanged) {
  CudfConfig::getInstance().exchangeStore = true;
  SCOPE_EXIT {
    CudfConfig::getInstance().exchangeStore = false;
  };
  // A materialized exchange's writer fragment: the received rows go to the
  // temporary table's writer through a renaming Project and a local exchange.
  const auto exchangeType = ROW({"k", "v"}, {BIGINT(), DOUBLE()});
  const std::string directory = "/data/" + std::string{kTable};
  core::PlanNodeId exchangeId;
  const auto writerPlan = [&](const std::vector<std::string>& projections,
                              const std::string& tableDirectory) {
    return PlanBuilder()
        .exchange(exchangeType, "Presto", std::string{core::TransportKind::kUcx})
        .capturePlanNodeId(exchangeId)
        .project(projections)
        .localPartition(std::vector<std::string>{})
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
  };
  const auto queryCtx = [](bool readByBucket) {
    std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
        sessions;
    sessions.emplace(
        kCudfHiveConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {"stream_temporary_tables", readByBucket ? "true" : "false"}}));
    return core::QueryCtx::create(nullptr, core::QueryConfig{{}}, sessions);
  };

  const auto renamed = writerPlan({"k AS key", "v AS value"}, directory);
  EXPECT_EQ(
      ucx_exchange::receivingTableInStore(*renamed, exchangeId, *queryCtx(true)),
      std::optional<std::string>{kTable});
  // Without reading by bucket, every bucket needs its file from the writer.
  EXPECT_EQ(
      ucx_exchange::receivingTableInStore(
          *renamed, exchangeId, *queryCtx(false)),
      std::nullopt);
  // A Project that moves columns changes the rows the writer gets.
  EXPECT_EQ(
      ucx_exchange::receivingTableInStore(
          *writerPlan({"v AS value", "k AS key"}, directory),
          exchangeId,
          *queryCtx(true)),
      std::nullopt);
  // A table that is not temporary is written as files.
  EXPECT_EQ(
      ucx_exchange::receivingTableInStore(
          *writerPlan({"k AS key", "v AS value"}, "/data/orders"),
          exchangeId,
          *queryCtx(true)),
      std::nullopt);
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
  // A column of differences, a dictionary column and a column of values.
  const auto outputType = ROW(
      {"value", "key", "name"}, {DOUBLE(), BIGINT(), VARCHAR()});
  const auto rows = AssertQueryBuilder(scanPlan(outputType, tableType))
                        .splits(splits)
                        .copyResults(pool());
  std::vector<RowVectorPtr> expected;
  for (const auto& batch : input) {
    expected.push_back(makeRowVector(
        {"value", "key", "name"},
        {batch->childAt(4), batch->childAt(0), batch->childAt(1)}));
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
