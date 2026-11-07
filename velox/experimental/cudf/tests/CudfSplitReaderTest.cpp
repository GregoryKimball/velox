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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilterAst();
  }

  cudf::ast::expression const* splitFilter() const {
    return pushdownFilter();
  }

  bool hasSplitFilter() const {
    return hasSplitSpecificPushdownFilter();
  }

 protected:
  void prepareSplitInternal(
      dwio::common::RuntimeStats& /*runtimeStats*/) override {}
};

class CudfSplitReaderTest : public ::facebook::velox::cudf_velox::exec::test::
                                CudfHiveConnectorTestBase {};

TEST_F(CudfSplitReaderTest, dynamicFilterChecksParsedIntegerSchema) {
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx context(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query",
      "task",
      "scan",
      0,
      "");
  FileHandleFactory files(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto check = [&](const VectorPtr& values,
                   const TypePtr& declaredType,
                   bool matches,
                   bool statistics = true) {
    SCOPED_TRACE(
        fmt::format(
            "file: {}, declared: {}",
            values->type()->toString(),
            declaredType->toString()));
    auto file = common::testutil::TempFilePath::create();
    auto data = makeRowVector({"c0"}, {values});
    if (statistics) {
      writeToFile(file->getPath(), data);
    } else {
      auto stream = cudf::get_default_stream();
      auto table = with_arrow::toCudfTable(
          data, pool_.get(), stream, cudf::get_current_device_resource_ref());
      cudf::io::table_input_metadata metadata(table->view());
      metadata.column_metadata[0].set_name("c0");
      auto options =
          cudf::io::parquet_writer_options::builder(
              cudf::io::sink_info(file->getPath()), table->view())
              .metadata(metadata)
              .stats_level(cudf::io::statistics_freq::STATISTICS_NONE)
              .build();
      cudf::io::write_parquet(options, stream);
    }
    auto rowType = ROW({"c0"}, {declaredType});
    auto split =
        CudfHiveConnectorSplitBuilder(file->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .build();
    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("c0"),
        std::make_shared<common::BigintRange>(1, 3, false));
    cudf::ast::column_reference staticFilter{0};
    MetadataOnlySplitReader reader(
        split,
        ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
            makeTableHandle("parquet_table", rowType),
        rowType,
        {"c0"},
        &files,
        ioExecutor_.get(),
        &context,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        &staticFilter,
        &filters);
    dwio::common::RuntimeStats stats;
    reader.prepareSplit(stats);
    EXPECT_EQ(reader.dynamicFiltersSatisfied(), matches);
  };
  std::vector<VectorPtr> values{
      makeFlatVector<int8_t>({1, 2, 3}),
      makeFlatVector<int16_t>({1, 2, 3}),
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int64_t>({1, 2, 3})};
  for (const auto& fileValues : values) {
    for (const auto& declared : values) {
      check(
          fileValues, declared->type(), fileValues->type() == declared->type());
    }
  }
  auto date = makeFlatVector<int32_t>({1, 2, 3});
  date->setType(DATE());
  check(date, INTEGER(), false);
  auto decimal = makeFlatVector<int64_t>({1, 2, 3});
  decimal->setType(DECIMAL(9, 2));
  check(decimal, BIGINT(), false);
  check(makeFlatVector<int64_t>({1, 2, 3}), BIGINT(), false, false);
}

TEST_F(CudfSplitReaderTest, setsPushdownFilterAfterReadingMetadata) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto connectorQueryCtx =
      ::facebook::velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool_.get())
          .connectorPool(pool_.get())
          .sessionProperties(properties.get())
          .queryId("query.CudfSplitReaderTest")
          .taskId("task.CudfSplitReaderTest")
          .planNodeId("plan.CudfSplitReaderTest")
          .build();
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto split =
      CudfHiveConnectorSplitBuilder(dataFile->getPath())
          .connectorId(
              ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
          .build();

  cudf::ast::column_reference logicalFilter{0};
  cudf::ast::column_reference splitFilter{0};
  MetadataOnlySplitReader reader(
      std::move(split),
      ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
          makeTableHandle("parquet_table", rowType),
      rowType,
      {"c0"},
      &fileHandleFactory,
      ioExecutor_.get(),
      connectorQueryCtx.get(),
      std::make_shared<CudfHiveConfig>(properties),
      std::make_shared<io::IoStatistics>(),
      std::make_shared<IoStats>(),
      &logicalFilter,
      nullptr);

  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  const auto& fileMetadatas = reader.fileMetaDatas();
  ASSERT_FALSE(fileMetadatas.empty());
  EXPECT_GT(fileMetadatas.front().schema.size(), 1);
  EXPECT_EQ(&reader.fileMetaDatas(), &fileMetadatas);

  reader.setPushdownFilter(&splitFilter);
  dwio::common::RuntimeStats runtimeStats;
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &splitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());
  EXPECT_EQ(runtimeStats.processedSplits, 1);
}

// The reader casts file timestamps to the session's
// parquet.reader.timestamp_type when the session sets it, and to the
// connector's parquet.reader.timestamp-type otherwise.
TEST_F(CudfSplitReaderTest, timestampTypeFollowsSession) {
  auto rowType = ROW({"c0"}, {TIMESTAMP()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector(
          {"c0"},
          {makeFlatVector<Timestamp>({Timestamp(1, 0), Timestamp(2, 0)})}));
  // Both settings take the enumerator's integer value.
  const auto seconds =
      std::to_string(static_cast<int32_t>(cudf::type_id::TIMESTAMP_SECONDS));
  const auto microseconds = std::to_string(
      static_cast<int32_t>(cudf::type_id::TIMESTAMP_MICROSECONDS));

  // Reads the file with the connector config 'connectorConfig' under
  // 'session' and returns the cuDF type of its column.
  const auto readTimestampType =
      [&](const std::unordered_map<std::string, std::string>& connectorConfig,
          const std::unordered_map<std::string, std::string>& session) {
        auto sessionProperties = std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>(session));
        auto connectorQueryCtx =
            ::facebook::velox::connector::ConnectorQueryCtx::Builder()
                .operatorPool(pool_.get())
                .connectorPool(pool_.get())
                .sessionProperties(sessionProperties.get())
                .queryId("query.CudfSplitReaderTest")
                .taskId("task.CudfSplitReaderTest")
                .planNodeId("plan.CudfSplitReaderTest")
                .build();
        FileHandleFactory fileHandleFactory(
            std::make_unique<FileHandleCache>(1000),
            std::make_unique<FileHandleGenerator>());
        auto split = CudfHiveConnectorSplitBuilder(dataFile->getPath())
                         .connectorId(
                             ::facebook::velox::cudf_velox::exec::test::
                                 kCudfHiveConnectorId)
                         .build();
        CudfSplitReader reader(
            std::move(split),
            ::facebook::velox::cudf_velox::exec::test::
                CudfHiveConnectorTestBase::makeTableHandle(
                    "parquet_table", rowType),
            rowType,
            {"c0"},
            &fileHandleFactory,
            ioExecutor_.get(),
            connectorQueryCtx.get(),
            std::make_shared<CudfHiveConfig>(
                std::make_shared<config::ConfigBase>(
                    std::unordered_map<std::string, std::string>(
                        connectorConfig))),
            std::make_shared<io::IoStatistics>(),
            std::make_shared<IoStats>(),
            nullptr);
        dwio::common::RuntimeStats runtimeStats;
        reader.prepareSplit(runtimeStats);
        auto chunk = reader.next(std::numeric_limits<uint64_t>::max());
        reader.stream().sync();
        if (!chunk.has_value()) {
          ADD_FAILURE() << "The reader returned no chunk";
          return cudf::type_id::EMPTY;
        }
        return chunk->table->view().column(0).type().id();
      };

  EXPECT_EQ(readTimestampType({}, {}), cudf::type_id::TIMESTAMP_MILLISECONDS);
  EXPECT_EQ(
      readTimestampType({{CudfHiveConfig::kTimestampType, seconds}}, {}),
      cudf::type_id::TIMESTAMP_SECONDS);
  EXPECT_EQ(
      readTimestampType(
          {{CudfHiveConfig::kTimestampType, seconds}},
          {{CudfHiveConfig::kTimestampTypeSession, microseconds}}),
      cudf::type_id::TIMESTAMP_MICROSECONDS);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
