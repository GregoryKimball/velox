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
#include "velox/experimental/ucx-exchange/ExchangeStore.h"
#include "velox/experimental/ucx-exchange/ExchangeStoreTables.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/utilities/memory_resource.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

#include <fmt/format.h>

#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::ucx_exchange {
namespace {

class ExchangeStoreTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static constexpr vector_size_t kNumRows = 1'000;
  static constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  static rmm::device_async_resource_ref memoryResource() {
    return cudf::get_current_device_resource_ref();
  }

  // Columns of every type TPC-H uses, some with nulls, and a fixed-width string
  // column, which the lanes codec splits into INT64 columns.
  RowVectorPtr makeInput() {
    std::vector<std::optional<std::string>> comments;
    std::vector<std::string> flags;
    for (vector_size_t row = 0; row < kNumRows; ++row) {
      comments.push_back(
          row % 17 == 0 ? std::nullopt
                        : std::optional<std::string>(std::string(row % 23, 'c')));
      flags.push_back(fmt::format("{:016d}", row));
    }
    return makeRowVector(
        {"orderkey",
         "quantity",
         "discount",
         "price",
         "total",
         "shipdate",
         "comment",
         "flag"},
        {
            makeFlatVector<int64_t>(kNumRows, [](auto row) { return row * 7; }),
            makeFlatVector<int32_t>(
                kNumRows, [](auto row) { return row % 50; }, nullEvery(11)),
            makeFlatVector<double>(
                kNumRows, [](auto row) { return row * 0.01; }),
            makeFlatVector<int64_t>(
                kNumRows,
                [](auto row) { return 100'000 + row; },
                nullEvery(13),
                DECIMAL(12, 2)),
            makeFlatVector<int128_t>(
                kNumRows,
                [](auto row) {
                  return static_cast<int128_t>(row) * 1'000'000'000'000'000'000;
                },
                nullptr,
                DECIMAL(38, 2)),
            makeFlatVector<int32_t>(
                kNumRows,
                [](auto row) { return 8'000 + row % 2'000; },
                nullptr,
                DATE()),
            makeNullableFlatVector<std::string>(comments),
            makeFlatVector<std::string>(flags),
        });
  }

  // Splits 'input' at 'splits' and packs each part with 'compression'.
  std::vector<PackedExchangePayload> pack(
      const RowVectorPtr& input,
      const std::vector<cudf::size_type>& splits,
      ExchangeCompression compression) {
    auto table = cudf_velox::with_arrow::toCudfTable(
        input, pool(), stream_, memoryResource());
    auto payloads = splitExchangePayloads(
        table->view(),
        splits,
        compression,
        stream_,
        memoryResource(),
        memoryResource());
    stream_.synchronize();
    return payloads;
  }

  // Rows 'offset' to 'offset' + 'size' of 'input' converted to cuDF and back
  // without the store. The conversion widens a short decimal's precision, so
  // the store's output is compared with this rather than with 'input'.
  RowVectorPtr expectedRows(
      const RowVectorPtr& input,
      vector_size_t offset,
      vector_size_t size) {
    auto slice = std::dynamic_pointer_cast<RowVector>(input->slice(offset, size));
    auto table = cudf_velox::with_arrow::toCudfTable(
        slice, pool(), stream_, memoryResource());
    return cudf_velox::with_arrow::toVeloxColumn(
        table->view(),
        pool(),
        asRowType(input->type()),
        "",
        stream_,
        memoryResource());
  }

  static std::size_t bytesOf(const PackedExchangePayload& payload) {
    return payload.packed->gpu_data ? payload.packed->gpu_data->size() : 0;
  }

  // Takes 'file' of 'table' from 'store' and converts each payload to a Velox
  // vector of 'type'.
  std::vector<RowVectorPtr> takeRows(
      ExchangeStore& store,
      std::string_view table,
      std::string_view file,
      const RowTypePtr& type) {
    std::vector<RowVectorPtr> rows;
    for (auto& payload : store.take(table, file, stream_, memoryResource())) {
      joinExchangeLanes(*payload, memoryResource());
      const auto view = payload->table != nullptr
          ? payload->table->view()
          : payload->packedTable->table;
      rows.push_back(
          cudf_velox::with_arrow::toVeloxColumn(
              view, pool(), type, "", stream_, memoryResource()));
    }
    return rows;
  }

  rmm::cuda_stream stream_;
};

std::vector<ExchangeCompression> compressions() {
  return {
      ExchangeCompression::kNone,
      ExchangeCompression::kCascaded,
#if __has_include(<cudf/detail/fused_for.hpp>)
      ExchangeCompression::kFusedForBitpacked,
      ExchangeCompression::kFusedForBitpackedLanes,
#endif
  };
}

TEST_F(ExchangeStoreTest, roundTripsEveryTpchColumnTypeInEitherTier) {
  const auto input = makeInput();
  const auto type = asRowType(input->type());
  const std::vector<cudf::size_type> splits{300, 700};
  const std::vector<vector_size_t> offsets{0, 300, 700, kNumRows};

  for (const auto compression : compressions()) {
    for (const bool device : {true, false}) {
      SCOPED_TRACE(
          fmt::format(
              "compression {}, {} tier",
              static_cast<int>(compression),
              device ? "device" : "host"));
      ExchangeStore store;
      store.setDeviceCapacity(device ? kUnlimited : 0);
      uint64_t totalBytes = 0;
      auto payloads = pack(input, splits, compression);
      ASSERT_EQ(payloads.size(), 3);
      for (int32_t bucket = 0; bucket < 3; ++bucket) {
        totalBytes += bytesOf(payloads[bucket]);
        store.add(
            "table",
            std::to_string(bucket),
            std::move(payloads[bucket]),
            offsets[bucket + 1] - offsets[bucket],
            stream_);
      }
      EXPECT_EQ(store.stats().deviceBytes, device ? totalBytes : 0);
      EXPECT_EQ(store.stats().hostBytes, device ? 0 : totalBytes);
      EXPECT_EQ(store.stats().payloads, 3);

      for (int32_t bucket = 0; bucket < 3; ++bucket) {
        const auto rows = takeRows(store, "table", std::to_string(bucket), type);
        ASSERT_EQ(rows.size(), 1);
        velox::test::assertEqualVectors(
            expectedRows(
                input, offsets[bucket], offsets[bucket + 1] - offsets[bucket]),
            rows[0]);
      }
      EXPECT_EQ(store.stats().deviceBytes, 0);
      EXPECT_EQ(store.stats().hostBytes, 0);
      EXPECT_EQ(store.stats().payloads, 0);
    }
  }
}

TEST_F(ExchangeStoreTest, overflowsToHostWhenDeviceTierIsFull) {
  const auto input = makeInput();
  const auto type = asRowType(input->type());
  auto payloads = pack(input, {300, 700}, compressions().back());
  const std::vector<std::size_t> bytes{
      bytesOf(payloads[0]), bytesOf(payloads[1]), bytesOf(payloads[2])};

  // Room for the first payload only. All three go to one file, so the file
  // spans both tiers and must still come back in order.
  ExchangeStore store;
  store.setDeviceCapacity(bytes[0]);
  const std::vector<vector_size_t> offsets{0, 300, 700, kNumRows};
  for (int part = 0; part < 3; ++part) {
    store.add(
        "table",
        "file",
        std::move(payloads[part]),
        offsets[part + 1] - offsets[part],
        stream_);
  }
  EXPECT_EQ(store.stats().deviceBytes, bytes[0]);
  EXPECT_EQ(store.stats().hostBytes, bytes[1] + bytes[2]);

  const auto rows = takeRows(store, "table", "file", type);
  ASSERT_EQ(rows.size(), 3);
  for (int part = 0; part < 3; ++part) {
    velox::test::assertEqualVectors(
        expectedRows(input, offsets[part], offsets[part + 1] - offsets[part]),
        rows[part]);
  }
  EXPECT_EQ(store.stats().deviceBytes, 0);
  EXPECT_EQ(store.stats().hostBytes, 0);
}

TEST_F(ExchangeStoreTest, dropFreesEveryFileOfOneTableInBothTiers) {
  const auto input = makeInput();
  const auto type = asRowType(input->type());
  auto dropped = pack(input, {300, 700}, compressions().back());
  auto kept = pack(input, {}, compressions().back());
  const auto keptBytes = bytesOf(kept[0]);

  ExchangeStore store;
  store.setDeviceCapacity(bytesOf(dropped[0]));
  for (int32_t bucket = 0; bucket < 3; ++bucket) {
    store.add(
        "dropped",
        std::to_string(bucket),
        std::move(dropped[bucket]),
        1,
        stream_);
  }
  ASSERT_GT(store.stats().hostBytes, 0);
  store.add("kept", "0", std::move(kept[0]), kNumRows, stream_);

  store.drop("dropped");
  EXPECT_EQ(store.stats().deviceBytes + store.stats().hostBytes, keptBytes);
  EXPECT_EQ(store.stats().payloads, 1);
  EXPECT_THAT(
      store.take("dropped", "0", stream_, memoryResource()),
      testing::IsEmpty());

  const auto rows = takeRows(store, "kept", "0", type);
  ASSERT_EQ(rows.size(), 1);
  velox::test::assertEqualVectors(expectedRows(input, 0, kNumRows), rows[0]);
  EXPECT_THAT(
      store.take("kept", "0", stream_, memoryResource()), testing::IsEmpty());
}

TEST_F(ExchangeStoreTest, failsWhenHostTierIsFull) {
  const auto input = makeInput();
  auto payloads = pack(input, {500}, ExchangeCompression::kNone);
  const auto firstBytes = bytesOf(payloads[0]);

  ExchangeStore store;
  store.setDeviceCapacity(0);
  store.setHostCapacity(firstBytes);
  store.add("table", "0", std::move(payloads[0]), 500, stream_);
  VELOX_ASSERT_THROW(
      store.add("table", "1", std::move(payloads[1]), 500, stream_),
      "The exchange store's host tier is full");
  EXPECT_EQ(store.stats().hostBytes, firstBytes);
  EXPECT_EQ(store.stats().payloads, 1);
  store.drop("table");
  EXPECT_EQ(store.stats().hostBytes, 0);
}

TEST(ExchangeStoreTablesTest, keysFilesByTableDirectoryAndFileName) {
  const std::pair<std::string, std::string> expected{
      "__presto_temporary_table_PARQUET_q_1", "000003_0_q"};
  // The writer's path and a split's path, which carries a scheme.
  EXPECT_EQ(
      storeKey("/data/hive/__presto_temporary_table_PARQUET_q_1/000003_0_q"),
      expected);
  EXPECT_EQ(
      storeKey(
          "file:/data/hive/__presto_temporary_table_PARQUET_q_1/000003_0_q"),
      expected);
  EXPECT_EQ(
      storeKey("__presto_temporary_table_PARQUET_q_1/000003_0_q"), expected);
  VELOX_ASSERT_THROW(storeKey("000003_0_q"), "Not a path to a temporary table");
  VELOX_ASSERT_THROW(
      storeKey("/data/__presto_temporary_table_PARQUET_q_1/"),
      "Not a path to a temporary table");
}

} // namespace
} // namespace facebook::velox::ucx_exchange
