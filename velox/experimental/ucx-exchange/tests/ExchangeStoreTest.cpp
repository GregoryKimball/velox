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
      const RowTypePtr& type,
      ExchangeStore::TakeStats* stats = nullptr) {
    std::vector<RowVectorPtr> rows;
    for (auto& taken :
         store.take(table, file, stream_, memoryResource(), stats)) {
      rows.push_back(toRows(taken, type));
    }
    return rows;
  }

  // Converts a taken payload to a Velox vector of 'type'.
  RowVectorPtr toRows(ExchangeStore::TakenPayload& taken, const RowTypePtr& type) {
    EXPECT_THAT(taken.encoding.columns, testing::IsEmpty());
    auto& payload = taken.data;
    joinExchangeLanes(*payload, memoryResource());
    const auto view = payload->table != nullptr ? payload->table->view()
                                                : payload->packedTable->table;
    return cudf_velox::with_arrow::toVeloxColumn(
        view, pool(), type, "", stream_, memoryResource());
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
            offsets[bucket + 1] - offsets[bucket], {}, stream_);
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
  std::vector<bool> toHost;
  for (int part = 0; part < 3; ++part) {
    toHost.push_back(store.add(
        "table",
        "file",
        std::move(payloads[part]),
        offsets[part + 1] - offsets[part], {}, stream_));
  }
  EXPECT_THAT(toHost, testing::ElementsAre(false, true, true));
  EXPECT_EQ(store.stats().deviceBytes, bytes[0]);
  EXPECT_EQ(store.stats().hostBytes, bytes[1] + bytes[2]);

  ExchangeStore::TakeStats takeStats;
  const auto rows = takeRows(store, "table", "file", type, &takeStats);
  ASSERT_EQ(rows.size(), 3);
  EXPECT_EQ(takeStats.deviceBytes, bytes[0]);
  EXPECT_EQ(takeStats.hostBytes, bytes[1] + bytes[2]);
  EXPECT_GT(takeStats.copyNanos, 0);
  for (int part = 0; part < 3; ++part) {
    velox::test::assertEqualVectors(
        expectedRows(input, offsets[part], offsets[part + 1] - offsets[part]),
        rows[part]);
  }
  EXPECT_EQ(store.stats().deviceBytes, 0);
  EXPECT_EQ(store.stats().hostBytes, 0);
}

TEST_F(ExchangeStoreTest, readsHostTierCopiedOnAnotherStream) {
  const auto input = makeInput();
  const auto type = asRowType(input->type());
  auto payloads = pack(input, {}, compressions().back());
  ExchangeStore store;
  store.setDeviceCapacity(0);
  // The copy to the host tier runs behind the writer on the writer's stream;
  // the reader neither shares that stream nor synchronizes it.
  rmm::cuda_stream writer;
  ASSERT_TRUE(
      store.add("table", "file", std::move(payloads[0]), kNumRows, {}, writer));
  const auto rows = takeRows(store, "table", "file", type);
  ASSERT_EQ(rows.size(), 1);
  velox::test::assertEqualVectors(expectedRows(input, 0, kNumRows), rows[0]);
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
        1, {}, stream_);
  }
  ASSERT_GT(store.stats().hostBytes, 0);
  store.add("kept", "0", std::move(kept[0]), kNumRows, {}, stream_);

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

TEST_F(ExchangeStoreTest, dropQueryFreesOnlyThatQuerysTables) {
  const auto input = makeInput();
  auto payloads = pack(input, {300, 700}, compressions().back());
  ExchangeStore store;
  store.setDeviceCapacity(bytesOf(payloads[0]));
  store.add("__presto_temporary_table_ORC_q1_a", "0", std::move(payloads[0]), 300, {}, stream_);
  store.add("__presto_temporary_table_ORC_q1_b", "0", std::move(payloads[1]), 400, {}, stream_);
  const auto keptBytes = bytesOf(payloads[2]);
  store.add("__presto_temporary_table_ORC_q2_a", "0", std::move(payloads[2]), 300, {}, stream_);
  EXPECT_EQ(store.stats().tables, 3);

  store.dropQuery("q1");
  EXPECT_EQ(store.stats().tables, 1);
  EXPECT_EQ(store.stats().payloads, 1);
  EXPECT_EQ(store.stats().deviceBytes + store.stats().hostBytes, keptBytes);
  EXPECT_THAT(
      store.take("__presto_temporary_table_ORC_q1_a", "0", stream_, memoryResource()),
      testing::IsEmpty());
  EXPECT_THAT(
      store.take("__presto_temporary_table_ORC_q2_a", "0", stream_, memoryResource()),
      testing::SizeIs(1));
}

TEST_F(ExchangeStoreTest, keepsNothingForADroppedQuery) {
  const auto input = makeInput();
  auto payloads = pack(input, {300, 700}, compressions().back());
  ExchangeStore store;
  store.add("__presto_temporary_table_ORC_q1_a", "0", std::move(payloads[0]), 300, {}, stream_);
  store.dropQuery("q1");

  // A writer of the query that ends after the drop, and a late reader.
  store.beginWrite("__presto_temporary_table_ORC_q1_b", "q1.2.0.0.0");
  store.add("__presto_temporary_table_ORC_q1_b", "0", std::move(payloads[1]), 400, {}, stream_);
  ContinueFuture future = ContinueFuture::makeEmpty();
  const auto taken = store.takeOrWait(
      "__presto_temporary_table_ORC_q1_c", {"0"}, stream_, memoryResource(), future);
  ASSERT_TRUE(taken.has_value());
  EXPECT_THAT(*taken, testing::IsEmpty());
  store.endWrites("q1.2.0.0.0");
  EXPECT_EQ(store.stats().tables, 0);
  EXPECT_EQ(store.stats().payloads, 0);
  EXPECT_EQ(store.stats().deviceBytes + store.stats().hostBytes, 0);

  // Other queries keep their data.
  store.add("__presto_temporary_table_ORC_q2_a", "0", std::move(payloads[2]), 300, {}, stream_);
  EXPECT_EQ(store.stats().tables, 1);
  EXPECT_EQ(store.stats().payloads, 1);
}

TEST_F(ExchangeStoreTest, keepsSharedKeysUntilTheirLastPayloadIsTaken) {
  using cudf_velox::connector::hive::FusedForEncoding;
  const auto keysInput = makeRowVector({makeFlatVector<int64_t>({10, 20, 30})});
  const auto indicesInput =
      makeRowVector({makeFlatVector<int32_t>({2, 0, 1, 1, 0, 2})});
  const auto keysType = asRowType(keysInput->type());

  for (const bool device : {true, false}) {
    SCOPED_TRACE(device ? "device tier" : "host tier");
    ExchangeStore store;
    store.setDeviceCapacity(device ? kUnlimited : 0);
    auto keys = pack(keysInput, {}, ExchangeCompression::kFusedForBitpacked);
    const auto keysBytes = bytesOf(keys[0]);
    // Two files of one batch share the keys, a third table has its own.
    const auto id = store.addKeys("table", std::move(keys[0]), 3, 2, stream_);
    auto indices =
        pack(indicesInput, {4}, ExchangeCompression::kFusedForBitpacked);
    const auto indicesBytes = bytesOf(indices[0]) + bytesOf(indices[1]);
    const ExchangeStore::PayloadEncoding encoding{
        {FusedForEncoding::kDictionary}, {id}, {}};
    store.add("table", "0", std::move(indices[0]), 4, encoding, stream_);
    store.add("table", "1", std::move(indices[1]), 2, encoding, stream_);
    EXPECT_EQ(store.stats().payloads, 3);
    EXPECT_EQ(
        device ? store.stats().deviceBytes : store.stats().hostBytes,
        keysBytes + indicesBytes);
    // Keys stay where they are when the device tier shrinks.
    store.demote(std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(store.stats().deviceBytes, device ? keysBytes : 0);

    for (const auto& [file, numRows] :
         {std::pair<std::string, int>{"0", 4}, {"1", 2}}) {
      auto taken = store.take("table", file, stream_, memoryResource());
      ASSERT_EQ(taken.size(), 1);
      EXPECT_THAT(
          taken[0].encoding.columns,
          testing::ElementsAre(FusedForEncoding::kDictionary));
      ASSERT_EQ(taken[0].keys.size(), 1);
      EXPECT_EQ(taken[0].data->numRows, numRows);
      auto& keysData = *taken[0].keys[0];
      EXPECT_EQ(keysData.numRows, 3);
      velox::test::assertEqualVectors(
          keysInput,
          cudf_velox::with_arrow::toVeloxColumn(
              keysData.tableView(), pool(), keysType, "", stream_, memoryResource()));
      // The keys stay while the other file still refers to them.
      EXPECT_EQ(store.stats().payloads, file == "0" ? 2 : 0);
    }
    EXPECT_EQ(store.stats().deviceBytes + store.stats().hostBytes, 0);

    // Dropping a table frees its keys too.
    auto otherKeys = pack(keysInput, {}, ExchangeCompression::kFusedForBitpacked);
    const auto otherId =
        store.addKeys("other", std::move(otherKeys[0]), 3, 1, stream_);
    auto otherIndices =
        pack(indicesInput, {}, ExchangeCompression::kFusedForBitpacked);
    store.add(
        "other",
        "0",
        std::move(otherIndices[0]),
        6,
        {{FusedForEncoding::kDictionary}, {otherId}, {}},
        stream_);
    store.drop("other");
    EXPECT_EQ(store.stats().payloads, 0);
    EXPECT_EQ(store.stats().deviceBytes + store.stats().hostBytes, 0);
  }
}

TEST_F(ExchangeStoreTest, demotesDeviceTierToHostAndStaysSmaller) {
  const auto input = makeInput();
  const auto type = asRowType(input->type());
  auto payloads = pack(input, {300, 700}, compressions().back());
  const std::vector<std::size_t> bytes{
      bytesOf(payloads[0]), bytesOf(payloads[1]), bytesOf(payloads[2])};
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.add("table", "0", std::move(payloads[0]), 300, {}, stream_);
  store.add("table", "1", std::move(payloads[1]), 400, {}, stream_);
  ASSERT_EQ(store.stats().deviceBytes, bytes[0] + bytes[1]);

  // A one-byte request moves one payload.
  const auto moved = store.demote(1);
  EXPECT_THAT(moved, testing::AnyOf(bytes[0], bytes[1]));
  EXPECT_EQ(store.stats().hostBytes, moved);
  EXPECT_EQ(store.stats().deviceBytes, bytes[0] + bytes[1] - moved);

  // The device tier stays at its smaller size: the next payload goes to host.
  store.add("table", "2", std::move(payloads[2]), 300, {}, stream_);
  EXPECT_EQ(store.stats().hostBytes, moved + bytes[2]);

  const std::vector<vector_size_t> offsets{0, 300, 700, kNumRows};
  for (int file = 0; file < 3; ++file) {
    const auto rows = takeRows(store, "table", std::to_string(file), type);
    ASSERT_EQ(rows.size(), 1);
    velox::test::assertEqualVectors(
        expectedRows(input, offsets[file], offsets[file + 1] - offsets[file]),
        rows[0]);
  }
  EXPECT_EQ(store.demote(1), 0);
}

TEST_F(ExchangeStoreTest, restoresDeviceTierOnceItHoldsNoTables) {
  const auto input = makeInput();
  auto payloads = pack(input, {500}, compressions().back());
  const auto secondBytes = bytesOf(payloads[1]);
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.add("query1_table", "0", std::move(payloads[0]), 500, {}, stream_);
  ASSERT_GT(store.demote(1), 0);
  store.drop("query1_table");

  // The shrink answered one query's lack of memory: once the store holds no
  // tables, the next payload gets the configured device tier again.
  EXPECT_FALSE(store.add(
      "query2_table",
      "0",
      std::move(payloads[1]),
      kNumRows - 500,
      {},
      stream_));
  EXPECT_EQ(store.stats().deviceBytes, secondBytes);
  EXPECT_EQ(store.stats().hostBytes, 0);
}

TEST_F(ExchangeStoreTest, takeOrWaitStreamsATableWhileItIsWritten) {
  const auto input = makeInput();
  auto payloads = pack(input, {500}, compressions().back());
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.beginWrite("table", "query.1.0.0.0");

  // Nothing written yet: the reader waits.
  auto future = ContinueFuture::makeEmpty();
  EXPECT_FALSE(
      store.takeOrWait("table", {"0"}, stream_, memoryResource(), future)
          .has_value());
  ASSERT_TRUE(future.valid());
  EXPECT_FALSE(future.isReady());

  // A write wakes the reader, which takes it while the table is still open.
  store.add("table", "0", std::move(payloads[0]), 500, {}, stream_);
  EXPECT_TRUE(future.isReady());
  future = ContinueFuture::makeEmpty();
  auto taken =
      store.takeOrWait("table", {"0"}, stream_, memoryResource(), future);
  ASSERT_TRUE(taken.has_value());
  EXPECT_EQ(taken->size(), 1);

  // Nothing new and the writer is still open: wait again, until the writer's
  // task finishes and completes the table.
  EXPECT_FALSE(
      store.takeOrWait("table", {"0"}, stream_, memoryResource(), future)
          .has_value());
  ASSERT_TRUE(future.valid());
  store.endWrites("query.1.0.0.0");
  EXPECT_TRUE(future.isReady());
  future = ContinueFuture::makeEmpty();
  auto done = store.takeOrWait("table", {"0"}, stream_, memoryResource(), future);
  ASSERT_TRUE(done.has_value());
  EXPECT_THAT(*done, testing::IsEmpty());
  EXPECT_FALSE(future.valid());
}

TEST_F(ExchangeStoreTest, takeOrWaitReadsAFileNeverWrittenAsDoneOnceComplete) {
  const auto input = makeInput();
  auto payloads = pack(input, {500}, compressions().back());
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.beginWrite("table", "query.1.0.0.0");
  store.add("table", "0", std::move(payloads[0]), 500, {}, stream_);
  store.endWrites("query.1.0.0.0");
  auto future = ContinueFuture::makeEmpty();
  ASSERT_EQ(
      store.takeOrWait("table", {"0"}, stream_, memoryResource(), future)
          ->size(),
      1);

  // The table was taken entirely and is gone, but a bucket that got no rows
  // still reads as done instead of waiting for a writer that has finished.
  auto empty =
      store.takeOrWait("table", {"1"}, stream_, memoryResource(), future);
  ASSERT_TRUE(empty.has_value());
  EXPECT_THAT(*empty, testing::IsEmpty());
  EXPECT_FALSE(future.valid());
}

TEST_F(ExchangeStoreTest, takeOrWaitTakesAnyFileOfASet) {
  const auto input = makeInput();
  auto payloads = pack(input, {300, 700}, compressions().back());
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.beginWrite("table", "query.1.0.0.0");
  const std::vector<std::string> files{"1", "3"};
  auto future = ContinueFuture::makeEmpty();
  ASSERT_FALSE(
      store.takeOrWait("table", files, stream_, memoryResource(), future)
          .has_value());

  // A write to a file outside the set wakes the reader, which finds nothing
  // and waits again.
  store.add("table", "2", std::move(payloads[0]), 300, {}, stream_);
  EXPECT_TRUE(future.isReady());
  future = ContinueFuture::makeEmpty();
  ASSERT_FALSE(
      store.takeOrWait("table", files, stream_, memoryResource(), future)
          .has_value());
  EXPECT_TRUE(future.valid());

  // One call takes what every file of the set got.
  store.add("table", "1", std::move(payloads[1]), 400, {}, stream_);
  store.add("table", "3", std::move(payloads[2]), kNumRows - 700, {}, stream_);
  future = ContinueFuture::makeEmpty();
  auto taken =
      store.takeOrWait("table", files, stream_, memoryResource(), future);
  ASSERT_TRUE(taken.has_value());
  EXPECT_EQ(taken->size(), 2);

  // Once the table is complete the set reads as done, and the file outside it
  // keeps its payload.
  store.endWrites("query.1.0.0.0");
  auto done =
      store.takeOrWait("table", files, stream_, memoryResource(), future);
  ASSERT_TRUE(done.has_value());
  EXPECT_THAT(*done, testing::IsEmpty());
  EXPECT_EQ(store.stats().payloads, 1);
}

TEST_F(ExchangeStoreTest, takeOrWaitTakesAFileABatchAtATime) {
  // A reader asks for a batch of 300 rows from a file of five 200-row
  // payloads. Each call returns the next payloads in order until they hold 300
  // rows, and the others stay in the store until the reader asks again.
  const auto input = makeInput();
  const auto type = asRowType(input->type());
  auto payloads = pack(input, {200, 400, 600, 800}, compressions().back());
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.beginWrite("table", "query.1.0.0.0");
  for (auto& payload : payloads) {
    store.add("table", "0", std::move(payload), 200, {}, stream_);
  }
  store.endWrites("query.1.0.0.0");

  std::vector<size_t> taken;
  std::vector<uint64_t> kept;
  vector_size_t offset = 0;
  auto future = ContinueFuture::makeEmpty();
  for (;;) {
    auto batch = store.takeOrWait(
        "table", {"0"}, stream_, memoryResource(), future, nullptr, 300);
    ASSERT_TRUE(batch.has_value());
    if (batch->empty()) {
      break;
    }
    taken.push_back(batch->size());
    kept.push_back(store.stats().payloads);
    for (auto& payload : *batch) {
      velox::test::assertEqualVectors(
          expectedRows(input, offset, 200), toRows(payload, type));
      offset += 200;
    }
  }
  EXPECT_THAT(taken, testing::ElementsAre(2, 2, 1));
  EXPECT_THAT(kept, testing::ElementsAre(3, 1, 0));
  EXPECT_EQ(offset, kNumRows);
}

TEST_F(ExchangeStoreTest, dropWakesReadersWaitingForTheTable) {
  ExchangeStore store;
  store.setDeviceCapacity(kUnlimited);
  store.beginWrite("query1_table", "query1.1.0.0.0");
  auto first = ContinueFuture::makeEmpty();
  ASSERT_FALSE(
      store.takeOrWait("query1_table", {"0"}, stream_, memoryResource(), first)
          .has_value());
  store.beginWrite("query1_other", "query1.2.0.0.0");
  auto second = ContinueFuture::makeEmpty();
  ASSERT_FALSE(
      store.takeOrWait("query1_other", {"0"}, stream_, memoryResource(), second)
          .has_value());

  // A failed task drops its query's tables. The readers wake up instead of
  // failing with a broken promise, which would hide the task's error.
  store.drop("query1_table");
  ASSERT_TRUE(first.isReady());
  EXPECT_TRUE(first.hasValue());
  store.dropQuery("query1");
  ASSERT_TRUE(second.isReady());
  EXPECT_TRUE(second.hasValue());
}

TEST_F(ExchangeStoreTest, failsWhenHostTierIsFull) {
  const auto input = makeInput();
  auto payloads = pack(input, {500}, ExchangeCompression::kNone);
  const auto firstBytes = bytesOf(payloads[0]);

  ExchangeStore store;
  store.setDeviceCapacity(0);
  store.setHostCapacity(firstBytes);
  store.add("table", "0", std::move(payloads[0]), 500, {}, stream_);
  VELOX_ASSERT_THROW(
      store.add("table", "1", std::move(payloads[1]), 500, {}, stream_),
      "The exchange store's host tier is full");
  EXPECT_EQ(store.stats().hostBytes, firstBytes);
  EXPECT_EQ(store.stats().payloads, 1);
  store.drop("table");
  EXPECT_EQ(store.stats().hostBytes, 0);
}

TEST(ExchangeStoreTablesTest, keysFilesByTableDirectoryAndFileName) {
  const std::pair<std::string, std::string> expected{
      "__presto_temporary_table_PARQUET_q_1", "3"};
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

TEST(ExchangeStoreTablesTest, keysBucketFilesByBucketNumber) {
  // A bucket's file is found by its bucket number whatever the padding and the
  // rest of the name, so that a split made before the file is written (the
  // reading section starting with the writing one) finds it.
  const std::pair<std::string, std::string> expected{
      "__presto_temporary_table_PARQUET_q_1", "3"};
  EXPECT_EQ(
      storeKey("/data/__presto_temporary_table_PARQUET_q_1/0003_0_query"),
      expected);
  EXPECT_EQ(
      storeKey("/split/__presto_temporary_table_PARQUET_q_1/03_0_query.parquet"),
      expected);
  EXPECT_EQ(
      storeKey("/data/__presto_temporary_table_PARQUET_q_1/012_0_q").second,
      "12");
  // Files that are not bucket files keep their name.
  EXPECT_EQ(storeKey("/data/table/part-0001.parquet").second, "part-0001.parquet");
  EXPECT_EQ(storeKey("/data/table/0003").second, "0003");
}

} // namespace
} // namespace facebook::velox::ucx_exchange
