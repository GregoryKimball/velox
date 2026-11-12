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
#include "velox/experimental/cudf/connectors/hive/FusedForColumnEncoding.h"

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

#include <fmt/format.h>

#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class FusedForColumnEncodingTest : public testing::Test,
                                   public velox::test::VectorTestBase {
 protected:
  static constexpr vector_size_t kNumRows = 20'000;

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  static rmm::device_async_resource_ref memoryResource() {
    return cudf::get_current_device_resource_ref();
  }

  std::unique_ptr<cudf::table> toCudf(const VectorPtr& vector) {
    return with_arrow::toCudfTable(
        makeRowVector({vector}), pool(), stream_, memoryResource());
  }

  VectorPtr toVelox(cudf::column_view column, const TypePtr& type) {
    return with_arrow::toVeloxColumn(
               cudf::table_view{{column}},
               pool(),
               ROW({"c0"}, {type}),
               "",
               stream_,
               memoryResource())
        ->childAt(0);
  }

  // Encodes 'vector' in segments starting at 'segmentStarts', expects
  // 'expected', and checks that each segment decodes to its rows.
  void expectRoundTrip(
      const VectorPtr& vector,
      FusedForEncoding expected,
      const std::vector<cudf::size_type>& segmentStarts = {}) {
    auto table = toCudf(vector);
    const auto column = table->view().column(0);
    auto encoded =
        encodeForFusedFor(column, segmentStarts, stream_, memoryResource());
    ASSERT_EQ(encoded.encoding, expected);
    ASSERT_TRUE(encoded.packedValues.has_value());
    EXPECT_EQ(
        encoded.packedKeys.has_value(),
        expected == FusedForEncoding::kDictionary);
    EXPECT_EQ(encoded.keys != nullptr, expected == FusedForEncoding::kDictionary);
    EXPECT_EQ(encoded.encoded == nullptr, expected == FusedForEncoding::kValues);
    if (expected == FusedForEncoding::kValues) {
      return;
    }
    // The column as cuDF holds it: short decimals come back wider.
    const auto expectedRows = toVelox(column, vector->type());

    auto starts = segmentStarts.empty() ? std::vector<cudf::size_type>{0}
                                        : segmentStarts;
    if (expected == FusedForEncoding::kDeltas && !segmentStarts.empty()) {
      ASSERT_EQ(encoded.segmentBases.size(), segmentStarts.size());
    } else {
      EXPECT_TRUE(encoded.segmentBases.empty());
    }
    starts.push_back(column.size());
    for (size_t segment = 0; segment + 1 < starts.size(); ++segment) {
      const auto begin = std::min(starts[segment], column.size());
      const auto end = std::min(starts[segment + 1], column.size());
      SCOPED_TRACE(fmt::format("rows {} to {}", begin, end));
      if (begin >= end) {
        // The base of an empty segment is never read. One that starts past
        // the end has none.
        if (!encoded.segmentBases.empty() && starts[segment] >= column.size()) {
          EXPECT_EQ(encoded.segmentBases[segment], 0);
        }
        continue;
      }
      const auto values =
          cudf::slice(encoded.encoded->view(), {begin, end}, stream_).front();
      const auto base = encoded.segmentBases.empty()
          ? 0
          : encoded.segmentBases[segment];
      auto decoded = decodeFusedForEncoding(
          encoded.encoding,
          values,
          encoded.keys ? std::optional(encoded.keys->view()) : std::nullopt,
          base,
          stream_,
          memoryResource(),
          memoryResource());
      ASSERT_NE(decoded, nullptr);
      velox::test::assertEqualVectors(
          expectedRows->slice(begin, end - begin),
          toVelox(decoded->view(), vector->type()));
    }
  }

  rmm::cuda_stream stream_;
};

TEST_F(FusedForColumnEncodingTest, encodesFewWidelySpreadValuesAsDictionary) {
  // Quantities: 50 values over a range of 4,900 take 13 bits as values and 6
  // as indices.
  expectRoundTrip(
      makeFlatVector<int64_t>(
          kNumRows,
          [](auto row) { return (row * 7 % 50 + 1) * 100; },
          nullptr,
          DECIMAL(12, 2)),
      FusedForEncoding::kDictionary);
  // Long decimals, which fused FOR does not narrow.
  expectRoundTrip(
      makeFlatVector<int128_t>(
          kNumRows,
          [](auto row) {
            return static_cast<int128_t>(row % 7) * 1'000'000'000'000'000'000;
          },
          nullptr,
          DECIMAL(38, 2)),
      FusedForEncoding::kDictionary);
}

TEST_F(FusedForColumnEncodingTest, encodesStringsOnlyAsDictionary) {
  expectRoundTrip(
      makeFlatVector<std::string>(
          kNumRows, [](auto row) { return fmt::format("mode {}", row % 7); }),
      FusedForEncoding::kDictionary);

  // Too many distinct strings: nothing to pack.
  auto table = toCudf(makeFlatVector<std::string>(
      kNumRows, [](auto row) { return fmt::format("comment {}", row); }));
  const auto encoded = encodeForFusedFor(
      table->view().column(0), {}, stream_, memoryResource());
  EXPECT_EQ(encoded.encoding, FusedForEncoding::kValues);
  EXPECT_FALSE(encoded.packedValues.has_value());
}

TEST_F(FusedForColumnEncodingTest, encodesAscendingKeysAsDifferences) {
  const auto keys = makeFlatVector<int64_t>(
      kNumRows, [](auto row) { return 1'000'000'000'000 + row * 4 + row % 3; });
  expectRoundTrip(keys, FusedForEncoding::kDeltas);
  // Segments, as buckets of a partitioned batch: the second is empty and the
  // last starts past the end.
  expectRoundTrip(
      keys, FusedForEncoding::kDeltas, {0, 3'000, 3'000, 11'111, kNumRows});
}

TEST_F(FusedForColumnEncodingTest, encodesAsAnEarlierChoice) {
  const auto ascending = toCudf(makeFlatVector<int64_t>(
      kNumRows, [](auto row) { return 1'000'000 + row * 4; }));
  const std::vector<cudf::size_type> starts{0, 5'000, 12'345};
  auto deltas = encodeAsFusedFor(
      FusedForEncoding::kDeltas,
      ascending->view().column(0),
      starts,
      stream_,
      memoryResource());
  ASSERT_EQ(deltas.encoding, FusedForEncoding::kDeltas);
  EXPECT_FALSE(deltas.packedValues.has_value());
  ASSERT_EQ(deltas.segmentBases.size(), starts.size());
  EXPECT_EQ(deltas.segmentBases[1], 1'000'000 + 5'000 * 4);
  auto restored = decodeFusedForEncoding(
      FusedForEncoding::kDeltas,
      cudf::slice(deltas.encoded->view(), {5'000, 12'345}, stream_).front(),
      std::nullopt,
      deltas.segmentBases[1],
      stream_,
      memoryResource(),
      memoryResource());
  velox::test::assertEqualVectors(
      toVelox(ascending->view().column(0), BIGINT())->slice(5'000, 7'345),
      toVelox(restored->view(), BIGINT()));

  const auto modes = toCudf(makeFlatVector<std::string>(
      kNumRows, [](auto row) { return fmt::format("mode {}", row % 7); }));
  auto dictionary = encodeAsFusedFor(
      FusedForEncoding::kDictionary,
      modes->view().column(0),
      {},
      stream_,
      memoryResource());
  ASSERT_EQ(dictionary.encoding, FusedForEncoding::kDictionary);
  EXPECT_EQ(dictionary.keys->size(), 7);
  EXPECT_FALSE(dictionary.packedKeys.has_value());

  // Data that no longer allows the earlier choice keeps its values.
  const auto negative = toCudf(makeFlatVector<int64_t>(
      kNumRows, [](auto row) { return row - 10; }));
  EXPECT_EQ(
      encodeAsFusedFor(
          FusedForEncoding::kDeltas,
          negative->view().column(0),
          {},
          stream_,
          memoryResource())
          .encoding,
      FusedForEncoding::kValues);
  const auto distinct = toCudf(makeFlatVector<std::string>(
      kNumRows, [](auto row) { return fmt::format("comment {}", row); }));
  const auto noDictionary = encodeAsFusedFor(
      FusedForEncoding::kDictionary,
      distinct->view().column(0),
      {},
      stream_,
      memoryResource());
  EXPECT_EQ(noDictionary.encoding, FusedForEncoding::kValues);
  EXPECT_EQ(noDictionary.encoded, nullptr);
}

TEST_F(FusedForColumnEncodingTest, keepsValuesWhenNoEncodingApplies) {
  // Nulls rule out both encodings.
  expectRoundTrip(
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return row % 50 * 100; }, nullEvery(5)),
      FusedForEncoding::kValues);
  // Negative keys are not stored as differences, and they have too many
  // distinct values for a dictionary.
  expectRoundTrip(
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return -1'000'000'000'000 + row * 4; }),
      FusedForEncoding::kValues);
  // A dictionary would not save a quarter of the bytes of small values.
  expectRoundTrip(
      makeFlatVector<int32_t>(kNumRows, [](auto row) { return row % 4; }),
      FusedForEncoding::kValues);
  expectRoundTrip(
      makeFlatVector<bool>(kNumRows, [](auto row) { return row % 3 == 0; }),
      FusedForEncoding::kValues);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
