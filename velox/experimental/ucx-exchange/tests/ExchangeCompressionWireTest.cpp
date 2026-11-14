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
#include "velox/experimental/ucx-exchange/ExchangeCompressionWire.h"

#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr std::size_t kVersionOffset = 8;
constexpr std::size_t kHeaderSize = 37;
constexpr std::size_t kHeaderSizeOffset = 10;
constexpr std::size_t kCodecOffset = 12;
constexpr std::size_t kLogicalSizeOffset = 13;
constexpr std::size_t kAuxiliaryOffset = 21;
constexpr std::size_t kMetadataSizeOffset = 29;

template <typename T>
void putScalar(std::vector<uint8_t>& bytes, std::size_t offset, T value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

auto cascadedEnvelope() {
  return wrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(
          std::initializer_list<uint8_t>{9, 7, 5}),
      ExchangePayloadCodec::kCascaded,
      1234,
      0);
}

// Encodes the expected header independently of the writer under test.
std::vector<uint8_t> referenceEnvelope(uint8_t codec, uint64_t auxiliaryCount) {
  std::vector<uint8_t> expected(kHeaderSize, 0);
  const std::array<uint8_t, 8> magic{'V', 'L', 'X', 'P', 'A', 'C', 'K', 0};
  std::copy(magic.begin(), magic.end(), expected.begin());
  putScalar(expected, kVersionOffset, uint16_t{1});
  putScalar(expected, kHeaderSizeOffset, static_cast<uint16_t>(kHeaderSize));
  putScalar(expected, kCodecOffset, codec);
  putScalar(expected, kLogicalSizeOffset, uint64_t{1234});
  putScalar(expected, kAuxiliaryOffset, auxiliaryCount);
  putScalar(expected, kMetadataSizeOffset, uint64_t{3});
  expected.insert(expected.end(), {9, 7, 5});
  return expected;
}

TEST(ExchangeCompressionWireTest, matchesReferenceEnvelopes) {
  struct Envelope {
    ExchangePayloadCodec codec;
    uint8_t wireCodec;
    uint64_t auxiliaryCount;
  };
  for (const auto& [codec, wireCodec, auxiliaryCount] :
       {Envelope{ExchangePayloadCodec::kCascaded, 2, 0},
        Envelope{ExchangePayloadCodec::kFusedFor, 1, 17}}) {
    SCOPED_TRACE(static_cast<int>(wireCodec));
    auto wrapped = wrapExchangePayloadMetadata(
        std::make_unique<std::vector<uint8_t>>(
            std::initializer_list<uint8_t>{9, 7, 5}),
        codec,
        1234,
        auxiliaryCount);
    EXPECT_EQ(*wrapped, referenceEnvelope(wireCodec, auxiliaryCount));

    auto decoded = unwrapExchangePayloadMetadata(std::move(wrapped));
    EXPECT_EQ(decoded.codec, codec);
    EXPECT_EQ(decoded.logicalDataSize, 1234);
    EXPECT_EQ(decoded.auxiliaryCount, auxiliaryCount);
    EXPECT_EQ(*decoded.cudfMetadata, (std::vector<uint8_t>{9, 7, 5}));
  }
}

TEST(ExchangeCompressionWireTest, forEnvelopePreservesEmptyAndSizeBounds) {
  for (auto size : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
    auto wrapped = wrapExchangePayloadMetadata(
        std::make_unique<std::vector<uint8_t>>(),
        ExchangePayloadCodec::kFusedFor,
        size,
        size);
    auto decoded = unwrapExchangePayloadMetadata(std::move(wrapped));
    EXPECT_EQ(decoded.logicalDataSize, size);
    EXPECT_EQ(decoded.auxiliaryCount, size);
    EXPECT_TRUE(decoded.cudfMetadata->empty());
  }
  EXPECT_THROW(
      wrapExchangePayloadMetadata(
          std::make_unique<std::vector<uint8_t>>(),
          ExchangePayloadCodec::kCascaded,
          10,
          1),
      VeloxRuntimeError);
}

TEST(ExchangeCompressionWireTest, leavesRawMetadataUnchanged) {
  auto raw = std::make_unique<std::vector<uint8_t>>(
      std::initializer_list<uint8_t>{1, 2, 3});
  const auto* original = raw.get();
  auto decoded = unwrapExchangePayloadMetadata(std::move(raw));
  EXPECT_EQ(decoded.codec, ExchangePayloadCodec::kNone);
  EXPECT_EQ(decoded.cudfMetadata.get(), original);
}

TEST(
    ExchangeCompressionWireTest,
    rejectsOtherCodecsInsteadOfTreatingThemAsRaw) {
  for (uint8_t codec : {0, 4, 255}) {
    auto wrapped = cascadedEnvelope();
    putScalar(*wrapped, kCodecOffset, codec);
    EXPECT_THROW(
        unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
    EXPECT_THROW(
        wrapExchangePayloadMetadata(
            std::make_unique<std::vector<uint8_t>>(),
            static_cast<ExchangePayloadCodec>(codec),
            10,
            0),
        VeloxRuntimeError);
  }
}

TEST(ExchangeCompressionWireTest, rejectsMalformedHeader) {
  for (int damage = 0; damage < 4; ++damage) {
    auto wrapped = cascadedEnvelope();
    if (damage == 0) {
      putScalar(*wrapped, kVersionOffset, uint16_t{2});
    } else if (damage == 1) {
      putScalar(*wrapped, kHeaderSizeOffset, uint16_t{36});
    } else if (damage == 2) {
      putScalar(*wrapped, kAuxiliaryOffset, uint64_t{1});
    } else {
      putScalar(*wrapped, kMetadataSizeOffset, uint64_t{4});
    }
    EXPECT_THROW(
        unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
  }
}

auto lanesEnvelope(const std::vector<ExchangeLaneColumn>& laneColumns) {
  return wrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(
          std::initializer_list<uint8_t>{9, 7, 5}),
      ExchangePayloadCodec::kFusedForLanes,
      1234,
      17,
      laneColumns);
}

TEST(ExchangeCompressionWireTest, roundTripsLaneColumns) {
  const std::vector<ExchangeLaneColumn> laneColumns{{1, 4}, {3, 2}};
  auto wrapped = lanesEnvelope(laneColumns);

  // The lane list follows the cuDF metadata: a count, then column and lane
  // count of each entry.
  auto expected = referenceEnvelope(3, 17);
  for (uint32_t value : {2, 1, 4, 3, 2}) {
    expected.resize(expected.size() + sizeof(value));
    putScalar(expected, expected.size() - sizeof(value), value);
  }
  EXPECT_EQ(*wrapped, expected);

  auto decoded = unwrapExchangePayloadMetadata(std::move(wrapped));
  EXPECT_EQ(decoded.codec, ExchangePayloadCodec::kFusedForLanes);
  EXPECT_EQ(decoded.logicalDataSize, 1234);
  EXPECT_EQ(decoded.auxiliaryCount, 17);
  EXPECT_EQ(*decoded.cudfMetadata, (std::vector<uint8_t>{9, 7, 5}));
  EXPECT_EQ(decoded.laneColumns, laneColumns);
}

TEST(ExchangeCompressionWireTest, rejectsInvalidLaneColumns) {
  const std::vector<ExchangeLaneColumn> laneColumns{{1, 4}};
  // Lane columns belong only to the lanes codec, which needs at least one.
  EXPECT_THROW(
      wrapExchangePayloadMetadata(
          std::make_unique<std::vector<uint8_t>>(),
          ExchangePayloadCodec::kFusedFor,
          10,
          1,
          laneColumns),
      VeloxRuntimeError);
  EXPECT_THROW(lanesEnvelope({}), VeloxRuntimeError);
  EXPECT_THROW(lanesEnvelope({{1, 0}}), VeloxRuntimeError);
  EXPECT_THROW(lanesEnvelope({{3, 4}, {1, 4}}), VeloxRuntimeError);
  EXPECT_THROW(lanesEnvelope({{1, 4}, {1, 4}}), VeloxRuntimeError);

  // A truncated or overlong lane list, and a lanes codec without a list.
  auto wrapped = lanesEnvelope(laneColumns);
  wrapped->pop_back();
  EXPECT_THROW(
      unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
  wrapped = lanesEnvelope(laneColumns);
  wrapped->push_back(0);
  EXPECT_THROW(
      unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
  wrapped = cascadedEnvelope();
  putScalar(*wrapped, kCodecOffset, uint8_t{3});
  EXPECT_THROW(
      unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
  wrapped = lanesEnvelope(laneColumns);
  putScalar(*wrapped, kMetadataSizeOffset, std::numeric_limits<uint64_t>::max());
  EXPECT_THROW(
      unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
}

TEST(ExchangeCompressionWireTest, tagsAndUntagsPayloadBucket) {
  // A tag wraps any metadata, including a compression envelope, and comes off
  // leaving it byte for byte.
  const auto envelope = wrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3}),
      ExchangePayloadCodec::kFusedFor,
      100,
      2);
  for (const auto& metadata :
       {std::vector<uint8_t>{}, std::vector<uint8_t>{9, 8, 7}, *envelope}) {
    auto [bucket, untagged] = untagPayloadBucket(tagPayloadBucket(
        std::make_unique<std::vector<uint8_t>>(metadata), 1'023));
    EXPECT_EQ(bucket, 1'023);
    ASSERT_NE(untagged, nullptr);
    EXPECT_EQ(*untagged, metadata);
  }
  // Metadata without a tag passes through.
  auto [bucket, untagged] =
      untagPayloadBucket(std::make_unique<std::vector<uint8_t>>(*envelope));
  EXPECT_FALSE(bucket.has_value());
  EXPECT_EQ(*untagged, *envelope);
  EXPECT_FALSE(untagPayloadBucket(nullptr).first.has_value());
}

TEST(ExchangeCompressionWireTest, rejectsTruncatedEnvelope) {
  auto wrapped = cascadedEnvelope();
  wrapped->resize(kHeaderSize - 1);
  EXPECT_THROW(
      unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
  wrapped = cascadedEnvelope();
  wrapped->pop_back();
  EXPECT_THROW(
      unwrapExchangePayloadMetadata(std::move(wrapped)), VeloxRuntimeError);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
