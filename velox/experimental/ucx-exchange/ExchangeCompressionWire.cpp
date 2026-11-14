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
#include <type_traits>
#include <utility>

namespace facebook::velox::ucx_exchange {
namespace {

constexpr std::array<uint8_t, 8> kMagic{'V', 'L', 'X', 'P', 'A', 'C', 'K', 0};
constexpr uint16_t kVersion = 1;
constexpr std::size_t kHeaderSize = kMagic.size() + sizeof(kVersion) +
    sizeof(uint16_t) + sizeof(uint8_t) + 3 * sizeof(uint64_t);

static_assert(kHeaderSize == 37);
static_assert(sizeof(ExchangeLaneColumn) == 2 * sizeof(uint32_t));

template <typename T>
void appendScalar(std::vector<uint8_t>& output, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto position = output.size();
  output.resize(position + sizeof(T));
  std::memcpy(output.data() + position, &value, sizeof(T));
}

// unwrapExchangePayloadMetadata() checks that the complete fixed-size header
// is present before calling this function.
template <typename T>
T readScalar(const uint8_t*& current) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value;
  std::memcpy(&value, current, sizeof(T));
  current += sizeof(T);
  return value;
}

void checkEnvelopeCodec(ExchangePayloadCodec codec, uint64_t auxiliaryCount) {
  VELOX_CHECK(
      codec == ExchangePayloadCodec::kFusedFor ||
          codec == ExchangePayloadCodec::kCascaded ||
          codec == ExchangePayloadCodec::kFusedForLanes,
      "Unsupported exchange payload codec: {}",
      static_cast<uint8_t>(codec));
  VELOX_CHECK(
      codec != ExchangePayloadCodec::kCascaded || auxiliaryCount == 0,
      "Cascaded cannot carry an auxiliary segment count: {}",
      auxiliaryCount);
}

// Only kFusedForLanes lists lane columns, and it lists at least one. Columns
// are in increasing order and each is split into at least one lane.
void checkLaneColumns(
    ExchangePayloadCodec codec,
    const std::vector<ExchangeLaneColumn>& laneColumns) {
  VELOX_CHECK_EQ(
      codec == ExchangePayloadCodec::kFusedForLanes,
      !laneColumns.empty(),
      "Lane columns must be present exactly for the fused FOR lanes codec");
  for (size_t i = 0; i < laneColumns.size(); ++i) {
    VELOX_CHECK_GT(laneColumns[i].lanes, 0, "Lane column without lanes");
    VELOX_CHECK(
        i == 0 || laneColumns[i - 1].column < laneColumns[i].column,
        "Lane columns are not in increasing column order");
  }
}

} // namespace

std::unique_ptr<std::vector<uint8_t>> wrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> cudfMetadata,
    ExchangePayloadCodec codec,
    std::size_t logicalDataSize,
    std::size_t auxiliaryCount,
    const std::vector<ExchangeLaneColumn>& laneColumns) {
  VELOX_CHECK_NOT_NULL(cudfMetadata);
  checkEnvelopeCodec(codec, auxiliaryCount);
  checkLaneColumns(codec, laneColumns);

  auto output = std::make_unique<std::vector<uint8_t>>();
  output->reserve(
      kHeaderSize + cudfMetadata->size() +
      (laneColumns.empty()
           ? 0
           : sizeof(uint32_t) +
               laneColumns.size() * sizeof(ExchangeLaneColumn)));
  output->insert(output->end(), kMagic.begin(), kMagic.end());
  appendScalar(*output, kVersion);
  appendScalar(*output, static_cast<uint16_t>(kHeaderSize));
  appendScalar(*output, static_cast<uint8_t>(codec));
  appendScalar(*output, static_cast<uint64_t>(logicalDataSize));
  appendScalar(*output, static_cast<uint64_t>(auxiliaryCount));
  appendScalar(*output, static_cast<uint64_t>(cudfMetadata->size()));
  output->insert(output->end(), cudfMetadata->begin(), cudfMetadata->end());
  if (!laneColumns.empty()) {
    appendScalar(*output, static_cast<uint32_t>(laneColumns.size()));
    for (const auto& laneColumn : laneColumns) {
      appendScalar(*output, laneColumn.column);
      appendScalar(*output, laneColumn.lanes);
    }
  }
  return output;
}

ExchangePayloadMetadata unwrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> metadata) {
  VELOX_CHECK_NOT_NULL(metadata);
  if (metadata->size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), metadata->begin())) {
    return ExchangePayloadMetadata{std::move(metadata)};
  }

  VELOX_CHECK_GE(
      metadata->size(),
      kHeaderSize,
      "Truncated exchange compression metadata envelope");
  const auto* current = metadata->data() + kMagic.size();
  const auto* end = metadata->data() + metadata->size();
  const auto version = readScalar<uint16_t>(current);
  const auto headerSize = readScalar<uint16_t>(current);
  const auto codecValue = readScalar<uint8_t>(current);
  const auto logicalDataSize = readScalar<uint64_t>(current);
  const auto auxiliaryCount = readScalar<uint64_t>(current);
  const auto cudfMetadataSize = readScalar<uint64_t>(current);

  VELOX_CHECK_EQ(
      version, kVersion, "Unsupported exchange compression metadata version");
  VELOX_CHECK_EQ(
      headerSize, kHeaderSize, "Invalid exchange compression header size");
  const auto codec = static_cast<ExchangePayloadCodec>(codecValue);
  checkEnvelopeCodec(codec, auxiliaryCount);
  const auto remaining = static_cast<uint64_t>(end - current);
  if (codec != ExchangePayloadCodec::kFusedForLanes) {
    VELOX_CHECK_EQ(
        cudfMetadataSize, remaining, "Invalid exchange cuDF metadata size");
  } else {
    VELOX_CHECK_LE(
        cudfMetadataSize, remaining, "Invalid exchange cuDF metadata size");
    VELOX_CHECK_GE(
        remaining - cudfMetadataSize,
        sizeof(uint32_t),
        "Truncated exchange lane column list");
  }
  const auto* cudfMetadataBegin = current;
  current += cudfMetadataSize;

  std::vector<ExchangeLaneColumn> laneColumns;
  if (codec == ExchangePayloadCodec::kFusedForLanes) {
    const auto count = readScalar<uint32_t>(current);
    VELOX_CHECK_EQ(
        static_cast<uint64_t>(end - current),
        static_cast<uint64_t>(count) * sizeof(ExchangeLaneColumn),
        "Invalid exchange lane column list size");
    laneColumns.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto column = readScalar<uint32_t>(current);
      const auto lanes = readScalar<uint32_t>(current);
      laneColumns.push_back(ExchangeLaneColumn{column, lanes});
    }
  }
  checkLaneColumns(codec, laneColumns);

  auto cudfMetadata = std::make_unique<std::vector<uint8_t>>(
      cudfMetadataBegin,
      cudfMetadataBegin + static_cast<std::size_t>(cudfMetadataSize));
  return ExchangePayloadMetadata{
      std::move(cudfMetadata),
      codec,
      static_cast<std::size_t>(logicalDataSize),
      static_cast<std::size_t>(auxiliaryCount),
      std::move(laneColumns)};
}

namespace {

constexpr std::array<uint8_t, 8> kBucketMagic{
    'V', 'L', 'X', 'B', 'U', 'C', 'K', 0};
constexpr std::size_t kBucketTagSize = kBucketMagic.size() + sizeof(int32_t);

} // namespace

std::unique_ptr<std::vector<uint8_t>> tagPayloadBucket(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    int32_t bucket) {
  VELOX_CHECK_NOT_NULL(metadata);
  VELOX_CHECK_GE(bucket, 0);
  auto tagged = std::make_unique<std::vector<uint8_t>>(
      kBucketMagic.begin(), kBucketMagic.end());
  tagged->reserve(kBucketTagSize + metadata->size());
  appendScalar(*tagged, bucket);
  tagged->insert(tagged->end(), metadata->begin(), metadata->end());
  return tagged;
}

std::pair<std::optional<int32_t>, std::unique_ptr<std::vector<uint8_t>>>
untagPayloadBucket(std::unique_ptr<std::vector<uint8_t>> metadata) {
  if (metadata == nullptr || metadata->size() < kBucketTagSize ||
      !std::equal(kBucketMagic.begin(), kBucketMagic.end(), metadata->begin())) {
    return {std::nullopt, std::move(metadata)};
  }
  const uint8_t* current = metadata->data() + kBucketMagic.size();
  const auto bucket = readScalar<int32_t>(current);
  VELOX_CHECK_GE(bucket, 0, "Malformed payload bucket tag");
  return {
      bucket,
      std::make_unique<std::vector<uint8_t>>(
          metadata->begin() + kBucketTagSize, metadata->end())};
}

} // namespace facebook::velox::ucx_exchange
