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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// Tells the receiver how to decode the GPU data in an exchange payload.
enum class ExchangePayloadCodec : uint8_t {
  kNone = 0,
  kFusedFor = 1,
  kCascaded = 2,
  /// Fused FOR of a table in which fixed-width string columns were replaced by
  /// 64-bit lane columns. The lane columns are listed after the cuDF metadata.
  kFusedForLanes = 3,
};

/// A string column whose rows all have the same width, a multiple of eight
/// bytes. The sender replaced it with `lanes` INT64 columns, where lane `l` of
/// row `r` holds bytes [8 * l, 8 * l + 8) of the row. `column` is the position
/// of the string column in the original table.
struct ExchangeLaneColumn {
  uint32_t column;
  uint32_t lanes;

  bool operator==(const ExchangeLaneColumn&) const = default;
};

/// Metadata needed to reconstruct a received exchange payload. `cudfMetadata`
/// contains the original cuDF metadata after the compression header is removed.
struct ExchangePayloadMetadata {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  ExchangePayloadCodec codec{ExchangePayloadCodec::kNone};
  /// Size of the packed GPU data before compression.
  std::size_t logicalDataSize{0};
  /// Number of fused FOR segments. Zero for other codecs.
  std::size_t auxiliaryCount{0};
  /// Columns split into lanes, in increasing column order. Empty unless the
  /// codec is kFusedForLanes.
  std::vector<ExchangeLaneColumn> laneColumns{};
};

/// Prepends the codec, uncompressed byte count, and number of fused FOR
/// segments to serialized cuDF metadata. The segment count is zero for
/// Cascaded. kFusedForLanes also appends `laneColumns`, which must be empty for
/// other codecs. This is used only for compressed payloads.
std::unique_ptr<std::vector<uint8_t>> wrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> cudfMetadata,
    ExchangePayloadCodec codec,
    std::size_t logicalDataSize,
    std::size_t auxiliaryCount,
    const std::vector<ExchangeLaneColumn>& laneColumns = {});

/// Removes and parses the compression header. If `metadata` has no compression
/// header, returns it unchanged with codec kNone.
ExchangePayloadMetadata unwrapExchangePayloadMetadata(
    std::unique_ptr<std::vector<uint8_t>> metadata);

/// Prepends to a payload's metadata the Hive bucket that holds all of its rows,
/// for a receiver that keeps the payload as it is (a materialized exchange in
/// the exchange store). The tag wraps whatever metadata the payload has,
/// including a compression header.
std::unique_ptr<std::vector<uint8_t>> tagPayloadBucket(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    int32_t bucket);

/// Removes the bucket tag of a payload's metadata. Returns the bucket and the
/// metadata as it was before tagPayloadBucket(), or std::nullopt and
/// 'metadata' unchanged when it has no tag.
std::pair<std::optional<int32_t>, std::unique_ptr<std::vector<uint8_t>>>
untagPayloadBucket(std::unique_ptr<std::vector<uint8_t>> metadata);

} // namespace facebook::velox::ucx_exchange
