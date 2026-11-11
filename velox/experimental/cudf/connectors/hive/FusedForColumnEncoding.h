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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/fused_for.hpp>
#include <cudf/types.hpp>

#include <cuda/stream_ref>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// How a column is transformed before it is packed with cuDF's fused
/// frame-of-reference (FOR) codec.
enum class FusedForEncoding : uint8_t {
  /// The values themselves.
  kValues,
  /// Indices into a column of the distinct values (the keys).
  kDictionary,
  /// The differences between consecutive values, restored by a prefix sum.
  kDeltas,
};

/// A column in the encoding chosen by encodeForFusedFor().
struct FusedForEncodedColumn {
  FusedForEncoding encoding{FusedForEncoding::kValues};
  /// The dictionary indices or the differences that take the column's place.
  /// Null for kValues.
  std::unique_ptr<cudf::column> encoded;
  /// The dictionary keys. Null unless the encoding is kDictionary.
  std::unique_ptr<cudf::column> keys;
  /// 'encoded', or the column itself for kValues, packed with bitpacked fused
  /// FOR. Not set for a string column with too many distinct values, which
  /// fused FOR does not compress.
  std::optional<cudf::detail::fused_for_packed_columns> packedValues;
  /// 'keys' packed with bitpacked fused FOR. Set for kDictionary.
  std::optional<cudf::detail::fused_for_packed_columns> packedKeys;
  /// For kDeltas of a column in segments, the value each segment starts from,
  /// one per segment start.
  std::vector<int64_t> segmentBases;
};

/// Returns 'column' without its null mask when it has no nulls. Readers and
/// operators often give a column without nulls a mask (cuDF's Parquet reader
/// does for every optional column), which would rule out every encoding.
cudf::column_view withoutEmptyNullMask(cudf::column_view column);

/// Encodes 'column' in the way that packs it into the fewest bitpacked fused
/// FOR bytes, and packs it. A column without nulls that has at most 65,536
/// distinct values and at least 8 rows per value becomes dictionary indices
/// plus keys: always for strings, and for other types except BOOLEAN when that
/// packs into at most three quarters of the bytes of the values. Otherwise, a
/// BIGINT column without nulls or negative values becomes the differences
/// between consecutive values when that packs into at most three quarters of
/// the bytes. A column without nulls is encoded and packed without its null
/// mask (see withoutEmptyNullMask()). Without 'segmentStarts', the differences start
/// with the first value itself. 'segmentStarts', when given, starts with row 0
/// and lists the rows at which the column is cut into segments that are
/// restored separately: each segment's differences start with zero, and its
/// first value is returned as its base (0 for a segment that starts past the
/// end). A large first value would otherwise widen the frame of reference of
/// the segment's first tile. Memory for the results and temporary memory come
/// from 'tempMr'.
FusedForEncodedColumn encodeForFusedFor(
    cudf::column_view column,
    const std::vector<cudf::size_type>& segmentStarts,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr);

/// Restores a column that was encoded as 'encoding' from 'values', which holds
/// the dictionary indices or the differences of one segment, 'keys', which
/// holds the keys of a dictionary, and 'base', the base of the segment (0
/// without segments). Returns a column from 'mr', or null for kValues, whose
/// column is 'values' itself.
std::unique_ptr<cudf::column> decodeFusedForEncoding(
    FusedForEncoding encoding,
    cudf::column_view values,
    std::optional<cudf::column_view> keys,
    int64_t base,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref tempMr);

} // namespace facebook::velox::cudf_velox::connector::hive
