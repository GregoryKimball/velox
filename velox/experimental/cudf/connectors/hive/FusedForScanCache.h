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
#include <cudf/contiguous_split.hpp>
#include <cudf/detail/fused_for.hpp>

#include <cuda/stream_ref>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Process-wide warm cache of Parquet file columns re-encoded with cuDF's
/// fused frame-of-reference codec (the exchange's fused-for-bitpacked layout)
/// and held in pinned host memory. A column is filled once per file from a
/// full, unfiltered read and then served by copying its encoded bytes to the
/// device and decoding them, in place of fetching, decompressing and decoding
/// the Parquet column chunks. String columns with few distinct values are
/// stored as frame-of-reference coded dictionary indices plus their keys and
/// gathered on read; other string columns are not cached. Fixed-width columns
/// with few distinct values are stored the same way when that is clearly
/// smaller than coding the values themselves, and non-negative BIGINT columns
/// (keys) as the differences between consecutive values, restored by a prefix
/// sum on read, under the same condition. Entries are never
/// evicted: once the capacity is reached, the cache is full and further
/// columns are read from Parquet.
class FusedForScanCache {
 public:
  static FusedForScanCache& instance();

  /// Raises the capacity in bytes of encoded data to at least 'bytes'.
  void reserveCapacity(uint64_t bytes);

  /// Returns whether every column in 'columns' of 'file' is cached.
  bool containsAll(
      std::string_view file,
      const std::vector<std::string>& columns) const;

  /// Returns whether the columns of 'file' that are not cached yet could still
  /// be: the cache is not full and none of them was found uncacheable.
  bool mayCache(std::string_view file, const std::vector<std::string>& columns)
      const;

  /// Encodes 'column' and caches it as column 'name' of 'file'. Returns false
  /// when the column is not cacheable or the cache is full. A column that is
  /// already cached is kept.
  bool insert(
      std::string_view file,
      std::string_view name,
      cudf::column_view column,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref tempMr);

  /// A cached column decoded on the device. 'view' points into memory the
  /// object owns: the decoded packed allocation, or 'column'.
  struct DecodedColumn {
    cudf::packed_columns packed;
    std::unique_ptr<cudf::column> column;
    cudf::column_view view;
  };

  /// Copies the cached column 'name' of 'file' to the device and decodes it
  /// into memory from 'tempMr', except that a dictionary-encoded column is
  /// gathered, and a difference-encoded column summed, into a column from
  /// 'mr'. The column must be cached.
  DecodedColumn decode(
      std::string_view file,
      std::string_view name,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      rmm::device_async_resource_ref tempMr) const;

 private:
  // One fused FOR payload: the packed-table metadata and the encoded bytes in
  // pinned host memory.
  struct Payload {
    std::vector<uint8_t> metadata;
    const void* wire{nullptr};
    std::size_t wireBytes{0};
    std::size_t segmentCount{0};
    std::size_t logicalDataSize{0};
  };

  // A column, the indices of a dictionary-encoded column and its keys, or the
  // differences between consecutive values of a column ('deltas').
  struct Entry {
    Payload values;
    std::optional<Payload> keys;
    bool deltas{false};
  };

  static std::string key(std::string_view file, std::string_view name);

  // Copies a payload to the device and decodes its single column into a
  // packed allocation from 'mr', returned with a view of the column.
  static std::pair<cudf::packed_columns, cudf::column_view> decodePayload(
      const Payload& payload,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  // Moves an encoded payload to pinned memory. Called with 'mutex_' held
  // exclusively.
  Payload store(
      cudf::detail::fused_for_packed_columns&& packed,
      cuda::stream_ref stream);

  // Returns pinned host memory for 'bytes', from the current block or a new
  // one. Called with 'mutex_' held exclusively.
  void* allocatePinned(std::size_t bytes);

  mutable std::shared_mutex mutex_;
  // Node map: an entry stays at its address while others are inserted.
  folly::F14NodeMap<std::string, Entry> entries_;
  folly::F14FastSet<std::string> uncacheable_;
  std::vector<void*> blocks_;
  char* blockCursor_{nullptr};
  std::size_t blockRemaining_{0};
  std::atomic<uint64_t> capacity_{0};
  std::atomic<uint64_t> usedBytes_{0};
  std::atomic<bool> full_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
