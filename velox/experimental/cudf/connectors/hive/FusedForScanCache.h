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

#include <cuda/stream_ref>
#include <folly/container/F14Map.h>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
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
/// the Parquet column chunks. Entries are never evicted: once the capacity is
/// reached, further columns are not cached and are read from Parquet.
class FusedForScanCache {
 public:
  static FusedForScanCache& instance();

  /// Raises the capacity in bytes of encoded data to at least 'bytes'.
  void reserveCapacity(uint64_t bytes);

  /// Returns whether every column in 'columns' of 'file' is cached.
  bool containsAll(
      std::string_view file,
      const std::vector<std::string>& columns) const;

  /// Encodes 'column' and caches it as column 'name' of 'file'. Returns false
  /// when the cache is full. A column that is already cached is kept.
  bool insert(
      std::string_view file,
      std::string_view name,
      cudf::column_view column,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref tempMr);

  /// Copies the cached column 'name' of 'file' to the device and decodes it.
  /// The column must be cached.
  std::unique_ptr<cudf::column> read(
      std::string_view file,
      std::string_view name,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      rmm::device_async_resource_ref tempMr) const;

  /// Bytes of encoded data cached so far.
  uint64_t usedBytes() const {
    return usedBytes_.load(std::memory_order_relaxed);
  }

 private:
  struct Entry {
    std::vector<uint8_t> metadata;
    const void* wire{nullptr};
    std::size_t wireBytes{0};
    std::size_t segmentCount{0};
    std::size_t logicalDataSize{0};
  };

  static std::string key(std::string_view file, std::string_view name);

  // Returns pinned host memory for 'bytes', from the current block or a new
  // one. Called with 'mutex_' held exclusively.
  void* allocatePinned(std::size_t bytes);

  mutable std::shared_mutex mutex_;
  // Node map: an entry stays at its address while others are inserted.
  folly::F14NodeMap<std::string, Entry> entries_;
  std::vector<void*> blocks_;
  char* blockCursor_{nullptr};
  std::size_t blockRemaining_{0};
  std::atomic<uint64_t> capacity_{0};
  std::atomic<uint64_t> usedBytes_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
