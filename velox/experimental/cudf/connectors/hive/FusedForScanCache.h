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
/// columns are read from Parquet. The first payloads up to a device capacity
/// are also copied to device memory, so that reading them skips the
/// host-to-device copy. When a device allocation fails, the cache frees its
/// newest device blocks (see releaseDeviceMemory() in GpuResources.h) and reads
/// their payloads from host memory, and the device tier stays at the smaller
/// size.
class FusedForScanCache {
 public:
  static FusedForScanCache& instance();

  /// Raises the capacity in bytes of encoded data to at least 'bytes'.
  void reserveCapacity(uint64_t bytes);

  /// Raises the bytes of encoded data held in device memory to at least
  /// 'bytes'.
  void reserveDeviceCapacity(uint64_t bytes);

  /// Sets the bytes of encoded data held in device memory to 'bytes' for the
  /// query about to run, which needs the rest of the device: frees device
  /// blocks above it now, and lets the device tier grow back up to it,
  /// whatever earlier queries shrank it to.
  void limitDeviceCapacity(uint64_t bytes);

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

  /// Returns about how many device bytes decoding one slice of 'columns' of
  /// 'file' takes: each column's decoded size over its slices, which for a
  /// dictionary-encoded column counts its 4-byte indices twice for the
  /// gathered values. The columns must be cached.
  std::size_t decodedSliceBytes(
      std::string_view file,
      const std::vector<std::string>& columns) const;

  /// Returns the rows per slice of a cached column: a column is cached and
  /// decoded in slices of this many rows (the last may be shorter), so that
  /// decoding a file takes device memory in proportion to the memory the
  /// process can use.
  static cudf::size_type sliceRows();

  /// Makes sliceRows() return 'rows' instead, for tests; zero undoes it.
  static void testingSetSliceRows(cudf::size_type rows);

  /// Returns the number of slices of the cached column 'name' of 'file'. The
  /// column must be cached.
  std::size_t numSlices(std::string_view file, std::string_view name) const;

  /// A cached column decoded on the device. 'view' points into memory the
  /// object owns: the decoded packed allocation, or 'column'.
  struct DecodedColumn {
    cudf::packed_columns packed;
    std::unique_ptr<cudf::column> column;
    cudf::column_view view;
  };

  /// Copies slice 'slice' of the cached column 'name' of 'file' to the device
  /// and decodes it into memory from 'tempMr', except that a dictionary-encoded
  /// column is gathered, and a difference-encoded column summed, into a column
  /// from 'mr'. The column must be cached.
  DecodedColumn decode(
      std::string_view file,
      std::string_view name,
      std::size_t slice,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      rmm::device_async_resource_ref tempMr) const;

 private:
  // One fused FOR payload: the packed-table metadata and the encoded bytes in
  // pinned host memory, with a copy in device memory while the device tier
  // holds one.
  struct Payload {
    std::vector<uint8_t> metadata;
    const void* wire{nullptr};
    const void* deviceWire{nullptr};
    std::size_t wireBytes{0};
    std::size_t segmentCount{0};
    std::size_t logicalDataSize{0};
  };

  // A column, the indices of a dictionary-encoded column and its keys, or the
  // differences between consecutive values of a column ('deltas'), in slices
  // of sliceRows() rows packed one per payload. The keys are packed whole.
  struct Entry {
    std::vector<Payload> values;
    std::optional<Payload> keys;
    bool deltas{false};
    // For differences in several slices, the value each slice starts from.
    std::vector<int64_t> sliceBases;
  };

  static std::string key(std::string_view file, std::string_view name);

  // Copies a payload to the device and decodes its single column into a
  // packed allocation from 'mr', returned with a view of the column.
  std::pair<cudf::packed_columns, cudf::column_view> decodePayload(
      const Payload& payload,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  // Frees the newest device blocks, dropping the device copies they hold,
  // until at least 'bytes' are freed or none is left. Returns whether any was
  // freed.
  bool releaseDeviceBlocks(std::size_t bytes);

  // Copies an encoded payload to pinned memory and, while the device capacity
  // allows, to device memory. Called with 'mutex_' held exclusively.
  Payload store(
      cudf::detail::fused_for_packed_columns&& packed,
      cuda::stream_ref stream);

  // Returns pinned host memory for 'bytes', from the current block or a new
  // one. Called with 'mutex_' held exclusively.
  void* allocatePinned(std::size_t bytes);

  // Returns device memory for 'bytes' within the device capacity, or null.
  // Called with 'mutex_' held exclusively.
  void* allocateDevice(std::size_t bytes);

  mutable std::shared_mutex mutex_;
  // Node map: an entry stays at its address while others are inserted.
  folly::F14NodeMap<std::string, Entry> entries_;
  folly::F14FastSet<std::string> uncacheable_;
  std::vector<void*> blocks_;
  char* blockCursor_{nullptr};
  std::size_t blockRemaining_{0};
  std::atomic<uint64_t> capacity_{0};
  // Device blocks and their sizes, oldest first.
  std::vector<std::pair<void*, std::size_t>> deviceBlocks_;
  bool releaserRegistered_{false};
  char* deviceCursor_{nullptr};
  std::size_t deviceRemaining_{0};
  std::size_t deviceAllocatedBytes_{0};
  std::atomic<uint64_t> deviceCapacity_{0};
  // Set once the device tier was capped for lack of device memory (a failed
  // block allocation or releaseDeviceBlocks()); reserveDeviceCapacity() then
  // leaves the capacity alone.
  std::atomic<bool> deviceTierShrunk_{false};
  std::atomic<uint64_t> usedBytes_{0};
  std::atomic<bool> full_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
