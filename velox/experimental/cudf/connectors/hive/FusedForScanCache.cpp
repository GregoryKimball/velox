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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/FusedForScanCache.h"
#include "velox/experimental/cudf/connectors/hive/FusedForColumnEncoding.h"

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/cudf/exec/GpuCapabilities.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <glog/logging.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Pinned memory is allocated in blocks of this size; larger entries get their
// own allocation.
constexpr std::size_t kBlockBytes = std::size_t{1} << 30;
constexpr std::size_t kAlignment = 256;

// Returns whether the device tier may hold 'tierBytes': at most half the
// device memory this process can use, and only while the operators'
// allocations plus the tier stay under three quarters of it, so that the tier
// does not take memory that queries need. A failed allocation shrinks the tier
// further (releaseDeviceBlocks()).
bool deviceTierFits(std::size_t tierBytes) {
  const auto available = gpuCapabilities().availableMemoryBytes;
  if (available <= 0) {
    return true;
  }
  const auto tier = static_cast<int64_t>(tierBytes);
  const auto allocated = std::max<int64_t>(cudfAllocatedBytes(), 0);
  return tier <= available / 2 && allocated + tier <= available / 4 * 3;
}

// A slice of a cached file decodes to about this many bytes per row, the
// width of a typical scan's columns; slices take 1/kSliceShare of the device
// memory available to the process. Concatenation downstream keeps batches
// within the same share (CudfBatchConcat).
constexpr int64_t kNominalRowBytes = 32;
constexpr int64_t kSliceShare = 64;
constexpr cudf::size_type kMinSliceRows = 1 << 20;

std::atomic<cudf::size_type> testingSliceRows{0};

std::size_t alignUp(std::size_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

void* allocatePinnedBlock(std::size_t bytes) {
  void* ptr = nullptr;
  const auto status = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
  VELOX_CHECK(
      status == cudaSuccess,
      "Failed to allocate {} bytes of pinned memory for the fused FOR scan cache: {}",
      bytes,
      cudaGetErrorString(status));
  return ptr;
}

} // namespace

FusedForScanCache& FusedForScanCache::instance() {
  static FusedForScanCache cache;
  return cache;
}

void FusedForScanCache::reserveCapacity(uint64_t bytes) {
  auto current = capacity_.load(std::memory_order_relaxed);
  while (current < bytes &&
         !capacity_.compare_exchange_weak(
             current, bytes, std::memory_order_relaxed)) {
  }
}

void FusedForScanCache::reserveDeviceCapacity(uint64_t bytes) {
  // A tier shrunk for a failed allocation stays at its smaller size: raising
  // it again would only take the memory back from the operators.
  if (deviceTierShrunk_.load(std::memory_order_relaxed)) {
    return;
  }
  auto current = deviceCapacity_.load(std::memory_order_relaxed);
  while (current < bytes &&
         !deviceCapacity_.compare_exchange_weak(
             current, bytes, std::memory_order_relaxed)) {
  }
}

void FusedForScanCache::limitDeviceCapacity(uint64_t bytes) {
  if (deviceCapacity_.load(std::memory_order_relaxed) == bytes &&
      !deviceTierShrunk_.load(std::memory_order_relaxed)) {
    return;
  }
  std::size_t allocated = 0;
  {
    std::shared_lock lock(mutex_);
    allocated = deviceAllocatedBytes_;
  }
  if (allocated > bytes) {
    releaseDeviceBlocks(allocated - bytes);
  }
  deviceCapacity_.store(bytes, std::memory_order_relaxed);
  deviceTierShrunk_.store(false, std::memory_order_relaxed);
}

std::string FusedForScanCache::key(
    std::string_view file,
    std::string_view name) {
  std::string key;
  key.reserve(file.size() + 1 + name.size());
  key.append(file).push_back('\0');
  key.append(name);
  return key;
}

bool FusedForScanCache::containsAll(
    std::string_view file,
    const std::vector<std::string>& columns) const {
  std::shared_lock lock(mutex_);
  for (const auto& name : columns) {
    if (!entries_.contains(key(file, name))) {
      return false;
    }
  }
  return true;
}

std::size_t FusedForScanCache::decodedSliceBytes(
    std::string_view file,
    const std::vector<std::string>& columns) const {
  std::shared_lock lock(mutex_);
  std::size_t bytes = 0;
  for (const auto& name : columns) {
    const auto it = entries_.find(key(file, name));
    VELOX_CHECK(it != entries_.end(), "Column is not cached: {}", name);
    const auto& entry = it->second;
    std::size_t values = 0;
    for (const auto& slice : entry.values) {
      values = std::max(values, slice.logicalDataSize);
    }
    bytes += entry.keys.has_value()
        ? 2 * values + entry.keys->logicalDataSize
        : values;
  }
  return bytes;
}

void FusedForScanCache::testingSetSliceRows(cudf::size_type rows) {
  testingSliceRows.store(rows, std::memory_order_relaxed);
}

cudf::size_type FusedForScanCache::sliceRows() {
  if (const auto rows = testingSliceRows.load(std::memory_order_relaxed);
      rows > 0) {
    return rows;
  }
  const auto available = gpuCapabilities().availableMemoryBytes;
  if (available <= 0) {
    return std::numeric_limits<cudf::size_type>::max();
  }
  return static_cast<cudf::size_type>(std::clamp<int64_t>(
      available / kSliceShare / kNominalRowBytes,
      kMinSliceRows,
      std::numeric_limits<cudf::size_type>::max()));
}

std::size_t FusedForScanCache::numSlices(
    std::string_view file,
    std::string_view name) const {
  std::shared_lock lock(mutex_);
  const auto it = entries_.find(key(file, name));
  VELOX_CHECK(it != entries_.end(), "Column is not cached: {}", name);
  return it->second.values.size();
}

bool FusedForScanCache::mayCache(
    std::string_view file,
    const std::vector<std::string>& columns) const {
  if (full_.load(std::memory_order_relaxed)) {
    return false;
  }
  std::shared_lock lock(mutex_);
  for (const auto& name : columns) {
    if (uncacheable_.contains(key(file, name))) {
      return false;
    }
  }
  return true;
}

bool FusedForScanCache::insert(
    std::string_view file,
    std::string_view name,
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr) {
  auto entryKey = key(file, name);
  {
    std::shared_lock lock(mutex_);
    if (entries_.contains(entryKey)) {
      return true;
    }
    if (uncacheable_.contains(entryKey)) {
      return false;
    }
  }
  if (full_.load(std::memory_order_relaxed)) {
    return false;
  }

  // Rows at which slices start, when the column takes more than one.
  std::vector<cudf::size_type> sliceStarts;
  const auto rowsPerSlice = sliceRows();
  if (column.size() > rowsPerSlice) {
    for (cudf::size_type start = 0; start < column.size();
         start += std::min(rowsPerSlice, column.size() - start)) {
      sliceStarts.push_back(start);
    }
  }
  auto encodedColumn = encodeForFusedFor(column, sliceStarts, stream, tempMr);
  if (!encodedColumn.packedValues.has_value()) {
    // A string column with many distinct values: fused FOR would not compress
    // it.
    std::unique_lock lock(mutex_);
    uncacheable_.insert(std::move(entryKey));
    return false;
  }
  // The encoding is chosen on the whole column; with several slices, the
  // encoded values are packed again, one payload per slice.
  std::vector<cudf::detail::fused_for_packed_columns> values;
  if (sliceStarts.size() > 1) {
    const auto source = encodedColumn.encoded != nullptr
        ? encodedColumn.encoded->view()
        : withoutEmptyNullMask(column);
    cudf::detail::fused_for_options options;
    options.layout = cudf::detail::fused_for_layout::bitpacked;
    values = cudf::detail::contiguous_split_fused_for(
        cudf::table_view{{source}},
        std::vector<cudf::size_type>(sliceStarts.begin() + 1, sliceStarts.end()),
        stream,
        tempMr,
        options);
  } else {
    values.push_back(std::move(*encodedColumn.packedValues));
  }
  auto& keys = encodedColumn.packedKeys;
  const bool deltas = encodedColumn.encoding == FusedForEncoding::kDeltas;
  const auto payloadBytes =
      [](const cudf::detail::fused_for_packed_columns& packed) {
        return packed.wire_data->size() + packed.metadata->size();
      };
  auto bytes = keys.has_value() ? payloadBytes(*keys) : 0;
  for (const auto& slice : values) {
    bytes += payloadBytes(slice);
  }

  std::unique_lock lock(mutex_);
  if (entries_.contains(entryKey)) {
    return true;
  }
  if (usedBytes_.load(std::memory_order_relaxed) + bytes >
      capacity_.load(std::memory_order_relaxed)) {
    full_.store(true, std::memory_order_relaxed);
    return false;
  }
  Entry entry;
  entry.values.reserve(values.size());
  for (auto& slice : values) {
    entry.values.push_back(store(std::move(slice), stream));
  }
  if (keys.has_value()) {
    entry.keys = store(std::move(*keys), stream);
  }
  entry.deltas = deltas;
  if (deltas && sliceStarts.size() > 1) {
    entry.sliceBases = std::move(encodedColumn.segmentBases);
  }
  stream.sync();
  entries_.emplace(std::move(entryKey), std::move(entry));
  usedBytes_.fetch_add(bytes, std::memory_order_relaxed);
  return true;
}

FusedForScanCache::Payload FusedForScanCache::store(
    cudf::detail::fused_for_packed_columns&& packed,
    cuda::stream_ref stream) {
  Payload payload;
  payload.wireBytes = packed.wire_data->size();
  void* host = allocatePinned(payload.wireBytes);
  auto status = cudaMemcpyAsync(
      host,
      packed.wire_data->data(),
      payload.wireBytes,
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));
  payload.wire = host;
  // The host copy stays, so that releasing the device copy needs no copy back.
  if (void* device = allocateDevice(payload.wireBytes)) {
    status = cudaMemcpyAsync(
        device,
        packed.wire_data->data(),
        payload.wireBytes,
        cudaMemcpyDeviceToDevice,
        stream.get());
    VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));
    payload.deviceWire = device;
  }
  payload.metadata = std::move(*packed.metadata);
  payload.segmentCount = packed.segment_count;
  payload.logicalDataSize = packed.logical_data_size;
  // The device buffer is released on 'stream' after the copies above.
  packed.wire_data.reset();
  return payload;
}

std::pair<cudf::packed_columns, cudf::column_view>
FusedForScanCache::decodePayload(
    const Payload& payload,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  // Allocate before taking the lock: a failed allocation makes the cache
  // release device blocks, which takes the lock exclusively.
  rmm::device_buffer wire(payload.wireBytes, stream, mr);
  {
    // The device copy can be released: copy from it under the lock. Releasing
    // a block synchronizes the device first, so no copy from it is in flight
    // when it is freed.
    std::shared_lock lock(mutex_);
    const bool onDevice = payload.deviceWire != nullptr;
    const auto status = cudaMemcpyAsync(
        wire.data(),
        onDevice ? payload.deviceWire : payload.wire,
        payload.wireBytes,
        onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
        stream.get());
    VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));
  }

  cudf::detail::fused_for_packed_columns packed;
  packed.metadata = std::make_unique<std::vector<uint8_t>>(payload.metadata);
  packed.wire_data = std::make_unique<rmm::device_buffer>(std::move(wire));
  packed.segment_count = payload.segmentCount;
  packed.logical_data_size = payload.logicalDataSize;
  auto columns = cudf::detail::decode_fused_for(std::move(packed), stream, mr);
  const auto view = cudf::unpack(columns).column(0);
  return {std::move(columns), view};
}

FusedForScanCache::DecodedColumn FusedForScanCache::decode(
    std::string_view file,
    std::string_view name,
    std::size_t slice,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref tempMr) const {
  const Entry* entry = nullptr;
  {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(key(file, name));
    VELOX_CHECK(
        it != entries_.end(),
        "Column {} of {} is not in the fused FOR scan cache",
        name,
        file);
    entry = &it->second;
  }

  VELOX_CHECK_LT(slice, entry->values.size(), "No such slice of {}", name);
  auto [values, valuesView] =
      decodePayload(entry->values[slice], stream, tempMr);
  if (entry->deltas) {
    auto column = decodeFusedForEncoding(
        FusedForEncoding::kDeltas,
        valuesView,
        std::nullopt,
        entry->sliceBases.empty() ? 0 : entry->sliceBases[slice],
        stream,
        mr,
        tempMr);
    const auto view = column->view();
    return DecodedColumn{{}, std::move(column), view};
  }
  if (!entry->keys.has_value()) {
    return DecodedColumn{std::move(values), nullptr, valuesView};
  }
  const auto [keys, keysView] = decodePayload(*entry->keys, stream, tempMr);
  auto column = decodeFusedForEncoding(
      FusedForEncoding::kDictionary, valuesView, keysView, 0, stream, mr, tempMr);
  const auto view = column->view();
  return DecodedColumn{{}, std::move(column), view};
}

void* FusedForScanCache::allocateDevice(std::size_t bytes) {
  bytes = alignUp(bytes);
  const auto capacity = deviceCapacity_.load(std::memory_order_relaxed);
  if (bytes > deviceRemaining_) {
    // Whole blocks count against the capacity, so the device never holds more
    // than it.
    const auto blockBytes = std::max(bytes, kBlockBytes);
    if (deviceAllocatedBytes_ + blockBytes > capacity ||
        !deviceTierFits(deviceAllocatedBytes_ + blockBytes)) {
      return nullptr;
    }
    void* block = nullptr;
    if (cudaMalloc(&block, blockBytes) != cudaSuccess) {
      // Out of device memory: keep the rest of the cache in host memory.
      (void)cudaGetLastError();
      deviceCapacity_.store(deviceAllocatedBytes_, std::memory_order_relaxed);
      deviceTierShrunk_.store(true, std::memory_order_relaxed);
      return nullptr;
    }
    if (!releaserRegistered_) {
      registerDeviceMemoryReleaser(
          [this](std::size_t bytes) { return releaseDeviceBlocks(bytes); });
      releaserRegistered_ = true;
    }
    deviceBlocks_.emplace_back(block, blockBytes);
    deviceAllocatedBytes_ += blockBytes;
    deviceCursor_ = static_cast<char*>(block);
    deviceRemaining_ = blockBytes;
  }
  void* ptr = deviceCursor_;
  deviceCursor_ += bytes;
  deviceRemaining_ -= bytes;
  return ptr;
}

bool FusedForScanCache::releaseDeviceBlocks(std::size_t bytes) {
  std::unique_lock lock(mutex_);
  std::size_t freed = 0;
  while (freed < bytes && !deviceBlocks_.empty()) {
    const auto [block, blockBytes] = deviceBlocks_.back();
    const auto* begin = static_cast<const char*>(block);
    const auto dropIfInBlock = [&](Payload& payload) {
      const auto* wire = static_cast<const char*>(payload.deviceWire);
      if (wire >= begin && wire < begin + blockBytes) {
        payload.deviceWire = nullptr;
      }
    };
    for (auto& [name, entry] : entries_) {
      for (auto& slice : entry.values) {
        dropIfInBlock(slice);
      }
      if (entry.keys.has_value()) {
        dropIfInBlock(*entry.keys);
      }
    }
    // Copies from the block were enqueued under the lock: wait for them.
    VELOX_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    VELOX_CHECK(cudaFree(block) == cudaSuccess);
    deviceBlocks_.pop_back();
    deviceAllocatedBytes_ -= blockBytes;
    freed += blockBytes;
  }
  if (freed > 0) {
    LOG(WARNING) << "Fused FOR scan cache: released " << freed
                 << " bytes of device blocks for a failed " << bytes
                 << "-byte allocation; device tier now "
                 << deviceAllocatedBytes_ << " bytes";
    // The newest block held the cursor: start the next one fresh, and keep
    // the tier at its new size under memory pressure.
    deviceCursor_ = nullptr;
    deviceRemaining_ = 0;
    deviceCapacity_.store(deviceAllocatedBytes_, std::memory_order_relaxed);
    deviceTierShrunk_.store(true, std::memory_order_relaxed);
  }
  return freed > 0;
}

void* FusedForScanCache::allocatePinned(std::size_t bytes) {
  bytes = alignUp(bytes);
  if (bytes > kBlockBytes / 4) {
    blocks_.push_back(allocatePinnedBlock(bytes));
    return blocks_.back();
  }
  if (bytes > blockRemaining_) {
    blocks_.push_back(allocatePinnedBlock(kBlockBytes));
    blockCursor_ = static_cast<char*>(blocks_.back());
    blockRemaining_ = kBlockBytes;
  }
  void* ptr = blockCursor_;
  blockCursor_ += bytes;
  blockRemaining_ -= bytes;
  return ptr;
}

} // namespace facebook::velox::cudf_velox::connector::hive
