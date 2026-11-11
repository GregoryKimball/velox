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
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <glog/logging.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Pinned memory is allocated in blocks of this size; larger entries get their
// own allocation.
constexpr std::size_t kBlockBytes = std::size_t{1} << 30;
constexpr std::size_t kAlignment = 256;

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
  auto current = deviceCapacity_.load(std::memory_order_relaxed);
  while (current < bytes &&
         !deviceCapacity_.compare_exchange_weak(
             current, bytes, std::memory_order_relaxed)) {
  }
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

  auto encodedColumn = encodeForFusedFor(column, {}, stream, tempMr);
  if (!encodedColumn.packedValues.has_value()) {
    // A string column with many distinct values: fused FOR would not compress
    // it.
    std::unique_lock lock(mutex_);
    uncacheable_.insert(std::move(entryKey));
    return false;
  }
  auto& values = *encodedColumn.packedValues;
  auto& keys = encodedColumn.packedKeys;
  const bool deltas = encodedColumn.encoding == FusedForEncoding::kDeltas;
  const auto payloadBytes =
      [](const cudf::detail::fused_for_packed_columns& packed) {
        return packed.wire_data->size() + packed.metadata->size();
      };
  const auto bytes =
      payloadBytes(values) + (keys.has_value() ? payloadBytes(*keys) : 0);

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
  entry.values = store(std::move(values), stream);
  if (keys.has_value()) {
    entry.keys = store(std::move(*keys), stream);
  }
  entry.deltas = deltas;
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

  auto [values, valuesView] = decodePayload(entry->values, stream, tempMr);
  if (entry->deltas) {
    auto column = decodeFusedForEncoding(
        FusedForEncoding::kDeltas,
        valuesView,
        std::nullopt,
        0,
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
    if (deviceAllocatedBytes_ + blockBytes > capacity) {
      return nullptr;
    }
    void* block = nullptr;
    if (cudaMalloc(&block, blockBytes) != cudaSuccess) {
      // Out of device memory: keep the rest of the cache in host memory.
      (void)cudaGetLastError();
      deviceCapacity_.store(deviceAllocatedBytes_, std::memory_order_relaxed);
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
      dropIfInBlock(entry.values);
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
