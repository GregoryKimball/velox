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

#include "velox/common/base/Exceptions.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/fused_for.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

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
  }
  cudf::detail::fused_for_options options;
  options.layout = cudf::detail::fused_for_layout::bitpacked;
  auto packed = cudf::detail::pack_fused_for(
      cudf::table_view{{column}}, stream, tempMr, options);
  const auto wireBytes = packed.wire_data->size();
  const auto bytes = wireBytes + packed.metadata->size();

  std::unique_lock lock(mutex_);
  if (entries_.contains(entryKey)) {
    return true;
  }
  if (usedBytes_.load(std::memory_order_relaxed) + bytes >
      capacity_.load(std::memory_order_relaxed)) {
    return false;
  }
  void* host = allocatePinned(wireBytes);
  const auto status = cudaMemcpyAsync(
      host,
      packed.wire_data->data(),
      wireBytes,
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));
  stream.sync();
  entries_.emplace(
      std::move(entryKey),
      Entry{
          std::move(*packed.metadata),
          host,
          wireBytes,
          packed.segment_count,
          packed.logical_data_size});
  usedBytes_.fetch_add(bytes, std::memory_order_relaxed);
  return true;
}

std::unique_ptr<cudf::column> FusedForScanCache::read(
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

  rmm::device_buffer wire(entry->wireBytes, stream, tempMr);
  const auto status = cudaMemcpyAsync(
      wire.data(),
      entry->wire,
      entry->wireBytes,
      cudaMemcpyHostToDevice,
      stream.get());
  VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));

  cudf::detail::fused_for_packed_columns payload;
  payload.metadata = std::make_unique<std::vector<uint8_t>>(entry->metadata);
  payload.wire_data = std::make_unique<rmm::device_buffer>(std::move(wire));
  payload.segment_count = entry->segmentCount;
  payload.logical_data_size = entry->logicalDataSize;
  auto packedColumns =
      cudf::detail::decode_fused_for(std::move(payload), stream, tempMr);
  return std::make_unique<cudf::column>(
      cudf::unpack(packedColumns).column(0), stream, mr);
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
