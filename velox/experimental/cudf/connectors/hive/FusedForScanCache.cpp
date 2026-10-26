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

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
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

// A column can be cached as dictionary indices when it has at most this many
// distinct values and at most one per this many rows.
constexpr cudf::size_type kMaxDictionaryKeys = 1 << 16;
constexpr cudf::size_type kMinRowsPerKey = 8;

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

cudf::detail::fused_for_packed_columns pack(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  cudf::detail::fused_for_options options;
  options.layout = cudf::detail::fused_for_layout::bitpacked;
  return cudf::detail::pack_fused_for(
      cudf::table_view{{column}}, stream, mr, options);
}

std::size_t payloadBytes(const cudf::detail::fused_for_packed_columns& packed) {
  return packed.wire_data->size() + packed.metadata->size();
}

// Returns 'column' dictionary encoded when it has few enough distinct values
// to be cached as dictionary indices, otherwise null.
std::unique_ptr<cudf::column> encodeDictionary(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr) {
  if (column.nullable() || column.type().id() == cudf::type_id::BOOL8) {
    return nullptr;
  }
  auto dictionary = cudf::dictionary::encode(
      column,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      cudf::memory_resources{tempMr, tempMr});
  const auto numKeys =
      cudf::dictionary_column_view(dictionary->view()).keys_size();
  if (numKeys > kMaxDictionaryKeys || numKeys * kMinRowsPerKey > column.size()) {
    return nullptr;
  }
  return dictionary;
}

// Returns the differences between consecutive values of 'column', starting
// with its first value, when it is a non-null BIGINT column in ascending order
// from a non-negative first value (sorted keys such as l_orderkey), otherwise
// null. The order and sign keep every difference within BIGINT, so an
// inclusive prefix sum restores the values exactly.
std::unique_ptr<cudf::column> encodeDeltas(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr) {
  if (column.nullable() || column.is_empty() ||
      column.type().id() != cudf::type_id::INT64 ||
      !cudf::is_sorted(
          cudf::table_view{{column}},
          {cudf::order::ASCENDING},
          {cudf::null_order::BEFORE},
          stream)) {
    return nullptr;
  }
  const auto first = cudf::get_element(column, 0, stream, tempMr);
  if (static_cast<const cudf::numeric_scalar<int64_t>&>(*first).value(stream) <
      0) {
    return nullptr;
  }
  const cudf::numeric_scalar<int64_t> zero(0, true, stream, tempMr);
  const auto previous = cudf::shift(column, 1, zero, stream, tempMr);
  return cudf::binary_operation(
      column,
      previous->view(),
      cudf::binary_operator::SUB,
      column.type(),
      stream,
      tempMr);
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

  std::optional<cudf::detail::fused_for_packed_columns> keys;
  cudf::detail::fused_for_packed_columns values;
  bool deltas = false;
  const auto dictionary = encodeDictionary(column, stream, tempMr);
  if (column.type().id() == cudf::type_id::STRING) {
    if (dictionary == nullptr) {
      std::unique_lock lock(mutex_);
      uncacheable_.insert(std::move(entryKey));
      return false;
    }
    const cudf::dictionary_column_view view(dictionary->view());
    values = pack(view.indices(), stream, tempMr);
    keys = pack(view.keys(), stream, tempMr);
  } else {
    values = pack(column, stream, tempMr);
    if (dictionary != nullptr) {
      // Few distinct values spread over a wide range (e.g. decimal quantities)
      // take fewer bits as dictionary indices. Gathering the keys on read
      // costs a pass over the column, so the dictionary must save at least a
      // quarter of the bytes.
      const cudf::dictionary_column_view view(dictionary->view());
      auto indices = pack(view.indices(), stream, tempMr);
      auto dictionaryKeys = pack(view.keys(), stream, tempMr);
      if (4 * (payloadBytes(indices) + payloadBytes(dictionaryKeys)) <=
          3 * payloadBytes(values)) {
        values = std::move(indices);
        keys = std::move(dictionaryKeys);
      }
    }
    if (!keys.has_value()) {
      // Ascending keys take a few bits per row as differences between
      // consecutive values where the values themselves take tens. Restoring
      // them costs a prefix sum on read, so the differences must also save at
      // least a quarter of the bytes.
      if (const auto differences = encodeDeltas(column, stream, tempMr)) {
        auto packedDifferences = pack(differences->view(), stream, tempMr);
        if (4 * payloadBytes(packedDifferences) <= 3 * payloadBytes(values)) {
          values = std::move(packedDifferences);
          deltas = true;
        }
      }
    }
  }
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
  const auto status = cudaMemcpyAsync(
      host,
      packed.wire_data->data(),
      payload.wireBytes,
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));
  payload.wire = host;
  payload.metadata = std::move(*packed.metadata);
  payload.segmentCount = packed.segment_count;
  payload.logicalDataSize = packed.logical_data_size;
  // The device buffer is released on 'stream' after the copy above.
  packed.wire_data.reset();
  return payload;
}

std::pair<cudf::packed_columns, cudf::column_view>
FusedForScanCache::decodePayload(
    const Payload& payload,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  rmm::device_buffer wire(payload.wireBytes, stream, mr);
  const auto status = cudaMemcpyAsync(
      wire.data(),
      payload.wire,
      payload.wireBytes,
      cudaMemcpyHostToDevice,
      stream.get());
  VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));

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
    auto column = cudf::scan(
        valuesView,
        *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
        cudf::scan_type::INCLUSIVE,
        cudf::null_policy::EXCLUDE,
        stream,
        mr);
    const auto view = column->view();
    return DecodedColumn{{}, std::move(column), view};
  }
  if (!entry->keys.has_value()) {
    return DecodedColumn{std::move(values), nullptr, valuesView};
  }
  const auto [keys, keysView] = decodePayload(*entry->keys, stream, tempMr);
  auto gathered = cudf::gather(
      cudf::table_view{{keysView}},
      valuesView,
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      cudf::memory_resources{mr, tempMr});
  auto column = std::move(gathered->release().front());
  const auto view = column->view();
  return DecodedColumn{{}, std::move(column), view};
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
