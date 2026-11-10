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
#include "velox/experimental/ucx-exchange/ExchangeStore.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <rmm/aligned.hpp>

namespace facebook::velox::ucx_exchange {

ExchangeStore::HostBuffer::HostBuffer(std::size_t bytes)
    : data_(cudf::get_pinned_memory_resource().allocate_sync(
          bytes, rmm::CUDA_ALLOCATION_ALIGNMENT)),
      bytes_(bytes) {}

ExchangeStore::HostBuffer::~HostBuffer() {
  cudf::get_pinned_memory_resource().deallocate_sync(
      data_, bytes_, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

ExchangeStore& ExchangeStore::instance() {
  static ExchangeStore store;
  return store;
}

void ExchangeStore::setDeviceCapacity(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  deviceCapacity_ = bytes;
}

void ExchangeStore::setHostCapacity(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  hostCapacity_ = bytes;
}

void ExchangeStore::add(
    std::string_view table,
    int32_t bucket,
    PackedExchangePayload payload,
    vector_size_t numRows,
    cuda::stream_ref stream) {
  VELOX_CHECK_NOT_NULL(payload.packed);
  StoredPayload stored;
  stored.metadata = std::move(payload.packed->metadata);
  stored.device = std::move(payload.packed->gpu_data);
  stored.numRows = numRows;
  VELOX_CHECK_NOT_NULL(stored.metadata);
  stored.bytes = stored.device ? stored.device->size() : 0;

  // A payload without data, such as one with no columns, takes no memory and
  // stays in the device tier.
  bool toHost;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    toHost = stored.bytes > 0 && deviceBytes_ + stored.bytes > deviceCapacity_;
    if (toHost) {
      VELOX_CHECK_LE(
          stored.bytes,
          hostCapacity_ - hostBytes_,
          "The exchange store's host tier is full: {} bytes held, {} bytes capacity",
          hostBytes_,
          hostCapacity_);
      hostBytes_ += stored.bytes;
    } else {
      deviceBytes_ += stored.bytes;
    }
  }

  if (toHost) {
    stored.host = std::make_unique<HostBuffer>(stored.bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        stored.host->data(),
        stored.device->data(),
        stored.bytes,
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    stored.device.reset();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  tables_[std::string{table}][bucket].push_back(std::move(stored));
  ++payloads_;
}

std::vector<std::unique_ptr<PackedTableWithStream>> ExchangeStore::take(
    std::string_view table,
    int32_t bucket,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<StoredPayload> payloads;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto buckets = tables_.find(table);
    if (buckets == tables_.end()) {
      return {};
    }
    auto it = buckets->second.find(bucket);
    if (it == buckets->second.end()) {
      return {};
    }
    payloads = std::move(it->second);
    buckets->second.erase(it);
    if (buckets->second.empty()) {
      tables_.erase(buckets);
    }
    forget(payloads);
  }

  // Copies the host tier's payloads up, then frees their host memory once the
  // copies are done.
  bool copied = false;
  for (auto& payload : payloads) {
    if (payload.host == nullptr) {
      continue;
    }
    payload.device =
        std::make_unique<rmm::device_buffer>(payload.bytes, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        payload.device->data(),
        payload.host->data(),
        payload.bytes,
        cudaMemcpyHostToDevice,
        stream.get()));
    copied = true;
  }
  if (copied) {
    stream.sync();
  }

  std::vector<std::unique_ptr<PackedTableWithStream>> result;
  result.reserve(payloads.size());
  for (auto& payload : payloads) {
    payload.host.reset();
    if (payload.device == nullptr) {
      payload.device = std::make_unique<rmm::device_buffer>(0, stream, mr);
    }
    // A device-tier payload was allocated on the producer's stream. Free it on
    // 'stream' so the release is ordered after the decode reads it.
    payload.device->set_stream(stream);
    result.push_back(restoreExchangePayload(
        std::move(payload.metadata),
        std::move(payload.device),
        stream,
        payload.numRows));
  }
  return result;
}

void ExchangeStore::drop(std::string_view table) {
  Buckets dropped;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tables_.find(table);
    if (it == tables_.end()) {
      return;
    }
    dropped = std::move(it->second);
    tables_.erase(it);
    for (const auto& [_, payloads] : dropped) {
      forget(payloads);
    }
  }
  // 'dropped' frees both tiers' memory outside the lock.
}

ExchangeStore::Stats ExchangeStore::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return Stats{
      .deviceBytes = deviceBytes_,
      .hostBytes = hostBytes_,
      .payloads = payloads_};
}

void ExchangeStore::forget(const std::vector<StoredPayload>& payloads) {
  for (const auto& payload : payloads) {
    if (payload.host != nullptr) {
      hostBytes_ -= payload.bytes;
    } else {
      deviceBytes_ -= payload.bytes;
    }
  }
  payloads_ -= payloads.size();
}

} // namespace facebook::velox::ucx_exchange
