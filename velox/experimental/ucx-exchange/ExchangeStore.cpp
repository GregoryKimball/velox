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

#include <glog/logging.h>

#include <chrono>
#include <iterator>

#include <cudf/utilities/error.hpp>
#include <rmm/aligned.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace facebook::velox::ucx_exchange {

namespace {
// The host pool's first block. The pool grows by at least its current size.
constexpr std::size_t kInitialHostPoolBytes{std::size_t{1} << 30};

// Host buffers are allocated and freed on one stream, so the pool reuses them
// without waiting; copies into and out of them are synchronized separately.
cuda::stream_ref hostPoolStream() {
  return cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
}
} // namespace

struct ExchangeStore::HostPool {
  HostPool() : pool(upstream, kInitialHostPoolBytes) {}

  rmm::mr::pinned_host_memory_resource upstream;
  rmm::mr::pool_memory_resource pool;
};

ExchangeStore::HostBuffer::HostBuffer(HostPool& pool, std::size_t bytes)
    : pool_(pool),
      data_(pool.pool.allocate(
          hostPoolStream(), bytes, rmm::CUDA_ALLOCATION_ALIGNMENT)),
      bytes_(bytes) {}

ExchangeStore::HostBuffer::~HostBuffer() {
  if (written_ != nullptr) {
    // A destructor must not throw: report a failed wait and release anyway.
    const auto status = cudaEventSynchronize(written_);
    LOG_IF(ERROR, status != cudaSuccess)
        << "Waiting for a copy into the exchange store's host tier failed: "
        << cudaGetErrorString(status);
    cudaEventDestroy(written_);
  }
  pool_.pool.deallocate(
      hostPoolStream(), data_, bytes_, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void ExchangeStore::HostBuffer::recordWrite(cuda::stream_ref stream) {
  if (written_ == nullptr) {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&written_, cudaEventDisableTiming));
  }
  CUDF_CUDA_TRY(cudaEventRecord(written_, stream.get()));
}

void ExchangeStore::HostBuffer::orderReadsOn(cuda::stream_ref stream) const {
  if (written_ != nullptr) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), written_, 0));
  }
}

ExchangeStore::ExchangeStore() = default;

ExchangeStore::~ExchangeStore() = default;

ExchangeStore::HostPool& ExchangeStore::hostPool() {
  if (hostPool_ == nullptr) {
    hostPool_ = std::make_unique<HostPool>();
  }
  return *hostPool_;
}

ExchangeStore& ExchangeStore::instance() {
  static ExchangeStore store;
  return store;
}

void ExchangeStore::setDeviceCapacity(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  deviceCapacity_ = bytes;
  configuredDeviceCapacity_ = bytes;
}

void ExchangeStore::restoreDeviceCapacityIfEmpty() {
  if (tables_.empty()) {
    deviceCapacity_ = configuredDeviceCapacity_;
  }
}

void ExchangeStore::setHostCapacity(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  hostCapacity_ = bytes;
}

namespace {

// Wraps the packed data of 'payload' for storing.
template <typename StoredPayload>
StoredPayload toStored(PackedExchangePayload payload, vector_size_t numRows) {
  VELOX_CHECK_NOT_NULL(payload.packed);
  StoredPayload stored;
  stored.metadata = std::move(payload.packed->metadata);
  stored.device = std::move(payload.packed->gpu_data);
  stored.numRows = numRows;
  VELOX_CHECK_NOT_NULL(stored.metadata);
  stored.bytes = stored.device ? stored.device->size() : 0;
  return stored;
}

} // namespace

bool ExchangeStore::place(StoredPayload& stored, cuda::stream_ref stream) {
  // A payload without data, such as one with no columns, takes no memory and
  // stays in the device tier.
  bool toHost;
  HostPool* pool = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    toHost = stored.bytes > 0 && deviceBytes_ + stored.bytes > deviceCapacity_;
    if (toHost) {
      pool = &hostPool();
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
    stored.host = std::make_unique<HostBuffer>(*pool, stored.bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        stored.host->data(),
        stored.device->data(),
        stored.bytes,
        cudaMemcpyDeviceToHost,
        stream.get()));
    // The copy runs behind the writer: readers and the host buffer's release
    // wait for it, and the device buffer is freed after it in stream order.
    stored.host->recordWrite(stream);
    stored.device->set_stream(stream);
    stored.device.reset();
  }
  return toHost;
}

bool ExchangeStore::add(
    std::string_view table,
    std::string_view file,
    PackedExchangePayload payload,
    vector_size_t numRows,
    PayloadEncoding encoding,
    cuda::stream_ref stream) {
  auto stored = toStored<StoredPayload>(std::move(payload), numRows);
  stored.encoding = std::move(encoding);
  const bool toHost = place(stored, stream);
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = tables_[std::string{table}];
    entry.files[std::string{file}].push_back(std::move(stored));
    ++payloads_;
    waiters = std::move(entry.waiters);
    entry.waiters.clear();
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
  return toHost;
}

uint64_t ExchangeStore::addKeys(
    std::string_view table,
    PackedExchangePayload keys,
    vector_size_t numKeys,
    uint32_t numReferences,
    cuda::stream_ref stream) {
  VELOX_CHECK_GT(numReferences, 0, "Keys must be shared by a payload");
  auto stored = toStored<StoredPayload>(std::move(keys), numKeys);
  place(stored, stream);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto id = nextKeysId_++;
  tables_[std::string{table}].keys.emplace(
      id,
      StoredKeys{
          std::make_shared<const StoredPayload>(std::move(stored)),
          numReferences});
  ++payloads_;
  return id;
}

std::vector<ExchangeStore::TakenPayload> ExchangeStore::take(
    std::string_view table,
    std::string_view file,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    TakeStats* stats) {
  std::vector<StoredPayload> payloads;
  // The keys each payload refers to, in its column order.
  std::vector<std::vector<std::shared_ptr<const StoredPayload>>> payloadKeys;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto tableIt = tables_.find(table);
    if (tableIt == tables_.end()) {
      return {};
    }
    auto& files = tableIt->second.files;
    auto it = files.find(file);
    if (it == files.end()) {
      return {};
    }
    payloads = std::move(it->second);
    files.erase(it);
    forget(payloads);
    auto& tableKeys = tableIt->second.keys;
    for (const auto& payload : payloads) {
      auto& keys = payloadKeys.emplace_back();
      for (const auto id : payload.encoding.keys) {
        auto keysIt = tableKeys.find(id);
        VELOX_CHECK(
            keysIt != tableKeys.end(),
            "Keys of a stored payload are missing: {} {}",
            table,
            id);
        keys.push_back(keysIt->second.payload);
        if (--keysIt->second.numReferences == 0) {
          forget(*keysIt->second.payload);
          tableKeys.erase(keysIt);
        }
      }
    }
    if (files.empty() && tableIt->second.writers.empty() &&
        tableIt->second.waiters.empty()) {
      forget(tableIt->second);
      tables_.erase(tableIt);
      restoreDeviceCapacityIfEmpty();
    }
  }

  // Copies the host tier's payloads up, and every payload's keys, then frees
  // the host memory once the copies are done.
  const auto copyStart = std::chrono::steady_clock::now();
  bool copied = false;
  const auto copyToDevice = [&](const StoredPayload& payload) {
    if (stats != nullptr) {
      (payload.host != nullptr ? stats->hostBytes : stats->deviceBytes) +=
          payload.bytes;
    }
    auto device = std::make_unique<rmm::device_buffer>(payload.bytes, stream, mr);
    if (payload.host != nullptr) {
      payload.host->orderReadsOn(stream);
    }
    if (payload.bytes > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          device->data(),
          payload.host != nullptr ? payload.host->data()
                                  : payload.device->data(),
          payload.bytes,
          payload.host != nullptr ? cudaMemcpyHostToDevice
                                  : cudaMemcpyDeviceToDevice,
          stream.get()));
      copied = true;
    }
    return device;
  };
  std::vector<std::vector<std::unique_ptr<rmm::device_buffer>>> keyBuffers;
  for (size_t i = 0; i < payloads.size(); ++i) {
    auto& payload = payloads[i];
    if (payload.host != nullptr) {
      payload.device = copyToDevice(payload);
    } else if (stats != nullptr) {
      stats->deviceBytes += payload.bytes;
    }
    auto& buffers = keyBuffers.emplace_back();
    for (const auto& keys : payloadKeys[i]) {
      buffers.push_back(copyToDevice(*keys));
    }
  }
  if (copied) {
    // Another take() may free the keys once this one lets go of them.
    stream.sync();
    if (stats != nullptr) {
      stats->copyNanos += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - copyStart)
                              .count();
    }
  }

  std::vector<TakenPayload> result;
  result.reserve(payloads.size());
  for (size_t i = 0; i < payloads.size(); ++i) {
    auto& payload = payloads[i];
    payload.host.reset();
    if (payload.device == nullptr) {
      payload.device = std::make_unique<rmm::device_buffer>(0, stream, mr);
    }
    // A device-tier payload was allocated on the producer's stream. Free it on
    // 'stream' so the release is ordered after the decode reads it.
    payload.device->set_stream(stream);
    TakenPayload taken;
    taken.data = restoreExchangePayload(
        std::move(payload.metadata),
        std::move(payload.device),
        stream,
        payload.numRows);
    taken.encoding = std::move(payload.encoding);
    for (size_t k = 0; k < payloadKeys[i].size(); ++k) {
      const auto& keys = *payloadKeys[i][k];
      taken.keys.push_back(restoreExchangePayload(
          std::make_unique<std::vector<uint8_t>>(*keys.metadata),
          std::move(keyBuffers[i][k]),
          stream,
          keys.numRows));
    }
    result.push_back(std::move(taken));
  }
  return result;
}

void ExchangeStore::beginWrite(
    std::string_view table,
    std::string_view taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  tables_[std::string{table}].writers.insert(std::string{taskId});
}

void ExchangeStore::endWrites(std::string_view taskId) {
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = tables_.begin(); it != tables_.end();) {
      auto& table = it->second;
      if (table.writers.erase(std::string{taskId}) == 0 ||
          !table.writers.empty()) {
        ++it;
        continue;
      }
      complete(it->first);
      for (auto& waiter : table.waiters) {
        waiters.push_back(std::move(waiter));
      }
      table.waiters.clear();
      if (table.files.empty()) {
        forget(table);
        it = tables_.erase(it);
        continue;
      }
      ++it;
    }
    restoreDeviceCapacityIfEmpty();
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

void ExchangeStore::complete(const std::string& name) {
  // Bounds the names kept for reads of files that were never written.
  constexpr size_t kMaxCompletedTables = 4096;
  if (completedTables_.insert(name).second) {
    completedOrder_.push_back(name);
    if (completedOrder_.size() > kMaxCompletedTables) {
      completedTables_.erase(completedOrder_.front());
      completedOrder_.pop_front();
    }
  }
}

std::optional<std::vector<ExchangeStore::TakenPayload>>
ExchangeStore::takeOrWait(
    std::string_view table,
    const std::vector<std::string>& files,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    ContinueFuture& future,
    TakeStats* stats) {
  for (;;) {
    std::vector<std::string_view> written;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto tableIt = tables_.find(table);
      if (tableIt != tables_.end()) {
        for (const auto& file : files) {
          if (tableIt->second.files.contains(file)) {
            written.push_back(file);
          }
        }
      }
      if (written.empty()) {
        const bool complete = completedTables_.count(std::string{table}) > 0 &&
            (tableIt == tables_.end() || tableIt->second.writers.empty());
        if (complete) {
          return std::vector<TakenPayload>{};
        }
        // A reader may come before the table's first writer: the entry it
        // creates holds its promise until a payload or the completion.
        auto& entry = tableIt != tables_.end() ? tableIt->second
                                               : tables_[std::string{table}];
        auto [promise, semiFuture] =
            makeVeloxContinuePromiseContract("ExchangeStore::takeOrWait");
        entry.waiters.push_back(std::move(promise));
        future = std::move(semiFuture);
        return std::nullopt;
      }
    }
    std::vector<TakenPayload> result;
    for (const auto file : written) {
      auto taken = take(table, file, stream, mr, stats);
      std::move(taken.begin(), taken.end(), std::back_inserter(result));
    }
    // Another reader of the same files may have taken them in between.
    if (!result.empty()) {
      return result;
    }
  }
}

void ExchangeStore::drop(std::string_view table) {
  Table dropped;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tables_.find(table);
    if (it == tables_.end()) {
      return;
    }
    dropped = std::move(it->second);
    tables_.erase(it);
    forget(dropped);
    restoreDeviceCapacityIfEmpty();
  }
  // 'dropped' frees both tiers' memory outside the lock. Its readers wake up:
  // a broken promise would fail their tasks with an error that hides the one
  // that dropped the table.
  for (auto& waiter : dropped.waiters) {
    waiter.setValue();
  }
}

void ExchangeStore::dropQuery(std::string_view queryId) {
  std::vector<Table> dropped;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = tables_.begin(); it != tables_.end();) {
      if (it->first.find(queryId) == std::string::npos) {
        ++it;
        continue;
      }
      forget(it->second);
      dropped.push_back(std::move(it->second));
      it = tables_.erase(it);
    }
    restoreDeviceCapacityIfEmpty();
  }
  // 'dropped' frees both tiers' memory outside the lock. Its readers wake up,
  // as in drop().
  for (auto& table : dropped) {
    for (auto& waiter : table.waiters) {
      waiter.setValue();
    }
  }
}

std::size_t ExchangeStore::demote(std::size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t moved = 0;
  for (auto& [_, table] : tables_) {
    for (auto& [__, payloads] : table.files) {
      for (auto& payload : payloads) {
        if (moved >= bytes) {
          break;
        }
        if (payload.host != nullptr || payload.bytes == 0 ||
            payload.bytes > hostCapacity_ - hostBytes_) {
          continue;
        }
        // The payload's device data is complete (see add()).
        payload.host = std::make_unique<HostBuffer>(hostPool(), payload.bytes);
        CUDF_CUDA_TRY(cudaMemcpy(
            payload.host->data(),
            payload.device->data(),
            payload.bytes,
            cudaMemcpyDeviceToHost));
        payload.device.reset();
        deviceBytes_ -= payload.bytes;
        hostBytes_ += payload.bytes;
        moved += payload.bytes;
      }
    }
  }
  if (moved > 0) {
    // Stay at the smaller device tier, so that the next payloads do not take
    // the memory back.
    deviceCapacity_ = deviceBytes_;
    // The frees are ordered on the payloads' streams. Wait for them, so that
    // an allocation on any stream can reuse the memory.
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
    LOG(WARNING) << "Exchange store: moved " << moved
                 << " bytes to host memory for a failed " << bytes
                 << "-byte device allocation; device tier now " << deviceBytes_
                 << " bytes";
  }
  return moved;
}

ExchangeStore::Stats ExchangeStore::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return Stats{
      .deviceBytes = deviceBytes_,
      .hostBytes = hostBytes_,
      .payloads = payloads_};
}

void ExchangeStore::forget(const StoredPayload& payload) {
  if (payload.host != nullptr) {
    hostBytes_ -= payload.bytes;
  } else {
    deviceBytes_ -= payload.bytes;
  }
  --payloads_;
}

void ExchangeStore::forget(const std::vector<StoredPayload>& payloads) {
  for (const auto& payload : payloads) {
    forget(payload);
  }
}

void ExchangeStore::forget(const Table& table) {
  for (const auto& [_, payloads] : table.files) {
    forget(payloads);
  }
  for (const auto& [_, keys] : table.keys) {
    forget(*keys.payload);
  }
}

} // namespace facebook::velox::ucx_exchange
