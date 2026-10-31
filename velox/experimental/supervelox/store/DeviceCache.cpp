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
#include "velox/experimental/supervelox/store/DeviceCache.h"

#include "velox/experimental/supervelox/common/Cuda.h"

namespace facebook::velox::supervelox {

namespace {

constexpr size_t kAlignment = 256;

size_t aligned(size_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

} // namespace


DeviceCache::DeviceCache(size_t budgetBytes, size_t stageBytes)
    : budgetBytes_(budgetBytes), stageBytes_(stageBytes) {
  SV_CUDA(cudaStreamCreateWithFlags(&copyStream_, cudaStreamNonBlocking));
  SV_CUDA(cudaMalloc(&arena_, budgetBytes_));
  free_[0] = budgetBytes_;
}

DeviceCache::~DeviceCache() {
  clear();
  cudaFree(arena_);
  cudaStreamDestroy(copyStream_);
  for (auto& [key, entry] : entries_) {
    if (entry.pinned) {
      cudaFreeHost(entry.pinned);
    }
  }
}

void DeviceCache::beginQuery() {
  ++query_;
}

void DeviceCache::finishCopy(Entry& entry) {
  if (entry.copied) {
    SV_CUDA(cudaEventSynchronize(entry.copied));
    SV_CUDA(cudaEventDestroy(entry.copied));
    entry.copied = nullptr;
    entry.host = nullptr;
  }
}

void DeviceCache::clear() {
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->second.device) {
      finishCopy(it->second);
      release(it->second.device, it->second.bytes);
      residentBytes_ -= aligned(it->second.bytes);
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void* DeviceCache::carve(size_t bytes) {
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second >= bytes) {
      const size_t offset = it->first;
      const size_t remaining = it->second - bytes;
      free_.erase(it);
      if (remaining) {
        free_[offset + bytes] = remaining;
      }
      return arena_ + offset;
    }
  }
  return nullptr;
}

void DeviceCache::release(void* device, size_t bytes) {
  bytes = aligned(bytes);
  size_t offset = static_cast<char*>(device) - arena_;
  auto next = free_.lower_bound(offset);
  if (next != free_.end() && offset + bytes == next->first) {
    bytes += next->second;
    next = free_.erase(next);
  }
  if (next != free_.begin()) {
    auto previous = std::prev(next);
    if (previous->first + previous->second == offset) {
      offset = previous->first;
      bytes += previous->second;
      free_.erase(previous);
    }
  }
  free_[offset] = bytes;
}

void* DeviceCache::allocate(size_t bytes) {
  bytes = aligned(bytes);
  while (true) {
    if (void* device = carve(bytes)) {
      residentBytes_ += bytes;
      return device;
    }
    auto victim = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->second.device && it->second.lastUse < query_ &&
          (victim == entries_.end() || it->second.lastUse < victim->second.lastUse)) {
        victim = it;
      }
    }
    if (victim == entries_.end()) {
      return nullptr;
    }
    finishCopy(victim->second);
    release(victim->second.device, victim->second.bytes);
    residentBytes_ -= aligned(victim->second.bytes);
    entries_.erase(victim);
  }
}

const void* DeviceCache::get(
    const std::string& key,
    const void* host,
    size_t bytes,
    cudaStream_t stream) {
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    Entry& entry = it->second;
    entry.lastUse = query_;
    if (entry.copied) {
      if (cudaEventQuery(entry.copied) == cudaErrorNotReady) {
        stats_.hostBytes += bytes;
        return entry.host;
      }
      finishCopy(entry);
    }
    return entry.device;
  }
  void* device = allocate(bytes);
  if (device == nullptr) {
    stats_.hostBytes += bytes;
    return host;
  }
  Entry entry{device, nullptr, bytes, query_};
  if (bytes < stageBytes_) {
    SV_CUDA(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream));
    // Callers on other streams may read the buffer as soon as it is cached.
    SV_CUDA(cudaStreamSynchronize(stream));
    stats_.copiedBytes += bytes;
    entries_[key] = entry;
    return device;
  }
  SV_CUDA(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, copyStream_));
  SV_CUDA(cudaEventCreateWithFlags(&entry.copied, cudaEventDisableTiming));
  SV_CUDA(cudaEventRecord(entry.copied, copyStream_));
  entry.host = host;
  stats_.stagedBytes += bytes;
  stats_.hostBytes += bytes;
  entries_[key] = entry;
  return host;
}

void* DeviceCache::derived(
    const std::string& key,
    size_t bytes,
    const std::function<void(void*)>& build,
    cudaStream_t stream) {
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    it->second.lastUse = query_;
    return it->second.device ? it->second.device : it->second.pinned;
  }
  Entry entry{allocate(bytes), nullptr, bytes, query_};
  if (entry.device == nullptr) {
    SV_CUDA(cudaHostAlloc(&entry.pinned, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  }
  void* buffer = entry.device ? entry.device : entry.pinned;
  build(buffer);
  SV_CUDA(cudaStreamSynchronize(stream));
  stats_.builtBytes += bytes;
  entries_[key] = entry;
  return buffer;
}

} // namespace facebook::velox::supervelox
