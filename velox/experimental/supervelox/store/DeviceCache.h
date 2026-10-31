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

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>

namespace facebook::velox::supervelox {

/// Keeps named buffers resident in HBM across queries up to a byte budget.
/// Buffers used by the running query are never evicted; when one does not fit,
/// kernels read it from pinned host memory instead. A large buffer missing
/// from HBM is staged: this query reads the pinned host copy while a copy
/// stream fills HBM for the next query.
class DeviceCache {
 public:
  struct Stats {
    size_t copiedBytes{0};
    size_t stagedBytes{0};
    size_t hostBytes{0};
    size_t builtBytes{0};
  };

  /// Buffers of at least 'stageBytes' are staged instead of copied before use.
  DeviceCache(size_t budgetBytes, size_t stageBytes = 2ULL << 30);
  ~DeviceCache();

  /// Starts a query. Buffers fetched after this call stay resident until the
  /// next call.
  void beginQuery();

  /// Returns a device-readable copy of 'bytes' at pinned 'host'.
  const void* get(const std::string& key, const void* host, size_t bytes, cudaStream_t stream);

  /// Returns a cached buffer of derived data, calling 'build' with the buffer
  /// to fill it the first time. The buffer is in HBM if it fits, else in
  /// mapped pinned host memory.
  void* derived(
      const std::string& key,
      size_t bytes,
      const std::function<void(void*)>& build,
      cudaStream_t stream);

  const Stats& stats() const {
    return stats_;
  }

  void resetStats() {
    stats_ = {};
  }

  size_t residentBytes() const {
    return residentBytes_;
  }

  size_t budgetBytes() const {
    return budgetBytes_;
  }

  /// Frees every HBM buffer.
  void clear();

 private:
  struct Entry {
    void* device{nullptr};
    void* pinned{nullptr};
    size_t bytes{0};
    uint64_t lastUse{0};
    /// Set while the copy into 'device' is in flight.
    const void* host{nullptr};
    cudaEvent_t copied{nullptr};
  };

  /// Waits for an in-flight copy into 'entry'.
  static void finishCopy(Entry& entry);

  /// Allocates 'bytes' of HBM from the arena, evicting buffers not used by
  /// the current query. Returns nullptr if that is not enough.
  void* allocate(size_t bytes);

  /// Returns a block to the arena, merging it with free neighbors.
  void release(void* device, size_t bytes);

  /// First fit in the arena, or nullptr.
  void* carve(size_t bytes);

  const size_t budgetBytes_;
  const size_t stageBytes_;
  cudaStream_t copyStream_{nullptr};
  /// The whole budget, reserved once: cudaMalloc maps HBM pages at about
  /// 70 ms per GB, too slow to do while a query runs.
  char* arena_{nullptr};
  /// Free blocks by offset into the arena.
  std::map<size_t, size_t> free_;
  size_t residentBytes_{0};
  uint64_t query_{0};
  std::unordered_map<std::string, Entry> entries_;
  Stats stats_;
};

} // namespace facebook::velox::supervelox
