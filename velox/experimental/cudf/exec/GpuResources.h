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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Statistics adaptors wrapping the main (and, if distinct, output) memory
/// resources so that live-allocated GPU bytes can be reported.
/// statistics_resource_adaptor is a cuda::mr::shared_resource: it is copyable
/// and copies share the same counter state, so the copies placed in
/// mr_/output_mr_ increment the same counters that these read.
extern std::optional<rmm::mr::statistics_resource_adaptor> statsMr_;
extern std::optional<rmm::mr::statistics_resource_adaptor> outputStatsMr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Live bytes currently allocated through the cuDF/RMM memory resource(s), or
/// -1 if cuDF is not registered. Counts the bytes callers hold rather than the
/// device memory a pool resource retains for reuse, so the value drops as
/// queries free their allocations.
int64_t cudfAllocatedBytes();

/// Counters reported by TieredMemoryResource.
struct TieredMemoryResourceStats {
  /// Bytes currently outstanding in the fast device tier.
  std::size_t deviceBytes;
  /// High-water mark of `deviceBytes`.
  std::size_t deviceBytesPeak;
  /// Bytes currently outstanding in the managed (overflow) tier.
  std::size_t overflowBytes;
  /// High-water mark of `overflowBytes`.
  std::size_t overflowBytesPeak;
  /// Number of allocations that were served by the overflow tier.
  std::size_t overflowAllocations;
  /// Size of the device tier cap, in bytes.
  std::size_t capBytes;
};

/**
 * @brief Two-tier memory resource: a device pool with a hard cap, backed by
 * managed memory for anything that does not fit.
 *
 * Allocations are served from a `cuda_async` device pool as long as the total
 * outstanding device-tier size stays at or below `deviceCapBytes`. Once the cap
 * is reached (or the pool itself reports `rmm::out_of_memory`) the allocation
 * is served from plain CUDA managed (unified) memory instead. This keeps
 * queries that fit in device memory on exactly today's fast path, while letting
 * queries that do not fit complete via oversubscription rather than failing
 * with `cudaErrorMemoryAllocation`. Pooling or prefetching the overflow tier is
 * left out on purpose: it is a performance optimization for the spilling path,
 * and the plain managed resource is the most robust oversubscription backing.
 *
 * For the "tiered" mode of createMemoryResource(), `cudf.memory_percent` is the
 * *device-tier cap* expressed as a percentage of total device memory (not of
 * free memory as for the pooling modes). 85-92 is the intended range: high
 * enough that fitting queries never touch managed memory, low enough to leave
 * headroom for cuDF/CUDA scratch allocations outside this resource.
 *
 * This class is copyable and cheap to copy: all state lives in a shared `Impl`,
 * so copies (including the one stored inside
 * `cuda::mr::any_resource<cuda::mr::device_accessible>`) share counters and the
 * overflow bookkeeping.
 *
 * Satisfies the CCCL `cuda::mr::resource_with<..., device_accessible>` concept.
 */
class TieredMemoryResource {
 public:
  /// @param deviceCapBytes Maximum bytes served from the device tier.
  explicit TieredMemoryResource(std::size_t deviceCapBytes);

  /// Stream-ordered allocation. Tries the device tier, falls back to managed.
  [[nodiscard]] void* allocate(
      cuda::stream_ref stream,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /// Stream-ordered deallocation. Routes to whichever tier owns @p ptr.
  void deallocate(
      cuda::stream_ref stream,
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /// Synchronous allocation on the default stream.
  [[nodiscard]] void* allocate_sync(
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /// Synchronous deallocation on the default stream.
  void deallocate_sync(
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  /// Two instances are equal iff they share the same underlying state.
  [[nodiscard]] bool operator==(
      TieredMemoryResource const& other) const noexcept {
    return impl_ == other.impl_;
  }

  /// @copydoc operator==
  [[nodiscard]] bool operator!=(
      TieredMemoryResource const& other) const noexcept {
    return impl_ != other.impl_;
  }

  /// Enables the `cuda::mr::device_accessible` property.
  friend void get_property(
      TieredMemoryResource const&,
      cuda::mr::device_accessible) noexcept {}

  /// Returns a snapshot of this resource's counters.
  [[nodiscard]] TieredMemoryResourceStats stats() const;

  /// Registers this instance as the process-wide tiered resource reported by
  /// tieredMemoryResourceStats(). Called by createMemoryResource().
  void registerGlobal() const;

  class Impl;

 private:
  std::shared_ptr<Impl> impl_;
};

/// Returns the counters of the process-wide tiered memory resource, or
/// std::nullopt when the "tiered" mode is not active.
[[nodiscard]] std::optional<TieredMemoryResourceStats>
tieredMemoryResourceStats();

/// LOG(INFO)s the counters of the process-wide tiered memory resource, if any.
void logTieredMemoryResourceStats();

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/// Returns the global CUDA stream pool used by cudf.
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

/// Records the CUDA device owning the primary context, so worker threads can
/// bind it via `ensureCudaContextForThread()`.
///
/// @param device The CUDA device ordinal that was current during registration.
void setCudfContextDevice(int device);

/// Binds the primary CUDA context on the calling thread for the registered
/// device, rebinding when cuDF is registered on another device.
/// Needed as `cuda::stream_ref` uses the driver API, which does not lazily
/// create a context like the runtime API does.
void ensureCudaContextForThread();

/// Ensures that the calling thread has a CUDA context and returns the
/// `cudf::get_default_stream()`. Use it for host-thread stream work outside
/// cuDF operators.
[[nodiscard]] cuda::stream_ref getDefaultStreamForCurrentThread();

} // namespace facebook::velox::cudf_velox
