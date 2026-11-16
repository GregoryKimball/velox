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
 */#pragma once

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

struct CudfSpillContext;

/// The device-to-host copies of the table slices spilled while one input is
/// processed. SpilledCudfVector::trySpillFlatSlice() queues such copies on the
/// slice's stream without waiting for them, and complete() waits for all of
/// them, once per stream. Until then this keeps the device tables the slices
/// belong to alive. A spilled vector with copies here completes them before
/// its host data is read or freed.
class DeferredSliceCopies {
 public:
  DeferredSliceCopies() = default;

  /// Completes the copies. A CUDA error cannot be reported from here.
  ~DeferredSliceCopies();

  DeferredSliceCopies(const DeferredSliceCopies&) = delete;
  DeferredSliceCopies& operator=(const DeferredSliceCopies&) = delete;

  /// Records the copies of one slice queued on 'stream'. 'owner' and
  /// 'tableOwner', whichever is set, keep the device source alive.
  void add(
      cuda::stream_ref stream,
      std::shared_ptr<void> owner,
      std::shared_ptr<void> tableOwner);

  /// Waits for every queued copy, synchronizing each stream once, and then
  /// releases the device sources. Throws if a synchronization fails, after
  /// trying the remaining streams.
  void complete();

  /// Returns the number of slices whose copies were added.
  uint64_t numSlices() const;

  /// Returns the number of stream synchronizations complete() has made.
  uint64_t numSyncs() const;

 private:
  mutable std::mutex mutex_;
  // Streams with copies that have not been waited for.
  std::vector<cuda::stream_ref> streams_;
  // Device owners of the sources of those copies.
  std::vector<std::shared_ptr<void>> parents_;
  uint64_t numSlices_{0};
  uint64_t numSyncs_{0};
};

/// A CudfVector that lives in pinned host memory instead of on the device.
/// Move-only.
///
/// Each device buffer of each column is copied to a pinned host buffer of its
/// own, so spilling a table needs no device allocation. The host bytes are
/// admitted by the query's host spill budget before anything is copied.
class SpilledCudfVector {
 public:
  ~SpilledCudfVector();

  SpilledCudfVector(SpilledCudfVector&&) noexcept;
  SpilledCudfVector& operator=(SpilledCudfVector&&) noexcept;
  SpilledCudfVector(const SpilledCudfVector&) = delete;
  SpilledCudfVector& operator=(const SpilledCudfVector&) = delete;

  /// Moves 'resident' to host memory and frees its device memory. 'resident'
  /// must be the only reference to a vector that owns its table. On failure
  /// 'resident' is a valid device vector again and the error propagates.
  static SpilledCudfVector spill(
      CudfVectorPtr& resident,
      std::shared_ptr<CudfSpillContext> context);

  /// Copies the rows of 'view' to host memory. 'view' is a slice of a device
  /// table that 'owner' or 'tableOwner' keeps alive; the table itself is
  /// neither copied on the device nor freed. Returns nullopt when a column is
  /// not fixed-width or has nulls. The result records bytes copied, not bytes
  /// released: other slices may still use the device table.
  ///
  /// Without 'deferred' the copies are complete on return. With it they are
  /// only queued; 'deferred' then keeps the owners alive, and the result
  /// completes 'deferred' before its host data is read or freed.
  static std::optional<SpilledCudfVector> trySpillFlatSlice(
      memory::MemoryPool* pool,
      const TypePtr& type,
      cudf::table_view view,
      cuda::stream_ref stream,
      std::shared_ptr<void> owner,
      std::shared_ptr<void> tableOwner,
      std::shared_ptr<CudfSpillContext> context,
      std::shared_ptr<DeferredSliceCopies> deferred = nullptr);

  /// Returns a device copy allocated from 'mr' on 'stream'. The host copy
  /// stays valid. The copies have completed on return.
  CudfVectorPtr restore(
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  /// Like restore() on a stream taken from the global stream pool.
  CudfVectorPtr restore(rmm::device_async_resource_ref mr) const;

  /// Restores 'sources' into one device table that holds their rows in order:
  /// one allocation per column and one stream synchronization. Returns
  /// nullptr, with nothing changed, unless every source has the same columns,
  /// all fixed-width and without null masks.
  static std::unique_ptr<cudf::table> tryRestoreConcatenated(
      const std::vector<const SpilledCudfVector*>& sources,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Returns the row count of the vector.
  vector_size_t numRows() const;

  /// Returns the device bytes the vector occupied, which restore() allocates
  /// again.
  uint64_t deviceBytes() const;

  /// Returns the pinned host bytes held.
  uint64_t hostBytes() const;

 private:
  struct Impl;

  explicit SpilledCudfVector(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace facebook::velox::cudf_velox
