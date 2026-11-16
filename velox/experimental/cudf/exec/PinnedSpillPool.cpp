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
#include "velox/experimental/cudf/exec/PinnedSpillPool.h"

#include "velox/common/base/Exceptions.h"

#include <cuda/memory_pool>

#include <limits>
#include <utility>

namespace facebook::velox::cudf_velox {

struct PinnedSpillPool::Impl {
  Impl(std::size_t limit, std::optional<int> node) {
    cuda::memory_pool_properties properties{};
    properties.max_pool_size = limit;
    properties.release_threshold = limit;
    if (node) {
      pool = std::make_unique<cuda::pinned_memory_pool>(*node, properties);
    } else {
#if CUDART_VERSION >= 13000
      pool = std::make_unique<cuda::pinned_memory_pool>(properties);
#else
      pool = std::make_unique<cuda::pinned_memory_pool>(0, properties);
#endif
    }
  }
  std::unique_ptr<cuda::pinned_memory_pool> pool;
};

PinnedSpillPool::PinnedSpillPool(std::size_t bytes, std::optional<int> node)
    : budget_(HostSpillBudget::createWorker(bytes)),
      impl_(std::make_unique<Impl>(bytes, node)) {}

std::shared_ptr<PinnedSpillPool> PinnedSpillPool::create(
    std::size_t bytes,
    std::optional<int> node) {
  VELOX_USER_CHECK_GT(
      bytes, 0, "Host spill budget must be explicitly configured");
  VELOX_USER_CHECK_LE(
      bytes, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  VELOX_USER_CHECK_EQ(
      bytes % 256, 0, "Host spill budget must be 256-byte aligned");
  if (node)
    VELOX_USER_CHECK_GE(*node, 0);
  return std::shared_ptr<PinnedSpillPool>(new PinnedSpillPool(bytes, node));
}

PinnedSpillPool::~PinnedSpillPool() = default;

std::size_t PinnedSpillPool::reservedBytes() const {
  return impl_->pool->attribute(
      cuda::memory_pool_attributes::reserved_mem_current);
}

PinnedSpillPool::Reservation PinnedSpillPool::reserve(
    const std::shared_ptr<HostSpillBudget>& queryBudget,
    std::size_t bytes) {
  if (!bytes) {
    return {};
  }
  VELOX_CHECK_NOT_NULL(queryBudget);
  auto admitted = queryBudget->reserve(bytes);
  VELOX_CHECK(
      admitted.belongsTo(budget_.get()),
      "Spill reservation belongs to another worker pool");
  auto owner = shared_from_this();
  return Reservation(std::move(owner), std::move(admitted));
}

PinnedSpillPool::Allocation PinnedSpillPool::allocate(
    Reservation& admitted,
    std::size_t bytes) {
  if (!bytes)
    return {};
  VELOX_CHECK(admitted.owner_.get() == this, "Spill reservation belongs to another pool");
  VELOX_CHECK_LE(bytes, admitted.bytes(), "Spill allocation exceeds admission");
  void* data;
  try {
    data = impl_->pool->allocate_sync(bytes);
  } catch (const cuda::cuda_error& error) {
    if (error.status() == cudaErrorMemoryAllocation) {
      _VELOX_THROW(
          VeloxRuntimeError,
          error_source::kErrorSourceRuntime.c_str(),
          error_code::kMemCapExceeded.c_str(),
          false,
          "Host spill pool exhausted: limitBytes={}, requestedBytes={}. Disk spilling is disabled.",
          budget_->limit(),
          bytes);
    }
    throw;
  }
  // split() cannot fail after the checked bound; the returned allocation owns
  // the actual charge while unused reservation remains with the operation.
  return Allocation(
      admitted.owner_,
      admitted.admitted_.split(bytes),
      data,
      bytes);
}

PinnedSpillPool::Reservation::Reservation(
    std::shared_ptr<PinnedSpillPool> owner,
    HostSpillBudget::Reservation admitted) noexcept
    : owner_(std::move(owner)),
      admitted_(std::move(admitted)) {}

PinnedSpillPool::Reservation::~Reservation() {
  release();
}

PinnedSpillPool::Reservation::Reservation(Reservation&& other) noexcept
    : owner_(std::move(other.owner_)),
      admitted_(std::move(other.admitted_)) {}

PinnedSpillPool::Reservation& PinnedSpillPool::Reservation::operator=(
    Reservation&& other) noexcept {
  if (this != &other) {
    release();
    owner_ = std::move(other.owner_);
    admitted_ = std::move(other.admitted_);
  }
  return *this;
}

void PinnedSpillPool::Reservation::release() noexcept {
  admitted_ = {};
}

PinnedSpillPool::Allocation PinnedSpillPool::Reservation::allocate(
    std::size_t bytes) {
  if (!bytes)
    return {};
  VELOX_CHECK_NOT_NULL(owner_);
  return owner_->allocate(*this, bytes);
}

PinnedSpillPool::Allocation::Allocation(
    std::shared_ptr<PinnedSpillPool> owner,
    HostSpillBudget::Reservation charge,
    void* data,
    std::size_t bytes) noexcept
    : owner_(std::move(owner)),
      charge_(std::move(charge)),
      data_(data),
      bytes_(bytes) {}

PinnedSpillPool::Allocation::~Allocation() {
  reset();
}

PinnedSpillPool::Allocation::Allocation(Allocation&& other) noexcept
    : owner_(std::move(other.owner_)),
      charge_(std::move(other.charge_)),
      data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

PinnedSpillPool::Allocation& PinnedSpillPool::Allocation::operator=(
    Allocation&& other) noexcept {
  if (this != &other) {
    reset();
    owner_ = std::move(other.owner_);
    charge_ = std::move(other.charge_);
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void PinnedSpillPool::Allocation::reset() noexcept {
  if (!data_)
    return;
  owner_->impl_->pool->deallocate_sync(data_, bytes_);
  data_ = nullptr;
  bytes_ = 0;
  charge_ = {};
}

} // namespace facebook::velox::cudf_velox
