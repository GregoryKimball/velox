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
#include "velox/experimental/cudf/exec/HostSpillBudget.h"

#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <utility>

namespace facebook::velox::cudf_velox {

HostSpillBudget::HostSpillBudget(
    std::string name,
    uint64_t limit,
    std::shared_ptr<HostSpillBudget> parent)
    : name_(std::move(name)), limit_(limit), parent_(std::move(parent)) {}

std::shared_ptr<HostSpillBudget> HostSpillBudget::createWorker(
    uint64_t limitBytes) {
  return std::shared_ptr<HostSpillBudget>(
      new HostSpillBudget("worker", limitBytes));
}

std::shared_ptr<HostSpillBudget> HostSpillBudget::createQuery(
    std::string queryId,
    uint64_t limitBytes) {
  VELOX_CHECK_NULL(parent_, "Only a worker budget can create query budgets");
  VELOX_USER_CHECK_LE(
      limitBytes, limit_, "Query spill limit exceeds worker spill limit");
  return std::shared_ptr<HostSpillBudget>(
      new HostSpillBudget(std::move(queryId), limitBytes, shared_from_this()));
}

void HostSpillBudget::checkLocked(uint64_t bytes) const {
  if (bytes > limit_ - used_) {
    _VELOX_THROW(
        VeloxRuntimeError,
        error_source::kErrorSourceRuntime.c_str(),
        error_code::kMemCapExceeded.c_str(),
        false, // Retry on the same host budget cannot make this query fit.
        "Host spill budget exceeded: scope={}, limitBytes={}, usedBytes={}, requestedBytes={}. "
        "Disk spilling is disabled.",
        name_,
        limit_,
        used_,
        bytes);
  }
}

HostSpillBudget::Reservation HostSpillBudget::reserve(uint64_t bytes) {
  if (!bytes)
    return {};
  auto self = shared_from_this();
  if (parent_) {
    std::scoped_lock lock(mutex_, parent_->mutex_);
    checkLocked(bytes);
    parent_->checkLocked(bytes);
    used_ += bytes;
    parent_->used_ += bytes;
    peak_ = std::max(peak_, used_);
    parent_->peak_ = std::max(parent_->peak_, parent_->used_);
  } else {
    std::lock_guard lock(mutex_);
    checkLocked(bytes);
    used_ += bytes;
    peak_ = std::max(peak_, used_);
  }
  return Reservation({std::move(self), parent_}, bytes);
}

uint64_t HostSpillBudget::used() const {
  std::lock_guard lock(mutex_);
  return used_;
}

uint64_t HostSpillBudget::peak() const {
  std::lock_guard lock(mutex_);
  return peak_;
}

HostSpillBudget::Reservation::Reservation(
    std::array<std::shared_ptr<HostSpillBudget>, 2> budgets,
    uint64_t bytes) noexcept
    : budgets_(std::move(budgets)), bytes_(bytes) {}

HostSpillBudget::Reservation::~Reservation() {
  release();
}

HostSpillBudget::Reservation::Reservation(Reservation&& other) noexcept
    : budgets_(std::move(other.budgets_)),
      bytes_(std::exchange(other.bytes_, 0)) {}

HostSpillBudget::Reservation& HostSpillBudget::Reservation::operator=(
    Reservation&& other) noexcept {
  if (this != &other) {
    release();
    budgets_ = std::move(other.budgets_);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

HostSpillBudget::Reservation HostSpillBudget::Reservation::split(
    uint64_t bytes) {
  VELOX_CHECK_LE(bytes, bytes_, "Allocation exceeds admitted spill bytes");
  if (!bytes)
    return {};
  Reservation result(budgets_, bytes);
  bytes_ -= bytes;
  return result;
}

bool HostSpillBudget::Reservation::belongsTo(
    const HostSpillBudget* worker) const {
  return worker &&
      (budgets_[1] ? budgets_[1].get() : budgets_[0].get()) == worker;
}

void HostSpillBudget::Reservation::release() noexcept {
  if (!bytes_)
    return;
  auto& self = *budgets_[0];
  if (budgets_[1]) {
    auto& parent = *budgets_[1];
    std::scoped_lock lock(self.mutex_, parent.mutex_);
    CHECK_GE(self.used_, bytes_);
    CHECK_GE(parent.used_, bytes_);
    self.used_ -= bytes_;
    parent.used_ -= bytes_;
  } else {
    std::lock_guard lock(self.mutex_);
    CHECK_GE(self.used_, bytes_);
    self.used_ -= bytes_;
  }
  bytes_ = 0;
}

} // namespace facebook::velox::cudf_velox
