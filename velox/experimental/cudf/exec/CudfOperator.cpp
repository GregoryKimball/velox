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
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/exec/Driver.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Task.h"

namespace facebook::velox::cudf_velox {

CudfOperatorMemory::CudfOperatorMemory(
    exec::DriverCtx* driverCtx,
    memory::MemoryPool* gpuPool) {
  if (gpuPool == nullptr) {
    return;
  }
  VELOX_CHECK(
      mr_.has_value() && output_mr_.has_value(),
      "cuDF memory resources must be initialized before creating operators");
  memoryResourceOwner_ =
      cudfMemoryResourceRegistry(*driverCtx->task->queryCtx());
  auto resources = memoryResourceOwner_->resourcesFor(
      *mr_, *output_mr_, gpuPool->shared_from_this());
  tempMemoryResource_ = resources.temp;
  outputMemoryResource_ = resources.output;
}

ScopedCudfMemoryResources CudfOperatorMemory::scoped() const {
  if (!accounted()) {
    return ScopedCudfMemoryResources{};
  }
  return ScopedCudfMemoryResources{
      *tempMemoryResource_, *outputMemoryResource_, memoryResourceOwner_};
}

// Releases device memory of one cuDF operator on behalf of the arbitrator of
// the GPU resource. Mirrors Operator::MemoryReclaimer, for the operator's GPU
// leaf pool instead of its CPU pool.
class CudfOperatorBase::GpuMemoryReclaimer final
    : public exec::MemoryReclaimer {
 public:
  static std::unique_ptr<memory::MemoryReclaimer> create(
      exec::DriverCtx* driverCtx,
      CudfOperatorBase* op) {
    VELOX_CHECK_NOT_NULL(driverCtx);
    VELOX_CHECK_NOT_NULL(driverCtx->driver);
    return std::unique_ptr<memory::MemoryReclaimer>(
        new GpuMemoryReclaimer(driverCtx->driver->shared_from_this(), op));
  }

  bool reclaimableBytes(
      const memory::MemoryPool& pool,
      uint64_t& reclaimableBytes) const override {
    reclaimableBytes = 0;
    auto driver = ensureDriver();
    if (FOLLY_UNLIKELY(driver == nullptr)) {
      return false;
    }

    VELOX_CHECK(
        std::addressof(pool) == op_->gpuPool(),
        "GPU reclaimer invoked for a pool other than its operator's cuDF pool");
    // Victim selection happens before the task is paused. Operator state is
    // not safe to walk here, also when the published value is stale or zero.
    reclaimableBytes =
        op_->gpuReclaimableEstimate_.load(std::memory_order_relaxed);
    return reclaimableBytes != 0;
  }

  uint64_t reclaim(
      memory::MemoryPool* pool,
      uint64_t targetBytes,
      uint64_t /*maxWaitMs*/,
      memory::MemoryReclaimer::Stats& stats) override {
    auto driver = ensureDriver();
    if (FOLLY_UNLIKELY(driver == nullptr)) {
      return 0;
    }

    VELOX_CHECK_NOT_NULL(pool);
    VELOX_CHECK(
        pool == op_->gpuPool(),
        "GPU reclaimer invoked for a pool other than its operator's cuDF pool");

    VELOX_CHECK(
        !driver->state().isOnThread() || driver->state().suspended() ||
            driver->state().isTerminated,
        "driverOnThread {}, driverSuspended {} driverTerminated {} {}",
        driver->state().isOnThread(),
        driver->state().suspended(),
        driver->state().isTerminated,
        pool->name());
    VELOX_CHECK(
        driver->task()->pauseRequested(),
        "GPU memory reclaim requires the owning task to be paused");

    if (op_->nonReclaimableSection_ &&
        !op_->supportsGpuSpillWithActiveCall()) {
      ++stats.numNonReclaimableAttempts;
      return 0;
    }

    uint64_t reclaimableBytes{0};
    if (!op_->gpuReclaimableBytes(reclaimableBytes) || reclaimableBytes == 0) {
      op_->publishGpuReclaimableEstimate(0);
      return 0;
    }

    RuntimeStatWriterScopeGuard opStatsGuard(op_);

    return memory::MemoryReclaimer::run(
        [&]() {
          int64_t reclaimedBytes{0};
          {
            memory::ScopedReclaimedBytesRecorder recorder(
                pool, &reclaimedBytes);
            ensureCudaContextForThread();
            auto memoryResources = op_->scopedMemoryResources();
            // spillGpu() returns the device bytes it released. Arbitration
            // credits the change in the pool's reservation instead, which
            // the recorder measures.
            op_->spillGpu(targetBytes);
            checkCudaErrorInDebug();
            uint64_t remainingBytes{0};
            op_->gpuReclaimableBytes(remainingBytes);
            op_->publishGpuReclaimableEstimate(remainingBytes);
          }
          VELOX_CHECK_GE(
              reclaimedBytes,
              0,
              "Unexpected GPU memory growth after reclaim from pool {}",
              pool->name());
          return reclaimedBytes;
        },
        stats);
  }

 private:
  GpuMemoryReclaimer(
      const std::shared_ptr<exec::Driver>& driver,
      CudfOperatorBase* op)
      : exec::MemoryReclaimer(0), driver_(driver), op_(op) {
    VELOX_CHECK_NOT_NULL(op_);
  }

  std::shared_ptr<exec::Driver> ensureDriver() const {
    return driver_.lock();
  }

  const std::weak_ptr<exec::Driver> driver_;
  CudfOperatorBase* const op_;
};

memory::MemoryPool* CudfOperatorBase::gpuPool() const {
  return customPool(kCudfMemoryResourceTag);
}

void CudfOperatorBase::maybeSetGpuMemoryReclaimer() {
  if (gpuMemoryReclaimerChecked_) {
    return;
  }
  gpuMemoryReclaimerChecked_ = true;
  auto* pool = gpuPool();
  if (pool == nullptr || pool->reclaimer() != nullptr) {
    return;
  }
  auto* parent = pool->parent();
  if (parent == nullptr || parent->reclaimer() == nullptr) {
    return;
  }
  pool->setReclaimer(
      GpuMemoryReclaimer::create(operatorCtx()->driverCtx(), this));
}

void CudfOperatorBase::recordGpuMemoryStats() {
  auto* pool = gpuPool();
  if (pool == nullptr) {
    return;
  }
  addRuntimeStat(
      "cudfOperatorGpuPeakBytes",
      RuntimeCounter(
          static_cast<int64_t>(pool->peakBytes()),
          RuntimeCounter::Unit::kBytes));
  auto* queryPool = pool->root();
  addRuntimeStat(
      "cudfQueryGpuPeakBytes",
      RuntimeCounter(
          static_cast<int64_t>(queryPool->peakBytes()),
          RuntimeCounter::Unit::kBytes));
  // The configured limit of the query's GPU root, not what it is granted
  // now. Aggregate copies from several operators with max, not sum.
  addRuntimeStat(
      "cudfQueryGpuMaxCapacityBytes",
      RuntimeCounter(queryPool->maxCapacity(), RuntimeCounter::Unit::kBytes));
}

memory::MemoryPool* CudfSourceOperatorBase::gpuPool() const {
  return customPool(kCudfMemoryResourceTag);
}

} // namespace facebook::velox::cudf_velox
