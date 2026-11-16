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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/DebugUtil.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/base/SpillConfig.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <glog/logging.h>

#include <atomic>
#include <memory>
#include <optional>
#include <type_traits>

namespace facebook::velox::cudf_velox {

class CudfMemoryResourceRegistry;

/// The device memory resources of one operator. When the query has a GPU
/// pool, the operator's allocations are charged to the operator's leaf of
/// that pool; otherwise they use the process resources as before.
class CudfOperatorMemory {
 public:
  /// 'gpuPool' is the operator's leaf of the query GPU pool, or nullptr.
  CudfOperatorMemory(exec::DriverCtx* driverCtx, memory::MemoryPool* gpuPool);

  /// Installs this operator's resources on the calling thread for the
  /// lifetime of the result. Installs nothing without a GPU pool.
  [[nodiscard]] ScopedCudfMemoryResources scoped() const;

  /// True if allocations are charged to a GPU pool.
  bool accounted() const {
    return tempMemoryResource_.has_value();
  }

 private:
  // Owns the accounting wrappers the two refs below point to.
  std::shared_ptr<CudfMemoryResourceRegistry> memoryResourceOwner_;
  std::optional<rmm::device_async_resource_ref> tempMemoryResource_;
  std::optional<rmm::device_async_resource_ref> outputMemoryResource_;
};

/// Bitmask controlling which operator methods get NVTX profiling ranges.
/// Operators pass flags for the do* methods they actually override, so
/// no-op methods don't pollute nsys profiles with empty ranges.
enum NvtxMethodFlag : uint8_t {
  kNone = 0,
  kAddInput = 1 << 0,
  kGetOutput = 1 << 1,
  kNoMoreInput = 1 << 2,
  kClose = 1 << 3,
  kAll = kAddInput | kGetOutput | kNoMoreInput | kClose,
};

inline NvtxMethodFlag operator|(NvtxMethodFlag a, NvtxMethodFlag b) {
  using EnumT = std::underlying_type_t<NvtxMethodFlag>;
  return static_cast<NvtxMethodFlag>(
      static_cast<EnumT>(a) | static_cast<EnumT>(b));
}

inline NvtxMethodFlag operator&(NvtxMethodFlag a, NvtxMethodFlag b) {
  using EnumT = std::underlying_type_t<NvtxMethodFlag>;
  return static_cast<NvtxMethodFlag>(
      static_cast<EnumT>(a) & static_cast<EnumT>(b));
}

/// The user defined operator will inherit this operator, the operator accepts
/// CudfOperator and output CudfVector.
class CudfOperator : public NvtxHelper {
 public:
  CudfOperator(
      int32_t operatorId,
      const core::PlanNodeId& nodeId,
      std::optional<nvtx3::color> color = std::nullopt)
      : NvtxHelper(color, operatorId, fmt::format("[{}]", nodeId)) {}
};

/// Base class for all built-in cuDF operators in Velox.
///
/// All cuDF operators MUST extend this class rather than extending
/// exec::Operator and NvtxHelper directly. This class implements the template
/// method pattern:
/// the public operator interface methods (addInput, getOutput, noMoreInput,
/// close) are marked final and must NOT be overridden by derived classes.
/// Instead, derived classes should ONLY override the corresponding protected
/// do* virtual methods:
///   - doAddInput()    -- receives input rows; called by addInput()
///   - doGetOutput()   -- produces output rows; called by getOutput()
///   - doNoMoreInput() -- signals end of input; called by noMoreInput()
///                        (defaults to Operator::noMoreInput())
///   - doClose()       -- releases resources; called by close()
///                        (defaults to Operator::close())
///
/// This design ensures that NVTX profiling ranges are applied uniformly
/// across all operators without each subclass having to manage them. The
/// nvtxMethods bitmask (NvtxMethodFlag) lets operators suppress NVTX ranges
/// for do* methods they do not override, keeping nsys profiles clean.
///
/// Example:
///   class MyCudfOperator : public CudfOperatorBase {
///    public:
///     MyCudfOperator(int32_t operatorId, exec::DriverCtx* ctx,
///                    RowTypePtr outputType, const core::PlanNodeId& nodeId)
///         : CudfOperatorBase(
///               operatorId, ctx, outputType, nodeId, "MyCudfOperator",
///               std::nullopt,
///               NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput) {}
///
///     bool needsInput() const override { return !noMoreInput_; }
///
///    protected:
///     void doAddInput(RowVectorPtr input) override { /* process input */ }
///     RowVectorPtr doGetOutput() override { /* return output or nullptr */ }
///   };
class CudfOperatorBase : public exec::Operator, public NvtxHelper {
 public:
  CudfOperatorBase(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId,
      const std::string& operatorName,
      std::optional<nvtx3::color> color = std::nullopt,
      NvtxMethodFlag nvtxMethods = NvtxMethodFlag::kAll,
      std::optional<common::SpillConfig> spillConfig = std::nullopt,
      std::optional<std::shared_ptr<const core::PlanNode>> planNode =
          std::nullopt)
      : Operator(
            driverCtx,
            outputType,
            operatorId,
            planNodeId,
            operatorName,
            spillConfig),
        NvtxHelper(color, operatorId, fmt::format("[{}]", planNodeId)),
        className_(operatorName),
        nvtxMethods_(nvtxMethods),
        gpuMemory_(driverCtx, gpuPool()) {}

  void addInput(RowVectorPtr input) final {
    ensureCudaContextForThread();
    maybeSetGpuMemoryReclaimer();
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kAddInput, className_);
    doAddInput(std::move(input));
    checkCudaErrorInDebug();
  }

  RowVectorPtr getOutput() final {
    ensureCudaContextForThread();
    maybeSetGpuMemoryReclaimer();
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kGetOutput, className_);
    auto result = doGetOutput();
    checkCudaErrorInDebug();
    return result;
  }

  void noMoreInput() final {
    ensureCudaContextForThread();
    maybeSetGpuMemoryReclaimer();
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kNoMoreInput, className_);
    doNoMoreInput();
    checkCudaErrorInDebug();
  }

  void close() final {
    // close() may run on a different thread than construction so bind the
    // context here too.
    ensureCudaContextForThread();
    publishGpuReclaimableEstimate(0);
    {
      auto memoryResources = scopedMemoryResources();
      VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
          nvtxMethods_ & NvtxMethodFlag::kClose, className_);
      doClose();
      checkCudaErrorInDebug();
    }
    recordGpuMemoryStats();
  }

 protected:
  virtual void doAddInput(RowVectorPtr input) = 0;

  virtual RowVectorPtr doGetOutput() = 0;

  virtual void doNoMoreInput() {
    Operator::noMoreInput();
  }

  virtual void doClose() {
    Operator::close();
  }

  /// Installs this operator's device memory resources on the calling thread.
  /// The four entry points above install them already. An override of
  /// initialize() or isBlocked() that allocates device memory installs them
  /// itself.
  [[nodiscard]] ScopedCudfMemoryResources scopedMemoryResources() const {
    return gpuMemory_.scoped();
  }

  /// Returns the operator's leaf of the query GPU pool, or nullptr when the
  /// query has no GPU pool.
  memory::MemoryPool* gpuPool() const;

  /// Publishes how many device bytes this operator could release, for victim
  /// selection. Called from the driver thread, or from a reclaimer while the
  /// task is paused, at stable points of the operator's state. Readers load
  /// only this value and never walk operator state. Publish newly reclaimable
  /// state before the allocations that follow it can start arbitration.
  void publishGpuReclaimableEstimate(uint64_t bytes) noexcept {
    gpuReclaimableEstimate_.store(bytes, std::memory_order_relaxed);
  }

  /// Sets 'reclaimableBytes' to the device bytes this operator can release
  /// now and returns whether that is more than zero. Called while the task is
  /// paused, so it may walk operator state. Excludes state that an active
  /// call still uses.
  ///
  /// Separate from Operator::reclaimableBytes(), which belongs to the
  /// operator's CPU pool. This one belongs to its GPU leaf pool.
  virtual bool gpuReclaimableBytes(uint64_t& reclaimableBytes) const {
    reclaimableBytes = 0;
    return false;
  }

  /// Returns true if inactive state can be released while a call of this
  /// operator is parked on the driver's stack with its operands in use. Such
  /// an operator is asked to release memory inside a non-reclaimable section
  /// too.
  virtual bool supportsGpuSpillWithActiveCall() const {
    return false;
  }

  /// Releases device memory by moving eligible state off the device. Called
  /// while the task is paused, with this operator's memory resources
  /// installed. 'targetBytes' of zero asks for everything eligible. Returns
  /// the device bytes released once the device has completed the work; zero
  /// is a valid result. Failures propagate.
  virtual uint64_t spillGpu(uint64_t /*targetBytes*/) {
    return 0;
  }

 private:
  class GpuMemoryReclaimer;

  // Gives the GPU leaf pool a reclaimer that calls spillGpu(), once, when the
  // query has a GPU pool whose parent takes part in arbitration.
  void maybeSetGpuMemoryReclaimer();

  // Reports the peak device bytes of this operator and of the query.
  void recordGpuMemoryStats();

  const std::string className_;
  const NvtxMethodFlag nvtxMethods_;
  const CudfOperatorMemory gpuMemory_;
  bool gpuMemoryReclaimerChecked_{false};
  std::atomic<uint64_t> gpuReclaimableEstimate_{0};
};

/// Base class for cuDF source operators (first operator in a Driver pipeline).
///
/// This mirrors CudfOperatorBase but inherits from exec::SourceOperator instead
/// of exec::Operator. Source operators never accept input (addInput/noMoreInput
/// are disabled by SourceOperator), so this class only wraps getOutput and
/// close with NVTX profiling and checkCudaErrorInDebug.
///
/// Derived classes override:
///   - doGetOutput()   -- produces output rows; called by getOutput()
///   - doClose()       -- releases resources; called by close()
///                        (defaults to SourceOperator::close())
class CudfSourceOperatorBase : public exec::SourceOperator, public NvtxHelper {
 public:
  CudfSourceOperatorBase(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId,
      const std::string& operatorName,
      std::optional<nvtx3::color> color = std::nullopt,
      NvtxMethodFlag nvtxMethods = NvtxMethodFlag::kGetOutput |
          NvtxMethodFlag::kClose)
      : SourceOperator(
            driverCtx,
            outputType,
            operatorId,
            planNodeId,
            operatorName),
        NvtxHelper(color, operatorId, fmt::format("[{}]", planNodeId)),
        className_(operatorName),
        nvtxMethods_(nvtxMethods),
        gpuMemory_(driverCtx, gpuPool()) {}

  RowVectorPtr getOutput() final {
    ensureCudaContextForThread();
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kGetOutput, className_);
    auto result = doGetOutput();
    checkCudaErrorInDebug();
    return result;
  }

  void close() final {
    // close() may run on a different thread than construction so bind the
    // context here too.
    ensureCudaContextForThread();
    auto memoryResources = scopedMemoryResources();
    VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(
        nvtxMethods_ & NvtxMethodFlag::kClose, className_);
    doClose();
    checkCudaErrorInDebug();
  }

 protected:
  virtual RowVectorPtr doGetOutput() = 0;

  virtual void doClose() {
    SourceOperator::close();
  }

  /// Installs this operator's device memory resources on the calling thread.
  /// getOutput() and close() install them already.
  [[nodiscard]] ScopedCudfMemoryResources scopedMemoryResources() const {
    return gpuMemory_.scoped();
  }

  /// Returns the operator's leaf of the query GPU pool, or nullptr when the
  /// query has no GPU pool.
  memory::MemoryPool* gpuPool() const;

 private:
  const std::string className_;
  const NvtxMethodFlag nvtxMethods_;
  const CudfOperatorMemory gpuMemory_;
};

} // namespace facebook::velox::cudf_velox
