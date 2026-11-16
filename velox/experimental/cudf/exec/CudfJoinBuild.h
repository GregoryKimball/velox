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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfJoinSpillState;
struct CudfJoinSpillStats;
struct CudfSpillContext;

/// Accumulates inputs and coordinates peer drivers for cuDF join builds.
///
/// When the query has a GPU pool and host spilling is on, the accumulated
/// input can be moved to pinned host memory while the build waits for more
/// input or for its peers. It is brought back when the build runs.
class CudfJoinBuild : public CudfOperatorBase {
 public:
  /// Counters of build input moved to host memory and back, in device bytes
  /// unless the name says host.
  static constexpr std::string_view kSpilledBytes{"cudfJoinSpilledBytes"};
  static constexpr std::string_view kRestoredBytes{"cudfJoinRestoredBytes"};
  static constexpr std::string_view kHostBytesWritten{
      "cudfJoinHostBytesWritten"};

  ~CudfJoinBuild() override;

  bool needsInput() const final;

  exec::BlockingReason isBlocked(ContinueFuture* future) final;

  bool isFinished() final;

 protected:
  CudfJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::PlanNode>& joinNode,
      const std::string& operatorName,
      NvtxMethodFlag nvtxMethods);

  void doAddInput(RowVectorPtr input) final;
  RowVectorPtr doGetOutput() final;
  void doNoMoreInput() final;
  void doClose() final;

  // Records per-input statistics before accumulation; no-op by default.
  virtual void recordInputStats(const CudfVector& input) {}

  // Consumes accumulated inputs and publishes the build result to the bridge.
  virtual void buildAndPublish(std::vector<CudfVectorPtr> inputs) = 0;

  // Called by the last driver with all build input when that input can spill,
  // before any of it is restored. A build that splits the input and publishes
  // it part by part takes 'state' and returns true. Otherwise the input is
  // restored whole and passed to buildAndPublish().
  virtual bool tryBuildAndPublishInParts(
      std::unique_ptr<CudfJoinSpillState>& /*state*/) {
    return false;
  }

  // Called from isBlocked() once the input has ended. A build that publishes
  // part by part does the work that is due and returns true, with 'future'
  // set, while it waits for the probes to finish the part it published.
  virtual bool waitForBuildWork(ContinueFuture* /*future*/) {
    return false;
  }

  // Returns false while a build that publishes part by part has parts left.
  virtual bool buildWorkFinished() {
    return true;
  }

  // The spill context of the query, or nullptr when the build cannot spill.
  const std::shared_ptr<CudfSpillContext>& spillContext() const {
    return spillContext_;
  }

  // The counters every spill state of this build reports to.
  const std::shared_ptr<CudfJoinSpillStats>& spillStats() const {
    return spillStats_;
  }

  bool gpuReclaimableBytes(uint64_t& reclaimableBytes) const override;

  bool supportsGpuSpillWithActiveCall() const override {
    return spillState_ != nullptr;
  }

  uint64_t spillGpu(uint64_t targetBytes) override;

 private:
  // Returns a copy of 'input' that this operator alone owns and is charged
  // for, so that it can be moved to host memory.
  CudfVectorPtr adoptInput(CudfVectorPtr input);

  // Reports the spill counters once.
  void recordSpillStats();

  // Build inputs accumulated by this driver and transferred from peer drivers.
  // Unused when 'spillState_' is set.
  std::vector<CudfVectorPtr> inputs_;
  // Set when the query has a GPU pool and host spilling is on.
  std::shared_ptr<CudfSpillContext> spillContext_;
  std::shared_ptr<CudfJoinSpillStats> spillStats_;
  // The accumulated build input when it can spill. Holds only input that no
  // call is using: a chunk is taken out before it is read.
  std::unique_ptr<CudfJoinSpillState> spillState_;

  // Waits for the last peer driver to finish collecting build inputs.
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

} // namespace facebook::velox::cudf_velox
