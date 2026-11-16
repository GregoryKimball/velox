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

#include "velox/experimental/cudf/exec/CudfJoinBuild.h"
#include "velox/experimental/cudf/exec/CudfJoinSpillState.h"
#include "velox/experimental/cudf/exec/CudfSpillContext.h"

#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Task.h"

#include <folly/ScopeGuard.h>

#include <iterator>
#include <typeinfo>
#include <utility>

namespace facebook::velox::cudf_velox {

CudfJoinBuild::CudfJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::PlanNode>& joinNode,
    const std::string& operatorName,
    NvtxMethodFlag nvtxMethods)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          operatorName,
          nvtx3::rgb{65, 105, 225},
          nvtxMethods,
          std::nullopt,
          joinNode) {
  if (gpuPool() == nullptr) {
    return;
  }
  spillContext_ = cudfSpillContext(*driverCtx->task->queryCtx());
  if (spillContext_ != nullptr) {
    spillStats_ = std::make_shared<CudfJoinSpillStats>();
    spillState_ =
        std::make_unique<CudfJoinSpillState>(spillContext_, spillStats_);
  }
}

CudfJoinBuild::~CudfJoinBuild() = default;

bool CudfJoinBuild::needsInput() const {
  return !noMoreInput_;
}

exec::BlockingReason CudfJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

void CudfJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  recordInputStats(*cudfInput);
  if (spillState_ == nullptr) {
    inputs_.push_back(std::move(cudfInput));
    return;
  }
  input.reset();
  spillState_->append(adoptInput(std::move(cudfInput)));
  publishGpuReclaimableEstimate(spillState_->reclaimableBytes());
}

CudfVectorPtr CudfJoinBuild::adoptInput(CudfVectorPtr input) {
  // The input is copied into a table of this operator, for three reasons. The
  // driver still holds the input during addInput(), and only a vector with
  // one owner can be moved to host memory. The memory of the input stays
  // charged to the operator that produced it, while this operator's reclaimer
  // answers for this operator's pool. And a packed input cannot be taken
  // apart into buffers, which must not need device memory during a reclaim.
  const auto stream = input->stream();
  auto table = std::make_unique<cudf::table>(
      input->getTableView(), stream, get_output_mr());
  // The copy reads the input, which the driver frees after addInput().
  stream.sync();
  return std::make_shared<CudfVector>(
      pool(), input->type(), input->size(), std::move(table), stream);
}

RowVectorPtr CudfJoinBuild::doGetOutput() {
  return nullptr;
}

void CudfJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Operator>> peerOperators;
  if (!operatorCtx_->allPeersArrived(&future_, promises, peerOperators)) {
    return;
  }

  SCOPE_EXIT {
    peerOperators.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  for (const auto& peer : peerOperators) {
    auto* build = peer->as<CudfJoinBuild>();
    VELOX_CHECK_NOT_NULL(build);
    VELOX_CHECK(
        typeid(*build) == typeid(*this),
        "Expected peer build type {}, got {}",
        typeid(*this).name(),
        typeid(*build).name());
    inputs_.insert(
        inputs_.end(),
        std::make_move_iterator(build->inputs_.begin()),
        std::make_move_iterator(build->inputs_.end()));
    build->inputs_.clear();
    if (spillState_ != nullptr && build->spillState_ != nullptr) {
      // A peer's input that is on the device stays there: it is charged
      // already, so taking it over needs no memory.
      spillState_->adopt(std::move(*build->spillState_));
      build->publishGpuReclaimableEstimate(0);
    }

    auto retainedInputBatches = build->inputs_.size();
    common::testutil::TestValue::adjust(
        "facebook::velox::cudf_velox::CudfJoinBuild::doNoMoreInput::sourceDriverRetainedInputBatchesAfterTransfer",
        &retainedInputBatches);
  }

  auto inputs = std::exchange(inputs_, {});
  if (spillState_ != nullptr) {
    // Take every chunk out of the state before any of it is read: what is
    // taken out can no longer be moved to host memory under the build.
    auto chunks = spillState_->takeChunks();
    publishGpuReclaimableEstimate(0);
    inputs.reserve(chunks.size());
    for (auto& chunk : chunks) {
      inputs.push_back(chunk.take(get_output_mr()));
    }
  }
  buildAndPublish(std::move(inputs));
}

void CudfJoinBuild::doClose() {
  recordSpillStats();
  inputs_.clear();
  spillState_.reset();
  spillContext_.reset();
  Operator::close();
}

bool CudfJoinBuild::gpuReclaimableBytes(uint64_t& reclaimableBytes) const {
  reclaimableBytes = spillState_ ? spillState_->reclaimableBytes() : 0;
  return reclaimableBytes != 0;
}

uint64_t CudfJoinBuild::spillGpu(uint64_t targetBytes) {
  return spillState_ ? spillState_->spill(targetBytes) : 0;
}

void CudfJoinBuild::recordSpillStats() {
  if (spillStats_ == nullptr) {
    return;
  }
  const auto add = [&](std::string_view name, uint64_t bytes) {
    if (bytes != 0) {
      addRuntimeStat(
          std::string(name),
          RuntimeCounter(
              static_cast<int64_t>(bytes), RuntimeCounter::Unit::kBytes));
    }
  };
  add(kSpilledBytes, spillStats_->spilledBytes.exchange(0));
  add(kRestoredBytes, spillStats_->restoredBytes.exchange(0));
  add(kHostBytesWritten, spillStats_->hostBytesWritten.exchange(0));
}

} // namespace facebook::velox::cudf_velox
