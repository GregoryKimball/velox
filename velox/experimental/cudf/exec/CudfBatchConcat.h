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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

#include <cstdint>
#include <optional>
#include <queue>

namespace facebook::velox::cudf_velox {

/// When a batch that concatenates inputs is complete: at
/// CudfConfig::batchSizeMinBytes estimated GPU bytes when that is set,
/// otherwise at CudfConfig::batchSizeMinThreshold rows or a fixed share of the
/// device memory available to the process, whichever comes first. The share
/// keeps a row target sized for ample memory from setting a memory floor on a
/// smaller device. A batch without columns holds no GPU buffers, so only its
/// rows count.
class BatchConcatTarget {
 public:
  explicit BatchConcatTarget(const RowTypePtr& outputType);

  /// Whether 'numRows' rows that hold 'numBytes' estimated GPU bytes complete
  /// a batch.
  bool isMet(size_t numRows, uint64_t numBytes) const;

  /// Whether isMet() reads its byte count, so that a caller can skip
  /// measuring.
  bool countsBytes() const {
    return targetBytes_.has_value() || availableMemoryShareBytes_ > 0;
  }

 private:
  // Byte target from batchSizeMinBytes. Unset for an output without columns.
  std::optional<uint64_t> targetBytes_;

  // Row target from batchSizeMinThreshold, used when targetBytes_ is unset.
  size_t targetRows_;

  // With the row target, the share of available device memory that also
  // completes a batch. Zero when that memory is unknown or the output has no
  // columns.
  uint64_t availableMemoryShareBytes_;
};

class CudfBatchConcat : public CudfOperatorBase {
 public:
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode);

  bool needsInput() const override {
    return !noMoreInput_ && outputQueue_.empty() && !targetReached();
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  bool targetReached() const {
    return target_.isMet(currentNumRows_, currentBytes_);
  }

  // Returns the estimated GPU bytes of 'vector', or 0 when no byte limit
  // applies.
  uint64_t estimateBytes(const CudfVector& vector) const;

  // Input vectors awaiting concatenation.
  std::vector<CudfVectorPtr> buffer_;

  // Concatenated vectors ready for downstream consumption.
  std::queue<CudfVectorPtr> outputQueue_;

  // Rows held in buffer_.
  size_t currentNumRows_{0};

  // Estimated GPU bytes held in buffer_. Tracked when target_ counts bytes.
  uint64_t currentBytes_{0};

  const BatchConcatTarget target_;
};

} // namespace facebook::velox::cudf_velox
