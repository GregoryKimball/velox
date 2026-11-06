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

#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <deque>

namespace facebook::velox::cudf_velox::connector::encoded {

/// Scans an encoded table on the GPU in place of a TableScan over an
/// EncodedConnector. Pulls splits from the task, prunes their segments by
/// min/max against the pushed-down filters, and decodes and filters the
/// surviving segments in batches that cross segment and split boundaries.
///
/// Each batch takes whole segments until the rows expected to pass the filters
/// reach CudfConfig::batchSizeMinBytes (estimated from the bytes per row of
/// earlier batches) or else CudfConfig::batchSizeMinThreshold. The expectation
/// uses the pass rate observed so far, starting from one segment. A batch
/// decodes at most CudfConfig::batchSizeMaxThreshold rows, or
/// kDefaultMaxDecodeRows when that is unset, but always at least one segment.
class CudfEncodedScan : public CudfSourceOperatorBase {
 public:
  static constexpr int64_t kDefaultMaxDecodeRows = 1 << 24;

  CudfEncodedScan(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TableScanNode>& tableScanNode);

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  RowVectorPtr doGetOutput() override;

 private:
  // Queues the surviving segments of splits until 'decodeRows' rows are
  // pending, the splits run out, or the task blocks. Returns false if blocked.
  bool fillPending(int64_t decodeRows);

  // Rows to decode for the next batch, from the targets and the pass rate.
  int64_t decodeRowsTarget() const;

  // Decodes and filters 'segments' into one batch, or null if no row passes.
  RowVectorPtr decodeBatch(const std::vector<int32_t>& segments);

  exec::DriverCtx* const driverCtx_;
  memory::MemoryPool* const connectorPool_;
  std::shared_ptr<velox::connector::ConnectorQueryCtx> connectorQueryCtx_;
  EncodedScanSpec spec_;

  cudf::ast::tree subfieldTree_;
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  const cudf::ast::expression* subfieldAst_{nullptr};
  std::shared_ptr<CudfExpression> remainingFilterExpression_;

  // Unpruned segments of the splits taken so far, not yet decoded.
  std::deque<int32_t> pendingSegments_;
  int64_t pendingRows_{0};
  bool noMoreSplits_{false};
  // Splits taken from the task since splitsFinished() was last called.
  int32_t unfinishedSplits_{0};

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};

  // Rows of the segments decoded so far, and of them those that passed.
  uint64_t decodedRows_{0};
  uint64_t passedRows_{0};
  // Rows and estimated bytes of the batches returned so far.
  uint64_t outputRows_{0};
  uint64_t outputBytes_{0};
};

} // namespace facebook::velox::cudf_velox::connector::encoded
