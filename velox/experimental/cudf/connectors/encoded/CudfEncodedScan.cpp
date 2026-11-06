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

#include "velox/experimental/cudf/connectors/encoded/CudfEncodedScan.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/time/Timer.h"
#include "velox/exec/Task.h"

#include <cudf/binaryop.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include "velox/experimental/cudf/CudfNoDefaults.h"

#include <algorithm>
#include <numeric>

namespace facebook::velox::cudf_velox::connector::encoded {

CudfEncodedScan::CudfEncodedScan(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TableScanNode>& tableScanNode)
    : CudfSourceOperatorBase(
          operatorId,
          driverCtx,
          tableScanNode->outputType(),
          tableScanNode->id(),
          "CudfEncodedScan",
          nvtx3::rgb{118, 185, 0}),
      driverCtx_(driverCtx),
      connectorPool_(driverCtx->task->addConnectorPoolLocked(
          planNodeId(),
          driverCtx->pipelineId,
          driverCtx->driverId,
          operatorType(),
          tableScanNode->tableHandle()->connectorId())),
      connectorQueryCtx_(operatorCtx_->createConnectorQueryCtx(
          tableScanNode->tableHandle()->connectorId(),
          planNodeId(),
          connectorPool_)),
      spec_(
          tableScanNode->outputType(),
          tableScanNode->tableHandle(),
          tableScanNode->assignments(),
          connectorQueryCtx_.get()) {
  ensureCudaContextForThread();
  if (!spec_.subfieldFilters.empty()) {
    subfieldAst_ = &createAstFromSubfieldFilters(
        spec_.subfieldFilters, subfieldTree_, subfieldScalars_, spec_.filterType);
  }
  if (spec_.remainingFilter) {
    remainingFilterExpression_ = createCudfExpression(
        spec_.remainingFilter,
        spec_.filterType,
        connectorQueryCtx_->memoryPool(),
        spec_.optimizeQueryCtx->queryConfig());
  }
  if (spec_.table->hostResident) {
    stagedColumnNames_ = spec_.filterColumnNames;
    stagedColumnNames_.insert(
        stagedColumnNames_.end(),
        spec_.outputColumnNames.begin(),
        spec_.outputColumnNames.end());
  }
}

void CudfEncodedScan::EventDeleter::operator()(cudaEvent_t event) const {
  cudaEventDestroy(event);
}

exec::BlockingReason CudfEncodedScan::isBlocked(ContinueFuture* future) {
  if (blockingFuture_.valid()) {
    *future = std::move(blockingFuture_);
    return blockingReason_;
  }
  return exec::BlockingReason::kNotBlocked;
}

bool CudfEncodedScan::isFinished() {
  return noMoreSplits_ && pendingSegments_.empty() && !prefetched_;
}

int64_t CudfEncodedScan::decodeRowsTarget() const {
  const auto& config = CudfConfig::getInstance();
  const double passRate = decodedRows_ == 0
      ? 1.0
      : std::max(1.0 * passedRows_ / decodedRows_, 1e-6);
  double targetRows = config.batchSizeMinThreshold;
  if (config.batchSizeMinBytes.has_value() && outputRows_ > 0 &&
      outputBytes_ > 0) {
    targetRows = 1.0 * *config.batchSizeMinBytes * outputRows_ / outputBytes_;
  }
  const int64_t maxRows =
      config.batchSizeMaxThreshold.value_or(kDefaultMaxDecodeRows);
  return std::min<int64_t>(maxRows, static_cast<int64_t>(targetRows / passRate));
}

bool CudfEncodedScan::fillPending(int64_t decodeRows) {
  while (!noMoreSplits_ && pendingRows_ < decodeRows) {
    exec::Split split;
    blockingReason_ = driverCtx_->task->getSplitOrFuture(
        driverCtx_->driverId,
        driverCtx_->splitGroupId,
        planNodeId(),
        /*maxPreloadSplits=*/0,
        /*preload=*/nullptr,
        split,
        blockingFuture_);
    if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
      return false;
    }
    VELOX_CHECK(
        !split.isBarrier(), "CudfEncodedScan does not support barriers");
    if (!split.hasConnectorSplit()) {
      noMoreSplits_ = true;
      break;
    }
    ++unfinishedSplits_;
    const auto encodedSplit =
        std::dynamic_pointer_cast<EncodedConnectorSplit>(split.connectorSplit);
    VELOX_CHECK_NOT_NULL(encodedSplit, "Expected an EncodedConnectorSplit");
    VELOX_CHECK_EQ(encodedSplit->table, spec_.table->name);
    const auto segments = spec_.rejectsAllRows
        ? std::vector<int32_t>{}
        : spec_.pruneSegments(encodedSplit->segmentBegin, encodedSplit->segmentEnd);
    for (auto segment : segments) {
      pendingSegments_.push_back(segment);
      pendingRows_ += spec_.table->segmentSize(segment);
    }
    auto lockedStats = stats_.wlock();
    ++lockedStats->numSplits;
    lockedStats->addRuntimeStat(
        "encodedSegmentsScanned", RuntimeCounter(segments.size()));
    lockedStats->addRuntimeStat(
        "encodedSegmentsPruned",
        RuntimeCounter(
            encodedSplit->segmentEnd - encodedSplit->segmentBegin -
            segments.size()));
  }
  return true;
}

bool CudfEncodedScan::takeBatch(Batch& batch) {
  const auto decodeRows = decodeRowsTarget();
  if (!fillPending(decodeRows)) {
    return false;
  }
  if (pendingSegments_.empty()) {
    return true;
  }
  const auto& table = *spec_.table;
  int64_t rows = 0;
  for (;;) {
    const auto segment = pendingSegments_.front();
    pendingSegments_.pop_front();
    batch.segments.push_back(segment);
    rows += table.segmentSize(segment);
    if (pendingSegments_.empty() ||
        table.segmentSize(segment) < table.segmentRows ||
        pendingSegments_.front() <= segment ||
        rows + table.segmentSize(pendingSegments_.front()) > decodeRows) {
      break;
    }
  }
  pendingRows_ -= rows;
  if (!table.hostResident) {
    return true;
  }
  uint64_t timeUs{0};
  {
    MicrosecondWallTimer timer(&timeUs);
    auto stream = cudfGlobalStreamPool().get_stream();
    batch.table = stageSegments(
        table, stagedColumnNames_, batch.segments, stream, get_temp_mr());
    cudaEvent_t event;
    VELOX_CHECK(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming) ==
            cudaSuccess,
        "Creating a staging event failed");
    batch.staged.reset(event);
    VELOX_CHECK(
        cudaEventRecord(event, stream.get()) == cudaSuccess,
        "Recording a staging event failed");
  }
  int64_t stagedBytes = 0;
  for (const auto& column : batch.table->columns) {
    stagedBytes += column->encodedBytes();
  }
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "encodedStagedBytes",
      RuntimeCounter(stagedBytes, RuntimeCounter::Unit::kBytes));
  lockedStats->addRuntimeStat(
      "encodedStageIssueTime",
      RuntimeCounter(timeUs * 1000, RuntimeCounter::Unit::kNanos));
  return true;
}

RowVectorPtr CudfEncodedScan::doGetOutput() {
  blockingReason_ = exec::BlockingReason::kNotBlocked;
  for (;;) {
    Batch batch;
    if (prefetched_) {
      batch = std::move(*prefetched_);
      prefetched_.reset();
    } else if (isFinished()) {
      break;
    } else {
      if (!takeBatch(batch)) {
        return nullptr;
      }
      if (batch.segments.empty()) {
        continue;
      }
    }
    if (spec_.table->hostResident) {
      Batch next;
      if (takeBatch(next) && !next.segments.empty()) {
        prefetched_ = std::move(next);
      }
    }
    auto output = decodeBatch(batch);
    if (pendingSegments_.empty() && !prefetched_ && unfinishedSplits_ > 0) {
      driverCtx_->task->splitsFinished(planNodeId(), *driverCtx_);
      unfinishedSplits_ = 0;
    }
    if (output != nullptr || blockingFuture_.valid()) {
      return output;
    }
  }
  if (unfinishedSplits_ > 0) {
    driverCtx_->task->splitsFinished(planNodeId(), *driverCtx_);
    unfinishedSplits_ = 0;
  }
  return nullptr;
}

RowVectorPtr CudfEncodedScan::decodeBatch(const Batch& batch) {
  uint64_t timeUs{0};
  std::optional<MicrosecondWallTimer> timer(std::in_place, &timeUs);
  auto stream = cudfGlobalStreamPool().get_stream();
  auto outputMr = get_output_mr();
  auto tempMr = get_temp_mr();

  // A staged batch holds its segments as segments 0 to n-1 of 'table'.
  const auto& table = batch.table ? *batch.table : *spec_.table;
  std::vector<int32_t> segments = batch.segments;
  if (batch.table) {
    std::iota(segments.begin(), segments.end(), 0);
    VELOX_CHECK(
        cudaStreamWaitEvent(stream.get(), batch.staged.get(), 0) ==
            cudaSuccess,
        "Waiting for staged segments failed");
  }
  rmm::device_uvector<int32_t> deviceSegments(segments.size(), stream, tempMr);
  VELOX_CHECK(
      cudaMemcpyAsync(
          deviceSegments.data(),
          segments.data(),
          segments.size() * sizeof(int32_t),
          cudaMemcpyHostToDevice,
          stream.get()) == cudaSuccess,
      "Copying segment ids to the device failed");
  int64_t numRows = 0;
  for (auto segment : batch.segments) {
    numRows += spec_.table->segmentSize(segment);
  }
  const int64_t inputRows = numRows;

  // Filter columns after filtering, by name, and the surviving rows as offsets
  // into the concatenated segments.
  std::unordered_map<std::string, std::unique_ptr<cudf::column>> filtered;
  std::unique_ptr<cudf::column> rows;
  if (!spec_.filterColumnNames.empty()) {
    std::vector<std::unique_ptr<cudf::column>> decoded;
    std::vector<cudf::column_view> views;
    for (const auto& name : spec_.filterColumnNames) {
      decoded.push_back(decodeColumn(
          table,
          table.column(name),
          segments,
          deviceSegments,
          std::nullopt,
          stream,
          tempMr));
      views.push_back(decoded.back()->view());
    }
    std::unique_ptr<cudf::column> mask;
    if (subfieldAst_ != nullptr) {
      mask = cudf::compute_column(
          cudf::table_view(views), *subfieldAst_, stream, tempMr);
    }
    if (remainingFilterExpression_) {
      auto result = remainingFilterExpression_->eval(views, stream, tempMr);
      if (mask) {
        mask = cudf::binary_operation(
            mask->view(),
            asView(result),
            cudf::binary_operator::NULL_LOGICAL_AND,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            tempMr);
      } else if (std::holds_alternative<std::unique_ptr<cudf::column>>(
                     result)) {
        mask = std::move(std::get<std::unique_ptr<cudf::column>>(result));
      } else {
        mask = std::make_unique<cudf::column>(
            std::get<cudf::column_view>(result), stream, tempMr);
      }
    }
    auto sequence = cudf::sequence(
        static_cast<cudf::size_type>(numRows),
        cudf::numeric_scalar<int32_t>(0, true, stream, tempMr),
        cudf::numeric_scalar<int32_t>(1, true, stream, tempMr),
        stream,
        tempMr);
    std::vector<cudf::column_view> withRows{sequence->view()};
    withRows.insert(withRows.end(), views.begin(), views.end());
    auto kept = cudf::apply_retention_mask(
                    cudf::table_view(withRows), mask->view(), stream, outputMr)
                    ->release();
    rows = std::move(kept[0]);
    for (size_t i = 0; i < spec_.filterColumnNames.size(); ++i) {
      filtered.emplace(spec_.filterColumnNames[i], std::move(kept[i + 1]));
    }
    numRows = rows->size();
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  std::unordered_map<std::string, int32_t> remainingUses;
  for (const auto& name : spec_.outputColumnNames) {
    ++remainingUses[name];
  }
  for (const auto& name : spec_.outputColumnNames) {
    const bool last = --remainingUses[name] == 0;
    if (auto it = filtered.find(name); it != filtered.end()) {
      columns.push_back(
          last ? std::move(it->second)
               : std::make_unique<cudf::column>(
                     it->second->view(), stream, outputMr));
      continue;
    }
    columns.push_back(decodeColumn(
        table,
        table.column(name),
        segments,
        deviceSegments,
        rows ? std::optional<cudf::column_view>(rows->view()) : std::nullopt,
        stream,
        outputMr));
  }
  stream.sync();
  timer.reset();

  decodedRows_ += inputRows;
  passedRows_ += numRows;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->rawInputPositions += inputRows;
    lockedStats->addRuntimeStat("encodedBatches", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "encodedBatchSegments", RuntimeCounter(batch.segments.size()));
    if (!spec_.filterColumnNames.empty()) {
      lockedStats->addRuntimeStat(
          "encodedRowsDecodedForFilter", RuntimeCounter(inputRows));
    }
    lockedStats->addRuntimeStat("encodedRowsPassed", RuntimeCounter(numRows));
    lockedStats->addRuntimeStat(
        "encodedDecodeFilterTime",
        RuntimeCounter(timeUs * 1000, RuntimeCounter::Unit::kNanos));
  }
  if (numRows == 0) {
    return nullptr;
  }
  auto output = std::make_shared<CudfVector>(
      pool(),
      outputType_,
      static_cast<vector_size_t>(numRows),
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
  outputRows_ += numRows;
  outputBytes_ += output->estimateFlatSize();
  return output;
}

} // namespace facebook::velox::cudf_velox::connector::encoded
