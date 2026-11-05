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

#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/time/Timer.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileConnectorUtil.h"
#include "velox/core/Expressions.h"
#include "velox/exec/TableScan.h"
#include "velox/expression/ExprOptimizer.h"

#include <cudf/binaryop.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include "velox/experimental/cudf/CudfNoDefaults.h"

#include <unordered_set>

namespace facebook::velox::cudf_velox::connector::encoded {

using namespace facebook::velox::connector;

namespace {

bool isTrueConstant(
    const core::ConstantTypedExpr& constant,
    memory::MemoryPool* pool) {
  return !constant.isNull() &&
      constant.toConstantVector(pool)->as<ConstantVector<bool>>()->valueAt(0);
}

bool prunesOnInt64Range(const common::Filter& filter) {
  switch (filter.kind()) {
    case common::FilterKind::kBigintRange:
    case common::FilterKind::kBigintValuesUsingHashTable:
    case common::FilterKind::kBigintValuesUsingBitmask:
    case common::FilterKind::kBigintMultiRange:
      return true;
    default:
      return false;
  }
}

} // namespace

std::string EncodedConnectorSplit::toString() const {
  return fmt::format(
      "[encoded split: {} segments {}-{}]", table, segmentBegin, segmentEnd);
}

std::vector<std::shared_ptr<ConnectorSplit>> makeEncodedSplits(
    const std::string& connectorId,
    const std::string& table,
    int32_t segmentsPerSplit) {
  const auto& encoded = EncodedColumnCache::instance().table(table);
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  for (int32_t begin = 0; begin < encoded.numSegments;
       begin += segmentsPerSplit) {
    splits.push_back(std::make_shared<EncodedConnectorSplit>(
        connectorId,
        table,
        begin,
        std::min(begin + segmentsPerSplit, encoded.numSegments)));
  }
  return splits;
}

std::unique_ptr<DataSource> EncodedConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  if (device_ == EncodedDevice::kCpu) {
    return std::make_unique<EncodedCpuDataSource>(
        outputType, tableHandle, columnHandles, connectorQueryCtx);
  }
  return std::make_unique<EncodedDataSource>(
      outputType, tableHandle, columnHandles, connectorQueryCtx);
}

std::unique_ptr<DataSink> EncodedConnector::createDataSink(
    RowTypePtr /*inputType*/,
    ConnectorInsertTableHandlePtr /*insertTableHandle*/,
    ConnectorQueryCtx* /*connectorQueryCtx*/,
    CommitStrategy /*commitStrategy*/) {
  VELOX_NYI("The encoded connector is read-only");
}

EncodedScanSpec::EncodedScanSpec(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& handle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  tableHandle =
      std::dynamic_pointer_cast<const velox::connector::hive::HiveTableHandle>(
          handle);
  VELOX_CHECK_NOT_NULL(
      tableHandle, "The encoded connector takes a HiveTableHandle");
  table = &EncodedColumnCache::instance().table(tableHandle->tableName());
  auto* pool = connectorQueryCtx->memoryPool();

  for (uint32_t i = 0; i < outputType->size(); ++i) {
    const auto it = columnHandles.find(outputType->nameOf(i));
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputType->nameOf(i));
    const auto& name =
        checkedPointerCast<const velox::connector::FileColumnHandle>(it->second)
            ->name();
    outputColumnNames.push_back(name);
    columnTypes.emplace(name, outputType->childAt(i));
  }

  for (const auto& [subfield, filter] : tableHandle->subfieldFilters()) {
    subfieldFilters.emplace(subfield.clone(), filter->clone());
  }
  double sampleRate = tableHandle->sampleRate();
  auto extracted = velox::connector::extractFiltersFromRemainingFilter(
      tableHandle->remainingFilter(),
      connectorQueryCtx->expressionEvaluator(),
      subfieldFilters,
      sampleRate);
  VELOX_CHECK_EQ(sampleRate, 1.0, "Sampling is not supported");

  std::unordered_map<std::string, std::string> sessionSettings{
      {core::QueryConfig::kAdjustTimestampToTimezone,
       connectorQueryCtx->adjustTimestampToTimezone() ? "true" : "false"}};
  if (!connectorQueryCtx->sessionTimezone().empty()) {
    sessionSettings.emplace(
        core::QueryConfig::kSessionTimezone,
        connectorQueryCtx->sessionTimezone());
  }
  optimizeQueryCtx = core::QueryCtx::create(
      nullptr, core::QueryConfig{std::move(sessionSettings)});
  remainingFilter = extracted
      ? expression::optimize(extracted, optimizeQueryCtx.get(), pool)
      : nullptr;
  if (const auto constant =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
              remainingFilter)) {
    rejectsAllRows = !isTrueConstant(*constant, pool);
    remainingFilter = nullptr;
  }

  std::unordered_set<std::string> filterColumnSet;
  auto addFilterColumn = [&](const std::string& name) {
    if (filterColumnSet.insert(name).second) {
      filterColumnNames.push_back(name);
    }
  };
  for (const auto& [subfield, _] : subfieldFilters) {
    VELOX_CHECK_EQ(
        subfield.path().size(), 1, "Only top-level column filters are supported");
    addFilterColumn(subfield.baseName());
  }
  if (remainingFilter) {
    for (const auto& name : referencedInputFields(remainingFilter)) {
      addFilterColumn(name);
    }
  }
  std::vector<TypePtr> filterTypes;
  for (const auto& name : filterColumnNames) {
    filterTypes.push_back(columnType(name));
  }
  filterType =
      ROW(std::vector<std::string>(filterColumnNames), std::move(filterTypes));
}

const TypePtr& EncodedScanSpec::columnType(const std::string& name) const {
  if (const auto it = columnTypes.find(name); it != columnTypes.end()) {
    return it->second;
  }
  const auto& dataColumns = tableHandle->dataColumns();
  VELOX_CHECK_NOT_NULL(
      dataColumns,
      "Table schema is required to resolve the type of filter-only column {}",
      name);
  return dataColumns->findChild(name);
}

std::vector<int32_t> EncodedScanSpec::pruneSegments(
    int32_t begin,
    int32_t end) const {
  std::vector<std::pair<const EncodedColumn*, const common::Filter*>> checks;
  for (const auto& [subfield, filter] : subfieldFilters) {
    const auto& column = table->column(subfield.baseName());
    if (column.codec == Codec::kForBitpack && prunesOnInt64Range(*filter)) {
      checks.emplace_back(&column, filter.get());
    }
  }
  std::vector<int32_t> segments;
  for (int32_t s = begin; s < end; ++s) {
    bool keep = true;
    for (const auto& [column, filter] : checks) {
      if (!filter->testInt64Range(
              column->segmentMin[s], column->segmentMax[s], false)) {
        keep = false;
        break;
      }
    }
    if (keep) {
      segments.push_back(s);
    }
  }
  return segments;
}

EncodedDataSource::EncodedDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx)
    : NvtxHelper(
          nvtx3::rgb{118, 185, 0},
          std::nullopt,
          fmt::format("[encoded {}]", tableHandle->name())),
      outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      spec_(outputType, tableHandle, columnHandles, connectorQueryCtx) {
  if (!spec_.subfieldFilters.empty()) {
    subfieldAst_ = &createAstFromSubfieldFilters(
        spec_.subfieldFilters, subfieldTree_, subfieldScalars_, spec_.filterType);
  }
  if (spec_.remainingFilter) {
    remainingFilterExpression_ = createCudfExpression(
        spec_.remainingFilter,
        spec_.filterType,
        pool_,
        spec_.optimizeQueryCtx->queryConfig());
  }
}

void EncodedDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  split_ = std::dynamic_pointer_cast<EncodedConnectorSplit>(split);
  VELOX_CHECK_NOT_NULL(split_, "Expected an EncodedConnectorSplit");
  VELOX_CHECK_EQ(split_->table, spec_.table->name);
}

std::optional<RowVectorPtr> EncodedDataSource::next(
    uint64_t /*size*/,
    ContinueFuture& /*future*/) {
  // Each split is returned as one batch; the next call ends it.
  if (split_ == nullptr) {
    return nullptr;
  }
  const auto split = std::move(split_);
  if (spec_.rejectsAllRows) {
    return nullptr;
  }
  ensureCudaContextForThread();
  const auto segments = spec_.pruneSegments(split->segmentBegin, split->segmentEnd);
  const auto splitSegments = split->segmentEnd - split->segmentBegin;
  segmentsScanned_ += segments.size();
  segmentsPruned_ += splitSegments - segments.size();
  if (segments.empty()) {
    return nullptr;
  }

  uint64_t timeUs{0};
  std::optional<MicrosecondWallTimer> timer(std::in_place, &timeUs);
  auto stream = cudfGlobalStreamPool().get_stream();
  auto outputMr = get_output_mr();
  auto tempMr = get_temp_mr();

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
  for (auto segment : segments) {
    numRows += spec_.table->segmentSize(segment);
  }

  // Filter columns after filtering, by name, and the surviving rows as offsets
  // into the concatenated segments.
  std::unordered_map<std::string, std::unique_ptr<cudf::column>> filtered;
  std::unique_ptr<cudf::column> rows;
  if (!spec_.filterColumnNames.empty()) {
    rowsDecodedForFilter_ += numRows;
    std::vector<std::unique_ptr<cudf::column>> decoded;
    std::vector<cudf::column_view> views;
    for (const auto& name : spec_.filterColumnNames) {
      decoded.push_back(decodeColumn(
          *spec_.table,
          spec_.table->column(name),
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
    rowsPassed_ += numRows;
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
        *spec_.table,
        spec_.table->column(name),
        segments,
        deviceSegments,
        rows ? std::optional<cudf::column_view>(rows->view()) : std::nullopt,
        stream,
        outputMr));
  }
  auto output = std::make_shared<CudfVector>(
      pool_,
      outputType_,
      static_cast<vector_size_t>(numRows),
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
  stream.sync();
  timer.reset();
  decodeFilterTimeNs_ += timeUs * 1000;
  completedRows_ += numRows;
  return output;
}

std::unordered_map<std::string, RuntimeMetric>
EncodedDataSource::getRuntimeStats() {
  return {
      {"encodedSegmentsScanned",
       RuntimeMetric(static_cast<int64_t>(segmentsScanned_))},
      {"encodedSegmentsPruned",
       RuntimeMetric(static_cast<int64_t>(segmentsPruned_))},
      {"encodedRowsDecodedForFilter",
       RuntimeMetric(static_cast<int64_t>(rowsDecodedForFilter_))},
      {"encodedRowsPassed", RuntimeMetric(static_cast<int64_t>(rowsPassed_))},
      {"encodedDecodeFilterTime",
       RuntimeMetric(
           static_cast<int64_t>(decodeFilterTimeNs_),
           RuntimeCounter::Unit::kNanos)},
  };
}

namespace {

class EncodedTableScanAdapter : public OperatorAdapter {
 public:
  EncodedTableScanAdapter() : OperatorAdapter("EncodedTableScan") {}

  bool canHandle(const exec::Operator* op) const override {
    return op->is<exec::TableScan>();
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    const auto scanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    if (scanNode != nullptr) {
      const auto connector = std::dynamic_pointer_cast<EncodedConnector>(
          ConnectorRegistry::tryGet(scanNode->tableHandle()->connectorId()));
      if (connector != nullptr) {
        return connector->device() == EncodedDevice::kGpu;
      }
    }
    const auto* next = nextAdapter(op);
    return next != nullptr && next->canRunOnGPU(op, planNode, ctx);
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

  bool keepOperator() const override {
    return true;
  }

 private:
  const OperatorAdapter* nextAdapter(const exec::Operator* op) const {
    bool seenSelf = false;
    for (const auto& adapter :
         OperatorAdapterRegistry::getInstance().getAdapters()) {
      if (adapter.get() == this) {
        seenSelf = true;
      } else if (seenSelf && adapter->canHandle(op)) {
        return adapter.get();
      }
    }
    return nullptr;
  }
};

} // namespace

void registerEncodedTableScanAdapter() {
  OperatorAdapterRegistry::getInstance().registerAdapterFront(
      std::make_unique<EncodedTableScanAdapter>());
}

} // namespace facebook::velox::cudf_velox::connector::encoded
