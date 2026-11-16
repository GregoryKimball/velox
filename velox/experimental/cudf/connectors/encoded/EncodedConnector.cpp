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
#include "velox/experimental/cudf/connectors/encoded/CudfEncodedScan.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileConnectorUtil.h"
#include "velox/core/Expressions.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/Task.h"
#include "velox/expression/ExprOptimizer.h"

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
    const RowTypePtr& /*outputType*/,
    const ConnectorTableHandlePtr& /*tableHandle*/,
    const ColumnHandleMap& /*columnHandles*/,
    ConnectorQueryCtx* /*connectorQueryCtx*/) {
  VELOX_FAIL(
      "Scans of the encoded connector run as CudfEncodedScan; call "
      "registerEncodedTableScanAdapter() after registerCudf()");
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

namespace {

// Whether 'op' is a TableScan over an EncodedConnector.
bool isEncodedScan(const exec::Operator* op) {
  if (!op->is<exec::TableScan>()) {
    return false;
  }
  const auto* scanNode = dynamic_cast<const core::TableScanNode*>(
      core::PlanNode::findNodeById(
          op->operatorCtx()->task()->planFragment().planNode.get(),
          op->planNodeId()));
  if (scanNode == nullptr) {
    return false;
  }
  return std::dynamic_pointer_cast<EncodedConnector>(
             ConnectorRegistry::tryGet(
                 scanNode->tableHandle()->connectorId())) != nullptr;
}

class EncodedTableScanAdapter : public OperatorAdapter {
 public:
  EncodedTableScanAdapter() : OperatorAdapter("EncodedTableScan") {}

  bool canHandle(const exec::Operator* op) const override {
    return isEncodedScan(op);
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto scanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    VELOX_CHECK_NOT_NULL(scanNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfEncodedScan>(operatorId, ctx, scanNode));
    return result;
  }
};

} // namespace

void registerEncodedTableScanAdapter() {
  OperatorAdapterRegistry::getInstance().registerAdapterFront(
      std::make_unique<EncodedTableScanAdapter>());
}

} // namespace facebook::velox::cudf_velox::connector::encoded
