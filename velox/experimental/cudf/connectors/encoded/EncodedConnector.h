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

#include "velox/experimental/cudf/connectors/encoded/EncodedColumnCache.h"

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::cudf_velox::connector::encoded {

/// A contiguous range of segments of one encoded table.
struct EncodedConnectorSplit : public velox::connector::ConnectorSplit {
  EncodedConnectorSplit(
      const std::string& connectorId,
      std::string table,
      int32_t segmentBegin,
      int32_t segmentEnd)
      : ConnectorSplit(connectorId),
        table(std::move(table)),
        segmentBegin(segmentBegin),
        segmentEnd(segmentEnd) {}

  std::string toString() const override;

  const std::string table;
  const int32_t segmentBegin;
  const int32_t segmentEnd;
};

/// Splits covering every segment of 'table', 'segmentsPerSplit' at a time.
std::vector<std::shared_ptr<velox::connector::ConnectorSplit>>
makeEncodedSplits(
    const std::string& connectorId,
    const std::string& table,
    int32_t segmentsPerSplit);

/// Scans tables of the EncodedColumnCache on the GPU: each TableScan over it is
/// replaced by CudfEncodedScan. Takes Hive table and column handles, so plans
/// built for the Hive connector run unchanged.
class EncodedConnector final : public velox::connector::Connector {
 public:
  EncodedConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config = nullptr)
      : Connector(id, std::move(config)) {}

  std::unique_ptr<velox::connector::DataSource> createDataSource(
      const RowTypePtr& outputType,
      const velox::connector::ConnectorTableHandlePtr& tableHandle,
      const velox::connector::ColumnHandleMap& columnHandles,
      velox::connector::ConnectorQueryCtx* connectorQueryCtx) override;

  std::unique_ptr<velox::connector::DataSink> createDataSink(
      RowTypePtr inputType,
      velox::connector::ConnectorInsertTableHandlePtr insertTableHandle,
      velox::connector::ConnectorQueryCtx* connectorQueryCtx,
      velox::connector::CommitStrategy commitStrategy) override;
};

/// The columns and filters of one CudfEncodedScan.
struct EncodedScanSpec {
  EncodedScanSpec(
      const RowTypePtr& outputType,
      const velox::connector::ConnectorTableHandlePtr& tableHandle,
      const velox::connector::ColumnHandleMap& columnHandles,
      velox::connector::ConnectorQueryCtx* connectorQueryCtx);

  const TypePtr& columnType(const std::string& name) const;

  /// Returns the segments in [begin, end) whose min/max the filters may match.
  std::vector<int32_t> pruneSegments(int32_t begin, int32_t end) const;

  std::shared_ptr<const velox::connector::hive::HiveTableHandle> tableHandle;
  const EncodedTable* table{nullptr};

  /// Table column read for each output column.
  std::vector<std::string> outputColumnNames;
  std::unordered_map<std::string, TypePtr> columnTypes;

  /// The table handle's filters plus those extracted from its remaining
  /// filter, all on top-level columns.
  common::SubfieldFilters subfieldFilters;
  /// Carries the session settings the remaining filter is folded against.
  std::shared_ptr<core::QueryCtx> optimizeQueryCtx;
  /// What is left of the remaining filter after extraction and folding.
  core::TypedExprPtr remainingFilter;
  bool rejectsAllRows{false};

  /// Columns the filters read: those of 'subfieldFilters', in order, then
  /// those only 'remainingFilter' reads.
  std::vector<std::string> filterColumnNames;
  RowTypePtr filterType;
};

/// Registers a TableScan adapter ahead of the built-in one that replaces scans
/// of an EncodedConnector with CudfEncodedScan and defers other scans to
/// the next adapter. Call after registerCudf().
void registerEncodedTableScanAdapter();

} // namespace facebook::velox::cudf_velox::connector::encoded
