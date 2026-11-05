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
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

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

/// Where scans of the encoded tables run.
enum class EncodedDevice {
  /// Device-resident tables, decoded and filtered by cuDF; batches are
  /// CudfVectors.
  kGpu,
  /// Host-resident tables, decoded and filtered on the CPU; batches are Velox
  /// vectors, with dictionary strings as DictionaryVectors.
  kCpu,
};

/// Scans tables of the EncodedColumnCache. Takes Hive table and column
/// handles, so plans built for the Hive connector run unchanged.
class EncodedConnector final : public velox::connector::Connector {
 public:
  EncodedConnector(
      const std::string& id,
      EncodedDevice device,
      std::shared_ptr<const config::ConfigBase> config = nullptr)
      : Connector(id, std::move(config)), device_(device) {}

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

  EncodedDevice device() const {
    return device_;
  }

 private:
  const EncodedDevice device_;
};

/// The columns and filters of one scan, shared by the GPU and CPU data
/// sources.
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

/// Per split: prunes segments by min/max against the pushed-down filters,
/// decodes the filter columns and evaluates the filters on the GPU, then
/// decodes only the surviving rows of the other output columns.
class EncodedDataSource : public velox::connector::DataSource,
                          public NvtxHelper {
 public:
  EncodedDataSource(
      const RowTypePtr& outputType,
      const velox::connector::ConnectorTableHandlePtr& tableHandle,
      const velox::connector::ColumnHandleMap& columnHandles,
      velox::connector::ConnectorQueryCtx* connectorQueryCtx);

  void addSplit(std::shared_ptr<velox::connector::ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, ContinueFuture& future)
      override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI("Dynamic filters are not supported by the encoded connector");
  }

  const common::SubfieldFilters* getFilters() const override {
    return &spec_.subfieldFilters;
  }

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 private:
  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  EncodedScanSpec spec_;

  cudf::ast::tree subfieldTree_;
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  const cudf::ast::expression* subfieldAst_{nullptr};
  std::shared_ptr<CudfExpression> remainingFilterExpression_;

  std::shared_ptr<EncodedConnectorSplit> split_;

  uint64_t completedRows_{0};
  uint64_t segmentsScanned_{0};
  uint64_t segmentsPruned_{0};
  uint64_t rowsDecodedForFilter_{0};
  uint64_t rowsPassed_{0};
  uint64_t decodeFilterTimeNs_{0};
};

/// The CPU counterpart of EncodedDataSource over the host copies of the
/// tables. Per batch of rows: applies each pushed-down filter to the rows that
/// passed the previous ones, decoding one value at a time (filters on
/// dictionary strings test the code against a per-code result computed once),
/// then evaluates the remaining filter, then decodes only the surviving rows
/// of the other output columns.
class EncodedCpuDataSource : public velox::connector::DataSource {
 public:
  EncodedCpuDataSource(
      const RowTypePtr& outputType,
      const velox::connector::ConnectorTableHandlePtr& tableHandle,
      const velox::connector::ColumnHandleMap& columnHandles,
      velox::connector::ConnectorQueryCtx* connectorQueryCtx);

  void addSplit(std::shared_ptr<velox::connector::ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, ContinueFuture& future)
      override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI("Dynamic filters are not supported by the encoded connector");
  }

  const common::SubfieldFilters* getFilters() const override {
    return &spec_.subfieldFilters;
  }

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 private:
  struct ColumnFilter {
    const EncodedColumn* column;
    const common::Filter* filter;
    // For kDictionary columns: whether each code passes the filter.
    std::vector<bool> codePasses;
  };

  // Decodes 'column' at 'rows' of 'segment' into a vector of 'type'.
  VectorPtr decode(
      const EncodedColumn& column,
      const TypePtr& type,
      int32_t segment,
      const std::vector<int32_t>& rows);

  // Returns the dictionary values of a kDictionary column as a flat vector,
  // built on first use.
  const VectorPtr& dictionaryValues(const EncodedColumn& column);

  // Returns the next non-empty batch of the current split, or null.
  RowVectorPtr nextBatch(uint64_t size);

  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  core::ExpressionEvaluator* const expressionEvaluator_;
  EncodedScanSpec spec_;

  std::vector<ColumnFilter> columnFilters_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  exec::FilterEvalCtx filterEvalCtx_;
  std::unordered_map<const EncodedColumn*, VectorPtr> dictionaryValues_;
  // Keeps the host string data of the tables referenced without copying.
  std::unordered_map<const EncodedColumn*, BufferPtr> charBuffers_;

  std::shared_ptr<EncodedConnectorSplit> split_;
  std::vector<int32_t> segments_;
  size_t segmentIndex_{0};
  int32_t rowInSegment_{0};

  uint64_t completedRows_{0};
  uint64_t segmentsScanned_{0};
  uint64_t segmentsPruned_{0};
  uint64_t rowsDecodedForFilter_{0};
  uint64_t rowsPassed_{0};
};

/// Registers a TableScan adapter ahead of the built-in one that keeps scans of
/// a GPU EncodedConnector on the GPU and defers other scans to the next
/// adapter. Call after registerCudf().
void registerEncodedTableScanAdapter();

} // namespace facebook::velox::cudf_velox::connector::encoded
