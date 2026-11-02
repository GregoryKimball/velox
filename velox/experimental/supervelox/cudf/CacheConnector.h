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

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <cudf/column/column.hpp>
#include <cudf/join/filtered_join.hpp>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/experimental/cudf/connectors/GpuConnector.h"
#include "velox/experimental/cudf/exec/FusedAggregation.h"
#include "velox/experimental/supervelox/store/ColumnStore.h"
#include "velox/experimental/supervelox/store/DeviceCache.h"

namespace facebook::velox::supervelox::cudf_backend {

/// A scan from the cache that reads columns in their stored encodings and
/// makes its output in generated kernels: one finding the rows that pass
/// 'filters' and 'columnFilters', and one computing the outputs that are
/// expressions from the rows that pass, decoding what they read. The
/// subfield filters of the handle only skip zones.
class FusedScanHandle : public connector::hive::HiveTableHandle {
 public:
  enum class Source {
    /// An input column as stored, for a consumer that decodes it.
    kStored,
    /// A string input column, copied, or looked up in its dictionary.
    kString,
    /// An expression over the input columns.
    kExpression,
  };

  struct Output {
    Source source;
    /// The input column for kStored and kString, else an expression.
    std::string column;
    core::TypedExprPtr expression;
  };

  /// 'input' names the stored columns the scan reads, typed as stored:
  /// integers of their stored width, codes for dictionary columns, or
  /// strings. 'outputs' follow the scan's output type.
  FusedScanHandle(
      const connector::hive::HiveTableHandle& scan,
      common::SubfieldFilters zoneFilters,
      RowTypePtr input,
      cudf_velox::Encodings encodings,
      std::vector<core::TypedExprPtr> filters,
      std::vector<cudf_velox::ColumnFilter> columnFilters,
      std::vector<Output> outputs);

  const RowTypePtr& input() const {
    return input_;
  }

  const cudf_velox::Encodings& encodings() const {
    return encodings_;
  }

  const std::vector<core::TypedExprPtr>& filters() const {
    return filters_;
  }

  const std::vector<cudf_velox::ColumnFilter>& columnFilters() const {
    return columnFilters_;
  }

  const std::vector<Output>& outputs() const {
    return outputs_;
  }

  /// Rows pass only if input column 'column' is one of the values of column
  /// 'key' of the rows 'keys', another scan from the cache, returns: the
  /// keys a join can match, as a dynamic filter would pass them.
  struct KeyFilter {
    std::string column;
    std::shared_ptr<const core::TableScanNode> keys;
    std::string key;
  };

  const std::vector<KeyFilter>& keyFilters() const {
    return keyFilters_;
  }

  /// This scan with 'filter' added.
  std::shared_ptr<const FusedScanHandle> withKeyFilter(KeyFilter filter) const;

  /// The expressions of the kExpression outputs, in order.
  std::vector<core::TypedExprPtr> expressions() const;

  std::string toString() const override;

 private:
  const RowTypePtr input_;
  const cudf_velox::Encodings encodings_;
  const std::vector<core::TypedExprPtr> filters_;
  const std::vector<cudf_velox::ColumnFilter> columnFilters_;
  const std::vector<Output> outputs_;
  std::vector<KeyFilter> keyFilters_;
};

/// Rows [begin, end) of a table in the column store.
struct CacheSplit : public connector::ConnectorSplit {
  CacheSplit(const std::string& connectorId, std::string table, int64_t begin, int64_t end)
      : ConnectorSplit(connectorId), table(std::move(table)), begin(begin), end(end) {}

  std::string toString() const override;

  const std::string table;
  const int64_t begin;
  const int64_t end;
};

/// Serves table scans from the HBM cache to the unchanged Velox-cuDF
/// operators. Each split becomes one CudfVector whose columns are decoded
/// from the store's encodings (narrow integers, unscaled decimals, day
/// counts, dictionary codes) into the types the scan asks for, one cuDF call
/// per column, or for a FusedScanHandle in its generated kernels. Takes the
/// table and column names from the Hive handles that TPC-H plans carry, and
/// accepts no dynamic filters.
class CacheConnector : public connector::Connector, public cudf_velox::connector::GpuConnector {
 public:
  CacheConnector(const std::string& id, ColumnStore& store, DeviceCache& cache, int64_t batchRows);

  std::unique_ptr<connector::DataSource> createDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      connector::ConnectorQueryCtx* connectorQueryCtx) override;

  std::unique_ptr<connector::DataSink> createDataSink(
      RowTypePtr /*inputType*/,
      connector::ConnectorInsertTableHandlePtr /*insertTableHandle*/,
      connector::ConnectorQueryCtx* /*connectorQueryCtx*/,
      connector::CommitStrategy /*commitStrategy*/) override {
    VELOX_UNSUPPORTED("The cache connector is read only");
  }

  /// Splits of about batchRows rows covering 'table'. With 'keys', no split
  /// boundary falls between two rows with the same values of 'keys'.
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits(
      const std::string& table,
      const std::vector<std::string>& keys = {});

  /// Whether the rows with equal values of an integer column are adjacent.
  /// Checked on first use.
  bool clustered(const std::string& table, const std::string& column);

  /// Device-readable values, or characters of a string column.
  const void* data(const Column& column, cudaStream_t stream);
  const int64_t* offsets(const Column& column, cudaStream_t stream);

  /// The dictionary of 'column' as a device strings column.
  const cudf::column& dictionary(const Column& column);

  /// {min, max} of each kZoneRows rows of an integer column, computed on
  /// first use like the statistics a file writer would keep.
  const std::vector<std::pair<int64_t, int64_t>>& zones(const Column& column);

  static constexpr int64_t kZoneRows = 8192;

  /// A table parameter that makes a scan use its filters only to skip zones,
  /// as a reader would skip row groups, leaving the rows of the remaining
  /// zones for a consumer that applies the filters itself.
  static constexpr const char* kZoneOnly = "supervelox.zone_only";

  /// A scan of a table with fewer rows applies its key filters only when
  /// getting keys for another scan's: reading all of it costs less than
  /// getting the keys.
  static constexpr int64_t kMinKeyFilterRows = 1 << 24;

  ColumnStore& store() {
    return store_;
  }

  /// The type 'column' is stored as: the integer of its width, or VARCHAR.
  static TypePtr storedVeloxType(const Column& column);

  /// The values of a key filter's key column over all splits of its scan,
  /// and a hash set of them to probe.
  struct JoinKeys {
    std::unique_ptr<cudf::column> values;
    std::unique_ptr<cudf::filtered_join> set;
  };

  /// The keys of 'filter' as 'type', computed by the first of the drivers of
  /// query 'queryId' to ask. Only the latest query's are kept.
  std::shared_ptr<const JoinKeys> joinKeys(
      const std::string& queryId,
      const FusedScanHandle::KeyFilter& filter,
      cudf::data_type type,
      connector::ConnectorQueryCtx* connectorQueryCtx);

 private:
  ColumnStore& store_;
  DeviceCache& cache_;
  const int64_t batchRows_;
  /// DeviceCache is not thread safe and drivers scan concurrently.
  std::mutex mutex_;
  std::map<std::string, std::unique_ptr<cudf::column>> dictionaries_;
  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> zones_;
  std::map<std::string, bool> clustered_;
  /// Held while computing join keys, which takes 'mutex_', and the keys of
  /// the key filters of the scan that returns them.
  std::recursive_mutex keysMutex_;
  std::string keysQuery_;
  std::map<std::string, std::shared_ptr<const JoinKeys>> joinKeys_;
};

} // namespace facebook::velox::supervelox::cudf_backend
