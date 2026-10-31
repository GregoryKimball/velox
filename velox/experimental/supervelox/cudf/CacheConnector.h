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

#include "velox/connectors/Connector.h"
#include "velox/experimental/cudf/connectors/GpuConnector.h"
#include "velox/experimental/supervelox/store/ColumnStore.h"
#include "velox/experimental/supervelox/store/DeviceCache.h"

namespace facebook::velox::supervelox::cudf_backend {

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
/// per column. Takes the table and column names from the Hive handles that
/// TPC-H plans carry, and accepts no pushed-down filters.
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

  ColumnStore& store() {
    return store_;
  }

 private:
  ColumnStore& store_;
  DeviceCache& cache_;
  const int64_t batchRows_;
  /// DeviceCache is not thread safe and drivers scan concurrently.
  std::mutex mutex_;
  std::map<std::string, std::unique_ptr<cudf::column>> dictionaries_;
  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> zones_;
  std::map<std::string, bool> clustered_;
};

} // namespace facebook::velox::supervelox::cudf_backend
