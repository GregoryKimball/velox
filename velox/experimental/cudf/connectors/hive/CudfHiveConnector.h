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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveDataSink.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::config;

/// Writes and reads Presto's temporary tables, the bucketed tables that hold a
/// materialized exchange, somewhere other than files.
class TemporaryTableStorage {
 public:
  virtual ~TemporaryTableStorage() = default;

  virtual std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const ::facebook::velox::connector::hive::
                          HiveInsertTableHandle> insertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy,
      const std::shared_ptr<const ::facebook::velox::connector::hive::
                                HiveConfig>& hiveConfig) = 0;

  virtual std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) = 0;
};

/// Routes the temporary tables of every CudfHiveConnector to 'storage', or
/// back to files when 'storage' is null.
void registerTemporaryTableStorage(
    std::shared_ptr<TemporaryTableStorage> storage);

/// Returns whether 'name', a table name or a path, is a Presto temporary
/// table or one of its files.
bool isTemporaryTable(std::string_view name);

class CudfHiveConnector final
    : public ::facebook::velox::connector::hive::HiveConnector {
 public:
  CudfHiveConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* executor);

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override final;

  bool canAddDynamicFilter() const override {
    return false;
  }

  bool supportsSplitPreload() const override {
    return true;
  }

  /// Writes temporary tables to the registered TemporaryTableStorage, and
  /// everything else as HiveConnector does.
  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      ConnectorInsertTableHandlePtr connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) override;

 protected:
  // TODO (dm): rename parquetconfig
  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
};

class CudfHiveConnectorFactory
    : public ::facebook::velox::connector::hive::HiveConnectorFactory {
 public:
  CudfHiveConnectorFactory()
      : ::facebook::velox::connector::hive::HiveConnectorFactory() {}

  explicit CudfHiveConnectorFactory(const char* connectorName)
      : ::facebook::velox::connector::hive::HiveConnectorFactory(
            connectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override;
};

} // namespace facebook::velox::cudf_velox::connector::hive
