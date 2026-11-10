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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/connectors/hive/HiveDataSource.h"

#include <mutex>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

CudfHiveConnector::CudfHiveConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* executor)
    : ::facebook::velox::connector::hive::HiveConnector(id, config, executor),
      cudfHiveConfig_(std::make_shared<CudfHiveConfig>(config)) {
  VLOG(1) << "cuDF Hive connector created";
}

namespace {

std::mutex temporaryTableStorageMutex;
std::shared_ptr<TemporaryTableStorage> temporaryTableStorageInstance;

std::shared_ptr<TemporaryTableStorage> temporaryTableStorage() {
  std::lock_guard<std::mutex> lock(temporaryTableStorageMutex);
  return temporaryTableStorageInstance;
}

} // namespace

void registerTemporaryTableStorage(
    std::shared_ptr<TemporaryTableStorage> storage) {
  std::lock_guard<std::mutex> lock(temporaryTableStorageMutex);
  temporaryTableStorageInstance = std::move(storage);
}

bool isTemporaryTable(std::string_view name) {
  // The prefix of every temporary table name, which is also the name of its
  // directory (HiveMetadata.createTemporaryTable in Presto).
  return name.find("__presto_temporary_table_") != std::string_view::npos;
}

std::unique_ptr<DataSink> CudfHiveConnector::createDataSink(
    RowTypePtr inputType,
    ConnectorInsertTableHandlePtr connectorInsertTableHandle,
    ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy) {
  if (auto storage = temporaryTableStorage()) {
    auto insertTableHandle = std::dynamic_pointer_cast<
        const ::facebook::velox::connector::hive::HiveInsertTableHandle>(
        connectorInsertTableHandle);
    if (insertTableHandle != nullptr &&
        isTemporaryTable(insertTableHandle->locationHandle()->targetPath())) {
      return storage->createDataSink(
          std::move(inputType),
          std::move(insertTableHandle),
          connectorQueryCtx,
          commitStrategy,
          hiveConfig_);
    }
  }
  return HiveConnector::createDataSink(
      std::move(inputType),
      std::move(connectorInsertTableHandle),
      connectorQueryCtx,
      commitStrategy);
}

std::unique_ptr<DataSource> CudfHiveConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  if (auto storage = temporaryTableStorage();
      storage != nullptr && isTemporaryTable(tableHandle->name())) {
    return storage->createDataSource(
        outputType, tableHandle, columnHandles, connectorQueryCtx);
  }

  // If it's parquet then return CudfHiveDataSource
  // If it's not parquet then return HiveDataSource
  // TODO (dm): Make this ^^^ happen
  // Problem: this information is in split, not table handle

  if (cudfIsRegistered() && isTypeSupportedByCudf(outputType)) {
    return std::make_unique<CudfHiveDataSource>(
        outputType,
        tableHandle,
        columnHandles,
        &fileHandleFactory_,
        ioExecutor_,
        connectorQueryCtx,
        cudfHiveConfig_);
  }

  return std::make_unique<::facebook::velox::connector::hive::HiveDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      &fileHandleFactory_,
      ioExecutor_,
      connectorQueryCtx,
      hiveConfig_);
}

std::shared_ptr<Connector> CudfHiveConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* cpuExecutor) {
  return std::make_shared<CudfHiveConnector>(id, config, ioExecutor);
}

} // namespace facebook::velox::cudf_velox::connector::hive
