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
#include "velox/experimental/ucx-exchange/ExchangeStoreTables.h"

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/ExchangeStore.h"

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/exec/Task.h"

#include <cudf/concatenate.hpp>

#include <cstring>
#include <numeric>

namespace facebook::velox::ucx_exchange {
namespace {

using connector::hive::HiveColumnHandle;
using connector::hive::HiveConnectorSplit;
using connector::hive::HiveTableHandle;

// The content of every placeholder file. A file the writer leaves empty is not
// reported to the coordinator (FileDataSink::finalizeWriterFile()).
constexpr std::string_view kPlaceholder{"velox exchange store\n"};

// Rows a writer collects from the CPU before packing them as one payload. The
// TableWriter hands each file a slice of every batch, which is a small part of
// the batch when the table has many buckets.
constexpr vector_size_t kPayloadRows{1'000'000};

// Packs what a temporary table writer receives into the exchange store under
// the key of its file. The file itself only gets the placeholder.
class ExchangeStoreWriter : public dwio::common::Writer {
 public:
  ExchangeStoreWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      memory::MemoryPool* writerPool,
      ExchangeCompression compression)
      : sink_(std::move(sink)),
        pool_(writerPool->addLeafChild(writerPool->name() + ".exchange_store")),
        compression_(compression) {
    std::tie(table_, file_) = storeKey(sink_->name());
    setState(State::kRunning);
  }

  void write(const VectorPtr& data) override {
    checkRunning();
    if (data->size() == 0) {
      return;
    }
    if (auto gpuData = std::dynamic_pointer_cast<cudf_velox::CudfVector>(data)) {
      add(gpuData->getTableView(), data->size(), gpuData->stream());
      return;
    }
    pending_.push_back(std::dynamic_pointer_cast<RowVector>(data));
    pendingRows_ += data->size();
    if (pendingRows_ >= kPayloadRows) {
      addPending();
    }
  }

  void flush() override {}

  bool finish() override {
    return true;
  }

  std::unique_ptr<dwio::common::FileMetadata> close() override {
    addPending();
    dwio::common::DataBuffer<char> placeholder(*pool_, kPlaceholder.size());
    std::memcpy(placeholder.data(), kPlaceholder.data(), kPlaceholder.size());
    sink_->write(std::move(placeholder));
    sink_->close();
    setState(State::kClosed);
    return nullptr;
  }

  void abort() override {
    // The query fails, so no file of the table will be read.
    pending_.clear();
    ExchangeStore::instance().drop(table_);
    sink_->close();
    setState(State::kAborted);
  }

 private:
  // Uploads the collected CPU rows and adds them as one payload.
  void addPending() {
    if (pending_.empty()) {
      return;
    }
    const auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    std::vector<std::unique_ptr<cudf::table>> tables;
    std::vector<cudf::table_view> views;
    for (const auto& rows : pending_) {
      tables.push_back(cudf_velox::with_arrow::toCudfTable(
          rows, pool_.get(), stream, cudf_velox::get_temp_mr()));
      views.push_back(tables.back()->view());
    }
    if (views.size() == 1) {
      add(views[0], pendingRows_, stream);
    } else {
      auto table =
          cudf::concatenate(views, stream, cudf_velox::get_temp_mr());
      add(table->view(), pendingRows_, stream);
    }
    pending_.clear();
    pendingRows_ = 0;
  }

  void add(cudf::table_view table, vector_size_t numRows, cuda::stream_ref stream) {
    auto payload = packExchangePayload(
        table,
        compression_,
        stream,
        cudf_velox::get_temp_mr(),
        cudf_velox::get_output_mr());
    stream.sync();
    ExchangeStore::instance().add(
        table_, file_, std::move(payload), numRows, stream);
  }

  const std::unique_ptr<dwio::common::FileSink> sink_;
  const std::shared_ptr<memory::MemoryPool> pool_;
  const ExchangeCompression compression_;
  std::string table_;
  std::string file_;
  // CPU rows not packed yet.
  std::vector<RowVectorPtr> pending_;
  vector_size_t pendingRows_{0};
};

class ExchangeStoreWriterFactory : public dwio::common::WriterFactory {
 public:
  ExchangeStoreWriterFactory(
      dwio::common::FileFormat format,
      ExchangeCompression compression)
      : WriterFactory(format), compression_(compression) {}

  std::unique_ptr<dwio::common::Writer> createWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      const std::shared_ptr<dwio::common::WriterOptions>& options) override {
    VELOX_CHECK_NOT_NULL(options->memoryPool);
    return std::make_unique<ExchangeStoreWriter>(
        std::move(sink), options->memoryPool, compression_);
  }

  std::unique_ptr<dwio::common::WriterOptions> createWriterOptions() override {
    return std::make_unique<dwio::common::WriterOptions>();
  }

 private:
  const ExchangeCompression compression_;
};

// Returns the path of the file a Hive split reads, whichever Hive connector
// made it.
const std::string& filePath(const connector::ConnectorSplit& split) {
  if (auto cudfSplit = dynamic_cast<
          const cudf_velox::connector::hive::CudfHiveConnectorSplit*>(&split)) {
    return cudfSplit->filePath;
  }
  auto hiveSplit = dynamic_cast<const HiveConnectorSplit*>(&split);
  VELOX_CHECK_NOT_NULL(hiveSplit, "Not a Hive split: {}", split.toString());
  return hiveSplit->filePath;
}

// Returns, for each split of a temporary table, the payloads the writer kept
// for its file, with the scan's columns.
class ExchangeStoreDataSource : public connector::DataSource {
 public:
  ExchangeStoreDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      connector::ConnectorQueryCtx* connectorQueryCtx)
      : outputType_(outputType), pool_(connectorQueryCtx->memoryPool()) {
    auto hiveTable =
        std::dynamic_pointer_cast<const HiveTableHandle>(tableHandle);
    VELOX_CHECK_NOT_NULL(hiveTable);
    VELOX_CHECK(
        hiveTable->subfieldFilters().empty() &&
            hiveTable->remainingFilter() == nullptr,
        "Filters on a temporary table are not supported: {}",
        hiveTable->name());
    dataColumns_ = hiveTable->dataColumns();
    VELOX_CHECK_NOT_NULL(dataColumns_);
    for (const auto& name : outputType_->names()) {
      auto column = std::dynamic_pointer_cast<const HiveColumnHandle>(
          columnHandles.at(name));
      VELOX_CHECK_NOT_NULL(column);
      columns_.push_back(
          static_cast<cudf::size_type>(
              dataColumns_->getChildIdx(column->name())));
    }
    std::vector<cudf::size_type> allColumns(dataColumns_->size());
    std::iota(allColumns.begin(), allColumns.end(), 0);
    allColumns_ = columns_ == allColumns;
  }

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    VELOX_CHECK_EQ(next_, payloads_.size(), "The previous split is not done");
    const auto [table, file] = storeKey(filePath(*split));
    stream_ = cudf_velox::cudfGlobalStreamPool().get_stream();
    payloads_ = ExchangeStore::instance().take(
        table, file, stream_, cudf_velox::get_output_mr());
    next_ = 0;
  }

  std::optional<RowVectorPtr> next(uint64_t /*size*/, ContinueFuture& /*future*/)
      override {
    while (next_ < payloads_.size()) {
      auto data = std::move(payloads_[next_++]);
      joinExchangeLanes(*data, cudf_velox::get_output_mr());
      if (data->numRows == 0) {
        continue;
      }
      completedRows_ += data->numRows;
      completedBytes_ += data->gpuDataSize();
      return toOutput(std::move(data));
    }
    payloads_.clear();
    next_ = 0;
    return nullptr;
  }

  // Takes over a source that the scan created to preload the next split. Its
  // addSplit() has already taken the split's payloads from the store and
  // copied the host tier's ones to the device.
  void setFromDataSource(std::unique_ptr<DataSource> source) override {
    auto* preloaded = dynamic_cast<ExchangeStoreDataSource*>(source.get());
    VELOX_CHECK_NOT_NULL(preloaded);
    VELOX_CHECK_EQ(next_, payloads_.size(), "The previous split is not done");
    stream_ = preloaded->stream_;
    payloads_ = std::move(preloaded->payloads_);
    next_ = preloaded->next_;
    completedRows_ += preloaded->completedRows_;
    completedBytes_ += preloaded->completedBytes_;
  }

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_UNSUPPORTED("Dynamic filters on a temporary table");
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override {
    return {};
  }

 private:
  RowVectorPtr toOutput(std::unique_ptr<PackedTableWithStream> data) {
    const auto view = data->tableView();
    VELOX_CHECK_EQ(
        view.num_columns(),
        dataColumns_->size(),
        "A stored payload does not have the temporary table's columns");
    const auto numRows = data->numRows;
    if (allColumns_) {
      if (data->table != nullptr) {
        return std::make_shared<cudf_velox::CudfVector>(
            pool_, outputType_, numRows, std::move(data->table), data->stream);
      }
      return std::make_shared<cudf_velox::CudfVector>(
          pool_,
          outputType_,
          numRows,
          std::move(data->packedTable),
          data->stream);
    }
    auto selected = std::make_unique<cudf::table>(
        view.select(columns_), data->stream, cudf_velox::get_output_mr());
    return std::make_shared<cudf_velox::CudfVector>(
        pool_, outputType_, numRows, std::move(selected), data->stream);
  }

  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  RowTypePtr dataColumns_;
  // Position in the table, and so in each payload, of each output column.
  std::vector<cudf::size_type> columns_;
  // Whether the output is every column of the table in order.
  bool allColumns_{false};
  cuda::stream_ref stream_{cudaStream_t{cudaStreamDefault}};
  std::vector<std::unique_ptr<PackedTableWithStream>> payloads_;
  size_t next_{0};
  uint64_t completedRows_{0};
  uint64_t completedBytes_{0};
};

class ExchangeStoreTables : public cudf_velox::connector::hive::TemporaryTableStorage {
 public:
  explicit ExchangeStoreTables(ExchangeCompression compression)
      : compression_(compression) {}

  std::unique_ptr<connector::DataSink> createDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const connector::hive::HiveInsertTableHandle>
          insertTableHandle,
      connector::ConnectorQueryCtx* connectorQueryCtx,
      connector::CommitStrategy commitStrategy,
      const std::shared_ptr<const connector::hive::HiveConfig>& hiveConfig)
      override {
    auto writerFactory = std::make_shared<ExchangeStoreWriterFactory>(
        insertTableHandle->storageFormat(), compression_);
    return std::make_unique<connector::hive::HiveDataSink>(
        std::move(inputType),
        std::move(insertTableHandle),
        connectorQueryCtx,
        commitStrategy,
        hiveConfig,
        std::move(writerFactory));
  }

  std::unique_ptr<connector::DataSource> createDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      connector::ConnectorQueryCtx* connectorQueryCtx) override {
    return std::make_unique<ExchangeStoreDataSource>(
        outputType, tableHandle, columnHandles, connectorQueryCtx);
  }

 private:
  const ExchangeCompression compression_;
};

// Drops the data of a query whose task did not finish, so that a failed or
// cancelled query does not leave its temporary tables in the store. A query that
// finishes reads every file, which takes its data out of the store.
class ExchangeStoreTaskListener : public exec::TaskListener {
 public:
  void onTaskCompletion(
      const std::string& /*taskUuid*/,
      const std::string& taskId,
      exec::TaskState state,
      std::exception_ptr /*error*/,
      exec::TaskStats /*stats*/) override {
    if (state == exec::TaskState::kFinished) {
      return;
    }
    // A Presto task id starts with the query id, which every temporary table
    // name of the query contains.
    const auto queryId = std::string_view{taskId}.substr(0, taskId.find('.'));
    if (!queryId.empty()) {
      ExchangeStore::instance().dropQuery(queryId);
    }
  }
};

std::shared_ptr<ExchangeStoreTaskListener>& taskListener() {
  static std::shared_ptr<ExchangeStoreTaskListener> listener;
  return listener;
}

} // namespace

std::pair<std::string, std::string> storeKey(std::string_view path) {
  const auto fileStart = path.find_last_of('/');
  VELOX_CHECK(
      fileStart != std::string_view::npos && fileStart > 0 &&
          fileStart + 1 < path.size(),
      "Not a path to a temporary table file: {}",
      path);
  const auto tableEnd = fileStart;
  const auto slashBeforeTable = path.find_last_of('/', tableEnd - 1);
  const auto tableStart =
      slashBeforeTable == std::string_view::npos ? 0 : slashBeforeTable + 1;
  return {
      std::string{path.substr(tableStart, tableEnd - tableStart)},
      std::string{path.substr(fileStart + 1)},
  };
}

void registerExchangeStoreTables(uint64_t deviceBytes) {
  ExchangeStore::instance().setDeviceCapacity(deviceBytes);
  if (taskListener() == nullptr) {
    taskListener() = std::make_shared<ExchangeStoreTaskListener>();
    exec::registerTaskListener(taskListener());
  }
  cudf_velox::connector::hive::registerTemporaryTableStorage(
      std::make_shared<ExchangeStoreTables>(parseExchangeCompression(
          cudf_velox::CudfConfig::getInstance().exchangeCompression)));
}

void unregisterExchangeStoreTables() {
  cudf_velox::connector::hive::registerTemporaryTableStorage(nullptr);
  if (taskListener() != nullptr) {
    exec::unregisterTaskListener(taskListener());
    taskListener() = nullptr;
  }
}

} // namespace facebook::velox::ucx_exchange
