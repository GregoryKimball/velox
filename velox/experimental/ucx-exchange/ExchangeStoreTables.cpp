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
    const auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    auto table = cudf_velox::with_arrow::toCudfTable(
        std::dynamic_pointer_cast<RowVector>(data),
        pool_.get(),
        stream,
        cudf_velox::get_temp_mr());
    add(table->view(), data->size(), stream);
  }

  void flush() override {}

  bool finish() override {
    return true;
  }

  std::unique_ptr<dwio::common::FileMetadata> close() override {
    dwio::common::DataBuffer<char> placeholder(*pool_, kPlaceholder.size());
    std::memcpy(placeholder.data(), kPlaceholder.data(), kPlaceholder.size());
    sink_->write(std::move(placeholder));
    sink_->close();
    setState(State::kClosed);
    return nullptr;
  }

  void abort() override {
    // The query fails, so no file of the table will be read.
    ExchangeStore::instance().drop(table_);
    sink_->close();
    setState(State::kAborted);
  }

 private:
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
    auto hiveSplit = std::dynamic_pointer_cast<HiveConnectorSplit>(split);
    VELOX_CHECK_NOT_NULL(hiveSplit);
    const auto [table, file] = storeKey(hiveSplit->filePath);
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
  cudf_velox::connector::hive::registerTemporaryTableStorage(
      std::make_shared<ExchangeStoreTables>(parseExchangeCompression(
          cudf_velox::CudfConfig::getInstance().exchangeCompression)));
}

void unregisterExchangeStoreTables() {
  cudf_velox::connector::hive::registerTemporaryTableStorage(nullptr);
}

} // namespace facebook::velox::ucx_exchange
