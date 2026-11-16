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
#include "velox/experimental/cudf/connectors/hive/FusedForColumnEncoding.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/CudfHivePartitionFunction.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/ExchangeStore.h"

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSink.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/config/Config.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Task.h"
#include "velox/exec/TableWriter.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cctype>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>

namespace facebook::velox::ucx_exchange {
namespace {

using connector::hive::HiveColumnHandle;
using connector::hive::HiveConnectorSplit;
using connector::hive::HiveTableHandle;
using cudf_velox::connector::hive::FusedForEncoding;

// The content of every placeholder file. A file the writer leaves empty is not
// reported to the coordinator (FileDataSink::finalizeWriterFile()).
constexpr std::string_view kPlaceholder{"velox exchange store\n"};

// Runtime stats of the TableWriter: the bytes it stored (uncompressed size),
// the bytes it stored in each tier (encoded and compressed, keys excluded),
// and the time adding them took (copies to the host tier included).
constexpr const char* kLogicalBytes{"exchangeStoreLogicalBytes"};
constexpr const char* kDeviceBytes{"exchangeStoreDeviceBytes"};
constexpr const char* kHostBytes{"exchangeStoreHostBytes"};
constexpr const char* kStoreAddNanos{"exchangeStoreAddNanos"};

// Runtime stats of the TableWriter: per batch, the columns it stored as
// dictionary indices or as differences, the stored bytes of the dictionaries'
// keys, and the time choosing and applying the encodings took.
constexpr const char* kDictionaryColumns{"exchangeStoreDictionaryColumns"};
constexpr const char* kDeltaColumns{"exchangeStoreDeltaColumns"};
constexpr const char* kKeysBytes{"exchangeStoreKeysBytes"};
constexpr const char* kEncodeNanos{"exchangeStoreEncodeNanos"};

// Runtime stats of a temporary table scan: the stored bytes it read from each
// tier and the time spent copying host-tier payloads to the device.
constexpr const char* kReadDeviceBytes{"exchangeStoreReadDeviceBytes"};
constexpr const char* kReadHostBytes{"exchangeStoreReadHostBytes"};
constexpr const char* kReadCopyNanos{"exchangeStoreReadCopyNanos"};

// Rows a writer collects before packing them. The TableWriter hands each file
// a slice of every batch, which is a small part of the batch when the table
// has many buckets, and the encodings need many rows per bucket: a dictionary
// at least 8 rows per key, differences runs of nearby keys.
constexpr vector_size_t kPayloadRows{1'000'000};

// Returns the bytes of 'column' uncompressed: its values or characters, null
// mask and children.
std::size_t uncompressedBytes(cudf::column_view column, cuda::stream_ref stream) {
  std::size_t bytes = column.nullable()
      ? cudf::bitmask_allocation_size_bytes(column.size())
      : 0;
  if (cudf::is_fixed_width(column.type())) {
    bytes += static_cast<std::size_t>(column.size()) *
        cudf::size_of(column.type());
  } else if (column.type().id() == cudf::type_id::STRING) {
    bytes += cudf::strings_column_view(column).chars_size(stream);
  }
  for (auto child = column.child_begin(); child != column.child_end();
       ++child) {
    bytes += uncompressedBytes(*child, stream);
  }
  return bytes;
}

std::size_t uncompressedBytes(cudf::table_view table, cuda::stream_ref stream) {
  std::size_t bytes = 0;
  for (const auto& column : table) {
    bytes += uncompressedBytes(column, stream);
  }
  return bytes;
}

// Returns whether 'compression' packs with fused FOR, the codec for which
// encodeForFusedFor() chooses encodings.
bool packsWithFusedFor(ExchangeCompression compression) {
  return compression == ExchangeCompression::kFusedForBitpacked ||
      compression == ExchangeCompression::kFusedForByteAligned ||
      compression == ExchangeCompression::kFusedForBitpackedLanes;
}

// Groups the rows of 'table' by their bucket in 'buckets', 0 to 'numBuckets'
// - 1, and returns the grouped rows with the first row of each bucket and the
// number of rows. Rows keep their order within a bucket, unlike with
// cudf::partition(), so that keys that arrive in key order keep the small
// differences they are stored as (see encodeForFusedFor()).
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
partitionInOrder(
    cudf::table_view table,
    cudf::column_view buckets,
    int32_t numBuckets,
    cuda::stream_ref stream) {
  const auto tempMr = cudf_velox::get_temp_mr();
  const auto order = cudf::stable_sorted_order(
      cudf::table_view{{buckets}},
      {cudf::order::ASCENDING},
      {cudf::null_order::AFTER},
      stream,
      tempMr);
  auto grouped = cudf::gather(
      table,
      order->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      cudf::memory_resources{tempMr, tempMr});
  const auto sortedBuckets = cudf::gather(
      cudf::table_view{{buckets}},
      order->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      cudf::memory_resources{tempMr, tempMr});
  const cudf::numeric_scalar<int32_t> zero(0, true, stream, tempMr);
  const auto bucketNumbers =
      cudf::sequence(numBuckets + 1, zero, stream, tempMr);
  const auto firstRows = cudf::lower_bound(
      sortedBuckets->view(),
      cudf::table_view{{bucketNumbers->view()}},
      {cudf::order::ASCENDING},
      {cudf::null_order::AFTER},
      stream,
      tempMr);
  std::vector<cudf::size_type> offsets(numBuckets + 1);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsets.data(),
      firstRows->view().data<cudf::size_type>(),
      offsets.size() * sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  return {std::move(grouped), std::move(offsets)};
}

// A batch packed for the store: the payload of each bucket, the encoding of
// each column, which the payloads share, the keys of each dictionary column in
// column order with their row counts, and for each column of differences in
// column order the base of each bucket.
struct PackedBatch {
  std::vector<PackedExchangePayload> payloads;
  std::vector<FusedForEncoding> encodings;
  std::vector<PackedExchangePayload> keys;
  std::vector<vector_size_t> numKeys;
  std::vector<std::vector<int64_t>> bases;
};

// Encodes the columns of 'table' as the scan cache does (see
// encodeForFusedFor()) and packs each bucket. Bucket b holds rows 'offsets'[b]
// to 'offsets'[b + 1], so the differences of a column start over at each
// bucket, and the dictionary of a column holds the keys of every bucket.
// 'chosen' holds a writer's encoding per column, chosen on its first batch:
// the batches of a temporary table carry the same kind of data, so later
// batches are encoded the same way without trying the others
// (encodeAsFusedFor()). Reports the encodings as runtime stats.
PackedBatch packBatch(
    cudf::table_view table,
    const std::vector<cudf::size_type>& offsets,
    ExchangeCompression compression,
    std::vector<std::optional<FusedForEncoding>>& chosen,
    cuda::stream_ref stream) {
  VELOX_CHECK_GE(offsets.size(), 2);
  PackedBatch batch;
  std::vector<cudf::column_view> columns;
  // The encoded columns, which 'columns' points into.
  std::vector<std::unique_ptr<cudf::column>> encodedColumns;
  if (packsWithFusedFor(compression)) {
    const auto start = std::chrono::steady_clock::now();
    const std::vector<cudf::size_type> segmentStarts(
        offsets.begin(), offsets.end() - 1);
    int64_t numDictionaryColumns = 0;
    int64_t numDeltaColumns = 0;
    chosen.resize(table.num_columns());
    for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
      const auto column =
          cudf_velox::connector::hive::withoutEmptyNullMask(table.column(i));
      // encodeForFusedFor() encodes top-level fixed-width and string columns.
      if (!cudf::is_fixed_width(column.type()) &&
          column.type().id() != cudf::type_id::STRING) {
        batch.encodings.push_back(FusedForEncoding::kValues);
        columns.push_back(column);
        continue;
      }
      auto encoded = chosen[i].has_value()
          ? cudf_velox::connector::hive::encodeAsFusedFor(
                *chosen[i], column, segmentStarts, stream, cudf_velox::get_temp_mr())
          : cudf_velox::connector::hive::encodeForFusedFor(
                column, segmentStarts, stream, cudf_velox::get_temp_mr());
      if (!chosen[i].has_value()) {
        chosen[i] = encoded.encoding;
      }
      batch.encodings.push_back(encoded.encoding);
      if (encoded.encoding == FusedForEncoding::kValues) {
        columns.push_back(column);
        continue;
      }
      columns.push_back(encoded.encoded->view());
      encodedColumns.push_back(std::move(encoded.encoded));
      if (encoded.encoding == FusedForEncoding::kDeltas) {
        ++numDeltaColumns;
        VELOX_CHECK_EQ(encoded.segmentBases.size(), segmentStarts.size());
        batch.bases.push_back(std::move(encoded.segmentBases));
        continue;
      }
      ++numDictionaryColumns;
      batch.numKeys.push_back(encoded.keys->size());
      batch.keys.push_back(packExchangePayload(
          cudf::table_view{{encoded.keys->view()}},
          compression,
          stream,
          cudf_velox::get_temp_mr(),
          cudf_velox::get_output_mr()));
    }
    addThreadLocalRuntimeStat(
        kDictionaryColumns, RuntimeCounter(numDictionaryColumns));
    addThreadLocalRuntimeStat(kDeltaColumns, RuntimeCounter(numDeltaColumns));
    addThreadLocalRuntimeStat(
        kEncodeNanos,
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count(),
            RuntimeCounter::Unit::kNanos));
    if (encodedColumns.empty()) {
      // Every column holds its values.
      batch.encodings.clear();
    }
  } else {
    columns.assign(table.begin(), table.end());
  }
  const std::vector<cudf::size_type> splits(offsets.begin() + 1, offsets.end() - 1);
  batch.payloads = splitExchangePayloads(
      cudf::table_view(columns),
      splits,
      compression,
      stream,
      cudf_velox::get_temp_mr(),
      cudf_velox::get_output_mr());
  return batch;
}

// Adds the keys of 'batch' to 'table' in the store, shared by
// 'numReferences' payloads, and returns the encoding those payloads are
// stored with, without the bases (see bucketEncoding()).
ExchangeStore::PayloadEncoding storeKeys(
    std::string_view table,
    PackedBatch& batch,
    uint32_t numReferences,
    cuda::stream_ref stream) {
  ExchangeStore::PayloadEncoding encoding;
  encoding.columns = batch.encodings;
  uint64_t keysBytes = 0;
  for (size_t i = 0; i < batch.keys.size(); ++i) {
    keysBytes += batch.keys[i].packed->gpu_data
        ? batch.keys[i].packed->gpu_data->size()
        : 0;
    encoding.keys.push_back(ExchangeStore::instance().addKeys(
        table, std::move(batch.keys[i]), batch.numKeys[i], numReferences, stream));
  }
  if (!batch.keys.empty()) {
    addThreadLocalRuntimeStat(
        kKeysBytes, RuntimeCounter(keysBytes, RuntimeCounter::Unit::kBytes));
  }
  return encoding;
}

// Returns 'encoding' with the bases of 'bucket' of 'batch'.
ExchangeStore::PayloadEncoding bucketEncoding(
    const ExchangeStore::PayloadEncoding& encoding,
    const PackedBatch& batch,
    int32_t bucket) {
  auto result = encoding;
  for (const auto& bases : batch.bases) {
    result.bases.push_back(bases[bucket]);
  }
  return result;
}

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

  // Adds a payload that the sink packed on the GPU.
  void addPayload(
      PackedExchangePayload payload,
      vector_size_t numRows,
      ExchangeStore::PayloadEncoding encoding,
      cuda::stream_ref stream) {
    checkRunning();
    store(std::move(payload), numRows, std::move(encoding), stream);
  }

  // The table under which the store keeps the file.
  const std::string& table() const {
    return table_;
  }

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

  // Packs 'table', all of which belongs to this writer's file, and adds it.
  void add(cudf::table_view table, vector_size_t numRows, cuda::stream_ref stream) {
    addThreadLocalRuntimeStat(
        kLogicalBytes,
        RuntimeCounter(
            uncompressedBytes(table, stream), RuntimeCounter::Unit::kBytes));
    auto batch =
        packBatch(table, {0, numRows}, compression_, encodings_, stream);
    VELOX_CHECK_EQ(batch.payloads.size(), 1);
    stream.sync();
    const auto encoding =
        bucketEncoding(storeKeys(table_, batch, 1, stream), batch, 0);
    store(std::move(batch.payloads[0]), numRows, encoding, stream);
  }

  // Adds a payload to the store and reports its size and tier as runtime
  // stats of the TableWriter.
  void store(
      PackedExchangePayload payload,
      vector_size_t numRows,
      ExchangeStore::PayloadEncoding encoding,
      cuda::stream_ref stream) {
    const auto storedBytes =
        payload.packed->gpu_data ? payload.packed->gpu_data->size() : 0;
    const auto start = std::chrono::steady_clock::now();
    const bool toHost = ExchangeStore::instance().add(
        table_, file_, std::move(payload), numRows, std::move(encoding), stream);
    addThreadLocalRuntimeStat(
        kStoreAddNanos,
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count(),
            RuntimeCounter::Unit::kNanos));
    addThreadLocalRuntimeStat(
        toHost ? kHostBytes : kDeviceBytes,
        RuntimeCounter(storedBytes, RuntimeCounter::Unit::kBytes));
  }

  const std::unique_ptr<dwio::common::FileSink> sink_;
  const std::shared_ptr<memory::MemoryPool> pool_;
  const ExchangeCompression compression_;
  // The encoding of each column, chosen on the first batch (see packBatch()).
  std::vector<std::optional<FusedForEncoding>> encodings_;
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

// Runtime stats of the TableWriter: the GPU batches it received, and the
// batches it bucketed and packed on the GPU after collecting them.
constexpr const char* kGpuInputBatches{"exchangeStoreGpuInputBatches"};
constexpr const char* kGpuBatches{"exchangeStoreGpuBatches"};

// A HiveDataSink whose files are kept in the exchange store. GPU input is
// collected up to kPayloadRows rows, then bucketed, encoded, split and packed
// on the GPU, one cuDF call per step for all buckets, and each bucket's
// payload goes to the writer of the bucket's file. CPU input takes
// HiveDataSink's path.
class ExchangeStoreDataSink : public connector::hive::HiveDataSink {
 public:
  ExchangeStoreDataSink(
      RowTypePtr inputType,
      std::shared_ptr<const connector::hive::HiveInsertTableHandle>
          insertTableHandle,
      const connector::ConnectorQueryCtx* connectorQueryCtx,
      connector::CommitStrategy commitStrategy,
      const std::shared_ptr<const connector::hive::HiveConfig>& hiveConfig,
      ExchangeCompression compression)
      : HiveDataSink(
            inputType,
            insertTableHandle,
            connectorQueryCtx,
            commitStrategy,
            hiveConfig,
            std::make_shared<ExchangeStoreWriterFactory>(
                insertTableHandle->storageFormat(), compression)),
        inputType_(inputType),
        pool_(connectorQueryCtx->memoryPool()),
        compression_(compression),
        gpuBucketFunction_(makeGpuBucketFunction(inputType, *insertTableHandle)) {
    // The table is complete on this worker once this task ends (see
    // ExchangeStoreTaskListener): readers that start while it runs wait for it.
    const auto& location = *insertTableHandle->locationHandle();
    const auto& directory = location.writePath().empty()
        ? location.targetPath()
        : location.writePath();
    ExchangeStore::instance().beginWrite(
        storeKey(directory + "/file").first, connectorQueryCtx->taskId());
  }

  void appendData(RowVectorPtr input) override {
    auto gpuInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
    if (gpuInput == nullptr) {
      HiveDataSink::appendData(std::move(input));
      return;
    }
    if (!gpuBucketFunction_.has_value()) {
      // The bucket function has no GPU version: bucket on the CPU.
      HiveDataSink::appendData(cudf_velox::with_arrow::toVeloxColumn(
          gpuInput->getTableView(),
          pool_,
          inputType_,
          "",
          gpuInput->stream(),
          cudf_velox::get_temp_mr()));
      return;
    }
    checkRunning();
    if (input->size() == 0) {
      return;
    }
    addThreadLocalRuntimeStat(kGpuInputBatches, RuntimeCounter(1));
    pending_.push_back(std::move(gpuInput));
    pendingRows_ += input->size();
    if (pendingRows_ >= kPayloadRows) {
      storePending();
    }
  }

  bool finish() override {
    if (!pending_.empty()) {
      storePending();
    }
    return HiveDataSink::finish();
  }

  void abort() override {
    pending_.clear();
    pendingRows_ = 0;
    HiveDataSink::abort();
  }

 private:
  // Stores the collected GPU batches as one batch.
  void storePending() {
    const auto stream = pending_.back()->stream();
    std::unique_ptr<cudf::table> concatenated;
    cudf::table_view table;
    if (pending_.size() == 1) {
      table = pending_[0]->getTableView();
    } else {
      // Each batch was produced on its own stream.
      std::vector<cudf::table_view> views;
      for (const auto& batch : pending_) {
        batch->stream().sync();
        views.push_back(batch->getTableView());
      }
      concatenated =
          cudf::concatenate(views, stream, cudf_velox::get_temp_mr());
      table = concatenated->view();
    }
    store(table, stream);
    // store() synchronizes 'stream' after its last read of the batches.
    pending_.clear();
    pendingRows_ = 0;
  }

  // Buckets, encodes and packs 'table' and hands each bucket's payload to the
  // writer of the bucket's file.
  void store(cudf::table_view table, cuda::stream_ref stream) {
    const auto buckets =
        gpuBucketFunction_->partitions(table, stream, cudf_velox::get_temp_mr());
    auto [partitioned, offsets] =
        partitionInOrder(table, buckets->view(), bucketCount_, stream);
    VELOX_CHECK_EQ(offsets.size(), static_cast<size_t>(bucketCount_) + 1);
    addThreadLocalRuntimeStat(
        kLogicalBytes,
        RuntimeCounter(
            uncompressedBytes(table, stream), RuntimeCounter::Unit::kBytes));
    auto batch = packBatch(
        partitioned->view(), offsets, compression_, encodings_, stream);
    VELOX_CHECK_EQ(batch.payloads.size(), static_cast<size_t>(bucketCount_));
    stream.sync();
    addThreadLocalRuntimeStat(kGpuBatches, RuntimeCounter(1));

    // The writer of each bucket with rows. The keys are stored under the
    // writers' table and shared by their payloads.
    std::vector<std::pair<int32_t, uint32_t>> bucketWriters;
    for (int32_t bucket = 0; bucket < bucketCount_; ++bucket) {
      if (offsets[bucket + 1] > offsets[bucket]) {
        bucketWriters.emplace_back(
            bucket,
            ensureWriter(
                connector::hive::WriterId{
                    std::nullopt, static_cast<uint32_t>(bucket)}));
      }
    }
    const auto writerAt = [&](uint32_t index) {
      auto* writer = dynamic_cast<ExchangeStoreWriter*>(writers_[index].get());
      VELOX_CHECK_NOT_NULL(writer);
      return writer;
    };
    const auto encoding = storeKeys(
        writerAt(bucketWriters.front().second)->table(),
        batch,
        bucketWriters.size(),
        stream);
    for (const auto& [bucket, index] : bucketWriters) {
      const auto numRows = offsets[bucket + 1] - offsets[bucket];
      const auto bytes = batch.payloads[bucket].logicalDataSize;
      writerAt(index)->addPayload(
          std::move(batch.payloads[bucket]),
          numRows,
          bucketEncoding(encoding, batch, bucket),
          stream);
      auto& info = writerInfo_[index];
      info->numWrittenRows += numRows;
      info->currentFileWrittenRows += numRows;
      info->inputSizeInBytes += bytes;
    }
  }

  // Returns the GPU version of the table's bucket function, or nothing when
  // there is none: Presto's native bucket function, partition columns, or a
  // key type the GPU cannot hash.
  static std::optional<cudf_velox::CudfHivePartitionFunction>
  makeGpuBucketFunction(
      const RowTypePtr& inputType,
      const connector::hive::HiveInsertTableHandle& insertTableHandle) {
    const auto& bucketProperty = insertTableHandle.bucketProperty();
    if (bucketProperty == nullptr ||
        bucketProperty->kind() !=
            connector::hive::HiveBucketProperty::Kind::kHiveCompatible ||
        !insertTableHandle.partitionChannels().empty()) {
      return std::nullopt;
    }
    std::vector<cudf::size_type> keyColumns;
    for (const auto& name : bucketProperty->bucketedBy()) {
      const auto channel = inputType->getChildIdx(name);
      if (!cudf_velox::CudfHivePartitionFunction::supports(
              cudf_velox::veloxToCudfDataType(inputType->childAt(channel)))) {
        return std::nullopt;
      }
      keyColumns.push_back(static_cast<cudf::size_type>(channel));
    }
    return cudf_velox::CudfHivePartitionFunction(
        bucketProperty->bucketCount(), {}, std::move(keyColumns));
  }

  const RowTypePtr inputType_;
  memory::MemoryPool* const pool_;
  const ExchangeCompression compression_;
  // The encoding of each column, chosen on the first batch (see packBatch()).
  std::vector<std::optional<FusedForEncoding>> encodings_;
  const std::optional<cudf_velox::CudfHivePartitionFunction> gpuBucketFunction_;
  // GPU batches not stored yet.
  std::vector<std::shared_ptr<cudf_velox::CudfVector>> pending_;
  vector_size_t pendingRows_{0};
};

// The TableWriter of a temporary table on the GPU: it hands GPU batches to the
// exchange store's sink without copying them to the CPU.
class ExchangeStoreTableWriter : public exec::TableWriter {
 public:
  ExchangeStoreTableWriter(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const core::TableWriteNodePtr& tableWriteNode)
      : exec::TableWriter(
            operatorId,
            driverCtx,
            tableWriteNode,
            "ExchangeStoreTableWrite") {}

 protected:
  RowVectorPtr mapInput(const RowVectorPtr& input) const override {
    auto gpuInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
    if (gpuInput == nullptr) {
      return exec::TableWriter::mapInput(input);
    }
    const auto& mapping = inputMapping();
    std::vector<cudf::size_type> columns(mapping.begin(), mapping.end());
    std::vector<cudf::size_type> allColumns(gpuInput->getTableView().num_columns());
    std::iota(allColumns.begin(), allColumns.end(), 0);
    if (columns == allColumns) {
      return input;
    }
    auto selected = std::make_unique<cudf::table>(
        gpuInput->getTableView().select(columns),
        gpuInput->stream(),
        cudf_velox::get_output_mr());
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    const auto& inputType = asRowType(input->type());
    for (const auto column : mapping) {
      names.push_back(inputType->nameOf(column));
      types.push_back(inputType->childAt(column));
    }
    return std::make_shared<cudf_velox::CudfVector>(
        input->pool(),
        ROW(std::move(names), std::move(types)),
        input->size(),
        std::move(selected),
        gpuInput->stream());
  }
};

// Puts ExchangeStoreTableWriter in place of the TableWriter of a temporary
// table, so that its input stays on the GPU.
class ExchangeStoreTableWriteAdapter : public cudf_velox::OperatorAdapter {
 public:
  ExchangeStoreTableWriteAdapter() : OperatorAdapter("ExchangeStoreTableWrite") {}

  bool canHandle(const exec::Operator* op) const override {
    return op->is<exec::TableWriter>();
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto tableWrite =
        std::dynamic_pointer_cast<const core::TableWriteNode>(planNode);
    if (tableWrite == nullptr) {
      return false;
    }
    auto insertTableHandle =
        std::dynamic_pointer_cast<const connector::hive::HiveInsertTableHandle>(
            tableWrite->insertTableHandle()->connectorInsertTableHandle());
    return insertTableHandle != nullptr &&
        cudf_velox::connector::hive::isTemporaryTable(
               insertTableHandle->locationHandle()->targetPath());
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(std::make_unique<ExchangeStoreTableWriter>(
        operatorId,
        ctx,
        std::dynamic_pointer_cast<const core::TableWriteNode>(planNode)));
    return result;
  }
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

// Returns the files of its table, as storeKey() names them, that 'split' of
// the file 'file' reads: that file, or, when the coordinator reads the table as
// read buckets (kReadBucketCountSplitInfo), every bucket file of its read
// bucket.
std::vector<std::string> splitFiles(
    const connector::ConnectorSplit& split,
    const std::string& file) {
  auto hiveSplit = dynamic_cast<const HiveConnectorSplit*>(&split);
  if (hiveSplit == nullptr) {
    return {file};
  }
  const auto& info = hiveSplit->customSplitInfo;
  const auto readBucketCountIt =
      info.find(std::string{kReadBucketCountSplitInfo});
  const auto tableBucketCountIt =
      info.find(std::string{kTableBucketCountSplitInfo});
  if (readBucketCountIt == info.end() || tableBucketCountIt == info.end()) {
    return {file};
  }
  const auto readBucketCount = folly::to<int32_t>(readBucketCountIt->second);
  const auto tableBucketCount = folly::to<int32_t>(tableBucketCountIt->second);
  const auto bucket = folly::to<int32_t>(file);
  VELOX_CHECK(
      readBucketCount > 0 && tableBucketCount % readBucketCount == 0 &&
          bucket < tableBucketCount,
      "A table bucket must belong to one read bucket: {} {} {}",
      bucket,
      readBucketCount,
      tableBucketCount);
  std::vector<std::string> files;
  for (auto tableBucket = bucket % readBucketCount;
       tableBucket < tableBucketCount;
       tableBucket += readBucketCount) {
    files.push_back(std::to_string(tableBucket));
  }
  return files;
}

// A payload taken from the store and ready to read: the columns that were
// encoded are decoded into columns of their own, the others stay in the
// payload's storage.
struct RestoredPayload {
  std::unique_ptr<PackedTableWithStream> data;
  // The decoded column at each encoded column's index, null elsewhere.
  std::vector<std::unique_ptr<cudf::column>> decoded;
  cudf::table_view view;
};

// Device bytes 'restored' holds: its payload's storage and its decoded
// columns.
uint64_t deviceBytes(const RestoredPayload& restored) {
  uint64_t bytes = restored.data->gpuDataSize();
  for (const auto& column : restored.decoded) {
    if (column != nullptr) {
      bytes += column->alloc_size();
    }
  }
  return bytes;
}

RestoredPayload restore(ExchangeStore::TakenPayload taken, cuda::stream_ref stream) {
  joinExchangeLanes(*taken.data, cudf_velox::get_output_mr());
  RestoredPayload restored;
  restored.view = taken.data->tableView();
  const auto& encodings = taken.encoding.columns;
  if (!encodings.empty()) {
    VELOX_CHECK_EQ(
        encodings.size(),
        restored.view.num_columns(),
        "A stored payload's encodings do not match its columns");
    std::vector<cudf::column_view> columns(
        restored.view.begin(), restored.view.end());
    restored.decoded.resize(columns.size());
    size_t nextKeys = 0;
    size_t nextBase = 0;
    for (size_t i = 0; i < columns.size(); ++i) {
      std::optional<cudf::column_view> keys;
      int64_t base = 0;
      if (encodings[i] == FusedForEncoding::kDictionary) {
        VELOX_CHECK_LT(nextKeys, taken.keys.size());
        auto& keysData = *taken.keys[nextKeys++];
        joinExchangeLanes(keysData, cudf_velox::get_output_mr());
        keys = keysData.tableView().column(0);
      } else if (encodings[i] == FusedForEncoding::kDeltas) {
        VELOX_CHECK_LT(nextBase, taken.encoding.bases.size());
        base = taken.encoding.bases[nextBase++];
      }
      restored.decoded[i] = cudf_velox::connector::hive::decodeFusedForEncoding(
          encodings[i],
          columns[i],
          keys,
          base,
          stream,
          cudf_velox::get_output_mr(),
          cudf_velox::get_temp_mr());
      if (restored.decoded[i] != nullptr) {
        columns[i] = restored.decoded[i]->view();
      }
    }
    restored.view = cudf::table_view(columns);
    // The keys are freed on 'stream', after the gathers that read them.
  }
  restored.data = std::move(taken.data);
  return restored;
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
      : outputType_(outputType),
        pool_(connectorQueryCtx->memoryPool()),
        batchTarget_(outputType_) {
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

  // Takes the first batch of the payloads of the split's files written so far
  // (see takeMore()). The rest, and a file that is still being written (its
  // table is read while the producer runs), are read in next(), which waits
  // for what is not written yet.
  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    VELOX_CHECK_EQ(next_, payloads_.size(), "The previous split is not done");
    std::string file;
    std::tie(table_, file) = storeKey(filePath(*split));
    files_ = splitFiles(*split, file);
    stream_ = cudf_velox::cudfGlobalStreamPool().get_stream();
    payloads_.clear();
    next_ = 0;
    splitDone_ = false;
    auto ignored = ContinueFuture::makeEmpty();
    takeMore(ignored);
  }

  // Returns the split's payloads concatenated into batches of CudfBatchConcat's
  // target where the payloads allow. A writer splits every batch it gets
  // across all buckets, so a file holds many small payloads, and each batch
  // costs the operators downstream a fixed overhead.
  std::optional<RowVectorPtr> next(uint64_t /*size*/, ContinueFuture& future)
      override {
    // Payloads without rows give no batch; take more until one does or the
    // split is done.
    for (;;) {
      while (next_ == payloads_.size() && !splitDone_) {
        if (!takeMore(future)) {
          return std::nullopt;
        }
      }
      std::vector<RestoredPayload> batch;
      vector_size_t batchRows = 0;
      uint64_t batchBytes = 0;
      while (next_ < payloads_.size() &&
             (batch.empty() || !batchTarget_.isMet(batchRows, batchBytes)) &&
             (batch.empty() ||
              static_cast<int64_t>(batchRows) +
                      payloads_[next_].data->numRows <=
                  std::numeric_limits<cudf::size_type>::max())) {
        auto restored = restore(std::move(payloads_[next_++]), stream_);
        const auto numRows = restored.data->numRows;
        if (numRows == 0) {
          continue;
        }
        completedRows_ += numRows;
        completedBytes_ += restored.data->gpuDataSize();
        batchRows += numRows;
        batchBytes += deviceBytes(restored);
        batch.push_back(std::move(restored));
      }
      if (batch.size() == 1) {
        return toOutput(std::move(batch[0]));
      }
      if (!batch.empty()) {
        return concatenate(batch, batchRows);
      }
      if (splitDone_) {
        payloads_.clear();
        next_ = 0;
        return nullptr;
      }
    }
  }

  // Takes over a source that the scan created to preload the next split. Its
  // addSplit() has already taken the split's first batch from the store and
  // copied the host tier's payloads of it to the device.
  void setFromDataSource(std::unique_ptr<DataSource> source) override {
    auto* preloaded = dynamic_cast<ExchangeStoreDataSource*>(source.get());
    VELOX_CHECK_NOT_NULL(preloaded);
    VELOX_CHECK_EQ(next_, payloads_.size(), "The previous split is not done");
    stream_ = preloaded->stream_;
    table_ = std::move(preloaded->table_);
    files_ = std::move(preloaded->files_);
    splitDone_ = preloaded->splitDone_;
    payloads_ = std::move(preloaded->payloads_);
    next_ = preloaded->next_;
    completedRows_ += preloaded->completedRows_;
    completedBytes_ += preloaded->completedBytes_;
    takeStats_.deviceBytes += preloaded->takeStats_.deviceBytes;
    takeStats_.hostBytes += preloaded->takeStats_.hostBytes;
    takeStats_.copyNanos += preloaded->takeStats_.copyNanos;
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
    return {
        {kReadDeviceBytes,
         RuntimeMetric(takeStats_.deviceBytes, RuntimeCounter::Unit::kBytes)},
        {kReadHostBytes,
         RuntimeMetric(takeStats_.hostBytes, RuntimeCounter::Unit::kBytes)},
        {kReadCopyNanos,
         RuntimeMetric(takeStats_.copyNanos, RuntimeCounter::Unit::kNanos)},
    };
  }

 private:
  RowVectorPtr toOutput(RestoredPayload restored) {
    const auto view = restored.view;
    VELOX_CHECK_EQ(
        view.num_columns(),
        dataColumns_->size(),
        "A stored payload does not have the temporary table's columns");
    auto& data = restored.data;
    const auto numRows = data->numRows;
    if (!restored.decoded.empty()) {
      // Decoded columns move to the output, the others are copied out of the
      // payload's storage.
      std::vector<std::unique_ptr<cudf::column>> columns;
      for (const auto column : columns_) {
        if (restored.decoded[column] != nullptr) {
          columns.push_back(std::move(restored.decoded[column]));
        } else {
          columns.push_back(std::make_unique<cudf::column>(
              view.column(column), data->stream, cudf_velox::get_output_mr()));
        }
      }
      return std::make_shared<cudf_velox::CudfVector>(
          pool_,
          outputType_,
          numRows,
          std::make_unique<cudf::table>(std::move(columns)),
          data->stream);
    }
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

  // Concatenates the scan's columns of 'batch' into one CudfVector.
  RowVectorPtr concatenate(
      const std::vector<RestoredPayload>& batch,
      vector_size_t numRows) {
    std::vector<cudf::table_view> views;
    views.reserve(batch.size());
    for (const auto& restored : batch) {
      const auto view = restored.view;
      VELOX_CHECK_EQ(
          view.num_columns(),
          dataColumns_->size(),
          "A stored payload does not have the temporary table's columns");
      views.push_back(view.select(columns_));
    }
    // A scan without columns, such as count(*), needs only the row count.
    // The payloads were restored on 'stream_' (see addSplit()).
    auto table = columns_.empty()
        ? std::make_unique<cudf::table>()
        : cudf::concatenate(views, stream_, cudf_velox::get_output_mr());
    return std::make_shared<cudf_velox::CudfVector>(
        pool_, outputType_, numRows, std::move(table), stream_);
  }

  // Replaces the taken payloads with the next ones of the split's files, one
  // batch of CudfConfig::batchSizeMinThreshold rows: the others stay in the
  // store, which can still move them to host memory, so the scan holds one
  // batch on the device instead of the whole split. Returns false with
  // 'future' set when the files have none yet and are still being written;
  // sets splitDone_ when they have none and are complete.
  bool takeMore(ContinueFuture& future) {
    auto taken = ExchangeStore::instance().takeOrWait(
        table_,
        files_,
        stream_,
        cudf_velox::get_output_mr(),
        future,
        &takeStats_,
        cudf_velox::CudfConfig::getInstance().batchSizeMinThreshold);
    if (!taken.has_value()) {
      return false;
    }
    if (taken->empty()) {
      splitDone_ = true;
    }
    payloads_ = std::move(*taken);
    next_ = 0;
    return true;
  }

  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  // When a batch of payloads is complete, as for CudfBatchConcat.
  const cudf_velox::BatchConcatTarget batchTarget_;
  RowTypePtr dataColumns_;
  // Position in the table, and so in each payload, of each output column.
  std::vector<cudf::size_type> columns_;
  // Whether the output is every column of the table in order.
  bool allColumns_{false};
  cuda::stream_ref stream_{cudaStream_t{cudaStreamDefault}};
  // The split's temporary table (storeKey()) and the files it reads
  // (splitFiles()).
  std::string table_;
  std::vector<std::string> files_;
  // Set once the split's files have no more payloads and its table is
  // complete.
  bool splitDone_{true};
  std::vector<ExchangeStore::TakenPayload> payloads_;
  size_t next_{0};
  uint64_t completedRows_{0};
  uint64_t completedBytes_{0};
  ExchangeStore::TakeStats takeStats_;
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
    return std::make_unique<ExchangeStoreDataSink>(
        std::move(inputType),
        std::move(insertTableHandle),
        connectorQueryCtx,
        commitStrategy,
        hiveConfig,
        compression_);
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
    // Completes the tables this task wrote, waking readers that wait for them.
    ExchangeStore::instance().endWrites(taskId);
    if (state != exec::TaskState::kFinished) {
      // A Presto task id starts with the query id, which every temporary table
      // name of the query contains.
      const auto queryId =
          std::string_view{taskId}.substr(0, taskId.find('.'));
      if (!queryId.empty()) {
        ExchangeStore::instance().dropQuery(queryId);
      }
    }
    logOccupancy(taskId, state);
    if (isWorkerIdle()) {
      ExchangeStore::instance().releaseHostPoolIfUnused();
    }
  }

 private:
  // Whether no task runs on this worker, as in the busy periods that
  // MemoryPeakLogger reports.
  static bool isWorkerIdle() {
    for (const auto& task : exec::Task::getRunningTasks()) {
      if (task->isRunning()) {
        return false;
      }
    }
    return true;
  }

  // What a query leaves in the store shows here: the next query has that much
  // less memory.
  static void logOccupancy(const std::string& taskId, exec::TaskState state) {
    const auto stats = ExchangeStore::instance().stats();
    LOG_IF(INFO, stats.tables > 0)
        << "Exchange store after task " << taskId << " ended ("
        << exec::taskStateString(state) << "): " << stats.tables << " tables, "
        << stats.payloads << " payloads, " << stats.deviceBytes
        << " device bytes, " << stats.hostBytes << " host bytes, "
        << stats.hostPoolBytes << " host pool bytes";
  }
};

std::shared_ptr<ExchangeStoreTaskListener>& taskListener() {
  static std::shared_ptr<ExchangeStoreTaskListener> listener;
  return listener;
}

} // namespace

namespace {

// Returns the bucket number of a bucketed table's file name, written as "0",
// the zero-padded bucket, "_", a number and "_" (computeBucketedFileName() in
// HiveDataSink.cpp), as a decimal without padding; returns the name itself for
// other files.
std::string fileKey(std::string_view name) {
  size_t digitsEnd = 1;
  while (digitsEnd < name.size() && std::isdigit(name[digitsEnd])) {
    ++digitsEnd;
  }
  size_t sequenceEnd = digitsEnd + 1;
  while (sequenceEnd < name.size() && std::isdigit(name[sequenceEnd])) {
    ++sequenceEnd;
  }
  const bool bucketFile = name.size() > 1 && name[0] == '0' &&
      digitsEnd > 1 && digitsEnd < name.size() && name[digitsEnd] == '_' &&
      sequenceEnd > digitsEnd + 1 && sequenceEnd < name.size() &&
      name[sequenceEnd] == '_';
  if (!bucketFile) {
    return std::string{name};
  }
  return std::to_string(
      std::stoll(std::string{name.substr(1, digitsEnd - 1)}));
}

} // namespace

void addReceivedBucketPayload(
    std::string_view table,
    int32_t bucket,
    std::string_view taskId,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    vector_size_t numRows,
    cuda::stream_ref stream) {
  auto& store = ExchangeStore::instance();
  store.beginWrite(table, taskId);
  PackedExchangePayload payload;
  payload.packed = std::make_unique<cudf::packed_columns>(
      std::move(metadata), std::move(data));
  store.add(
      table,
      std::to_string(bucket),
      std::move(payload),
      numRows,
      ExchangeStore::PayloadEncoding{},
      stream);
}

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
      fileKey(path.substr(fileStart + 1)),
  };
}

namespace {

// Returns the source of 'node' when 'node' passes its rows on unchanged: a
// Project whose output column i is its input column i under another name, or a
// local exchange of one source. Returns nullptr otherwise.
const core::PlanNode* passThroughSource(const core::PlanNode& node) {
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(&node)) {
    const auto& inputType = project->sources()[0]->outputType();
    const auto& projections = project->projections();
    if (projections.size() != inputType->size()) {
      return nullptr;
    }
    for (column_index_t i = 0; i < projections.size(); ++i) {
      const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(
          projections[i].get());
      if (field == nullptr || !field->isInputColumn() ||
          inputType->getChildIdx(field->name()) != i) {
        return nullptr;
      }
    }
    return project->sources()[0].get();
  }
  if (dynamic_cast<const core::LocalPartitionNode*>(&node) != nullptr &&
      node.sources().size() == 1) {
    return node.sources()[0].get();
  }
  return nullptr;
}

const core::TableWriteNode* findTableWriter(const core::PlanNode& node) {
  if (const auto* writer = dynamic_cast<const core::TableWriteNode*>(&node)) {
    return writer;
  }
  for (const auto& source : node.sources()) {
    if (const auto* writer = findTableWriter(*source)) {
      return writer;
    }
  }
  return nullptr;
}

} // namespace

std::optional<std::string> receivingTableInStore(
    const core::PlanNode& root,
    std::string_view exchangeId,
    const core::QueryCtx& queryCtx) {
  if (!cudf_velox::CudfConfig::getInstance().exchangeStore) {
    return std::nullopt;
  }
  const auto* writer = findTableWriter(root);
  if (writer == nullptr) {
    return std::nullopt;
  }
  const auto& writerInput = writer->sources()[0]->outputType();
  if (writer->columns()->names() != writerInput->names()) {
    return std::nullopt;
  }
  const core::PlanNode* node = writer->sources()[0].get();
  while (node != nullptr && node->id() != exchangeId) {
    node = passThroughSource(*node);
  }
  if (node == nullptr ||
      dynamic_cast<const core::ExchangeNode*>(node) == nullptr) {
    return std::nullopt;
  }
  const auto insert =
      std::dynamic_pointer_cast<const connector::hive::HiveInsertTableHandle>(
          writer->insertTableHandle()->connectorInsertTableHandle());
  if (insert == nullptr || insert->bucketProperty() == nullptr) {
    return std::nullopt;
  }
  const auto& location = *insert->locationHandle();
  const auto& directory = location.writePath().empty()
      ? location.targetPath()
      : location.writePath();
  if (!cudf_velox::connector::hive::isTemporaryTable(directory)) {
    return std::nullopt;
  }
  const auto* session = queryCtx.connectorSessionProperties(
      writer->insertTableHandle()->connectorId());
  if (session == nullptr ||
      !(session->get<bool>("stream_temporary_tables", false) ||
        session->get<bool>("list_temporary_tables_by_bucket", false))) {
    return std::nullopt;
  }
  return storeKey(directory + "/file").first;
}

void registerExchangeStoreTables(uint64_t deviceBytes, uint64_t hostBytes) {
  ExchangeStore::instance().setDeviceCapacity(deviceBytes);
  ExchangeStore::instance().setHostCapacity(
      hostBytes == 0 ? std::numeric_limits<uint64_t>::max() : hostBytes);
  // unregisterCudf() clears the adapters, so this is added on every
  // registration.
  cudf_velox::OperatorAdapterRegistry::getInstance().registerAdapterFront(
      std::make_unique<ExchangeStoreTableWriteAdapter>());
  if (taskListener() == nullptr) {
    taskListener() = std::make_shared<ExchangeStoreTaskListener>();
    exec::registerTaskListener(taskListener());
  }
  // Releasers cannot be removed, so this one is registered once per process.
  static std::once_flag releaserRegistered;
  std::call_once(releaserRegistered, [] {
    cudf_velox::registerDeviceMemoryReleaser([](std::size_t bytes) {
      return ExchangeStore::instance().demote(bytes) > 0;
    });
  });
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
