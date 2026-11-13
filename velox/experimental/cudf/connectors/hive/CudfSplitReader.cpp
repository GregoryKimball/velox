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
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CachingDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/FusedForScanCache.h"
#include "velox/experimental/cudf/exec/GpuCapabilities.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/functions/lib/string/StringImpl.h"
#ifdef VELOX_ENABLE_ABFS
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#endif

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

// Checks whether the `path` uses an ABFS scheme
bool isAbfsPath([[maybe_unused]] const std::string_view path) {
#ifdef VELOX_ENABLE_ABFS
  return ::facebook::velox::filesystems::isAbfsFile(path);
#else
  return false;
#endif
}

// Rebuilds a struct/list column in-place after possibly transforming (e.g.,
// decimal-casting) its children.
template <typename TransformChildrenFn>
std::unique_ptr<cudf::column> rebuildWithTransformedChildren(
    std::unique_ptr<cudf::column> col,
    TransformChildrenFn&& transformFn) {
  auto const type = col->type();
  auto const size = col->size();
  auto const nullCount = col->null_count();
  auto contents = col->release();
  transformFn(contents.children);
  return std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nullCount,
      std::move(contents.children));
}

// Recursively casts columns to the expected Velox type iff the column is:
//  - Decimal type but not the expected Velox type.
//  - Struct type: with any of its children being decimal type but not the
//  expected Velox type. Rebuilt in place with the casted children.
//  - List type: with its `child` being decimal type but not the expected Velox
//  type. Rebuilt in place with the casted children.
std::unique_ptr<cudf::column> castDecimalColumns(
    std::unique_ptr<cudf::column> col,
    const TypePtr& veloxType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  // Decimal type (base case)
  if (veloxType->isDecimal()) {
    auto const targetType = veloxToCudfDataType(veloxType);
    if (col->type() != targetType) {
      return cudf::cast(col->view(), targetType, stream, mr);
    }
    return col;
  }

  // Struct type
  if (veloxType->kind() == TypeKind::ROW) {
    auto const& rowType = veloxType->asRow();
    auto const numChildren = static_cast<size_t>(col->num_children());
    VELOX_CHECK_EQ(
        numChildren,
        rowType.size(),
        "Scanned STRUCT column has {} fields but the expected schema has {}.",
        numChildren,
        rowType.size());
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      for (size_t i = 0; i < numChildren; ++i) {
        children[i] = castDecimalColumns(
            std::move(children[i]), rowType.childAt(i), stream, mr);
      }
    });
  }

  // List type
  if (veloxType->kind() == TypeKind::ARRAY) {
    // A LIST column stores [offsets, child]; only the child may hold decimal
    // data.
    VELOX_CHECK_EQ(
        col->num_children(),
        2,
        "LIST column must have exactly 2 children: [offsets, child]");
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      auto const childIdx = cudf::lists_column_view::child_column_index;
      children[childIdx] = castDecimalColumns(
          std::move(children[childIdx]), veloxType->childAt(0), stream, mr);
    });
  }

  return col;
}

// Returns a greatest-priority CUDA stream of the calling thread. Each thread
// hands out its two streams in turn, as cuDF's per-thread stream pool does, so
// that consecutive splits prepared on one thread get different streams. They
// are never destroyed: destroying them from a thread_local destructor could
// race driver teardown when the process exits.
cuda::stream_ref highPriorityStreamForCurrentThread() {
  static thread_local std::array<cudaStream_t, 2> streams{};
  static thread_local std::size_t next = 0;
  auto& stream = streams[next++ % streams.size()];
  if (stream == nullptr) {
    int leastPriority = 0;
    int greatestPriority = 0;
    VELOX_CHECK_EQ(
        static_cast<int>(cudaSuccess),
        static_cast<int>(
            cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority)),
        "Failed to get the CUDA stream priority range");
    VELOX_CHECK_EQ(
        static_cast<int>(cudaSuccess),
        static_cast<int>(cudaStreamCreateWithPriority(
            &stream, cudaStreamNonBlocking, greatestPriority)),
        "Failed to create a high-priority CUDA stream");
  }
  return cuda::stream_ref{stream};
}

} // namespace

std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    std::span<const TypePtr> columnTypes,
    size_t numPrependedColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(
      numPrependedColumns + columnTypes.size(),
      table->view().num_columns(),
      "Read column types must match the cuDF table width");
  auto columns = table->release();
  for (size_t i = 0; i < columnTypes.size(); ++i) {
    const auto columnIndex = numPrependedColumns + i;
    columns[columnIndex] = castDecimalColumns(
        std::move(columns[columnIndex]), columnTypes[i], stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

CudfSplitReader::CudfSplitReader(
    std::shared_ptr<CudfHiveConnectorSplit> split,
    std::shared_ptr<const HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    const cudf::ast::expression* subfieldFilterAst)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241},
          std::nullopt,
          fmt::format("[split:{}]", split ? split->filePath : "unknown")),
      split_(std::move(split)),
      tableHandle_(std::move(tableHandle)),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      readColumnTypes_(outputType->children()),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      cudfHiveConfig_(cudfHiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      baseReaderOpts_(pool_),
      subfieldFilterAst_(subfieldFilterAst),
      pushdownFilterExpr_(subfieldFilterAst) {
  VELOX_CHECK_GE(
      readColumnNames_.size(),
      readColumnTypes_.size(),
      "Read columns must include all output columns");
  if (readColumnNames_.size() > readColumnTypes_.size()) {
    const auto& dataColumns = tableHandle_->dataColumns();
    VELOX_CHECK_NOT_NULL(
        dataColumns,
        "Table schema is required to resolve filter-only column types");
    for (size_t i = readColumnTypes_.size(); i < readColumnNames_.size(); ++i) {
      readColumnTypes_.push_back(dataColumns->findChild(readColumnNames_[i]));
    }
  }
  VELOX_DCHECK_EQ(readColumnNames_.size(), readColumnTypes_.size());
  baseReaderOpts_.setDataIoStats(ioStatistics_);
  baseReaderOpts_.setMetadataIoStats(ioStatistics_);

  // The cuDF path doesn't use connector::hive::configureReaderOptions() to
  // apply configurations so load-quantum, max-coalesced-bytes and
  // max-coalesced-distance-bytes are applied here so a tuned load-quantum
  // reaches CachedBufferedInput.
  const ::facebook::velox::connector::hive::HiveConfig hiveConfig(
      cudfHiveConfig_->config());
  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();
  baseReaderOpts_.setLoadQuantum(hiveConfig.loadQuantum(sessionProperties));
  baseReaderOpts_.setMaxCoalesceBytes(
      hiveConfig.maxCoalescedBytes(sessionProperties));
  baseReaderOpts_.setMaxCoalesceDistance(
      hiveConfig.maxCoalescedDistanceBytes(sessionProperties));
  caseInsensitiveColumnNames_ =
      hiveConfig.isFileColumnNamesReadAsLowerCase(sessionProperties);
  highPriorityScanStreams_ =
      cudfHiveConfig_->highPriorityScanStreamsSession(sessionProperties);
}

CudfSplitReader::~CudfSplitReader() {
  // A split abandoned before it is read, e.g. when the task is cancelled while
  // the preloader prepares it, can still have reads in flight or queued.
  if (passState_ == nullptr) {
    return;
  }
  try {
    releaseCurrentPassData();
  } catch (const std::exception& e) {
    // The data of a failed read is being dropped anyway, so the failure must
    // not propagate out of the destructor.
    LOG(ERROR) << fmt::format(
        "Failed to discard the reads of an abandoned split. Path: {}. Error: {}.",
        split_ != nullptr ? split_->filePath : "unknown",
        e.what());
  }
}

void CudfSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  // Read file metadata and cache schema information
  cacheSchemaFromMetadata();

  // No columns to read (e.g. `count(*)`): take the row count from the footer.
  noColumnsToRead_ = readColumnNames_.empty();
  if (noColumnsToRead_) {
    VELOX_CHECK_NULL(
        pushdownFilter(),
        "A footer-derived row count cannot honor a pushed-down filter");
    return;
  }

  createCudfReader();

  if (const auto capacity = cudfHiveConfig_->fusedForScanCacheBytes();
      capacity > 0 and fusedForCacheEligible()) {
    auto& cache = FusedForScanCache::instance();
    cache.reserveCapacity(capacity);
    // A query that states how much device memory it needs bounds the device
    // tier for its run (session property scan_cache_device_limit_mb).
    const auto limitMb = connectorQueryCtx_->sessionProperties()->get<uint64_t>(
        CudfHiveConfig::kScanCacheDeviceLimitMbSession, 0);
    if (limitMb > 0) {
      cache.limitDeviceCapacity(std::min<uint64_t>(
          limitMb << 20, cudfHiveConfig_->fusedForScanCacheDeviceBytes()));
    } else {
      cache.reserveDeviceCapacity(
          cudfHiveConfig_->fusedForScanCacheDeviceBytes());
    }
    fusedForCacheHit_ = cache.containsAll(split_->filePath, readColumnNames_);
    fusedForCacheRead_ = fusedForCacheHit_ or
        cache.mayCache(split_->filePath, readColumnNames_);
  }
}

void CudfSplitReader::cacheSchemaFromMetadata() {
  // Read file metadatas if not already
  fileMetaDatas();

  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "Expected a single parquet footer for the split's data file");
  const auto& meta = fileMetaData_.front();
  VELOX_CHECK(not meta.schema.empty(), "Parquet footer schema is empty");
  VELOX_CHECK_GE(meta.num_rows, 0, "Parquet footer reports negative row count");
  std::tie(baseReadOffset_, splitRowCount_) = computeSplitRowRange();
  rowGroupNumRows_.clear();
  for (const auto& rowGroup : meta.row_groups) {
    rowGroupNumRows_.push_back(rowGroup.num_rows);
  }

  const auto& root = meta.schema.front();
  fileColumnNames_.clear();
  fileColumnNames_.reserve(root.children_idx.size());
  for (const auto childIdx : root.children_idx) {
    VELOX_CHECK_LT(
        childIdx,
        meta.schema.size(),
        "Parquet schema child index out of range");
    const auto& name = meta.schema[childIdx].name;
    fileColumnNames_.insert(
        caseInsensitiveColumnNames_
            ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
            : name);
  }
}

std::pair<std::size_t, std::size_t> CudfSplitReader::computeSplitRowRange()
    const {
  // Note: This function implements the same logic as cuDF's hybrid scan
  // reader's `filter_row_groups_with_byte_range()` API
  const auto rowGroupOffset = [](const auto& rowGroup) {
    if (rowGroup.file_offset.has_value()) {
      return rowGroup.file_offset.value();
    }
    if (rowGroup.columns.front().file_offset != 0) {
      return rowGroup.columns.front().file_offset;
    }
    const auto& column = rowGroup.columns.front().meta_data;
    return column.dictionary_page_offset != 0
        ? std::min(column.dictionary_page_offset, column.data_page_offset)
        : column.data_page_offset;
  };

  std::size_t startRow{0};
  std::size_t numRows{0};
  for (const auto& rowGroup : fileMetaData_.front().row_groups) {
    VELOX_CHECK(
        not rowGroup.columns.empty(),
        "Parquet footer reports a row group with no column chunks");
    const auto offset = rowGroupOffset(rowGroup);
    if (std::cmp_less(offset, split_->start)) {
      startRow += rowGroup.num_rows;
    } else if (offset - split_->start < split_->size()) {
      numRows += rowGroup.num_rows;
    }
  }
  return {startRow, numRows};
}

void CudfSplitReader::prepareSplit(dwio::common::RuntimeStats& runtimeStats) {
  // Reset existing split and split readers, if any
  resetSplit();

  // Acquire a stream from the global stream pool, or a high-priority one
  stream_ = highPriorityScanStreams_ ? highPriorityStreamForCurrentThread()
                                     : cudfGlobalStreamPool().get_stream();

  // Perform split-specific setup.
  prepareSplitInternal(runtimeStats);

  // Update runtime stats.
  if (isSplitSkipped()) {
    runtimeStats.skippedSplits++;
    // An unbounded length means the whole file, whose size the split does not
    // carry, so it contributes no byte count.
    if (split_->length != std::numeric_limits<uint64_t>::max()) {
      runtimeStats.skippedSplitBytes += static_cast<int64_t>(split_->length);
    }
  } else {
    runtimeStats.processedSplits++;
  }
}

std::optional<CudfSplitReader::TableChunk> CudfSplitReader::next(
    uint64_t /*size*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  auto chunkOpt = readNextChunk();
  if (!chunkOpt.has_value()) {
    return std::nullopt;
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.get(), &CudfSplitReader::totalScanTimeCalculator, callbackData);

  return chunkOpt;
}

void CudfSplitReader::setConnectorQueryCtx(
    const ConnectorQueryCtx* connectorQueryCtx) {
  VELOX_CHECK_NOT_NULL(connectorQueryCtx);
  // A preloaded split can be adopted by a driver with a different memory pool.
  // Only the context pointer changes; 'pool_' and its buffers stay with the
  // pool that allocated them.
  connectorQueryCtx_ = connectorQueryCtx;
}

std::optional<CudfSplitReader::TableChunk> CudfSplitReader::readNextChunk() {
  if (noColumnsToRead_) {
    // A Velox vector holds at most `vector_size_t` rows, so the footer row
    // count of a larger split is returned over several tables.
    const auto numRows = std::min<std::size_t>(
        splitRowCount_,
        static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
    if (numRows == 0) {
      return std::nullopt;
    }
    splitRowCount_ -= numRows;
    return TableChunk{
        std::make_unique<cudf::table>(
            std::vector<std::unique_ptr<cudf::column>>{}),
        static_cast<vector_size_t>(numRows)};
  }

  VELOX_CHECK_NOT_NULL(splitReader_, "cuDF parquet reader not present");
  VELOX_CHECK_NOT_NULL(passState_, "Row group pass state not present");

  if (fusedForCacheChunk_.has_value()) {
    auto chunk = std::move(fusedForCacheChunk_);
    fusedForCacheChunk_.reset();
    return chunk;
  }

  auto outputMr = determineCudfMemoryResource();

  if (passState_->currentPass >= passState_->passes.size()) {
    return std::nullopt;
  }

  if (fusedForCacheRead_) {
    if (auto chunk = readFromFusedForCache(outputMr)) {
      return chunk;
    }
    fusedForCacheRead_ = false;
  }

  if (not passState_->isChunkingSetup) {
    setupChunkingForCurrentPass(outputMr);
    VELOX_CHECK(
        splitReader_->has_next_table_chunk(),
        "cuDF row group pass did not produce a table chunk");
  }

  auto tableWithMetadata = splitReader_->materialize_all_columns_chunk();

  // This was the last chunk of the pass. Drop its fetch buffers and begin
  // I/O for the next pass while the caller consumes this table.
  if (not splitReader_->has_next_table_chunk()) {
    releaseCurrentPassData();
    passState_->isChunkingSetup = false;
    ++passState_->currentPass;
    if (passState_->currentPass < passState_->passes.size()) {
      startColumnChunkFetch();
    } else {
      passState_->passes.clear();
    }
  }

  auto table = castDecimalColumnsToVeloxTypes(
      std::move(tableWithMetadata.tbl),
      readColumnTypes_,
      prependRowIndex_ ? 1 : 0,
      stream_,
      outputMr);
  const auto numRows = table->num_rows();
  return TableChunk{std::move(table), numRows};
}

void CudfSplitReader::startColumnChunkFetch() {
  // Return if the split is served from the fused FOR scan cache, there's no
  // passes, or the fetch is already started or the chunking is already setup.
  if (fusedForCacheHit_ or passState_ == nullptr or
      passState_->currentPass >= passState_->passes.size() or
      passState_->fetch.pending.valid() or passState_->isChunkingSetup) {
    return;
  }

  const auto& rowGroupIndices = passState_->passes[passState_->currentPass];

  // Byte ranges are flattened across sources; the source index map is only
  // needed once a reader spans multiple data sources.
  const auto columnChunkByteRanges =
      splitReader_
          ->all_column_chunks_byte_ranges(rowGroupIndices, readerOptions_)
          .first;

  nvtxRangePush("startColumnChunkFetch");
  passState_->fetch = fetchByteRangesAsync(
      dataSource_,
      columnChunkByteRanges,
      serializeIoRequests_,
      stream_,
      {get_temp_mr(), get_temp_mr()});
  nvtxRangePop();
}

void CudfSplitReader::setupChunkingForCurrentPass(
    rmm::device_async_resource_ref mr) {
  // No-op when the fetch was already started while preparing the split.
  startColumnChunkFetch();

  // Wait for all reads of the pass to complete.
  passState_->fetch.wait();

  splitReader_->setup_chunking_for_all_columns(
      chunkReadLimit_,
      passReadLimit_,
      passState_->passes[passState_->currentPass],
      passState_->fetch.data,
      readerOptions_,
      stream_,
      mr);

  passState_->isChunkingSetup = true;
}

void CudfSplitReader::setupFirstPassChunking() {
  if (noColumnsToRead_ or passState_ == nullptr or
      passState_->currentPass >= passState_->passes.size() or
      passState_->isChunkingSetup) {
    return;
  }
  if (fusedForCacheHit_) {
    // A split served from the fused FOR scan cache is read here instead, if
    // its decoded file fits: it stays allocated until the driver reads it.
    // Ahead-of-time decoding hides latency while device memory is ample and
    // gives way to the operators when it is not (allocated bytes plus the file
    // over half of what the process can use); the file is then decoded when
    // read.
    if (fusedForCacheRead_ and not fusedForCacheChunk_.has_value()) {
      const auto available = gpuCapabilities().availableMemoryBytes;
      const auto decodedBytes = static_cast<int64_t>(
          FusedForScanCache::instance().decodedSliceBytes(
              split_->filePath, readColumnNames_));
      if (available <= 0 or
          std::max<int64_t>(cudfAllocatedBytes(), 0) + decodedBytes <=
              available / 2) {
        fusedForCacheChunk_ =
            readFromFusedForCache(determineCudfMemoryResource());
      }
    }
    return;
  }
  setupChunkingForCurrentPass(determineCudfMemoryResource());
}

bool CudfSplitReader::fusedForCacheEligible() const {
  if (prependRowIndex_ or hasSplitSpecificPushdownFilter_ or
      baseReadOffset_ != 0 or
      splitRowCount_ !=
          std::accumulate(
              rowGroupNumRows_.begin(), rowGroupNumRows_.end(), std::size_t{0})) {
    return false;
  }
  for (size_t i = 0; i < readColumnNames_.size(); ++i) {
    if (not fileColumnNames_.contains(readColumnNames_[i]) or
        not readColumnTypes_[i]->isPrimitiveType()) {
      return false;
    }
  }
  return true;
}

void CudfSplitReader::fillFusedForCache() {
  nvtxRangePush("fillFusedForCache");
  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();
  auto options =
      cudf::io::parquet_reader_options::builder(
          cudf::io::source_info{dataSource_.get()})
          .columns(readColumnNames_)
          .use_pandas_metadata(
              cudfHiveConfig_->isUsePandasMetadataSession(sessionProperties))
          .use_arrow_schema(
              cudfHiveConfig_->isUseArrowSchemaSession(sessionProperties))
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemasSession(
                  sessionProperties))
          .timestamp_type(cudfHiveConfig_->timestampType())
          .case_sensitive_names(not caseInsensitiveColumnNames_)
          .build();
  const auto tempMr = get_temp_mr();
  // The file is read and encoded one slice at a time, so that filling takes
  // device memory for a slice: the reads of a file that does not fit would
  // otherwise fail every time and never fill the cache. The split covers the
  // whole file (fusedForCacheEligible()).
  const auto numFileRows = static_cast<int64_t>(splitRowCount_);
  const int64_t rowsPerSlice = FusedForScanCache::sliceRows();
  std::vector<FusedForScanCache::ColumnSlices> columns(readColumnNames_.size());
  int64_t numSlices = 0;
  for (int64_t sliceBegin = 0; numSlices == 0 or sliceBegin < numFileRows;
       sliceBegin += rowsPerSlice, ++numSlices) {
    if (numFileRows > rowsPerSlice) {
      options.set_skip_rows(sliceBegin);
      options.set_num_rows(static_cast<cudf::size_type>(
          std::min(rowsPerSlice, numFileRows - sliceBegin)));
    }
    auto result = cudf::io::read_parquet(options, stream_, tempMr);
    for (size_t i = 0; i < columns.size(); ++i) {
      columns[i].add(result.tbl->view().column(i), stream_, tempMr);
    }
  }
  auto& cache = FusedForScanCache::instance();
  // A column that is not cacheable does not keep the others out.
  for (size_t i = 0; i < columns.size(); ++i) {
    cache.insert(
        split_->filePath, readColumnNames_[i], std::move(columns[i]), stream_);
  }
  if (ioStats_) {
    ioStats_->addCounter("cudfFusedForCacheFills", RuntimeCounter(1));
    ioStats_->addCounter(
        "cudfFusedForCacheFillSlices", RuntimeCounter(numSlices));
  }
  nvtxRangePop();
}

std::optional<CudfSplitReader::TableChunk>
CudfSplitReader::readFromFusedForCache(rmm::device_async_resource_ref mr) {
  auto& cache = FusedForScanCache::instance();
  if (not fusedForCacheHit_) {
    fillFusedForCache();
    if (not cache.containsAll(split_->filePath, readColumnNames_)) {
      return std::nullopt;
    }
    // The next slices are read from what this fill cached.
    fusedForCacheHit_ = true;
  }
  nvtxRangePush("readFromFusedForCache");
  const auto tempMr = get_temp_mr();
  // The cached columns hold the whole file in slices; each call reads the
  // next slice.
  const auto numSlices =
      cache.numSlices(split_->filePath, readColumnNames_.front());
  const auto slice = fusedForCacheSlice_;

  // Keep the selected row groups.
  std::vector<cudf::size_type> bounds;
  std::vector<bool> selected(rowGroupNumRows_.size(), false);
  std::size_t numSelected = 0;
  for (const auto& pass : passState_->passes) {
    for (const auto& rowGroups : pass) {
      for (const auto rowGroup : rowGroups) {
        if (not selected[rowGroup]) {
          selected[rowGroup] = true;
          ++numSelected;
        }
      }
    }
  }
  if (numSelected < rowGroupNumRows_.size()) {
    cudf::size_type begin = 0;
    for (size_t rowGroup = 0; rowGroup < rowGroupNumRows_.size(); ++rowGroup) {
      const auto end =
          begin + static_cast<cudf::size_type>(rowGroupNumRows_[rowGroup]);
      if (selected[rowGroup]) {
        if (not bounds.empty() and bounds.back() == begin) {
          bounds.back() = end;
        } else {
          bounds.push_back(begin);
          bounds.push_back(end);
        }
      }
      begin = end;
    }
  }

  // Selecting rows writes the output from the decoded columns; without a
  // selection they are copied to it.
  const auto* filter = pushdownFilter();
  const bool selectsRows = filter != nullptr or not bounds.empty();
  std::vector<FusedForScanCache::DecodedColumn> decoded;
  std::vector<cudf::column_view> views;
  decoded.reserve(readColumnNames_.size());
  views.reserve(readColumnNames_.size());
  for (const auto& name : readColumnNames_) {
    decoded.push_back(cache.decode(
        split_->filePath,
        name,
        slice,
        stream_,
        selectsRows ? tempMr : mr,
        tempMr));
    views.push_back(decoded.back().view);
  }
  // The file rows of this slice, and the selected row ranges within it.
  const auto sliceBegin = fusedForCacheSliceRow_;
  const auto sliceEnd = sliceBegin + views.front().size();
  if (numSlices > 1 and not bounds.empty()) {
    std::vector<cudf::size_type> sliceBounds;
    for (size_t i = 0; i < bounds.size(); i += 2) {
      const auto begin = std::max(bounds[i], sliceBegin);
      const auto end = std::min(bounds[i + 1], sliceEnd);
      if (begin < end) {
        sliceBounds.push_back(begin - sliceBegin);
        sliceBounds.push_back(end - sliceBegin);
      }
    }
    if (sliceBounds.empty()) {
      // No selected row in this slice: an empty range keeps the columns.
      sliceBounds = {0, 0};
    }
    bounds = std::move(sliceBounds);
  }
  fusedForCacheSlice_ = slice + 1;
  fusedForCacheSliceRow_ = sliceEnd;
  std::unique_ptr<cudf::table> table;
  if (not selectsRows) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(decoded.size());
    for (auto& column : decoded) {
      columns.push_back(
          column.column != nullptr
              ? std::move(column.column)
              : std::make_unique<cudf::column>(column.view, stream_, mr));
    }
    table = std::make_unique<cudf::table>(std::move(columns));
  } else {
    cudf::table_view input{views};
    std::unique_ptr<cudf::table> rowGroups;
    if (not bounds.empty()) {
      auto pieces = cudf::slice(input, bounds, stream_);
      rowGroups =
          cudf::concatenate(pieces, stream_, filter != nullptr ? tempMr : mr);
      input = rowGroups->view();
    }
    if (filter != nullptr) {
      auto keep = cudf::compute_column(input, *filter, stream_, tempMr);
      table = cudf::apply_boolean_mask(input, keep->view(), stream_, mr);
    } else {
      table = std::move(rowGroups);
    }
  }
  decoded.clear();

  // After the last slice, nothing else of the split is read.
  if (fusedForCacheSlice_ == numSlices) {
    releaseCurrentPassData();
    passState_->currentPass = passState_->passes.size();
  }
  if (ioStats_) {
    ioStats_->addCounter("cudfFusedForCacheReads", RuntimeCounter(1));
  }
  nvtxRangePop();

  table = castDecimalColumnsToVeloxTypes(
      std::move(table), readColumnTypes_, 0, stream_, mr);
  const auto numRows = table->num_rows();
  return TableChunk{std::move(table), numRows};
}

void CudfSplitReader::releaseCurrentPassData() {
  // Queued or active reads can still write into the buffers being released.
  passState_->fetch.wait();
  passState_->fetch = {};
}

void CudfSplitReader::resetSplit() {
  if (passState_ != nullptr) {
    releaseCurrentPassData();
  }
  splitReader_.reset();
  passState_.reset();
  dataSource_.reset();
  fileMetaData_.clear();
  pushdownFilterExpr_ = subfieldFilterAst_;
  hasSplitSpecificPushdownFilter_ = false;
  fileColumnNames_.clear();
  baseReadOffset_ = 0;
  splitRowCount_ = 0;
  readAllFileColumns_ = false;
  noColumnsToRead_ = false;
  fusedForCacheRead_ = false;
  fusedForCacheHit_ = false;
  fusedForCacheChunk_.reset();
  fusedForCacheSlice_ = 0;
  fusedForCacheSliceRow_ = 0;
  rowGroupNumRows_.clear();
}

cudf::ast::expression const* CudfSplitReader::pushdownFilter() const {
  return pushdownFilterExpr_;
}

const cudf::ast::expression* CudfSplitReader::subfieldFilterAst() const {
  return subfieldFilterAst_;
}

bool CudfSplitReader::isSplitSkipped() const {
  return false;
}

bool CudfSplitReader::hasSplitSpecificPushdownFilter() const {
  return hasSplitSpecificPushdownFilter_;
}

namespace {

std::unique_ptr<cudf::io::datasource> makeKvikioDataSource(
    const CudfHiveConnectorSplit& split,
    const std::shared_ptr<IoStats>& ioStats) {
  const auto knownSize = knownKvikioFileSize(split);
  // Open the whole file (offset 0, no size estimate), not this split's range,
  // because the reader also reads the footer at the end of the file. With the
  // file size known, opening sends no HEAD request to S3.
  auto source = cudf::io::datasource::create(
      split.filePath, /*offset=*/0, /*max_size_estimate=*/0, knownSize);
  if (ioStats) {
    ioStats->addCounter(
        knownSize.has_value() ? "cudfKvikioKnownSizeOpens"
                              : "cudfKvikioUnknownSizeOpens",
        RuntimeCounter(1));
  }
  return maybeCacheKvikioDataSource(
      std::move(source),
      split.filePath,
      velox::cache::AsyncDataCache::getInstance(),
      split.cacheable,
      ioStats);
}

} // namespace

void CudfSplitReader::setupCudfDataSource() {
  if (dataSource_) {
    return;
  }

  const auto useBufferedInput = cudfHiveConfig_->useBufferedInputSession(
      connectorQueryCtx_->sessionProperties());

  VELOX_CHECK(
      not isAbfsPath(split_->filePath) or useBufferedInput,
      "ABFS blobs require buffered input data source. "
      "Set the session property '{}' (or connector property '{}') to 'true'. "
      "Blob Path: {}.",
      CudfHiveConfig::kUseBufferedInputSession,
      CudfHiveConfig::kUseBufferedInput,
      split_->filePath);

  // Use KvikIO data source if we don't want to use the BufferedInput source
  if (not useBufferedInput) {
    VLOG(1) << fmt::format(
        "Using KvikIO data source for file: {}", split_->filePath);
    dataSource_ = makeKvikioDataSource(*split_, ioStats_);
    return;
  }

  auto fileHandleCachePtr = FileHandleCachedPtr{};
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  } catch (const VeloxRuntimeError& e) {
    // ABFS blobs can not fall back to KvikIO. Throw the original error.
    if (isAbfsPath(split_->filePath)) {
      VELOX_USER_FAIL(
          "Failed to generate file handle cache for ABFS blob. Ensure "
          "registerAbfsFileSystem() and registerAzureClientProvider() have "
          "been called and the connector config provides Azure credentials. "
          "Blob path: {}. Error: {}.",
          split_->filePath,
          e.what());
    }

    LOG(WARNING) << fmt::format(
        "Failed to generate file handle cache for file. Falling back to KvikIO. Path: {}",
        split_->filePath);
    dataSource_ = makeKvikioDataSource(*split_, ioStats_);
    return;
  }

  // Here we keep adding new entries to CacheTTLController when new
  // fileHandles are generated, if CacheTTLController was created. Creator of
  // CacheTTLController needs to make sure a size control strategy was
  // available such as removing aged out entries.
  if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
    cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
  }

  auto bufferedInput =
      velox::connector::hive::BufferedInputBuilder::getInstance()->create(
          *fileHandleCachePtr,
          baseReaderOpts_,
          connectorQueryCtx_,
          ioStatistics_,
          ioStats_,
          executor_);
  if (not bufferedInput) {
    // ABFS blobs can not fall back to KvikIO
    if (isAbfsPath(split_->filePath)) {
      VELOX_USER_FAIL(
          "Failed to create buffered input data source for the ABFS blob. Ensure that the registered "
          "BufferedInputBuilder is ABFS-aware. Blob path: {}.",
          split_->filePath);
    }

    LOG(WARNING) << fmt::format(
        "Failed to create buffered input data source for file. Falling back to the KvikIO. Path: {}",
        split_->filePath);
    dataSource_ = makeKvikioDataSource(*split_, ioStats_);
    return;
  }
  dataSource_ =
      std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
}

void CudfSplitReader::setupReaderOptions() {
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder()
          .use_pandas_metadata(
              cudfHiveConfig_->isUsePandasMetadataSession(sessionProperties))
          .use_arrow_schema(
              cudfHiveConfig_->isUseArrowSchemaSession(sessionProperties))
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemasSession(
                  sessionProperties))
          .timestamp_type(cudfHiveConfig_->timestampType())
          // cuDF currently only folds ASCII letters, unlike the CPU reader's
          // UTF-8 folding, so non-ASCII names (rare case) that differ in case
          // do not match.
          .case_sensitive_names(not caseInsensitiveColumnNames_)
          .build();

  // Set skip_bytes and num_bytes if available
  if (split_->start != 0) {
    readerOptions_.set_skip_bytes(split_->start);
  }
  if (split_->size() != std::numeric_limits<uint64_t>::max()) {
    readerOptions_.set_num_bytes(split_->size());
  }

  if (auto* filter = pushdownFilter(); filter != nullptr) {
    readerOptions_.set_filter(*filter);
  }

  // Skip column selection when reading every file column in file order
  if (not readAllFileColumns_) {
    readerOptions_.set_column_names(readColumnNames_);
  }

  if (prependRowIndex_) {
    readerOptions_.enable_prepend_row_index_column(true);
  }
}

rmm::device_async_resource_ref CudfSplitReader::determineCudfMemoryResource()
    const {
  return get_output_mr();
}

void CudfSplitReader::fileMetaDatas() {
  if (not fileMetaData_.empty()) {
    return;
  }

  // Setup the datasource
  setupCudfDataSource();

  // Check that the datasource is set up
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  // Wrap the existing datasource without transferring ownership.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));
  fileMetaData_ = cudf::io::read_parquet_footers(sources);
  VELOX_CHECK_GE(
      fileMetaData_.size(),
      1,
      "CudfSplitReader failed to read any parquet metadatas");

  if (pushdownFilterBuilder_) {
    VELOX_CHECK_EQ(
        fileMetaData_.size(),
        1,
        "Split-specific pushdown filters require exactly one Parquet metadata");
    pushdownFilterExpr_ = pushdownFilterBuilder_(fileMetaData_.front());
    VELOX_CHECK_NOT_NULL(
        pushdownFilterExpr_,
        "Split-specific pushdown filter builder must return an expression");
    hasSplitSpecificPushdownFilter_ = true;
  }
}

void CudfSplitReader::createCudfReader() {
  // Read file metadatas
  fileMetaDatas();

  // Check if we are reading all columns in order
  const auto& schema = fileMetaData_.front().schema;
  readAllFileColumns_ = std::ranges::equal(
      schema.front().children_idx, readColumnNames_, {}, [&](auto childIdx) {
        const auto& name = schema[childIdx].name;
        return caseInsensitiveColumnNames_
            ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
            : name;
      });

  // Setup reader options
  setupReaderOptions();

  // A reader spanning multiple sources also needs the byte range source index
  // map to fetch column chunks from the matching data source.
  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "cuDF parquet reader supports a single parquet metadata per split");

  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();
  chunkReadLimit_ =
      cudfHiveConfig_->maxChunkReadLimitSession(sessionProperties);
  passReadLimit_ = cudfHiveConfig_->maxPassReadLimitSession(sessionProperties);
  serializeIoRequests_ =
      cudfHiveConfig_->serializeIoRequestsSession(sessionProperties);

  // Create a hybrid scan reader over all sources of the split
  splitReader_ = std::make_unique<CudfParquetReader>(
      std::move(fileMetaData_), readerOptions_);

  setupPageIndexes();

  passState_ = std::make_unique<RowGroupPassState>();
  passState_->passes = selectRowGroupPasses();
}

void CudfSplitReader::setupPageIndexes() {
  const auto pageIndexByteRanges = splitReader_->page_index_byte_ranges();

  // No page index bytes
  if (std::ranges::all_of(pageIndexByteRanges, [](const auto& byteRange) {
        return byteRange.is_empty();
      })) {
    return;
  }

  [[maybe_unused]] auto [pageIndexBuffers, pageIndexData] =
      fetchPageIndexes(dataSource_, pageIndexByteRanges);
  splitReader_->setup_page_indexes(pageIndexData);
}

CudfSplitReader::RowGroupPasses CudfSplitReader::selectRowGroupPasses() {
  auto rowGroupIndices = splitReader_->all_row_groups(readerOptions_);

  // Filter row groups using row group byte ranges
  if (readerOptions_.get_skip_bytes() > 0 or
      readerOptions_.get_num_bytes().has_value()) {
    rowGroupIndices = splitReader_->filter_row_groups_with_byte_range(
        rowGroupIndices, readerOptions_);
  }

  // Filter row groups using column chunk statistics
  if (readerOptions_.get_filter().has_value()) {
    rowGroupIndices = splitReader_->filter_row_groups_with_stats(
        rowGroupIndices, readerOptions_, stream_);
  }

  const auto numRowGroups = std::accumulate(
      rowGroupIndices.begin(),
      rowGroupIndices.end(),
      std::size_t{0},
      [](auto sum, const auto& sourceRowGroups) {
        return sum + sourceRowGroups.size();
      });

  // No row groups to read.
  if (numRowGroups == 0) {
    return {};
  }

  // Construct row group passes using all (or selected) columns for pass memory
  // estimation.
  return splitReader_->construct_row_group_passes(
      cudf::io::parquet::experimental::read_columns_mode::ALL_COLUMNS,
      rowGroupIndices,
      passReadLimit_,
      readerOptions_);
}

void CudfSplitReader::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStatistics->incTotalScanTimeNs(elapsedNs);

  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
