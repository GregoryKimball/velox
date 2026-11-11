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
#include "velox/experimental/cudf/connectors/hive/CudfNimbleSplitReader.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/connectors/common/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/dwio/common/ReaderFactory.h"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Rows requested from the CPU reader per chunk. A read stops early at the end
// of the split.
constexpr uint64_t kNimbleReadBatchRows = 1 << 20;

} // namespace

const std::vector<cudf::io::parquet::FileMetaData>&
CudfNimbleSplitReader::fileMetaDatas() {
  return fileMetaDatas_;
}

std::unique_ptr<dwio::common::BufferedInput>
CudfNimbleSplitReader::makeBufferedInput(
    const dwio::common::ReaderOptions& readerOptions) {
  const auto fileHandleKey = FileHandleKey{
      .filename = split_->filePath,
      .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
  auto fileProperties = split_->properties.value_or(FileProperties{});
  fileProperties.fileSize = split_->fileSize();
  auto fileHandle = fileHandleFactory_->generate(
      fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
  VELOX_CHECK_NOT_NULL(fileHandle.get());

  auto input = velox::connector::BufferedInputBuilder::getInstance()->create(
      *fileHandle,
      readerOptions,
      connectorQueryCtx_,
      ioStatistics_,
      ioStats_,
      executor_);
  if (input == nullptr) {
    input = std::make_unique<dwio::common::BufferedInput>(
        fileHandle->file, readerOptions.memoryPool());
  }
  return input;
}

void CudfNimbleSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  nvtx3::scoped_range range{"CudfNimbleSplitReader::prepareSplit"};
  auto* pool = connectorQueryCtx_->memoryPool();

  scanSpec_ = std::make_shared<common::ScanSpec>("<root>");
  for (size_t i = 0; i < readColumnNames_.size(); ++i) {
    scanSpec_->addFieldRecursively(
        readColumnNames_[i], *readColumnTypes_[i], i);
  }
  noColumnsToRead_ = readColumnNames_.empty();

  dwio::common::ReaderOptions readerOptions(pool);
  readerOptions.setFileFormat(dwio::common::FileFormat::NIMBLE);
  readerOptions.setSelectiveNimbleReaderEnabled(true);
  readerOptions.setScanSpec(scanSpec_);
  readerOptions.setDataIoStats(ioStatistics_);
  readerOptions.setMetadataIoStats(ioStatistics_);

  reader_ = dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE)
                ->createReader(makeBufferedInput(readerOptions), readerOptions);

  dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec_);
  auto requestedType = ROW(
      std::vector<std::string>(readColumnNames_),
      std::vector<TypePtr>(readColumnTypes_));
  rowReaderOptions.setRequestedType(requestedType);
  rowReaderOptions.range(split_->start, split_->length);
  rowReader_ = reader_->createRowReader(rowReaderOptions);
  batch_ = BaseVector::create(requestedType, 0, pool);
}

std::optional<CudfSplitReader::TableChunk>
CudfNimbleSplitReader::readNextChunk() {
  VELOX_CHECK_NOT_NULL(rowReader_, "Nimble row reader not present");

  uint64_t numRows{0};
  {
    nvtx3::scoped_range range{"CudfNimbleSplitReader::cpuDecode"};
    numRows = rowReader_->next(kNimbleReadBatchRows, batch_);
  }
  if (numRows == 0) {
    return std::nullopt;
  }

  auto rowVector = std::dynamic_pointer_cast<RowVector>(batch_);
  VELOX_CHECK_NOT_NULL(rowVector, "Nimble reader returned a non-row batch");
  if (noColumnsToRead_) {
    return TableChunk{
        std::make_unique<cudf::table>(
            std::vector<std::unique_ptr<cudf::column>>{}),
        rowVector->size()};
  }
  for (auto& child : rowVector->children()) {
    child = BaseVector::loadedVectorShared(child);
  }

  const auto* filter = subfieldFilterAst();
  std::unique_ptr<cudf::table> table;
  {
    nvtx3::scoped_range range{"CudfNimbleSplitReader::toDevice"};
    table = with_arrow::toCudfTable(
        rowVector,
        connectorQueryCtx_->memoryPool(),
        stream_,
        filter != nullptr ? get_temp_mr() : get_output_mr());
  }

  if (filter != nullptr) {
    nvtx3::scoped_range range{"CudfNimbleSplitReader::filter"};
    auto mask =
        cudf::compute_column(table->view(), *filter, stream_, get_temp_mr());
    table = cudf::apply_retention_mask(
        table->view(), mask->view(), stream_, get_output_mr());
  }
  const auto tableRows = table->num_rows();
  return TableChunk{std::move(table), tableRows};
}

void CudfNimbleSplitReader::resetSplit() {
  rowReader_.reset();
  reader_.reset();
  scanSpec_.reset();
  batch_.reset();
  CudfSplitReader::resetSplit();
}

} // namespace facebook::velox::cudf_velox::connector::hive
