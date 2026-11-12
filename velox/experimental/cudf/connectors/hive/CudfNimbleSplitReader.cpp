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

#include "velox/connectors/hive/FileHandle.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include <nvtx3/nvtx3.hpp>

#include <cstdlib>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

folly::Executor* readExecutor() {
  static auto* executor = new folly::CPUThreadPoolExecutor(16);
  return executor;
}

} // namespace

CudfNimbleSplitReader::~CudfNimbleSplitReader() {
  dropPending();
}

const std::vector<cudf::io::parquet::FileMetaData>&
CudfNimbleSplitReader::fileMetaDatas() {
  return fileMetaDatas_;
}

void CudfNimbleSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  nvtx3::scoped_range range{"CudfNimbleSplitReader::prepareSplit"};
  const auto fileHandleKey = FileHandleKey{
      .filename = split_->filePath,
      .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
  auto fileProperties = split_->properties.value_or(FileProperties{});
  fileProperties.fileSize = split_->fileSize();
  auto fileHandle = fileHandleFactory_->generate(
      fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
  VELOX_CHECK_NOT_NULL(fileHandle.get());
  readFile_ = fileHandle->file;

  facebook::nimble::TabletReader::Options tabletOptions;
  tabletOptions.ioOptions.emplace(connectorQueryCtx_->memoryPool())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
  tablet_ = facebook::nimble::TabletReader::create(
      readFile_, connectorQueryCtx_->memoryPool(), tabletOptions);
  compactRowCount_ = tablet_->properties().compactRowCountEncoding();

  auto schemaSection = tablet_->loadOptionalSection(
      std::string(facebook::nimble::kSchemaSection));
  VELOX_CHECK(schemaSection.has_value(), "Nimble file has no schema");
  const auto schema = facebook::nimble::SchemaDeserializer::deserialize(
      schemaSection->content().data());
  const auto& root = schema->asRow();

  streamIds_.clear();
  cudfTypes_.clear();
  for (size_t i = 0; i < readColumnNames_.size(); ++i) {
    const auto child = root.findChild(readColumnNames_[i]);
    VELOX_CHECK(
        child.has_value(),
        "Column {} not found in Nimble file {}",
        readColumnNames_[i],
        split_->filePath);
    const auto& type = root.childAt(*child);
    VELOX_CHECK(
        type->kind() == facebook::nimble::Kind::Scalar,
        "GPU Nimble reader supports only scalar columns, not {}",
        readColumnNames_[i]);
    streamIds_.push_back(type->asScalar().scalarDescriptor().offset());
    cudfTypes_.push_back(veloxToCudfDataType(readColumnTypes_[i]));
  }
  noColumnsToRead_ = readColumnNames_.empty();

  stripes_.clear();
  nextStripe_ = 0;
  const uint64_t splitEnd = split_->start + split_->length;
  for (uint32_t stripe = 0; stripe < tablet_->stripeCount(); ++stripe) {
    const auto offset = tablet_->stripeOffset(stripe);
    if (offset >= split_->start && offset < splitEnd) {
      stripes_.push_back(stripe);
    }
  }
}

std::optional<CudfNimbleSplitReader::PendingBatch>
CudfNimbleSplitReader::startBatch() {
  if (nextStripe_ >= stripes_.size()) {
    return std::nullopt;
  }
  // Stripes are batched up to a stored-byte budget so that one decode sees
  // many compressed frames.
  static const size_t kBatchBytes = [] {
    const char* mb = std::getenv("NIMBLE_GPU_BATCH_MB");
    return size_t{mb != nullptr ? std::stoul(mb) : 512} << 20;
  }();
  // Well-compressed columns decode to many times their stored size, so the
  // decoded size is bounded too, like Parquet's chunk read limit.
  static const size_t kDecodedBytes = [] {
    const char* mb = std::getenv("NIMBLE_GPU_BATCH_DECODED_MB");
    return size_t{mb != nullptr ? std::stoul(mb) : 2048} << 20;
  }();
  constexpr int64_t kBatchRows = int64_t{1} << 28;
  struct Read {
    uint64_t fileOffset;
    uint64_t size;
  };
  std::vector<Read> reads;
  std::vector<NimbleStripeStreams> batch;
  size_t batchBytes = 0;
  size_t decodedBytes = 0;
  int64_t batchRows = 0;
  while (nextStripe_ < stripes_.size() && batchBytes < kBatchBytes) {
    const auto stripe = stripes_[nextStripe_];
    const auto numRows = static_cast<int64_t>(tablet_->stripeRowCount(stripe));
    const auto stripeId = tablet_->stripeIdentifier(stripe);
    size_t stripeDecoded = 0;
    for (size_t i = 0; i < streamIds_.size(); ++i) {
      stripeDecoded += cudfTypes_[i].id() == cudf::type_id::STRING
          ? numRows * sizeof(cudf::size_type) +
              tablet_->streamLocation(stripeId, streamIds_[i]).size
          : numRows * cudf::size_of(cudfTypes_[i]);
    }
    if (!batch.empty() &&
        (batchRows + numRows > kBatchRows ||
         decodedBytes + stripeDecoded > kDecodedBytes)) {
      break;
    }
    ++nextStripe_;
    const auto stripeOffset = tablet_->stripeOffset(stripe);
    batch.emplace_back().numRows = static_cast<cudf::size_type>(numRows);
    for (size_t i = 0; i < streamIds_.size(); ++i) {
      const auto location = tablet_->streamLocation(stripeId, streamIds_[i]);
      reads.push_back({stripeOffset + location.offset, location.size});
      batchBytes += location.size;
    }
    batchRows += numRows;
    decodedBytes += stripeDecoded;
  }

  // The streams are read straight into the pinned buffer that goes to the
  // device, in pieces spread over the read threads.
  PendingBatch pending;
  pending.pinned =
      std::make_unique<NimbleGpuDecoder::PinnedBuffer>(batchBytes);
  pending.bytes = batchBytes;
  auto* host = pending.pinned->data();
  constexpr uint64_t kPieceBytes = uint64_t{4} << 20;
  std::vector<folly::SemiFuture<folly::Unit>> pieces;
  size_t offset = 0;
  size_t next = 0;
  for (auto& entry : batch) {
    for (size_t i = 0; i < streamIds_.size(); ++i) {
      const auto& read = reads[next++];
      for (uint64_t done = 0; done < read.size; done += kPieceBytes) {
        const auto size = std::min(kPieceBytes, read.size - done);
        auto* out = host + offset + done;
        const auto fileOffset = read.fileOffset + done;
        pieces.push_back(
            folly::via(readExecutor(), [this, fileOffset, size, out] {
              readFile_->pread(fileOffset, size, out);
            }).semi());
      }
      entry.streams.emplace_back(host + offset, read.size);
      offset += read.size;
    }
  }
  pending.stripes = std::move(batch);
  pending.reads = folly::collect(std::move(pieces));
  return pending;
}

void CudfNimbleSplitReader::dropPending() {
  if (pending_) {
    pending_->reads.wait();
    pending_.reset();
  }
}

std::optional<CudfSplitReader::TableChunk>
CudfNimbleSplitReader::readNextChunk() {
  VELOX_CHECK_NOT_NULL(tablet_, "Nimble tablet not open");
  if (noColumnsToRead_) {
    if (nextStripe_ >= stripes_.size()) {
      return std::nullopt;
    }
    const auto stripe = stripes_[nextStripe_++];
    return TableChunk{
        std::make_unique<cudf::table>(
            std::vector<std::unique_ptr<cudf::column>>{}),
        static_cast<cudf::size_type>(tablet_->stripeRowCount(stripe))};
  }

  if (!pending_) {
    pending_ = startBatch();
  }
  if (!pending_) {
    return std::nullopt;
  }
  auto current = std::move(*pending_);
  pending_ = startBatch();
  {
    nvtx3::scoped_range range{"CudfNimbleSplitReader::read"};
    std::move(current.reads).get();
  }
  if (ioStatistics_) {
    ioStatistics_->incRawBytesRead(current.bytes);
  }

  const auto* filter = subfieldFilterAst();
  auto columns = decoder_.decode(
      current.stripes,
      std::span<const uint8_t>{current.pinned->data(), current.bytes},
      cudfTypes_,
      compactRowCount_,
      stream_,
      filter != nullptr ? get_temp_mr() : get_output_mr());
  auto table = std::make_unique<cudf::table>(std::move(columns));

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
  dropPending();
  tablet_.reset();
  readFile_.reset();
  streamIds_.clear();
  cudfTypes_.clear();
  stripes_.clear();
  nextStripe_ = 0;
  CudfSplitReader::resetSplit();
}

} // namespace facebook::velox::cudf_velox::connector::hive
