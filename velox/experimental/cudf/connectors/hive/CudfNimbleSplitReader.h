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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/NimbleGpuDecoder.h"

#include "velox/common/file/File.h"

#include <folly/futures/Future.h>

#include <deque>
#include <optional>

namespace facebook::nimble {
class TabletReader;
class Type;
} // namespace facebook::nimble

namespace facebook::velox::cudf_velox::connector::hive {

/// Reads a Nimble split for the cuDF Hive data source, one stripe per chunk.
/// The host reads the stripe's stream bytes and walks their encoding headers;
/// NimbleGpuDecoder decompresses and decodes the payloads on the device, where
/// the subfield filters are then applied. Only scalar columns are supported.
class CudfNimbleSplitReader : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  ~CudfNimbleSplitReader() override;

  void resetSplit() override;

  /// Starts the reads of the first batches of the split, so that a split
  /// prepared in the background has its stream bytes in flight before a
  /// driver adopts it.
  void startColumnChunkFetch() override;

 protected:
  void prepareSplitInternal(dwio::common::RuntimeStats& runtimeStats) override;

  std::optional<TableChunk> readNextChunk() override;

  bool isSplitSkipped() const override {
    return splitSkipped_;
  }

 private:
  // Returns false when the file-level column statistics show that no row of
  // the file passes a subfield filter.
  bool fileStatsMayMatch(
      const std::shared_ptr<const facebook::nimble::Type>& schema) const;

  // A batch of stripes whose stream bytes are being read into `pinned`.
  struct PendingBatch {
    std::unique_ptr<NimbleGpuDecoder::PinnedBuffer> pinned;
    std::vector<NimbleStripeStreams> stripes;
    size_t bytes{0};
    // Ready once every read has finished, failed or not, so that `pinned` can
    // be freed.
    folly::SemiFuture<std::vector<folly::Try<folly::Unit>>> reads{
        folly::makeSemiFuture(std::vector<folly::Try<folly::Unit>>{})};
  };

  // Opens the split's file through a BufferedInput over the AsyncDataCache,
  // when the process has one.
  void openBufferedReadFile();

  // Starts reading the next batch, if any stripes are left.
  std::optional<PendingBatch> startBatch();

  // Starts batches until the prefetch depth is in flight or no stripes are
  // left.
  void fillPending();

  // Waits for the reads of every pending batch, ignoring their errors, and
  // frees them.
  void dropPending();

  // Reads through KvikIO and its async data cache when the connector uses
  // KvikIO, like the Parquet reader, and otherwise through a BufferedInput
  // over the AsyncDataCache and its SSD tier.
  std::shared_ptr<ReadFile> readFile_;
  std::shared_ptr<facebook::nimble::TabletReader> tablet_;
  // The stream id of each read column.
  std::vector<uint32_t> streamIds_;
  std::vector<cudf::data_type> cudfTypes_;
  // Stripes that start inside the split's byte range, in file order.
  std::vector<uint32_t> stripes_;
  size_t nextStripe_{0};
  bool compactRowCount_{false};
  bool splitSkipped_{false};
  NimbleGpuDecoder decoder_;
  // Read, in file order, while the previous batch is decoded and consumed.
  std::deque<PendingBatch> pending_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
