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

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ScanSpec.h"

namespace facebook::velox::cudf_velox::connector::hive {

/// Reads a Nimble split for the cuDF Hive data source. The stripes of the
/// split are decoded on the CPU by the selective Nimble reader and copied to
/// the device, where the subfield filters are applied.
class CudfNimbleSplitReader : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  /// A Nimble file has no Parquet footer, so there is never any metadata.
  const std::vector<cudf::io::parquet::FileMetaData>& fileMetaDatas() override;

  void resetSplit() override;

 protected:
  void prepareSplitInternal(dwio::common::RuntimeStats& runtimeStats) override;

  std::optional<TableChunk> readNextChunk() override;

 private:
  // Opens the split's file through the file handle cache and the registered
  // BufferedInput builder, so reads go through the AsyncDataCache.
  std::unique_ptr<dwio::common::BufferedInput> makeBufferedInput(
      const dwio::common::ReaderOptions& readerOptions);

  std::shared_ptr<common::ScanSpec> scanSpec_;
  std::unique_ptr<dwio::common::Reader> reader_;
  std::unique_ptr<dwio::common::RowReader> rowReader_;
  VectorPtr batch_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
