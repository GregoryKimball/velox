/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Rewrites a directory of Parquet tables as Nimble. Each table is a
// subdirectory of --input_dir, and each Parquet file becomes one Nimble file of
// the same name, with a .nimble extension, in the same table subdirectory of
// --output_dir. This is the layout the TPC-H and TPC-DS benchmarks read.

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

DEFINE_string(input_dir, "", "Directory of Parquet tables, one per subdirectory");
DEFINE_string(output_dir, "", "Directory to write the Nimble tables to");
DEFINE_uint64(
    stripe_raw_size_mb,
    256,
    "Raw (decoded) bytes buffered before a stripe is written");
DEFINE_string(
    compression,
    "Zstd",
    "Compression of encoded streams: Uncompressed, Zstd or Lz4");
DEFINE_uint32(zstd_level, 3, "Zstd compression level");
DEFINE_uint64(
    max_chunk_raw_kb,
    0,
    "Cap on the raw size of a stream chunk, enforced eagerly; 0 keeps the "
    "writer default. Small chunks give GPU decompression more frames per "
    "stripe");
DEFINE_uint64(
    min_chunk_raw_kb,
    0,
    "Raw size below which a stream is not chunked; 0 keeps the writer default");
DEFINE_bool(
    exact_bits,
    false,
    "Pack FixedBitWidth values at their exact bit width instead of rounding "
    "the width up to whole bytes");
DEFINE_string(
    read_factors,
    "",
    "Candidate encodings and their read factors for encoding selection, as "
    "<Encoding>=<factor>;...; empty keeps the writer default");
DEFINE_uint64(read_batch_rows, 1 << 16, "Rows per Parquet read batch");
DEFINE_int32(threads, 8, "Files converted in parallel");

namespace fs = std::filesystem;
using namespace facebook::velox;

namespace {

struct FileTask {
  fs::path input;
  fs::path output;
};

std::vector<FileTask> listFiles(const fs::path& inputDir, const fs::path& outputDir) {
  std::vector<FileTask> tasks;
  for (const auto& table : fs::directory_iterator(inputDir)) {
    if (!table.is_directory()) {
      continue;
    }
    for (const auto& file : fs::directory_iterator(table.path())) {
      const auto name = file.path().filename().string();
      if (!file.is_regular_file() || name.empty() || name[0] == '.' ||
          name[0] == '_') {
        continue;
      }
      auto output = outputDir / table.path().filename() / file.path().stem();
      output += ".nimble";
      tasks.push_back({file.path(), std::move(output)});
    }
  }
  return tasks;
}

uint64_t convertFile(const FileTask& task, memory::MemoryPool& rootPool) {
  auto leafPool = rootPool.addLeafChild(task.input.filename().string());

  dwio::common::ReaderOptions readerOptions(leafPool.get());
  readerOptions.setFileFormat(dwio::common::FileFormat::PARQUET);
  std::shared_ptr<ReadFile> readFile =
      filesystems::getFileSystem(task.input.string(), nullptr)
          ->openFileForRead(task.input.string());
  auto reader =
      dwio::common::getReaderFactory(dwio::common::FileFormat::PARQUET)
          ->createReader(
              std::make_unique<dwio::common::BufferedInput>(
                  readFile, *leafPool),
              readerOptions);
  const auto rowType = reader->rowType();

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addAllChildFields(*rowType);
  dwio::common::RowReaderOptions rowReaderOptions;
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setRequestedType(rowType);
  auto rowReader = reader->createRowReader(rowReaderOptions);

  facebook::nimble::WriterOptions writerOptions;
  writerOptions.compressionOptions.compressionType =
      facebook::nimble::toCompressionType(FLAGS_compression);
  writerOptions.compressionOptions.zstdCompressionLevel = FLAGS_zstd_level;
  // The default encoding selection policy carries its own compression
  // options, so it is rebuilt with these.
  writerOptions.fixedBitWidthUseExactBits = FLAGS_exact_bits;
  using facebook::nimble::ManualEncodingSelectionPolicyFactory;
  writerOptions.encodingSelectionPolicyCreator =
      [factory = ManualEncodingSelectionPolicyFactory{
           FLAGS_read_factors.empty()
               ? ManualEncodingSelectionPolicyFactory::
                     defaultEncodingReadFactors()
               : ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
                     FLAGS_read_factors),
           writerOptions.compressionOptions}](
          facebook::nimble::DataType dataType) {
        return factory.createPolicy(dataType);
      };
  if (FLAGS_max_chunk_raw_kb > 0) {
    writerOptions.eagerChunking = true;
    writerOptions.maxStreamChunkRawSize = FLAGS_max_chunk_raw_kb << 10;
  }
  if (FLAGS_min_chunk_raw_kb > 0) {
    writerOptions.minStreamChunkRawSize = FLAGS_min_chunk_raw_kb << 10;
  }
  const uint64_t stripeRawSize = FLAGS_stripe_raw_size_mb << 20;
  // stripeRawSize counts only data not yet encoded, which chunking drains, so
  // the chunks already encoded in the stripe count toward the cut as well.
  writerOptions.flushPolicyFactory = [stripeRawSize]() {
    return std::make_unique<facebook::nimble::LambdaFlushPolicy>(
        [stripeRawSize](const facebook::nimble::StripeProgress& progress) {
          return progress.stripeRawSize + progress.stripeEncodedLogicalSize >=
              stripeRawSize;
        });
  };

  fs::create_directories(task.output.parent_path());
  auto writeFile = filesystems::getFileSystem(task.output.string(), nullptr)
                       ->openFileForWrite(task.output.string());
  facebook::nimble::Writer writer(
      rowType, std::move(writeFile), rootPool, std::move(writerOptions));

  uint64_t numRows{0};
  VectorPtr batch = BaseVector::create(rowType, 0, leafPool.get());
  while (rowReader->next(FLAGS_read_batch_rows, batch) > 0) {
    auto rowVector = std::dynamic_pointer_cast<RowVector>(batch);
    for (auto& child : rowVector->children()) {
      child = BaseVector::loadedVectorShared(child);
    }
    writer.write(rowVector);
    numRows += rowVector->size();
  }
  writer.close();
  return numRows;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  CHECK(!FLAGS_input_dir.empty()) << "--input_dir is required";
  CHECK(!FLAGS_output_dir.empty()) << "--output_dir is required";

  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  filesystems::registerLocalFileSystem();
  parquet::registerParquetReaderFactory();

  const auto tasks = listFiles(FLAGS_input_dir, FLAGS_output_dir);
  LOG(INFO) << "Converting " << tasks.size() << " files";

  std::atomic<uint64_t> totalRows{0};
  std::atomic<int32_t> failures{0};
  {
    folly::CPUThreadPoolExecutor executor(FLAGS_threads);
    for (const auto& task : tasks) {
      executor.add([&task, &totalRows, &failures]() {
        try {
          auto rootPool = memory::memoryManager()->addRootPool(
              task.output.string());
          const auto rows = convertFile(task, *rootPool);
          totalRows += rows;
          LOG(INFO) << task.input << " -> " << task.output << ": " << rows
                    << " rows";
        } catch (const std::exception& e) {
          ++failures;
          LOG(ERROR) << "Failed to convert " << task.input << ": " << e.what();
        }
      });
    }
    executor.join();
  }

  LOG(INFO) << "Converted " << tasks.size() - failures << " of " << tasks.size()
            << " files, " << totalRows << " rows";
  return failures == 0 ? 0 : 1;
}
