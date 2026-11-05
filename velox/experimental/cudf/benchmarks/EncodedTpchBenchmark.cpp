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

// Runs the TPC-H plans of TpchQueryBuilder with every table scan reading the
// EncodedColumnCache instead of Parquet. The tables are loaded from
// --data_path and encoded on the GPU before any query runs. With
// --encoded_device=gpu the plans run on cuDF and the tables stay in device
// memory, or with --encoded_memory=pinned_host move to pinned host memory;
// with --encoded_device=cpu they are copied to host memory, the device copies
// are freed, and the plans run on CPU Velox.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <fstream>

DECLARE_string(data_path);

DEFINE_string(
    encoded_device,
    "gpu",
    "Where the encoded tables live and the plans run: gpu or cpu.");

DEFINE_string(
    encoded_memory,
    "device",
    "With --encoded_device=gpu, where the encoded tables are kept: device, or "
    "pinned_host to have the decode kernels read them from host memory.");

DEFINE_int32(
    encoded_segment_rows,
    1 << 20,
    "Rows per encoded segment, the unit of min/max pruning.");

DEFINE_int32(
    encoded_segments_per_split,
    4,
    "Segments per split; each split is decoded and filtered as one batch.");

DEFINE_bool(
    encoded_cluster,
    true,
    "Sort lineitem by l_shipdate and orders by o_orderdate when loading.");

DEFINE_string(
    encoded_size_report,
    "",
    "If set, writes per-column codec and encoded bytes as CSV to this path.");

DEFINE_uint64(
    cudf_local_exchange_buffer_size,
    1UL << 30,
    "Maximum buffered bytes per local exchange before applying backpressure.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value.");

using namespace facebook::velox;
namespace encoded = facebook::velox::cudf_velox::connector::encoded;

namespace {

const std::vector<std::pair<std::string, std::string>> kTables = {
    {"lineitem", "l_shipdate"},
    {"orders", "o_orderdate"},
    {"customer", ""},
    {"part", ""},
    {"partsupp", ""},
    {"supplier", ""},
    {"nation", ""},
    {"region", ""},
};

std::vector<std::string> parquetFiles(const std::string& tableDir) {
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(tableDir)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  VELOX_CHECK(!files.empty(), "No data files in {}", tableDir);
  return files;
}

class EncodedTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override {
    if (!FLAGS_cudf_properties.empty()) {
      cudf_velox::CudfConfig::getInstance().initialize(
          cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
    }
    TpchBenchmark::initialize();

    VELOX_USER_CHECK(
        FLAGS_encoded_device == "gpu" || FLAGS_encoded_device == "cpu",
        "--encoded_device must be gpu or cpu");
    VELOX_USER_CHECK(
        FLAGS_encoded_memory == "device" ||
            FLAGS_encoded_memory == "pinned_host",
        "--encoded_memory must be device or pinned_host");
    const bool gpu = FLAGS_encoded_device == "gpu";
    connector::ConnectorRegistry::global().erase(exec::test::kHiveConnectorId);
    connector::ConnectorRegistry::global().insert(
        exec::test::kHiveConnectorId,
        std::make_shared<encoded::EncodedConnector>(
            exec::test::kHiveConnectorId,
            gpu ? encoded::EncodedDevice::kGpu : encoded::EncodedDevice::kCpu));
    if (!gpu) {
      loadTables(/*gpu=*/false);
      return;
    }

    cudf_velox::registerCudf();
    encoded::registerEncodedTableScanAdapter();
    cudf_velox::registerPrestoFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);

    queryConfigs_[cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
        std::to_string(FLAGS_cudf_gpu_batch_size_rows);
    queryConfigs_[core::QueryConfig::kMaxLocalExchangeBufferSize] =
        std::to_string(FLAGS_cudf_local_exchange_buffer_size);

    loadTables(/*gpu=*/true);
  }

  std::vector<std::shared_ptr<connector::ConnectorSplit>> listSplits(
      const std::string& path,
      int32_t /*numSplitsPerFile*/,
      const exec::test::TpchPlan& /*plan*/) override {
    const auto tableDir = std::filesystem::path(path).parent_path();
    // A table with several files gets its splits once, for its first file.
    if (parquetFiles(tableDir.string()).front() != path) {
      return {};
    }
    return encoded::makeEncodedSplits(
        exec::test::kHiveConnectorId,
        tableDir.filename().string(),
        FLAGS_encoded_segments_per_split);
  }

  void shutdown() override {
    encoded::EncodedColumnCache::instance().clear();
    if (FLAGS_encoded_device == "gpu") {
      cudf_velox::unregisterCudf();
    }
    TpchBenchmark::shutdown();
  }

 private:
  void loadTables(bool gpu) {
    if (gpu) {
      cudf_velox::ensureCudaContextForThread();
    } else {
      VELOX_CHECK(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice failed");
    }
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = gpu ? cudf_velox::get_output_mr()
                        : rmm::mr::get_current_device_resource_ref();
    auto& cache = encoded::EncodedColumnCache::instance();
    for (const auto& [table, clusterKey] : kTables) {
      const auto start = std::chrono::steady_clock::now();
      cache.loadTable(
          table,
          parquetFiles(FLAGS_data_path + "/" + table),
          FLAGS_encoded_cluster ? clusterKey : "",
          FLAGS_encoded_segment_rows,
          stream,
          mr);
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      const auto& loaded = cache.table(table);
      int64_t bytes = 0;
      for (const auto& column : loaded.columns) {
        bytes += column->encodedBytes();
      }
      std::cout << fmt::format(
                       "Encoded {}: {} rows, {} segments, {} bytes, {} ms",
                       table,
                       loaded.numRows,
                       loaded.numSegments,
                       bytes,
                       ms)
                << std::endl;
    }
    if (gpu && FLAGS_encoded_memory == "pinned_host") {
      static rmm::mr::pinned_host_memory_resource pinnedHost;
      const auto start = std::chrono::steady_clock::now();
      cache.moveTo(stream, pinnedHost);
      std::cout << fmt::format(
                       "Moved encoded tables to pinned host memory: {} ms",
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count())
                << std::endl;
    }
    if (!gpu) {
      const auto start = std::chrono::steady_clock::now();
      cache.copyToHost(stream, /*releaseDevice=*/true);
      std::cout << fmt::format(
                       "Copied encoded tables to host and freed the device "
                       "copies: {} ms",
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count())
                << std::endl;
    }
    if (!FLAGS_encoded_size_report.empty()) {
      std::ofstream(FLAGS_encoded_size_report) << cache.sizeReport();
    }
  }
};

} // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Benchmarks TPC-H on cuDF over the encoded in-memory column cache.");
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<EncodedTpchBenchmark>();
  tpchBenchmarkMain();
}
