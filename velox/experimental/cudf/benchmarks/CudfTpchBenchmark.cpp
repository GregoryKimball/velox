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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <fmt/format.h>
#include <folly/String.h>

#include <chrono>
#include <filesystem>
#include <sstream>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);
DECLARE_string(data_path);
DECLARE_int32(num_repeats);
DECLARE_int32(run_query_verbose);

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;
namespace encoded = facebook::velox::cudf_velox::connector::encoded;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_uint64(
    cudf_local_exchange_buffer_size,
    1UL << 30,
    "Maximum buffered bytes per local exchange before applying backpressure.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");
DEFINE_bool(
    cudf_hive_use_buffered_input,
    true,
    "Use BufferedInputDataSource instead of KvikIO for cuDF Hive reads");

DEFINE_bool(
    cudf_hive_preload_column_chunks,
    false,
    "Load column chunks while preloading cuDF Hive splits");

DEFINE_bool(
    cudf_hive_serialize_io_requests,
    false,
    "Serialize I/O submissions to prevent interleaving between concurrent splits");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

DEFINE_bool(
    encoded,
    false,
    "Scan the tables from the encoded in-memory column cache instead of "
    "Parquet. The tables are loaded from --data_path a few row groups at a "
    "time, in file order, and encoded on the GPU before any query runs; sort "
    "the files beforehand to cluster them.");

DEFINE_string(
    encoded_memory,
    "device",
    "With --encoded, where the encoded tables are kept: device, or "
    "pinned_host to have scans copy each batch to the device before decoding.");

DEFINE_int32(
    encoded_segment_rows,
    1 << 20,
    "Rows per encoded segment, the unit of min/max pruning.");

DEFINE_int32(
    encoded_segments_per_split,
    4,
    "Segments per split, the unit of work handed to scan drivers. GPU scans "
    "batch segments across splits up to the cuDF batch-size targets.");

DEFINE_string(
    run_queries,
    "",
    "Comma-separated query numbers to run verbosely one after another in this "
    "process, initializing (and with --encoded, loading the tables) once. Each "
    "query's output starts with a 'Running query N' line. Overrides "
    "--run_query_verbose.");

namespace {

const std::vector<std::string> kTables = {
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
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

void loadEncodedTables() {
  VELOX_USER_CHECK(
      FLAGS_encoded_memory == "device" || FLAGS_encoded_memory == "pinned_host",
      "--encoded_memory must be device or pinned_host");
  cudf_velox::ensureCudaContextForThread();
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  auto& cache = encoded::EncodedColumnCache::instance();
  static rmm::mr::pinned_host_memory_resource pinnedHost;
  const bool hostStorage = FLAGS_encoded_memory == "pinned_host";
  for (const auto& table : kTables) {
    const auto start = std::chrono::steady_clock::now();
    cache.loadTable(
        table,
        parquetFiles(FLAGS_data_path + "/" + table),
        FLAGS_encoded_segment_rows,
        stream,
        mr,
        hostStorage ? rmm::device_async_resource_ref(pinnedHost) : mr,
        hostStorage);
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
}

// Unlike QueryBenchmarkBase::printResults(), prints every column; that
// truncates rows to 5 columns.
void printRows(const std::vector<RowVectorPtr>& results, std::ostream& out) {
  out << "Results:" << std::endl;
  if (!results.empty()) {
    out << results.front()->type()->asRow().toString() << std::endl;
  }
  for (const auto& vector : results) {
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      std::vector<std::string> fields;
      for (const auto& child : vector->children()) {
        fields.push_back(child->toString(i));
      }
      out << "{" << folly::join(", ", fields) << "}" << std::endl;
    }
  }
  out << std::endl;
}

} // namespace

void CudfTpchBenchmark::initialize() {
  if (!FLAGS_cudf_properties.empty()) {
    cudf_velox::CudfConfig::getInstance().initialize(
        cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  TpchBenchmark::initialize();
  queryBuilder_ = std::make_shared<TpchQueryBuilder>(
      toFileFormat(FLAGS_data_format));
  queryBuilder_->initialize(FLAGS_data_path);

  if (FLAGS_encoded) {
    connector::ConnectorRegistry::global().erase(
        exec::test::kHiveConnectorId);
    connector::ConnectorRegistry::global().insert(
        exec::test::kHiveConnectorId,
        std::make_shared<encoded::EncodedConnector>(
            exec::test::kHiveConnectorId));
  } else if (FLAGS_velox_cudf_table_scan) {
    connector::ConnectorRegistry::global().erase(
        exec::test::kHiveConnectorId);

    auto cudfHiveConfigurationValues =
        std::unordered_map<std::string, std::string>();
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput] =
            std::to_string(FLAGS_cudf_hive_use_buffered_input);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kPreloadColumnChunks] =
            std::to_string(FLAGS_cudf_hive_preload_column_chunks);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kSerializeIoRequests] =
            std::to_string(FLAGS_cudf_hive_serialize_io_requests);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        exec::test::kHiveConnectorId,
        cudfHiveProperties,
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  cudf_velox::registerCudf();
  if (FLAGS_encoded) {
    encoded::registerEncodedTableScanAdapter();
  }
  cudf_velox::registerPrestoFunctions(
      cudf_velox::CudfConfig::getInstance().functionNamePrefix);

  queryConfigs_[cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  queryConfigs_[core::QueryConfig::kMaxLocalExchangeBufferSize] =
      std::to_string(FLAGS_cudf_local_exchange_buffer_size);

  if (FLAGS_encoded) {
    loadEncodedTables();
  }
}

std::shared_ptr<config::ConfigBase>
CudfTpchBenchmark::makeConnectorProperties() {
  auto cfg = TpchBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  // CuDF-specific properties.
  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(
      CudfHiveCfg::kUseBufferedInput,
      std::to_string(FLAGS_cudf_hive_use_buffered_input));
  cfg->set(
      CudfHiveCfg::kPreloadColumnChunks,
      std::to_string(FLAGS_cudf_hive_preload_column_chunks));
  cfg->set(
      CudfHiveCfg::kSerializeIoRequests,
      std::to_string(FLAGS_cudf_hive_serialize_io_requests));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  return cfg;
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
CudfTpchBenchmark::listSplits(
    const std::string& path,
    int32_t numSplitsPerFile,
    const exec::test::TpchPlan& plan) {
  if (FLAGS_encoded) {
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
  // TODO (dm): Figure out a way to enforce 1 split per file in
  // CudfHiveDataSource outside of this benchmark
  if (FLAGS_velox_cudf_table_scan) {
    // TODO (dm): Instead of this, we can maybe use
    // makeHiveConnectorSplits(vector<shared_ptr<TempFilePath>>&
    // filePaths)
    std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
    auto temp = HiveConnectorTestBase::makeHiveConnectorSplits(
        path, 1, plan.dataFileFormat);
    for (auto& i : temp) {
      result.push_back(i);
    }
    return result;
  }

  return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
}

void CudfTpchBenchmark::runMain(std::ostream& out, RunStats& runStats) {
  if (FLAGS_run_query_verbose == -1) {
    TpchBenchmark::runMain(out, runStats);
    return;
  }
  const auto plan = queryBuilder_->getQueryPlan(FLAGS_run_query_verbose);
  const auto repeats = FLAGS_num_repeats;
  FLAGS_num_repeats = 1;
  std::pair<
      std::unique_ptr<exec::TaskCursor>,
      std::vector<RowVectorPtr>>
      result;
  for (int32_t repeat = 0; repeat < repeats; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    result = run(plan, queryConfigs_);
    if (result.first) {
      ensureTaskCompletion(result.first->task().get());
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    if (!result.first) {
      break;
    }
    out << fmt::format("Repeat {} wall time us: {}", repeat, us) << std::endl;
  }
  FLAGS_num_repeats = repeats;
  VELOX_CHECK_NOT_NULL(
      result.first, "Query {} failed", FLAGS_run_query_verbose);

  if (FLAGS_include_results) {
    printRows(result.second, out);
  }
  const auto stats = result.first->task()->taskStats();
  out << fmt::format(
             "Execution time: {}",
             succinctMillis(
                 stats.executionEndTimeMs - stats.executionStartTimeMs))
      << std::endl;
  out << fmt::format(
             "Splits total: {}, finished: {}",
             stats.numTotalSplits,
             stats.numFinishedSplits)
      << std::endl;
  out << exec::printPlanWithStats(
             *plan.plan, stats, FLAGS_include_custom_stats)
      << std::endl;
}

void CudfTpchBenchmark::shutdown() {
  encoded::EncodedColumnCache::instance().clear();
  cudf_velox::unregisterCudf();
  queryBuilder_.reset();
  TpchBenchmark::shutdown();
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  if (FLAGS_run_queries.empty()) {
    tpchBenchmarkMain();
    return 0;
  }
  std::vector<int32_t> queries;
  std::stringstream list(FLAGS_run_queries);
  for (std::string query; std::getline(list, query, ',');) {
    queries.push_back(std::stoi(query));
  }
  benchmark->initialize();
  for (const auto query : queries) {
    FLAGS_run_query_verbose = query;
    std::cout << "Running query " << query << std::endl;
    try {
      RunStats ignore;
      benchmark->runMain(std::cout, ignore);
    } catch (const std::exception& e) {
      std::cout << "Query " << query << " failed: " << e.what() << std::endl;
    }
    std::cout << "Finished query " << query << std::endl;
  }
  benchmark->shutdown();
  return 0;
}
