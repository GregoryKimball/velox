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
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/experimental/supervelox/exec/FusedQuery.h"
#include "velox/experimental/supervelox/exec/PlanMatcher.h"
#ifdef SUPERVELOX_CUDF
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/supervelox/cudf/CacheConnector.h"
#include "velox/experimental/supervelox/cudf/AggregationRules.h"
#include "velox/experimental/supervelox/cudf/CudfBackend.h"
#endif
#ifdef SUPERVELOX_MELD
#include "velox/experimental/supervelox/meld/MeldBackend.h"
#endif
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

DEFINE_string(data, "", "Directory written by prep.py");
DEFINE_string(queries, "", "Comma-separated TPC-H query numbers; empty runs every fused query");
DEFINE_int32(runs, 5, "Runs per query; the first may copy columns into HBM");
DEFINE_double(cache_gb, 80, "HBM cache budget in GB");
DEFINE_double(stage_gb, 2, "Columns at least this large are staged into HBM behind the query that misses them");
DEFINE_string(out, "", "Directory for qNN.csv results");
DEFINE_bool(preload, false, "Read every column into pinned host memory before the first query");
DEFINE_string(
    tpch_plans,
    "",
    "Directory of TPC-H Parquet tables, one subdirectory per table. When set, each query runs the plan "
    "Velox's TpchQueryBuilder makes, with fused queries in place of the subtrees they match; the rest "
    "runs on Velox's CPU operators over the Parquet files");
DEFINE_bool(verify_cpu, false, "With --tpch_plans, also run each plan unchanged on the CPU and compare answers");
DEFINE_bool(filters_as_node, false, "With --tpch_plans, put filters in filter nodes rather than in the table scans");
DEFINE_string(backend, "handwritten", "Who runs the fused queries: handwritten, cudf or meld");
DEFINE_string(query_backends, "", "Per-query backends overriding --backend, as 1=cudf,6=handwritten");
DEFINE_bool(compare_handwritten, false, "Also run each query with the hand-written kernels and compare answers");
DEFINE_string(meld_library, "", "libmeld-velox.so to load for the meld backend");
DEFINE_int64(
    batch_rows,
    0,
    "Run the hand-written kernels batch by batch, about this many lineitem rows a batch; 0 runs each kernel once");
DEFINE_bool(
    velox_cudf,
    false,
    "With --tpch_plans, run the plans unchanged on the Velox-cuDF operators, scanning the HBM cache instead of "
    "Parquet");
DEFINE_int64(scan_rows, 1 << 24, "With --velox_cudf, rows per table scan split, each one batch");
DEFINE_int32(drivers, 16, "Drivers per pipeline for --tpch_plans");
DEFINE_string(
    rules,
    "",
    "With --velox_cudf, the comma separated rules to apply, none by default: decode_filter_gather (read a scan's "
    "columns encoded, evaluate its filters in one kernel, compact and decode the rows that pass), "
    "expression_project (also evaluate the filters and projections above the scan in those kernels), order_aware "
    "(aggregate in one step over pre-grouped input where a scanned table keeps rows with equal grouping keys "
    "adjacent), direct (aggregate by keys that the store bounds to small ranges into slots indexed by key; dense "
    "is the old name), streaming_groupby (cuDF's streaming group-by for final aggregations), fuse (evaluate the "
    "filters and projections below an aggregation into few groups inside one reduction), exists_aggregation (semi "
    "and anti joins filtered by an inequality, as for exists subqueries, join with the min and max per key of a "
    "table clustered by it), key_pushdown (an aggregation inner joined on its grouping key with a side whose keys "
    "come from a filtered scan semi joins its input with that scan), key_filter (with decode_filter_gather, the scan "
    "below a join's probe side passes only the keys of the filtered scan its build side keys come from)");
DEFINE_bool(plan_stats, false, "With --tpch_plans, print each run's plan with the time and rows of every node");
DEFINE_string(cudf_config, "", "With --velox_cudf, more CudfConfig entries as key=value,key=value");
DEFINE_bool(cudf_debug, false, "With --velox_cudf, log which operators Velox-cuDF replaces");
DEFINE_int32(cudf_memory_percent, 50, "With --velox_cudf, percent of GPU memory the cuDF memory pool starts with");
using namespace facebook::velox;
using namespace facebook::velox::supervelox;
using exec::test::PlanBuilder;

namespace {

/// The rules named by --rules.
const std::set<std::string>& rules() {
  static const std::set<std::string> parsed = [] {
    static const std::set<std::string> known{
        "decode_filter_gather",
        "expression_project",
        "order_aware",
        "direct",
        "dense",
        "streaming_groupby",
        "fuse",
        "exists_aggregation",
        "key_pushdown",
        "key_filter"};
    std::set<std::string> out;
    std::stringstream names(FLAGS_rules);
    std::string name;
    while (std::getline(names, name, ',')) {
      VELOX_CHECK(known.count(name), "Unknown rule {} in --rules", name);
      out.insert(name == "dense" ? "direct" : name);
    }
    return out;
  }();
  return parsed;
}

} // namespace

namespace {

core::PlanNodePtr fused(int32_t query, const core::PlanNodeId& id) {
  return std::make_shared<FusedQueryNode>(id, query);
}

/// The Velox plan for 'query': the fused GPU node plus whatever Velox
/// operators finish the query on its output.
core::PlanNodePtr plan(int32_t query) {
  PlanBuilder builder;
  builder.addNode([&](std::string id, core::PlanNodePtr) { return fused(query, id); });
  switch (query) {
    case 1:
      builder.orderBy({"l_returnflag", "l_linestatus"}, false);
      break;
    case 4:
      builder.orderBy({"o_orderpriority"}, false);
      break;
    case 5:
      builder.orderBy({"revenue DESC"}, false);
      break;
    case 12:
      builder.orderBy({"l_shipmode"}, false);
      break;
    default:
      break;
  }
  return builder.planNode();
}

std::string csvField(const VectorPtr& vector, vector_size_t row) {
  if (vector->isNullAt(row)) {
    return "";
  }
  const auto& type = vector->type();
  if (type->isDate()) {
    return DATE()->toString(vector->as<SimpleVector<int32_t>>()->valueAt(row));
  }
  if (type->isShortDecimal()) {
    return DecimalUtil::toString(vector->as<SimpleVector<int64_t>>()->valueAt(row), type);
  }
  if (type->isLongDecimal()) {
    return DecimalUtil::toString(vector->as<SimpleVector<int128_t>>()->valueAt(row), type);
  }
  switch (type->kind()) {
    case TypeKind::BIGINT:
      return std::to_string(vector->as<SimpleVector<int64_t>>()->valueAt(row));
    case TypeKind::INTEGER:
      return std::to_string(vector->as<SimpleVector<int32_t>>()->valueAt(row));
    case TypeKind::DOUBLE: {
      char buffer[64];
      snprintf(buffer, sizeof(buffer), "%.17g", vector->as<SimpleVector<double>>()->valueAt(row));
      return buffer;
    }
    case TypeKind::VARCHAR: {
      std::string text = vector->as<SimpleVector<StringView>>()->valueAt(row).str();
      std::string quoted = "\"";
      for (char c : text) {
        quoted += c;
        if (c == '"') {
          quoted += '"';
        }
      }
      return quoted + "\"";
    }
    default:
      return vector->toString(row);
  }
}

void writeCsv(const RowVectorPtr& result, const std::string& path) {
  std::ofstream out(path);
  const auto& type = *result->rowType();
  for (size_t i = 0; i < type.size(); ++i) {
    out << (i ? "," : "") << type.nameOf(i);
  }
  out << "\n";
  for (vector_size_t row = 0; row < result->size(); ++row) {
    for (size_t i = 0; i < type.size(); ++i) {
      out << (i ? "," : "") << csvField(result->childAt(i), row);
    }
    out << "\n";
  }
}

void collectScans(const core::PlanNodePtr& node, std::vector<core::PlanNodeId>& ids) {
  if (dynamic_cast<const core::TableScanNode*>(node.get())) {
    ids.push_back(node->id());
  }
  for (const auto& source : node->sources()) {
    collectScans(source, ids);
  }
}

const core::TableScanNode* findScan(const core::PlanNodePtr& node, const core::PlanNodeId& id) {
  if (node->id() == id) {
    return dynamic_cast<const core::TableScanNode*>(node.get());
  }
  for (const auto& source : node->sources()) {
    if (auto* scan = findScan(source, id)) {
      return scan;
    }
  }
  return nullptr;
}

/// The connector that plans' table scans name, swapped between Parquet and
/// the HBM cache.
void useConnector(const std::shared_ptr<connector::Connector>& connector) {
  connector::ConnectorRegistry::global().erase(connector->connectorId());
  connector::ConnectorRegistry::global().insert(connector->connectorId(), connector);
}

/// Runs `plan`, giving each table scan left in it the Parquet files that
/// `tables` lists for it, or with `splits` set, the splits it returns for the
/// scanned table. `cudf` says whether Velox-cuDF may take over the plan.
RowVectorPtr execute(
    const core::PlanNodePtr& plan,
    const exec::test::TpchPlan& tables,
    memory::MemoryPool* pool,
    const std::function<std::vector<std::shared_ptr<connector::ConnectorSplit>>(const std::string&)>& splits = nullptr,
    bool cudf = true) {
  exec::test::AssertQueryBuilder builder(plan);
  std::vector<core::PlanNodeId> scans;
  collectScans(plan, scans);
  for (const auto& id : scans) {
    if (splits) {
      builder.splits(id, splits(findScan(plan, id)->tableHandle()->name()));
      continue;
    }
    std::vector<std::shared_ptr<connector::ConnectorSplit>> files;
    for (const auto& file : tables.dataFiles.at(id)) {
      for (auto& split : exec::test::HiveConnectorTestBase::makeHiveConnectorSplits(file, 1, tables.dataFileFormat)) {
        files.push_back(split);
      }
    }
    builder.splits(id, files);
  }
  if (!cudf) {
    builder.config("cudf.enabled", "false");
  }
#ifdef SUPERVELOX_CUDF
  if (FLAGS_velox_cudf && cudf) {
    // As the Velox-cuDF TPC-H benchmark that measured the Parquet baseline.
    builder.config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "100000");
    builder.config(core::QueryConfig::kMaxLocalExchangeBufferSize, std::to_string(1UL << 30));
  }
#endif
  std::shared_ptr<exec::Task> task;
  auto result = builder.maxDrivers(scans.empty() ? 1 : FLAGS_drivers).copyResults(pool, task);
  if (FLAGS_plan_stats) {
    printf("%s\n", exec::printPlanWithStats(*plan, task->taskStats(), true).c_str());
  }
  return result;
}

std::vector<int32_t> queryList() {
  std::vector<int32_t> queries;
  if (FLAGS_queries.empty()) {
    if (!FLAGS_tpch_plans.empty()) {
      for (int32_t query = 1; query <= 22; ++query) {
        queries.push_back(query);
      }
      return queries;
    }
    for (const auto& spec : querySpecs()) {
      queries.push_back(spec.number);
    }
    return queries;
  }
  std::stringstream list(FLAGS_queries);
  std::string item;
  while (std::getline(list, item, ',')) {
    queries.push_back(std::stoi(item));
  }
  return queries;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  registerFusedQuery();

  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor;
  std::unique_ptr<exec::test::TpchQueryBuilder> tpch;
  std::shared_ptr<connector::Connector> parquetConnector;
  if (!FLAGS_tpch_plans.empty()) {
    filesystems::registerLocalFileSystem();
    parquet::registerParquetReaderFactory();
    ioExecutor = std::make_unique<folly::IOThreadPoolExecutor>(8);
    auto hive = connector::hive::HiveConnectorFactory().newConnector(
        exec::test::kHiveConnectorId,
        std::make_shared<config::ConfigBase>(std::unordered_map<std::string, std::string>()),
        ioExecutor.get());
    parquetConnector = hive;
    connector::ConnectorRegistry::global().insert(hive->connectorId(), hive);
    tpch = std::make_unique<exec::test::TpchQueryBuilder>(dwio::common::FileFormat::PARQUET, FLAGS_filters_as_node);
    tpch->initialize(FLAGS_tpch_plans);
    for (int32_t query = 1; query <= 22; ++query) {
      registerReferencePlan(query, tpch->getQueryPlan(query).plan);
    }
  }

  Engine engine(
      FLAGS_data,
      static_cast<size_t>(FLAGS_cache_gb * (1ULL << 30)),
      static_cast<size_t>(FLAGS_stage_gb * (1ULL << 30)));
  Engine::setInstance(&engine);
  setBatchRows(FLAGS_batch_rows);
#ifdef SUPERVELOX_CUDF
  cudf_backend::registerCudfBackend();
#endif
#ifdef SUPERVELOX_MELD
  if (!FLAGS_meld_library.empty()) {
    std::string error;
    VELOX_CHECK(meld_backend::registerMeldBackend(FLAGS_meld_library, error), "cannot load Meld: {}", error);
  }
#endif
  engine.setBackend(parseBackend(FLAGS_backend));
  std::shared_ptr<connector::Connector> cacheConnector;
#ifdef SUPERVELOX_CUDF
  std::shared_ptr<cudf_backend::CacheConnector> cache;
#endif
  if (FLAGS_velox_cudf) {
#ifdef SUPERVELOX_CUDF
    VELOX_CHECK(tpch != nullptr, "--velox_cudf needs --tpch_plans");
    std::unordered_map<std::string, std::string> cudfConfig{
        {cudf_velox::CudfConfig::kCudfMemoryPercent, std::to_string(FLAGS_cudf_memory_percent)},
        {cudf_velox::CudfConfig::kCudfDebugEnabled, FLAGS_cudf_debug ? "true" : "false"}};
    if (rules().count("streaming_groupby")) {
      cudfConfig["cudf.streaming_groupby_enabled"] = "true";
    }
    std::stringstream entries(FLAGS_cudf_config);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
      const auto equals = entry.find('=');
      VELOX_CHECK_NE(equals, std::string::npos, "--cudf_config entry {} is not key=value", entry);
      cudfConfig[entry.substr(0, equals)] = entry.substr(equals + 1);
    }
    cudf_velox::CudfConfig::getInstance().initialize(std::move(cudfConfig));
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions(cudf_velox::CudfConfig::getInstance().functionNamePrefix);
    cache = std::make_shared<cudf_backend::CacheConnector>(
        exec::test::kHiveConnectorId, engine.store(), engine.cache(), FLAGS_scan_rows);
    cacheConnector = cache;
    useConnector(cacheConnector);
#else
    VELOX_FAIL("--velox_cudf needs the cuDF build");
#endif
  }
  {
    std::stringstream overrides(FLAGS_query_backends);
    std::string item;
    while (std::getline(overrides, item, ',')) {
      const auto equals = item.find('=');
      VELOX_CHECK_NE(equals, std::string::npos, "--query_backends item {} is not query=backend", item);
      engine.setBackend(std::stoi(item.substr(0, equals)), parseBackend(item.substr(equals + 1)));
    }
  }
  if (FLAGS_preload) {
    const auto start = std::chrono::steady_clock::now();
    engine.store().preload(engine.store().columns(
        {"region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"}));
    printf(
        "preloaded %.1f GB pinned in %.1f s\n",
        engine.store().pinnedBytes() / 1e9,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  }
  auto pool = memory::memoryManager()->addLeafPool();

  double totalBest = 0;
  for (int32_t query : queryList()) {
    core::PlanNodePtr planNode;
    exec::test::TpchPlan tables;
    std::function<std::vector<std::shared_ptr<connector::ConnectorSplit>>(const std::string&)> cacheSplits;
    if (tpch) {
      tables = tpch->getQueryPlan(query);
#ifdef SUPERVELOX_CUDF
      if (cache) {
        planNode = tables.plan;
        std::map<std::string, std::vector<std::string>> splitKeys;
        auto rewrite = cudf_backend::rewriteAggregations(
            tables.plan,
            *cache,
            {.decode = rules().count("decode_filter_gather") > 0,
             .project = rules().count("expression_project") > 0,
             .preGroup = rules().count("order_aware") > 0,
             .direct = rules().count("direct") > 0,
             .fuse = rules().count("fuse") > 0,
             .existsAggregation = rules().count("exists_aggregation") > 0,
             .keyPushdown = rules().count("key_pushdown") > 0,
             .keyFilter = rules().count("key_filter") > 0});
        planNode = rewrite.plan;
        splitKeys = std::move(rewrite.splitKeys);
        for (const auto& line : rewrite.log) {
          printf("q%d: %s\n", query, line.c_str());
        }
        cacheSplits = [cache, splitKeys](const std::string& table) {
          const auto keys = splitKeys.find(table);
          return cache->splits(table, keys == splitKeys.end() ? std::vector<std::string>{} : keys->second);
        };
        printf("q%d: Velox-cuDF over the HBM cache, rules: %s\n", query, FLAGS_rules.empty() ? "none" : FLAGS_rules.c_str());
      }
#endif
    }
    if (tpch && !cacheConnector) {
      std::vector<std::string> log;
      planNode = rewriteWithFusedQueries(tables.plan, engine.store(), &log);
      for (const auto& line : log) {
        printf("q%d: %s\n", query, line.c_str());
      }
      std::vector<core::PlanNodeId> scans;
      collectScans(planNode, scans);
      printf("q%d: %s\n", query, scans.empty() ? "fully fused" : "runs on the CPU over Parquet");
    } else if (!tpch) {
      planNode = plan(query);
    }
    double best = 1e30;
    double bestGpu = 1e30;
    RowVectorPtr result;
    for (int32_t run = 0; run < FLAGS_runs; ++run) {
      Engine::Run cacheRun;
      if (cacheConnector) {
        engine.cache().beginQuery();
        engine.cache().resetStats();
      }
      const auto start = std::chrono::steady_clock::now();
      result = tpch ? execute(planNode, tables, pool.get(), cacheSplits)
                    : exec::test::AssertQueryBuilder(planNode).maxDrivers(1).copyResults(pool.get());
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      if (cacheConnector) {
        cacheRun.gpuMs = ms;
        cacheRun.cache = engine.cache().stats();
      }
      const auto& last = cacheConnector ? cacheRun : engine.lastRun();
      printf(
          "q%d run %d: %.2f ms (%s gpu %.2f ms) copied %.2f GB staged %.2f GB host %.2f GB built %.2f GB resident %.1f GB\n",
          query,
          run,
          ms,
          cacheConnector ? "velox-cudf" : backendName(last.backend),
          last.gpuMs,
          last.cache.copiedBytes / 1e9,
          last.cache.stagedBytes / 1e9,
          last.cache.hostBytes / 1e9,
          last.cache.builtBytes / 1e9,
          engine.cache().residentBytes() / 1e9);
      fflush(stdout);
      if (run > 0 || FLAGS_runs == 1) {
        best = std::min(best, ms);
        bestGpu = std::min(bestGpu, last.gpuMs);
      }
    }
    totalBest += best;
    printf("q%d best %.2f ms (gpu %.2f ms) rows %d\n", query, best, bestGpu, result->size());
    if (tpch && FLAGS_verify_cpu) {
      if (cacheConnector) {
        useConnector(parquetConnector);
      }
      auto expected = execute(tables.plan, tables, pool.get(), nullptr, false);
      if (cacheConnector) {
        useConnector(cacheConnector);
      }
      const bool same = exec::test::assertEqualResults({expected}, {result});
      printf("q%d matches the CPU plan: %s\n", query, same ? "yes" : "NO");
    }
    if (FLAGS_compare_handwritten && engine.backend(query) != Backend::kHandwritten) {
      const Backend chosen = engine.backend(query);
      engine.setBackend(query, Backend::kHandwritten);
      auto expected = tpch ? execute(planNode, tables, pool.get())
                           : exec::test::AssertQueryBuilder(planNode).maxDrivers(1).copyResults(pool.get());
      engine.setBackend(query, chosen);
      const bool same = exec::test::assertEqualResults({expected}, {result});
      printf("q%d matches the hand-written kernels: %s\n", query, same ? "yes" : "NO");
    }
    if (!FLAGS_out.empty()) {
      char name[32];
      snprintf(name, sizeof(name), "/q%02d.csv", query);
      writeCsv(result, FLAGS_out + name);
    }
  }
  printf("total best %.2f ms\n", totalBest);
  return 0;
}
