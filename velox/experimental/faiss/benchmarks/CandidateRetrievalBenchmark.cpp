/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "velox/experimental/faiss/FaissNvtx.h"
#include "velox/experimental/faiss/FaissOperators.h"

#if defined(VELOX_ENABLE_FAISS_GPU)
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#endif

#include "velox/common/file/LocalFile.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include <folly/init/Init.h>
#include <folly/json.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>

DEFINE_string(execution, "cpu", "cpu|gpu|gpu_build_cpu_search");
DEFINE_string(provider, "build", "build|load");
DEFINE_string(
    strategy,
    "flat",
    "flat|ivf_flat|ivf_pq|cagra|hnsw|cagra_hnsw");
DEFINE_int64(candidates, 100000, "Candidate rows");
DEFINE_int64(queries, 10000, "Query rows");
DEFINE_int32(dimension, 128, "Embedding dimension");
DEFINE_int32(topK, 100, "Neighbors per query");
DEFINE_string(preset, "", "ci|baseline|validation|headline");
DEFINE_string(artifacts, "", "FAISS artifact directory");
DEFINE_string(input, "", "Parquet input directory");
DEFINE_string(output, "", "JSON report path; stdout when empty");
DEFINE_int32(warmups, 1, "Warmup repetitions");
DEFINE_int32(repetitions, 3, "Measured repetitions");
DEFINE_double(
    min_recall,
    0.1,
    "Minimum recall@k required for approximate strategies");
DEFINE_bool(
    prepare_artifacts,
    false,
    "Generate reusable Parquet inputs and a strategy artifact");

namespace facebook::velox::faiss::benchmark {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::HiveConnectorTestBase;
using exec::test::PlanBuilder;
using Clock = std::chrono::steady_clock;

constexpr int64_t kMaxListChildren = 2'000'000'000LL - 1;
constexpr int32_t kMarket = 0;

struct Options {
  int64_t candidates;
  int64_t queries;
  int32_t dimension;
  int32_t topK;
};

struct RunResult {
  double fullQueryMs{0};
  double scanMs{0};
  double assignmentMs{0};
  double providerMs{0};
  double conversionMs{0};
  double searchMs{0};
  double outputMs{0};
  double trainMs{0};
  double addMs{0};
  double loadReadMs{0};
  double loadDeserializeMs{0};
  double loadUploadMs{0};
  double cagraToHnswMs{0};
  RowVectorPtr rows;
};

double nanosToMs(int64_t nanos) {
  return static_cast<double>(nanos) / 1'000'000.0;
}

FaissAlgorithm algorithm() {
  if (FLAGS_strategy == "flat") {
    return FaissAlgorithm::kFlat;
  }
  if (FLAGS_strategy == "ivf_flat") {
    return FaissAlgorithm::kIvfFlat;
  }
  if (FLAGS_strategy == "ivf_pq") {
    return FaissAlgorithm::kIvfPq;
  }
  if (FLAGS_strategy == "cagra") {
    return FaissAlgorithm::kCagra;
  }
  if (FLAGS_strategy == "hnsw") {
    return FaissAlgorithm::kHnsw;
  }
  if (FLAGS_strategy == "cagra_hnsw") {
    return FaissAlgorithm::kHnswCagra;
  }
  VELOX_USER_FAIL("Unknown --strategy={}", FLAGS_strategy);
}

Options optionsFromFlags() {
  if (FLAGS_preset == "ci") {
    return {2'000, 200, 32, std::min(FLAGS_topK, 10)};
  }
  if (FLAGS_preset.empty()) {
    return {FLAGS_candidates, FLAGS_queries, FLAGS_dimension, FLAGS_topK};
  }
  if (FLAGS_preset == "baseline") {
    return {100'000, 10'000, 128, FLAGS_topK};
  }
  if (FLAGS_preset == "validation") {
    return {100'000, 10'000, 512, FLAGS_topK};
  }
  if (FLAGS_preset == "headline") {
    VELOX_USER_FAIL(
        "The headline 21M-candidate/512-dimension preset is defined but "
        "execution is deferred until streaming index construction avoids "
        "materializing the full candidate corpus");
  }
  VELOX_USER_FAIL("Unknown --preset={}", FLAGS_preset);
}

void validateOptions(const Options& o) {
  VELOX_USER_CHECK_GT(o.candidates, 0);
  VELOX_USER_CHECK_GT(o.queries, 0);
  VELOX_USER_CHECK_GT(o.dimension, 0);
  VELOX_USER_CHECK_GT(o.topK, 0);
  VELOX_USER_CHECK(
      FLAGS_execution == "cpu" || FLAGS_execution == "gpu" ||
          FLAGS_execution == "gpu_build_cpu_search",
      "Unknown --execution={}",
      FLAGS_execution);
  VELOX_USER_CHECK(
      FLAGS_provider == "build" || FLAGS_provider == "load",
      "Unknown --provider={}",
      FLAGS_provider);
  VELOX_USER_CHECK(
      FLAGS_execution != "gpu_build_cpu_search" ||
          FLAGS_strategy == "cagra_hnsw",
      "gpu_build_cpu_search requires --strategy=cagra_hnsw");
  VELOX_USER_CHECK(
      o.dimension <= kMaxListChildren,
      "Embedding dimension exceeds Parquet/Velox list-child capacity");
}

FaissIndexConfig indexConfig(const Options& o) {
  FaissIndexConfig config;
  config.algorithm = algorithm();
  config.executionDevice = FLAGS_execution == "cpu"
      ? FaissExecutionDevice::kCpu
      : FaissExecutionDevice::kGpu;
  config.dimension = o.dimension;
  const auto clusterCount =
      std::max<int64_t>(1, std::min<int64_t>(8, o.candidates / 2'000));
  const auto estimatedRowsPerCluster =
      std::max<int64_t>(1, o.candidates * 10 / 44 / clusterCount);
  config.nlist = std::max<int32_t>(
      1, std::min<int64_t>(4096, std::sqrt(estimatedRowsPerCluster)));
  config.nprobe = std::max(1, config.nlist / 16);
  config.pqSubquantizers =
      o.dimension % 16 == 0 ? 16 : (o.dimension % 8 == 0 ? 8 : 1);
  // Small routed partitions cannot train the 256 codewords implied by
  // eight-bit PQ. Use 16 codewords until partitions are large enough.
  config.pqBits = estimatedRowsPerCluster >= 1'024 ? 8 : 4;
  config.hnswM = 32;
  config.efConstruction = 80;
  config.efSearch = std::max(64, o.topK * 4);
  config.validate();
  return config;
}

class CandidateRetrievalBenchmark final : public HiveConnectorTestBase {
 public:
  CandidateRetrievalBenchmark() {
    HiveConnectorTestBase::SetUp();
  }

  ~CandidateRetrievalBenchmark() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  static RowTypePtr candidateType() {
    return ROW(
        {"candidate_id", "embedding", "market", "active"},
        {BIGINT(), ARRAY(REAL()), BIGINT(), BOOLEAN()});
  }

  static RowTypePtr queryType() {
    return ROW(
        {"query_id", "embedding", "market"},
        {BIGINT(), ARRAY(REAL()), BIGINT()});
  }

  static RowTypePtr centroidType() {
    return ROW(
        {"centroid_id", "embedding"}, {BIGINT(), ARRAY(REAL())});
  }

  std::vector<RowVectorPtr> generate(
      int64_t rows,
      int32_t dimension,
      bool candidates,
      uint32_t seed) {
    const int64_t rowsPerBatch =
        std::max<int64_t>(1, std::min<int64_t>(4096, kMaxListChildren / dimension));
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> value(0, 1);
    std::vector<RowVectorPtr> batches;
    for (int64_t base = 0; base < rows; base += rowsPerBatch) {
      const auto count =
          static_cast<vector_size_t>(std::min(rowsPerBatch, rows - base));
      auto ids = makeFlatVector<int64_t>(count, [&](auto row) {
        return base + row;
      });
      auto embeddings = makeArrayVector<float>(
          count,
          [&](auto) { return dimension; },
          [&](auto, auto) { return value(random); });
      auto markets = makeFlatVector<int64_t>(count, [&](auto row) {
        return (base + row) % 4;
      });
      if (candidates) {
        auto active = makeFlatVector<bool>(count, [&](auto row) {
          return (base + row) % 11 != 0;
        });
        batches.push_back(makeRowVector(
            {"candidate_id", "embedding", "market", "active"},
            {ids, embeddings, markets, active}));
      } else {
        batches.push_back(makeRowVector(
            {"query_id", "embedding", "market"}, {ids, embeddings, markets}));
      }
    }
    return batches;
  }

  std::vector<float> centroids(const Options& o) {
    // Keep enough post-filter rows in every routed partition for PQ training
    // and CAGRA's graph degree. Eight partitions preserve the prior prototype
    // shape while the small CI preset intentionally uses one.
    const auto count =
        std::max<int64_t>(1, std::min<int64_t>(8, o.candidates / 2'000));
    std::vector<float> result(count * o.dimension);
    std::mt19937 random(41);
    std::uniform_real_distribution<float> value(0, 1);
    for (auto& v : result) {
      v = value(random);
    }
    return result;
  }

  void writeParquet(
      const std::string& path,
      const RowTypePtr& type,
      const std::vector<RowVectorPtr>& batches) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::make_unique<LocalWriteFile>(path, true, false), path);
    auto writerPool = rootPool_->addAggregateChild("candidate-writer");
    dwio::common::WriterOptions writerOptions;
    writerOptions.memoryPool = writerPool.get();
    writerOptions.formatSpecificOptions =
        std::make_shared<parquet::ParquetWriterOptions>();
    parquet::Writer writer(std::move(sink), writerOptions, writerPool, type);
    for (const auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
  }

  void prepareInputs(const Options& o, const std::string& directory) {
    VELOX_USER_CHECK(!directory.empty(), "--input is required for preparation");
    writeParquet(
        directory + "/candidates.parquet",
        candidateType(),
        generate(o.candidates, o.dimension, true, 17));
    writeParquet(
        directory + "/retrieval_queries.parquet",
        queryType(),
        generate(o.queries, o.dimension, false, 23));
    const auto values = centroids(o);
    const auto count = values.size() / o.dimension;
    auto ids = makeFlatVector<int64_t>(count, [&](auto row) { return row; });
    auto arrays = makeArrayVector<float>(
        count,
        [&](auto) { return o.dimension; },
        [&](auto row, auto element) {
          return values[row * o.dimension + element];
        });
    writeParquet(
        directory + "/centroids.parquet",
        centroidType(),
        {makeRowVector({"centroid_id", "embedding"}, {ids, arrays})});
  }

  std::vector<float> loadCentroids(
      const Options& o,
      const std::string& directory) {
    auto plan = scan(centroidType(), "", pool_.get());
    auto result = AssertQueryBuilder(plan)
                      .split(
                          connector::hive::HiveConnectorSplitBuilder(
                              directory + "/centroids.parquet")
                              .connectorId(exec::test::kHiveConnectorId)
                              .fileFormat(dwio::common::FileFormat::PARQUET)
                              .build())
                      .copyResults(pool_.get());
    const auto* arrays = validateFaissEmbeddings(
        result, centroidType(), "embedding", o.dimension);
    const auto* elements = arrays->elements()->as<SimpleVector<float>>();
    std::vector<float> centers(
        static_cast<size_t>(result->size()) * o.dimension);
    for (vector_size_t row = 0; row < result->size(); ++row) {
      for (int32_t d = 0; d < o.dimension; ++d) {
        centers[static_cast<size_t>(row) * o.dimension + d] =
            elements->valueAt(arrays->offsetAt(row) + d);
      }
    }
    return centers;
  }

  core::PlanNodePtr scan(
      const RowTypePtr& type,
      const std::string& filter,
      memory::MemoryPool* pool,
      std::shared_ptr<core::PlanNodeIdGenerator> idGenerator = nullptr) {
    if (!idGenerator) {
      idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    }
    auto builder = PlanBuilder(std::move(idGenerator), pool).tableScan(type);
    if (!filter.empty()) {
      builder.filter(filter);
    }
    return builder.planNode();
  }

  core::PlanNodePtr leaf(const core::PlanNodePtr& node) {
    if (node->sources().empty()) {
      return node;
    }
    VELOX_CHECK_EQ(node->sources().size(), 1);
    return leaf(node->sources().front());
  }

  RunResult runOnce(
      const Options& o,
      const std::vector<float>& centers,
      const std::string& input,
      bool captureRows) {
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto candidateSource = scan(
        candidateType(),
        "active AND market = 0",
        pool_.get(),
        idGenerator);
    auto querySource =
        scan(queryType(), "market = 0", pool_.get(), idGenerator);
    const auto candidateLeaf = leaf(candidateSource)->id();
    const auto queryLeaf = leaf(querySource)->id();

    candidateSource = std::make_shared<AssignClustersNode>(
        "candidate_assignment",
        candidateSource,
        "embedding",
        "cluster_id",
        o.dimension,
        centers);
    querySource = std::make_shared<AssignClustersNode>(
        "query_assignment",
        querySource,
        "embedding",
        "cluster_id",
        o.dimension,
        centers);

#if defined(VELOX_ENABLE_FAISS_GPU)
    if (FLAGS_execution != "cpu") {
      candidateSource = std::make_shared<cudf_velox::CudfFromVeloxNode>(
          "candidate_to_cudf", candidateSource);
    }
    if (FLAGS_execution == "gpu") {
      querySource = std::make_shared<cudf_velox::CudfFromVeloxNode>(
          "query_to_cudf", querySource);
    }
#else
    VELOX_USER_CHECK(
        FLAGS_execution == "cpu",
        "GPU execution requested, but VELOX_ENABLE_FAISS_GPU is disabled");
#endif

    core::PlanNodePtr provider;
    if (FLAGS_provider == "build") {
      provider = std::make_shared<BuildIndexNode>(
          "index_provider",
          candidateSource,
          "candidate_id",
          "embedding",
          "cluster_id",
          indexConfig(o),
          FLAGS_prepare_artifacts ? std::optional<std::string>(FLAGS_artifacts)
                                  : std::nullopt);
    } else {
      VELOX_USER_CHECK(!FLAGS_artifacts.empty(), "--artifacts is required");
      provider = std::make_shared<LoadIndexNode>(
          "index_provider", FLAGS_artifacts, indexConfig(o));
    }
    auto plan = std::make_shared<SearchIndexNode>(
        "candidate_retrieval",
        querySource,
        provider,
        "query_id",
        "embedding",
        "cluster_id",
        o.topK);

    AssertQueryBuilder builder(plan);
    builder.maxDrivers(1)
        .split(
            queryLeaf,
            connector::hive::HiveConnectorSplitBuilder(
                input + "/retrieval_queries.parquet")
                .connectorId(exec::test::kHiveConnectorId)
                .fileFormat(dwio::common::FileFormat::PARQUET)
                .build());
    if (FLAGS_provider == "build") {
      builder.split(
          candidateLeaf,
          connector::hive::HiveConnectorSplitBuilder(
              input + "/candidates.parquet")
              .connectorId(exec::test::kHiveConnectorId)
              .fileFormat(dwio::common::FileFormat::PARQUET)
              .build());
    }
    std::shared_ptr<exec::Task> task;
    RunResult result;
    {
      FaissNvtxRange complete("complete candidate retrieval");
      const auto start = Clock::now();
      auto rows = builder.copyResults(pool_.get(), task);
      result.fullQueryMs =
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      if (captureRows) {
        result.rows = std::move(rows);
      }
    }

    for (const auto& pipeline : task->taskStats().pipelineStats) {
      const bool providerPipeline = std::any_of(
          pipeline.operatorStats.begin(),
          pipeline.operatorStats.end(),
          [](const auto& op) { return op.operatorType == "FaissIndexBuild"; });
      for (const auto& op : pipeline.operatorStats) {
        const auto wall = nanosToMs(
            op.addInputTiming.wallNanos + op.getOutputTiming.wallNanos +
            op.finishTiming.wallNanos);
        if (providerPipeline) {
          // Inclusive by design: build includes scan, assignment and
          // construction; load includes source, deserialize and upload.
          result.providerMs += wall;
        }
        if (op.operatorType == "TableScan") {
          result.scanMs += wall;
        } else if (op.operatorType == "FaissAssignClusters") {
          result.assignmentMs += wall;
        }
        for (const auto& [name, stat] : op.runtimeStats) {
          if (name == "faissConversionWallNanos") {
            result.conversionMs += nanosToMs(stat.sum);
          } else if (name == "faissSearchWallNanos") {
            result.searchMs += nanosToMs(stat.sum);
          } else if (name == "faissIdGatherWallNanos") {
            result.outputMs += nanosToMs(stat.sum);
          } else if (name == "faissTrainWallNanos") {
            result.trainMs += nanosToMs(stat.sum);
          } else if (name == "faissAddWallNanos") {
            result.addMs += nanosToMs(stat.sum);
          } else if (name == "faissLoadReadWallNanos") {
            result.loadReadMs += nanosToMs(stat.sum);
          } else if (name == "faissLoadDeserializeWallNanos") {
            result.loadDeserializeMs += nanosToMs(stat.sum);
          } else if (name == "faissLoadUploadWallNanos") {
            result.loadUploadMs += nanosToMs(stat.sum);
          } else if (name == "faissCagraToHnswWallNanos") {
            result.cagraToHnswMs += nanosToMs(stat.sum);
          }
        }
      }
    }
    return result;
  }

  void validate(
      const Options& o,
      const std::vector<float>& centers,
      const RowVectorPtr& actual) {
    if (FLAGS_preset != "ci") {
      return;
    }
    if (FLAGS_strategy == "flat") {
      // Flat is its own exact ground truth. Check complete IDs and cardinality
      // deterministically across a second fresh repetition.
      const auto savedExecution = FLAGS_execution;
      const auto savedProvider = FLAGS_provider;
      FLAGS_execution = "cpu";
      FLAGS_provider = "build";
      const auto second = runOnce(o, centers, FLAGS_input, true).rows;
      FLAGS_execution = savedExecution;
      FLAGS_provider = savedProvider;
      VELOX_USER_CHECK_EQ(actual->size(), second->size());
      for (vector_size_t row = 0; row < actual->size(); ++row) {
        for (const auto child : {0, 1, 3}) {
          VELOX_USER_CHECK(
              actual->childAt(child)->equalValueAt(
                  second->childAt(child).get(), row, row),
              "Flat result ID/rank mismatch at output row {}",
              row);
        }
      }
      return;
    }
    const auto savedStrategy = FLAGS_strategy;
    const auto savedExecution = FLAGS_execution;
    const auto savedProvider = FLAGS_provider;
    FLAGS_strategy = "flat";
    FLAGS_execution = "cpu";
    FLAGS_provider = "build";
    auto truth = runOnce(o, centers, FLAGS_input, true).rows;
    FLAGS_strategy = savedStrategy;
    FLAGS_execution = savedExecution;
    FLAGS_provider = savedProvider;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> expected;
    auto truthQuery = truth->childAt(0)->as<SimpleVector<int64_t>>();
    auto truthId = truth->childAt(1)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < truth->size(); ++row) {
      expected[truthQuery->valueAt(row)].insert(truthId->valueAt(row));
    }
    auto actualQuery = actual->childAt(0)->as<SimpleVector<int64_t>>();
    auto actualId = actual->childAt(1)->as<SimpleVector<int64_t>>();
    int64_t hits = 0;
    for (vector_size_t row = 0; row < actual->size(); ++row) {
      hits += expected[actualQuery->valueAt(row)].count(actualId->valueAt(row));
    }
    const auto denominator = std::max<int64_t>(1, truth->size());
    const auto recall = static_cast<double>(hits) / denominator;
    LOG(INFO) << "recall@" << o.topK << "=" << recall;
    VELOX_USER_CHECK_GE(
        recall,
        FLAGS_min_recall,
        "{} recall@{} is below the required threshold",
        FLAGS_strategy,
        o.topK);
  }

 private:
  using HiveConnectorTestBase::makeHiveConnectorSplit;
};

folly::dynamic summarize(
    const Options& o,
    const std::vector<RunResult>& measured) {
  folly::dynamic report = folly::dynamic::object;
  report["execution"] = FLAGS_execution;
  report["provider"] = FLAGS_provider;
  report["strategy"] = FLAGS_strategy;
  report["candidates"] = o.candidates;
  report["queries"] = o.queries;
  report["dimension"] = o.dimension;
  report["topK"] = o.topK;
  report["repetitions"] = measured.size();
  folly::dynamic phases = folly::dynamic::object;
  for (const auto& [name, getter] :
       std::vector<std::pair<std::string, std::function<double(const RunResult&)>>>{
           {"full_query", [](const auto& r) { return r.fullQueryMs; }},
           {"scan", [](const auto& r) { return r.scanMs; }},
           {"assignment", [](const auto& r) { return r.assignmentMs; }},
           {"provider", [](const auto& r) { return r.providerMs; }},
           {"conversion", [](const auto& r) { return r.conversionMs; }},
           {"search", [](const auto& r) { return r.searchMs; }},
           {"output", [](const auto& r) { return r.outputMs; }},
           {"faiss_train", [](const auto& r) { return r.trainMs; }},
           {"faiss_add", [](const auto& r) { return r.addMs; }},
           {"load_read", [](const auto& r) { return r.loadReadMs; }},
           {"load_deserialize",
            [](const auto& r) { return r.loadDeserializeMs; }},
           {"load_upload", [](const auto& r) { return r.loadUploadMs; }},
           {"cagra_to_hnsw",
            [](const auto& r) { return r.cagraToHnswMs; }}}) {
    double total = 0;
    for (const auto& run : measured) {
      total += getter(run);
    }
    phases[name + "_mean_ms"] = total / measured.size();
  }
  report["phases"] = std::move(phases);
  return report;
}

} // namespace
} // namespace facebook::velox::faiss::benchmark

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  using namespace facebook::velox;
  using namespace facebook::velox::faiss;
  using namespace facebook::velox::faiss::benchmark;
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  memory::SharedArbitrator::registerFactory();
  functions::prestosql::registerAllScalarFunctions();
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
  facebook::velox::faiss::registerFaiss();
#if defined(VELOX_ENABLE_FAISS_GPU)
  if (FLAGS_execution != "cpu") {
    cudf_velox::registerCudf();
  }
#endif

  const auto options = optionsFromFlags();
  validateOptions(options);
  CandidateRetrievalBenchmark benchmark;
  if (FLAGS_prepare_artifacts) {
    benchmark.prepareInputs(options, FLAGS_input);
    VELOX_USER_CHECK(!FLAGS_artifacts.empty(), "--artifacts is required");
    FLAGS_provider = "build";
    benchmark.runOnce(options, benchmark.centroids(options), FLAGS_input, false);
    return 0;
  }
  VELOX_USER_CHECK(!FLAGS_input.empty(), "--input is required");
  const auto centers = benchmark.loadCentroids(options, FLAGS_input);
  for (int32_t i = 0; i < FLAGS_warmups; ++i) {
    benchmark.runOnce(options, centers, FLAGS_input, false);
  }
  std::vector<RunResult> measured;
  for (int32_t i = 0; i < FLAGS_repetitions; ++i) {
    measured.push_back(
        benchmark.runOnce(options, centers, FLAGS_input, i == 0));
  }
  benchmark.validate(options, centers, measured.front().rows);
  const auto output = folly::toPrettyJson(summarize(options, measured));
  if (FLAGS_output.empty()) {
    std::cout << output;
  } else {
    std::ofstream(FLAGS_output) << output;
  }
  return 0;
}
