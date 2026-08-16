/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "velox/experimental/faiss/FaissOperators.h"

#if defined(VELOX_ENABLE_FAISS_GPU)
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include <nvtx3/nvToolsExt.h>
#endif

#include "velox/common/file/LocalFile.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>

DEFINE_int64(candidates, 10'000, "Candidate rows");
DEFINE_int64(queries, 1'000, "Query rows");
DEFINE_int32(dimension, 128, "Embedding dimension");
DEFINE_int32(topK, 10, "Neighbors per query");
DEFINE_int32(clusters, 8, "Routed cluster count");
DEFINE_string(
    data_directory,
    "",
    "Persistent input/artifact directory; default data is temporary");
DEFINE_double(
    min_recall,
    0.1,
    "Minimum recall@k required for approximate strategies");

namespace facebook::velox::faiss::benchmark {
namespace {

using common::testutil::TempDirectoryPath;
using exec::test::AssertQueryBuilder;
using exec::test::HiveConnectorTestBase;
using exec::test::PlanBuilder;

constexpr int64_t kMaxListChildren = 2'000'000'000LL - 1;
constexpr int32_t kRowsPerBatch = 4'096;
#if defined(VELOX_ENABLE_FAISS_GPU)
constexpr std::string_view kGpuHiveConnectorId = "faiss-gpu-hive";
#endif

class FaissNvtxProcessRange {
 public:
  explicit FaissNvtxProcessRange(const char* name) {
#if defined(VELOX_ENABLE_FAISS_GPU)
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = 0xFFFFA500; // Orange.
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    rangeId_ = nvtxDomainRangeStartEx(domain(), &attributes);
#else
    (void)name;
#endif
  }

  ~FaissNvtxProcessRange() {
#if defined(VELOX_ENABLE_FAISS_GPU)
    nvtxDomainRangeEnd(domain(), rangeId_);
#endif
  }

  FaissNvtxProcessRange(const FaissNvtxProcessRange&) = delete;
  FaissNvtxProcessRange& operator=(const FaissNvtxProcessRange&) = delete;

 private:
#if defined(VELOX_ENABLE_FAISS_GPU)
  static nvtxDomainHandle_t domain() {
    static const auto handle = nvtxDomainCreateA("velox");
    return handle;
  }

  nvtxRangeId_t rangeId_;
#endif
};

enum class Provider { kBuild, kLoad };

struct Options {
  int64_t candidates;
  int64_t queries;
  int32_t dimension;
  int32_t topK;
  int32_t clusters;
};

struct BenchmarkCase {
  std::string name;
  FaissExecutionDevice device;
  FaissAlgorithm algorithm;
  Provider provider;
};

struct PreparedRun {
  core::PlanNodePtr plan;
  core::PlanNodeId queryScanId;
  std::optional<core::PlanNodeId> candidateScanId;
  std::string connectorId;
};

std::string algorithmName(FaissAlgorithm algorithm) {
  switch (algorithm) {
    case FaissAlgorithm::kFlat:
      return "flat";
    case FaissAlgorithm::kHnsw:
      return "hnsw";
    case FaissAlgorithm::kCagra:
      return "cagra";
    default:
      VELOX_UNREACHABLE();
  }
}

Options optionsFromFlags() {
  return {
      FLAGS_candidates,
      FLAGS_queries,
      FLAGS_dimension,
      FLAGS_topK,
      FLAGS_clusters};
}

void validateOptions(const Options& options) {
  VELOX_USER_CHECK_GT(options.candidates, 0, "Candidates must be positive");
  VELOX_USER_CHECK_GT(options.queries, 0, "Queries must be positive");
  VELOX_USER_CHECK_GT(options.dimension, 0, "Dimension must be positive");
  VELOX_USER_CHECK_GT(options.topK, 0, "topK must be positive");
  VELOX_USER_CHECK_GT(options.clusters, 0, "Clusters must be positive");
  VELOX_USER_CHECK_LE(
      options.dimension,
      kMaxListChildren,
      "Embedding dimension exceeds Parquet/Velox list-child capacity");
  const auto rowsPerCluster =
      options.candidates * 19 / 20 / options.clusters;
  VELOX_USER_CHECK_GT(
      rowsPerCluster,
      64,
      "The workload leaves too few post-filter candidates per cluster for "
      "HNSW/CAGRA graph degree 64");
  VELOX_USER_CHECK_GE(
      rowsPerCluster,
      options.topK,
      "topK exceeds post-filter candidates per cluster");
}

class CandidateRetrievalBenchmark final : public HiveConnectorTestBase {
 public:
  explicit CandidateRetrievalBenchmark(Options options)
      : options_(options), centers_(makeCentroids()) {
    HiveConnectorTestBase::SetUp();
    if (FLAGS_data_directory.empty()) {
      temporaryDirectory_ = TempDirectoryPath::create();
      dataDirectory_ = temporaryDirectory_->getPath();
    } else {
      dataDirectory_ = FLAGS_data_directory;
      std::filesystem::create_directories(dataDirectory_);
    }
#if defined(VELOX_ENABLE_FAISS_GPU)
    cudf_velox::connector::hive::CudfHiveConnectorFactory factory;
    auto gpuConnector = factory.newConnector(
        std::string(kGpuHiveConnectorId),
        std::make_shared<const config::ConfigBase>(
            std::unordered_map<std::string, std::string>{}),
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        gpuConnector->connectorId(), gpuConnector);
#endif
    prepareInputs();
    for (const auto& benchmarkCase : cases()) {
      ensureArtifact(benchmarkCase);
    }
  }

  ~CandidateRetrievalBenchmark() override {
#if defined(VELOX_ENABLE_FAISS_GPU)
    connector::ConnectorRegistry::global().erase(
        std::string(kGpuHiveConnectorId));
    cudf_velox::unregisterCudf();
#endif
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  void addBenchmarks() {
    for (const auto& benchmarkCase : cases()) {
      cases_.push_back(std::make_unique<BenchmarkCase>(benchmarkCase));
      const auto* testCase = cases_.back().get();
      folly::addBenchmark(
          __FILE__,
          testCase->name,
          [this, testCase](
              folly::UserCounters& counters, unsigned iterations) {
            folly::BenchmarkSuspender setupSuspender;
            ensureArtifact(*testCase);
            ensureValidated(*testCase);
            setupSuspender.dismiss();

            uint64_t outputRows = 0;
            for (unsigned iteration = 0; iteration < iterations; ++iteration) {
              folly::BenchmarkSuspender planSuspender;
              auto run = prepareRun(*testCase);
              planSuspender.dismiss();
              const auto repetitionName =
                  fmt::format("BenchmarkIteration::{}", testCase->name);
              FaissNvtxProcessRange repetition(repetitionName.c_str());
              auto result = execute(run);
              outputRows += result->size();
            }
            BENCHMARK_SUSPEND {
              counters["output_rows"] = folly::UserMetric(
                  static_cast<int64_t>(outputRows / iterations),
                  folly::UserMetric::Type::METRIC);
              folly::doNotOptimizeAway(outputRows);
            }
            return iterations;
          });
    }
  }

 private:
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

  std::vector<BenchmarkCase> cases() const {
    std::vector<BenchmarkCase> result{
        {"CandidateRetrieval/CPU/FlatBruteForce/Build",
         FaissExecutionDevice::kCpu,
         FaissAlgorithm::kFlat,
         Provider::kBuild},
        {"CandidateRetrieval/CPU/FlatBruteForce/Load",
         FaissExecutionDevice::kCpu,
         FaissAlgorithm::kFlat,
         Provider::kLoad},
        {"CandidateRetrieval/CPU/HNSW/Build",
         FaissExecutionDevice::kCpu,
         FaissAlgorithm::kHnsw,
         Provider::kBuild},
        {"CandidateRetrieval/CPU/HNSW/Load",
         FaissExecutionDevice::kCpu,
         FaissAlgorithm::kHnsw,
         Provider::kLoad}};
#if defined(VELOX_ENABLE_FAISS_GPU)
    result.insert(
        result.end(),
        {{"CandidateRetrieval/GPU/FlatCuVSBruteForce/Build",
          FaissExecutionDevice::kGpu,
          FaissAlgorithm::kFlat,
          Provider::kBuild},
         {"CandidateRetrieval/GPU/FlatCuVSBruteForce/LoadCpuArtifact",
          FaissExecutionDevice::kGpu,
          FaissAlgorithm::kFlat,
          Provider::kLoad},
         {"CandidateRetrieval/GPU/CAGRA/Build",
          FaissExecutionDevice::kGpu,
          FaissAlgorithm::kCagra,
          Provider::kBuild},
         {"CandidateRetrieval/GPU/CAGRA/LoadCpuGraphArtifact",
          FaissExecutionDevice::kGpu,
          FaissAlgorithm::kCagra,
          Provider::kLoad}});
#endif
    return result;
  }

  std::vector<float> makeCentroids() const {
    std::mt19937 random(41);
    std::normal_distribution<float> value(0.0F, 1.0F);
    std::vector<float> result(
        static_cast<size_t>(options_.clusters) * options_.dimension);
    for (int32_t cluster = 0; cluster < options_.clusters; ++cluster) {
      float norm = 0;
      for (int32_t dimension = 0; dimension < options_.dimension;
           ++dimension) {
        auto& element =
            result[static_cast<size_t>(cluster) * options_.dimension +
                   dimension];
        element = value(random);
        norm += element * element;
      }
      const auto scale = 10.0F / std::sqrt(norm);
      for (int32_t dimension = 0; dimension < options_.dimension;
           ++dimension) {
        result[static_cast<size_t>(cluster) * options_.dimension + dimension] *=
            scale;
      }
    }
    return result;
  }

  RowVectorPtr generateBatch(
      int64_t base,
      vector_size_t count,
      bool candidates,
      std::mt19937& random) {
    std::normal_distribution<float> noise(0.0F, 0.05F);
    auto ids = makeFlatVector<int64_t>(
        count, [&](auto row) { return base + row; });
    auto embeddings = makeArrayVector<float>(
        count,
        [&](auto) { return options_.dimension; },
        [&](auto row, auto dimension) {
          const auto id = base + row;
          const auto cluster = id % options_.clusters;
          return centers_[cluster * options_.dimension + dimension] +
              noise(random);
        });
    auto markets = makeFlatVector<int64_t>(count, [&](auto row) {
      return (base + row) % 40 == 0 ? 1 : 0;
    });
    if (!candidates) {
      return makeRowVector(
          {"query_id", "embedding", "market"}, {ids, embeddings, markets});
    }
    auto active = makeFlatVector<bool>(
        count, [&](auto row) { return (base + row) % 40 != 1; });
    return makeRowVector(
        {"candidate_id", "embedding", "market", "active"},
        {ids, embeddings, markets, active});
  }

  template <typename BatchFactory>
  void writeParquet(
      const std::string& path,
      const RowTypePtr& type,
      int64_t rows,
      BatchFactory&& batchFactory) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::make_unique<LocalWriteFile>(path, true, false), path);
    auto writerPool = rootPool_->addAggregateChild("faiss-benchmark-writer");
    dwio::common::WriterOptions writerOptions;
    writerOptions.memoryPool = writerPool.get();
    writerOptions.formatSpecificOptions =
        std::make_shared<parquet::ParquetWriterOptions>();
    parquet::Writer writer(std::move(sink), writerOptions, writerPool, type);
    const auto rowsPerBatch = std::max<int64_t>(
        1,
        std::min<int64_t>(
            kRowsPerBatch, kMaxListChildren / options_.dimension));
    for (int64_t base = 0; base < rows; base += rowsPerBatch) {
      const auto count = static_cast<vector_size_t>(
          std::min<int64_t>(rowsPerBatch, rows - base));
      writer.write(batchFactory(base, count));
    }
    writer.close();
  }

  void prepareInputs() {
    std::mt19937 candidateRandom(17);
    writeParquet(
        dataDirectory_ + "/candidates.parquet",
        candidateType(),
        options_.candidates,
        [&](int64_t base, vector_size_t count) {
          return generateBatch(base, count, true, candidateRandom);
        });
    std::mt19937 queryRandom(23);
    writeParquet(
        dataDirectory_ + "/retrieval_queries.parquet",
        queryType(),
        options_.queries,
        [&](int64_t base, vector_size_t count) {
          return generateBatch(base, count, false, queryRandom);
        });
  }

  FaissIndexConfig indexConfig(const BenchmarkCase& benchmarkCase) const {
    FaissIndexConfig config;
    config.algorithm = benchmarkCase.algorithm;
    config.executionDevice = benchmarkCase.device;
    config.dimension = options_.dimension;
    config.hnswM = 32;
    config.efConstruction = 80;
    config.efSearch = std::max(64, options_.topK * 4);
    config.validate();
    return config;
  }

  std::string connectorId(FaissExecutionDevice device) const {
#if defined(VELOX_ENABLE_FAISS_GPU)
    if (device == FaissExecutionDevice::kGpu) {
      return std::string(kGpuHiveConnectorId);
    }
#endif
    return exec::test::kHiveConnectorId;
  }

  core::PlanNodePtr scan(
      const RowTypePtr& type,
      const std::string& filter,
      const std::string& connector,
      const std::shared_ptr<core::PlanNodeIdGenerator>& idGenerator) {
    PlanBuilder builder(idGenerator, pool_.get());
    PlanBuilder::TableScanBuilder(builder)
        .connectorId(connector)
        .outputType(type)
        .endTableScan();
    if (!filter.empty()) {
      builder.filter(filter);
    }
    return builder.planNode();
  }

  core::PlanNodePtr leaf(const core::PlanNodePtr& node) const {
    if (node->sources().empty()) {
      return node;
    }
    VELOX_CHECK_EQ(node->sources().size(), 1);
    return leaf(node->sources().front());
  }

  std::string artifactDirectory(const BenchmarkCase& benchmarkCase) const {
    return fmt::format(
        "{}/artifacts/{}-{}",
        dataDirectory_,
        benchmarkCase.device == FaissExecutionDevice::kCpu ? "cpu" : "gpu",
        algorithmName(benchmarkCase.algorithm));
  }

  PreparedRun prepareRun(
      const BenchmarkCase& benchmarkCase,
      std::optional<std::string> writeArtifact = std::nullopt) {
    const auto connector = connectorId(benchmarkCase.device);
    auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto querySource =
        scan(queryType(), "market = 0", connector, idGenerator);
    const auto queryScanId = leaf(querySource)->id();
    querySource = std::make_shared<AssignClustersNode>(
        "query_assignment",
        querySource,
        "embedding",
        "cluster_id",
        options_.dimension,
        centers_,
        FaissMetric::kL2,
        benchmarkCase.device);

    core::PlanNodePtr provider;
    std::optional<core::PlanNodeId> candidateScanId;
    if (benchmarkCase.provider == Provider::kBuild || writeArtifact) {
      auto candidateSource =
          scan(candidateType(), "active AND market = 0", connector, idGenerator);
      candidateScanId = leaf(candidateSource)->id();
      candidateSource = std::make_shared<AssignClustersNode>(
          "candidate_assignment",
          candidateSource,
          "embedding",
          "cluster_id",
          options_.dimension,
          centers_,
          FaissMetric::kL2,
          benchmarkCase.device);
      provider = std::make_shared<BuildIndexNode>(
          "index_provider",
          candidateSource,
          "candidate_id",
          "embedding",
          "cluster_id",
          indexConfig(benchmarkCase),
          std::move(writeArtifact));
    } else {
      provider = std::make_shared<LoadIndexNode>(
          "index_provider",
          artifactDirectory(benchmarkCase),
          indexConfig(benchmarkCase));
    }
    return {
        std::make_shared<SearchIndexNode>(
            "candidate_retrieval",
            querySource,
            provider,
            "query_id",
            "embedding",
            "cluster_id",
            options_.topK),
        queryScanId,
        candidateScanId,
        connector};
  }

  RowVectorPtr execute(
      const PreparedRun& run,
      std::shared_ptr<exec::Task>* taskOut = nullptr) {
    AssertQueryBuilder builder(run.plan);
    builder.serialExecution(true).maxDrivers(1).split(
        run.queryScanId,
        connector::hive::HiveConnectorSplitBuilder(
            dataDirectory_ + "/retrieval_queries.parquet")
            .connectorId(run.connectorId)
            .fileFormat(dwio::common::FileFormat::PARQUET)
            .build());
    if (run.candidateScanId) {
      builder.split(
          *run.candidateScanId,
          connector::hive::HiveConnectorSplitBuilder(
              dataDirectory_ + "/candidates.parquet")
              .connectorId(run.connectorId)
              .fileFormat(dwio::common::FileFormat::PARQUET)
              .build());
    }
#if defined(VELOX_ENABLE_FAISS_GPU)
    builder.config(
        cudf_velox::CudfConfig::kCudfEnabled,
        run.connectorId == kGpuHiveConnectorId);
#endif
    std::shared_ptr<exec::Task> task;
    auto result = builder.copyResults(pool_.get(), task);
    if (taskOut) {
      *taskOut = std::move(task);
    }
    return result;
  }

  void ensureArtifact(const BenchmarkCase& benchmarkCase) {
    if (benchmarkCase.provider != Provider::kLoad) {
      return;
    }
    const auto directory = artifactDirectory(benchmarkCase);
    if (preparedArtifacts_.insert(directory).second) {
      std::filesystem::remove_all(directory);
      auto buildCase = benchmarkCase;
      buildCase.provider = Provider::kBuild;
      execute(prepareRun(buildCase, directory));
    }
  }

  static std::map<std::pair<int64_t, int32_t>, int64_t> exactResults(
      const RowVectorPtr& rows) {
    std::map<std::pair<int64_t, int32_t>, int64_t> result;
    const auto* queries = rows->childAt(0)->as<SimpleVector<int64_t>>();
    const auto* ids = rows->childAt(1)->as<SimpleVector<int64_t>>();
    const auto* ranks = rows->childAt(3)->as<SimpleVector<int32_t>>();
    for (vector_size_t row = 0; row < rows->size(); ++row) {
      result[{queries->valueAt(row), ranks->valueAt(row)}] = ids->valueAt(row);
    }
    return result;
  }

  void verifyResidency(
      const BenchmarkCase& benchmarkCase,
      const std::shared_ptr<exec::Task>& task) const {
    bool sawAssignment = false;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        VELOX_USER_CHECK(
            op.operatorType.find("CudfToVelox") == std::string::npos &&
                op.operatorType.find("CudfFromVelox") == std::string::npos,
            "{} unexpectedly contains vector conversion operator {}",
            benchmarkCase.name,
            op.operatorType);
        if (op.operatorType == "FaissAssignClusters" ||
            op.operatorType == "FaissGpuAssignClusters") {
          sawAssignment = true;
          const auto expected =
              benchmarkCase.device == FaissExecutionDevice::kGpu
              ? "FaissGpuAssignClusters"
              : "FaissAssignClusters";
          VELOX_USER_CHECK_EQ(op.operatorType, expected);
        }
      }
    }
    VELOX_USER_CHECK(
        sawAssignment, "{} did not execute cluster assignment", benchmarkCase.name);
  }

  void ensureValidated(const BenchmarkCase& benchmarkCase) {
    if (!validatedCases_.insert(benchmarkCase.name).second) {
      return;
    }
    std::shared_ptr<exec::Task> task;
    auto actual = execute(prepareRun(benchmarkCase), &task);
    verifyResidency(benchmarkCase, task);

    BenchmarkCase truthCase{
        "validation-truth",
        FaissExecutionDevice::kCpu,
        FaissAlgorithm::kFlat,
        Provider::kBuild};
    auto truth = execute(prepareRun(truthCase));
    if (benchmarkCase.algorithm == FaissAlgorithm::kFlat &&
        benchmarkCase.device == FaissExecutionDevice::kCpu) {
      VELOX_USER_CHECK(
          exactResults(actual) == exactResults(truth),
          "{} exact result mismatch",
          benchmarkCase.name);
      return;
    }

    std::unordered_map<int64_t, std::unordered_set<int64_t>> expected;
    const auto* truthQueries =
        truth->childAt(0)->as<SimpleVector<int64_t>>();
    const auto* truthIds = truth->childAt(1)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < truth->size(); ++row) {
      expected[truthQueries->valueAt(row)].insert(truthIds->valueAt(row));
    }
    const auto* actualQueries =
        actual->childAt(0)->as<SimpleVector<int64_t>>();
    const auto* actualIds = actual->childAt(1)->as<SimpleVector<int64_t>>();
    int64_t hits = 0;
    for (vector_size_t row = 0; row < actual->size(); ++row) {
      hits += expected[actualQueries->valueAt(row)].count(
          actualIds->valueAt(row));
    }
    const auto recall =
        static_cast<double>(hits) / std::max<vector_size_t>(1, truth->size());
    LOG(INFO) << benchmarkCase.name << " recall@" << options_.topK << "="
              << recall;
    const auto requiredRecall =
        benchmarkCase.algorithm == FaissAlgorithm::kFlat ? 0.995
                                                         : FLAGS_min_recall;
    VELOX_USER_CHECK_GE(
        recall,
        requiredRecall,
        "{} recall@{} is below the required threshold",
        benchmarkCase.name,
        options_.topK);
  }

  Options options_;
  std::vector<float> centers_;
  std::string dataDirectory_;
  std::shared_ptr<TempDirectoryPath> temporaryDirectory_;
  std::vector<std::unique_ptr<BenchmarkCase>> cases_;
  std::unordered_set<std::string> preparedArtifacts_;
  std::unordered_set<std::string> validatedCases_;
};

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
#if defined(VELOX_ENABLE_FAISS_GPU)
  cudf_velox::registerCudf();
#endif
  registerFaiss();

  const auto options = optionsFromFlags();
  validateOptions(options);
  const auto benchmarkRangeName = fmt::format(
      "CandidateRetrievalBenchmark candidates={} queries={} dimension={} "
      "topK={} clusters={}",
      options.candidates,
      options.queries,
      options.dimension,
      options.topK,
      options.clusters);
  FaissNvtxProcessRange benchmarkRange(benchmarkRangeName.c_str());
  CandidateRetrievalBenchmark benchmark(options);
  benchmark.addBenchmarks();
  folly::runBenchmarks();
  return 0;
}
