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

// Times Velox's RE2-based regex functions on the CPU against their cuDF
// versions on the GPU over synthetic string columns, and checks that both
// return the same values. Writes one TSV row per dataset and expression.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/RegisterSpecialForm.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/FlatVector.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

DEFINE_int32(rows, 4'000'000, "Rows in each dataset.");
DEFINE_int32(repeats, 5, "Timed GPU runs per expression.");
DEFINE_int32(cpu_repeats, 3, "Timed CPU runs per expression.");
DEFINE_int32(
    threads,
    0,
    "Threads for the multi-threaded CPU runs; 0 uses every hardware thread.");
DEFINE_string(output, "regex_benchmark.tsv", "TSV file to write.");
DEFINE_string(filter, "", "Only run expressions whose name contains this.");
DEFINE_string(datasets, "short,log", "Comma-separated datasets to run.");

using namespace facebook::velox;

namespace {

using Clock = std::chrono::steady_clock;

double millisSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

const std::vector<std::string> kWords = {
    "alpha", "bravo",  "charlie", "delta",  "echo",   "foxtrot", "golf",
    "hotel", "india",  "juliet",  "lima",   "mike",   "oscar",   "papa",
    "romeo", "tango",  "victor",  "whisky", "yankee", "zulu",    "error",
    "retry", "cache",  "server",  "client", "timeout"};
const std::vector<std::string> kLevels = {
    "INFO", "INFO", "INFO", "DEBUG", "DEBUG", "WARN", "ERROR", "FATAL"};
const std::vector<std::string> kHosts = {
    "www.example.com", "api.test.org", "cdn.nvidia.com", "shop.example.org"};
const std::vector<std::string> kMethods = {"GET", "GET", "POST", "PUT"};

template <typename T>
const T& pick(const std::vector<T>& values, std::mt19937& rng) {
  return values[std::uniform_int_distribution<size_t>(
      0, values.size() - 1)(rng)];
}

int32_t uniform(int32_t low, int32_t high, std::mt19937& rng) {
  return std::uniform_int_distribution<int32_t>(low, high)(rng);
}

// About 120 bytes: a timestamp, a level, and fields that the patterns look
// for, some of them only on some lines.
std::string makeLogLine(std::mt19937& rng) {
  std::string line = fmt::format(
      "2026-{:02}-{:02} {:02}:{:02}:{:02} {} ",
      uniform(1, 12, rng),
      uniform(1, 28, rng),
      uniform(0, 23, rng),
      uniform(0, 59, rng),
      uniform(0, 59, rng),
      pick(kLevels, rng));
  if (uniform(0, 1, rng) == 0) {
    line += fmt::format(
        "user={}.{}@{} ", pick(kWords, rng), pick(kWords, rng), pick(kHosts, rng));
  }
  line += fmt::format(
      "ip={}.{}.{}.{} ",
      uniform(1, 254, rng),
      uniform(0, 255, rng),
      uniform(0, 255, rng),
      uniform(1, 254, rng));
  if (uniform(0, 3, rng) != 0) {
    line += fmt::format(
        "{} https://{}/{}/{} ",
        pick(kMethods, rng),
        pick(kHosts, rng),
        pick(kWords, rng),
        uniform(1, 99999, rng));
  }
  line += fmt::format(
      "status={} latency={}ms {}",
      pick(std::vector<int32_t>{200, 200, 200, 201, 404, 500, 503}, rng),
      uniform(1, 2000, rng),
      pick(kWords, rng));
  return line;
}

// About 20 bytes: one to three short tokens.
std::string makeShortString(std::mt19937& rng) {
  std::string s;
  const auto tokens = uniform(1, 3, rng);
  for (int32_t i = 0; i < tokens; ++i) {
    if (i > 0) {
      s += ' ';
    }
    switch (uniform(0, 5, rng)) {
      case 0:
        s += fmt::format("{}{}", pick(kWords, rng), uniform(0, 99, rng));
        break;
      case 1:
        s += fmt::format("{}-{:04}", uniform(100, 999, rng), uniform(0, 9999, rng));
        break;
      case 2:
        s += fmt::format("{}@{}", pick(kWords, rng), pick(kHosts, rng));
        break;
      case 3:
        s += fmt::format("2026-{:02}-{:02}", uniform(1, 12, rng), uniform(1, 28, rng));
        break;
      default:
        s += pick(kWords, rng);
        break;
    }
  }
  return s;
}

RowVectorPtr makeDataset(
    const std::string& name,
    int32_t rows,
    memory::MemoryPool* pool,
    double& averageBytes) {
  std::mt19937 rng(20261125);
  auto vector = BaseVector::create<FlatVector<StringView>>(VARCHAR(), rows, pool);
  int64_t totalBytes = 0;
  for (int32_t i = 0; i < rows; ++i) {
    const auto s = name == "log" ? makeLogLine(rng) : makeShortString(rng);
    totalBytes += s.size();
    vector->set(i, StringView(s));
  }
  averageBytes = static_cast<double>(totalBytes) / rows;
  return std::make_shared<RowVector>(
      pool, ROW({"c0"}, {VARCHAR()}), nullptr, rows, std::vector<VectorPtr>{vector});
}

struct Case {
  std::string name;
  std::string function;
  std::string expression;
};

const std::vector<Case>& cases() {
  static const std::vector<Case> kCases = {
      {"literal", "regexp_like", "regexp_like(c0, 'error')"},
      {"anchored", "regexp_like", "regexp_like(c0, '^2026-0[1-6]')"},
      {"alternation", "regexp_like", "regexp_like(c0, 'ERROR|WARN|FATAL')"},
      {"ignore_case", "regexp_like", "regexp_like(c0, '(?i)error')"},
      {"dot_star", "regexp_like", "regexp_like(c0, 'user=.*example.*status=5')"},
      {"email", "regexp_like", "regexp_like(c0, '[\\w.+-]+@[\\w-]+\\.(com|org)')"},
      {"ipv4",
       "regexp_like",
       "regexp_like(c0, '\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}')"},
      {"word_list",
       "regexp_like",
       "regexp_like(c0, 'alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|"
       "india|juliet|lima|mike|oscar|papa|romeo|tango|victor|whisky|yankee|"
       "zulu')"},
      {"url_host",
       "regexp_extract",
       "regexp_extract(c0, 'https?://([^/\\s]+)', 1)"},
      {"latency",
       "regexp_extract",
       "regexp_extract(c0, 'latency=(\\d+)ms', 1)"},
      {"date", "regexp_extract", "regexp_extract(c0, '\\d{4}-\\d{2}-\\d{2}')"},
      {"numbers", "regexp_extract_all", "regexp_extract_all(c0, '\\d+')"},
      {"words", "regexp_extract_all", "regexp_extract_all(c0, '[a-z]+')"},
      {"digits", "regexp_replace", "regexp_replace(c0, '\\d', '#')"},
      {"backrefs",
       "regexp_replace",
       "regexp_replace(c0, '(\\w+)@([\\w.]+)', '$2 at $1')"},
      {"delimiters", "regexp_split", "regexp_split(c0, '[\\s=/]+')"},
  };
  return kCases;
}

class RegexBenchmark {
 public:
  RegexBenchmark() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
    parse::registerTypeResolver();
    exec::registerFunctionCallToSpecialForms();
    functions::prestosql::registerAllScalarFunctions();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions("");
    rootPool_ = memory::memoryManager()->addRootPool("regex_benchmark");
    pool_ = rootPool_->addLeafChild("main", true);
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    threads_ = FLAGS_threads > 0
        ? FLAGS_threads
        : static_cast<int32_t>(std::thread::hardware_concurrency());
  }

  void run() {
    std::ofstream out(FLAGS_output);
    out << "dataset\trows\tavg_bytes\tcase\tfunction\texpression\tgpu_supported"
           "\tcpu_1t_ms\tcpu_mt_ms\tcpu_threads\tgpu_ms\tgpu_to_host_ms"
           "\th2d_ms\tmismatches\n";
    for (const auto& dataset : splitList(FLAGS_datasets)) {
      double averageBytes;
      auto input = makeDataset(dataset, FLAGS_rows, pool_.get(), averageBytes);
      auto rowType = asRowType(input->type());
      std::cout << fmt::format(
                       "dataset {}: {} rows, {:.1f} bytes per row",
                       dataset,
                       FLAGS_rows,
                       averageBytes)
                << std::endl;

      auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
      auto start = Clock::now();
      auto table = cudf_velox::with_arrow::toCudfTable(
          input, pool_.get(), stream, cudf_velox::get_output_mr());
      stream.sync();
      const double h2dMs = millisSince(start);
      std::vector<cudf::column_view> inputViews;
      for (cudf::size_type i = 0; i < table->num_columns(); ++i) {
        inputViews.push_back(table->view().column(i));
      }

      for (const auto& c : cases()) {
        if (!FLAGS_filter.empty() &&
            c.name.find(FLAGS_filter) == std::string::npos &&
            c.function.find(FLAGS_filter) == std::string::npos) {
          continue;
        }
        runCase(
            out, dataset, averageBytes, input, rowType, inputViews, h2dMs, c, stream);
      }
    }
  }

 private:
  static std::vector<std::string> splitList(const std::string& list) {
    std::vector<std::string> items;
    size_t start = 0;
    while (start <= list.size()) {
      const auto end = std::min(list.find(',', start), list.size());
      if (end > start) {
        items.push_back(list.substr(start, end - start));
      }
      start = end + 1;
    }
    return items;
  }

  core::TypedExprPtr parse(const std::string& sql, const RowTypePtr& rowType) {
    auto untyped = parse::DuckSqlExpressionsParser().parseExpr(sql);
    return core::Expressions::inferTypes(untyped, rowType, pool_.get());
  }

  VectorPtr evalCpu(
      const core::TypedExprPtr& expr,
      const RowVectorPtr& input,
      core::ExecCtx* execCtx) {
    exec::ExprSet exprSet({expr}, execCtx);
    exec::EvalCtx evalCtx(execCtx, &exprSet, input.get());
    SelectivityVector rows(input->size());
    std::vector<VectorPtr> results(1);
    exprSet.eval(rows, evalCtx, results);
    return results[0];
  }

  // Splits the input into one slice per thread, each evaluated with its own
  // memory pool and contexts.
  double timeCpuParallel(
      const core::TypedExprPtr& expr,
      const RowVectorPtr& input) {
    const vector_size_t rowsPerThread =
        (input->size() + threads_ - 1) / threads_;
    std::vector<std::shared_ptr<memory::MemoryPool>> pools;
    std::vector<std::shared_ptr<core::QueryCtx>> queryCtxs;
    std::vector<std::unique_ptr<core::ExecCtx>> execCtxs;
    std::vector<RowVectorPtr> slices;
    for (int32_t t = 0; t < threads_; ++t) {
      const vector_size_t offset = t * rowsPerThread;
      if (offset >= input->size()) {
        break;
      }
      const auto length = std::min(rowsPerThread, input->size() - offset);
      slices.push_back(
          std::dynamic_pointer_cast<RowVector>(input->slice(offset, length)));
      pools.push_back(rootPool_->addLeafChild(fmt::format("cpu{}", t), true));
      queryCtxs.push_back(core::QueryCtx::create());
      execCtxs.push_back(std::make_unique<core::ExecCtx>(
          pools.back().get(), queryCtxs.back().get()));
    }
    auto start = Clock::now();
    std::vector<std::thread> workers;
    for (size_t t = 0; t < slices.size(); ++t) {
      workers.emplace_back([&, t] {
        auto result = evalCpu(expr, slices[t], execCtxs[t].get());
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    return millisSince(start);
  }

  void runCase(
      std::ofstream& out,
      const std::string& dataset,
      double averageBytes,
      const RowVectorPtr& input,
      const RowTypePtr& rowType,
      const std::vector<cudf::column_view>& inputViews,
      double h2dMs,
      const Case& c,
      cuda::stream_ref stream) {
    auto typed = parse(c.expression, rowType);
    auto optimized =
        expression::optimize(typed, queryCtx_.get(), pool_.get());
    const bool gpuSupported =
        cudf_velox::canExprRunOnGpu(typed, queryCtx_.get(), pool_.get());

    std::vector<double> cpuTimes;
    VectorPtr cpuResult;
    for (int32_t i = 0; i < FLAGS_cpu_repeats; ++i) {
      auto start = Clock::now();
      cpuResult = evalCpu(optimized, input, execCtx_.get());
      cpuTimes.push_back(millisSince(start));
    }
    std::vector<double> cpuParallelTimes;
    for (int32_t i = 0; i < FLAGS_cpu_repeats; ++i) {
      cpuParallelTimes.push_back(timeCpuParallel(optimized, input));
    }

    double gpuMs = -1;
    double toHostMs = -1;
    int64_t mismatches = -1;
    if (gpuSupported) {
      auto mr = cudf_velox::get_output_mr();
      auto cudfExpr = cudf_velox::createCudfExpression(
          optimized, rowType, pool_.get(), queryCtx_->queryConfig());
      // The first run builds the device program and warms the allocator.
      {
        auto warm = cudfExpr->eval(inputViews, stream, mr);
        stream.sync();
      }
      std::vector<double> gpuTimes;
      for (int32_t i = 0; i < FLAGS_repeats; ++i) {
        auto start = Clock::now();
        auto result = cudfExpr->eval(inputViews, stream, mr);
        stream.sync();
        gpuTimes.push_back(millisSince(start));
      }
      gpuMs = median(gpuTimes);

      auto result = cudfExpr->eval(inputViews, stream, mr);
      auto start = Clock::now();
      cudf::table_view resultTable({cudf_velox::asView(result)});
      auto outputType = ROW({"c0"}, {optimized->type()});
      auto gpuRow = cudf_velox::with_arrow::toVeloxColumn(
          resultTable, pool_.get(), outputType, "", stream, mr);
      stream.sync();
      toHostMs = millisSince(start);
      auto gpuResult = gpuRow->childAt(0);
      mismatches = 0;
      for (vector_size_t row = 0; row < input->size(); ++row) {
        if (!cpuResult->equalValueAt(gpuResult.get(), row, row)) {
          if (mismatches < 3) {
            std::cout << fmt::format(
                             "  mismatch row {}: input '{}' cpu {} gpu {}",
                             row,
                             input->childAt(0)->toString(row),
                             cpuResult->toString(row),
                             gpuResult->toString(row))
                      << std::endl;
          }
          ++mismatches;
        }
      }
    }

    const double cpuMs = median(cpuTimes);
    const double cpuParallelMs = median(cpuParallelTimes);
    std::cout << fmt::format(
                     "{:6} {:20} {:12} cpu {:9.1f} ms  cpu x{} {:8.1f} ms  "
                     "gpu {:8.2f} ms  speedup {:6.1f}x / {:5.1f}x  mismatches {}",
                     dataset,
                     c.function,
                     c.name,
                     cpuMs,
                     threads_,
                     cpuParallelMs,
                     gpuMs,
                     gpuMs > 0 ? cpuMs / gpuMs : 0.0,
                     gpuMs > 0 ? cpuParallelMs / gpuMs : 0.0,
                     mismatches)
              << std::endl;
    out << fmt::format(
        "{}\t{}\t{:.1f}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\n",
        dataset,
        input->size(),
        averageBytes,
        c.name,
        c.function,
        c.expression,
        gpuSupported,
        cpuMs,
        cpuParallelMs,
        threads_,
        gpuMs,
        toHostMs,
        h2dMs,
        mismatches);
    out.flush();
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  int32_t threads_;
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  RegexBenchmark().run();
  return 0;
}
