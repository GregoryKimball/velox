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

// Measures FilterProject and GroupedAggregation from a Values source:
//
//   Values -> filter -> M polynomial projections -> sum of each, grouped by key
//
// The schema is one BIGINT key of cardinality k plus N DOUBLE columns. The
// filter compares the first DOUBLE column against a constant. Each of the M
// projections reads --projection_fanin of those columns and sums one degree-D
// polynomial per column, in Horner form. The aggregation sums each projection,
// grouped by the key.
//
// Three arms, one per process, since cuDF reads the evaluator choice once when
// its adapter registers:
//
//   --engine=cpu
//   --engine=cudf --evaluator=ast
//   --engine=cudf --evaluator=jit
//
// Correctness is a checksum within a tolerance rather than an exact match,
// because NVRTC contracts a*b+c into an FMA and the CPU does not, so the
// results differ in the last ulp by design. Run cpu first and pass its
// checksum with --expect_checksum.
//
// Sweeping comes from QueryBenchmarkBase: put one flag per line in a
// --test_flags_file and every combination is run, then reported at the end,
// sorted and labelled with the flag values that produced it.
//
//   printf 'num_projections:8,32,128\nbatch_size:100000,1000000\n' > grid
//   velox_cudf_filter_project_benchmark --engine=cudf --evaluator=jit \
//       --test_flags_file=grid

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/benchmarks/QueryBenchmarkBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <sys/resource.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <random>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

DECLARE_int32(num_drivers);

DEFINE_string(engine, "cudf", "One of: cpu, cudf.");

DEFINE_int64(num_rows, 20'000'000, "Total rows in the input.");

DEFINE_int64(
    total_bytes,
    0,
    "If set, derive --num_rows so the input holds this many bytes at any "
    "column count, as rows = bytes / (8 * (columns + 1)). Keeps the data "
    "volume constant when a sweep changes the schema.");

DEFINE_int32(batch_size, 100'000, "Rows per input RowVector.");

DEFINE_int32(num_columns, 8, "Number of DOUBLE columns, N.");

DEFINE_int32(
    num_projections,
    8,
    "Number of polynomial projections, M. 0 matches --num_columns, which is "
    "what a sweep of the column count wants: the projections have to grow with "
    "the schema to keep reading all of it.");

DEFINE_int32(poly_degree, 4, "Polynomial order, D. Evaluated in Horner form.");

DEFINE_int32(
    projection_fanin,
    4,
    "Distinct input columns each projection reads; the projection is a sum of "
    "that many polynomials, one per column. Clamped to --num_columns. Every "
    "column has to be read by something: an unread one still costs data "
    "movement that no expression accounts for.");

DEFINE_int32(
    projection_stride,
    1,
    "How far the reading window advances between consecutive projections, "
    "which sets how much input the projection list reuses. 0 strides by the "
    "fan-in, so consecutive projections read disjoint column groups and share "
    "nothing. 1 slides one column at a time, so adjacent projections share all "
    "but one input.");

DEFINE_double(
    selectivity,
    1.0,
    "Fraction of rows passing the filter, in [0, 1]. Column values are "
    "uniform on [0, 1), so the filter constant is this value.");

DEFINE_int64(key_cardinality, 5000, "Distinct group keys, k.");

DEFINE_int32(
    warmup_iters,
    1,
    "Untimed iterations. The first one pays for JIT compilation.");

DEFINE_int32(
    timed_iters,
    4,
    "Timed iterations. Their mean is reported, excluding the first, which runs "
    "high even behind a warmup. All of them are printed so the spread and the "
    "excluded one stay visible.");

DEFINE_int32(
    gpu_batch_size_rows,
    100'000,
    "velox.cudf.gpu_batch_size_rows. cuDF coalesces host batches to this many "
    "rows before converting.");

DEFINE_string(
    evaluator,
    "default",
    "cuDF expression evaluator: 'jit' for a generated kernel, 'ast' for the "
    "runtime AST interpreter, or 'default' to leave both registered, which "
    "selects JIT on priority.");

DEFINE_double(
    expect_checksum,
    std::numeric_limits<double>::quiet_NaN(),
    "If set, fail unless the result checksum matches within "
    "--checksum_tolerance. Take the value from a cpu run.");

DEFINE_double(checksum_tolerance, 1e-9, "Relative checksum tolerance.");

DEFINE_bool(print_plan, false, "Print the plan before running.");

namespace {

// Velox's parser types "1" as BIGINT and there is no
// lt(DOUBLE, BIGINT) to fall back on. Force the point.
std::string doubleLiteral(double value) {
  auto text = fmt::format("{:.17g}", value);
  if (text.find_first_of(".eE") == std::string::npos) {
    text += ".0";
  }
  return text;
}

// Lands in [1.5, 15.5], never 0 and never 1. Those are the two values a
// compiler could use to shorten the chain by an identity rather than evaluate
// it: zero drops an add through x + 0, one drops a multiply through 1 * x.
// Nothing here folds, since every node of the chain reads a column.
//
// Distinct per (projection, term, power), so no two projections share a
// subexpression that could be evaluated once and reused.
std::string coefficient(int32_t projection, int32_t term, int32_t power) {
  const auto seed = 1 + ((projection * 7 + term * 13 + power * 3) % 29);
  return doubleLiteral(1.0 + 0.5 * seed);
}

// Which column term `term` of projection `projection` reads. Wrapping keeps
// every index in range and lets a projection list shorter than the table still
// come back around to cover it.
int32_t projectionColumn(int32_t projection, int32_t term, int32_t fanin) {
  const auto stride =
      FLAGS_projection_stride > 0 ? FLAGS_projection_stride : fanin;
  return (projection * stride + term) % FLAGS_num_columns;
}

// Resolved on every read rather than written back to the flag. A sweep sets
// the flags it names once per combination and leaves the rest alone, so a
// derived value stored into --num_projections would outlive the cell that
// produced it and silently apply to the next one.
int32_t numProjections() {
  return FLAGS_num_projections == 0 ? FLAGS_num_columns : FLAGS_num_projections;
}

bool useCudf() {
  if (FLAGS_engine == "cudf") {
    return true;
  }
  if (FLAGS_engine == "cpu") {
    return false;
  }
  VELOX_FAIL("Unknown --engine '{}', expected cpu or cudf", FLAGS_engine);
}

class CudfFilterProjectBenchmark : public QueryBenchmarkBase {
 public:
  void initialize() override {
    QueryBenchmarkBase::initialize();

    rootPool_ =
        memory::memoryManager()->addRootPool("CudfFilterProjectBenchmark");
    pool_ = rootPool_->addLeafChild("leaf");

    if (!useCudf()) {
      return;
    }
    // CudfConfig is read once when the adapter registers rather than per query,
    // so these have to be set before registerCudf() and cannot be swept.
    auto& config = cudf_velox::CudfConfig::getInstance();
    if (FLAGS_evaluator == "ast") {
      config.jitExpressionEnabled = false;
    } else if (FLAGS_evaluator == "jit") {
      config.astExpressionEnabled = false;
    } else if (FLAGS_evaluator != "default") {
      VELOX_FAIL("Unknown --evaluator '{}'", FLAGS_evaluator);
    }
    cudf_velox::registerCudf();
  }

  void runMain(std::ostream& out, RunStats& runStats) override {
    resolveShape();
    ensureBaseColumns();
    rechunk();
    auto plan = makePlan();
    if (FLAGS_print_plan) {
      out << plan->toString(true, true) << std::endl;
    }

    for (auto i = 0; i < FLAGS_warmup_iters; ++i) {
      execute(plan);
    }

    double checksum = 0;
    int64_t groups = 0;
    std::vector<double> wallMs;
    struct rusage before;
    getrusage(RUSAGE_SELF, &before);
    for (auto i = 0; i < FLAGS_timed_iters; ++i) {
      const auto start = std::chrono::steady_clock::now();
      auto output = execute(plan);
      const auto end = std::chrono::steady_clock::now();
      wallMs.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
      checksum = summarize(output, groups);
    }
    struct rusage after;
    getrusage(RUSAGE_SELF, &after);

    // The first timed iteration is excluded. It runs high even behind a warmup
    // and even once JIT compilation is paid for: at 32 columns the JIT arm
    // takes around 145 ms on it against a steady 99 ms after. A single timed
    // iteration has nothing to exclude and is reported as it is.
    const size_t skip = wallMs.size() > 1 ? 1 : 0;
    const auto mean =
        std::accumulate(wallMs.begin() + skip, wallMs.end(), 0.0) /
        static_cast<double>(wallMs.size() - skip);
    std::string all;
    for (const auto ms : wallMs) {
      if (!all.empty()) {
        all += ' ';
      }
      all += fmt::format("{:.1f}", ms);
    }
    verifyChecksum(checksum);

    // Reported through RunStats so a swept run is sorted and printed by the
    // base. The base fills these in only when they are left at zero, and its
    // own measurement would span all of runMain, charging the generation of a
    // multi-gigabyte input to a query that takes milliseconds. The CPU totals
    // are divided down to one iteration to match, or the share it prints would
    // read several hundred percent.
    runStats.micros = static_cast<int64_t>(mean * 1000);
    runStats.rawInputBytes = inputBytes();
    runStats.userNanos = cpuNanos(before.ru_utime, after.ru_utime) /
        std::max(FLAGS_timed_iters, 1);
    runStats.systemNanos = cpuNanos(before.ru_stime, after.ru_stime) /
        std::max(FLAGS_timed_iters, 1);

    out << fmt::format(
        "engine={} evaluator={} rows={} batch={} columns={} projections={} "
        "degree={} selectivity={} groups={} checksum={:.6f} mean={:.1f}ms "
        "all=[{}] rows/s={:.1f}M\n",
        FLAGS_engine,
        FLAGS_evaluator,
        FLAGS_num_rows,
        FLAGS_batch_size,
        FLAGS_num_columns,
        numProjections(),
        FLAGS_poly_degree,
        FLAGS_selectivity,
        groups,
        checksum,
        mean,
        all,
        FLAGS_num_rows / mean / 1000.0);
  }

 private:
  // Derives what depends on the column count, which a sweep sets per
  // combination, so this has to run per combination rather than once.
  static void resolveShape() {
    // At a fixed row count, adding columns adds data; a byte budget holds the
    // volume still so that the schema is the only thing that changed.
    if (FLAGS_total_bytes > 0) {
      const int64_t rowBytes = sizeof(double) * (FLAGS_num_columns + 1);
      FLAGS_num_rows = std::max<int64_t>(FLAGS_total_bytes / rowBytes, 1);
    }
  }

  static int64_t cpuNanos(timeval before, timeval after) {
    const auto nanos = [](timeval time) {
      return time.tv_sec * 1'000'000'000L + time.tv_usec * 1'000L;
    };
    return nanos(after) - nanos(before);
  }

  static int64_t inputBytes() {
    return FLAGS_num_rows * static_cast<int64_t>(sizeof(double)) *
        (FLAGS_num_columns + 1);
  }

  // One BIGINT key plus N DOUBLE columns, uniform on [0, 1).
  //
  // Uniform values make the filter constant equal the selectivity directly and
  // keep a degree-D Horner polynomial bounded, so nothing overflows and the
  // checksum stays meaningful at any depth.
  //
  // The contents depend only on the column count, the row count and the
  // cardinality. Batch size decides how rows are chunked rather than what they
  // hold, so a sweep of batch size alone regenerates nothing.
  void ensureBaseColumns() {
    if (baseColumns_.size() == static_cast<size_t>(FLAGS_num_columns) + 1 &&
        baseRows_ == FLAGS_num_rows &&
        baseCardinality_ == FLAGS_key_cardinality) {
      return;
    }

    // Release the old columns before allocating the new ones, so peak host use
    // stays at one width's worth rather than two.
    vectors_.clear();
    baseColumns_.clear();

    std::vector<std::string> names{"k"};
    std::vector<TypePtr> types{BIGINT()};
    for (auto i = 0; i < FLAGS_num_columns; ++i) {
      names.push_back(fmt::format("c{}", i));
      types.push_back(DOUBLE());
    }
    rowType_ = ROW(std::move(names), std::move(types));

    const auto rows = static_cast<vector_size_t>(FLAGS_num_rows);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    auto keys =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), rows, pool_.get());
    auto* rawKeys = keys->mutableRawValues();
    for (vector_size_t i = 0; i < rows; ++i) {
      rawKeys[i] = i % FLAGS_key_cardinality;
    }
    baseColumns_.push_back(std::move(keys));

    for (auto c = 0; c < FLAGS_num_columns; ++c) {
      auto values =
          BaseVector::create<FlatVector<double>>(DOUBLE(), rows, pool_.get());
      auto* raw = values->mutableRawValues();
      for (vector_size_t i = 0; i < rows; ++i) {
        raw[i] = uniform(rng);
      }
      baseColumns_.push_back(std::move(values));
    }

    baseRows_ = FLAGS_num_rows;
    baseCardinality_ = FLAGS_key_cardinality;
  }

  // Cuts the base columns into batches. Slicing a flat vector shares the values
  // buffer through a view, so this copies nothing regardless of batch size.
  void rechunk() {
    vectors_.clear();
    for (int64_t offset = 0; offset < FLAGS_num_rows;
         offset += FLAGS_batch_size) {
      const auto size = static_cast<vector_size_t>(
          std::min<int64_t>(FLAGS_batch_size, FLAGS_num_rows - offset));
      std::vector<VectorPtr> children;
      children.reserve(baseColumns_.size());
      for (auto& column : baseColumns_) {
        children.push_back(
            column->slice(static_cast<vector_size_t>(offset), size));
      }
      vectors_.push_back(
          std::make_shared<RowVector>(
              pool_.get(), rowType_, nullptr, size, std::move(children)));
    }
  }

  core::PlanNodePtr makePlan() {
    // Horner form: (((a_D * x + a_D-1) * x + a_D-2) ... ) * x + a_0.
    auto polynomial = [](const std::string& x,
                         int32_t degree,
                         int32_t projection,
                         int32_t term) {
      std::string expr = coefficient(projection, term, degree);
      for (auto i = degree - 1; i >= 0; --i) {
        expr = fmt::format(
            "({} * {} + {})", expr, x, coefficient(projection, term, i));
      }
      return expr;
    };

    const auto count = numProjections();
    const auto fanin = std::min(
        FLAGS_projection_fanin > 0 ? FLAGS_projection_fanin
                                   : (FLAGS_num_columns + count - 1) / count,
        FLAGS_num_columns);
    VELOX_CHECK_GT(fanin, 0, "--projection_fanin must be positive");

    std::vector<std::string> projections{"k"};
    std::vector<std::string> aggregates;
    std::vector<bool> covered(FLAGS_num_columns, false);
    for (auto m = 0; m < count; ++m) {
      std::string expr;
      for (auto j = 0; j < fanin; ++j) {
        const auto index = projectionColumn(m, j, fanin);
        covered[index] = true;
        const auto term =
            polynomial(fmt::format("c{}", index), FLAGS_poly_degree, m, j);
        expr = expr.empty() ? term : fmt::format("({} + {})", expr, term);
      }
      projections.push_back(fmt::format("{} as p{}", expr, m));
      aggregates.push_back(fmt::format("sum(p{})", m));
    }

    // A column that no expression reads is still generated and handed to the
    // engine, so it adds data movement that the measurement cannot attribute to
    // expression evaluation. Cheap to check, and easy to introduce by changing
    // the fan-in or the stride.
    const auto unread = std::count(covered.begin(), covered.end(), false);
    VELOX_CHECK_EQ(
        unread,
        0,
        "{} of {} columns are never read at fanin {} with {} projections",
        unread,
        FLAGS_num_columns,
        fanin,
        count);

    const auto filter =
        fmt::format("c0 < {}", doubleLiteral(FLAGS_selectivity));

    return PlanBuilder(pool_.get())
        .values(vectors_)
        .filter(filter)
        .project(projections)
        .singleAggregation({"k"}, aggregates)
        .planNode();
  }

  RowVectorPtr execute(const core::PlanNodePtr& plan) {
    AssertQueryBuilder builder(plan);
    builder.maxDrivers(FLAGS_num_drivers);
    if (useCudf()) {
      builder.config(
          cudf_velox::CudfFromVelox::kGpuBatchSizeRows,
          std::to_string(FLAGS_gpu_batch_size_rows));
    }
    std::shared_ptr<exec::Task> task;
    return builder.copyResults(pool_.get(), task);
  }

  // Reduces the result to one number, so the arms can be compared without the
  // exact vector match that FMA contraction rules out. The long double
  // accumulator keeps the sum from drifting with the order the groups arrive
  // in.
  static double summarize(const RowVectorPtr& output, int64_t& groups) {
    long double total = 0;
    for (auto c = 1; c < output->type()->size(); ++c) {
      auto* column = output->childAt(c)->asFlatVector<double>();
      VELOX_CHECK_NOT_NULL(column, "Aggregate output is not a flat double");
      for (vector_size_t i = 0; i < output->size(); ++i) {
        total += column->valueAt(i);
      }
    }
    groups = output->size();
    return static_cast<double>(total);
  }

  static void verifyChecksum(double checksum) {
    if (std::isnan(FLAGS_expect_checksum)) {
      return;
    }
    const auto scale = std::max(std::abs(FLAGS_expect_checksum), 1.0);
    const auto error = std::abs(checksum - FLAGS_expect_checksum) / scale;
    VELOX_CHECK_LE(
        error,
        FLAGS_checksum_tolerance,
        "Checksum {:.6f} differs from the expected {:.6f} by a relative {}",
        checksum,
        FLAGS_expect_checksum,
        error);
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
  RowTypePtr rowType_;
  std::vector<VectorPtr> baseColumns_;
  std::vector<RowVectorPtr> vectors_;
  int64_t baseRows_{0};
  int64_t baseCardinality_{0};
};

} // namespace

int main(int argc, char** argv) {
  std::string kUsage(
      "velox-cudf's expression evaluators against CPU Velox on a filter, a "
      "projection list and a grouped aggregation. Run with --help for "
      "available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};

  CudfFilterProjectBenchmark benchmark;
  benchmark.initialize();
  if (FLAGS_test_flags_file.empty()) {
    RunStats stats;
    benchmark.runMain(std::cout, stats);
  } else {
    benchmark.runAllCombinations();
  }
  benchmark.shutdown();
  return 0;
}
