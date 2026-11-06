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

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/PlanNodeStats.h"

#include <fmt/format.h>
#include <folly/String.h>

#include <chrono>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

DECLARE_string(data_path);
DECLARE_int32(num_repeats);
DECLARE_int32(run_query_verbose);

/// A TpchBenchmark whose --run_query_verbose run prints the wall time of each
/// of the --num_repeats runs ('Repeat N wall time us: T'), and with
/// --include_results every column of every result row as '{a, b, ...}'.
class VerboseTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override {
    TpchBenchmark::initialize();
    queryBuilder_ = std::make_shared<facebook::velox::exec::test::TpchQueryBuilder>(
        facebook::velox::dwio::common::toFileFormat(FLAGS_data_format));
    queryBuilder_->initialize(FLAGS_data_path);
  }

  void shutdown() override {
    queryBuilder_.reset();
    TpchBenchmark::shutdown();
  }

  void runMain(std::ostream& out, facebook::velox::RunStats& runStats)
      override {
    if (FLAGS_run_query_verbose == -1) {
      TpchBenchmark::runMain(out, runStats);
      return;
    }
    const auto plan = queryBuilder_->getQueryPlan(FLAGS_run_query_verbose);
    const auto repeats = FLAGS_num_repeats;
    FLAGS_num_repeats = 1;
    std::pair<
        std::unique_ptr<facebook::velox::exec::TaskCursor>,
        std::vector<facebook::velox::RowVectorPtr>>
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
      out << fmt::format("Repeat {} wall time us: {}", repeat, us)
          << std::endl;
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
               facebook::velox::succinctMillis(
                   stats.executionEndTimeMs - stats.executionStartTimeMs))
        << std::endl;
    out << fmt::format(
               "Splits total: {}, finished: {}",
               stats.numTotalSplits,
               stats.numFinishedSplits)
        << std::endl;
    out << facebook::velox::exec::printPlanWithStats(
               *plan.plan, stats, FLAGS_include_custom_stats)
        << std::endl;
  }

 private:
  // Unlike QueryBenchmarkBase::printResults(), prints every column; that
  // truncates rows to 5 columns.
  static void printRows(
      const std::vector<facebook::velox::RowVectorPtr>& results,
      std::ostream& out) {
    out << "Results:" << std::endl;
    if (!results.empty()) {
      out << results.front()->type()->asRow().toString() << std::endl;
    }
    for (const auto& vector : results) {
      for (facebook::velox::vector_size_t i = 0; i < vector->size(); ++i) {
        std::vector<std::string> fields;
        for (const auto& child : vector->children()) {
          fields.push_back(child->toString(i));
        }
        out << "{" << folly::join(", ", fields) << "}" << std::endl;
      }
    }
    out << std::endl;
  }

  std::shared_ptr<facebook::velox::exec::test::TpchQueryBuilder> queryBuilder_;
};
