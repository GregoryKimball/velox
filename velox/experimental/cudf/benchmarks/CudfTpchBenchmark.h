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

#include "velox/experimental/cudf/benchmarks/CudfTpchQueryBuilder.h"

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// Runs the TPC-H plans of cudf_velox::CudfTpchQueryBuilder. Every entry point
/// runs those plans: --run_query_verbose and --run_queries run the given
/// queries, and without either all 22 queries run, each --num_repeats times.
/// TpchBenchmark's own query builder, runQuery() and folly benchmarks, which
/// run the plans of exec::test::TpchQueryBuilder, are not used.
///
/// For each query it prints the wall time of each of the --num_repeats runs
/// ('Repeat N wall time us: T'). With --run_query_verbose or --run_queries it
/// also prints the plan with its runtime statistics, with --include_results
/// every column of every result row as '{a, b, ...}', and with
/// --results_json_dir it writes the result rows to <dir>/q<N>.json.
class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override;

  void runMain(std::ostream& out, facebook::velox::RunStats& runStats) override;

  std::shared_ptr<facebook::velox::config::ConfigBase> makeConnectorProperties()
      override;

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const facebook::velox::exec::test::TpchPlan& plan) override;

  void shutdown() override;

 private:
  // Runs 'queryId' --num_repeats times and prints the time of each run. With
  // 'verbose', also prints the results and statistics of the last run.
  void executeQuery(int32_t queryId, bool verbose, std::ostream& out);

  std::unique_ptr<facebook::velox::cudf_velox::CudfTpchQueryBuilder>
      queryBuilder_;
};

namespace facebook::velox::cudf_velox {

/// Parse a properties file into a key-value map.
/// Each line should be key=value. Lines starting with '#' and blank lines are
/// skipped. Lines without '=' are logged as warnings and ignored.
inline std::unordered_map<std::string, std::string> loadPropertiesFile(
    const std::string& path) {
  auto fsPath = std::filesystem::path(path);
  VELOX_USER_CHECK(
      std::filesystem::exists(fsPath), "Properties file not found: {}", path);
  std::unordered_map<std::string, std::string> properties;
  std::string line;
  std::ifstream configFile(fsPath);
  while (std::getline(configFile, line)) {
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line.empty() || line[0] == '#') {
      continue;
    }
    LOG(INFO) << "Setting property " << line;
    const auto delimiterPos = line.find('=');
    if (delimiterPos == std::string::npos) {
      LOG(WARNING) << "Skipping malformed config line (no '='): " << line;
      continue;
    }
    properties.emplace(
        line.substr(0, delimiterPos), line.substr(delimiterPos + 1));
  }
  return properties;
}

} // namespace facebook::velox::cudf_velox
