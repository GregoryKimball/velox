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

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <unordered_set>

#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include "velox/common/memory/SharedArbitrator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/MemoryArbitrationFuzzer.h"
#include "velox/exec/fuzzer/PrestoQueryRunner.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"

DEFINE_int64(allocator_capacity, 32L << 30, "Allocator capacity in bytes.");

DECLARE_int64(arbitrator_capacity);

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

using namespace facebook::velox::exec;

int main(int argc, char** argv) {
  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for a better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);
  test::setupMemory(FLAGS_allocator_capacity, FLAGS_arbitrator_capacity);
  const size_t seed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

  facebook::velox::serializer::presto::PrestoVectorSerde::registerVectorSerde();
  facebook::velox::filesystems::registerLocalFileSystem();
  facebook::velox::tests::utils::registerFaultyFileSystem();
  facebook::velox::functions::prestosql::registerAllScalarFunctions();
  facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
  memoryArbitrationFuzzer(seed);
}
