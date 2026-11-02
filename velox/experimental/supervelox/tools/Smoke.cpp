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

#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <folly/init/Init.h>

#include <iostream>

int64_t smokeDeviceSum(int64_t n);

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();

  auto pool = memory::memoryManager()->addLeafPool();
  test::VectorMaker maker(pool.get());
  auto input = maker.rowVector({"x"}, {maker.flatVector<int64_t>({1, 2, 3, 4})});
  auto plan = exec::test::PlanBuilder().values({input}).project({"x * 10 AS y"}).planNode();
  auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool.get());

  std::cout << "velox: " << result->toString(0, 4) << std::endl;
  std::cout << "gpu sum(0..999): " << smokeDeviceSum(1000) << std::endl;
  return 0;
}
