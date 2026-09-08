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
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <cudf/utilities/memory_resource.hpp>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class AdapterOperatorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }
};

TEST_F(AdapterOperatorTest, adapterStatsMergedIntoPlanNode) {
  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);

  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_TRUE(projStats.operatorStats.count("CudfToVelox"));
}

// Guards an invariant of the host-to-device boundary rather than a reachable
// defect: CudfFromVelox must never be handed data that is already on the device.
//
// The zero-column shortcut exists for a genuinely empty projection -- no columns
// to convert, only a row count to carry across. A CudfVector is indistinguishable
// from that by child count alone, since it is always childless while reporting a
// real size, so a child-count gate would apply the shortcut to a vector whose
// schema still had columns and drop every value. The schema gate cannot, and this
// check refuses the input outright.
//
// No plan built by Presto reaches here: CudfFromVelox is only inserted after an
// operator that declares it does not produce GPU output, and every adapter that
// accepts GPU input while declaring CPU output is a pipeline sink, so nothing
// forwards a CudfVector into one. Values is the sole exception -- it emits the
// vectors it was constructed with -- and Presto builds those from SQL literals
// with BaseVector::create, one child per column. That makes Values the only way
// to construct the situation at all, which is what this test does.
TEST_F(AdapterOperatorTest, fromVeloxRejectsDeviceResidentInput) {
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  auto hostData =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3, 4, 5})});
  auto table = cudf_velox::with_arrow::toCudfTable(
      hostData, pool(), stream, cudf::get_current_device_resource_ref());
  ASSERT_NE(table, nullptr);
  stream.synchronize();

  // Childless, as every CudfVector is, but typed with one BIGINT column.
  const RowVectorPtr deviceData = std::make_shared<cudf_velox::CudfVector>(
      pool(),
      ROW({"c0"}, {BIGINT()}),
      hostData->size(),
      std::move(table),
      stream);

  auto plan = PlanBuilder()
                  .values({deviceData})
                  .singleAggregation({}, {"sum(c0)"})
                  .planNode();

  // The conversion must refuse the input rather than convert it to nothing.
  // Before the fix it took the shortcut and the aggregation summed an empty
  // table, so no exception was raised at all.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "CudfFromVelox got 0 children for a 1-column schema");
}
