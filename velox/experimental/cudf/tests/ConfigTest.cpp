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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, batchConcatThresholdDefaults) {
  CudfConfig config;
  EXPECT_EQ(config.batchSizeMinThreshold, 100'000);
  EXPECT_FALSE(config.batchSizeMinBytes);
}

TEST(ConfigTest, cudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfStreamingGroupbyEnabled, "true"},
      {CudfConfig::kCudfStreamingGroupbyCapacityMultiplier, "3.5"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kCudfBatchSizeMinThreshold, "123456"},
      {CudfConfig::kCudfBatchSizeMinBytes, "2147483648"},
      {CudfConfig::kCudfHashJoinLoadFactor, "0.8"}};

  CudfConfig config;
  ASSERT_FALSE(config.streamingGroupbyEnabled);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 2.0);
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.streamingGroupbyEnabled, true);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 3.5);
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_EQ(config.batchSizeMinThreshold, 123'456);
  ASSERT_EQ(config.batchSizeMinBytes.value(), 2'147'483'648);
  ASSERT_DOUBLE_EQ(config.hashJoinLoadFactor, 0.8);
}

// The leaf bound of a spillable group-by is a setting of its own. The global
// batch maximum, set or not, does not change it.
TEST(ConfigTest, spillGroupbyMaxRowsPerLeaf) {
  {
    CudfConfig config;
    EXPECT_EQ(config.spillGroupbyMaxRowsPerLeaf, 1 << 24);
    EXPECT_FALSE(config.batchSizeMaxThreshold.has_value());
  }
  {
    CudfConfig config;
    config.initialize({{CudfConfig::kCudfBatchSizeMaxThreshold, "1000"}});
    EXPECT_EQ(config.batchSizeMaxThreshold.value(), 1'000);
    EXPECT_EQ(config.spillGroupbyMaxRowsPerLeaf, 1 << 24);
  }
  {
    CudfConfig config;
    config.initialize({{CudfConfig::kCudfSpillGroupbyMaxRowsPerLeaf, "4096"}});
    EXPECT_EQ(config.spillGroupbyMaxRowsPerLeaf, 4'096);
    EXPECT_FALSE(config.batchSizeMaxThreshold.has_value());
  }
}

TEST(ConfigTest, rejectsNonPositiveBatchConcatTargets) {
  auto initialize = [](const char* key, const char* value) {
    CudfConfig config;
    config.initialize({{key, value}});
  };

  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinThreshold, "0"),
      "cuDF BatchConcat minimum row target must be positive");
  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinThreshold, "-5"),
      "cuDF BatchConcat minimum row target must be positive");
  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinBytes, "0"),
      "cuDF BatchConcat minimum byte target must be a positive integer: 0");
  VELOX_ASSERT_USER_THROW(
      initialize(CudfConfig::kCudfBatchSizeMinBytes, "-1"),
      "cuDF BatchConcat minimum byte target must be a positive integer: -1");
}

TEST(ConfigTest, rejectsInvalidSpillGroupbyMaxRowsPerLeaf) {
  auto initialize = [](const char* value) {
    CudfConfig config;
    config.initialize({{CudfConfig::kCudfSpillGroupbyMaxRowsPerLeaf, value}});
  };

  VELOX_ASSERT_USER_THROW(
      initialize("0"), "cudf.spill_groupby_max_rows_per_leaf must be positive");
  VELOX_ASSERT_USER_THROW(
      initialize("-16"),
      "cudf.spill_groupby_max_rows_per_leaf must be positive");
  VELOX_ASSERT_USER_THROW(
      initialize("2147483648"),
      "cudf.spill_groupby_max_rows_per_leaf must be a 32-bit integer: "
      "2147483648");
  VELOX_ASSERT_USER_THROW(
      initialize("many"),
      "cudf.spill_groupby_max_rows_per_leaf must be a 32-bit integer: many");
}

} // namespace facebook::velox::cudf_velox::test
