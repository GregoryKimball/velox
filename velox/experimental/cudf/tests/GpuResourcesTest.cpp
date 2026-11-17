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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox {
namespace {

constexpr int64_t kAllocSize = 8 << 20; // 8MB.

class GpuResourcesTest : public OperatorTestBase {
 protected:
  // The memory resource configuration has to be in place before registerCudf()
  // creates the resources, so each test registers cuDF itself.
  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = CudfConfig::getInstance();
    savedMemoryResource_ = config.memoryResource;
    savedOutputMemoryResource_ = config.outputMemoryResource;
  }

  void TearDown() override {
    unregisterCudf();
    auto& config = CudfConfig::getInstance();
    config.memoryResource = savedMemoryResource_;
    config.outputMemoryResource = savedOutputMemoryResource_;
    OperatorTestBase::TearDown();
  }

  // Asserts that an allocation from each of 'resources' raises the live byte
  // count, and that freeing them returns it to the value it started at.
  void assertTracksAllocations(
      const std::vector<rmm::device_async_resource_ref>& resources) {
    const auto baseline = cudfAllocatedBytes();
    ASSERT_GE(baseline, 0);

    auto stream = cudfGlobalStreamPool().get_stream();
    {
      std::vector<rmm::device_buffer> buffers;
      buffers.reserve(resources.size());
      for (const auto& resource : resources) {
        buffers.emplace_back(kAllocSize, stream, resource);
      }
      // A resource may round the request up, hence the inequality.
      EXPECT_GE(
          cudfAllocatedBytes(),
          baseline + static_cast<int64_t>(resources.size()) * kAllocSize);
    }
    EXPECT_EQ(cudfAllocatedBytes(), baseline);
  }

  std::string savedMemoryResource_;
  std::string savedOutputMemoryResource_;
};

TEST_F(GpuResourcesTest, allocatedBytesWithoutCudf) {
  {
    SCOPED_TRACE("Before registration");
    ASSERT_FALSE(cudfIsRegistered());
    EXPECT_EQ(cudfAllocatedBytes(), -1);
  }

  registerCudf();
  ASSERT_GE(cudfAllocatedBytes(), 0);
  unregisterCudf();

  {
    SCOPED_TRACE("After unregistration");
    EXPECT_EQ(cudfAllocatedBytes(), -1);
  }
}

TEST_F(GpuResourcesTest, allocatedBytesTracksLiveAllocations) {
  registerCudf();
  ASSERT_TRUE(cudfIsRegistered());

  assertTracksAllocations({cudf::get_current_device_resource_ref()});
}

TEST_F(GpuResourcesTest, peakAllocatedBytesCoversEachWindow) {
  EXPECT_EQ(takeCudfPeakAllocatedBytes(), -1);
  registerCudf();
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto resource = cudf::get_current_device_resource_ref();
  // Allocated before the window, so it counts in the window's peak as it does
  // in the live count.
  rmm::device_buffer before(kAllocSize, stream, resource);
  takeCudfPeakAllocatedBytes();
  int64_t largest{0};
  {
    rmm::device_buffer large(2 * kAllocSize, stream, resource);
    largest = cudfAllocatedBytes();
  }
  rmm::device_buffer small(kAllocSize, stream, resource);
  const auto withSmall = cudfAllocatedBytes();
  EXPECT_GT(largest, withSmall);
  EXPECT_EQ(takeCudfPeakAllocatedBytes(), largest);

  // The next window starts from the bytes still allocated and only frees.
  EXPECT_EQ(cudfAllocatedBytes(), withSmall);
  before = rmm::device_buffer(0, stream, resource);
  EXPECT_LT(cudfAllocatedBytes(), withSmall);
  EXPECT_EQ(takeCudfPeakAllocatedBytes(), withSmall);

  unregisterCudf();
  EXPECT_EQ(takeCudfPeakAllocatedBytes(), -1);
}

// A task starts a split group next to running ones once it has completed one
// and while one more running split group's share of the allocated memory fits.
TEST_F(GpuResourcesTest, admitsASplitGroupWhileItsShareFits) {
  constexpr int64_t kAvailableBytes{100};
  // No split group has completed: what one allocates is not known yet.
  EXPECT_FALSE(admitSplitGroup(0, kAvailableBytes, 1, 0, 0.8));
  // Two split groups hold 40 bytes: a third one's 20 reaches 60 of 80.
  EXPECT_TRUE(admitSplitGroup(40, kAvailableBytes, 2, 1, 0.8));
  // Two hold 60 bytes: a third one's 30 would reach 90 of 80.
  EXPECT_FALSE(admitSplitGroup(60, kAvailableBytes, 2, 1, 0.8));
  // Exactly the fraction fits.
  EXPECT_TRUE(admitSplitGroup(50, kAvailableBytes, 1, 1, 1.0));
  // Unknown memory admits nothing.
  EXPECT_FALSE(admitSplitGroup(-1, kAvailableBytes, 1, 1, 0.8));
  EXPECT_FALSE(admitSplitGroup(10, 0, 1, 1, 0.8));
}

// The pool of the "async" memory resource keeps a freed buffer's memory for
// reuse. The worker returns it to the device only when the device is short of
// free memory.
TEST_F(GpuResourcesTest, trimsTheAsyncPoolOnlyWhenTheDeviceIsShort) {
  CudfConfig::getInstance().memoryResource = "async";
  registerCudf();
  constexpr std::size_t kBytes = std::size_t{512} << 20;
  auto freeBytes = [] {
    std::size_t free{0};
    std::size_t total{0};
    EXPECT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
    return free;
  };
  auto stream = cudfGlobalStreamPool().get_stream();
  {
    rmm::device_buffer buffer(
        kBytes, stream, cudf::get_current_device_resource_ref());
  }
  stream.sync();

  const auto freeBefore = freeBytes();
  EXPECT_FALSE(trimDeviceMemoryPoolWhenShort(0));
  EXPECT_LT(freeBytes(), freeBefore + kBytes / 2);

  EXPECT_TRUE(trimDeviceMemoryPoolWhenShort(freeBefore + kBytes));
  EXPECT_GE(freeBytes(), freeBefore + kBytes / 2);
}

TEST_F(GpuResourcesTest, allocatedBytesIncludesSeparateOutputResource) {
  auto& config = CudfConfig::getInstance();
  config.memoryResource = "async";
  config.outputMemoryResource = "cuda";
  registerCudf();

  assertTracksAllocations(
      {cudf::get_current_device_resource_ref(), get_output_mr()});
}

} // namespace
} // namespace facebook::velox::cudf_velox
