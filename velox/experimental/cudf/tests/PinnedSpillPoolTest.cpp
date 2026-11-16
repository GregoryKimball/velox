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
#include "velox/experimental/cudf/exec/PinnedSpillPool.h"

#include "velox/common/memory/Memory.h"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstring>

namespace facebook::velox::cudf_velox {
namespace {
class PinnedSpillPoolTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize({});
  }
};

TEST_F(PinnedSpillPoolTest, tracksLiveBudgetAndReportsCachedMemory) {
  constexpr std::size_t kLimit = 32u << 20;
  auto storage = PinnedSpillPool::create(kLimit, 0);
  auto query = storage->budget()->createQuery("query", kLimit);
  for (std::size_t size :
       {4u << 20, 8u << 20, 12u << 20, 16u << 20, 6u << 20}) {
    {
      auto admission = storage->reserve(query, size);
      auto allocation = admission.allocate(size);
      ASSERT_NE(allocation.data(), nullptr);
      std::memset(allocation.data(), 42, size);
      EXPECT_EQ(query->used(), size);
      EXPECT_EQ(storage->budget()->used(), size);
      EXPECT_GE(storage->reservedBytes(), size);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(query->used(), 0);
    EXPECT_EQ(storage->budget()->used(), 0);
    EXPECT_LE(storage->budget()->used(), kLimit);
  }
}

TEST_F(PinnedSpillPoolTest, admissionFailureDoesNotPoisonAllocator) {
  constexpr std::size_t kLimit = 16u << 20;
  auto storage = PinnedSpillPool::create(kLimit, 0);
  auto query = storage->budget()->createQuery("query", 8u << 20);
  {
    auto admitted = storage->reserve(query, 8u << 20);
    auto allocation = admitted.allocate(8u << 20);
    EXPECT_THROW(storage->reserve(query, 1), VeloxException);
    EXPECT_EQ(query->used(), 8u << 20);
  }
  auto next = storage->reserve(query, 4u << 20);
  auto reused = next.allocate(4u << 20);
  EXPECT_NE(reused.data(), nullptr);
  auto wrong = HostSpillBudget::createWorker(kLimit);
  EXPECT_THROW(storage->reserve(wrong, 256), VeloxException);
}

TEST_F(PinnedSpillPoolTest, allocationRetainsPoolAndBudgetOwners) {
  auto storage = PinnedSpillPool::create(16u << 20, 0);
  auto query = storage->budget()->createQuery("query", 16u << 20);
  std::weak_ptr<PinnedSpillPool> weakStorage = storage;
  std::weak_ptr<HostSpillBudget> weakQuery = query;
  auto admitted = storage->reserve(query, 1u << 20);
  auto allocation = admitted.allocate(1u << 20);
  admitted = {};
  query.reset();
  storage.reset();
  EXPECT_FALSE(weakStorage.expired());
  EXPECT_FALSE(weakQuery.expired());
  allocation = {};
  EXPECT_TRUE(weakStorage.expired());
  EXPECT_TRUE(weakQuery.expired());
}

TEST_F(PinnedSpillPoolTest, returnsUnusedAdmissionWithoutCpuCharges) {
  auto storage = PinnedSpillPool::create(16u << 20, 0);
  auto query = storage->budget()->createQuery("query", 16u << 20);
  PinnedSpillPool::Allocation retained;
  {
    auto admitted = storage->reserve(query, 4u << 20);
    retained = admitted.allocate(1u << 20);
    EXPECT_EQ(query->used(), 4u << 20);
    EXPECT_EQ(storage->budget()->used(), 4u << 20);
  }
  EXPECT_EQ(query->used(), 1u << 20);
  EXPECT_EQ(storage->budget()->used(), 1u << 20);
  retained = {};
  EXPECT_EQ(query->used(), 0);
  EXPECT_EQ(storage->budget()->used(), 0);
}

TEST_F(PinnedSpillPoolTest, exhaustedCpuBudgetDoesNotBlockPinnedSpill) {
  constexpr uint64_t kCpuLimit = 8u << 20;
  auto cpuRoot = memory::memoryManager()->addRootPool("cpu-budget", kCpuLimit);
  auto cpuLeaf = cpuRoot->addLeafChild("cpu-operator");
  cpuLeaf->reportExternalAllocation(kCpuLimit);
  // The CPU side really is full; the independent spill allowance must neither
  // borrow capacity from it nor make another CPU allocation admissible.
  EXPECT_THROW(cpuLeaf->reportExternalAllocation(1), VeloxRuntimeError);
  const auto capacity = cpuRoot->capacity();
  const auto growths = cpuRoot->stats().numCapacityGrowths;
  {
    auto storage = PinnedSpillPool::create(16u << 20, 0);
    auto query = storage->budget()->createQuery("spill-query", 16u << 20);
    auto admission = storage->reserve(query, 4u << 20);
    auto allocation = admission.allocate(4u << 20);
    ASSERT_NE(allocation.data(), nullptr);
    EXPECT_EQ(query->used(), 4u << 20);
    EXPECT_EQ(cpuLeaf->usedBytes(), kCpuLimit);
    EXPECT_EQ(cpuRoot->capacity(), capacity);
    EXPECT_EQ(cpuRoot->stats().numCapacityGrowths, growths);
  }
  EXPECT_EQ(cpuLeaf->usedBytes(), kCpuLimit);
  cpuLeaf->reportExternalFree(kCpuLimit);
}

TEST_F(PinnedSpillPoolTest, physicalRoundingDoesNotIncreaseLogicalAdmission) {
  // CUDA may round a small requested physical maximum up (e.g. 16 -> 32MiB).
  // That storage must never grant extra logical capacity to a query.
  constexpr uint64_t kLimit = 16u << 20;
  auto storage = PinnedSpillPool::create(kLimit, 0);
  auto query = storage->budget()->createQuery("rounded", kLimit);
  auto admission = storage->reserve(query, kLimit);
  auto allocation = admission.allocate(8u << 20);
  EXPECT_GE(storage->reservedBytes(), allocation.size());
  EXPECT_EQ(storage->budget()->limit(), kLimit);
  EXPECT_EQ(storage->budget()->used(), kLimit);
  EXPECT_THROW(storage->reserve(query, 1), VeloxException);
  admission = {};
  EXPECT_EQ(storage->budget()->used(), allocation.size());
  allocation = {};
  EXPECT_EQ(storage->budget()->used(), 0);
}

TEST_F(PinnedSpillPoolTest, independentQueriesShareTheWorkerLimit) {
  auto storage = PinnedSpillPool::create(16u << 20, 0);
  auto queryA = storage->budget()->createQuery("a", 16u << 20);
  auto queryB = storage->budget()->createQuery("b", 16u << 20);
  auto a = storage->reserve(queryA, 12u << 20);
  EXPECT_THROW(storage->reserve(queryB, 8u << 20), VeloxException);
  EXPECT_EQ(queryB->used(), 0);
  EXPECT_EQ(storage->budget()->used(), 12u << 20);
  a = {};
  auto b = storage->reserve(queryB, 8u << 20);
  auto allocation = b.allocate(8u << 20);
  EXPECT_EQ(queryA->used(), 0);
  EXPECT_EQ(queryB->used(), 8u << 20);
  EXPECT_GE(storage->reservedBytes(), allocation.size());
}
} // namespace
} // namespace facebook::velox::cudf_velox
