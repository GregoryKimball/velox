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
#include "velox/experimental/cudf/exec/HostSpillBudget.h"

#include "velox/common/memory/MemoryArbitrator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <latch>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {
TEST(HostSpillBudgetTest, enforcedDuringArbitrationAndNonRetriable) {
  auto worker = HostSpillBudget::createWorker(100);
  auto query = worker->createQuery("query-a", 80);
  auto admitted = query->reserve(80);
  memory::ScopedMemoryArbitrationContext arbitration;
  try {
    query->reserve(1);
    FAIL() << "Expected a hard host-budget failure";
  } catch (const VeloxException& error) {
    EXPECT_EQ(error.errorCode(), error_code::kMemCapExceeded);
    EXPECT_FALSE(error.isRetriable());
    EXPECT_NE(
        std::string(error.what()).find("Host spill budget exceeded"),
        std::string::npos);
  }
  EXPECT_EQ(query->used(), 80);
  EXPECT_EQ(worker->used(), 80);
}

TEST(HostSpillBudgetTest, retainedAllocationAndCancellationReleaseExactlyOnce) {
  auto worker = HostSpillBudget::createWorker(100);
  auto query = worker->createQuery("query-a", 80);
  std::optional<HostSpillBudget::Reservation> allocation;
  {
    auto pending = query->reserve(80);
    allocation.emplace(pending.split(30));
    EXPECT_EQ(pending.bytes(), 50);
    EXPECT_TRUE(allocation->belongsTo(worker.get()));
    EXPECT_EQ(worker->used(), 80);
  }
  EXPECT_EQ(worker->used(), 30);
  EXPECT_EQ(query->used(), 30);
  allocation.reset();
  EXPECT_EQ(worker->used(), 0);
  EXPECT_EQ(query->used(), 0);
  EXPECT_EQ(query->peak(), 80);
  auto retry = query->reserve(80);
  EXPECT_EQ(retry.bytes(), 80);
}

TEST(HostSpillBudgetTest, concurrentQueriesCannotOvercommitWorker) {
  auto worker = HostSpillBudget::createWorker(300);
  std::atomic<int> accepted{0}, rejected{0};
  std::latch ready(4), release(1);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i] {
      auto query = worker->createQuery(std::to_string(i), 100);
      std::optional<HostSpillBudget::Reservation> token;
      try {
        token.emplace(query->reserve(100));
        ++accepted;
      } catch (const VeloxException&) {
        ++rejected;
      }
      ready.count_down();
      release.wait();
    });
  }
  ready.wait();
  EXPECT_EQ(accepted, 3);
  EXPECT_EQ(rejected, 1);
  EXPECT_EQ(worker->used(), 300);
  release.count_down();
  for (auto& thread : threads)
    thread.join();
  EXPECT_EQ(worker->used(), 0);
}

TEST(HostSpillBudgetTest, zeroDisablesAndArithmeticCannotWrap) {
  auto disabled = HostSpillBudget::createWorker(0);
  EXPECT_EQ(disabled->reserve(0).bytes(), 0);
  EXPECT_THROW(disabled->reserve(1), VeloxException);
  auto large =
      HostSpillBudget::createWorker(std::numeric_limits<uint64_t>::max());
  auto all = large->reserve(std::numeric_limits<uint64_t>::max());
  EXPECT_THROW(large->reserve(1), VeloxException);
  EXPECT_EQ(large->used(), std::numeric_limits<uint64_t>::max());
}
} // namespace
} // namespace facebook::velox::cudf_velox
