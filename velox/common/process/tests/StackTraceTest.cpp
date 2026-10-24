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

#include "velox/common/process/StackTrace.h"

#include <gtest/gtest.h>

// <arm_sve.h> can be used with a per-function target attribute only when the
// whole translation unit is not built for SVE starting with GCC 14.
#if defined(__aarch64__) && defined(__linux__) && \
    (defined(__ARM_FEATURE_SVE) ||                  \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 14))
#define VELOX_TEST_SVE_FRAMES 1
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#else
#define VELOX_TEST_SVE_FRAMES 0
#endif

namespace facebook::velox::process {
namespace {

TEST(StackTraceTest, capture) {
  uintptr_t addresses[64];
  const auto count = StackTrace::capture(addresses, 64);
  ASSERT_GT(count, 1);
  ASSERT_LE(count, 64);
  ASSERT_EQ(StackTrace::capture(addresses, 1), 1);
}

#if VELOX_TEST_SVE_FRAMES
__attribute__((noipa)) ssize_t captureStack() {
  return StackTrace().getStack().size();
}

// A function with an SVE argument follows the SVE calling convention, so it
// saves z8-z23 and p4-p15 and its CFI locates them relative to the vector
// granule register (DWARF register 46). The clobbers after the call keep the
// saves live while the stack is captured.
__attribute__((noipa, target("+sve"))) ssize_t captureBelowSveFrame(
    svbool_t pg) {
  const auto frames = captureStack();
  asm volatile("" ::: "z8", "z16", "p4");
  return frames + static_cast<ssize_t>(svcntp_b8(pg, pg)) * 0;
}

__attribute__((noipa, target("+sve"))) ssize_t captureThroughSveFrame() {
  return captureBelowSveFrame(svptrue_b8());
}

TEST(StackTraceTest, captureThroughSveFrame) {
  if ((getauxval(AT_HWCAP) & HWCAP_SVE) == 0) {
    GTEST_SKIP() << "CPU does not support SVE";
  }
  // At least captureStack, captureBelowSveFrame, captureThroughSveFrame and
  // this test.
  ASSERT_GE(captureThroughSveFrame(), 4);
}
#endif

} // namespace
} // namespace facebook::velox::process
