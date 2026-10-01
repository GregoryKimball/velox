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

#include <cstddef>
#include <cstdint>

namespace facebook::velox::functions::aggregate {

// The historical decimal SUM/AVG intermediate state is an unmarked 32-byte
// tuple of count, overflow, and a two-word signed sum. Its layout cannot change
// without breaking already serialized states.
inline constexpr size_t kLegacyDecimalIntermediateStateSize = 32;

// Aggregate counts are non-negative signed int64 values. The otherwise-unused
// sign bit is an explicit compact-format marker, so the remaining bits preserve
// every valid count without reducing its range.
inline constexpr size_t kCompactDecimalIntermediateStateSize = 16;
inline constexpr uint64_t kCompactDecimalIntermediateStateTag =
    UINT64_C(0x8000000000000000);
inline constexpr uint64_t kCompactDecimalIntermediateStateTagMask =
    kCompactDecimalIntermediateStateTag;
inline constexpr uint64_t kCompactDecimalIntermediateStateCountMask =
    UINT64_C(0x7fffffffffffffff);

struct CompactDecimalIntermediateState {
  uint64_t taggedCount;
  int64_t sum;
};

static_assert(
    sizeof(CompactDecimalIntermediateState) ==
    kCompactDecimalIntermediateStateSize);

} // namespace facebook::velox::functions::aggregate
