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

// Device side of DenseGroupby, free of Velox headers for nvcc.

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/stream_ref>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <vector>

namespace facebook::velox::cudf_velox::dense {

enum class Kind { kCountRows, kCount, kSum, kAvg, kMin, kMax };

void fill(int64_t* out, int64_t count, int64_t value, cuda::stream_ref stream);

/// The slot of each row, sum over keys of (key - lows[k]) * strides[k], into
/// 'out', or 'slots' for a row with a key outside [lows[k], lows[k] +
/// sizes[k]).
void combine(
    const std::vector<cudf::column_view>& keys,
    const std::vector<int64_t>& lows,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t slots,
    int32_t* out,
    cuda::stream_ref stream);

/// One aggregate for addShared: 'value' is unused for kCountRows.
struct Target {
  cudf::column_view value;
  Kind kind;
  bool floating;
  void* accumulators;
  unsigned long long* counts;
};

/// The largest slot count addShared takes for 'targets' aggregates.
int64_t sharedSlots(size_t targets);

/// Like addRows and add for every target in one pass, accumulating per block
/// in shared memory first. For few slots, where atomics on global memory
/// would contend.
void addShared(
    const cudf::column_view& key,
    int64_t low,
    int64_t slots,
    const std::vector<Target>& targets,
    unsigned long long* rows,
    int* outOfRange,
    cuda::stream_ref stream);

/// Counts the rows of each key - low in 'rows', and sets 'outOfRange' for a
/// key outside [low, low + slots).
void addRows(
    const cudf::column_view& key,
    int64_t low,
    int64_t slots,
    unsigned long long* rows,
    int* outOfRange,
    cuda::stream_ref stream);

/// Adds the non-null 'value' of each row to the slot of its key: to the
/// count, and for sums and averages to an int128 or double sum, for minima
/// and maxima to an int64 or double extreme.
void add(
    const cudf::column_view& key,
    int64_t low,
    int64_t slots,
    const cudf::column_view& value,
    Kind kind,
    bool floating,
    void* accumulators,
    unsigned long long* counts,
    cuda::stream_ref stream);

/// How a partial state field combines into an accumulator.
enum class MergeOp : int32_t {
  kAddCount,
  kAddInt128,
  kAddDouble,
  kMinInteger,
  kMaxInteger,
  kMinDouble,
  kMaxDouble
};

/// One partial state field, or for kAddInt128 its low and high halves, and
/// the accumulator it combines into. Sources point to device memory.
struct Merge {
  MergeOp op;
  const void* source;
  const void* sourceHigh;
  void* target;
};

/// Combines every merge in 'merges', a device array of 'count', atomically
/// into its target.
void merge(const Merge* merges, int32_t count, cuda::stream_ref stream);

/// The slots with rows, in order.
rmm::device_uvector<cudf::size_type> occupied(
    const unsigned long long* rows,
    int64_t slots,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// low + slot / stride % size for each of 'slots', into 'out'.
void keys(
    const rmm::device_uvector<cudf::size_type>& slots,
    int64_t low,
    int64_t stride,
    int64_t size,
    cudf::mutable_column_view out,
    cuda::stream_ref stream);

/// The result of each of 'slots' into 'out', with 'valid' false where no
/// input was counted. Averages of integers round half away from zero.
void results(
    const rmm::device_uvector<cudf::size_type>& slots,
    Kind kind,
    bool floating,
    const void* accumulators,
    const unsigned long long* counts,
    cudf::mutable_column_view out,
    bool* valid,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox::dense
