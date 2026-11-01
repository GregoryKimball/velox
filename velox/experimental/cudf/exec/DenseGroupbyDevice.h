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

namespace facebook::velox::cudf_velox::dense {

enum class Kind { kCountRows, kCount, kSum, kAvg, kMin, kMax };

void fill(int64_t* out, int64_t count, int64_t value, cuda::stream_ref stream);

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

/// The slots with rows, in order.
rmm::device_uvector<cudf::size_type> occupied(
    const unsigned long long* rows,
    int64_t slots,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// low + slot for each of 'slots', into 'out'.
void keys(
    const rmm::device_uvector<cudf::size_type>& slots,
    int64_t low,
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
