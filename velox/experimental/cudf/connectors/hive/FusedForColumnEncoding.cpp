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

#include "velox/experimental/cudf/connectors/hive/FusedForColumnEncoding.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include "velox/experimental/cudf/CudfNoDefaults.h"

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// A column can be encoded as dictionary indices when it has at most this many
// distinct values and at most one per this many rows.
constexpr cudf::size_type kMaxDictionaryKeys = 1 << 16;
constexpr cudf::size_type kMinRowsPerKey = 8;

cudf::detail::fused_for_packed_columns pack(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  cudf::detail::fused_for_options options;
  options.layout = cudf::detail::fused_for_layout::bitpacked;
  return cudf::detail::pack_fused_for(
      cudf::table_view{{column}}, stream, mr, options);
}

std::size_t payloadBytes(const cudf::detail::fused_for_packed_columns& packed) {
  return packed.wire_data->size() + packed.metadata->size();
}

// Returns 'column' dictionary encoded when it has few enough distinct values
// to be encoded as dictionary indices, otherwise null.
std::unique_ptr<cudf::column> encodeDictionary(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr) {
  if (column.nullable() || column.type().id() == cudf::type_id::BOOL8) {
    return nullptr;
  }
  const auto tooManyKeys = [&](cudf::size_type numKeys) {
    return numKeys > kMaxDictionaryKeys ||
        numKeys * kMinRowsPerKey > column.size();
  };
  // Counting the distinct values takes one hash pass. Encoding sorts them, so
  // it is skipped for columns such as keys, which have too many.
  if (tooManyKeys(cudf::distinct_count(
          column,
          cudf::null_policy::INCLUDE,
          cudf::nan_policy::NAN_IS_VALID,
          stream))) {
    return nullptr;
  }
  auto dictionary = cudf::dictionary::encode(
      column,
      cudf::data_type{cudf::type_id::INT32},
      stream,
      cudf::memory_resources{tempMr, tempMr});
  if (tooManyKeys(
          cudf::dictionary_column_view(dictionary->view()).keys_size())) {
    return nullptr;
  }
  return dictionary;
}

// The differences between consecutive values of a column, and the base of
// each of its segments when it has segments.
struct Deltas {
  std::unique_ptr<cudf::column> differences;
  std::vector<int64_t> segmentBases;
};

// Returns the differences between consecutive values of 'column' when it is a
// BIGINT column without a null mask or negative values (keys such as
// l_orderkey), otherwise nothing (see encodeForFusedFor() for the segments).
// The difference of two non-negative BIGINTs fits in BIGINT, so an inclusive
// prefix sum over a segment plus its base restores its values exactly. Rows
// need not be sorted: keys stored roughly in key order (SF3K lineitem files)
// still differ by much less than their range.
std::optional<Deltas> encodeDeltas(
    cudf::column_view column,
    const std::vector<cudf::size_type>& segmentStarts,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr) {
  if (column.nullable() || column.is_empty() ||
      column.type().id() != cudf::type_id::INT64) {
    return std::nullopt;
  }
  const auto minimum = cudf::minmax(column, stream, tempMr).first;
  if (static_cast<const cudf::numeric_scalar<int64_t>&>(*minimum).value(
          stream) < 0) {
    return std::nullopt;
  }
  const cudf::numeric_scalar<int64_t> zero(0, true, stream, tempMr);
  auto previous = cudf::shift(column, 1, zero, stream, tempMr);
  Deltas deltas;
  if (!segmentStarts.empty()) {
    VELOX_CHECK_EQ(segmentStarts.front(), 0, "The first segment starts at row 0");
    // Each segment's first row is subtracted from itself; its value is the
    // segment's base. Rows past the end start empty segments.
    std::vector<cudf::size_type> starts;
    for (const auto start : segmentStarts) {
      if (start < column.size() && (starts.empty() || start > starts.back())) {
        starts.push_back(start);
      }
    }
    const rmm::device_buffer startsBuffer(
        starts.data(), starts.size() * sizeof(cudf::size_type), stream, tempMr);
    const cudf::column_view startsView(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(starts.size()),
        startsBuffer.data(),
        nullptr,
        0);
    auto firstValues = cudf::gather(
        cudf::table_view{{column}},
        startsView,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        cudf::memory_resources{tempMr, tempMr});
    std::vector<int64_t> bases(starts.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        bases.data(),
        firstValues->view().column(0).data<int64_t>(),
        bases.size() * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.get()));
    previous = std::move(
        cudf::scatter(
            firstValues->view(),
            startsView,
            cudf::table_view{{previous->view()}},
            stream,
            tempMr)
            ->release()
            .front());
    stream.sync();
    size_t next = 0;
    for (const auto start : segmentStarts) {
      while (next < starts.size() && starts[next] < start) {
        ++next;
      }
      deltas.segmentBases.push_back(
          next < starts.size() && starts[next] == start ? bases[next] : 0);
    }
  }
  deltas.differences = cudf::binary_operation(
      column,
      previous->view(),
      cudf::binary_operator::SUB,
      column.type(),
      stream,
      tempMr);
  return deltas;
}

// Moves the indices and keys of 'dictionary' into 'result'.
void takeDictionary(
    std::unique_ptr<cudf::column> dictionary,
    FusedForEncodedColumn& result) {
  auto children = std::move(dictionary->release().children);
  result.encoding = FusedForEncoding::kDictionary;
  result.encoded = std::move(
      children[cudf::dictionary_column_view::indices_column_index]);
  result.keys =
      std::move(children[cudf::dictionary_column_view::keys_column_index]);
}

} // namespace

cudf::column_view withoutEmptyNullMask(cudf::column_view column) {
  if (!column.nullable() || column.null_count() > 0) {
    return column;
  }
  const std::vector<cudf::column_view> children(
      column.child_begin(), column.child_end());
  return cudf::column_view(
      column.type(),
      column.size(),
      column.head(),
      nullptr,
      0,
      column.offset(),
      children);
}

FusedForEncodedColumn encodeForFusedFor(
    cudf::column_view input,
    const std::vector<cudf::size_type>& segmentStarts,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref tempMr) {
  const auto column = withoutEmptyNullMask(input);
  FusedForEncodedColumn result;
  auto dictionary = encodeDictionary(column, stream, tempMr);
  if (column.type().id() == cudf::type_id::STRING) {
    // Fused FOR copies strings as they are, so their indices are always
    // smaller.
    if (dictionary != nullptr) {
      const cudf::dictionary_column_view view(dictionary->view());
      result.packedValues = pack(view.indices(), stream, tempMr);
      result.packedKeys = pack(view.keys(), stream, tempMr);
      takeDictionary(std::move(dictionary), result);
    }
    return result;
  }

  auto values = pack(column, stream, tempMr);
  if (dictionary != nullptr) {
    // Few distinct values spread over a wide range (e.g. decimal quantities)
    // take fewer bits as dictionary indices. Gathering the keys on read costs
    // a pass over the column, so the dictionary must save at least a quarter
    // of the bytes.
    const cudf::dictionary_column_view view(dictionary->view());
    auto indices = pack(view.indices(), stream, tempMr);
    auto keys = pack(view.keys(), stream, tempMr);
    if (4 * (payloadBytes(indices) + payloadBytes(keys)) <=
        3 * payloadBytes(values)) {
      result.packedValues = std::move(indices);
      result.packedKeys = std::move(keys);
      takeDictionary(std::move(dictionary), result);
      return result;
    }
  }
  // Keys stored near key order take a few bits per row as differences between
  // consecutive values where the values themselves take tens. Restoring them
  // costs a prefix sum on read, so the differences must also save at least a
  // quarter of the bytes.
  if (auto deltas = encodeDeltas(column, segmentStarts, stream, tempMr)) {
    auto packedDifferences = pack(deltas->differences->view(), stream, tempMr);
    if (4 * payloadBytes(packedDifferences) <= 3 * payloadBytes(values)) {
      result.encoding = FusedForEncoding::kDeltas;
      result.encoded = std::move(deltas->differences);
      result.packedValues = std::move(packedDifferences);
      result.segmentBases = std::move(deltas->segmentBases);
      return result;
    }
  }
  result.packedValues = std::move(values);
  return result;
}

std::unique_ptr<cudf::column> decodeFusedForEncoding(
    FusedForEncoding encoding,
    cudf::column_view values,
    std::optional<cudf::column_view> keys,
    int64_t base,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref tempMr) {
  switch (encoding) {
    case FusedForEncoding::kValues:
      return nullptr;
    case FusedForEncoding::kDeltas: {
      auto sums = cudf::scan(
          values,
          *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
          cudf::scan_type::INCLUSIVE,
          cudf::null_policy::EXCLUDE,
          stream,
          base == 0 ? mr : tempMr);
      if (base == 0) {
        return sums;
      }
      const cudf::numeric_scalar<int64_t> baseScalar(base, true, stream, tempMr);
      return cudf::binary_operation(
          sums->view(),
          baseScalar,
          cudf::binary_operator::ADD,
          values.type(),
          stream,
          mr);
    }
    case FusedForEncoding::kDictionary: {
      VELOX_CHECK(keys.has_value(), "A dictionary column needs its keys");
      auto gathered = cudf::gather(
          cudf::table_view{{*keys}},
          values,
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          cudf::memory_resources{mr, tempMr});
      return std::move(gathered->release().front());
    }
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::cudf_velox::connector::hive
