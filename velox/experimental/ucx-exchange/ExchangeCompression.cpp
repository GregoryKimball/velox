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

#include "velox/experimental/ucx-exchange/ExchangeCompression.h"

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#if __has_include(<cudf/detail/fused_for.hpp>)
#define VELOX_UCX_HAS_FUSED_FOR 1
#include <cudf/detail/fused_for.hpp>
#else
#define VELOX_UCX_HAS_FUSED_FOR 0
#endif

namespace facebook::velox::ucx_exchange {
namespace {

constexpr std::string_view kFusedForUnavailable{
    "Fused FOR exchange compression is not available in this build"};

bool isFusedFor(ExchangeCompression compression) {
  return compression == ExchangeCompression::kFusedForBitpacked ||
      compression == ExchangeCompression::kFusedForByteAligned ||
      compression == ExchangeCompression::kFusedForBitpackedLanes;
}

#if VELOX_UCX_HAS_FUSED_FOR
constexpr std::size_t kFusedForTileBytes = 32 * 1024;

cudf::detail::fused_for_options fusedForOptions(
    ExchangeCompression compression) {
  return cudf::detail::fused_for_options{
      kFusedForTileBytes,
      compression == ExchangeCompression::kFusedForByteAligned
          ? cudf::detail::fused_for_layout::byte_aligned
          : cudf::detail::fused_for_layout::bitpacked};
}

// Rows wider than this many lanes are sent as strings. The decimal aggregation
// states are four lanes wide.
constexpr uint32_t kMaxLanes = 16;

// Columns with fewer rows are sent as strings: they fill less than one fused
// FOR tile of 32-byte rows, so lanes save little and the check below costs a
// synchronization per column.
constexpr cudf::size_type kMinLaneRows = 1024;

// Returns the number of eight-byte lanes in each row of `column`, or zero when
// the column cannot be split into lanes. That is the case unless it is an
// unsliced strings column without nulls, of at least kMinLaneRows rows, whose
// rows all have the same width, a multiple of eight bytes.
uint32_t laneCount(
    cudf::column_view column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref memoryResource) {
  if (column.type().id() != cudf::type_id::STRING ||
      column.size() < kMinLaneRows || column.has_nulls() ||
      column.offset() != 0) {
    return 0;
  }
  const cudf::strings_column_view strings{column};
  if (strings.offsets().type().id() != cudf::type_id::INT32 ||
      reinterpret_cast<std::uintptr_t>(strings.chars_begin(stream)) %
              alignof(int64_t) !=
          0) {
    return 0;
  }

  // Every row has the same width when the narrowest equals the widest. Read
  // both with one synchronization.
  using SizeScalar = cudf::numeric_scalar<cudf::size_type>;
  auto widths = cudf::strings::count_bytes(strings, stream, memoryResource);
  auto [minWidth, maxWidth] =
      cudf::minmax(widths->view(), stream, memoryResource);
  std::array<cudf::size_type, 2> bounds;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &bounds[0],
      static_cast<SizeScalar&>(*minWidth).data(),
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &bounds[1],
      static_cast<SizeScalar&>(*maxWidth).data(),
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  const auto rowBytes = bounds[0];
  if (rowBytes != bounds[1] || rowBytes <= 0 ||
      rowBytes % sizeof(int64_t) != 0 ||
      rowBytes / sizeof(int64_t) > kMaxLanes) {
    return 0;
  }
  return static_cast<uint32_t>(rowBytes / sizeof(int64_t));
}

// A table in which some string columns were replaced by lane columns.
struct LaneTable {
  cudf::table_view view;
  std::vector<ExchangeLaneColumn> laneColumns;
  // Owns the lane columns in `view`.
  std::vector<std::unique_ptr<cudf::column>> lanes;
};

// Replaces each string column of `input` that laneCount() accepts with one
// INT64 column per lane. Lane `l` of row `r` is the eight bytes at offset
// 8 * l of the row. Other columns are unchanged.
LaneTable splitIntoLanes(
    cudf::table_view input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource) {
  LaneTable result;
  std::vector<cudf::column_view> columns;
  for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
    const auto column = input.column(i);
    const auto lanes = laneCount(column, stream, temporaryMemoryResource);
    if (lanes == 0 ||
        static_cast<int64_t>(column.size()) * lanes >
            std::numeric_limits<cudf::size_type>::max()) {
      columns.push_back(column);
      continue;
    }

    // View the characters as row-major INT64 values and gather every lanes-th
    // value starting at each lane.
    const cudf::column_view words{
        cudf::data_type{cudf::type_id::INT64},
        static_cast<cudf::size_type>(column.size() * lanes),
        cudf::strings_column_view{column}.chars_begin(stream),
        nullptr,
        0};
    const cudf::table_view source{std::vector<cudf::column_view>{words}};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      auto gatherMap = cudf::sequence(
          column.size(),
          cudf::numeric_scalar<cudf::size_type>(
              static_cast<cudf::size_type>(lane),
              true,
              stream,
              temporaryMemoryResource),
          cudf::numeric_scalar<cudf::size_type>(
              static_cast<cudf::size_type>(lanes),
              true,
              stream,
              temporaryMemoryResource),
          stream,
          temporaryMemoryResource);
      auto gathered = cudf::gather(
                          source,
                          gatherMap->view(),
                          cudf::out_of_bounds_policy::DONT_CHECK,
                          stream,
                          cudf::memory_resources{
                              temporaryMemoryResource,
                              temporaryMemoryResource})
                          ->release();
      columns.push_back(gathered.front()->view());
      result.lanes.push_back(std::move(gathered.front()));
    }
    result.laneColumns.push_back(
        ExchangeLaneColumn{static_cast<uint32_t>(i), lanes});
  }
  result.view = cudf::table_view{columns};
  return result;
}

// Reverses splitIntoLanes(): returns an owning table with `encoded`'s columns,
// in which the lane columns of each entry of `laneColumns` are joined into the
// original strings column.
std::unique_ptr<cudf::table> joinLanes(
    cudf::table_view encoded,
    const std::vector<ExchangeLaneColumn>& laneColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref memoryResource) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto laneColumn = laneColumns.begin();
  cudf::size_type next = 0;
  while (next < encoded.num_columns()) {
    if (laneColumn == laneColumns.end() ||
        laneColumn->column != columns.size()) {
      columns.push_back(std::make_unique<cudf::column>(
          encoded.column(next), stream, memoryResource));
      ++next;
      continue;
    }

    VELOX_CHECK_LE(
        static_cast<int64_t>(next) + laneColumn->lanes,
        encoded.num_columns(),
        "Exchange payload is missing lanes of column {}",
        laneColumn->column);
    std::vector<cudf::column_view> lanes;
    for (uint32_t lane = 0; lane < laneColumn->lanes; ++lane) {
      const auto& column = encoded.column(next + lane);
      VELOX_CHECK(
          column.type().id() == cudf::type_id::INT64 && !column.has_nulls(),
          "Exchange payload lane {} of column {} is not a non-null INT64 column",
          lane,
          laneColumn->column);
      lanes.push_back(column);
    }
    const auto rows = lanes.front().size();
    const auto rowBytes =
        static_cast<int64_t>(laneColumn->lanes) * sizeof(int64_t);
    VELOX_CHECK_LE(
        rows * rowBytes,
        std::numeric_limits<cudf::size_type>::max(),
        "Exchange payload strings column is too large");

    // Interleaving the lanes restores the row-major characters.
    auto chars = cudf::interleave_columns(
                     cudf::table_view{lanes}, stream, memoryResource)
                     ->release();
    auto offsets = cudf::sequence(
        rows + 1,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, memoryResource),
        cudf::numeric_scalar<cudf::size_type>(
            static_cast<cudf::size_type>(rowBytes),
            true,
            stream,
            memoryResource),
        stream,
        memoryResource);
    columns.push_back(cudf::make_strings_column(
        rows,
        std::move(offsets),
        std::move(*chars.data),
        0,
        cudf::create_null_mask(
            rows, cudf::mask_state::UNALLOCATED, stream, memoryResource)));
    next += static_cast<cudf::size_type>(laneColumn->lanes);
    ++laneColumn;
  }
  VELOX_CHECK(
      laneColumn == laneColumns.end(),
      "Exchange payload has no lanes for column {}",
      laneColumn->column);
  return std::make_unique<cudf::table>(std::move(columns));
}
#endif

[[noreturn]] void fusedForUnavailable() {
  VELOX_UNSUPPORTED(kFusedForUnavailable);
}

PackedExchangePayload makePayload(
    cudf::packed_columns packed,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource) {
  VELOX_CHECK_NOT_NULL(packed.metadata);
  VELOX_CHECK_NOT_NULL(packed.gpu_data);
  const auto logicalDataSize = packed.gpu_data->size();
  PackedExchangePayload raw{
      std::make_unique<cudf::packed_columns>(std::move(packed)),
      ExchangePayloadCodec::kNone,
      logicalDataSize,
      0};
  if (compression != ExchangeCompression::kCascaded || logicalDataSize == 0) {
    return raw;
  }

  auto options = cudf::experimental::pack_options{};
  options.compression = cudf::experimental::pack_compression::cascaded;
  auto plan = cudf::experimental::make_pack_plan_builder(
                  *raw.packed,
                  options,
                  stream,
                  cudf::memory_resources{
                      temporaryMemoryResource, temporaryMemoryResource})
                  .build();
  const auto sizes = plan.sizes();
  VELOX_CHECK_EQ(sizes.uncompressed_payload_bytes, logicalDataSize);
  auto output = std::make_unique<rmm::device_buffer>(
      sizes.payload_bytes, stream, outputMemoryResource);
  auto result = cudf::experimental::pack_into(
      plan,
      std::span<uint8_t>{static_cast<uint8_t*>(output->data()), output->size()},
      cudf::memory_resources{temporaryMemoryResource, temporaryMemoryResource});
  VELOX_CHECK_LE(result.payload_bytes, output->size());

  // pack_into() finishes reading the original packed data before it returns.
  // If the compressed data is not smaller, send the original data instead.
  if (result.payload_bytes >= logicalDataSize) {
    return raw;
  }

  output->resize(result.payload_bytes, stream);
  auto metadata = wrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(std::move(result.metadata)),
      ExchangePayloadCodec::kCascaded,
      logicalDataSize,
      0);
  return PackedExchangePayload{
      std::make_unique<cudf::packed_columns>(
          std::move(metadata), std::move(output)),
      ExchangePayloadCodec::kCascaded,
      logicalDataSize,
      0};
}

#if VELOX_UCX_HAS_FUSED_FOR
PackedExchangePayload makeFusedForPayload(
    cudf::detail::fused_for_packed_columns packed,
    const std::vector<ExchangeLaneColumn>& laneColumns) {
  VELOX_CHECK_NOT_NULL(packed.wire_data);
  const auto logicalDataSize = packed.logical_data_size;
  const auto segmentCount = packed.segment_count;
  const auto codec = laneColumns.empty() ? ExchangePayloadCodec::kFusedFor
                                         : ExchangePayloadCodec::kFusedForLanes;
  auto metadata = wrapExchangePayloadMetadata(
      std::move(packed.metadata),
      codec,
      logicalDataSize,
      segmentCount,
      laneColumns);
  return PackedExchangePayload{
      std::make_unique<cudf::packed_columns>(
          std::move(metadata), std::move(packed.wire_data)),
      codec,
      logicalDataSize,
      segmentCount};
}

// Returns `input` with string columns split into lanes when `compression`
// uses lanes, and `input` unchanged otherwise.
LaneTable fusedForInput(
    cudf::table_view input,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource) {
  if (compression != ExchangeCompression::kFusedForBitpackedLanes) {
    return LaneTable{input, {}, {}};
  }
  return splitIntoLanes(input, stream, temporaryMemoryResource);
}
#endif

std::unique_ptr<PackedTableWithStream> retainPackedPayload(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    vector_size_t numRows) {
  VELOX_CHECK_NOT_NULL(metadata);
  VELOX_CHECK_NOT_NULL(data);
  auto packedColumns =
      cudf::packed_columns(std::move(metadata), std::move(data));
  auto tableView = cudf::unpack(packedColumns);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});

  // Keep the packed table with the CUDA stream used for later GPU operations.
  // Use the producer's row count because a table with no columns reports zero.
  return std::make_unique<PackedTableWithStream>(
      std::move(packedTable), stream, numRows);
}

} // namespace

ExchangeCompression parseExchangeCompression(std::string_view value) {
  if (value == "none") {
    return ExchangeCompression::kNone;
  }
  if (value == "cascaded") {
    return ExchangeCompression::kCascaded;
  }
  if (value == "fused-for-bitpacked" || value == "fused-for-byte-aligned" ||
      value == "fused-for-bitpacked-lanes") {
    VELOX_USER_CHECK(VELOX_UCX_HAS_FUSED_FOR, kFusedForUnavailable);
    if (value == "fused-for-bitpacked-lanes") {
      return ExchangeCompression::kFusedForBitpackedLanes;
    }
    return value == "fused-for-bitpacked"
        ? ExchangeCompression::kFusedForBitpacked
        : ExchangeCompression::kFusedForByteAligned;
  }
  VELOX_USER_FAIL(
      "Unsupported cuDF exchange compression '{}'. Expected none, "
      "fused-for-bitpacked, fused-for-byte-aligned, "
      "fused-for-bitpacked-lanes, or cascaded",
      value);
}

PackedExchangePayload packExchangePayload(
    cudf::table_view input,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource) {
  // A table without columns has no data to compress, and fused FOR needs at
  // least one column.
  const auto effectiveCompression =
      input.num_columns() == 0 ? ExchangeCompression::kNone : compression;
  switch (effectiveCompression) {
    case ExchangeCompression::kNone:
    case ExchangeCompression::kCascaded:
      return makePayload(
          cudf::pack(input, stream, outputMemoryResource),
          effectiveCompression,
          stream,
          temporaryMemoryResource,
          outputMemoryResource);
    case ExchangeCompression::kFusedForBitpacked:
    case ExchangeCompression::kFusedForByteAligned:
    case ExchangeCompression::kFusedForBitpackedLanes: {
#if VELOX_UCX_HAS_FUSED_FOR
      auto encoded = fusedForInput(
          input, effectiveCompression, stream, temporaryMemoryResource);
      return makeFusedForPayload(
          cudf::detail::pack_fused_for(
              encoded.view,
              stream,
              outputMemoryResource,
              fusedForOptions(effectiveCompression)),
          encoded.laneColumns);
#else
      fusedForUnavailable();
#endif
    }
  }
  VELOX_UNREACHABLE();
}

std::vector<PackedExchangePayload> splitExchangePayloads(
    cudf::table_view input,
    const std::vector<cudf::size_type>& splits,
    ExchangeCompression compression,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource) {
  VELOX_CHECK_GT(input.num_columns(), 0);

  if (isFusedFor(compression)) {
#if VELOX_UCX_HAS_FUSED_FOR
    auto encoded =
        fusedForInput(input, compression, stream, temporaryMemoryResource);
    auto packed = cudf::detail::contiguous_split_fused_for(
        encoded.view,
        splits,
        stream,
        outputMemoryResource,
        fusedForOptions(compression));
    std::vector<PackedExchangePayload> result;
    result.reserve(packed.size());
    for (auto& partition : packed) {
      result.push_back(
          makeFusedForPayload(std::move(partition), encoded.laneColumns));
    }
    return result;
#else
    fusedForUnavailable();
#endif
  }

  VELOX_CHECK(
      compression == ExchangeCompression::kNone ||
          compression == ExchangeCompression::kCascaded,
      "Unsupported exchange compression mode");

  auto packed =
      cudf::contiguous_split(input, splits, stream, outputMemoryResource);
  std::vector<PackedExchangePayload> result;
  result.reserve(packed.size());
  for (auto& partition : packed) {
    result.push_back(makePayload(
        std::move(partition.data),
        compression,
        stream,
        temporaryMemoryResource,
        outputMemoryResource));
  }
  return result;
}

std::unique_ptr<PackedTableWithStream> restoreExchangePayload(
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    cuda::stream_ref stream,
    vector_size_t numRows) {
  auto envelope = unwrapExchangePayloadMetadata(std::move(metadata));
  switch (envelope.codec) {
    case ExchangePayloadCodec::kNone:
      return retainPackedPayload(
          std::move(envelope.cudfMetadata), std::move(data), stream, numRows);
    case ExchangePayloadCodec::kCascaded: {
      // 'data' may have been allocated on the producer's stream. Free it on
      // 'stream' so the release is ordered after materialize() reads it.
      data->set_stream(stream);
      auto memoryResource = data->memory_resource();
      auto packed = cudf::experimental::packed_data_view{
          *envelope.cudfMetadata,
          std::span<uint8_t const>{
              static_cast<uint8_t const*>(data->data()), data->size()}};
      auto table = cudf::experimental::materialize(
          packed,
          stream,
          cudf::memory_resources{memoryResource, memoryResource});
      return std::make_unique<PackedTableWithStream>(
          std::move(table), stream, envelope.logicalDataSize, numRows);
    }
    case ExchangePayloadCodec::kFusedFor:
    case ExchangePayloadCodec::kFusedForLanes:
#if VELOX_UCX_HAS_FUSED_FOR
    {
      // 'memoryResource' refers to the resource owned by 'data', which
      // decode_fused_for() consumes and frees. Do not use it after the call.
      auto memoryResource = data->memory_resource();
      auto decoded = cudf::detail::decode_fused_for(
          cudf::detail::fused_for_packed_columns{
              std::move(envelope.cudfMetadata),
              std::move(data),
              envelope.auxiliaryCount,
              envelope.logicalDataSize},
          stream,
          memoryResource);
      auto result = retainPackedPayload(
          std::move(decoded.metadata),
          std::move(decoded.gpu_data),
          stream,
          numRows);
      result->laneColumns = std::move(envelope.laneColumns);
      return result;
    }
#else
      fusedForUnavailable();
#endif
  }
  VELOX_UNREACHABLE();
}

void joinExchangeLanes(
    PackedTableWithStream& data,
    rmm::device_async_resource_ref memoryResource) {
  if (data.laneColumns.empty()) {
    return;
  }
#if VELOX_UCX_HAS_FUSED_FOR
  VELOX_CHECK_NOT_NULL(data.packedTable);
  data.table = joinLanes(
      data.packedTable->table, data.laneColumns, data.stream, memoryResource);
  // The packed buffer was allocated on data.stream, so it is freed after the
  // join has read it.
  data.packedTable.reset();
  data.laneColumns.clear();
#else
  fusedForUnavailable();
#endif
}

} // namespace facebook::velox::ucx_exchange
