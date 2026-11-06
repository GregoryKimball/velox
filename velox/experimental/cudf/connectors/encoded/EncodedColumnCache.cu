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

#include "velox/experimental/cudf/connectors/encoded/EncodedColumnCache.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/algorithm>
#include <cuda/std/limits>

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace facebook::velox::cudf_velox::connector::encoded {

namespace {

constexpr int kThreads = 256;

// This file is compiled by nvcc, which cannot parse Velox's folly-based
// headers on aarch64, so errors are reported with standard exceptions.
void check(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int64_t numBlocks(int64_t n) {
  return std::max<int64_t>(1, (n + kThreads - 1) / kThreads);
}

void checkCuda(cudaError_t error, const char* what) {
  check(
      error == cudaSuccess,
      std::string(what) + " failed: " + cudaGetErrorString(error));
}

template <typename T>
__global__ void segmentMinMaxKernel(
    const T* data,
    int64_t numRows,
    int32_t segmentRows,
    int64_t* outMin,
    int64_t* outMax) {
  using BlockReduce = cub::BlockReduce<int64_t, kThreads>;
  __shared__ typename BlockReduce::TempStorage minStorage;
  __shared__ typename BlockReduce::TempStorage maxStorage;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * segmentRows;
  const int64_t end = cuda::std::min<int64_t>(begin + segmentRows, numRows);
  int64_t localMin = cuda::std::numeric_limits<int64_t>::max();
  int64_t localMax = cuda::std::numeric_limits<int64_t>::min();
  for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
    const int64_t value = static_cast<int64_t>(data[row]);
    localMin = min(localMin, value);
    localMax = max(localMax, value);
  }
  const int64_t blockMin = BlockReduce(minStorage).Reduce(localMin, cuda::minimum<>{});
  const int64_t blockMax = BlockReduce(maxStorage).Reduce(localMax, cuda::maximum<>{});
  if (threadIdx.x == 0) {
    outMin[blockIdx.x] = blockMin;
    outMax[blockIdx.x] = blockMax;
  }
}

template <typename T>
__global__ void packKernel(
    const T* data,
    int64_t numRows,
    int32_t segmentRows,
    const SegmentParams* params,
    unsigned long long* words) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (row >= numRows) {
    return;
  }
  const auto& p = params[row / segmentRows];
  if (p.bitWidth == 0) {
    return;
  }
  const uint64_t value =
      static_cast<uint64_t>(static_cast<int64_t>(data[row]) - p.base);
  const uint64_t bit = p.bitOffset + (row % segmentRows) * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  atomicOr(&words[word], static_cast<unsigned long long>(value << shift));
  if (shift + p.bitWidth > 64) {
    atomicOr(
        &words[word + 1], static_cast<unsigned long long>(value >> (64 - shift)));
  }
}

__device__ __forceinline__ int64_t
unpack(const uint64_t* words, const SegmentParams& p, int64_t index) {
  if (p.bitWidth == 0) {
    return p.base;
  }
  const uint64_t bit = p.bitOffset + index * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  uint64_t value = words[word] >> shift;
  if (shift + p.bitWidth > 64) {
    value |= words[word + 1] << (64 - shift);
  }
  const uint64_t mask =
      p.bitWidth == 64 ? ~0ULL : ((1ULL << p.bitWidth) - 1);
  return p.base + static_cast<int64_t>(value & mask);
}

/// Decodes 'numOut' values. Output j is row 'rows[j]' of the concatenated
/// segments, or row j without 'rows'.
template <typename OutT>
__global__ void unpackKernel(
    const uint64_t* words,
    const SegmentParams* params,
    const int32_t* segments,
    int32_t segmentRows,
    const int32_t* rows,
    int64_t numOut,
    OutT* out) {
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= numOut) {
    return;
  }
  const int64_t row = rows ? rows[j] : j;
  const auto& p = params[segments[row / segmentRows]];
  out[j] = static_cast<OutT>(unpack(words, p, row % segmentRows));
}

/// Maps output j to the table row it reads, for gathers from kPlain columns.
__global__ void globalRowsKernel(
    const int32_t* segments,
    int32_t segmentRows,
    const int32_t* rows,
    int64_t numOut,
    int32_t* out) {
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= numOut) {
    return;
  }
  const int64_t row = rows ? rows[j] : j;
  out[j] = static_cast<int32_t>(
      static_cast<int64_t>(segments[row / segmentRows]) * segmentRows +
      row % segmentRows);
}

bool isIntegerLike(cudf::data_type type) {
  switch (type.id()) {
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
      return true;
    default:
      return false;
  }
}

uint32_t bitWidthOf(uint64_t range) {
  return range == 0 ? 0 : 64 - __builtin_clzll(range);
}

/// Fills the kForBitpack streams of 'column' from 'values', a 4- or 8-byte
/// integer-like column.
void encodeForBitpack(
    EncodedColumn& column,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const int64_t numRows = values.size();
  const int32_t numSegments =
      static_cast<int32_t>((numRows + segmentRows - 1) / segmentRows);
  const bool wide = cudf::size_of(values.type()) == 8;
  check(
      wide || cudf::size_of(values.type()) == 4,
      "Unsupported width for bit-packing: " +
          std::to_string(cudf::size_of(values.type())));

  rmm::device_uvector<int64_t> deviceMin(numSegments, stream, mr);
  rmm::device_uvector<int64_t> deviceMax(numSegments, stream, mr);
  if (numSegments > 0 && numRows > 0) {
    if (wide) {
      segmentMinMaxKernel<int64_t>
          <<<numSegments, kThreads, 0, stream.get()>>>(
              values.data<int64_t>(),
              numRows,
              segmentRows,
              deviceMin.data(),
              deviceMax.data());
    } else {
      segmentMinMaxKernel<int32_t>
          <<<numSegments, kThreads, 0, stream.get()>>>(
              values.data<int32_t>(),
              numRows,
              segmentRows,
              deviceMin.data(),
              deviceMax.data());
    }
    checkCuda(cudaGetLastError(), "segmentMinMaxKernel");
  }
  column.segmentMin.resize(numSegments);
  column.segmentMax.resize(numSegments);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentMin.data(),
          deviceMin.data(),
          numSegments * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.get()),
      "copy segment minimums");
  checkCuda(
      cudaMemcpyAsync(
          column.segmentMax.data(),
          deviceMax.data(),
          numSegments * sizeof(int64_t),
          cudaMemcpyDeviceToHost,
          stream.get()),
      "copy segment maximums");
  stream.sync();

  std::vector<SegmentParams> params(numSegments);
  column.segmentBitWidth.resize(numSegments);
  uint64_t bitOffset = 0;
  for (int32_t s = 0; s < numSegments; ++s) {
    const uint64_t range = static_cast<uint64_t>(column.segmentMax[s]) -
        static_cast<uint64_t>(column.segmentMin[s]);
    const uint32_t width = bitWidthOf(range);
    const int64_t rows =
        std::min<int64_t>(segmentRows, numRows - int64_t{s} * segmentRows);
    params[s] = SegmentParams{column.segmentMin[s], bitOffset, width, 0};
    column.segmentBitWidth[s] = width;
    bitOffset += ((rows * width + 63) / 64) * 64;
  }
  const int64_t numWords = static_cast<int64_t>(bitOffset / 64) + 1;

  column.segmentParams = rmm::device_uvector<SegmentParams>(numSegments, stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentParams.data(),
          params.data(),
          numSegments * sizeof(SegmentParams),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy segment params");
  column.packed = rmm::device_uvector<uint64_t>(numWords, stream, mr);
  checkCuda(
      cudaMemsetAsync(
          column.packed.data(), 0, numWords * sizeof(uint64_t), stream.get()),
      "clear packed words");
  if (numRows > 0) {
    auto* words =
        reinterpret_cast<unsigned long long*>(column.packed.data());
    if (wide) {
      packKernel<int64_t><<<numBlocks(numRows), kThreads, 0, stream.get()>>>(
          values.data<int64_t>(),
          numRows,
          segmentRows,
          column.segmentParams.data(),
          words);
    } else {
      packKernel<int32_t><<<numBlocks(numRows), kThreads, 0, stream.get()>>>(
          values.data<int32_t>(),
          numRows,
          segmentRows,
          column.segmentParams.data(),
          words);
    }
    checkCuda(cudaGetLastError(), "packKernel");
  }
  stream.sync();
}

/// Decodes the bit-packed stream of 'column' into a new column of 'type'.
std::unique_ptr<cudf::column> unpackColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    cudf::data_type type,
    const rmm::device_uvector<int32_t>& deviceSegments,
    const int32_t* rows,
    int64_t numOut,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto out = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(numOut),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (numOut == 0) {
    return out;
  }
  auto view = out->mutable_view();
  if (cudf::size_of(type) == 8) {
    unpackKernel<int64_t><<<numBlocks(numOut), kThreads, 0, stream.get()>>>(
        column.packed.data(),
        column.segmentParams.data(),
        deviceSegments.data(),
        table.segmentRows,
        rows,
        numOut,
        view.data<int64_t>());
  } else {
    check(cudf::size_of(type) == 4, "Unsupported decode width");
    unpackKernel<int32_t><<<numBlocks(numOut), kThreads, 0, stream.get()>>>(
        column.packed.data(),
        column.segmentParams.data(),
        deviceSegments.data(),
        table.segmentRows,
        rows,
        numOut,
        view.data<int32_t>());
  }
  checkCuda(cudaGetLastError(), "unpackKernel");
  return out;
}

/// Converts a column read from Parquet to the type it is stored and decoded
/// as. Short decimals are kept in 64 bits, as Velox holds them.
std::unique_ptr<cudf::column> normalize(
    std::unique_ptr<cudf::column> column,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (column->type().id() == cudf::type_id::DECIMAL128) {
    return cudf::cast(
        column->view(),
        cudf::data_type{cudf::type_id::DECIMAL64, column->type().scale()},
        stream,
        mr);
  }
  return column;
}

} // namespace

const char* codecName(Codec codec) {
  switch (codec) {
    case Codec::kPlain:
      return "plain";
    case Codec::kForBitpack:
      return "for_bitpack";
    case Codec::kDictionary:
      return "dictionary";
  }
  return "unknown";
}

EncodedColumn::EncodedColumn(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : codec(Codec::kPlain), packed(0, stream, mr), segmentParams(0, stream, mr) {}

int64_t EncodedColumn::encodedBytes() const {
  int64_t bytes = packed.size() * sizeof(uint64_t) +
      segmentParams.size() * sizeof(SegmentParams);
  if (values) {
    bytes += values->alloc_size();
  }
  return bytes;
}

int64_t EncodedColumn::hostBytes() const {
  return hostPacked.size() * sizeof(uint64_t) +
      hostSegmentParams.size() * sizeof(SegmentParams) +
      hostOffsets.size() * sizeof(int64_t) + hostChars.size();
}

const EncodedColumn& EncodedTable::column(const std::string& columnName) const {
  auto it = columnIndex.find(columnName);
  check(
      it != columnIndex.end(),
      "Column " + columnName + " not found in encoded table " + name);
  return *columns[it->second];
}

int32_t EncodedTable::segmentSize(int32_t segment) const {
  return static_cast<int32_t>(std::min<int64_t>(
      segmentRows, numRows - static_cast<int64_t>(segment) * segmentRows));
}

std::unique_ptr<cudf::column> decodeColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  int64_t numOut = 0;
  if (rows.has_value()) {
    numOut = rows->size();
  } else {
    for (auto segment : segments) {
      numOut += table.segmentSize(segment);
    }
  }
  const int32_t* rowData = rows.has_value() ? rows->data<int32_t>() : nullptr;

  switch (column.codec) {
    case Codec::kForBitpack:
      return unpackColumn(
          table,
          column,
          column.type,
          deviceSegments,
          rowData,
          numOut,
          stream,
          mr);
    case Codec::kDictionary: {
      auto codes = unpackColumn(
          table,
          column,
          cudf::data_type{cudf::type_id::INT32},
          deviceSegments,
          rowData,
          numOut,
          stream,
          mr);
      auto gathered = cudf::gather(
          cudf::table_view({column.values->view()}),
          codes->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      return std::move(gathered->release()[0]);
    }
    case Codec::kPlain: {
      auto map = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(numOut),
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      if (numOut > 0) {
        globalRowsKernel<<<numBlocks(numOut), kThreads, 0, stream.get()>>>(
            deviceSegments.data(),
            table.segmentRows,
            rowData,
            numOut,
            map->mutable_view().data<int32_t>());
        checkCuda(cudaGetLastError(), "globalRowsKernel");
      }
      auto gathered = cudf::gather(
          cudf::table_view({column.values->view()}),
          map->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      return std::move(gathered->release()[0]);
    }
  }
  throw std::logic_error("Unknown codec");
}

EncodedColumnCache& EncodedColumnCache::instance() {
  static EncodedColumnCache cache;
  return cache;
}

void EncodedColumnCache::loadTable(
    const std::string& name,
    const std::vector<std::string>& parquetFiles,
    const std::string& clusterKey,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  check(segmentRows > 0, "segmentRows must be positive");
  auto options = cudf::io::parquet_reader_options::builder(
                     cudf::io::source_info(parquetFiles))
                     .build();
  auto read = cudf::io::read_parquet(options, stream, mr);
  std::vector<std::string> names;
  for (const auto& info : read.metadata.schema_info) {
    names.push_back(info.name);
  }
  auto columns = read.tbl->release();
  for (auto& column : columns) {
    check(column->null_count() == 0, "Encoded tables do not support nulls yet");
    column = normalize(std::move(column), stream, mr);
  }
  auto loaded = std::make_unique<cudf::table>(std::move(columns));

  if (!clusterKey.empty()) {
    auto it = std::find(names.begin(), names.end(), clusterKey);
    check(it != names.end(), "Cluster key " + clusterKey + " not in " + name);
    const auto keyIndex = static_cast<cudf::size_type>(it - names.begin());
    loaded = cudf::stable_sort_by_key(
        loaded->view(),
        loaded->view().select({keyIndex}),
        {},
        {},
        stream,
        mr);
  }

  auto table = std::make_unique<EncodedTable>();
  table->name = name;
  table->numRows = loaded->num_rows();
  table->segmentRows = segmentRows;
  table->numSegments =
      static_cast<int32_t>((table->numRows + segmentRows - 1) / segmentRows);
  table->clusterKey = clusterKey;

  columns = loaded->release();
  for (size_t i = 0; i < columns.size(); ++i) {
    auto encoded = std::make_unique<EncodedColumn>(stream, mr);
    encoded->name = names[i];
    encoded->type = columns[i]->type();
    const auto view = columns[i]->view();
    if (isIntegerLike(encoded->type)) {
      encoded->codec = Codec::kForBitpack;
      encodeForBitpack(*encoded, view, segmentRows, stream, mr);
    } else if (encoded->type.id() == cudf::type_id::STRING) {
      const auto distinct = cudf::distinct_count(
          view,
          cudf::null_policy::INCLUDE,
          cudf::nan_policy::NAN_IS_VALID,
          stream);
      if (distinct <= (1 << 16) && int64_t{distinct} * 8 < table->numRows) {
        encoded->codec = Codec::kDictionary;
        auto dictionary = cudf::dictionary::encode(
            view, cudf::data_type{cudf::type_id::INT32}, stream, mr);
        cudf::dictionary_column_view dictionaryView(dictionary->view());
        encoded->values =
            std::make_unique<cudf::column>(dictionaryView.keys(), stream, mr);
        encodeForBitpack(
            *encoded, dictionaryView.indices(), segmentRows, stream, mr);
      } else {
        encoded->codec = Codec::kPlain;
        encoded->values = std::move(columns[i]);
      }
    } else {
      encoded->codec = Codec::kPlain;
      encoded->values = std::move(columns[i]);
    }
    columns[i].reset();
    table->columnIndex.emplace(encoded->name, static_cast<int32_t>(i));
    table->columns.push_back(std::move(encoded));
  }
  stream.sync();
  tables_[name] = std::move(table);
}

namespace {

constexpr cudf::size_type kMaxDictionaryKeys = 1 << 16;

// Reads row groups of 'files' one at a time and calls 'consume' with each,
// after normalize(), and whether it is the last.
template <typename Consume>
void forEachRowGroup(
    const std::vector<std::string>& files,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    Consume consume) {
  std::vector<std::pair<size_t, cudf::size_type>> rowGroups;
  for (size_t f = 0; f < files.size(); ++f) {
    const auto metadata =
        cudf::io::read_parquet_metadata(cudf::io::source_info(files[f]));
    for (cudf::size_type g = 0; g < metadata.num_rowgroups(); ++g) {
      rowGroups.emplace_back(f, g);
    }
  }
  for (size_t i = 0; i < rowGroups.size(); ++i) {
    const auto [file, group] = rowGroups[i];
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info(files[file]))
                       .row_groups({{group}})
                       .build();
    auto read = cudf::io::read_parquet(options, stream, mr);
    auto columns = read.tbl->release();
    for (auto& column : columns) {
      check(
          column->null_count() == 0, "Encoded tables do not support nulls yet");
      column = normalize(std::move(column), stream, mr);
    }
    consume(
        std::make_unique<cudf::table>(std::move(columns)),
        read.metadata,
        i + 1 == rowGroups.size());
  }
}

// Bit-packed segments encoded piece by piece and stored in 'storageMr'.
struct PackedPieces {
  std::vector<rmm::device_uvector<uint64_t>> words;
  std::vector<SegmentParams> params;
  uint64_t numWords{0};
};

void appendPacked(
    EncodedColumn& column,
    PackedPieces& pieces,
    cudf::column_view values,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  EncodedColumn piece(stream, mr);
  encodeForBitpack(piece, values, segmentRows, stream, mr);
  std::vector<SegmentParams> params(piece.segmentParams.size());
  checkCuda(
      cudaMemcpyAsync(
          params.data(),
          piece.segmentParams.data(),
          params.size() * sizeof(SegmentParams),
          cudaMemcpyDeviceToHost,
          stream.get()),
      "copy piece segment params");
  const uint64_t words = piece.packed.size() - 1;
  rmm::device_uvector<uint64_t> stored(words, stream, storageMr);
  checkCuda(
      cudaMemcpyAsync(
          stored.data(),
          piece.packed.data(),
          words * sizeof(uint64_t),
          cudaMemcpyDefault,
          stream.get()),
      "store packed words");
  stream.sync();
  for (auto p : params) {
    p.bitOffset += pieces.numWords * 64;
    pieces.params.push_back(p);
  }
  pieces.words.push_back(std::move(stored));
  pieces.numWords += words;
  column.segmentMin.insert(
      column.segmentMin.end(), piece.segmentMin.begin(), piece.segmentMin.end());
  column.segmentMax.insert(
      column.segmentMax.end(), piece.segmentMax.begin(), piece.segmentMax.end());
  column.segmentBitWidth.insert(
      column.segmentBitWidth.end(),
      piece.segmentBitWidth.begin(),
      piece.segmentBitWidth.end());
}

void finishPacked(
    EncodedColumn& column,
    PackedPieces& pieces,
    bool hostResident,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr) {
  column.packed =
      rmm::device_uvector<uint64_t>(pieces.numWords + 1, stream, storageMr);
  uint64_t offset = 0;
  for (const auto& words : pieces.words) {
    checkCuda(
        cudaMemcpyAsync(
            column.packed.data() + offset,
            words.data(),
            words.size() * sizeof(uint64_t),
            cudaMemcpyDefault,
            stream.get()),
        "concatenate packed words");
    offset += words.size();
  }
  const uint64_t zero = 0;
  checkCuda(
      cudaMemcpyAsync(
          column.packed.data() + pieces.numWords,
          &zero,
          sizeof(uint64_t),
          cudaMemcpyDefault,
          stream.get()),
      "clear padding word");
  column.segmentParams =
      rmm::device_uvector<SegmentParams>(pieces.params.size(), stream, mr);
  checkCuda(
      cudaMemcpyAsync(
          column.segmentParams.data(),
          pieces.params.data(),
          pieces.params.size() * sizeof(SegmentParams),
          cudaMemcpyHostToDevice,
          stream.get()),
      "copy segment params");
  stream.sync();
  if (hostResident) {
    column.hostSegmentParams = std::move(pieces.params);
  }
  pieces = PackedPieces{};
}

// A string column coded piece by piece before its dictionary is known: each
// piece holds codes into its own sorted distinct values, recoded to the
// table's dictionary once every piece is read.
struct DictionaryPieces {
  std::vector<std::unique_ptr<cudf::column>> keys;
  std::vector<std::unique_ptr<cudf::column>> codes;
  std::unique_ptr<cudf::column> allKeys;
};

// The strings of each piece of 'pieces', stored in 'storageMr'.
void decodePieces(
    DictionaryPieces& pieces,
    std::vector<std::unique_ptr<cudf::column>>& plain,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref storageMr) {
  for (size_t p = 0; p < pieces.keys.size(); ++p) {
    auto strings = cudf::gather(
        cudf::table_view({pieces.keys[p]->view()}),
        pieces.codes[p]->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        storageMr);
    plain.push_back(std::move(strings->release()[0]));
  }
  stream.sync();
  pieces = DictionaryPieces{};
}

} // namespace

void EncodedColumnCache::loadTableByRowGroups(
    const std::string& name,
    const std::vector<std::string>& parquetFiles,
    int32_t segmentRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    rmm::device_async_resource_ref storageMr,
    bool hostResident) {
  check(segmentRows > 0, "segmentRows must be positive");

  auto table = std::make_unique<EncodedTable>();
  table->name = name;
  table->segmentRows = segmentRows;
  table->hostResident = hostResident;
  // String columns start as kDictionary and become kPlain once they have more
  // distinct values than a dictionary may hold.
  std::vector<PackedPieces> packed;
  std::vector<DictionaryPieces> dictionaries;
  std::vector<std::vector<std::unique_ptr<cudf::column>>> plain;
  auto toPlain = [&](size_t i) {
    decodePieces(dictionaries[i], plain[i], stream, storageMr);
    table->columns[i]->codec = Codec::kPlain;
  };

  auto encodePiece = [&](cudf::table_view piece) {
    for (size_t i = 0; i < table->columns.size(); ++i) {
      auto& column = *table->columns[i];
      const auto values = piece.column(i);
      if (column.codec == Codec::kDictionary) {
        auto& dictionary = dictionaries[i];
        auto distinct = cudf::distinct(
            cudf::table_view({values}),
            {0},
            cudf::duplicate_keep_option::KEEP_ANY,
            cudf::null_equality::EQUAL,
            cudf::nan_equality::ALL_EQUAL,
            stream,
            mr);
        if (distinct->num_rows() > kMaxDictionaryKeys) {
          toPlain(i);
        } else {
          auto sorted = cudf::sort(distinct->view(), {}, {}, stream, mr);
          auto keys = std::move(sorted->release()[0]);
          auto codes = cudf::lower_bound(
              cudf::table_view({keys->view()}),
              cudf::table_view({values}),
              {cudf::order::ASCENDING},
              {cudf::null_order::AFTER},
              stream,
              storageMr);
          std::vector<cudf::column_view> parts{keys->view()};
          if (dictionary.allKeys) {
            parts.push_back(dictionary.allKeys->view());
          }
          auto all = cudf::concatenate(parts, stream, mr);
          auto allDistinct = cudf::distinct(
              cudf::table_view({all->view()}),
              {0},
              cudf::duplicate_keep_option::KEEP_ANY,
              cudf::null_equality::EQUAL,
              cudf::nan_equality::ALL_EQUAL,
              stream,
              mr);
          dictionary.allKeys = std::move(allDistinct->release()[0]);
          dictionary.keys.push_back(std::move(keys));
          dictionary.codes.push_back(std::move(codes));
          if (dictionary.allKeys->size() > kMaxDictionaryKeys) {
            toPlain(i);
          }
          continue;
        }
      }
      if (column.codec == Codec::kForBitpack) {
        appendPacked(
            column, packed[i], values, segmentRows, stream, mr, storageMr);
      } else {
        std::vector<cudf::column_view> parts{values};
        plain[i].push_back(cudf::concatenate(parts, stream, storageMr));
      }
    }
    stream.sync();
  };

  // Encode whole segments as they fill, carrying the rest of each row group
  // into the next.
  std::unique_ptr<cudf::table> carry;
  forEachRowGroup(
      parquetFiles,
      stream,
      mr,
      [&](std::unique_ptr<cudf::table> rows, const auto& metadata, bool last) {
        if (table->columns.empty()) {
          for (size_t i = 0; i < metadata.schema_info.size(); ++i) {
            auto encoded = std::make_unique<EncodedColumn>(stream, mr);
            encoded->name = metadata.schema_info[i].name;
            encoded->type = rows->view().column(i).type();
            encoded->codec = isIntegerLike(encoded->type) ? Codec::kForBitpack
                : encoded->type.id() == cudf::type_id::STRING
                ? Codec::kDictionary
                : Codec::kPlain;
            table->columnIndex.emplace(
                encoded->name, static_cast<int32_t>(i));
            table->columns.push_back(std::move(encoded));
          }
          packed.resize(table->columns.size());
          dictionaries.resize(table->columns.size());
          plain.resize(table->columns.size());
        }
        table->numRows += rows->num_rows();
        if (carry) {
          std::vector<cudf::table_view> parts{carry->view(), rows->view()};
          rows = cudf::concatenate(parts, stream, mr);
          carry.reset();
        }
        const auto total = rows->num_rows();
        const auto whole = last ? total : total / segmentRows * segmentRows;
        if (whole > 0) {
          encodePiece(cudf::slice(rows->view(), {0, whole}, stream)[0]);
        }
        if (whole < total) {
          std::vector<cudf::table_view> rest{
              cudf::slice(rows->view(), {whole, total}, stream)[0]};
          carry = cudf::concatenate(rest, stream, mr);
        }
      });
  check(!table->columns.empty(), "No row groups in " + name);
  table->numSegments = static_cast<int32_t>(
      (table->numRows + segmentRows - 1) / segmentRows);

  for (size_t i = 0; i < table->columns.size(); ++i) {
    auto& column = *table->columns[i];
    if (column.codec == Codec::kDictionary &&
        int64_t{dictionaries[i].allKeys->size()} * 8 >= table->numRows) {
      toPlain(i);
    }
    if (column.codec == Codec::kDictionary) {
      auto& dictionary = dictionaries[i];
      auto sorted = cudf::stable_sort(
          cudf::table_view({dictionary.allKeys->view()}), {}, {}, stream, mr);
      column.values = std::move(sorted->release()[0]);
      for (size_t p = 0; p < dictionary.keys.size(); ++p) {
        auto recode = cudf::lower_bound(
            cudf::table_view({column.values->view()}),
            cudf::table_view({dictionary.keys[p]->view()}),
            {cudf::order::ASCENDING},
            {cudf::null_order::AFTER},
            stream,
            mr);
        auto codes = cudf::gather(
            cudf::table_view({recode->view()}),
            dictionary.codes[p]->view(),
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        appendPacked(
            column,
            packed[i],
            codes->view().column(0),
            segmentRows,
            stream,
            mr,
            storageMr);
        dictionary.keys[p].reset();
        dictionary.codes[p].reset();
      }
      dictionaries[i] = DictionaryPieces{};
    }
    if (column.codec == Codec::kPlain) {
      if (plain[i].size() == 1) {
        column.values = std::move(plain[i][0]);
      } else {
        std::vector<cudf::column_view> parts;
        for (const auto& part : plain[i]) {
          parts.push_back(part->view());
        }
        column.values = cudf::concatenate(parts, stream, storageMr);
      }
      plain[i].clear();
    } else {
      finishPacked(column, packed[i], hostResident, stream, mr, storageMr);
    }
    stream.sync();
  }
  tables_[name] = std::move(table);
}

const EncodedTable* EncodedColumnCache::findTable(const std::string& name) const {
  auto it = tables_.find(name);
  return it == tables_.end() ? nullptr : it->second.get();
}

const EncodedTable& EncodedColumnCache::table(const std::string& name) const {
  const auto* table = findTable(name);
  check(table != nullptr, "Encoded table " + name + " is not loaded");
  return *table;
}

void EncodedColumnCache::copyToHost(cuda::stream_ref stream, bool releaseDevice) {
  auto mr = rmm::mr::get_current_device_resource_ref();
  for (auto& [_, table] : tables_) {
    for (auto& column : table->columns) {
      column->hostPacked.resize(column->packed.size());
      checkCuda(
          cudaMemcpyAsync(
              column->hostPacked.data(),
              column->packed.data(),
              column->packed.size() * sizeof(uint64_t),
              cudaMemcpyDefault,
              stream.get()),
          "copy packed words to host");
      column->hostSegmentParams.resize(column->segmentParams.size());
      checkCuda(
          cudaMemcpyAsync(
              column->hostSegmentParams.data(),
              column->segmentParams.data(),
              column->segmentParams.size() * sizeof(SegmentParams),
              cudaMemcpyDefault,
              stream.get()),
          "copy segment params to host");
      if (column->values) {
        check(
            column->values->type().id() == cudf::type_id::STRING,
            "Host copies support only string value columns");
        cudf::strings_column_view strings(column->values->view());
        const auto numOffsets = strings.size() + 1;
        const auto offsets = strings.offsets();
        column->hostOffsets.resize(numOffsets);
        if (offsets.type().id() == cudf::type_id::INT64) {
          checkCuda(
              cudaMemcpyAsync(
                  column->hostOffsets.data(),
                  offsets.data<int64_t>(),
                  numOffsets * sizeof(int64_t),
                  cudaMemcpyDefault,
                  stream.get()),
              "copy string offsets to host");
        } else {
          std::vector<int32_t> narrow(numOffsets);
          checkCuda(
              cudaMemcpyAsync(
                  narrow.data(),
                  offsets.data<int32_t>(),
                  numOffsets * sizeof(int32_t),
                  cudaMemcpyDefault,
                  stream.get()),
              "copy string offsets to host");
          stream.sync();
          std::copy(narrow.begin(), narrow.end(), column->hostOffsets.begin());
        }
        column->hostChars.resize(strings.chars_size(stream));
        checkCuda(
            cudaMemcpyAsync(
                column->hostChars.data(),
                strings.chars_begin(stream),
                column->hostChars.size(),
                cudaMemcpyDefault,
                stream.get()),
            "copy string characters to host");
      }
      stream.sync();
      if (releaseDevice) {
        column->packed = rmm::device_uvector<uint64_t>(0, stream, mr);
        column->segmentParams =
            rmm::device_uvector<SegmentParams>(0, stream, mr);
        column->values.reset();
      }
    }
  }
  stream.sync();
}

void EncodedColumnCache::moveTo(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  for (auto& [_, table] : tables_) {
    for (auto& column : table->columns) {
      column->hostSegmentParams.resize(column->segmentParams.size());
      checkCuda(
          cudaMemcpyAsync(
              column->hostSegmentParams.data(),
              column->segmentParams.data(),
              column->segmentParams.size() * sizeof(SegmentParams),
              cudaMemcpyDeviceToHost,
              stream.get()),
          "copy segment params to host");
      column->packed = rmm::device_uvector<uint64_t>(column->packed, stream, mr);
      if (column->values && column->codec == Codec::kPlain) {
        column->values =
            std::make_unique<cudf::column>(column->values->view(), stream, mr);
      }
      stream.sync();
    }
    table->hostResident = true;
  }
}

namespace {

template <typename T>
__global__ void addToOffsetsKernel(T* offsets, int64_t n, T delta) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    offsets[i] += delta;
  }
}

// Ascending runs of adjacent segments, as [first, last] segment ids.
std::vector<std::pair<int32_t, int32_t>> segmentRuns(
    std::span<const int32_t> segments) {
  std::vector<std::pair<int32_t, int32_t>> runs;
  for (auto segment : segments) {
    if (!runs.empty() && runs.back().second + 1 == segment) {
      runs.back().second = segment;
    } else {
      runs.emplace_back(segment, segment);
    }
  }
  return runs;
}

void copyToDevice(void* dst, const void* src, size_t bytes, cuda::stream_ref stream) {
  if (bytes > 0) {
    checkCuda(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream.get()),
        "stage to device");
  }
}

// Work queued after every copy from pageable memory. A copy from pageable
// memory waits for the work already on the stream, so the bulk copies from
// pinned memory are queued last to keep them asynchronous.
using Deferred = std::vector<std::function<void()>>;

// Stages the packed words and params of 'segments' of a bit-packed stream.
void stagePacked(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    EncodedColumn& staged,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto& params = column.hostSegmentParams;
  check(
      params.size() == static_cast<size_t>(table.numSegments),
      "Host segment params missing for " + column.name);
  auto wordsOf = [&](int32_t segment) {
    return (static_cast<uint64_t>(table.segmentSize(segment)) *
                params[segment].bitWidth +
            63) /
        64;
  };
  std::vector<SegmentParams> stagedParams;
  uint64_t numWords = 0;
  for (auto segment : segments) {
    auto p = params[segment];
    p.bitOffset = numWords * 64;
    stagedParams.push_back(p);
    numWords += wordsOf(segment);
  }
  staged.segmentParams =
      rmm::device_uvector<SegmentParams>(stagedParams.size(), stream, mr);
  copyToDevice(
      staged.segmentParams.data(),
      stagedParams.data(),
      stagedParams.size() * sizeof(SegmentParams),
      stream);
  staged.packed = rmm::device_uvector<uint64_t>(numWords + 1, stream, mr);
  checkCuda(
      cudaMemsetAsync(
          staged.packed.data() + numWords, 0, sizeof(uint64_t), stream.get()),
      "clear padding word");
  uint64_t word = 0;
  for (const auto& [first, last] : segmentRuns(segments)) {
    uint64_t runWords = 0;
    for (int32_t s = first; s <= last; ++s) {
      runWords += wordsOf(s);
    }
    deferred.push_back([dst = staged.packed.data() + word,
                        src = column.packed.data() + params[first].bitOffset / 64,
                        bytes = runWords * sizeof(uint64_t),
                        stream] { copyToDevice(dst, src, bytes, stream); });
    word += runWords;
  }
}

// Each run copies one offset past its end, which the next run overwrites, so
// the last offset of the staged column is its character count.
template <typename OffsetT>
std::unique_ptr<cudf::column> stageStrings(
    const EncodedTable& table,
    cudf::strings_column_view strings,
    std::span<const int32_t> segments,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto* offsets = strings.offsets().data<OffsetT>();
  const char* chars = strings.parent().data<char>();
  std::vector<std::pair<int64_t, int64_t>> rowRuns;
  int64_t numRows = 0;
  int64_t numChars = 0;
  for (const auto& [first, last] : segmentRuns(segments)) {
    const int64_t begin = int64_t{first} * table.segmentRows;
    const int64_t end = int64_t{last} * table.segmentRows + table.segmentSize(last);
    rowRuns.emplace_back(begin, end);
    numRows += end - begin;
    numChars += offsets[end] - offsets[begin];
  }
  auto stagedOffsets = cudf::make_fixed_width_column(
      strings.offsets().type(),
      static_cast<cudf::size_type>(numRows + 1),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* outOffsets = stagedOffsets->mutable_view().data<OffsetT>();
  rmm::device_buffer stagedChars(numChars, stream, mr);
  auto* outChars = static_cast<char*>(stagedChars.data());
  int64_t row = 0;
  int64_t charBase = 0;
  for (const auto& [begin, end] : rowRuns) {
    const int64_t runRows = end - begin;
    const int64_t runChars = offsets[end] - offsets[begin];
    const auto delta = static_cast<OffsetT>(charBase - offsets[begin]);
    deferred.push_back([=] {
      copyToDevice(outChars + charBase, chars + offsets[begin], runChars, stream);
      copyToDevice(
          outOffsets + row,
          offsets + begin,
          (runRows + 1) * sizeof(OffsetT),
          stream);
      addToOffsetsKernel<OffsetT>
          <<<numBlocks(runRows + 1), kThreads, 0, stream.get()>>>(
              outOffsets + row, runRows + 1, delta);
      checkCuda(cudaGetLastError(), "addToOffsetsKernel");
    });
    row += runRows;
    charBase += runChars;
  }
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(numRows),
      std::move(stagedOffsets),
      std::move(stagedChars),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
}

std::unique_ptr<cudf::column> stageFixedWidth(
    const EncodedTable& table,
    cudf::column_view values,
    std::span<const int32_t> segments,
    Deferred& deferred,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto width = cudf::size_of(values.type());
  int64_t numRows = 0;
  for (auto segment : segments) {
    numRows += table.segmentSize(segment);
  }
  auto staged = cudf::make_fixed_width_column(
      values.type(),
      static_cast<cudf::size_type>(numRows),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* out = static_cast<char*>(staged->mutable_view().head());
  const auto* in = static_cast<const char*>(values.head()) + values.offset() * width;
  int64_t row = 0;
  for (const auto& [first, last] : segmentRuns(segments)) {
    const int64_t begin = int64_t{first} * table.segmentRows;
    const int64_t end = int64_t{last} * table.segmentRows + table.segmentSize(last);
    deferred.push_back([dst = out + row * width,
                        src = in + begin * width,
                        bytes = (end - begin) * width,
                        stream] { copyToDevice(dst, src, bytes, stream); });
    row += end - begin;
  }
  return staged;
}

} // namespace

std::unique_ptr<EncodedTable> stageSegments(
    const EncodedTable& table,
    const std::vector<std::string>& columnNames,
    std::span<const int32_t> segments,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  check(table.hostResident, "stageSegments needs a host-resident table");
  check(!segments.empty(), "stageSegments needs segments");
  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    check(
        segments[i] < segments[i + 1] &&
            table.segmentSize(segments[i]) == table.segmentRows,
        "Staged segments must be ascending with only the last one short");
  }
  auto staged = std::make_unique<EncodedTable>();
  staged->name = table.name;
  staged->segmentRows = table.segmentRows;
  staged->numSegments = static_cast<int32_t>(segments.size());
  staged->numRows = int64_t{table.segmentRows} * (staged->numSegments - 1) +
      table.segmentSize(segments.back());
  staged->clusterKey = table.clusterKey;
  Deferred deferred;
  for (const auto& name : columnNames) {
    if (staged->columnIndex.count(name)) {
      continue;
    }
    const auto& column = table.column(name);
    auto out = std::make_unique<EncodedColumn>(stream, mr);
    out->name = column.name;
    out->type = column.type;
    out->codec = column.codec;
    switch (column.codec) {
      case Codec::kForBitpack:
        stagePacked(table, column, segments, *out, deferred, stream, mr);
        break;
      case Codec::kDictionary:
        stagePacked(table, column, segments, *out, deferred, stream, mr);
        out->values = std::make_unique<cudf::column>(
            column.values->view(), stream, mr);
        break;
      case Codec::kPlain:
        if (column.values->type().id() == cudf::type_id::STRING) {
          cudf::strings_column_view strings(column.values->view());
          out->values = strings.offsets().type().id() == cudf::type_id::INT64
              ? stageStrings<int64_t>(
                    table, strings, segments, deferred, stream, mr)
              : stageStrings<int32_t>(
                    table, strings, segments, deferred, stream, mr);
        } else {
          out->values = stageFixedWidth(
              table, column.values->view(), segments, deferred, stream, mr);
        }
        break;
    }
    staged->columnIndex.emplace(
        name, static_cast<int32_t>(staged->columns.size()));
    staged->columns.push_back(std::move(out));
  }
  for (auto& work : deferred) {
    work();
  }
  return staged;
}

std::string EncodedColumnCache::sizeReport() const {
  std::vector<std::string> tableNames;
  for (const auto& [name, _] : tables_) {
    tableNames.push_back(name);
  }
  std::sort(tableNames.begin(), tableNames.end());
  std::string report = "table,column,codec,rows,segments,encoded_bytes,"
                       "host_bytes,max_bit_width\n";
  for (const auto& tableName : tableNames) {
    const auto& table = *tables_.at(tableName);
    for (const auto& column : table.columns) {
      uint32_t maxWidth = 0;
      for (auto width : column->segmentBitWidth) {
        maxWidth = std::max(maxWidth, width);
      }
      std::ostringstream line;
      line << tableName << ',' << column->name << ','
           << codecName(column->codec) << ',' << table.numRows << ','
           << table.numSegments << ',' << column->encodedBytes() << ','
           << column->hostBytes() << ',' << maxWidth << '\n';
      report += line.str();
    }
  }
  return report;
}

void EncodedColumnCache::clear() {
  tables_.clear();
}

} // namespace facebook::velox::cudf_velox::connector::encoded
