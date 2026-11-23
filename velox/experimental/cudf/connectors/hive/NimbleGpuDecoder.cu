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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/NimbleGpuDecoder.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/buffer>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <glog/logging.h>

#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// Numeric values of facebook::nimble::EncodingType, DataType and
// CompressionType (velox/dwio/nimble/common/Types.h). Only the values the
// decoder handles are listed.
enum class EncodingType : uint8_t {
  kTrivial = 0,
  kRle = 1,
  kDictionary = 2,
  kFixedBitWidth = 3,
  kNullable = 5,
  kSparseBool = 6,
  kVarint = 7,
  kConstant = 9,
  kMainlyConstant = 10,
  kAlp = 12,
  kFsst = 19,
  kDeltaBlock = 21,
};

enum class DataType : uint8_t {
  kInt8 = 1,
  kUint8 = 2,
  kInt16 = 3,
  kUint16 = 4,
  kInt32 = 5,
  kUint32 = 6,
  kInt64 = 7,
  kUint64 = 8,
  kFloat = 9,
  kDouble = 10,
  kBool = 11,
  kString = 12,
};

enum class Codec : uint8_t {
  kUncompressed = 0,
  kZstd = 1,
  kLz4 = 3,
};

constexpr int kBlockSize = 256;

#define NIMBLE_GPU_CHECK(cond, msg) CUDF_EXPECTS((cond), std::string("Nimble GPU decode: ") + (msg))

void checkCuda(cudaError_t error) {
  CUDF_EXPECTS(error == cudaSuccess, cudaGetErrorString(error));
}

// Bytes per value of a fixed-width Nimble data type, or 0 for Bool and String.
int physicalWidth(DataType type) {
  switch (type) {
    case DataType::kInt8:
    case DataType::kUint8:
      return 1;
    case DataType::kInt16:
    case DataType::kUint16:
      return 2;
    case DataType::kInt32:
    case DataType::kUint32:
    case DataType::kFloat:
      return 4;
    case DataType::kInt64:
    case DataType::kUint64:
    case DataType::kDouble:
      return 8;
    case DataType::kBool:
    case DataType::kString:
      return 0;
  }
  NIMBLE_GPU_CHECK(false, "unknown data type " + std::to_string(static_cast<int>(type)));
  return 0;
}

// Reads little-endian fields from host bytes.
struct ByteReader {
  const uint8_t* pos;
  const uint8_t* end;

  explicit ByteReader(std::span<const uint8_t> bytes)
      : pos{bytes.data()}, end{bytes.data() + bytes.size()} {}

  size_t remaining() const {
    return static_cast<size_t>(end - pos);
  }

  void need(size_t n) const {
    NIMBLE_GPU_CHECK(n <= remaining(), "truncated encoding");
  }

  uint8_t u8() {
    need(1);
    return *pos++;
  }

  uint32_t u32() {
    return static_cast<uint32_t>(uN(4));
  }

  uint64_t uN(int width) {
    need(width);
    uint64_t value = 0;
    std::memcpy(&value, pos, width);
    pos += width;
    return value;
  }

  uint32_t varint() {
    uint32_t value = 0;
    for (int shift = 0; shift < 35; shift += 7) {
      const uint8_t byte = u8();
      value |= static_cast<uint32_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      }
    }
    NIMBLE_GPU_CHECK(false, "varint too long");
    return 0;
  }

  uint64_t varint64() {
    uint64_t value = 0;
    for (int shift = 0; shift < 70; shift += 7) {
      const uint8_t byte = u8();
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      }
    }
    NIMBLE_GPU_CHECK(false, "varint too long");
    return 0;
  }

  std::span<const uint8_t> take(size_t n) {
    need(n);
    std::span<const uint8_t> bytes{pos, n};
    pos += n;
    return bytes;
  }

  std::span<const uint8_t> rest() {
    return take(remaining());
  }

  std::string string() {
    const auto length = u32();
    auto bytes = take(length);
    return std::string(reinterpret_cast<const char*>(bytes.data()), length);
  }
};

template <typename T>
__device__ T loadUnaligned(const uint8_t* p) {
  T value;
  memcpy(&value, p, sizeof(T));
  return value;
}

template <typename U>
__global__ void unpackBitsKernel(
    const uint8_t* packed,
    int bitWidth,
    U baseline,
    U* out,
    int64_t numValues) {
  const uint64_t mask =
      bitWidth == 64 ? ~uint64_t{0} : (uint64_t{1} << bitWidth) - 1;
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    const uint64_t bit = static_cast<uint64_t>(i) * bitWidth;
    const uint8_t* p = packed + (bit >> 3);
    const int shift = bit & 7;
    uint64_t word = loadUnaligned<uint64_t>(p) >> shift;
    if (shift + bitWidth > 64) {
      word |= static_cast<uint64_t>(p[8]) << (64 - shift);
    }
    out[i] = static_cast<U>(static_cast<U>(word & mask) + baseline);
  }
}

__global__ void unpackBoolsKernel(
    const uint8_t* bitmap,
    bool* out,
    int64_t numValues) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = (bitmap[i >> 3] >> (i & 7)) & 1;
  }
}

template <typename U>
__global__ void fillKernel(U* out, U value, int64_t numValues) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = value;
  }
}

__global__ void invertBoolsKernel(const bool* in, bool* out, int64_t numValues) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = !in[i];
  }
}

// Run r of a bool RLE holds `first` when r is even and its negation otherwise.
__global__ void alternateBoolsKernel(bool first, bool* out, int64_t numValues) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = first != static_cast<bool>(i & 1);
  }
}

__global__ void scatterBoolKernel(
    const int32_t* indices,
    bool value,
    bool* out,
    int64_t numIndices) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numIndices;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[indices[i]] = value;
  }
}

// 1 for the last byte of each LEB128 varint.
__global__ void varintEndsKernel(
    const uint8_t* bytes,
    int32_t* ends,
    int64_t numBytes) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numBytes;
       i += int64_t{gridDim.x} * blockDim.x) {
    ends[i] = (bytes[i] & 0x80) == 0;
  }
}

// Decodes each varint at the thread holding its last byte; `index` is the
// exclusive scan of the end flags, so it numbers the values in order.
template <typename U>
__global__ void varintDecodeKernel(
    const uint8_t* bytes,
    const int32_t* index,
    U baseline,
    U* out,
    int64_t numBytes) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numBytes;
       i += int64_t{gridDim.x} * blockDim.x) {
    if (bytes[i] & 0x80) {
      continue;
    }
    int64_t start = i;
    while (start > 0 && (bytes[start - 1] & 0x80)) {
      --start;
    }
    uint64_t value = 0;
    for (int64_t j = start, shift = 0; j <= i; ++j, shift += 7) {
      value |= static_cast<uint64_t>(bytes[j] & 0x7f) << shift;
    }
    out[index[i]] = static_cast<U>(static_cast<U>(value) + baseline);
  }
}

// Powers of ten ALP scales by (kPow10Double in ALPEncoding.h).
constexpr double kAlpPow10[24] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23};

// value = zigzag^-1(encoded) * 10^factor / 10^exponent, rounded as the CPU
// decoder rounds, then the exceptions are patched in.
template <typename F>
__global__ void alpKernel(
    const uint64_t* encoded,
    double factor,
    double exponent,
    F* out,
    int64_t numValues) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    const uint64_t zigzag = encoded[i];
    const auto value =
        static_cast<int64_t>(zigzag >> 1) ^ -static_cast<int64_t>(zigzag & 1);
    out[i] = static_cast<F>(
        __ddiv_rn(__dmul_rn(static_cast<double>(value), factor), exponent));
  }
}

template <typename F>
__global__ void scatterKernel(
    const int32_t* positions,
    const F* values,
    F* out,
    int64_t numValues) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numValues;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[positions[i]] = values[i];
  }
}

// One block of a DeltaBlock encoding: its first value, in order-preserving
// form, and the bit-packed deltas from each row to the next.
struct DeltaBlock {
  uint64_t base;
  uint32_t offset;
  uint8_t bitWidth;
};

// Decodes block b of a DeltaBlock encoding in thread block b. `signBit` undoes
// the order-preserving form of signed types.
template <typename U>
__global__ void deltaBlockKernel(
    const uint8_t* packed,
    const DeltaBlock* blocks,
    uint32_t blockSize,
    uint64_t signBit,
    U* out,
    uint32_t numRows) {
  using Scan = cub::BlockScan<uint64_t, kBlockSize>;
  __shared__ typename Scan::TempStorage temp;
  const DeltaBlock block = blocks[blockIdx.x];
  const uint32_t start = blockIdx.x * blockSize;
  const uint32_t end = min(start + blockSize, numRows);
  const uint8_t* data = packed + block.offset;
  const uint64_t mask = block.bitWidth == 64
      ? ~uint64_t{0}
      : (uint64_t{1} << block.bitWidth) - 1;
  uint64_t carry = block.base;
  for (uint32_t first = start; first < end; first += kBlockSize) {
    const uint32_t row = first + threadIdx.x;
    uint64_t delta = 0;
    if (row > start && row < end && block.bitWidth > 0) {
      const uint64_t bit = static_cast<uint64_t>(row - start - 1) * block.bitWidth;
      const uint8_t* p = data + (bit >> 3);
      const int shift = bit & 7;
      uint64_t word = loadUnaligned<uint64_t>(p) >> shift;
      if (shift + block.bitWidth > 64) {
        word |= static_cast<uint64_t>(p[8]) << (64 - shift);
      }
      delta = word & mask;
    }
    uint64_t sum;
    uint64_t total;
    Scan(temp).InclusiveSum(delta, sum, total);
    if (row < end) {
      out[row] = static_cast<U>((carry + sum) ^ signBit);
    }
    carry += total;
    __syncthreads();
  }
}

// An FSST symbol table, laid out as fsst_import (velox/external/fsst) builds
// it: code c expands to the low len[c] bytes of symbol[c], and the escape code
// stands for the literal byte that follows it.
struct FsstTable {
  uint64_t symbol[255];
  uint8_t len[255];
};

constexpr uint8_t kFsstEscape = 255;

// Parses the serialized table: an 8-byte version, a zero-terminated flag, the
// count of symbols of each length (1-byte symbols counted last) and then the
// symbol bytes.
FsstTable fsstTable(std::span<const uint8_t> bytes) {
  ByteReader reader{bytes};
  reader.take(8);
  const bool zeroTerminated = reader.u8() & 1;
  uint8_t counts[8];
  for (auto& count : counts) {
    count = reader.u8();
  }
  FsstTable table{};
  table.len[0] = 1;
  uint32_t code = zeroTerminated ? 1 : 0;
  if (zeroTerminated) {
    NIMBLE_GPU_CHECK(counts[0] > 0, "FSST table has no terminator symbol");
    --counts[0];
  }
  for (uint32_t l = 1; l <= 8; ++l) {
    const uint32_t length = (l & 7) + 1;
    for (uint32_t i = 0; i < counts[l & 7]; ++i, ++code) {
      NIMBLE_GPU_CHECK(code < 255, "FSST table has too many symbols");
      table.len[code] = length;
      std::memcpy(&table.symbol[code], reader.take(length).data(), length);
    }
  }
  for (; code < 255; ++code) {
    table.len[code] = 8;
  }
  return table;
}

__device__ void loadFsstTable(const FsstTable& table, FsstTable& shared) {
  for (int c = threadIdx.x; c < 255; c += blockDim.x) {
    shared.symbol[c] = table.symbol[c];
    shared.len[c] = table.len[c];
  }
  __syncthreads();
}

// The decoded length of each string; offsets are where each string's codes
// start in the blob.
__global__ void fsstLengthsKernel(
    const FsstTable table,
    const uint8_t* blob,
    const int32_t* offsets,
    const int32_t* compressedLengths,
    int32_t* lengths,
    int64_t numRows) {
  __shared__ FsstTable shared;
  loadFsstTable(table, shared);
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numRows;
       i += int64_t{gridDim.x} * blockDim.x) {
    const uint8_t* p = blob + offsets[i];
    const uint8_t* end = p + compressedLengths[i];
    int32_t length = 0;
    while (p < end) {
      const uint8_t code = *p++;
      if (code == kFsstEscape) {
        ++p;
        ++length;
      } else {
        length += shared.len[code];
      }
    }
    lengths[i] = length;
  }
}

__global__ void fsstDecodeKernel(
    const FsstTable table,
    const uint8_t* blob,
    const int32_t* offsets,
    const int32_t* compressedLengths,
    const int32_t* charOffsets,
    char* chars,
    int64_t numRows) {
  __shared__ FsstTable shared;
  loadFsstTable(table, shared);
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
       i < numRows;
       i += int64_t{gridDim.x} * blockDim.x) {
    const uint8_t* p = blob + offsets[i];
    const uint8_t* end = p + compressedLengths[i];
    char* out = chars + charOffsets[i];
    while (p < end) {
      const uint8_t code = *p++;
      if (code == kFsstEscape) {
        *out++ = static_cast<char>(*p++);
        continue;
      }
      const uint64_t symbol = shared.symbol[code];
      const int length = shared.len[code];
      for (int k = 0; k < length; ++k) {
        out[k] = static_cast<char>(symbol >> (8 * k));
      }
      out += length;
    }
  }
}

// A leaf encoding that gives O(1) access to value i.
// Varint and DeltaBlock leaves are decoded into 8-byte trivial leaves before
// the chunks that read them.
struct FastLeaf {
  enum Kind : uint8_t {
    kTrivial,
    kFixedBitWidth,
    kConstant,
    kIdentity,
    kVarint,
    kDeltaBlock
  };
  const uint8_t* data{nullptr};
  // FixedBitWidth or Varint baseline, the Constant value, or where a
  // DeltaBlock leaf's values start in the batch's decoded DeltaBlock values.
  uint64_t baseline{0};
  // Values of a Varint or DeltaBlock leaf, and payload bytes of a Varint leaf.
  uint32_t count{0};
  uint32_t numBytes{0};
  Kind kind{kTrivial};
  uint8_t width{0};
  uint8_t bitWidth{0};
};

// The payload of a Varint leaf, at byteBase in the batch's varint bytes.
struct VarintSegment {
  const uint8_t* data;
  uint64_t baseline;
  uint32_t numBytes;
  uint32_t byteBase;
  uint8_t width;
};

constexpr int kVarintBytesPerBlock = 4096;

__device__ int findSegment(const uint32_t* blockStarts, int numSegments) {
  int lo = 0;
  int hi = numSegments - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (blockStarts[mid] <= blockIdx.x) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

__global__ void varintEndFlagsKernel(
    const VarintSegment* segments,
    const uint32_t* blockStarts,
    int numSegments,
    uint32_t* flags) {
  const int s = findSegment(blockStarts, numSegments);
  const VarintSegment segment = segments[s];
  const uint32_t begin = (blockIdx.x - blockStarts[s]) * kVarintBytesPerBlock;
  const uint32_t end = min(begin + kVarintBytesPerBlock, segment.numBytes);
  for (uint32_t j = begin + threadIdx.x; j < end; j += blockDim.x) {
    flags[segment.byteBase + j] = (segment.data[j] & 0x80) == 0;
  }
}

// `index` is the exclusive scan of the end flags, so it numbers the values of
// all segments in order.
__global__ void varintValuesKernel(
    const VarintSegment* segments,
    const uint32_t* blockStarts,
    int numSegments,
    const uint32_t* index,
    uint64_t* out,
    uint32_t numValues) {
  const int s = findSegment(blockStarts, numSegments);
  const VarintSegment segment = segments[s];
  const uint32_t begin = (blockIdx.x - blockStarts[s]) * kVarintBytesPerBlock;
  const uint32_t end = min(begin + kVarintBytesPerBlock, segment.numBytes);
  for (uint32_t j = begin + threadIdx.x; j < end; j += blockDim.x) {
    if (segment.data[j] & 0x80) {
      continue;
    }
    uint32_t start = j;
    while (start > 0 && j - start < 9 && (segment.data[start - 1] & 0x80)) {
      --start;
    }
    uint64_t value = 0;
    for (uint32_t k = start, shift = 0; k <= j; ++k, shift += 7) {
      value |= static_cast<uint64_t>(segment.data[k] & 0x7f) << shift;
    }
    const uint32_t v = index[segment.byteBase + j];
    if (v < numValues) {
      value += segment.baseline;
      out[v] = segment.width == 4 ? static_cast<uint32_t>(value) : value;
    }
  }
}

// One chunk decoded by decodeFastChunksKernel. Row i maps to a run when the
// chunk is run-length encoded, then through dictionary indices, then to a
// value leaf:
//   leaf, Dictionary(leaf, leaf), RLE(leaf), RLE(Dictionary(leaf, leaf)) and
//   Dictionary(leaf, RLE(leaf)).
struct FastChunk {
  FastLeaf values;
  FastLeaf indices;
  // Inclusive run ends, offset by rowBase modulo 2^32.
  const uint32_t* runEnds{nullptr};
  uint8_t* out{nullptr};
  uint64_t add{0};
  uint32_t numRows{0};
  uint32_t numRuns{0};
  uint32_t rowBase{0};
  uint8_t outWidth{0};
  bool dictionary{false};
  bool rle{false};
};

constexpr int kFastRowsPerBlock = 4096;

__device__ uint64_t readLeaf(const FastLeaf& leaf, uint64_t i) {
  switch (leaf.kind) {
    case FastLeaf::kTrivial:
      // Uncompressed payloads sit at arbitrary offsets in the file bytes.
      switch (leaf.width) {
        case 1:
          return leaf.data[i];
        case 2:
          return loadUnaligned<uint16_t>(leaf.data + 2 * i);
        case 4:
          return loadUnaligned<uint32_t>(leaf.data + 4 * i);
        default:
          return loadUnaligned<uint64_t>(leaf.data + 8 * i);
      }
    case FastLeaf::kFixedBitWidth: {
      const uint64_t bit = i * leaf.bitWidth;
      const uint8_t* p = leaf.data + (bit >> 3);
      const int shift = bit & 7;
      uint64_t word = loadUnaligned<uint64_t>(p) >> shift;
      if (shift + leaf.bitWidth > 64) {
        word |= static_cast<uint64_t>(p[8]) << (64 - shift);
      }
      const uint64_t mask = leaf.bitWidth == 64
          ? ~uint64_t{0}
          : (uint64_t{1} << leaf.bitWidth) - 1;
      return (word & mask) + leaf.baseline;
    }
    case FastLeaf::kIdentity:
      return i;
    default:
      return leaf.baseline;
  }
}

// Each block decodes up to kFastRowsPerBlock rows of one chunk; blockStarts[c]
// is the first block of chunk c, with a final entry for the total.
__global__ void decodeFastChunksKernel(
    const FastChunk* chunks,
    const uint32_t* blockStarts,
    int numChunks) {
  int lo = 0;
  int hi = numChunks - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (blockStarts[mid] <= blockIdx.x) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const FastChunk chunk = chunks[lo];
  const uint64_t begin =
      static_cast<uint64_t>(blockIdx.x - blockStarts[lo]) * kFastRowsPerBlock;
  const uint64_t end =
      min(begin + kFastRowsPerBlock, static_cast<uint64_t>(chunk.numRows));
  for (uint64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    uint64_t position = i;
    if (chunk.rle) {
      uint32_t lo = 0;
      uint32_t hi = chunk.numRuns - 1;
      while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (chunk.runEnds[mid] - chunk.rowBase > i) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      position = lo;
    }
    if (chunk.dictionary) {
      position = readLeaf(chunk.indices, position);
    }
    const uint64_t value = readLeaf(chunk.values, position) + chunk.add;
    switch (chunk.outWidth) {
      case 1:
        chunk.out[i] = static_cast<uint8_t>(value);
        break;
      case 2:
        reinterpret_cast<uint16_t*>(chunk.out)[i] = static_cast<uint16_t>(value);
        break;
      case 4:
        reinterpret_cast<uint32_t*>(chunk.out)[i] = static_cast<uint32_t>(value);
        break;
      default:
        reinterpret_cast<uint64_t*>(chunk.out)[i] = value;
        break;
    }
  }
}

// One block of a DeltaBlock leaf, decoded by one thread block of
// deltaTasksKernel into the batch's 8-byte DeltaBlock values from `first` on.
struct DeltaTask {
  const uint8_t* data{nullptr};
  uint64_t base{0};
  uint64_t signBit{0};
  uint64_t first{0};
  uint32_t numRows{0};
  uint8_t bitWidth{0};
};

__global__ void deltaTasksKernel(const DeltaTask* tasks, uint64_t* values) {
  using Scan = cub::BlockScan<uint64_t, kBlockSize>;
  __shared__ typename Scan::TempStorage temp;
  const DeltaTask task = tasks[blockIdx.x];
  const uint64_t mask = task.bitWidth == 64
      ? ~uint64_t{0}
      : (uint64_t{1} << task.bitWidth) - 1;
  uint64_t* out = values + task.first;
  uint64_t carry = task.base;
  for (uint32_t first = 0; first < task.numRows; first += kBlockSize) {
    const uint32_t row = first + threadIdx.x;
    uint64_t delta = 0;
    if (row > 0 && row < task.numRows && task.bitWidth > 0) {
      const uint64_t bit = static_cast<uint64_t>(row - 1) * task.bitWidth;
      const uint8_t* p = task.data + (bit >> 3);
      const int shift = bit & 7;
      uint64_t word = loadUnaligned<uint64_t>(p) >> shift;
      if (shift + task.bitWidth > 64) {
        word |= static_cast<uint64_t>(p[8]) << (64 - shift);
      }
      delta = word & mask;
    }
    uint64_t sum;
    uint64_t total;
    Scan(temp).InclusiveSum(delta, sum, total);
    if (row < task.numRows) {
      out[row] = (carry + sum) ^ task.signBit;
    }
    carry += total;
    __syncthreads();
  }
}

// The string bytes of one chunk of a column that has FSST chunks, whose
// decoded sizes are known only on the device. FSST chunks expand their codes
// with tables[table]; others (table -1) copy their bytes.
struct StringChunk {
  const uint8_t* src{nullptr};
  // An FSST chunk's code lengths and their exclusive scan over the batch, at
  // the chunk's first entry.
  const uint32_t* codeLengths{nullptr};
  const uint32_t* codeOffsets{nullptr};
  // The column's entry lengths and then, once scanned, string offsets, at the
  // chunk's first entry.
  int32_t* entries{nullptr};
  char* chars{nullptr};
  uint32_t numEntries{0};
  int32_t table{-1};
};

constexpr int kStringsPerBlock = 1024;

__global__ void fsstLengthsBatchKernel(
    const StringChunk* chunks,
    const uint32_t* blockStarts,
    int numChunks,
    const FsstTable* tables) {
  __shared__ FsstTable shared;
  const int c = findSegment(blockStarts, numChunks);
  const StringChunk chunk = chunks[c];
  loadFsstTable(tables[chunk.table], shared);
  const uint32_t begin = (blockIdx.x - blockStarts[c]) * kStringsPerBlock;
  const uint32_t end = min(begin + kStringsPerBlock, chunk.numEntries);
  const uint32_t base = chunk.codeOffsets[0];
  for (uint32_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const uint8_t* p = chunk.src + (chunk.codeOffsets[i] - base);
    const uint8_t* stop = p + chunk.codeLengths[i];
    int32_t length = 0;
    while (p < stop) {
      const uint8_t code = *p++;
      if (code == kFsstEscape) {
        ++p;
        ++length;
      } else {
        length += shared.len[code];
      }
    }
    chunk.entries[i] = length;
  }
}

__global__ void stringCharsKernel(
    const StringChunk* chunks,
    const uint32_t* blockStarts,
    int numChunks,
    const FsstTable* tables) {
  __shared__ FsstTable shared;
  const int c = findSegment(blockStarts, numChunks);
  const StringChunk chunk = chunks[c];
  if (chunk.table >= 0) {
    loadFsstTable(tables[chunk.table], shared);
  }
  const uint32_t begin = (blockIdx.x - blockStarts[c]) * kStringsPerBlock;
  const uint32_t end = min(begin + kStringsPerBlock, chunk.numEntries);
  for (uint32_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    char* out = chunk.chars + chunk.entries[i];
    if (chunk.table < 0) {
      const uint8_t* p = chunk.src + (chunk.entries[i] - chunk.entries[0]);
      const int32_t length = chunk.entries[i + 1] - chunk.entries[i];
      for (int32_t k = 0; k < length; ++k) {
        out[k] = static_cast<char>(p[k]);
      }
      continue;
    }
    const uint8_t* p = chunk.src + (chunk.codeOffsets[i] - chunk.codeOffsets[0]);
    const uint8_t* stop = p + chunk.codeLengths[i];
    while (p < stop) {
      const uint8_t code = *p++;
      if (code == kFsstEscape) {
        *out++ = static_cast<char>(*p++);
        continue;
      }
      const uint64_t symbol = shared.symbol[code];
      const int length = shared.len[code];
      for (int k = 0; k < length; ++k) {
        out[k] = static_cast<char>(symbol >> (8 * k));
      }
      out += length;
    }
  }
}

template <typename T>
void scanInPlace(
    T* data,
    int64_t n,
    bool inclusive,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (n == 0) {
    return;
  }
  size_t tempBytes = 0;
  if (inclusive) {
    checkCuda(cub::DeviceScan::InclusiveSum(
        nullptr, tempBytes, data, data, n, stream.get()));
  } else {
    checkCuda(cub::DeviceScan::ExclusiveSum(
        nullptr, tempBytes, data, data, n, stream.get()));
  }
  rmm::device_buffer temp(tempBytes, stream, mr);
  if (inclusive) {
    checkCuda(cub::DeviceScan::InclusiveSum(
        temp.data(), tempBytes, data, data, n, stream.get()));
  } else {
    checkCuda(cub::DeviceScan::ExclusiveSum(
        temp.data(), tempBytes, data, data, n, stream.get()));
  }
}

// A chunk as decodeFastChunksKernel sees it, plus the parts decoded before it:
// run lengths and, for strings, the entries its indices point at.
struct ChunkShape {
  FastChunk chunk;
  FastLeaf runLengths;
  FastLeaf entryLengths;
  // String bytes, or FSST codes. charsSize is 0 for FSST entries, whose
  // decoded size is known only on the device.
  const uint8_t* chars{nullptr};
  uint64_t charsSize{0};
  uint32_t numEntries{0};
  // FSST entries: entryLengths holds their code lengths, and fsstTable
  // indexes the batch's FSST tables once resolved.
  bool fsst{false};
  int32_t fsstTable{-1};
  // A trivial or FSST string chunk whose entries are its rows.
  bool direct{false};
};

int gridSize(int64_t numValues) {
  return static_cast<int>(std::min<int64_t>(
      (numValues + kBlockSize - 1) / kBlockSize, int64_t{1} << 20));
}

// Calls f with a zero of the unsigned integer type of the given byte width.
template <typename F>
void dispatchWidth(int width, F&& f) {
  switch (width) {
    case 1:
      return f(uint8_t{});
    case 2:
      return f(uint16_t{});
    case 4:
      return f(uint32_t{});
    case 8:
      return f(uint64_t{});
  }
  NIMBLE_GPU_CHECK(false, "unsupported value width " + std::to_string(width));
}

// Bytes on the device holding a payload in its decoded form.
struct DeviceBytes {
  const uint8_t* data{nullptr};
  size_t size{0};
};

// A leaf payload found while walking the encodings.
struct Payload {
  std::span<const uint8_t> stored;
  Codec codec;
  // Size after decompression.
  size_t size;
  // Offset of the stored bytes in the staging buffer.
  size_t stagingOffset{0};
  DeviceBytes device;
};

size_t alignUp(size_t n) {
  return (n + 15) & ~size_t{15};
}

// Pinned host buffers for launch descriptors and values read back from the
// device, in a pool of their own so that they do not take the large stream
// buffers. cudaMallocHost is slow, so buffers are never freed.
class SmallPinned {
 public:
  explicit SmallPinned(size_t size) {
    auto& pool = smallPool();
    {
      std::lock_guard<std::mutex> lock{pool.mutex};
      auto best = pool.free.end();
      for (auto it = pool.free.begin(); it != pool.free.end(); ++it) {
        if (it->second >= size &&
            (best == pool.free.end() || it->second < best->second)) {
          best = it;
        }
      }
      if (best != pool.free.end()) {
        data_ = best->first;
        size_ = best->second;
        pool.free.erase(best);
        return;
      }
    }
    size_ = std::max(std::bit_ceil(size), size_t{1} << 20);
    void* data = nullptr;
    checkCuda(cudaMallocHost(&data, size_));
    data_ = static_cast<uint8_t*>(data);
  }

  ~SmallPinned() {
    auto& pool = smallPool();
    std::lock_guard<std::mutex> lock{pool.mutex};
    pool.free.emplace_back(data_, size_);
  }

  SmallPinned(const SmallPinned&) = delete;
  SmallPinned& operator=(const SmallPinned&) = delete;

  uint8_t* data() const {
    return data_;
  }

 private:
  struct Pool {
    std::mutex mutex;
    std::vector<std::pair<uint8_t*, size_t>> free;
  };

  static Pool& smallPool() {
    static auto* pool = new Pool();
    return *pool;
  }

  uint8_t* data_;
  size_t size_;
};

// Small host arrays read by a set of launches, packed into one pinned buffer
// and copied to the device in one transfer: a copy from pageable memory
// blocks the host.
class Metadata {
 public:
  template <typename T>
  size_t add(std::span<const T> values) {
    const size_t offset = alignUp(bytes_.size());
    bytes_.resize(offset + values.size_bytes());
    if (!values.empty()) {
      std::memcpy(bytes_.data() + offset, values.data(), values.size_bytes());
    }
    return offset;
  }

  template <typename T>
  size_t add(const std::vector<T>& values) {
    return add(std::span<const T>{values});
  }

  // The pinned buffer and the device copy live until the object is
  // destroyed, which must be after the stream has run the launches.
  void upload(cuda::stream_ref stream, rmm::device_async_resource_ref mr) {
    if (bytes_.empty()) {
      return;
    }
    pinned_.emplace(bytes_.size());
    std::memcpy(pinned_->data(), bytes_.data(), bytes_.size());
    device_ = rmm::device_buffer(bytes_.size(), stream, mr);
    checkCuda(cudaMemcpyAsync(
        device_.data(),
        pinned_->data(),
        bytes_.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
  }

  template <typename T>
  T* device(size_t offset) const {
    return reinterpret_cast<T*>(
        static_cast<uint8_t*>(const_cast<void*>(device_.data())) + offset);
  }

 private:
  std::vector<uint8_t> bytes_;
  std::optional<SmallPinned> pinned_;
  rmm::device_buffer device_;
};

// Items launched as one or more blocks each: item i starts at block starts[i].
struct BlockLaunch {
  size_t items{0};
  size_t starts{0};
  int count{0};
  uint32_t numBlocks{0};
};

template <typename T, typename Units>
BlockLaunch addBlocks(
    Metadata& metadata,
    const std::vector<T>& items,
    Units units,
    uint64_t unitsPerBlock) {
  std::vector<uint32_t> starts;
  starts.reserve(items.size());
  uint64_t numBlocks = 0;
  for (const auto& item : items) {
    starts.push_back(static_cast<uint32_t>(numBlocks));
    numBlocks += (units(item) + unitsPerBlock - 1) / unitsPerBlock;
  }
  NIMBLE_GPU_CHECK(
      numBlocks <= std::numeric_limits<int32_t>::max(), "too many blocks");
  BlockLaunch launch;
  launch.count = static_cast<int>(items.size());
  launch.numBlocks = static_cast<uint32_t>(numBlocks);
  launch.items = metadata.add(items);
  launch.starts = metadata.add(starts);
  return launch;
}

// Walks the chunks and encodings of a stripe's streams twice: first to find
// the payloads, which are then staged and decompressed, and then to decode
// them on the device.
class StripeDecode {
 public:
  StripeDecode(
      bool compactRowCount,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr)
      : compactRowCount_{compactRowCount}, stream_{stream}, mr_{mr} {}

  bool collecting{true};
  std::vector<Payload> payloads;
  // Found while resolving fast chunk shapes.
  std::vector<FsstTable> fsstTables;
  std::vector<DeltaTask> deltaTasks;
  uint64_t numDeltaValues{0};

  std::unique_ptr<cudf::column> decodeStream(
      std::span<const uint8_t> bytes,
      cudf::data_type type,
      cudf::size_type numRows) {
    if (bytes.empty()) {
      return collecting ? nullptr : allNulls(type, numRows);
    }
    std::vector<std::unique_ptr<cudf::column>> parts;
    ByteReader reader{bytes};
    while (reader.remaining() > 0) {
      const auto length = reader.u32();
      const auto codec = reader.u8();
      NIMBLE_GPU_CHECK(codec == 0, "chunk-level compression is not supported");
      parts.push_back(decodeEncoding(reader.take(length), type));
    }
    if (collecting) {
      return nullptr;
    }
    std::unique_ptr<cudf::column> result;
    if (parts.size() == 1) {
      result = std::move(parts[0]);
    } else {
      std::vector<cudf::column_view> views;
      views.reserve(parts.size());
      for (const auto& part : parts) {
        views.push_back(part->view());
      }
      result = cudf::concatenate(views, stream_, mr_);
    }
    NIMBLE_GPU_CHECK(
        result->size() == numRows,
        "stream has " + std::to_string(result->size()) + " rows, stripe has " +
            std::to_string(numRows));
    return result;
  }

  // The encodings of a stream's chunks, or nothing if a chunk is compressed
  // at the chunk level.
  static std::optional<std::vector<std::span<const uint8_t>>> chunks(
      std::span<const uint8_t> bytes) {
    std::vector<std::span<const uint8_t>> result;
    ByteReader reader{bytes};
    while (reader.remaining() > 0) {
      const auto length = reader.u32();
      if (reader.u8() != 0) {
        return std::nullopt;
      }
      result.push_back(reader.take(length));
    }
    return result;
  }

  // Maps host pointers into the file bytes to their copies on the device.
  void setStaging(const uint8_t* host, const uint8_t* device) {
    hostBase_ = host;
    deviceBase_ = device;
  }

  static bool fastType(cudf::data_type type) {
    return type.id() == cudf::type_id::STRING ||
        (cudf::is_fixed_width(type) && type.id() != cudf::type_id::BOOL8);
  }

  // Describes a chunk for decodeFastChunksKernel if its shape allows. Before
  // the payloads are decompressed `resolve` is false and only the shape is
  // checked; afterwards the leaves get their device bytes.
  bool fastShape(
      std::span<const uint8_t> bytes,
      cudf::data_type type,
      bool resolve,
      ChunkShape& shape) {
    if (!fastType(type)) {
      return false;
    }
    const bool isString = type.id() == cudf::type_id::STRING;
    ByteReader reader{bytes};
    const auto encoding = static_cast<EncodingType>(reader.u8());
    const auto dataType = static_cast<DataType>(reader.u8());
    const uint32_t rows = compactRowCount_ ? reader.varint() : reader.u32();
    auto& chunk = shape.chunk;
    chunk.numRows = rows;
    chunk.outWidth = isString ? 4 : cudf::size_of(type);
    uint32_t count = 0;
    switch (encoding) {
      case EncodingType::kDictionary: {
        const auto alphabet = reader.take(reader.u32());
        chunk.dictionary = true;
        return valueLeaf(alphabet, type, resolve, shape, count) &&
            indexLeaf(reader.rest(), resolve, shape, rows);
      }
      case EncodingType::kRle: {
        if (dataType == DataType::kBool) {
          return false;
        }
        if (!runs(reader, resolve, shape)) {
          return false;
        }
        const auto values = reader.rest();
        ByteReader inner{values};
        const auto innerEncoding = static_cast<EncodingType>(inner.u8());
        inner.u8();
        const uint32_t innerRows =
            compactRowCount_ ? inner.varint() : inner.u32();
        if (innerRows != chunk.numRuns) {
          return false;
        }
        if (innerEncoding == EncodingType::kDictionary) {
          const auto alphabet = inner.take(inner.u32());
          uint32_t indexRows = 0;
          chunk.dictionary = true;
          return valueLeaf(alphabet, type, resolve, shape, count) &&
              fastLeaf(inner.rest(), 4, resolve, chunk.indices, indexRows) &&
              indexRows == innerRows;
        }
        return valueLeaf(values, type, resolve, shape, count) &&
            count == innerRows;
      }
      default:
        if (!valueLeaf(bytes, type, resolve, shape, count) || count != rows) {
          return false;
        }
        shape.direct = isString &&
            (encoding == EncodingType::kTrivial ||
             encoding == EncodingType::kFsst);
        return true;
    }
  }

  // A short description of an encoding tree, for logging.
  std::string describe(std::span<const uint8_t> bytes) {
    ByteReader reader{bytes};
    const auto encoding = static_cast<EncodingType>(reader.u8());
    const auto dataType = static_cast<DataType>(reader.u8());
    static_cast<void>(compactRowCount_ ? reader.varint() : reader.u32());
    const auto type = std::to_string(static_cast<int>(dataType));
    switch (encoding) {
      case EncodingType::kTrivial:
        if (dataType == DataType::kString) {
          reader.u8();
          return "Trivial<" + type + ">(" + describe(reader.take(reader.u32())) +
              ")";
        }
        return "Trivial<" + type + ">";
      case EncodingType::kDictionary: {
        const auto alphabet = describe(reader.take(reader.u32()));
        return "Dictionary<" + type + ">(" + alphabet + "," +
            describe(reader.rest()) + ")";
      }
      case EncodingType::kRle: {
        const auto runLengths = describe(reader.take(reader.u32()));
        return "RLE<" + type + ">(" + runLengths +
            (dataType == DataType::kBool ? "" : "," + describe(reader.rest())) +
            ")";
      }
      case EncodingType::kMainlyConstant: {
        const auto isCommon = describe(reader.take(reader.u32()));
        return "MainlyConstant<" + type + ">(" + isCommon + "," +
            describe(reader.take(reader.u32())) + ")";
      }
      case EncodingType::kNullable: {
        const auto nonNulls = describe(reader.take(reader.u32()));
        return "Nullable<" + type + ">(" + nonNulls + "," +
            describe(reader.rest()) + ")";
      }
      default:
        return "E" + std::to_string(static_cast<int>(encoding)) + "<" + type +
            ">";
    }
  }

 private:
  // RLE run lengths, leaving the reader at the values.
  bool runs(ByteReader& reader, bool resolve, ChunkShape& shape) {
    const auto runLengths = reader.take(reader.u32());
    uint32_t numRuns = 0;
    if (!fastLeaf(runLengths, 4, resolve, shape.runLengths, numRuns) ||
        numRuns == 0) {
      return false;
    }
    shape.chunk.rle = true;
    shape.chunk.numRuns = numRuns;
    return true;
  }

  // Dictionary indices: a leaf, or run-length encoded leaf.
  bool indexLeaf(
      std::span<const uint8_t> bytes,
      bool resolve,
      ChunkShape& shape,
      uint32_t rows) {
    ByteReader reader{bytes};
    const auto encoding = static_cast<EncodingType>(reader.u8());
    reader.u8();
    const uint32_t indexRows = compactRowCount_ ? reader.varint() : reader.u32();
    if (indexRows != rows) {
      return false;
    }
    uint32_t count = 0;
    if (encoding != EncodingType::kRle) {
      return fastLeaf(bytes, 4, resolve, shape.chunk.indices, count);
    }
    return runs(reader, resolve, shape) &&
        fastLeaf(reader.rest(), 4, resolve, shape.chunk.indices, count) &&
        count == shape.chunk.numRuns;
  }

  // The leaf values of a chunk: numeric leaves, or string entries.
  bool valueLeaf(
      std::span<const uint8_t> bytes,
      cudf::data_type type,
      bool resolve,
      ChunkShape& shape,
      uint32_t& count) {
    if (type.id() != cudf::type_id::STRING) {
      return fastLeaf(bytes, cudf::size_of(type), resolve, shape.chunk.values, count);
    }
    ByteReader reader{bytes};
    const auto encoding = static_cast<EncodingType>(reader.u8());
    const auto dataType = static_cast<DataType>(reader.u8());
    count = compactRowCount_ ? reader.varint() : reader.u32();
    if (dataType != DataType::kString) {
      return false;
    }
    shape.chunk.values.kind = FastLeaf::kIdentity;
    shape.numEntries = count;
    if (encoding == EncodingType::kConstant) {
      const auto length = reader.u32();
      const auto value = reader.take(length);
      shape.entryLengths.kind = FastLeaf::kConstant;
      shape.entryLengths.width = 4;
      shape.entryLengths.baseline = length;
      shape.charsSize = length;
      shape.numEntries = 1;
      shape.chunk.values.kind = FastLeaf::kConstant;
      shape.chunk.values.width = 4;
      if (resolve) {
        shape.chars = deviceBase_ + (value.data() - hostBase_);
      }
      return true;
    }
    if (encoding == EncodingType::kFsst) {
      const auto codec = static_cast<Codec>(reader.u8());
      const auto table = reader.take(reader.varint());
      const auto codeLengths = reader.take(reader.varint());
      uint32_t lengthRows = 0;
      if (!fastLeaf(codeLengths, 4, resolve, shape.entryLengths, lengthRows) ||
          lengthRows != count) {
        return false;
      }
      shape.fsst = true;
      if (resolve) {
        shape.chars = payload(codec, reader.rest()).data;
        shape.fsstTable = static_cast<int32_t>(fsstTables.size());
        fsstTables.push_back(fsstTable(table));
      }
      return true;
    }
    if (encoding != EncodingType::kTrivial) {
      return false;
    }
    const auto codec = static_cast<Codec>(reader.u8());
    const auto lengths = reader.take(reader.u32());
    uint32_t lengthRows = 0;
    if (!fastLeaf(lengths, 4, resolve, shape.entryLengths, lengthRows) ||
        lengthRows != count) {
      return false;
    }
    const auto blob = reader.rest();
    if (codec == Codec::kUncompressed) {
      shape.charsSize = blob.size();
    } else {
      ByteReader header{blob};
      shape.charsSize = header.u32();
    }
    if (resolve) {
      const auto chars = payload(codec, blob);
      NIMBLE_GPU_CHECK(chars.size == shape.charsSize, "string blob size");
      shape.chars = chars.data;
    }
    return true;
  }

  const uint8_t* hostBase_{nullptr};
  const uint8_t* deviceBase_{nullptr};

  bool fastLeaf(
      std::span<const uint8_t> bytes,
      int width,
      bool resolve,
      FastLeaf& leaf,
      uint32_t& rows) {
    ByteReader reader{bytes};
    const auto encoding = static_cast<EncodingType>(reader.u8());
    const auto dataType = static_cast<DataType>(reader.u8());
    rows = compactRowCount_ ? reader.varint() : reader.u32();
    if (dataType == DataType::kBool || dataType == DataType::kString ||
        physicalWidth(dataType) != width) {
      return false;
    }
    leaf.width = width;
    switch (encoding) {
      case EncodingType::kTrivial: {
        leaf.kind = FastLeaf::kTrivial;
        const auto codec = static_cast<Codec>(reader.u8());
        if (resolve) {
          const auto bytes = payload(codec, reader.rest());
          NIMBLE_GPU_CHECK(
              bytes.size == static_cast<size_t>(rows) * width,
              "trivial payload size mismatch");
          leaf.data = bytes.data;
        }
        return true;
      }
      case EncodingType::kFixedBitWidth: {
        leaf.kind = FastLeaf::kFixedBitWidth;
        const auto codec = static_cast<Codec>(reader.u8());
        leaf.baseline = reader.uN(width);
        leaf.bitWidth = reader.u8();
        NIMBLE_GPU_CHECK(
            leaf.bitWidth <= 8 * width, "bit width exceeds the type");
        if (resolve) {
          const auto packed = payload(codec, reader.rest());
          NIMBLE_GPU_CHECK(
              packed.size >= (static_cast<uint64_t>(rows) * leaf.bitWidth + 7) / 8,
              "fixed bit width payload too small");
          leaf.data = packed.data;
        }
        return true;
      }
      case EncodingType::kConstant:
        leaf.kind = FastLeaf::kConstant;
        leaf.baseline = reader.uN(width);
        return true;
      case EncodingType::kVarint: {
        if (width != 4 && width != 8) {
          return false;
        }
        leaf.kind = FastLeaf::kVarint;
        leaf.baseline = reader.uN(width);
        leaf.count = rows;
        if (resolve) {
          const auto bytes = payload(Codec::kUncompressed, reader.rest());
          leaf.data = bytes.data;
          leaf.numBytes = static_cast<uint32_t>(bytes.size);
        }
        return true;
      }
      case EncodingType::kDeltaBlock: {
        leaf.kind = FastLeaf::kDeltaBlock;
        leaf.count = rows;
        if (resolve) {
          addDeltaTasks(reader, dataType, rows, leaf);
        }
        return true;
      }
      default:
        return false;
    }
  }

  // Lists the blocks of a DeltaBlock leaf for deltaTasksKernel, which decodes
  // them into the batch's DeltaBlock values from leaf.baseline on.
  void addDeltaTasks(
      ByteReader& reader,
      DataType dataType,
      uint32_t rows,
      FastLeaf& leaf) {
    const auto blockSize = reader.varint();
    const auto numBlocks = reader.varint();
    NIMBLE_GPU_CHECK(
        blockSize > 0 &&
            numBlocks == (static_cast<uint64_t>(rows) + blockSize - 1) / blockSize,
        "DeltaBlock block count");
    const size_t firstTask = deltaTasks.size();
    deltaTasks.resize(firstTask + numBlocks);
    auto* tasks = deltaTasks.data() + firstTask;
    for (uint32_t b = 0; b < numBlocks; ++b) {
      tasks[b].base = reader.varint64();
    }
    for (uint32_t b = 0; b < numBlocks; ++b) {
      tasks[b].bitWidth = reader.u8();
      NIMBLE_GPU_CHECK(tasks[b].bitWidth <= 64, "DeltaBlock bit width");
    }
    std::vector<uint32_t> offsets(numBlocks);
    for (auto& offset : offsets) {
      offset = reader.varint();
    }
    const auto packed = payload(Codec::kUncompressed, reader.rest());
    const bool isSigned = dataType == DataType::kInt8 ||
        dataType == DataType::kInt16 || dataType == DataType::kInt32 ||
        dataType == DataType::kInt64;
    const uint64_t signBit = isSigned
        ? uint64_t{1} << (8 * physicalWidth(dataType) - 1)
        : 0;
    leaf.baseline = numDeltaValues;
    for (uint32_t b = 0; b < numBlocks; ++b) {
      NIMBLE_GPU_CHECK(offsets[b] <= packed.size, "DeltaBlock offset");
      tasks[b].data = packed.data + offsets[b];
      tasks[b].signBit = signBit;
      tasks[b].first = numDeltaValues + uint64_t{b} * blockSize;
      tasks[b].numRows = std::min<uint32_t>(blockSize, rows - b * blockSize);
    }
    numDeltaValues += rows;
  }

  std::unique_ptr<cudf::column> decodeEncoding(
      std::span<const uint8_t> bytes,
      cudf::data_type type) {
    ByteReader reader{bytes};
    const auto encoding = static_cast<EncodingType>(reader.u8());
    const auto dataType = static_cast<DataType>(reader.u8());
    const uint32_t rows = compactRowCount_ ? reader.varint() : reader.u32();
    NIMBLE_GPU_CHECK(
        rows <= static_cast<uint32_t>(std::numeric_limits<cudf::size_type>::max()),
        "too many rows in a chunk");
    const auto numRows = static_cast<cudf::size_type>(rows);
    switch (encoding) {
      case EncodingType::kTrivial:
        if (dataType == DataType::kString) {
          return trivialString(reader, numRows);
        }
        if (dataType == DataType::kBool) {
          return trivialBool(reader, numRows);
        }
        return trivialNumeric(reader, dataType, type, numRows);
      case EncodingType::kFixedBitWidth:
        return fixedBitWidth(reader, dataType, type, numRows);
      case EncodingType::kConstant:
        return constant(reader, dataType, type, numRows);
      case EncodingType::kVarint:
        return varint(reader, dataType, type, numRows);
      case EncodingType::kDictionary:
        return dictionary(reader, type);
      case EncodingType::kRle:
        return rle(reader, dataType, type);
      case EncodingType::kMainlyConstant:
        return mainlyConstant(reader, dataType, type, numRows);
      case EncodingType::kNullable:
        return nullable(reader, type, numRows);
      case EncodingType::kSparseBool:
        return sparseBool(reader, numRows);
      case EncodingType::kFsst:
        return fsst(reader, numRows);
      case EncodingType::kDeltaBlock:
        return deltaBlock(reader, dataType, type, numRows);
      case EncodingType::kAlp:
        return alp(reader, dataType, type, numRows);
    }
    NIMBLE_GPU_CHECK(
        false,
        "unsupported encoding " + std::to_string(static_cast<int>(encoding)));
    return nullptr;
  }

  // Registers a payload while collecting and returns its device bytes when
  // decoding.
  DeviceBytes payload(Codec codec, std::span<const uint8_t> stored) {
    if (collecting) {
      size_t size = stored.size();
      if (codec != Codec::kUncompressed) {
        NIMBLE_GPU_CHECK(
            codec == Codec::kZstd || codec == Codec::kLz4,
            "unsupported compression " +
                std::to_string(static_cast<int>(codec)));
        ByteReader reader{stored};
        size = reader.u32();
        stored = reader.rest();
      }
      payloadIndex_[stored.data()] = payloads.size();
      payloads.push_back(Payload{stored, codec, size});
      return {};
    }
    if (codec != Codec::kUncompressed) {
      stored = stored.subspan(4);
    }
    return payloads.at(payloadIndex_.at(stored.data())).device;
  }

  DeviceBytes compressedPayload(ByteReader& reader) {
    const auto codec = static_cast<Codec>(reader.u8());
    return payload(codec, reader.rest());
  }

  static void checkWidth(DataType dataType, cudf::data_type type) {
    NIMBLE_GPU_CHECK(
        cudf::is_fixed_width(type) &&
            physicalWidth(dataType) == static_cast<int>(cudf::size_of(type)),
        "data type " + std::to_string(static_cast<int>(dataType)) +
            " does not match the requested type");
  }

  std::unique_ptr<cudf::column> fixedWidth(
      cudf::data_type type,
      cudf::size_type numRows,
      cudf::mask_state state = cudf::mask_state::UNALLOCATED) {
    return cudf::make_fixed_width_column(type, numRows, state, stream_, mr_);
  }

  std::unique_ptr<cudf::column> allNulls(
      cudf::data_type type,
      cudf::size_type numRows) {
    if (type.id() == cudf::type_id::STRING) {
      cudf::string_scalar null{"", false, stream_, mr_};
      return cudf::make_column_from_scalar(null, numRows, stream_, mr_);
    }
    return fixedWidth(type, numRows, cudf::mask_state::ALL_NULL);
  }

  // A column of numRows copies of a value given by its bits.
  std::unique_ptr<cudf::column>
  filled(cudf::data_type type, uint64_t bits, cudf::size_type numRows) {
    auto column = fixedWidth(type, numRows);
    if (numRows > 0) {
      dispatchWidth(cudf::size_of(type), [&](auto zero) {
        using U = decltype(zero);
        fillKernel<U><<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
            column->mutable_view().template head<U>(),
            static_cast<U>(bits),
            numRows);
      });
    }
    return column;
  }

  std::unique_ptr<cudf::column> filledString(
      const std::string& value,
      cudf::size_type numRows) {
    cudf::string_scalar scalar{value, true, stream_, mr_};
    return cudf::make_column_from_scalar(scalar, numRows, stream_, mr_);
  }

  std::unique_ptr<cudf::column> trivialNumeric(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    const auto bytes = compressedPayload(reader);
    if (collecting) {
      return nullptr;
    }
    checkWidth(dataType, type);
    const size_t size = static_cast<size_t>(numRows) * cudf::size_of(type);
    NIMBLE_GPU_CHECK(bytes.size == size, "trivial payload size mismatch");
    auto column = fixedWidth(type, numRows);
    if (size > 0) {
      checkCuda(cudaMemcpyAsync(
          column->mutable_view().head<uint8_t>(),
          bytes.data,
          size,
          cudaMemcpyDeviceToDevice,
          stream_.get()));
    }
    return column;
  }

  std::unique_ptr<cudf::column> trivialBool(
      ByteReader& reader,
      cudf::size_type numRows) {
    const auto bytes = compressedPayload(reader);
    if (collecting) {
      return nullptr;
    }
    NIMBLE_GPU_CHECK(
        bytes.size >= (static_cast<size_t>(numRows) + 7) / 8,
        "bool bitmap too small");
    auto column = fixedWidth(cudf::data_type{cudf::type_id::BOOL8}, numRows);
    if (numRows > 0) {
      unpackBoolsKernel<<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
          bytes.data, column->mutable_view().head<bool>(), numRows);
    }
    return column;
  }

  std::unique_ptr<cudf::column> trivialString(
      ByteReader& reader,
      cudf::size_type numRows) {
    const auto codec = static_cast<Codec>(reader.u8());
    const auto lengthsSize = reader.u32();
    auto lengths = decodeEncoding(
        reader.take(lengthsSize), cudf::data_type{cudf::type_id::INT32});
    const auto blob = payload(codec, reader.rest());
    if (collecting) {
      return nullptr;
    }
    NIMBLE_GPU_CHECK(lengths->size() == numRows, "string lengths row count");
    auto offsets = fixedWidth(cudf::data_type{cudf::type_id::INT32}, numRows + 1);
    auto* offsetsData = offsets->mutable_view().head<int32_t>();
    checkCuda(cudaMemsetAsync(offsetsData, 0, sizeof(int32_t), stream_.get()));
    if (numRows > 0) {
      const auto* lengthsData = lengths->view().head<int32_t>();
      size_t tempBytes = 0;
      checkCuda(cub::DeviceScan::InclusiveSum(
          nullptr, tempBytes, lengthsData, offsetsData + 1, numRows, stream_.get()));
      rmm::device_buffer temp(tempBytes, stream_, mr_);
      checkCuda(cub::DeviceScan::InclusiveSum(
          temp.data(),
          tempBytes,
          lengthsData,
          offsetsData + 1,
          numRows,
          stream_.get()));
    }
    rmm::device_buffer chars(blob.size, stream_, mr_);
    if (blob.size > 0) {
      checkCuda(cudaMemcpyAsync(
          chars.data(),
          blob.data,
          blob.size,
          cudaMemcpyDeviceToDevice,
          stream_.get()));
    }
    return cudf::make_strings_column(
        numRows,
        std::move(offsets),
        std::move(chars),
        0,
        cuda::device_buffer<std::byte>(stream_, mr_, 0, cuda::no_init));
  }

  std::unique_ptr<cudf::column> alp(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    NIMBLE_GPU_CHECK(
        dataType == DataType::kDouble || dataType == DataType::kFloat,
        "ALP needs a floating point type");
    uint32_t control = reader.u8();
    control |= reader.u8() << 8;
    control |= reader.u8() << 16;
    const int exponent = control & 0x1f;
    const int factor = (control >> 5) & 0x1f;
    NIMBLE_GPU_CHECK(exponent < 24 && factor < 24, "ALP exponent or factor");
    const uint32_t numExceptions = (control & (1u << 16)) ? reader.varint() : 0;
    auto encoded = decodeEncoding(
        reader.take(reader.varint()), cudf::data_type{cudf::type_id::UINT64});
    std::unique_ptr<cudf::column> positions;
    std::unique_ptr<cudf::column> exceptions;
    if (numExceptions > 0) {
      positions = decodeEncoding(
          reader.take(reader.varint()), cudf::data_type{cudf::type_id::INT32});
      exceptions = decodeEncoding(reader.take(reader.varint()), type);
    }
    if (collecting) {
      return nullptr;
    }
    checkWidth(dataType, type);
    NIMBLE_GPU_CHECK(encoded->size() == numRows, "ALP encoded row count");
    auto column = fixedWidth(type, numRows);
    if (numRows == 0) {
      return column;
    }
    auto decode = [&](auto zero) {
      using F = decltype(zero);
      auto* out = column->mutable_view().template head<F>();
      alpKernel<F><<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
          encoded->view().template head<uint64_t>(),
          kAlpPow10[factor],
          kAlpPow10[exponent],
          out,
          numRows);
      if (numExceptions > 0) {
        NIMBLE_GPU_CHECK(
            positions->size() == static_cast<cudf::size_type>(numExceptions) &&
                exceptions->size() == positions->size(),
            "ALP exception count");
        scatterKernel<F><<<gridSize(numExceptions), kBlockSize, 0, stream_.get()>>>(
            positions->view().template head<int32_t>(),
            exceptions->view().template head<F>(),
            out,
            numExceptions);
      }
    };
    if (dataType == DataType::kDouble) {
      decode(double{});
    } else {
      decode(float{});
    }
    checkCuda(cudaGetLastError());
    return column;
  }

  std::unique_ptr<cudf::column> deltaBlock(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    const int width = physicalWidth(dataType);
    NIMBLE_GPU_CHECK(width > 0, "DeltaBlock needs an integer type");
    const auto blockSize = reader.varint();
    const auto numBlocks = reader.varint();
    NIMBLE_GPU_CHECK(
        blockSize > 0 &&
            numBlocks == (static_cast<uint64_t>(numRows) + blockSize - 1) / blockSize,
        "DeltaBlock block count");
    std::vector<DeltaBlock> blocks(numBlocks);
    for (auto& block : blocks) {
      block.base = reader.varint64();
    }
    for (auto& block : blocks) {
      block.bitWidth = reader.u8();
      NIMBLE_GPU_CHECK(block.bitWidth <= 64, "DeltaBlock bit width");
    }
    for (auto& block : blocks) {
      block.offset = reader.varint();
    }
    const auto packed = payload(Codec::kUncompressed, reader.rest());
    if (collecting) {
      return nullptr;
    }
    checkWidth(dataType, type);
    auto column = fixedWidth(type, numRows);
    if (numRows == 0) {
      return column;
    }
    rmm::device_uvector<DeltaBlock> deviceBlocks(numBlocks, stream_, mr_);
    checkCuda(cudaMemcpyAsync(
        deviceBlocks.data(),
        blocks.data(),
        numBlocks * sizeof(DeltaBlock),
        cudaMemcpyHostToDevice,
        stream_.get()));
    const bool isSigned = dataType == DataType::kInt8 ||
        dataType == DataType::kInt16 || dataType == DataType::kInt32 ||
        dataType == DataType::kInt64;
    const uint64_t signBit = isSigned ? uint64_t{1} << (8 * width - 1) : 0;
    dispatchWidth(width, [&](auto zero) {
      using U = decltype(zero);
      deltaBlockKernel<U><<<numBlocks, kBlockSize, 0, stream_.get()>>>(
          packed.data,
          deviceBlocks.data(),
          blockSize,
          signBit,
          column->mutable_view().template head<U>(),
          static_cast<uint32_t>(numRows));
    });
    checkCuda(cudaGetLastError());
    return column;
  }

  std::unique_ptr<cudf::column> fsst(
      ByteReader& reader,
      cudf::size_type numRows) {
    const auto codec = static_cast<Codec>(reader.u8());
    const auto table = fsstTable(reader.take(reader.varint()));
    auto lengths = decodeEncoding(
        reader.take(reader.varint()), cudf::data_type{cudf::type_id::INT32});
    const auto blob = payload(codec, reader.rest());
    if (collecting) {
      return nullptr;
    }
    NIMBLE_GPU_CHECK(lengths->size() == numRows, "FSST lengths row count");
    NIMBLE_GPU_CHECK(
        blob.size <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "FSST blob too large");
    auto offsets = fixedWidth(cudf::data_type{cudf::type_id::INT32}, numRows + 1);
    auto* offsetsData = offsets->mutable_view().head<int32_t>();
    checkCuda(cudaMemsetAsync(offsetsData, 0, sizeof(int32_t), stream_.get()));
    if (numRows == 0) {
      return cudf::make_strings_column(
          0,
          std::move(offsets),
          rmm::device_buffer(0, stream_, mr_),
          0,
          cuda::device_buffer<std::byte>(stream_, mr_, 0, cuda::no_init));
    }
    const auto* compressedLengths = lengths->view().head<int32_t>();
    rmm::device_uvector<int32_t> codeOffsets(numRows, stream_, mr_);
    checkCuda(cudaMemcpyAsync(
        codeOffsets.data(),
        compressedLengths,
        numRows * sizeof(int32_t),
        cudaMemcpyDeviceToDevice,
        stream_.get()));
    scanInPlace(codeOffsets.data(), numRows, false, stream_, mr_);
    fsstLengthsKernel<<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
        table,
        blob.data,
        codeOffsets.data(),
        compressedLengths,
        offsetsData + 1,
        numRows);
    checkCuda(cudaGetLastError());
    scanInPlace(offsetsData + 1, numRows, true, stream_, mr_);
    int32_t numChars = 0;
    checkCuda(cudaMemcpyAsync(
        &numChars,
        offsetsData + numRows,
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream_.get()));
    checkCuda(cudaStreamSynchronize(stream_.get()));
    rmm::device_buffer chars(numChars, stream_, mr_);
    fsstDecodeKernel<<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
        table,
        blob.data,
        codeOffsets.data(),
        compressedLengths,
        offsetsData,
        static_cast<char*>(chars.data()),
        numRows);
    checkCuda(cudaGetLastError());
    return cudf::make_strings_column(
        numRows,
        std::move(offsets),
        std::move(chars),
        0,
        cuda::device_buffer<std::byte>(stream_, mr_, 0, cuda::no_init));
  }

  std::unique_ptr<cudf::column> fixedBitWidth(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    const auto codec = static_cast<Codec>(reader.u8());
    const int width = physicalWidth(dataType);
    NIMBLE_GPU_CHECK(width > 0, "fixed bit width needs a numeric type");
    const uint64_t baseline = reader.uN(width);
    const int bitWidth = reader.u8();
    NIMBLE_GPU_CHECK(bitWidth <= 8 * width, "bit width exceeds the type");
    const auto packed = payload(codec, reader.rest());
    if (collecting) {
      return nullptr;
    }
    checkWidth(dataType, type);
    NIMBLE_GPU_CHECK(
        packed.size >=
            (static_cast<uint64_t>(numRows) * bitWidth + 7) / 8,
        "fixed bit width payload too small");
    auto column = fixedWidth(type, numRows);
    if (numRows > 0) {
      dispatchWidth(width, [&](auto zero) {
        using U = decltype(zero);
        unpackBitsKernel<U><<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
            packed.data,
            bitWidth,
            static_cast<U>(baseline),
            column->mutable_view().template head<U>(),
            numRows);
      });
    }
    return column;
  }

  std::unique_ptr<cudf::column> constant(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    if (dataType == DataType::kString) {
      const auto value = reader.string();
      return collecting ? nullptr : filledString(value, numRows);
    }
    if (dataType == DataType::kBool) {
      const auto value = reader.u8();
      return collecting
          ? nullptr
          : filled(cudf::data_type{cudf::type_id::BOOL8}, value != 0, numRows);
    }
    const auto bits = reader.uN(physicalWidth(dataType));
    if (collecting) {
      return nullptr;
    }
    checkWidth(dataType, type);
    return filled(type, bits, numRows);
  }

  std::unique_ptr<cudf::column> varint(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    const int width = physicalWidth(dataType);
    NIMBLE_GPU_CHECK(width == 4 || width == 8, "varint needs a 4 or 8 byte type");
    const uint64_t baseline = reader.uN(width);
    const auto bytes = payload(Codec::kUncompressed, reader.rest());
    if (collecting) {
      return nullptr;
    }
    checkWidth(dataType, type);
    auto column = fixedWidth(type, numRows);
    const auto numBytes = static_cast<int64_t>(bytes.size);
    if (numRows == 0 || numBytes == 0) {
      return column;
    }
    rmm::device_uvector<int32_t> ends(numBytes, stream_, mr_);
    rmm::device_uvector<int32_t> index(numBytes, stream_, mr_);
    varintEndsKernel<<<gridSize(numBytes), kBlockSize, 0, stream_.get()>>>(
        bytes.data, ends.data(), numBytes);
    size_t tempBytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(
        nullptr, tempBytes, ends.data(), index.data(), numBytes, stream_.get()));
    rmm::device_buffer temp(tempBytes, stream_, mr_);
    checkCuda(cub::DeviceScan::ExclusiveSum(
        temp.data(), tempBytes, ends.data(), index.data(), numBytes, stream_.get()));
    dispatchWidth(width, [&](auto zero) {
      using U = decltype(zero);
      varintDecodeKernel<U><<<gridSize(numBytes), kBlockSize, 0, stream_.get()>>>(
          bytes.data,
          index.data(),
          static_cast<U>(baseline),
          column->mutable_view().template head<U>(),
          numBytes);
    });
    return column;
  }

  std::unique_ptr<cudf::column> dictionary(
      ByteReader& reader,
      cudf::data_type type) {
    const auto alphabetSize = reader.u32();
    auto alphabet = decodeEncoding(reader.take(alphabetSize), type);
    auto indices =
        decodeEncoding(reader.rest(), cudf::data_type{cudf::type_id::INT32});
    if (collecting) {
      return nullptr;
    }
    auto gathered = cudf::gather(
        cudf::table_view{{alphabet->view()}},
        indices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream_,
        cudf::memory_resources{mr_, mr_});
    return std::move(gathered->release()[0]);
  }

  std::unique_ptr<cudf::column>
  rle(ByteReader& reader, DataType dataType, cudf::data_type type) {
    const auto runLengthsSize = reader.u32();
    auto runLengths = decodeEncoding(
        reader.take(runLengthsSize), cudf::data_type{cudf::type_id::INT32});
    std::unique_ptr<cudf::column> values;
    if (dataType == DataType::kBool) {
      const bool first = reader.u8() != 0;
      if (!collecting) {
        const auto numRuns = runLengths->size();
        values = fixedWidth(cudf::data_type{cudf::type_id::BOOL8}, numRuns);
        if (numRuns > 0) {
          alternateBoolsKernel<<<gridSize(numRuns), kBlockSize, 0, stream_.get()>>>(
              first, values->mutable_view().head<bool>(), numRuns);
        }
      }
    } else {
      values = decodeEncoding(reader.rest(), type);
    }
    if (collecting) {
      return nullptr;
    }
    auto repeated = cudf::repeat(
        cudf::table_view{{values->view()}}, runLengths->view(), stream_, mr_);
    return std::move(repeated->release()[0]);
  }

  std::unique_ptr<cudf::column> mainlyConstant(
      ByteReader& reader,
      DataType dataType,
      cudf::data_type type,
      cudf::size_type numRows) {
    const auto isCommonSize = reader.u32();
    auto isCommon = decodeEncoding(
        reader.take(isCommonSize), cudf::data_type{cudf::type_id::BOOL8});
    const auto otherValuesSize = reader.u32();
    auto otherValues = decodeEncoding(reader.take(otherValuesSize), type);
    std::unique_ptr<cudf::column> common;
    if (dataType == DataType::kString) {
      const auto value = reader.string();
      if (!collecting) {
        common = filledString(value, numRows);
      }
    } else {
      const auto bits = reader.uN(physicalWidth(dataType));
      if (!collecting) {
        checkWidth(dataType, type);
        common = filled(type, bits, numRows);
      }
    }
    if (collecting) {
      return nullptr;
    }
    if (otherValues->size() == 0) {
      return common;
    }
    auto isOther = fixedWidth(cudf::data_type{cudf::type_id::BOOL8}, numRows);
    invertBoolsKernel<<<gridSize(numRows), kBlockSize, 0, stream_.get()>>>(
        isCommon->view().head<bool>(),
        isOther->mutable_view().head<bool>(),
        numRows);
    auto scattered = cudf::boolean_mask_scatter(
        cudf::table_view{{otherValues->view()}},
        cudf::table_view{{common->view()}},
        isOther->view(),
        stream_,
        mr_);
    return std::move(scattered->release()[0]);
  }

  std::unique_ptr<cudf::column> nullable(
      ByteReader& reader,
      cudf::data_type type,
      cudf::size_type numRows) {
    const auto nonNullsSize = reader.u32();
    auto nonNulls = decodeEncoding(reader.take(nonNullsSize), type);
    auto present =
        decodeEncoding(reader.rest(), cudf::data_type{cudf::type_id::BOOL8});
    if (collecting) {
      return nullptr;
    }
    auto target = allNulls(type, numRows);
    if (nonNulls->size() == 0) {
      return target;
    }
    auto scattered = cudf::boolean_mask_scatter(
        cudf::table_view{{nonNulls->view()}},
        cudf::table_view{{target->view()}},
        present->view(),
        stream_,
        mr_);
    return std::move(scattered->release()[0]);
  }

  std::unique_ptr<cudf::column> sparseBool(
      ByteReader& reader,
      cudf::size_type numRows) {
    const bool sparseValue = reader.u8() != 0;
    auto indices =
        decodeEncoding(reader.rest(), cudf::data_type{cudf::type_id::INT32});
    if (collecting) {
      return nullptr;
    }
    auto column =
        filled(cudf::data_type{cudf::type_id::BOOL8}, !sparseValue, numRows);
    // The last index is a sentinel equal to the row count.
    const auto numSparse = indices->size() - 1;
    if (numSparse > 0) {
      scatterBoolKernel<<<gridSize(numSparse), kBlockSize, 0, stream_.get()>>>(
          indices->view().head<int32_t>(),
          sparseValue,
          column->mutable_view().head<bool>(),
          numSparse);
    }
    return column;
  }

  const bool compactRowCount_;
  cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  std::unordered_map<const uint8_t*, size_t> payloadIndex_;
};

using CodecResult = cudf::io::detail::codec_exec_result;

// The payloads of one codec, decompressed in one batch. Their spans and
// results are in the decode's Metadata.
struct Decompression {
  Codec codec;
  size_t count{0};
  size_t maxSize{0};
  size_t totalSize{0};
  size_t inputs{0};
  size_t outputs{0};
  size_t results{0};
  std::vector<size_t> expected;
  std::vector<size_t> storedSizes;
};

// Points each payload of the codec at its decompressed slot and lists the
// batch's spans and results.
Decompression prepareDecompression(
    std::vector<Payload>& payloads,
    Codec codec,
    const uint8_t* staging,
    uint8_t* output,
    std::span<const size_t> outputOffsets,
    Metadata& metadata) {
  Decompression batch{codec};
  std::vector<cudf::device_span<uint8_t const>> inputs;
  std::vector<cudf::device_span<uint8_t>> outputs;
  for (size_t i = 0; i < payloads.size(); ++i) {
    auto& payload = payloads[i];
    if (payload.codec != codec) {
      continue;
    }
    inputs.emplace_back(staging + payload.stagingOffset, payload.stored.size());
    outputs.emplace_back(output + outputOffsets[i], payload.size);
    payload.device = {output + outputOffsets[i], payload.size};
    batch.expected.push_back(payload.size);
    batch.storedSizes.push_back(payload.stored.size());
    batch.maxSize = std::max(batch.maxSize, payload.size);
    batch.totalSize += payload.size;
  }
  batch.count = inputs.size();
  if (batch.count == 0) {
    return batch;
  }
  batch.inputs = metadata.add(inputs);
  batch.outputs = metadata.add(outputs);
  // Entries that enter as SKIPPED are left untouched by the codec.
  batch.results = metadata.add(std::vector<CodecResult>(
      batch.count,
      CodecResult{0, cudf::io::detail::codec_status::FAILURE}));
  return batch;
}

// Decompresses the batch and copies its results to `readback`, a pinned
// buffer checked by checkDecompression once the stream has run.
void launchDecompression(
    const Decompression& batch,
    const Metadata& metadata,
    CodecResult* readback,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (batch.count == 0) {
    return;
  }
  nvtx3::scoped_range range{"NimbleGpuDecoder::decompress"};
  auto* results = metadata.device<CodecResult>(batch.results);
  cudf::io::detail::decompress(
      batch.codec == Codec::kZstd ? cudf::io::compression_type::ZSTD
                                  : cudf::io::compression_type::LZ4,
      cudf::device_span<cudf::device_span<uint8_t const> const>{
          metadata.device<cudf::device_span<uint8_t const>>(batch.inputs),
          batch.count},
      cudf::device_span<cudf::device_span<uint8_t> const>{
          metadata.device<cudf::device_span<uint8_t>>(batch.outputs),
          batch.count},
      cudf::device_span<CodecResult>{results, batch.count},
      batch.maxSize,
      batch.totalSize,
      stream,
      cudf::memory_resources{mr, mr});
  checkCuda(cudaMemcpyAsync(
      readback,
      results,
      batch.count * sizeof(CodecResult),
      cudaMemcpyDeviceToHost,
      stream.get()));
}

void checkDecompression(const Decompression& batch, const CodecResult* results) {
  for (size_t i = 0; i < batch.count; ++i) {
    NIMBLE_GPU_CHECK(
        results[i].status == cudf::io::detail::codec_status::SUCCESS &&
            results[i].bytes_written == batch.expected[i],
        "decompression failed for payload " + std::to_string(i) + " of " +
            std::to_string(batch.count) + ": status " +
            std::to_string(static_cast<int>(results[i].status)) + ", wrote " +
            std::to_string(results[i].bytes_written) + " of " +
            std::to_string(batch.expected[i]) + " bytes from " +
            std::to_string(batch.storedSizes[i]) + " compressed, max " +
            std::to_string(batch.maxSize) + ", total " +
            std::to_string(batch.totalSize));
  }
}

} // namespace

namespace {
// Pinned buffers not in use. They are never freed: cudaMallocHost costs about
// 100 ms per buffer.
struct PinnedPool {
  std::mutex mutex;
  std::vector<std::pair<uint8_t*, size_t>> free;
};

PinnedPool& pinnedPool() {
  static auto* pool = new PinnedPool();
  return *pool;
}
} // namespace

NimbleGpuDecoder::PinnedBuffer::PinnedBuffer(size_t size) {
  auto& pool = pinnedPool();
  {
    std::lock_guard<std::mutex> lock{pool.mutex};
    auto best = pool.free.end();
    for (auto it = pool.free.begin(); it != pool.free.end(); ++it) {
      if (it->second >= size &&
          (best == pool.free.end() || it->second < best->second)) {
        best = it;
      }
    }
    if (best != pool.free.end()) {
      data_ = best->first;
      size_ = best->second;
      pool.free.erase(best);
      return;
    }
  }
  constexpr size_t kGranularity = size_t{64} << 20;
  size_ = std::max<size_t>(
      (size + kGranularity - 1) / kGranularity * kGranularity, kGranularity);
  void* data = nullptr;
  checkCuda(cudaMallocHost(&data, size_));
  data_ = static_cast<uint8_t*>(data);
}

NimbleGpuDecoder::PinnedBuffer::~PinnedBuffer() {
  auto& pool = pinnedPool();
  std::lock_guard<std::mutex> lock{pool.mutex};
  pool.free.emplace_back(data_, size_);
}

std::vector<std::unique_ptr<cudf::column>> NimbleGpuDecoder::decode(
    std::span<const NimbleStripeStreams> stripes,
    std::span<const uint8_t> hostBytes,
    std::span<const cudf::data_type> types,
    bool compactRowCount,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  for (const auto& stripe : stripes) {
    NIMBLE_GPU_CHECK(
        stripe.streams.size() == types.size(), "one type per stream");
  }
  StripeDecode walk{compactRowCount, stream, mr};
  // Columns whose every chunk decodeFastChunksKernel handles.
  std::vector<bool> fast(types.size(), true);
  // String bytes of each column's fast chunks.
  std::vector<uint64_t> chars(types.size(), 0);
  {
    nvtx3::scoped_range range{"NimbleGpuDecoder::plan"};
    for (const auto& stripe : stripes) {
      for (size_t i = 0; i < types.size(); ++i) {
        walk.decodeStream(stripe.streams[i], types[i], stripe.numRows);
        if (!fast[i]) {
          continue;
        }
        const auto chunks = StripeDecode::chunks(stripe.streams[i]);
        fast[i] = chunks.has_value() && !chunks->empty();
        for (size_t c = 0; fast[i] && c < chunks->size(); ++c) {
          ChunkShape shape;
          fast[i] = walk.fastShape((*chunks)[c], types[i], false, shape);
          chars[i] += shape.charsSize;
          static std::atomic<int> logged{0};
          if (!fast[i] && logged++ < 32) {
            LOG(INFO) << "Nimble GPU decode: column " << i << " type "
                      << static_cast<int>(types[i].id())
                      << " on the general path: "
                      << walk.describe((*chunks)[c]);
          }
        }
      }
    }
    for (size_t i = 0; i < types.size(); ++i) {
      if (chars[i] > static_cast<uint64_t>(
                         std::numeric_limits<cudf::size_type>::max())) {
        fast[i] = false;
      }
    }
  }

  // The whole host buffer goes over in one transfer, and payloads are used in
  // place. Decompressed payloads get their own aligned slots.
  auto& payloads = walk.payloads;
  size_t outputSize = 0;
  std::vector<size_t> outputOffsets(payloads.size());
  for (size_t i = 0; i < payloads.size(); ++i) {
    const auto* stored = payloads[i].stored.data();
    NIMBLE_GPU_CHECK(
        stored >= hostBytes.data() &&
            stored + payloads[i].stored.size() <=
                hostBytes.data() + hostBytes.size(),
        "payload outside the host buffer");
    payloads[i].stagingOffset = stored - hostBytes.data();
    if (payloads[i].codec != Codec::kUncompressed) {
      outputOffsets[i] = outputSize;
      // The padding covers 8-byte loads past the end of bit-packed data.
      outputSize += alignUp(payloads[i].size + 8);
    }
  }
  // The padding covers 8-byte loads past the end of the last payload.
  rmm::device_buffer staging(hostBytes.size() + 8, stream, mr);
  rmm::device_buffer output(outputSize, stream, mr);
  if (!hostBytes.empty()) {
    nvtx3::scoped_range range{"NimbleGpuDecoder::stage"};
    checkCuda(cudaMemcpyAsync(
        staging.data(),
        hostBytes.data(),
        hostBytes.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
  }
  auto* stagingData = static_cast<const uint8_t*>(staging.data());
  for (auto& payload : payloads) {
    if (payload.codec == Codec::kUncompressed) {
      payload.device = {stagingData + payload.stagingOffset, payload.size};
    }
  }
  auto* outputData = static_cast<uint8_t*>(output.data());
  // Every descriptor the fast path launches with goes over in one transfer
  // before the launches. The decompression results and the decoded sizes of
  // FSST columns come back in `readback`, read after the next wait on the
  // stream: a failed decompression is reported after the kernels that read
  // its output have run.
  Metadata metadata;
  Metadata charsMetadata;
  const auto zstd = prepareDecompression(
      payloads, Codec::kZstd, stagingData, outputData, outputOffsets, metadata);
  const auto lz4 = prepareDecompression(
      payloads, Codec::kLz4, stagingData, outputData, outputOffsets, metadata);
  std::optional<SmallPinned> readback;
  CodecResult* results = nullptr;
  bool decompressionChecked = false;
  auto checkDecompressed = [&] {
    if (!decompressionChecked && results != nullptr) {
      checkDecompression(zstd, results);
      checkDecompression(lz4, results + zstd.count);
    }
    decompressionChecked = true;
  };

  std::vector<std::unique_ptr<cudf::column>> columns(types.size());
  walk.collecting = false;
  cudf::size_type totalRows = 0;
  for (const auto& stripe : stripes) {
    totalRows += stripe.numRows;
  }
  {
    nvtx3::scoped_range range{"NimbleGpuDecoder::decode"};
    walk.setStaging(hostBytes.data(), stagingData);
    struct ColumnShapes {
      std::vector<ChunkShape> shapes;
      uint32_t numEntries{0};
      uint64_t numChars{0};
      bool direct{true};
      // Has FSST chunks, so its string bytes are sized on the device.
      bool fsst{false};
    };
    std::vector<ColumnShapes> fastColumns(types.size());
    size_t totalRuns = 0;
    uint64_t numCodeLengths = 0;
    for (size_t i = 0; i < types.size(); ++i) {
      if (!fast[i]) {
        continue;
      }
      auto& column = fastColumns[i];
      size_t row = 0;
      for (const auto& stripe : stripes) {
        const size_t stripeEnd = row + stripe.numRows;
        const auto chunks = StripeDecode::chunks(stripe.streams[i]);
        for (const auto chunkBytes : *chunks) {
          ChunkShape shape;
          NIMBLE_GPU_CHECK(
              walk.fastShape(chunkBytes, types[i], true, shape),
              "chunk shape changed");
          row += shape.chunk.numRows;
          if (shape.chunk.numRows == 0) {
            continue;
          }
          totalRuns += shape.chunk.numRuns;
          column.numEntries += shape.numEntries;
          column.numChars += shape.charsSize;
          column.direct = column.direct && shape.direct;
          column.fsst = column.fsst || shape.fsst;
          if (shape.fsst) {
            numCodeLengths += shape.numEntries;
          }
          column.shapes.push_back(shape);
        }
        NIMBLE_GPU_CHECK(
            row == stripeEnd,
            "stream has " + std::to_string(row - (stripeEnd - stripe.numRows)) +
                " rows, stripe has " + std::to_string(stripe.numRows));
      }
    }

    // Varint and DeltaBlock leaves are decoded first and then read as 8-byte
    // trivial leaves.
    std::vector<VarintSegment> segments;
    std::vector<std::pair<FastLeaf*, uint64_t>> varintLeaves;
    std::vector<FastLeaf*> deltaLeaves;
    uint64_t numVarintBytes = 0;
    uint64_t numVarints = 0;
    for (auto& column : fastColumns) {
      for (auto& shape : column.shapes) {
        for (FastLeaf* leaf :
             {&shape.chunk.values,
              &shape.chunk.indices,
              &shape.runLengths,
              &shape.entryLengths}) {
          if (leaf->kind == FastLeaf::kDeltaBlock) {
            deltaLeaves.push_back(leaf);
            continue;
          }
          if (leaf->kind != FastLeaf::kVarint) {
            continue;
          }
          segments.push_back(VarintSegment{
              leaf->data,
              leaf->baseline,
              leaf->numBytes,
              static_cast<uint32_t>(numVarintBytes),
              leaf->width});
          numVarintBytes += leaf->numBytes;
          varintLeaves.emplace_back(leaf, numVarints);
          numVarints += leaf->count;
        }
      }
    }
    NIMBLE_GPU_CHECK(
        numVarintBytes <= std::numeric_limits<uint32_t>::max() &&
            numVarints <= std::numeric_limits<uint32_t>::max(),
        "too many varint bytes in a batch");
    rmm::device_uvector<uint64_t> varints(numVarints, stream, mr);
    for (auto [leaf, offset] : varintLeaves) {
      leaf->kind = FastLeaf::kTrivial;
      leaf->width = 8;
      leaf->baseline = 0;
      leaf->data = reinterpret_cast<const uint8_t*>(varints.data() + offset);
    }
    rmm::device_uvector<uint64_t> deltaValues(walk.numDeltaValues, stream, mr);
    for (FastLeaf* leaf : deltaLeaves) {
      leaf->kind = FastLeaf::kTrivial;
      leaf->width = 8;
      leaf->data =
          reinterpret_cast<const uint8_t*>(deltaValues.data() + leaf->baseline);
      leaf->baseline = 0;
    }

    // Run lengths, string entry lengths and string bytes go first; the
    // run lengths and entry lengths are then scanned into run ends and
    // string offsets.
    rmm::device_uvector<uint32_t> runEnds(totalRuns, stream, mr);
    std::vector<std::unique_ptr<cudf::column>> offsets(types.size());
    std::vector<rmm::device_buffer> stringChars(types.size());
    std::vector<std::unique_ptr<cudf::column>> indices(types.size());
    std::vector<FastChunk> before;
    std::vector<FastChunk> main;
    // The FSST chunks, and every string chunk of the columns that have them
    // with the column it belongs to.
    NIMBLE_GPU_CHECK(
        numCodeLengths <= std::numeric_limits<uint32_t>::max(),
        "too many FSST strings in a batch");
    rmm::device_uvector<uint32_t> codeLengths(numCodeLengths, stream, mr);
    rmm::device_uvector<uint32_t> codeOffsets(numCodeLengths, stream, mr);
    std::vector<StringChunk> fsstChunks;
    std::vector<StringChunk> stringChunks;
    std::vector<size_t> stringChunkColumns;
    uint32_t codeBase = 0;
    uint32_t runBase = 0;
    uint32_t rowBase = 0;
    for (size_t i = 0; i < types.size(); ++i) {
      if (!fast[i]) {
        continue;
      }
      auto& column = fastColumns[i];
      const bool isString = types[i].id() == cudf::type_id::STRING;
      uint8_t* out = nullptr;
      int32_t* entryOffsets = nullptr;
      uint8_t* chars = nullptr;
      if (isString) {
        offsets[i] = cudf::make_fixed_width_column(
            cudf::data_type{cudf::type_id::INT32},
            column.numEntries + 1,
            cudf::mask_state::UNALLOCATED,
            stream,
            mr);
        entryOffsets = offsets[i]->mutable_view().head<int32_t>();
        checkCuda(cudaMemsetAsync(
            entryOffsets + column.numEntries, 0, sizeof(int32_t), stream.get()));
        if (!column.fsst) {
          stringChars[i] = rmm::device_buffer(column.numChars, stream, mr);
          chars = static_cast<uint8_t*>(stringChars[i].data());
        }
        if (!column.direct) {
          indices[i] = cudf::make_fixed_width_column(
              cudf::data_type{cudf::type_id::INT32},
              totalRows,
              cudf::mask_state::UNALLOCATED,
              stream,
              mr);
          out = indices[i]->mutable_view().head<uint8_t>();
        }
      } else {
        columns[i] = cudf::make_fixed_width_column(
            types[i], totalRows, cudf::mask_state::UNALLOCATED, stream, mr);
        out = columns[i]->mutable_view().head<uint8_t>();
      }
      size_t row = 0;
      uint32_t entryBase = 0;
      uint64_t charBase = 0;
      for (auto& shape : column.shapes) {
        auto chunk = shape.chunk;
        if (chunk.rle) {
          FastChunk lengths;
          lengths.values = shape.runLengths;
          lengths.out = reinterpret_cast<uint8_t*>(runEnds.data() + runBase);
          lengths.numRows = chunk.numRuns;
          lengths.outWidth = 4;
          before.push_back(lengths);
          chunk.runEnds = runEnds.data() + runBase;
          chunk.rowBase = rowBase;
          runBase += chunk.numRuns;
          rowBase += chunk.numRows;
        }
        if (isString && shape.fsst) {
          FastChunk lengths;
          lengths.values = shape.entryLengths;
          lengths.out = reinterpret_cast<uint8_t*>(codeLengths.data() + codeBase);
          lengths.numRows = shape.numEntries;
          lengths.outWidth = 4;
          before.push_back(lengths);
          StringChunk strings;
          strings.src = shape.chars;
          strings.codeLengths = codeLengths.data() + codeBase;
          strings.codeOffsets = codeOffsets.data() + codeBase;
          strings.entries = entryOffsets + entryBase;
          strings.numEntries = shape.numEntries;
          strings.table = shape.fsstTable;
          fsstChunks.push_back(strings);
          stringChunks.push_back(strings);
          stringChunkColumns.push_back(i);
          codeBase += shape.numEntries;
        } else if (isString) {
          FastChunk lengths;
          lengths.values = shape.entryLengths;
          lengths.out = reinterpret_cast<uint8_t*>(entryOffsets + entryBase);
          lengths.numRows = shape.numEntries;
          lengths.outWidth = 4;
          before.push_back(lengths);
          if (column.fsst) {
            StringChunk strings;
            strings.src = shape.chars;
            strings.entries = entryOffsets + entryBase;
            strings.numEntries = shape.numEntries;
            stringChunks.push_back(strings);
            stringChunkColumns.push_back(i);
          } else if (shape.charsSize > 0) {
            FastChunk bytes;
            bytes.values.kind = FastLeaf::kTrivial;
            bytes.values.width = 1;
            bytes.values.data = shape.chars;
            bytes.out = chars + charBase;
            bytes.numRows = static_cast<uint32_t>(shape.charsSize);
            bytes.outWidth = 1;
            before.push_back(bytes);
          }
        }
        if (isString) {
          chunk.add = entryBase;
          entryBase += shape.numEntries;
          charBase += shape.charsSize;
        }
        if (out != nullptr) {
          chunk.out = out + row * chunk.outWidth;
          main.push_back(chunk);
        }
        row += chunk.numRows;
      }
    }
    const auto varintLaunch = addBlocks(
        metadata,
        segments,
        [](const VarintSegment& segment) { return segment.numBytes; },
        kVarintBytesPerBlock);
    const auto deltaTasks = metadata.add(walk.deltaTasks);
    const auto fastRows = [](const FastChunk& chunk) { return chunk.numRows; };
    const auto beforeLaunch =
        addBlocks(metadata, before, fastRows, kFastRowsPerBlock);
    const auto mainLaunch = addBlocks(metadata, main, fastRows, kFastRowsPerBlock);
    const auto entryCount = [](const StringChunk& chunk) {
      return chunk.numEntries;
    };
    const auto fsstLaunch =
        addBlocks(metadata, fsstChunks, entryCount, kStringsPerBlock);
    const auto tablesOffset = metadata.add(walk.fsstTables);
    metadata.upload(stream, mr);
    const auto* tables = metadata.device<FsstTable>(tablesOffset);

    std::vector<size_t> fsstColumns;
    for (size_t i = 0; i < types.size(); ++i) {
      if (fast[i] && fastColumns[i].fsst) {
        fsstColumns.push_back(i);
      }
    }
    const size_t numResults = zstd.count + lz4.count;
    const size_t numCharsOffset = alignUp(numResults * sizeof(CodecResult));
    if (numResults > 0 || !fsstColumns.empty()) {
      readback.emplace(numCharsOffset + fsstColumns.size() * sizeof(int32_t));
      results = reinterpret_cast<CodecResult*>(readback->data());
    }
    launchDecompression(zstd, metadata, results, stream, mr);
    launchDecompression(lz4, metadata, results + zstd.count, stream, mr);

    if (varintLaunch.numBlocks > 0) {
      rmm::device_uvector<uint32_t> index(numVarintBytes, stream, mr);
      const auto* deviceSegments =
          metadata.device<VarintSegment>(varintLaunch.items);
      const auto* starts = metadata.device<uint32_t>(varintLaunch.starts);
      varintEndFlagsKernel<<<varintLaunch.numBlocks, kBlockSize, 0, stream.get()>>>(
          deviceSegments, starts, varintLaunch.count, index.data());
      scanInPlace(index.data(), numVarintBytes, false, stream, mr);
      varintValuesKernel<<<varintLaunch.numBlocks, kBlockSize, 0, stream.get()>>>(
          deviceSegments,
          starts,
          varintLaunch.count,
          index.data(),
          varints.data(),
          static_cast<uint32_t>(numVarints));
      checkCuda(cudaGetLastError());
    }
    if (!walk.deltaTasks.empty()) {
      NIMBLE_GPU_CHECK(
          walk.deltaTasks.size() <= std::numeric_limits<int32_t>::max(),
          "too many DeltaBlock blocks in a batch");
      deltaTasksKernel<<<
          static_cast<uint32_t>(walk.deltaTasks.size()),
          kBlockSize,
          0,
          stream.get()>>>(
          metadata.device<DeltaTask>(deltaTasks), deltaValues.data());
      checkCuda(cudaGetLastError());
    }
    auto launchFast = [&](const BlockLaunch& launch) {
      if (launch.numBlocks == 0) {
        return;
      }
      decodeFastChunksKernel<<<launch.numBlocks, kBlockSize, 0, stream.get()>>>(
          metadata.device<FastChunk>(launch.items),
          metadata.device<uint32_t>(launch.starts),
          launch.count);
      checkCuda(cudaGetLastError());
    };
    launchFast(beforeLaunch);
    scanInPlace(runEnds.data(), totalRuns, true, stream, mr);
    if (fsstLaunch.numBlocks > 0) {
      checkCuda(cudaMemcpyAsync(
          codeOffsets.data(),
          codeLengths.data(),
          numCodeLengths * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice,
          stream.get()));
      scanInPlace(codeOffsets.data(), numCodeLengths, false, stream, mr);
      fsstLengthsBatchKernel<<<fsstLaunch.numBlocks, kBlockSize, 0, stream.get()>>>(
          metadata.device<StringChunk>(fsstLaunch.items),
          metadata.device<uint32_t>(fsstLaunch.starts),
          fsstLaunch.count,
          tables);
      checkCuda(cudaGetLastError());
    }
    for (size_t i = 0; i < types.size(); ++i) {
      if (offsets[i]) {
        scanInPlace(
            offsets[i]->mutable_view().head<int32_t>(),
            fastColumns[i].numEntries + 1,
            false,
            stream,
            mr);
      }
    }
    launchFast(mainLaunch);

    // One wait for the string sizes of every FSST column.
    if (!fsstColumns.empty()) {
      auto* numChars =
          reinterpret_cast<int32_t*>(readback->data() + numCharsOffset);
      for (size_t k = 0; k < fsstColumns.size(); ++k) {
        const auto i = fsstColumns[k];
        checkCuda(cudaMemcpyAsync(
            numChars + k,
            offsets[i]->view().head<int32_t>() + fastColumns[i].numEntries,
            sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream.get()));
      }
      checkCuda(cudaStreamSynchronize(stream.get()));
      checkDecompressed();
      for (size_t k = 0; k < fsstColumns.size(); ++k) {
        NIMBLE_GPU_CHECK(numChars[k] >= 0, "too many string bytes in a column");
        stringChars[fsstColumns[k]] =
            rmm::device_buffer(numChars[k], stream, mr);
      }
      for (size_t c = 0; c < stringChunks.size(); ++c) {
        stringChunks[c].chars =
            static_cast<char*>(stringChars[stringChunkColumns[c]].data());
      }
      const auto charsLaunch =
          addBlocks(charsMetadata, stringChunks, entryCount, kStringsPerBlock);
      charsMetadata.upload(stream, mr);
      if (charsLaunch.numBlocks > 0) {
        stringCharsKernel<<<charsLaunch.numBlocks, kBlockSize, 0, stream.get()>>>(
            charsMetadata.device<StringChunk>(charsLaunch.items),
            charsMetadata.device<uint32_t>(charsLaunch.starts),
            charsLaunch.count,
            tables);
        checkCuda(cudaGetLastError());
      }
    }

    for (size_t i = 0; i < types.size(); ++i) {
      if (!offsets[i]) {
        continue;
      }
      auto entries = cudf::make_strings_column(
          fastColumns[i].numEntries,
          std::move(offsets[i]),
          std::move(stringChars[i]),
          0,
          cuda::device_buffer<std::byte>(stream, mr, 0, cuda::no_init));
      if (fastColumns[i].direct) {
        columns[i] = std::move(entries);
        continue;
      }
      auto gathered = cudf::gather(
          cudf::table_view{{entries->view()}},
          indices[i]->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          cudf::memory_resources{mr, mr});
      columns[i] = std::move(gathered->release()[0]);
    }

    for (size_t i = 0; i < types.size(); ++i) {
      if (fast[i]) {
        continue;
      }
      std::vector<std::unique_ptr<cudf::column>> parts;
      parts.reserve(stripes.size());
      for (const auto& stripe : stripes) {
        parts.push_back(
            walk.decodeStream(stripe.streams[i], types[i], stripe.numRows));
      }
      if (parts.size() == 1) {
        columns[i] = std::move(parts[0]);
        continue;
      }
      std::vector<cudf::column_view> views;
      views.reserve(parts.size());
      for (const auto& part : parts) {
        views.push_back(part->view());
      }
      columns[i] = cudf::concatenate(views, stream, mr);
    }
  }
  checkCuda(cudaGetLastError());
  checkCuda(cudaStreamSynchronize(stream.get()));
  checkDecompressed();
  return columns;
}

} // namespace facebook::velox::cudf_velox::connector::hive
