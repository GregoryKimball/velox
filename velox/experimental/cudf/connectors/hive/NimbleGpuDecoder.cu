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

#include <cub/device/device_scan.cuh>
#include <cuda/buffer>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cstring>
#include <limits>
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

 private:
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

// Decompresses every payload of one codec in one batch.
void decompressPayloads(
    std::vector<Payload>& payloads,
    Codec codec,
    const uint8_t* staging,
    uint8_t* output,
    std::span<const size_t> outputOffsets,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::device_span<uint8_t const>> inputs;
  std::vector<cudf::device_span<uint8_t>> outputs;
  std::vector<size_t> expected;
  size_t maxSize = 0;
  size_t totalSize = 0;
  for (size_t i = 0; i < payloads.size(); ++i) {
    auto& payload = payloads[i];
    if (payload.codec != codec) {
      continue;
    }
    inputs.emplace_back(staging + payload.stagingOffset, payload.stored.size());
    outputs.emplace_back(output + outputOffsets[i], payload.size);
    payload.device = {output + outputOffsets[i], payload.size};
    expected.push_back(payload.size);
    maxSize = std::max(maxSize, payload.size);
    totalSize += payload.size;
  }
  if (inputs.empty()) {
    return;
  }
  nvtx3::scoped_range range{"NimbleGpuDecoder::decompress"};
  const auto count = inputs.size();
  rmm::device_uvector<cudf::device_span<uint8_t const>> deviceInputs(
      count, stream, mr);
  rmm::device_uvector<cudf::device_span<uint8_t>> deviceOutputs(
      count, stream, mr);
  rmm::device_uvector<cudf::io::detail::codec_exec_result> results(
      count, stream, mr);
  checkCuda(cudaMemcpyAsync(
      deviceInputs.data(),
      inputs.data(),
      count * sizeof(inputs[0]),
      cudaMemcpyHostToDevice,
      stream.get()));
  checkCuda(cudaMemcpyAsync(
      deviceOutputs.data(),
      outputs.data(),
      count * sizeof(outputs[0]),
      cudaMemcpyHostToDevice,
      stream.get()));
  cudf::io::detail::decompress(
      codec == Codec::kZstd ? cudf::io::compression_type::ZSTD
                            : cudf::io::compression_type::LZ4,
      cudf::device_span<cudf::device_span<uint8_t const> const>{
          deviceInputs.data(), count},
      cudf::device_span<cudf::device_span<uint8_t> const>{
          deviceOutputs.data(), count},
      cudf::device_span<cudf::io::detail::codec_exec_result>{
          results.data(), count},
      maxSize,
      totalSize,
      stream,
      cudf::memory_resources{mr, mr});
  std::vector<cudf::io::detail::codec_exec_result> hostResults(count);
  checkCuda(cudaMemcpyAsync(
      hostResults.data(),
      results.data(),
      count * sizeof(hostResults[0]),
      cudaMemcpyDeviceToHost,
      stream.get()));
  checkCuda(cudaStreamSynchronize(stream.get()));
  for (size_t i = 0; i < count; ++i) {
    NIMBLE_GPU_CHECK(
        hostResults[i].status == cudf::io::detail::codec_status::SUCCESS &&
            hostResults[i].bytes_written == expected[i],
        "decompression failed for payload " + std::to_string(i) + " of " +
            std::to_string(count) + ": status " +
            std::to_string(static_cast<int>(hostResults[i].status)) +
            ", wrote " + std::to_string(hostResults[i].bytes_written) +
            " of " + std::to_string(expected[i]) + " bytes from " +
            std::to_string(inputs[i].size()) + " compressed, max " +
            std::to_string(maxSize) + ", total " + std::to_string(totalSize));
  }
}

} // namespace

NimbleGpuDecoder::~NimbleGpuDecoder() {
  if (pinned_ != nullptr) {
    cudaFreeHost(pinned_);
  }
}

std::vector<std::unique_ptr<cudf::column>> NimbleGpuDecoder::decode(
    std::span<const std::span<const uint8_t>> streams,
    std::span<const cudf::data_type> types,
    cudf::size_type numRows,
    bool compactRowCount,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  NIMBLE_GPU_CHECK(streams.size() == types.size(), "one type per stream");
  StripeDecode walk{compactRowCount, stream, mr};
  {
    nvtx3::scoped_range range{"NimbleGpuDecoder::plan"};
    for (size_t i = 0; i < streams.size(); ++i) {
      walk.decodeStream(streams[i], types[i], numRows);
    }
  }

  // Stage the stored payload bytes, each 16-byte aligned, and copy them over
  // in one transfer. Decompressed payloads get their own aligned slots.
  auto& payloads = walk.payloads;
  size_t stagingSize = 0;
  size_t outputSize = 0;
  std::vector<size_t> outputOffsets(payloads.size());
  for (size_t i = 0; i < payloads.size(); ++i) {
    payloads[i].stagingOffset = stagingSize;
    // The padding covers 8-byte loads past the end of bit-packed data.
    stagingSize += alignUp(payloads[i].stored.size() + 8);
    if (payloads[i].codec != Codec::kUncompressed) {
      outputOffsets[i] = outputSize;
      outputSize += alignUp(payloads[i].size + 8);
    }
  }
  // The previous stripe's transfer from the pinned buffer has completed:
  // decode() synchronizes before returning.
  if (stagingSize > pinnedSize_) {
    if (pinned_ != nullptr) {
      checkCuda(cudaFreeHost(pinned_));
      pinned_ = nullptr;
    }
    pinnedSize_ = std::max(stagingSize, 2 * pinnedSize_);
    checkCuda(cudaMallocHost(&pinned_, pinnedSize_));
  }
  rmm::device_buffer staging(stagingSize, stream, mr);
  rmm::device_buffer output(outputSize, stream, mr);
  {
    nvtx3::scoped_range range{"NimbleGpuDecoder::stage"};
    for (const auto& payload : payloads) {
      std::memcpy(
          pinned_ + payload.stagingOffset,
          payload.stored.data(),
          payload.stored.size());
    }
    if (stagingSize > 0) {
      checkCuda(cudaMemcpyAsync(
          staging.data(),
          pinned_,
          stagingSize,
          cudaMemcpyHostToDevice,
          stream.get()));
    }
  }
  auto* stagingData = static_cast<const uint8_t*>(staging.data());
  for (auto& payload : payloads) {
    if (payload.codec == Codec::kUncompressed) {
      payload.device = {stagingData + payload.stagingOffset, payload.size};
    }
  }
  auto* outputData = static_cast<uint8_t*>(output.data());
  decompressPayloads(
      payloads, Codec::kZstd, stagingData, outputData, outputOffsets, stream, mr);
  decompressPayloads(
      payloads, Codec::kLz4, stagingData, outputData, outputOffsets, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(streams.size());
  walk.collecting = false;
  {
    nvtx3::scoped_range range{"NimbleGpuDecoder::decode"};
    for (size_t i = 0; i < streams.size(); ++i) {
      columns.push_back(walk.decodeStream(streams[i], types[i], numRows));
    }
  }
  checkCuda(cudaGetLastError());
  checkCuda(cudaStreamSynchronize(stream.get()));
  return columns;
}

} // namespace facebook::velox::cudf_velox::connector::hive
