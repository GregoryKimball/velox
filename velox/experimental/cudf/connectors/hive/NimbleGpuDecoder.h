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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// The scalar streams of one Nimble stripe, one per output column.
struct NimbleStripeStreams {
  std::vector<std::span<const uint8_t>> streams;
  cudf::size_type numRows;
};

/// Decodes the scalar streams of a batch of Nimble stripes on the GPU.
///
/// Encoding headers are not compressed, so the host walks each stream's
/// chunks and encoding trees directly from the file bytes. Only the leaf
/// payloads go to the device: they are staged into one pinned buffer, copied
/// in one transfer, decompressed in one batch per codec, and then decoded by
/// CUDA kernels and cuDF primitives. Batching stripes gives the decompressor
/// more frames to spread across the GPU.
///
/// Supported encodings: Trivial, FixedBitWidth, Constant, Varint, Dictionary,
/// RLE, MainlyConstant, Nullable and SparseBool, with Zstd or LZ4 payloads.
/// Anything else throws.
class NimbleGpuDecoder {
 public:
  NimbleGpuDecoder() = default;
  ~NimbleGpuDecoder();

  NimbleGpuDecoder(const NimbleGpuDecoder&) = delete;
  NimbleGpuDecoder& operator=(const NimbleGpuDecoder&) = delete;

  /// Decodes column i of every stripe as types[i] and concatenates the stripes
  /// in order. An empty stream decodes to all nulls. `compactRowCount` is the
  /// file's compactRowCountEncoding property.
  std::vector<std::unique_ptr<cudf::column>> decode(
      std::span<const NimbleStripeStreams> stripes,
      std::span<const cudf::data_type> types,
      bool compactRowCount,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

 private:
  // Pinned staging buffer for the payloads, reused across stripes.
  uint8_t* pinned_{nullptr};
  size_t pinnedSize_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
