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
#include "velox/experimental/cudf/exec/CudfHivePartitionFunction.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/iterator>

#include <cstdint>
#include <optional>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

// HiveHash::hashInt64: the low 32 bits of the value's upper half XOR the value.
__device__ uint32_t hashInt64(int64_t value) {
  const auto bits = static_cast<uint64_t>(value);
  return static_cast<uint32_t>((bits >> 32) ^ bits);
}

// HiveHash::hashBytes: 31 * hash + byte over the bytes as signed values.
__device__ uint32_t hashBytes(cudf::string_view value) {
  uint32_t hash = 0;
  const char* data = value.data();
  for (cudf::size_type i = 0; i < value.size_bytes(); ++i) {
    hash = hash * 31 + static_cast<uint32_t>(static_cast<int8_t>(data[i]));
  }
  return hash;
}

// Folds the Hive hash of one key column into 'hashes', as HiveHash::
// hashPrimitive does for the Velox type the cuDF type comes from.
struct FoldKeyHash {
  cudf::column_device_view column;
  uint32_t* hashes;
  bool mix;

  __device__ uint32_t keyHash(cudf::size_type row) const {
    switch (column.type().id()) {
      case cudf::type_id::BOOL8:
        return column.element<bool>(row) ? 1 : 0;
      case cudf::type_id::INT8:
        return static_cast<uint32_t>(column.element<int8_t>(row));
      case cudf::type_id::INT16:
        return static_cast<uint32_t>(column.element<int16_t>(row));
      case cudf::type_id::INT32:
        return static_cast<uint32_t>(column.element<int32_t>(row));
      case cudf::type_id::TIMESTAMP_DAYS:
        return static_cast<uint32_t>(
            column.element<cudf::timestamp_D>(row).time_since_epoch().count());
      case cudf::type_id::INT64:
        return hashInt64(column.element<int64_t>(row));
      case cudf::type_id::DECIMAL64:
        // Velox stores a short decimal as a BIGINT and hashes it like one.
        return hashInt64(column.data<int64_t>()[row]);
      case cudf::type_id::FLOAT32:
        return static_cast<uint32_t>(__float_as_int(column.element<float>(row)));
      case cudf::type_id::FLOAT64:
        return hashInt64(__double_as_longlong(column.element<double>(row)));
      case cudf::type_id::STRING:
        return hashBytes(column.element<cudf::string_view>(row));
      default:
        // Excluded by CudfHivePartitionFunction::supports().
        return 0;
    }
  }

  __device__ void operator()(cudf::size_type row) const {
    const uint32_t hash = column.is_null(row) ? 0 : keyHash(row);
    hashes[row] = mix ? hashes[row] * 31 + hash : hash;
  }
};

struct ToPartition {
  const uint32_t* hashes;
  int32_t* partitions;
  int32_t numBuckets;
  // Null when buckets are the partitions.
  const int32_t* bucketToPartition;

  __device__ void operator()(cudf::size_type row) const {
    const auto bucket = static_cast<int32_t>(
        (hashes[row] & 0x7FFF'FFFFu) % static_cast<uint32_t>(numBuckets));
    partitions[row] =
        bucketToPartition != nullptr ? bucketToPartition[bucket] : bucket;
  }
};

template <typename Op>
void forEachRow(cudf::size_type numRows, Op op, cuda::stream_ref stream) {
  if (numRows == 0) {
    return;
  }
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0}, numRows, op, stream.get());
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace

CudfHivePartitionFunction::CudfHivePartitionFunction(
    int numBuckets,
    std::vector<int> bucketToPartition,
    std::vector<cudf::size_type> keyColumns)
    : numBuckets_(numBuckets),
      bucketToPartition_(std::move(bucketToPartition)),
      keyColumns_(std::move(keyColumns)) {
  CUDF_EXPECTS(numBuckets_ > 0, "The bucket count must be positive");
  CUDF_EXPECTS(
      bucketToPartition_.empty() ||
          bucketToPartition_.size() == static_cast<size_t>(numBuckets_),
      "The bucket to partition map must have one entry per bucket");
  CUDF_EXPECTS(!keyColumns_.empty(), "Hive partitioning needs a key column");
}

bool CudfHivePartitionFunction::supports(cudf::data_type type) {
  switch (type.id()) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::STRING:
      return true;
    default:
      return false;
  }
}

std::unique_ptr<cudf::column> CudfHivePartitionFunction::partitions(
    cudf::table_view input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  const auto numRows = input.num_rows();
  rmm::device_uvector<uint32_t> hashes(numRows, stream);
  for (size_t i = 0; i < keyColumns_.size(); ++i) {
    const auto& column = input.column(keyColumns_[i]);
    CUDF_EXPECTS(
        supports(column.type()),
        "Hive partitioning on the GPU does not support this key type");
    auto deviceColumn = cudf::column_device_view::create(column, stream);
    forEachRow(
        numRows,
        FoldKeyHash{*deviceColumn, hashes.data(), /*mix=*/i > 0},
        stream);
  }

  std::optional<rmm::device_uvector<int32_t>> bucketToPartition;
  if (!bucketToPartition_.empty()) {
    bucketToPartition.emplace(bucketToPartition_.size(), stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        bucketToPartition->data(),
        bucketToPartition_.data(),
        bucketToPartition_.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.get()));
  }

  auto result = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  forEachRow(
      numRows,
      ToPartition{
          hashes.data(),
          result->mutable_view().data<int32_t>(),
          numBuckets_,
          bucketToPartition ? bucketToPartition->data() : nullptr},
      stream);
  return result;
}

} // namespace facebook::velox::cudf_velox
