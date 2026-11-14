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
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

// Declared here because this header is compiled by nvcc, which cannot compile
// velox/type/Type.h.
namespace facebook::velox {
class RowType;
} // namespace facebook::velox

namespace facebook::velox::connector::hive {
class HivePartitionFunctionSpec;
} // namespace facebook::velox::connector::hive

namespace facebook::velox::cudf_velox {

/// Assigns rows to partitions on the GPU exactly as
/// connector::hive::HivePartitionFunction does on the CPU, so that a stage that
/// reads a bucketed table finds each row in the partition that owns its
/// bucket. A row's hash folds the Hive hash of each key column in order as
/// 31 * hash + keyHash, with a null key hashing to 0. Its bucket is
/// (hash & INT32_MAX) % numBuckets, and its partition is
/// bucketToPartition[bucket], or the bucket when 'bucketToPartition' is empty.
class CudfHivePartitionFunction {
 public:
  CudfHivePartitionFunction(
      int numBuckets,
      std::vector<int> bucketToPartition,
      std::vector<cudf::size_type> keyColumns);

  /// Returns whether a key column of 'type' can be hashed on the GPU: the
  /// integer, boolean, floating point, DATE (TIMESTAMP_DAYS), short decimal
  /// (DECIMAL64) and string types.
  static bool supports(cudf::data_type type);

  /// Returns whether the GPU can compute the partitions of 'spec' over rows of
  /// 'inputType': it has no constant keys and every key can be hashed.
  static bool supports(
      const velox::connector::hive::HivePartitionFunctionSpec& spec,
      const std::shared_ptr<const RowType>& inputType);

  /// Returns the GPU form of the function 'spec' creates for 'numPartitions'
  /// partitions of a remote or a local exchange
  /// (HivePartitionFunctionSpec::create()), for an operator that partitions
  /// rows of 'outputType'. The keys of 'spec', channels of 'inputType', are
  /// located in 'outputType' by name. 'spec' must be supported.
  static CudfHivePartitionFunction fromSpec(
      const velox::connector::hive::HivePartitionFunctionSpec& spec,
      int numPartitions,
      bool localExchange,
      const std::shared_ptr<const RowType>& inputType,
      const std::shared_ptr<const RowType>& outputType);

  /// Returns the partition of each row of 'input' as an INT32 column without
  /// nulls. Every key column must be supported.
  std::unique_ptr<cudf::column> partitions(
      cudf::table_view input,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

 private:
  const int numBuckets_;
  const std::vector<int> bucketToPartition_;
  const std::vector<cudf::size_type> keyColumns_;
};

} // namespace facebook::velox::cudf_velox
