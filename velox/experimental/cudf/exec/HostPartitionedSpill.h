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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Holds the input of a final aggregation in host memory, hash-partitioned on
/// the grouping keys. All rows of a group land in the same partition, so
/// aggregating each partition on its own gives the same result as aggregating
/// all of the input, while the GPU holds one partition at a time.
class HostPartitionedSpill {
 public:
  HostPartitionedSpill(std::vector<cudf::size_type> keys, int32_t numPartitions);

  /// Hash-partitions 'table' on the keys and copies every partition to host
  /// memory. Returns once the copies are complete, so the caller may free
  /// 'table'.
  void spill(
      cudf::table_view table,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  int32_t numPartitions() const {
    return static_cast<int32_t>(partitions_.size());
  }

  /// Host bytes currently held.
  uint64_t bytes() const {
    return bytes_;
  }

  /// Copies 'partition' back to the GPU as one table and releases its host
  /// memory. Returns nullptr when the partition holds no rows.
  std::unique_ptr<cudf::table> read(
      int32_t partition,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

 private:
  struct Chunk {
    std::vector<uint8_t> metadata;
    std::unique_ptr<uint8_t[]> data;
    size_t size;
  };

  const std::vector<cudf::size_type> keys_;
  std::vector<std::vector<Chunk>> partitions_;
  uint64_t bytes_{0};
};

} // namespace facebook::velox::cudf_velox
