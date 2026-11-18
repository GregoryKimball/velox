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

#include "velox/experimental/cudf/exec/HostPartitionedSpill.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {
namespace {

// The rows reaching one operator were already hash-partitioned on these keys
// by the exchange and local partitioning; a different seed keeps them spread
// over all spill partitions.
constexpr uint32_t kSpillHashSeed = 0x9e3779b9;

} // namespace

HostPartitionedSpill::HostPartitionedSpill(
    std::vector<cudf::size_type> keys,
    int32_t numPartitions)
    : keys_(std::move(keys)), partitions_(numPartitions) {
  VELOX_CHECK_GT(numPartitions, 0);
  VELOX_CHECK(!keys_.empty());
}

void HostPartitionedSpill::spill(
    cudf::table_view table,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (table.num_rows() == 0) {
    return;
  }
  auto [partitioned, offsets] = cudf::hash_partition(
      table,
      keys_,
      numPartitions(),
      cudf::hash_id::HASH_MURMUR3,
      kSpillHashSeed,
      stream,
      mr);
  std::vector<cudf::size_type> splits(offsets.begin() + 1, offsets.end());
  auto packed = cudf::contiguous_split(partitioned->view(), splits, stream, mr);
  partitioned.reset();
  VELOX_CHECK_EQ(packed.size(), partitions_.size());

  for (size_t i = 0; i < packed.size(); ++i) {
    if (packed[i].table.num_rows() == 0) {
      continue;
    }
    auto& columns = packed[i].data;
    Chunk chunk;
    chunk.metadata = std::move(*columns.metadata);
    chunk.size = columns.gpu_data->size();
    chunk.data = std::make_unique_for_overwrite<uint8_t[]>(chunk.size);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        chunk.data.get(),
        columns.gpu_data->data(),
        chunk.size,
        cudaMemcpyDeviceToHost,
        stream.get()));
    bytes_ += chunk.size;
    partitions_[i].push_back(std::move(chunk));
  }
  stream.sync();
}

std::unique_ptr<cudf::table> HostPartitionedSpill::read(
    int32_t partition,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto chunks = std::exchange(partitions_.at(partition), {});
  if (chunks.empty()) {
    return nullptr;
  }
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::table_view> views;
  buffers.reserve(chunks.size());
  views.reserve(chunks.size());
  for (auto& chunk : chunks) {
    buffers.emplace_back(chunk.data.get(), chunk.size, stream, mr);
    views.push_back(cudf::unpack(
        chunk.metadata.data(),
        static_cast<const uint8_t*>(buffers.back().data())));
    bytes_ -= chunk.size;
  }
  auto table = views.size() == 1
      ? std::make_unique<cudf::table>(views[0], stream, mr)
      : cudf::concatenate(views, stream, mr);
  // The host chunks are freed on return; the copies must be done by then.
  stream.sync();
  return table;
}

} // namespace facebook::velox::cudf_velox
