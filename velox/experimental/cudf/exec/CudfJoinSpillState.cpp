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
#include "velox/experimental/cudf/exec/CudfJoinSpillState.h"
#include "velox/experimental/cudf/exec/CudfSpillContext.h"

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {
CudfJoinSpillState::Chunk::Chunk(CudfVectorPtr input)
    : resident(std::move(input)) {
  VELOX_CHECK_NOT_NULL(resident);
  bytes = resident->estimateFlatSize();
  rows = resident->size();
}
CudfVectorPtr CudfJoinSpillState::Chunk::take(
    rmm::device_async_resource_ref mr) {
  if (!resident) {
    VELOX_CHECK(spilled.has_value());
    resident = spilled->restore(mr);
    if (stats) {
      stats->restoredBytes.fetch_add(spilled->deviceBytes(),
                                     std::memory_order_relaxed);
    }
  }
  spilled.reset();
  return std::move(resident);
}
uint64_t CudfJoinSpillState::Chunk::park(
    const std::shared_ptr<CudfSpillContext>& context) {
  if (!resident) {
    return 0;
  }
  VELOX_CHECK_EQ(resident.use_count(), 1);
  auto replacement = SpilledCudfVector::spill(resident, context);
  const auto released = replacement.deviceBytes();
  if (stats) {
    stats->spilledBytes.fetch_add(released, std::memory_order_relaxed);
    stats->hostBytesWritten.fetch_add(replacement.hostBytes(),
                                      std::memory_order_relaxed);
  }
  spilled.emplace(std::move(replacement));
  return released;
}
CudfJoinSpillState::CudfJoinSpillState(
    std::shared_ptr<CudfSpillContext> context,
    std::shared_ptr<CudfJoinSpillStats> stats)
    : context_(std::move(context)),
      stats_(stats ? std::move(stats)
                   : std::make_shared<CudfJoinSpillStats>()) {
  VELOX_CHECK_NOT_NULL(context_);
}
void CudfJoinSpillState::append(CudfVectorPtr input) {
  if (input && input->size() != 0) {
    append(Chunk(std::move(input)));
  }
}
void CudfJoinSpillState::append(Chunk chunk) {
  if (!chunk.stats) {
    chunk.stats = stats_;
  }
  // Keep new input on host after reclamation instead of reconstructing an
  // ever-growing resident relation on each addInput.
  if (spilled_) {
    chunk.park(context_);
  }
  spilled_ |= chunk.spilled.has_value();
  chunks_.push_back(std::move(chunk));
}
void CudfJoinSpillState::append(CudfJoinSpillState&& other) {
  spilled_ |= other.spilled_;
  while (!other.empty()) {
    append(other.takeFront());
  }
}
void CudfJoinSpillState::adopt(CudfJoinSpillState&& other) {
  spilled_ |= other.spilled_;
  while (!other.empty()) {
    auto chunk = other.takeFront();
    if (!chunk.stats) {
      chunk.stats = stats_;
    }
    spilled_ |= chunk.spilled.has_value();
    chunks_.push_back(std::move(chunk));
  }
}
uint64_t CudfJoinSpillState::reclaimableBytes() const {
  uint64_t bytes = 0;
  for (const auto& chunk : chunks_) {
    if (chunk.resident && chunk.resident.use_count() == 1 &&
        chunk.resident->hasOwnedTable()) {
      bytes += chunk.bytes;
    }
  }
  return bytes;
}
uint64_t CudfJoinSpillState::spill(uint64_t targetBytes) {
  uint64_t released = 0;
  for (auto& chunk : chunks_) {
    if (chunk.resident && chunk.resident.use_count() == 1 &&
        chunk.resident->hasOwnedTable()) {
      released += chunk.park(context_);
      spilled_ = true;
    }
    if (targetBytes != 0 && released >= targetBytes) {
      break;
    }
  }
  return released;
}
bool CudfJoinSpillState::hasSpilled() const {
  return spilled_;
}
bool CudfJoinSpillState::empty() const {
  return chunks_.empty();
}
uint64_t CudfJoinSpillState::rows() const {
  uint64_t rows = 0;
  for (const auto& chunk : chunks_) {
    rows += chunk.rows;
  }
  return rows;
}
uint64_t CudfJoinSpillState::bytes() const {
  uint64_t bytes = 0;
  for (const auto& chunk : chunks_) {
    bytes += chunk.bytes;
  }
  return bytes;
}
uint64_t CudfJoinSpillState::largestChunkBytes() const {
  uint64_t result = 0;
  for (const auto& chunk : chunks_) {
    result = std::max(result, chunk.bytes);
  }
  return result;
}
uint64_t CudfJoinSpillState::largestChunkRows() const {
  uint64_t result = 0;
  for (const auto& chunk : chunks_) {
    result = std::max(result, chunk.rows);
  }
  return result;
}
size_t CudfJoinSpillState::size() const {
  return chunks_.size();
}
void CudfJoinSpillState::clear() {
  chunks_.clear();
  spilled_ = false;
}
CudfJoinSpillState::Chunk CudfJoinSpillState::takeFront() {
  VELOX_CHECK(!empty());
  auto chunk = std::move(chunks_.front());
  chunks_.pop_front();
  return chunk;
}
std::vector<CudfJoinSpillState::Chunk> CudfJoinSpillState::takeChunks() {
  std::vector<Chunk> chunks;
  chunks.reserve(size());
  while (!empty()) {
    chunks.push_back(takeFront());
  }
  return chunks;
}
std::array<CudfVectorPtr, 2> CudfJoinSpillState::partition(
    CudfVectorPtr input,
    const std::vector<cudf::size_type>& keys,
    uint32_t depth,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK(!keys.empty());
  const auto stream = input->stream();
  auto* pool = input->pool();
  auto type = input->type();
  const uint32_t seed = cudf::DEFAULT_HASH_SEED + 0x9e3779b9U * (depth + 1);
  auto [table, offsets] = cudf::hash_partition(input->getTableView(),
                                               keys,
                                               2,
                                               cudf::hash_id::HASH_MURMUR3,
                                               seed,
                                               stream,
                                               mr);
  VELOX_CHECK_EQ(offsets.size(), 3);
  stream.sync();
  input.reset();
  auto views = cudf::split(table->view(), {offsets[1]}, stream);
  std::array<CudfVectorPtr, 2> result;
  for (size_t i = 0; i < result.size(); ++i) {
    if (views[i].num_rows() != 0) {
      auto owned = std::make_unique<cudf::table>(views[i], stream, mr);
      stream.sync();
      result[i] = std::make_shared<CudfVector>(
          pool, type, views[i].num_rows(), std::move(owned), stream);
    }
  }
  return result;
}
} // namespace facebook::velox::cudf_velox
