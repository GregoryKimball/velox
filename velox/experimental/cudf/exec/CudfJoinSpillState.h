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
#include "velox/experimental/cudf/exec/SpilledCudfVector.h"

#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {
struct CudfJoinSpillStats {
  std::atomic<uint64_t> spilledBytes{0};
  std::atomic<uint64_t> restoredBytes{0};
  std::atomic<uint64_t> hostBytesWritten{0};
};

// Data mechanics independent of task pause, driver lifetime and victim policy.
// Mutations require exclusive access. Detach active chunks before a cuDF call;
// only the inactive chunks retained here may be reclaimed while it is parked.
class CudfJoinSpillState {
 public:
  struct Chunk {
    CudfVectorPtr resident;
    std::shared_ptr<CudfJoinSpillStats> stats;
    std::optional<SpilledCudfVector> spilled;
    uint64_t bytes{0};
    uint64_t rows{0};
    explicit Chunk(CudfVectorPtr input);
    Chunk(Chunk&&) noexcept = default;
    Chunk& operator=(Chunk&&) noexcept = default;
    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;
    // Keep the host copy until allocation and transfer have succeeded.
    CudfVectorPtr take(rmm::device_async_resource_ref mr);
    uint64_t park(const std::shared_ptr<CudfSpillContext>& context);
  };
  explicit CudfJoinSpillState(
      std::shared_ptr<CudfSpillContext> context,
      std::shared_ptr<CudfJoinSpillStats> stats = nullptr);
  void append(CudfVectorPtr input);
  void append(CudfJoinSpillState&& other);
  void append(Chunk chunk);
  // Takes over the chunks of 'other' as they are: a resident chunk stays
  // resident even when either state has spilled. For the inactive state of a
  // finished peer, whose chunks are already on the device and charged, so
  // that moving them here adds no device memory. append() keeps its policy of
  // parking new input once the state has spilled. The chunks keep their own
  // counters, and hasSpilled() is true afterwards if it was for either state.
  void adopt(CudfJoinSpillState&& other);
  uint64_t reclaimableBytes() const;
  uint64_t spill(uint64_t targetBytes);
  bool hasSpilled() const;
  bool empty() const;
  uint64_t rows() const;
  uint64_t bytes() const;
  uint64_t largestChunkBytes() const;
  uint64_t largestChunkRows() const;
  size_t size() const;
  void clear();
  // Detaching a chunk pins it against reclamation through this state.
  Chunk takeFront();
  std::vector<Chunk> takeChunks();
  // Same depth-dependent hash seed on build and probe. One bounded input
  // chunk at a time; returned vectors own their exact device buffers.
  static std::array<CudfVectorPtr, 2> partition(
      CudfVectorPtr input,
      const std::vector<cudf::size_type>& keys,
      uint32_t depth,
      rmm::device_async_resource_ref mr);

 private:
  std::shared_ptr<CudfSpillContext> context_;
  std::shared_ptr<CudfJoinSpillStats> stats_;
  std::deque<Chunk> chunks_;
  bool spilled_{false};
};
} // namespace facebook::velox::cudf_velox
