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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {
struct CudfSpillContext;

// Input may already be hash-partitioned by remote and local exchanges. Start
// buffered-state partitioning in a separate hash domain so those routing bits
// do not restrict which spill children can receive rows. In particular, avoid
// both DEFAULT_HASH_SEED (remote) and 0x9e3779b9U (local exchange).
inline constexpr uint32_t kBufferedStateHashSeed = 0x85ebca6bU;

struct PartitionSpec {
  int32_t numPartitions;
  std::vector<cudf::size_type> keyIndices;
  cudf::hash_id hashId{cudf::hash_id::HASH_MURMUR3};
  uint32_t seed{kBufferedStateHashSeed};
};

enum class InputChunkStorage : uint8_t {
  // The chunk is a view whose storage is retained by 'owner' or 'tableOwner'.
  kBorrowed,

  // 'owner' describes the complete table referenced by 'view'. The owner can
  // still be shared temporarily while routing the chunk, but a persistent leaf
  // must materialize if it is not the sole owner.
  kOwned,
};

struct InputChunk {
  InputChunk() : stream(cuda::stream_ref{cudaStream_t{cudaStreamDefault}}) {}

  InputChunk(memory::MemoryPool* pool,
             TypePtr type,
             cudf::table_view view,
             cuda::stream_ref stream,
             CudfVectorPtr owner,
             std::shared_ptr<cudf::table> tableOwner,
             InputChunkStorage storage)
      : pool(pool),
        type(std::move(type)),
        view(view),
        stream(stream),
        owner(std::move(owner)),
        tableOwner(std::move(tableOwner)),
        storage(storage) {}

  InputChunk(const InputChunk&) = delete;
  InputChunk& operator=(const InputChunk&) = delete;
  InputChunk(InputChunk&&) = default;
  InputChunk& operator=(InputChunk&&) = default;

  memory::MemoryPool* pool{nullptr};
  TypePtr type;
  cudf::table_view view;
  cuda::stream_ref stream;
  CudfVectorPtr owner;
  std::shared_ptr<cudf::table> tableOwner;
  InputChunkStorage storage{InputChunkStorage::kBorrowed};

  size_t size() const {
    return static_cast<size_t>(view.num_rows());
  }

  bool empty() const {
    return size() == 0;
  }

  // Returns true when 'view' is the complete table view of 'owner'.
  bool ownsFullTable() const;

  // Consumes this chunk and returns one backed by an independently owned
  // CudfVector. Reuses an already complete, unique owner; otherwise copies the
  // exact view through 'mr'.
  InputChunk materialize(rmm::device_async_resource_ref mr) &&;
};

// Serve as the opaque base type for strategy-owned leaf state.
class BufferedState {
 public:
  virtual ~BufferedState() = default;
};

class BufferedStateOps {
 public:
  virtual ~BufferedStateOps() = default;


  // Convert one raw input batch into a prepared chunk that PBS can route.
  virtual InputChunk prepareInput(CudfVectorPtr rawInput) = 0;

  // Return a cheap, conservative upper bound on the row count after adding
  // `input` to `leaf`.
  virtual size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const = 0;

  // Create a new leaf from the first chunk routed to it.
  virtual std::unique_ptr<BufferedState> createLeaf(InputChunk input) = 0;

  // Create one leaf directly from two transient chunks routed to the same
  // child. Strategies should override this when they can merge the chunks
  // without first materializing either one. The default preserves the old
  // create-then-add behavior.
  virtual std::unique_ptr<BufferedState> createLeafFromInputs(
      InputChunk first,
      InputChunk second) {
    auto leaf = createLeaf(std::move(first));
    addInputToLeaf(*leaf, std::move(second));
    return leaf;
  }

  // Absorb one prepared chunk into an existing leaf.
  virtual void addInputToLeaf(BufferedState& leaf, InputChunk input) = 0;

  // Reduce a resident leaf to its smallest form, for example one row for each
  // key. PBS calls this when a leaf exceeds its row limit and hash
  // partitioning cannot split it, which happens when its rows share few keys.
  // The default does nothing, for strategies whose leaves are always compact.
  virtual void compactLeaf(BufferedState& /*leaf*/) {}

  // Optionally consume a routed chunk without restoring a spilled leaf.
  // Returning true tells PBS that the implementation retained the chunk and
  // deliberately kept the leaf spilled. This applies both to normal input and
  // to chunks replayed while draining. Such a leaf may grow beyond
  // maxRowsPerLeaf because repartitioning its stored state would require
  // restoring it first.
  //
  // The default preserves the restore-before-mutation behavior.
  virtual bool addInputToSpilledLeaf(BufferedState& /*leaf*/,
                                     InputChunk& /*input*/) {
    return false;
  }

  // PBS calls beginInput() before it routes one prepared input into the tree
  // and endInput() after the last chunk of that input was routed, also when
  // the routing throws. An implementation may postpone work that every
  // routed chunk needs, such as waiting for its copies, to endInput(), which
  // must leave nothing of the input in flight. Not called for chunks replayed
  // while draining. The defaults do nothing.
  virtual void beginInput() {}
  virtual void endInput() {}

  // Return true when a leaf kept spilled during input processing has another
  // independently restorable chunk that PBS must replay while draining. PBS
  // detaches such a leaf from the partition tree and feeds each restored chunk
  // back through its normal insertion and row-limit enforcement path.
  virtual bool hasNextDrainChunk(const BufferedState& /*leaf*/) const {
    return false;
  }

  // Restore and consume the next chunk advertised by hasNextDrainChunk().
  // The returned chunk must be suitable for insert() without another call to
  // prepareInput().
  virtual InputChunk restoreNextDrainChunk(BufferedState& /*leaf*/) {
    VELOX_UNSUPPORTED("Buffered state does not support drain replay");
  }

  // Optional external-aggregation drain. Oversized host-backed leaves can be
  // repartitioned a chunk at a time without rebuilding aggregate prefixes.
  // The caller protects the leaf throughout each operation. The partition
  // operation consumes its host chunks and returns exactly numPartitions
  // children (nullptr for empty children); a failure must abort the query.
  virtual bool supportsBulkSpilledDrain(const BufferedState& /*leaf*/) const {
    return false;
  }

  virtual std::vector<std::unique_ptr<BufferedState>> repartitionSpilledLeaf(
      BufferedState& /*leaf*/, const PartitionSpec& /*spec*/) {
    VELOX_UNSUPPORTED("Buffered state does not support external repartitioning");
  }

  // An engaged optional means the fitted leaf was consumed, even if its final
  // result is empty. A disengaged optional leaves the host representation
  // intact and requests the existing bounded replay path (e.g. no admission).
  virtual std::optional<CudfVectorPtr> tryFinalizeSpilledLeaf(
      std::unique_ptr<BufferedState>& /*leaf*/) {
    return std::nullopt;
  }

  // Report the logical row count PBS should track for this leaf.
  virtual size_t leafRowCount(const BufferedState& leaf) const = 0;

  // Report the approximate flat size of this leaf state in bytes.
  // TODO (dm): This should be part of BufferedState
  virtual uint64_t leafFlatSize(const BufferedState& leaf) const = 0;

  // Return independently reclaimable device bytes for a resident leaf. The
  // default keeps non-spill-aware strategies out of GPU reclamation.
  virtual uint64_t leafReclaimableBytes(const BufferedState& /*leaf*/) const {
    return 0;
  }

  // Evict a resident leaf to non-device storage. Implementations must not
  // release the resident state until the replacement is complete.
  virtual void spillLeaf(BufferedState& /*leaf*/) {
    VELOX_UNSUPPORTED("Buffered state does not support spilling");
  }

  // Make a spilled leaf resident before an operation reads or mutates it.
  // The default is a no-op for strategies that never spill.
  virtual void restoreLeaf(BufferedState& /*leaf*/) {}

  // Partition one prepared chunk according to an internal node's partition
  // spec and return one child chunk per partition. The input remains valid so
  // PBS can retry with a different seed if partitioning makes no progress.
  virtual std::vector<InputChunk> partitionInput(const InputChunk& input,
                                                 const PartitionSpec& spec) = 0;

  // Partition one overflowing leaf into transient child chunks according to
  // `spec`. The leaf remains valid so PBS can retry with a different seed if
  // partitioning makes no progress. After a successful attempt, PBS destroys
  // the old leaf before turning these chunks into persistent child states.
  virtual std::vector<InputChunk> partitionLeaf(const BufferedState& leaf,
                                                const PartitionSpec& spec) = 0;

  // Finalize one leaf and return one output batch.
  virtual CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) = 0;

  // Return partition-key indices in the prepared input schema.
  virtual const std::vector<cudf::size_type>& keyIndices() const = 0;
};

// Owns a recursive hash-partition tree for operators that must buffer grouped
// state until all input is received.
//
// PBS only manages the partition tree and leaf row limits. It does not know
// the concrete type of state stored in a leaf. The caller supplies a
// `BufferedStateOps` implementation that defines:
// - how raw input becomes an `InputChunk`,
// - what concrete `BufferedState` subclass each leaf holds,
// - how a chunk is added to a leaf,
// - how a leaf reports its logical row count,
// - how prepared input is partitioned at internal nodes,
// - how an overflowing leaf is repartitioned into child leaves,
// - and how one leaf is finalized into one output batch.
//
// Use PBS when all rows for the same grouping key must continue to route to the
// same subtree, but one in-memory leaf may grow past a safe row limit. PBS
// keeps splitting oversized leaves into more hash partitions and then drains
// one finalized leaf at a time at the end.
//
// Typical usage:
// 1. Define a `BufferedStateOps` implementation for the operator.
// 2. Store any operator-specific state inside a `BufferedState` subclass.
// 3. Construct PBS with that strategy and a max leaf row count.
// 4. Call `addInput()` for each incoming `CudfVector`.
// 5. At the end, call `drainNextOutput()` until it returns `nullptr`.
//
// `InputChunk` may hold either a borrowed `cudf::table_view` or a materialized
// table through its owning `CudfVectorPtr`. This allows callers such as final
// groupby to avoid eager materialization when a permuted view is sufficient.
class PartitionedBufferedState {
 public:
  struct Node {
    size_t leafRows{0};
    std::unique_ptr<BufferedState> leafState;
    std::optional<PartitionSpec> partitionSpec;
    std::vector<std::unique_ptr<Node>> children;
    // Last published size of this leaf. Mutated only by the driver or paused
    // reclaimer, never read by concurrent victim selection.
    uint64_t estimatedReclaimableBytes{0};

    bool isLeaf() const {
      return !partitionSpec.has_value();
    }
  };

  PartitionedBufferedState(
      std::unique_ptr<BufferedStateOps> ops,
      size_t maxRowsPerLeaf,
      uint32_t initialHashSeed = kBufferedStateHashSeed,
      std::function<void(uint64_t)> publishReclaimableEstimate = {},
      size_t minPartitionsPerSplit = 2);

  void addInput(CudfVectorPtr rawInput);

  CudfVectorPtr drainNextOutput();

  bool empty() const;

  // Returns device bytes held by independently spillable resident leaves.
  // A leaf currently being read or mutated is deliberately excluded. Caller
  // must have exclusive access to this state; this walks the tree. Local sole
  // ownership or a management adapter's established access protocol can supply
  // that exclusion. No task/arbitrator state is consulted by this mechanism.
  uint64_t reclaimableBytes() const;

  // Spills largest resident leaves first until at least 'targetBytes' have
  // been selected. A zero target spills every eligible resident leaf.
  // Requires the same exclusive access and active-use protection as inspection.
  // Returns logical device bytes released after successful replacement and
  // CUDA completion, not an arbitrator's pool-reservation credit.
  uint64_t reclaim(uint64_t targetBytes);

 private:
  class ActiveLeafGuard {
   public:
    ActiveLeafGuard(PartitionedBufferedState& owner, Node& node);
    ~ActiveLeafGuard();

    ActiveLeafGuard(const ActiveLeafGuard&) = delete;
    ActiveLeafGuard& operator=(const ActiveLeafGuard&) = delete;

   private:
    PartitionedBufferedState& owner_;
    Node& node_;
    Node* previous_;
  };

  void insert(Node& node, InputChunk bufferedInput);

  void splitLeaf(Node& node);

  void splitLeafAndAddInput(Node& node, InputChunk bufferedInput);

  CudfVectorPtr drainNextOutput(Node& node);

  PartitionSpec makePartitionSpec(size_t totalRows);

  void ensureLeafWithinLimit(Node& node);

  std::vector<InputChunk> partitionInput(const InputChunk& input,
                                         const PartitionSpec& spec);

  void restoreLeaf(Node& node);

  // Updates the published hint in O(1) per changed leaf, including transitions
  // within an addInput call. The callback must not throw, allocate, or inspect
  // the tree. The hint may include a currently active leaf; reclaim() always
  // rechecks current eligibility after pause.
  void updateReclaimableEstimate(Node& node);

  void collectReclaimableLeaves(
      Node& node,
      std::vector<std::pair<Node*, uint64_t>>& leaves) const;

  std::unique_ptr<BufferedStateOps> ops_;
  const size_t maxRowsPerLeaf_;
  // A split makes at least this many children. More children per split make
  // a shallower tree, in which each row is partitioned fewer times.
  const size_t minPartitionsPerSplit_;
  std::unique_ptr<Node> root_;
  uint32_t nextHashSeed_;
  const std::function<void(uint64_t)> publishReclaimableEstimate_;
  // Wider internal sum preserves correct subtraction if the public uint64_t
  // estimate saturates. This is accounting metadata, not a memory allowance.
  __uint128_t estimatedReclaimableBytes_{0};

  // This is intentionally operator/PBS state, not thread-local state. Velox
  // task pause guarantees prevent concurrent driver mutation during external
  // reclaim; reentrant reclaim on the calling thread observes this guard.
  Node* activeLeaf_{nullptr};
};

} // namespace facebook::velox::cudf_velox
