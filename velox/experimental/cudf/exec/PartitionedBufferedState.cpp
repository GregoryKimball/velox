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
#include "velox/experimental/cudf/exec/CudfSpillContext.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"
#include "velox/experimental/cudf/exec/SpilledCudfVector.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/common/testutil/TestValue.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/contiguous_split.hpp>
#include <cstring>
#include <span>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <folly/ScopeGuard.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

// Avoid spinning forever when a single partition key cannot be split.
constexpr uint32_t kMaxSplitSeedAttempts = 4;

bool referencesSameColumn(const cudf::column_view& left,
                          const cudf::column_view& right) {
  if (left.type() != right.type() || left.size() != right.size() ||
      left.offset() != right.offset() ||
      left.head<void>() != right.head<void>() ||
      left.null_mask() != right.null_mask() ||
      left.num_children() != right.num_children()) {
    return false;
  }

  for (cudf::size_type i = 0; i < left.num_children(); ++i) {
    if (!referencesSameColumn(left.child(i), right.child(i))) {
      return false;
    }
  }
  return true;
}

bool nodeEmpty(const PartitionedBufferedState::Node& node) {
  if (node.isLeaf()) {
    return node.leafState == nullptr;
  }

  for (const auto& child : node.children) {
    if (child && !nodeEmpty(*child)) {
      return false;
    }
  }
  return true;
}

size_t countNonEmptyChildren(
    const std::vector<InputChunk>& storedPartitions,
    const std::vector<InputChunk>& incomingPartitions) {
  VELOX_CHECK_EQ(storedPartitions.size(), incomingPartitions.size());

  size_t nonEmptyChildren = 0;
  for (size_t i = 0; i < incomingPartitions.size(); ++i) {
    if (!storedPartitions[i].empty() || !incomingPartitions[i].empty()) {
      ++nonEmptyChildren;
    }
  }
  return nonEmptyChildren;
}

struct SplitLeafAttempt {
  PartitionSpec spec;
  std::vector<InputChunk> storedPartitions;
  std::vector<InputChunk> incomingPartitions;
};

} // namespace

bool InputChunk::ownsFullTable() const {
  if (storage != InputChunkStorage::kOwned || owner == nullptr ||
      tableOwner != nullptr) {
    return false;
  }

  const auto ownerView = owner->getTableView();
  if (view.num_rows() != ownerView.num_rows() ||
      view.num_columns() != ownerView.num_columns()) {
    return false;
  }
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    if (!referencesSameColumn(view.column(i), ownerView.column(i))) {
      return false;
    }
  }
  return true;
}

InputChunk InputChunk::materialize(rmm::device_async_resource_ref mr) && {
  if (empty()) {
    return InputChunk{};
  }

  if (storage == InputChunkStorage::kOwned) {
    VELOX_CHECK(
        ownsFullTable(),
        "An owned InputChunk must reference its owner's complete table");
    if (owner.use_count() == 1) {
      return std::move(*this);
    }
  }

  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NOT_NULL(type);
  VELOX_CHECK(owner != nullptr || tableOwner != nullptr,
              "A borrowed InputChunk must retain its source storage");

  auto materializedTable = std::make_unique<cudf::table>(view, stream, mr);
  auto materialized =
      std::make_shared<CudfVector>(pool,
                                   type,
                                   materializedTable->num_rows(),
                                   std::move(materializedTable),
                                   stream);
  auto result = InputChunk{materialized->pool(),
                           type,
                           materialized->getTableView(),
                           materialized->stream(),
                           std::move(materialized),
                           nullptr,
                           InputChunkStorage::kOwned};
  owner.reset();
  tableOwner.reset();
  return result;
}

PartitionedBufferedState::PartitionedBufferedState(
    std::unique_ptr<BufferedStateOps> ops,
    size_t maxRowsPerLeaf,
    uint32_t initialHashSeed,
    std::function<void(uint64_t)> publishReclaimableEstimate,
    size_t minPartitionsPerSplit)
    : ops_(std::move(ops)),
      maxRowsPerLeaf_(maxRowsPerLeaf),
      minPartitionsPerSplit_(minPartitionsPerSplit),
      root_(std::make_unique<Node>()),
      nextHashSeed_(initialHashSeed),
      publishReclaimableEstimate_(std::move(publishReclaimableEstimate)) {
  VELOX_CHECK_NOT_NULL(ops_);
  VELOX_CHECK_GT(maxRowsPerLeaf_, 0);
  VELOX_CHECK_GE(minPartitionsPerSplit_, 2);
}

PartitionedBufferedState::ActiveLeafGuard::ActiveLeafGuard(
    PartitionedBufferedState& owner,
    Node& node)
    : owner_(owner), node_(node), previous_(owner.activeLeaf_) {
  owner_.activeLeaf_ = &node;
}

PartitionedBufferedState::ActiveLeafGuard::~ActiveLeafGuard() {
  owner_.activeLeaf_ = previous_;
}

void PartitionedBufferedState::addInput(CudfVectorPtr rawInput) {
  if (!rawInput || rawInput->size() == 0) {
    return;
  }

  auto compacted = ops_->prepareInput(std::move(rawInput));
  if (compacted.empty()) {
    return;
  }

  ops_->beginInput();
  // Whatever the ops postponed for this input is finished before this call
  // returns, also when routing throws. That failure is the one reported.
  auto endInputOnFailure = folly::makeGuard([&]() noexcept {
    try {
      ops_->endInput();
    } catch (...) {
    }
  });
  insert(*root_, std::move(compacted));
  endInputOnFailure.dismiss();
  ops_->endInput();
  common::testutil::TestValue::adjust(
      "facebook::velox::cudf_velox::PartitionedBufferedState::afterAddInput", this);
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput() {
  auto output = drainNextOutput(*root_);
  return output;
}

bool PartitionedBufferedState::empty() const {
  return nodeEmpty(*root_);
}

uint64_t PartitionedBufferedState::reclaimableBytes() const {
  std::vector<std::pair<Node*, uint64_t>> leaves;
  collectReclaimableLeaves(*root_, leaves);
  uint64_t bytes = 0;
  for (const auto& [node, leafBytes] : leaves) {
    if (std::numeric_limits<uint64_t>::max() - bytes < leafBytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    bytes += leafBytes;
  }
  return bytes;
}

uint64_t PartitionedBufferedState::reclaim(uint64_t targetBytes) {
  std::vector<std::pair<Node*, uint64_t>> leaves;
  collectReclaimableLeaves(*root_, leaves);
  std::sort(
      leaves.begin(), leaves.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
      });

  uint64_t reclaimedBytes = 0;
  for (const auto& [node, leafBytes] : leaves) {
    if (targetBytes != 0 && reclaimedBytes >= targetBytes) {
      break;
    }
    VELOX_CHECK_NOT_NULL(node->leafState);
    ops_->spillLeaf(*node->leafState);
    updateReclaimableEstimate(*node);
    VELOX_CHECK_EQ(
        ops_->leafReclaimableBytes(*node->leafState),
        0,
        "A spilled PBS leaf must not retain reclaimable device bytes");
    reclaimedBytes += leafBytes;
  }
  return reclaimedBytes;
}

void PartitionedBufferedState::collectReclaimableLeaves(
    Node& node,
    std::vector<std::pair<Node*, uint64_t>>& leaves) const {
  if (!node.isLeaf()) {
    for (auto& child : node.children) {
      collectReclaimableLeaves(*child, leaves);
    }
    return;
  }

  if (node.leafState && &node != activeLeaf_) {
    const auto bytes = ops_->leafReclaimableBytes(*node.leafState);
    if (bytes != 0) {
      leaves.emplace_back(&node, bytes);
    }
  }
}

void PartitionedBufferedState::insert(Node& node, InputChunk bufferedInput) {
  if (bufferedInput.empty()) {
    return;
  }
  const auto inputRows = bufferedInput.size();

  // If node is not a leaf, it is an internal node and any input has to be
  // partitioned and inserted into child nodes.
  if (!node.isLeaf()) {
    auto partitions = partitionInput(bufferedInput, *node.partitionSpec);
    VELOX_CHECK_EQ(partitions.size(), node.children.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (!partitions[i].empty()) {
        insert(*node.children[i], std::move(partitions[i]));
      }
    }
    return;
  }

  // If node is a leaf but empty, initialize it with the input.
  if (!node.leafState) {
    node.leafState = ops_->createLeaf(std::move(bufferedInput));
    if (node.leafState) {
      node.leafRows = ops_->leafRowCount(*node.leafState);
      updateReclaimableEstimate(node);
      ensureLeafWithinLimit(node);
    }
    return;
  }

  ActiveLeafGuard activeLeaf(*this, node);
  if (ops_->addInputToSpilledLeaf(*node.leafState, bufferedInput)) {
    node.leafRows = ops_->leafRowCount(*node.leafState);
    updateReclaimableEstimate(node);
    return;
  }
  restoreLeaf(node);
  // If node is a leaf and not empty, check if the input can be added to the
  // leaf. If not, split the leaf and insert the input into the new subtree.
  const auto projectedRows =
      ops_->estimatedMergedRowUpperBound(*node.leafState, bufferedInput);
  if (projectedRows > maxRowsPerLeaf_) {
    splitLeafAndAddInput(node, std::move(bufferedInput));
    return;
  }

  ops_->addInputToLeaf(*node.leafState, std::move(bufferedInput));
  node.leafRows = ops_->leafRowCount(*node.leafState);
  updateReclaimableEstimate(node);
  ensureLeafWithinLimit(node);
}

void PartitionedBufferedState::splitLeaf(Node& node) {
  splitLeafAndAddInput(node, InputChunk{});
}

void PartitionedBufferedState::splitLeafAndAddInput(Node& node,
                                                    InputChunk bufferedInput) {
  VELOX_CHECK(node.isLeaf());
  VELOX_CHECK(node.leafState || !bufferedInput.empty());
  ActiveLeafGuard activeLeaf(*this, node);
  restoreLeaf(node);

  const auto totalRows = node.leafRows + bufferedInput.size();
  std::optional<SplitLeafAttempt> splitAttempt;
  for (uint32_t attempt = 1; attempt <= kMaxSplitSeedAttempts; ++attempt) {
    auto spec = makePartitionSpec(totalRows);
    auto storedPartitions = node.leafState
        ? ops_->partitionLeaf(*node.leafState, spec)
        : std::vector<InputChunk>(spec.numPartitions);
    auto incomingPartitions = partitionInput(bufferedInput, spec);
    const auto nonEmptyChildren =
        countNonEmptyChildren(storedPartitions, incomingPartitions);

    if (nonEmptyChildren > 1) {
      // Found a partition spec that can split the leaf into more than one
      // child. If nonEmptyChildren = 1, then the partition spec could not
      // split and we'd have no meaningful progress.
      splitAttempt.emplace(SplitLeafAttempt{std::move(spec),
                                            std::move(storedPartitions),
                                            std::move(incomingPartitions)});
      break;
    }
  }

  if (!splitAttempt.has_value()) {
    // Partitioning cannot separate rows that share few keys. Reduce the leaf
    // to what it actually has to hold and decide from that whether a split is
    // necessary: the row count used so far is an upper bound that counts
    // every stored and incoming row of the same key.
    const bool hadInput = !bufferedInput.empty();
    if (hadInput) {
      ops_->addInputToLeaf(*node.leafState, std::move(bufferedInput));
    }
    const auto rowsBeforeCompaction = ops_->leafRowCount(*node.leafState);
    node.leafRows = rowsBeforeCompaction;
    if (node.leafRows > maxRowsPerLeaf_) {
      // A leaf that only appends its input still holds every row.
      ops_->compactLeaf(*node.leafState);
      node.leafRows = ops_->leafRowCount(*node.leafState);
    }
    updateReclaimableEstimate(node);
    if (node.leafRows <= maxRowsPerLeaf_) {
      return;
    }
    if (hadInput || node.leafRows < rowsBeforeCompaction) {
      // The leaf now holds the groups that must be partitioned, not an upper
      // bound from overlapping chunks. Try once more to split those.
      splitLeafAndAddInput(node, InputChunk{});
      return;
    }
  }

  VELOX_CHECK(
      splitAttempt.has_value(),
      "Partitioning buffered state made no progress after {} hash seed "
      "attempts: {} rows exceeded the per-leaf limit of {} rows. This can "
      "happen when every row has the same partition key.",
      kMaxSplitSeedAttempts,
      totalRows,
      maxRowsPerLeaf_);

  node.leafRows = 0;
  node.leafState.reset();
  updateReclaimableEstimate(node);
  node.partitionSpec = splitAttempt->spec;
  node.children.clear();
  node.children.reserve(splitAttempt->spec.numPartitions);
  for (int32_t i = 0; i < splitAttempt->spec.numPartitions; ++i) {
    node.children.push_back(std::make_unique<Node>());
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    auto& child = *node.children[i];
    auto& stored = splitAttempt->storedPartitions[i];
    auto& incoming = splitAttempt->incomingPartitions[i];
    if (!stored.empty() && !incoming.empty()) {
      child.leafState =
          ops_->createLeafFromInputs(std::move(stored), std::move(incoming));
    } else if (!stored.empty()) {
      child.leafState = ops_->createLeaf(std::move(stored));
    } else if (!incoming.empty()) {
      child.leafState = ops_->createLeaf(std::move(incoming));
    }

    if (child.leafState) {
      child.leafRows = ops_->leafRowCount(*child.leafState);
      updateReclaimableEstimate(child);
      ensureLeafWithinLimit(child);
    }
  }
}

CudfVectorPtr PartitionedBufferedState::drainNextOutput(Node& node) {
  if (!node.isLeaf()) {
    for (auto& child : node.children) {
      if (child) {
        auto output = drainNextOutput(*child);
        if (output) {
          return output;
        }
      }
    }
    return nullptr;
  }

  if (!node.leafState) {
    return nullptr;
  }

  if (ops_->supportsBulkSpilledDrain(*node.leafState)) {
    ActiveLeafGuard activeLeaf(*this, node);
    for (uint32_t attempt = 0;
         node.leafRows > maxRowsPerLeaf_ && attempt < kMaxSplitSeedAttempts;
         ++attempt) {
      const auto rows = node.leafRows;
      auto spec = makePartitionSpec(rows);
      auto children = ops_->repartitionSpilledLeaf(*node.leafState, spec);
      VELOX_CHECK_EQ(children.size(), spec.numPartitions);
      size_t totalRows = 0;
      size_t nonEmpty = 0;
      size_t onlyChild = 0;
      for (size_t i = 0; i < children.size(); ++i) {
        if (children[i]) {
          const auto childRows = ops_->leafRowCount(*children[i]);
          VELOX_CHECK_GT(childRows, 0);
          VELOX_CHECK_LE(childRows, rows - totalRows);
          totalRows += childRows;
          ++nonEmpty;
          onlyChild = i;
        }
      }
      VELOX_CHECK_EQ(totalRows, rows);
      VELOX_CHECK_GT(nonEmpty, 0);
      if (nonEmpty == 1) {
        // Keep every row and retry a different seed. If all attempts fail,
        // the original replay/compaction path below handles unsplittable keys.
        node.leafState = std::move(children[onlyChild]);
        updateReclaimableEstimate(node);
        continue;
      }
      node.leafState.reset();
      node.leafRows = 0;
      updateReclaimableEstimate(node);
      node.partitionSpec = std::move(spec);
      node.children.reserve(children.size());
      for (auto& state : children) {
        auto child = std::make_unique<Node>();
        child->leafState = std::move(state);
        if (child->leafState) {
          child->leafRows = ops_->leafRowCount(*child->leafState);
          updateReclaimableEstimate(*child);
        }
        node.children.push_back(std::move(child));
      }
      return drainNextOutput(node);
    }
    if (node.leafRows <= maxRowsPerLeaf_) {
      auto output = ops_->tryFinalizeSpilledLeaf(node.leafState);
      if (output.has_value()) {
        VELOX_CHECK_NULL(node.leafState);
        node.leafRows = 0;
        updateReclaimableEstimate(node);
        return std::move(*output);
      }
    }
  }

  if (ops_->hasNextDrainChunk(*node.leafState)) {
    ActiveLeafGuard activeLeaf(*this, node);
    auto replayLeaf = std::move(node.leafState);
    node.leafRows = 0;
    updateReclaimableEstimate(node);
    while (ops_->hasNextDrainChunk(*replayLeaf)) {
      auto restored = ops_->restoreNextDrainChunk(*replayLeaf);
      insert(node, std::move(restored));
    }
    return drainNextOutput(node);
  }

  ActiveLeafGuard activeLeaf(*this, node);
  restoreLeaf(node);
  node.leafRows = 0;
  auto leaf = std::move(node.leafState);
  updateReclaimableEstimate(node);
  auto output = ops_->finalizeLeaf(std::move(leaf));
  return output;
}

PartitionSpec PartitionedBufferedState::makePartitionSpec(size_t totalRows) {
  VELOX_CHECK_GT(totalRows, maxRowsPerLeaf_);

  const auto requiredPartitions =
      (totalRows + maxRowsPerLeaf_ - 1) / maxRowsPerLeaf_;
  const auto numPartitions =
      std::min(totalRows, std::max(minPartitionsPerSplit_, requiredPartitions));
  VELOX_CHECK_LE(numPartitions, std::numeric_limits<int32_t>::max());

  return PartitionSpec{static_cast<int32_t>(numPartitions),
                       ops_->keyIndices(),
                       cudf::hash_id::HASH_MURMUR3,
                       nextHashSeed_++};
}

void PartitionedBufferedState::ensureLeafWithinLimit(Node& node) {
  if (node.isLeaf() && node.leafState && node.leafRows > maxRowsPerLeaf_) {
    splitLeaf(node);
  }
}

std::vector<InputChunk> PartitionedBufferedState::partitionInput(
    const InputChunk& input,
    const PartitionSpec& spec) {
  return input.empty() ? std::vector<InputChunk>(spec.numPartitions)
                       : ops_->partitionInput(input, spec);
}

void PartitionedBufferedState::restoreLeaf(Node& node) {
  if (node.leafState) {
    ops_->restoreLeaf(*node.leafState);
    updateReclaimableEstimate(node);
  }
}

void PartitionedBufferedState::updateReclaimableEstimate(Node& node) {
  if (!publishReclaimableEstimate_) {
    return;
  }
  const auto bytes =
      node.leafState ? ops_->leafReclaimableBytes(*node.leafState) : 0;
  estimatedReclaimableBytes_ -= node.estimatedReclaimableBytes;
  estimatedReclaimableBytes_ += bytes;
  node.estimatedReclaimableBytes = bytes;
  publishReclaimableEstimate_(static_cast<uint64_t>(std::min<__uint128_t>(
      estimatedReclaimableBytes_, std::numeric_limits<uint64_t>::max())));
}

} // namespace facebook::velox::cudf_velox
