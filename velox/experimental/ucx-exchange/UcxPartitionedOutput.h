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

#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <optional>

namespace facebook::velox::ucx_exchange {

/// This is the cudf equivalent of the PartitionedOutput operator for cudf.
/// Instead of serializing and segmenting the partitioned data into an
/// OutputBuffer, the UcxPartitionedOutput operator transfers entire
/// cudf::packed_columns corresponding to CudfVectors to other workers.
class UcxPartitionedOutput : public exec::Operator,
                             public cudf_velox::NvtxHelper {
 public:
  // Default minimum rows to accumulate before flushing. Matches HTTP
  // PartitionedOutput's ~10,000 row target. Overridable via
  // CudfConfig::kUcxPartitionedOutputBatchRows.
  // PartitionedOutput's ~10,000 row target. The cuDF exchange system property
  // provides the cluster default and can be overridden per query.
  static constexpr int64_t kDefaultTargetRowsPerChunk = 10'000;

  /// @param queueManager Output queue manager the partitions are enqueued to.
  /// Comes from the same exec::OutputTransportEntry that builds this operator,
  /// so the operator and the manager can never diverge. Held weakly, matching
  /// exec::PartitionedOutput.
  UcxPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      const std::shared_ptr<UcxOutputQueueManager>& queueManager);

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr. The action is to further process
  /// unprocessed input. If all input has been processed, 'this' is in
  /// a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  /// The caller checks isBlocked before adding input, so this only needs to
  /// report whether the operator can still accept rows once unblocked.
  bool needsInput() const override {
    return !finished_ && !noMoreInput_ && !shouldDrainPending() &&
        blockingReason_ == exec::BlockingReason::kNotBlocked;
  }

  // The operator is blocked when its output queue is over capacity.
  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  // The operaor is finished when the queue manager say the queues have all been
  // drained ?
  bool isFinished() override;

 private:
  std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
  sharedQueueManager();

  bool usesHashPartitioning() const;

  // Heuristic method to derive the partition keys from the PartitionNode
  // specification.
  void initPartitionKeys(
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  // Partitions the cudf table view using the partition keys and a hash
  // function using the given stream.
  void hashPartition(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream,
      int64_t estimatedBytes);

  // Splits the cudf table view into equal sizes. This is used when
  // RoundRobin partitioning is requested but round robin on a
  // row-by-row basis is not meaningful for UCX exchange.
  void equalPartition(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream,
      std::unique_ptr<cudf::table> tableOwner,
      std::vector<cudf_velox::CudfVectorPtr> vectorOwners,
      int64_t estimatedBytes);

  struct PendingPartitionedBatch {
    std::unique_ptr<cudf::table> tableOwner;
    std::vector<cudf_velox::CudfVectorPtr> vectorOwners;
    cudf::table_view tableView;
    std::vector<cudf::size_type> offsets;
    int64_t estimatedBytes{0};
    bool conservativeChunkSizing{false};
    rmm::cuda_stream_view stream;
    size_t nextPartition{0};
    std::vector<cudf::size_type> nextRows;
    std::vector<uint64_t> nextPayloadBytes;
    std::vector<uint64_t> drainDeficits;
    size_t remainingPartitions{0};
  };

  struct PendingInputBatch {
    std::unique_ptr<cudf::table> tableOwner;
    std::vector<cudf_velox::CudfVectorPtr> vectorOwners;
    cudf::table_view tableView;
    int64_t estimatedBytes{0};
    rmm::cuda_stream_view stream;
  };

  void partitionPendingInputBatch();

  bool drainPendingPartitionedBatch();

  bool tryDrainWithFullContiguousSplit(PendingPartitionedBatch& batch);

  bool tryDrainWithContiguousSplit(PendingPartitionedBatch& batch);

  void initializePartitionDrainState(PendingPartitionedBatch& batch) const;

  uint64_t estimatedPartitionBytes(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end) const;

  uint64_t exactPartitionBytes(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end) const;

  cudf::size_type rowsForPayloadTarget(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end,
      uint64_t targetBytes) const;

  bool shouldSplitPayload(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end) const;

  uint64_t baseTransferWindowBytes(const PendingPartitionedBatch& batch) const;

  uint64_t normalTransferWindowBytes(
      const PendingPartitionedBatch& batch) const;

  uint64_t maxTransferWindowBytes(const PendingPartitionedBatch& batch) const;

  uint64_t transferWindowBytes(
      const PendingPartitionedBatch& batch,
      int destination,
      const std::shared_ptr<UcxOutputQueueManager>& queueManager) const;

  void recordAllocationPressure(uint64_t waitBytes);

  // Routes the table to destinations by partition, hashing on the partition
  // keys when they are known and splitting into equal sizes otherwise.
  // 'numRows' is the logical row count of 'tableView', which a table with no
  // columns cannot report for itself.
  void partitionAndEnqueue(
      cudf::table_view tableView,
      vector_size_t numRows,
      rmm::cuda_stream_view stream);

  // Splits a column-less payload across the destinations by row count alone.
  // There is no GPU data to move and no partition key to hash, so each
  // destination receives its own packed empty table plus its share of the
  // rows, using the same boundaries equalPartition() would have used.
  void equalPartitionRowCountOnly(
      cudf::table_view tableView,
      vector_size_t numRows,
      rmm::cuda_stream_view stream);

  // Sends the rows that must reach every destination -- rows with a null
  // partition key, plus one arbitrary row over the lifetime of this operator --
  // to all destinations, then routes the remaining rows by partition.
  void replicateNullsAndAnyThenPartition(
      cudf::table_view tableView,
      vector_size_t numRows,
      rmm::cuda_stream_view stream);

  // Packs the table separately for each destination and enqueues one private
  // copy per destination. A shared packed_columns cannot be used: the
  // intra-node transfer path moves its members out, which would corrupt the
  // data for every other destination.
  void packAndEnqueueToAllDestinations(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream);

  const std::weak_ptr<UcxOutputQueueManager> queueManager_;
  std::vector<column_index_t> partitionKeyIndices_;
  const size_t numPartitions_;

  // True when rows with a null partition key, plus one arbitrary row, must
  // reach every destination. Mirrors
  // core::PartitionedOutputNode::isReplicateNullsAndAny().
  const bool replicateNullsAndAny_;

  // Set once the arbitrary row has been replicated. Matches
  // exec::PartitionedOutput::replicatedAny_: the arbitrary row is replicated
  // once per operator instance, not once per flush.
  bool replicatedAnyRow_{false};

  const int pipelineId_;
  const int driverId_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;

  bool finished_{false};
  std::string spec_;
  bool isGatherPartition_{false};

  // Used for switching columns when column order differs between input and
  // output.
  std::vector<uint32_t> remap_;

  /// Concatenates pending inputs and partitions/enqueues the merged result.
  void flushPending();

  bool shouldFlushPending() const;

  bool shouldDrainPending() const;

  /// Accumulated CudfVectors awaiting flush.
  std::vector<cudf_velox::CudfVectorPtr> pendingInputs_;
  /// Total rows across pendingInputs_.
  int64_t pendingRows_{0};
  /// Estimated bytes across pendingInputs_.
  int64_t pendingBytes_{0};
  /// Configured row threshold for flushing (from QueryConfig).
  const int64_t targetRowsPerChunk_;
  /// Initial target GPU payload size for geometrically chunked UCX messages.
  const uint64_t initialPayloadBytes_;
  /// Maximum adaptive window as a multiple of the per-batch base window.
  const uint64_t transferWindowMultiplier_;

  /// Partitioned table waiting to be packed and enqueued destination-by-
  /// destination. This avoids materializing the full fanout before UCX output
  /// backpressure can take effect.
  std::optional<PendingPartitionedBatch> pendingPartitionedBatch_;

  /// Cudf input that has been materialized enough to retry partitioning after
  /// GPU allocation pressure without asking the upstream driver for the row
  /// again.
  std::optional<PendingInputBatch> pendingInputBatch_;
};

} // namespace facebook::velox::ucx_exchange
