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

#include "velox/common/future/VeloxPromise.h"
#include "velox/experimental/cudf/connectors/hive/FusedForColumnEncoding.h"
#include "velox/experimental/ucx-exchange/ExchangeCompression.h"
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// Holds exchange data that a later stage reads back file by file, in place of
/// the files of a temporary table. Payloads stay in the exchange's packed and
/// compressed format (see splitExchangePayloads()). Their columns may have been
/// encoded before packing, as dictionary indices or differences (see
/// encodeForFusedFor()); the keys of a dictionary are stored once for all the
/// payloads that share them. A payload is kept in device memory while the
/// device tier is below its capacity and is moved to pinned host memory
/// otherwise. Nothing is written to disk: adding a payload that fits in neither
/// tier fails. The host tier has a pinned memory pool of its own, which grows
/// in large blocks.
class ExchangeStore {
 public:
  /// How the columns of a payload were encoded before packing.
  struct PayloadEncoding {
    /// The encoding of each column. Empty when every column holds its values.
    std::vector<cudf_velox::connector::hive::FusedForEncoding> columns;
    /// For each dictionary column, in column order, the id of its keys (see
    /// addKeys()).
    std::vector<uint64_t> keys;
    /// For each column of differences, in column order, the value the
    /// payload's differences start from (see encodeForFusedFor()).
    std::vector<int64_t> bases;
  };

  ExchangeStore();
  ~ExchangeStore();

  /// The process-wide store.
  static ExchangeStore& instance();

  /// Sets the bytes of payload data the device tier may hold. A payload added
  /// while it does not fit goes to the host tier. Defaults to 0.
  void setDeviceCapacity(uint64_t bytes);

  /// Sets the bytes of payload data the host tier may hold. Defaults to no
  /// limit.
  void setHostCapacity(uint64_t bytes);

  /// Adds 'payload', which holds 'numRows' rows encoded as 'encoding', to
  /// 'file' of 'table'. The payload's device data must be complete. A payload
  /// that goes to the host tier is copied on 'stream', which is synchronized
  /// before its device memory is freed. Returns whether the payload went to the
  /// host tier.
  bool add(
      std::string_view table,
      std::string_view file,
      PackedExchangePayload payload,
      vector_size_t numRows,
      PayloadEncoding encoding,
      cuda::stream_ref stream);

  /// Adds the keys of a dictionary column, a payload of one column with
  /// 'numKeys' rows, which 'numReferences' payloads of 'table' will share.
  /// Returns the id under which those payloads refer to them. The keys stay
  /// until the last of those payloads is taken or the table is dropped. They
  /// are placed like a payload but stay in their tier (see demote()).
  uint64_t addKeys(
      std::string_view table,
      PackedExchangePayload keys,
      vector_size_t numKeys,
      uint32_t numReferences,
      cuda::stream_ref stream);

  /// A payload taken from the store, restored on the device: its data, the
  /// encoding of its columns, and the keys of each dictionary column in
  /// column order. Fused FOR lanes payloads, keys included, are returned
  /// before joinExchangeLanes().
  struct TakenPayload {
    std::unique_ptr<PackedTableWithStream> data;
    PayloadEncoding encoding;
    std::vector<std::unique_ptr<PackedTableWithStream>> keys;
  };

  /// Removes the payloads of 'file' of 'table' and returns them restored on
  /// the device, in the order in which they were added. Payloads in the host
  /// tier, and the keys they refer to, are copied to device memory from 'mr'
  /// on 'stream'. Returns nothing for a file without payloads. Adds what it
  /// took from each tier to 'stats' when given.
  struct TakeStats {
    uint64_t deviceBytes{0};
    uint64_t hostBytes{0};
    /// Time spent copying host-tier payloads to the device.
    uint64_t copyNanos{0};
  };

  std::vector<TakenPayload> take(
      std::string_view table,
      std::string_view file,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      TakeStats* stats = nullptr);

  /// Records that the task 'taskId' writes 'table' on this worker. The table
  /// is complete once every task that began writing it has ended its writes
  /// (endWrites()); until then takeOrWait() waits for more payloads instead of
  /// treating a file without payloads as done.
  void beginWrite(std::string_view table, std::string_view taskId);

  /// Ends the writes of the task 'taskId' to every table it began writing, and
  /// wakes the readers of each table that is now complete.
  void endWrites(std::string_view taskId);

  /// Removes the payloads of 'files' of 'table' added so far, like take(), so
  /// that a reader can consume a table while it is being written. Returns them
  /// when there are any; an empty vector when there are none and the table is
  /// complete (the files got no rows or were taken already); and std::nullopt
  /// with 'future' set when there are none yet: it completes on the table's
  /// next payload or when the table completes. Several readers may read the
  /// same files: each payload goes to one of them.
  std::optional<std::vector<TakenPayload>> takeOrWait(
      std::string_view table,
      const std::vector<std::string>& files,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      ContinueFuture& future,
      TakeStats* stats = nullptr);

  /// Frees the payloads of every file of 'table' that was not taken.
  void drop(std::string_view table);

  /// Frees the payloads of every table whose name contains 'queryId'.
  void dropQuery(std::string_view queryId);

  /// Moves device-tier payloads to the host tier until at least 'bytes' are
  /// moved or none is left, and lowers the device tier's capacity to what it
  /// then holds, until the store holds no tables again. Keys are not moved:
  /// they are small, and payloads in several files may be copying from them.
  /// Returns the bytes moved. Called when a device allocation fails (see
  /// releaseDeviceMemory() in GpuResources.h).
  std::size_t demote(std::size_t bytes);

  struct Stats {
    /// Payload bytes in each tier, keys included.
    uint64_t deviceBytes{0};
    uint64_t hostBytes{0};
    /// Payloads held in either tier, keys included.
    uint64_t payloads{0};
    /// Pinned host memory the host tier's pool holds, in use or not.
    uint64_t hostPoolBytes{0};
  };

  Stats stats() const;

 private:
  // The host tier's pinned memory pool.
  struct HostPool;

  // Pinned host memory from the host tier's pool.
  class HostBuffer {
   public:
    HostBuffer(HostPool& pool, std::size_t bytes);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    void* data() const {
      return data_;
    }

    // Marks the end of an asynchronous copy into the buffer, enqueued on
    // 'stream'. Readers wait for it (orderReadsOn()), and so does the
    // destructor, so that the buffer does not return to its pool while the
    // copy still writes to it.
    void recordWrite(cuda::stream_ref stream);

    // Makes work enqueued on 'stream' from now on wait for the copy recorded
    // by recordWrite(), if any.
    void orderReadsOn(cuda::stream_ref stream) const;

   private:
    HostPool& pool_;
    void* data_;
    const std::size_t bytes_;
    cudaEvent_t written_{nullptr};
  };

  // Returns the host tier's pool, creating it on first use. Called with
  // 'mutex_' held.
  HostPool& hostPool();

  // A payload's metadata, which names its codec, its data in exactly one of
  // the tiers, and the encoding of its columns.
  struct StoredPayload {
    std::unique_ptr<std::vector<uint8_t>> metadata;
    std::unique_ptr<rmm::device_buffer> device;
    std::unique_ptr<HostBuffer> host;
    std::size_t bytes{0};
    vector_size_t numRows{0};
    PayloadEncoding encoding;
  };

  // Keys of a dictionary and the number of payloads that still refer to them.
  // Shared, so that a take() copies from them after releasing 'mutex_' while
  // a concurrent take() may remove them.
  struct StoredKeys {
    std::shared_ptr<const StoredPayload> payload;
    uint32_t numReferences{0};
  };

  using Files = folly::F14FastMap<std::string, std::vector<StoredPayload>>;

  struct Table {
    Files files;
    folly::F14FastMap<uint64_t, StoredKeys> keys;
    // Tasks that began writing the table and have not ended their writes.
    folly::F14FastSet<std::string> writers;
    // Readers waiting in takeOrWait() for the next payload or the completion.
    std::vector<ContinuePromise> waiters;
  };

  // Moves 'payload' from the device to the host tier unless the device tier
  // has room for it, and counts its bytes in its tier. Returns whether it went
  // to the host tier.
  bool place(StoredPayload& payload, cuda::stream_ref stream);

  // Subtracts the bytes and count of 'payloads' from the totals. Called with
  // 'mutex_' held.
  void forget(const std::vector<StoredPayload>& payloads);
  void forget(const StoredPayload& payload);
  void forget(const Table& table);

  // Records the table 'name' as complete, for takeOrWait(). Called with
  // 'mutex_' held when its last writer ends.
  void complete(const std::string& name);

  // Once 'tables_' is empty, restores the device tier's capacity to the
  // configured one and releases the host tier's pool when no host buffer is
  // left. A demote() answered one query's lack of memory, and keeping the
  // smaller tier would send later queries' payloads to host memory although
  // the device has room again. The pool keeps the pinned memory of the largest
  // spill until it is released, which would hold it for the life of the
  // process. Called with 'mutex_' held.
  void resetTiersIfEmpty();

  mutable std::mutex mutex_;
  // Declared before 'tables_' so that it outlives the host buffers.
  std::unique_ptr<HostPool> hostPool_;
  folly::F14FastMap<std::string, Table> tables_;
  uint64_t nextKeysId_{0};
  uint64_t deviceCapacity_{0};
  // The capacity setDeviceCapacity() gave; demote() lowers deviceCapacity_
  // below it until the store is empty again.
  uint64_t configuredDeviceCapacity_{0};
  // Names of tables whose writers have all ended, so that a file never written
  // (a bucket without rows) reads as done after its table was taken and
  // erased. Bounded by kMaxCompletedTables, oldest first out.
  folly::F14FastSet<std::string> completedTables_;
  std::deque<std::string> completedOrder_;
  uint64_t hostCapacity_{std::numeric_limits<uint64_t>::max()};
  // Bytes in each tier, including payloads being added.
  uint64_t deviceBytes_{0};
  uint64_t hostBytes_{0};
  uint64_t payloads_{0};
};

} // namespace facebook::velox::ucx_exchange
