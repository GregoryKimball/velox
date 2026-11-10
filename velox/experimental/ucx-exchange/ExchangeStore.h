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

#include "velox/experimental/ucx-exchange/ExchangeCompression.h"
#include "velox/experimental/ucx-exchange/UcxExchangeQueue.h"

#include <folly/container/F14Map.h>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::ucx_exchange {

/// Holds exchange data that a later stage reads back file by file, in place of
/// the files of a temporary table. Payloads stay in the exchange's packed and
/// compressed format (see splitExchangePayloads()). A payload is kept in device
/// memory while the device tier is below its capacity and is moved to pinned
/// host memory otherwise. Nothing is written to disk: adding a payload that
/// fits in neither tier fails.
class ExchangeStore {
 public:
  /// The process-wide store.
  static ExchangeStore& instance();

  /// Sets the bytes of payload data the device tier may hold. A payload added
  /// while it does not fit goes to the host tier. Defaults to 0.
  void setDeviceCapacity(uint64_t bytes);

  /// Sets the bytes of payload data the host tier may hold. Defaults to no
  /// limit.
  void setHostCapacity(uint64_t bytes);

  /// Adds 'payload', which holds 'numRows' rows, to 'file' of 'table'. The
  /// payload's device data must be complete. A payload that goes to the host
  /// tier is copied on 'stream', which is synchronized before its device
  /// memory is freed.
  void add(
      std::string_view table,
      std::string_view file,
      PackedExchangePayload payload,
      vector_size_t numRows,
      cuda::stream_ref stream);

  /// Removes the payloads of 'file' of 'table' and returns them restored on
  /// the device, in the order in which they were added. Payloads in the host
  /// tier are copied to device memory from 'mr' on 'stream'. Fused FOR lanes
  /// payloads are returned before joinExchangeLanes(). Returns nothing for a
  /// file without payloads.
  std::vector<std::unique_ptr<PackedTableWithStream>> take(
      std::string_view table,
      std::string_view file,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Frees the payloads of every file of 'table' that was not taken.
  void drop(std::string_view table);

  /// Frees the payloads of every table whose name contains 'queryId'.
  void dropQuery(std::string_view queryId);

  struct Stats {
    /// Payload bytes in each tier.
    uint64_t deviceBytes{0};
    uint64_t hostBytes{0};
    /// Payloads held in either tier.
    uint64_t payloads{0};
  };

  Stats stats() const;

 private:
  // Pinned host memory from cuDF's pinned memory resource.
  class HostBuffer {
   public:
    explicit HostBuffer(std::size_t bytes);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    void* data() const {
      return data_;
    }

   private:
    void* data_;
    const std::size_t bytes_;
  };

  // A payload's metadata, which names its codec, and its data in exactly one
  // of the tiers.
  struct StoredPayload {
    std::unique_ptr<std::vector<uint8_t>> metadata;
    std::unique_ptr<rmm::device_buffer> device;
    std::unique_ptr<HostBuffer> host;
    std::size_t bytes{0};
    vector_size_t numRows{0};
  };

  using Files = folly::F14FastMap<std::string, std::vector<StoredPayload>>;

  // Subtracts the bytes and count of 'payloads' from the totals. Called with
  // 'mutex_' held.
  void forget(const std::vector<StoredPayload>& payloads);

  mutable std::mutex mutex_;
  folly::F14FastMap<std::string, Files> tables_;
  uint64_t deviceCapacity_{0};
  uint64_t hostCapacity_{std::numeric_limits<uint64_t>::max()};
  // Bytes in each tier, including payloads being added.
  uint64_t deviceBytes_{0};
  uint64_t hostBytes_{0};
  uint64_t payloads_{0};
};

} // namespace facebook::velox::ucx_exchange
