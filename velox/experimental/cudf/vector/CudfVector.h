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

#include "velox/common/memory/MemoryPool.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <cuda/stream>

#include <memory>
#include <span>
#include <utility>
#include <variant>

namespace facebook::velox::cudf_velox {

// Vector class which holds GPU data from cuDF.
// Can be constructed either from an owned cudf::table or from packed_table.
// When constructed from packed_table, the data remains packed and tabView_
// references the table view inside packed_table without copying the underlying
// GPU data.
//
// CudfVector assumes that the input table data is already ready to consume.
// Constructors and rebindStream() do not insert CUDA stream waits for work that
// populated the input buffers; callers must establish that ordering before
// constructing or rebinding a CudfVector.
class CudfVector : public RowVector {
 public:
  /// Constructs a CudfVector from an owned cudf::table. The vector retains
  /// 'allocationOwner', or threadAllocationOwner() when it is null, for as
  /// long as it holds the table.
  /// The table's buffers are freed on 'stream', or on the stream passed to a
  /// later rebindStream().
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::table>&& table,
      cuda::stream_ref stream,
      std::shared_ptr<void> allocationOwner = nullptr);

  /// Constructs a CudfVector from packed_table.
  /// The packed data is retained and tabView_ references the table view inside
  /// packed_table. This avoids copying the underlying GPU data.
  /// The packed buffer is freed on 'stream', or on the stream passed to a later
  /// rebindStream().
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::packed_table>&& packedTable,
      cuda::stream_ref stream,
      std::shared_ptr<void> allocationOwner = nullptr);

  /// Returns what a vector constructed on this thread retains besides its
  /// table, or nullptr. The device memory resources that are in effect on the
  /// thread set it to whatever must outlive memory allocated from them.
  static std::shared_ptr<void> threadAllocationOwner();

  /// Replaces threadAllocationOwner() and returns the previous value.
  static std::shared_ptr<void> exchangeThreadAllocationOwner(
      std::shared_ptr<void> owner);

  cuda::stream_ref stream() const {
    return stream_;
  }

  /// Returns a view of the underlying table. Throws after release().
  cudf::table_view getTableView() const;

  /// Releases ownership of the underlying table.
  /// If constructed from packed_table, materializes a table from the view
  /// first (which copies the data). Throws if called more than once.
  std::unique_ptr<cudf::table> release();

  /// Rebinds the vector's device buffers (owned or packed) to use 'stream' for
  /// future deallocation.
  /// This only changes future stream/deallocation association. It does not make
  /// 'stream' wait on the previous stream or any producer stream.
  /// Returns false if the storage was already taken by release().
  bool rebindStream(cuda::stream_ref stream);

  uint64_t estimateFlatSize() const override;

  /// Returns true if the vector holds a table it owns. release() then moves
  /// the table out. For a packed table release() copies it on the device,
  /// which is not safe to start while memory is being reclaimed.
  bool hasOwnedTable() const;

 private:
  uint64_t retainedSizeImpl(uint64_t& totalStringBufferSize) const override;

  // Declared before the table storage so that they are destroyed after the
  // table's buffers.
  std::shared_ptr<void> allocationOwner_;
  // RowVector keeps only a raw pointer to its pool. A vector that is parked
  // off the device or handed to another operator can outlive the operator
  // whose pool it was made with.
  std::shared_ptr<velox::memory::MemoryPool> cpuPoolOwner_;

  // Storage for either an owned table or packed table.
  // Only one is active at a time - using variant enforces this at compile time.
  using TableStorage = std::variant<
      std::unique_ptr<cudf::table>,
      std::unique_ptr<cudf::packed_table>>;
  TableStorage tableStorage_;

  // Views either the owned table or the packed table. Invalid after release().
  cudf::table_view tabView_;

  cuda::stream_ref stream_;
  uint64_t flatSize_;
};

using CudfVectorPtr = std::shared_ptr<CudfVector>;

/// Orders CudfVector deallocations after work on a target stream.
///
/// Rebinds each input's device buffers to 'stream' so stream-ordered memory
/// resources free the inputs after prior work on that stream. Throws if any
/// input was passed to CudfVector::release(), after rebinding the others.
void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox
