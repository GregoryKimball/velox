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
 */#include "velox/experimental/cudf/exec/CudfSpillContext.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/SpilledCudfVector.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/ScopeGuard.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <type_traits>
#include <utility>

#include "velox/experimental/cudf/CudfNoDefaults.h"

namespace facebook::velox::cudf_velox {
namespace {

// The buffer types a cudf::column is made of, taken from cudf::column itself
// so that this file does not depend on how a cuDF version types null masks.
using ColumnDataBufferPtr = decltype(cudf::column::contents::data);
using ColumnMaskBufferPtr = decltype(cudf::column::contents::null_mask);

// Returns the size in bytes of a device buffer whose size() counts either
// bytes or elements.
template <typename Buffer>
std::size_t bufferBytes(const Buffer& buffer) {
  using Element =
      std::remove_cv_t<std::remove_pointer_t<decltype(buffer.data())>>;
  if constexpr (std::is_void_v<Element>) {
    return buffer.size();
  } else {
    return buffer.size() * sizeof(Element);
  }
}

struct HostColumn;

// The device buffers of one column and of its children, taken out of a
// cudf::column and put back into one.
struct DeviceColumn {
  explicit DeviceColumn(std::unique_ptr<cudf::column> column)
      : type(column->type()),
        size(column->size()),
        nullCount(column->null_count()) {
    auto contents = column->release();
    data = std::move(contents.data);
    nullMask = std::move(contents.null_mask);
    children.reserve(contents.children.size());
    for (auto& child : contents.children) {
      children.emplace_back(std::move(child));
    }
  }

  // Allocates buffers for the column 'host' describes. Does not copy.
  DeviceColumn(
      const HostColumn& host,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  uint64_t bytes() const {
    uint64_t result = bufferBytes(*data) + bufferBytes(*nullMask);
    for (const auto& child : children) {
      result += child.bytes();
    }
    return result;
  }

  std::unique_ptr<cudf::column> releaseColumn() && {
    std::vector<std::unique_ptr<cudf::column>> releasedChildren;
    releasedChildren.reserve(children.size());
    for (auto& child : children) {
      releasedChildren.push_back(std::move(child).releaseColumn());
    }
    return std::make_unique<cudf::column>(
        type,
        size,
        std::move(*data),
        std::move(*nullMask),
        nullCount,
        std::move(releasedChildren));
  }

  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type nullCount;
  ColumnDataBufferPtr data;
  ColumnMaskBufferPtr nullMask;
  std::vector<DeviceColumn> children;
};

// One device buffer's bytes in pinned host memory. 'data' is empty until the
// buffer has been admitted and allocated.
struct HostBuffer {
  explicit HostBuffer(std::size_t bytes) : bytes(bytes) {}

  std::size_t bytes;
  PinnedSpillPool::Allocation data;
};

// The host copy of one column and of its children.
struct HostColumn {
  // A fixed-width column without nulls: the rows 'view' selects.
  explicit HostColumn(cudf::column_view view)
      : type(view.type()),
        size(view.size()),
        nullCount(0),
        data(static_cast<std::size_t>(view.size()) * cudf::size_of(view.type())),
        nullMask(0) {}

  explicit HostColumn(const DeviceColumn& device)
      : type(device.type),
        size(device.size),
        nullCount(device.nullCount),
        data(bufferBytes(*device.data)),
        nullMask(bufferBytes(*device.nullMask)) {
    children.reserve(device.children.size());
    for (const auto& child : device.children) {
      children.emplace_back(child);
    }
  }

  uint64_t bytes() const {
    uint64_t result = data.data.size() + nullMask.data.size();
    for (const auto& child : children) {
      result += child.bytes();
    }
    return result;
  }

  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type nullCount;
  HostBuffer data;
  HostBuffer nullMask;
  std::vector<HostColumn> children;
};

DeviceColumn::DeviceColumn(
    const HostColumn& host,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : type(host.type), size(host.size), nullCount(host.nullCount) {
  data = std::make_unique<typename ColumnDataBufferPtr::element_type>(
      host.data.bytes, stream, mr);
  // The mask is made by cuDF so that it has the type and the padding a column
  // expects. An unallocated mask stands for "no nulls".
  nullMask = std::make_unique<typename ColumnMaskBufferPtr::element_type>(
      cudf::create_null_mask(
          host.nullMask.bytes == 0 ? 0 : size,
          host.nullMask.bytes == 0 ? cudf::mask_state::UNALLOCATED
                                   : cudf::mask_state::UNINITIALIZED,
          stream,
          mr));
  children.reserve(host.children.size());
  for (const auto& child : host.children) {
    children.emplace_back(child, stream, mr);
  }
}

// One device-to-host copy: 'source' is 'destination->bytes' long.
struct HostWrite {
  const void* source;
  HostBuffer* destination;
};

// One host-to-device copy of 'bytes' bytes.
struct HostRead {
  const HostBuffer* source;
  void* destination;
  std::size_t bytes;
};

void collectWrites(
    const DeviceColumn& device,
    HostColumn& host,
    std::vector<HostWrite>& writes) {
  if (host.data.bytes != 0) {
    writes.push_back({device.data->data(), &host.data});
  }
  if (host.nullMask.bytes != 0) {
    writes.push_back({device.nullMask->data(), &host.nullMask});
  }
  for (size_t i = 0; i < host.children.size(); ++i) {
    collectWrites(device.children[i], host.children[i], writes);
  }
}

void collectReads(
    const HostColumn& host,
    DeviceColumn& device,
    std::vector<HostRead>& reads) {
  if (host.data.bytes != 0) {
    reads.push_back({&host.data, device.data->data(), host.data.bytes});
  }
  if (host.nullMask.bytes != 0) {
    // cuDF may pad a mask differently than the one that was spilled. Both
    // cover every row, so the bytes they share are the ones that matter.
    reads.push_back(
        {&host.nullMask,
         device.nullMask->data(),
         std::min(host.nullMask.bytes, bufferBytes(*device.nullMask))});
  }
  for (size_t i = 0; i < host.children.size(); ++i) {
    collectReads(host.children[i], device.children[i], reads);
  }
}

// Allocates the admitted host buffers and queues the copies on 'stream'.
// Waits for them unless 'deferSynchronize' is set, in which case the caller
// keeps sources and destinations alive until it has waited.
void writeBuffers(
    const std::vector<HostWrite>& writes,
    PinnedSpillPool::Reservation& admission,
    cuda::stream_ref stream,
    bool deferSynchronize) {
  for (const auto& write : writes) {
    write.destination->data = admission.allocate(write.destination->bytes);
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(
            write.destination->data.data(),
            write.source,
            write.destination->bytes,
            cudaMemcpyDeviceToHost,
            stream.get()));
  }
  if (!deferSynchronize) {
    stream.sync();
  }
}

// Completes the deferred copies of a directly spilled slice when destroyed.
struct DeferredCopyGuard {
  DeferredCopyGuard() = default;
  DeferredCopyGuard(DeferredCopyGuard&&) noexcept = default;
  DeferredCopyGuard& operator=(DeferredCopyGuard&&) noexcept = default;

  ~DeferredCopyGuard() {
    if (copies) {
      try {
        copies->complete();
      } catch (...) {
      }
    }
  }

  std::shared_ptr<DeferredSliceCopies> copies;
};

} // namespace

DeferredSliceCopies::~DeferredSliceCopies() {
  try {
    complete();
  } catch (...) {
  }
}

void DeferredSliceCopies::add(
    cuda::stream_ref stream,
    std::shared_ptr<void> owner,
    std::shared_ptr<void> tableOwner) {
  std::lock_guard lock(mutex_);
  const bool known = std::any_of(
      streams_.begin(), streams_.end(), [&](const cuda::stream_ref& other) {
        return other.get() == stream.get();
      });
  if (!known) {
    streams_.push_back(stream);
  }
  // The slices of one partitioned table arrive one after the other.
  for (auto* parent : {&owner, &tableOwner}) {
    if (*parent && (parents_.empty() || parents_.back() != *parent)) {
      parents_.push_back(std::move(*parent));
    }
  }
  ++numSlices_;
}

void DeferredSliceCopies::complete() {
  std::vector<std::shared_ptr<void>> parents;
  std::exception_ptr error;
  {
    // Held while waiting, so that a second caller returns only when the
    // copies are done.
    std::lock_guard lock(mutex_);
    for (auto stream : streams_) {
      try {
        stream.sync();
      } catch (...) {
        if (!error) {
          error = std::current_exception();
        }
      }
      ++numSyncs_;
    }
    streams_.clear();
    parents.swap(parents_);
  }
  // Every copy that read these has finished or failed.
  parents.clear();
  if (error) {
    std::rethrow_exception(error);
  }
}

uint64_t DeferredSliceCopies::numSlices() const {
  std::lock_guard lock(mutex_);
  return numSlices_;
}

uint64_t DeferredSliceCopies::numSyncs() const {
  std::lock_guard lock(mutex_);
  return numSyncs_;
}

struct SpilledCudfVector::Impl {
  std::shared_ptr<memory::MemoryPool> vectorPool;
  std::shared_ptr<CudfSpillContext> context;
  TypePtr type;
  vector_size_t numRows{0};
  uint64_t deviceBytes{0};
  std::vector<HostColumn> columns;
  // Set by trySpillFlatSlice() when the copies into 'columns' were only
  // queued. Declared after 'columns' so that the copies are completed before
  // the host buffers are freed.
  DeferredCopyGuard deferredCopies;

  void waitCopied() const {
    if (deferredCopies.copies) {
      deferredCopies.copies->complete();
    }
  }
};

SpilledCudfVector::SpilledCudfVector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
  VELOX_CHECK_NOT_NULL(impl_);
}

SpilledCudfVector::~SpilledCudfVector() = default;

SpilledCudfVector::SpilledCudfVector(SpilledCudfVector&&) noexcept = default;

SpilledCudfVector& SpilledCudfVector::operator=(SpilledCudfVector&&) noexcept =
    default;

std::optional<SpilledCudfVector> SpilledCudfVector::trySpillFlatSlice(
    memory::MemoryPool* pool,
    const TypePtr& type,
    cudf::table_view view,
    cuda::stream_ref stream,
    std::shared_ptr<void> owner,
    std::shared_ptr<void> tableOwner,
    std::shared_ptr<CudfSpillContext> context,
    std::shared_ptr<DeferredSliceCopies> deferred) {
  if (context == nullptr || context->storage == nullptr ||
      context->budget == nullptr || pool == nullptr || type == nullptr ||
      view.num_rows() == 0 || view.num_columns() == 0 ||
      (owner == nullptr && tableOwner == nullptr)) {
    return std::nullopt;
  }
  uint64_t bytes = 0;
  for (const auto& column : view) {
    if (!cudf::is_fixed_width(column.type()) || column.num_children() != 0 ||
        (column.nullable() && column.null_count() != 0)) {
      return std::nullopt;
    }
    bytes +=
        static_cast<uint64_t>(column.size()) * cudf::size_of(column.type());
  }

  auto impl = std::make_unique<Impl>();
  impl->vectorPool = pool->shared_from_this();
  impl->context = context;
  impl->type = type;
  impl->numRows = static_cast<vector_size_t>(view.num_rows());
  impl->deviceBytes = bytes;
  auto admission = context->storage->reserve(context->budget, bytes);

  // A copy may be queued when a later step fails. The host destinations are
  // freed when 'impl' unwinds, so wait for the stream first.
  bool copied = false;
  SCOPE_EXIT {
    if (!copied) {
      try {
        stream.sync();
      } catch (...) {
      }
    }
  };

  impl->columns.reserve(view.num_columns());
  for (const auto& column : view) {
    impl->columns.emplace_back(column);
  }
  std::vector<HostWrite> writes;
  writes.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    const auto column = view.column(i);
    const auto* begin = static_cast<const uint8_t*>(column.head<void>()) +
        static_cast<size_t>(column.offset()) * cudf::size_of(column.type());
    writes.push_back({begin, &impl->columns[i].data});
  }
  // The slice needs no device allocation, copy or free: the table it belongs
  // to stays with the caller's owners.
  writeBuffers(writes, admission, stream, deferred != nullptr);
  if (deferred) {
    // From here 'deferred' keeps the table alive and 'impl' completes it
    // before the host buffers are read or freed. If add() throws, the guard
    // above still waits for the copies queued so far.
    deferred->add(stream, std::move(owner), std::move(tableOwner));
    impl->deferredCopies.copies = std::move(deferred);
  }
  copied = true;
  return SpilledCudfVector(std::move(impl));
}

SpilledCudfVector SpilledCudfVector::spill(
    CudfVectorPtr& resident,
    std::shared_ptr<CudfSpillContext> context) {
  VELOX_CHECK_NOT_NULL(resident);
  VELOX_CHECK_NOT_NULL(context, "Host spilling is disabled or unconfigured");
  VELOX_CHECK_NOT_NULL(context->storage);
  VELOX_CHECK_NOT_NULL(context->budget);
  VELOX_CHECK(
      resident->hasOwnedTable(),
      "A packed vector must be turned into an owned table before it spills");
  VELOX_CHECK_EQ(
      resident.use_count(),
      1,
      "Only an independently owned CudfVector can be spilled");

  const auto vectorPool = resident->pool()->shared_from_this();
  auto admission = context->storage->reserve(
      context->budget, resident->estimateFlatSize());
  const auto type = resident->type();
  const auto numRows = resident->size();
  const auto stream = resident->stream();

  auto table = resident->release();
  resident.reset();
  auto releasedColumns = table->release();
  std::vector<DeviceColumn> deviceColumns;
  deviceColumns.reserve(releasedColumns.size());
  for (auto& column : releasedColumns) {
    deviceColumns.emplace_back(std::move(column));
  }

  auto restoreResident = [&]() {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(deviceColumns.size());
    for (auto& column : deviceColumns) {
      columns.push_back(std::move(column).releaseColumn());
    }
    resident = std::make_shared<CudfVector>(
        vectorPool.get(),
        type,
        numRows,
        std::make_unique<cudf::table>(std::move(columns)),
        stream);
  };

  auto impl = std::make_unique<Impl>();
  impl->vectorPool = vectorPool;
  impl->context = std::move(context);
  impl->type = type;
  impl->numRows = numRows;
  try {
    impl->columns.reserve(deviceColumns.size());
    for (const auto& column : deviceColumns) {
      impl->columns.emplace_back(column);
      impl->deviceBytes += column.bytes();
    }
    std::vector<HostWrite> writes;
    for (size_t i = 0; i < impl->columns.size(); ++i) {
      collectWrites(deviceColumns[i], impl->columns[i], writes);
    }
    writeBuffers(writes, admission, stream, false);

    // The frees are queued on the vector's stream. Wait for them so that the
    // memory is back before the caller reports it released.
    deviceColumns.clear();
    stream.sync();
    return SpilledCudfVector(std::move(impl));
  } catch (...) {
    // A copy may have been queued before the failure. Keep its pinned
    // destination alive until the stream no longer uses it.
    try {
      stream.sync();
    } catch (...) {
    }
    restoreResident();
    throw;
  }
}

CudfVectorPtr SpilledCudfVector::restore(
    rmm::device_async_resource_ref mr) const {
  return restore(cudfGlobalStreamPool().get_stream(), mr);
}

CudfVectorPtr SpilledCudfVector::restore(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_NOT_NULL(impl_);
  impl_->waitCopied();

  std::vector<DeviceColumn> deviceColumns;
  deviceColumns.reserve(impl_->columns.size());
  for (const auto& column : impl_->columns) {
    deviceColumns.emplace_back(column, stream, mr);
  }
  std::vector<HostRead> reads;
  for (size_t i = 0; i < impl_->columns.size(); ++i) {
    collectReads(impl_->columns[i], deviceColumns[i], reads);
  }
  {
    // The destinations are freed when 'deviceColumns' unwinds. Wait for the
    // copies queued so far before that happens.
    bool completed = false;
    SCOPE_EXIT {
      if (!completed) {
        try {
          stream.sync();
        } catch (...) {
        }
      }
    };
    for (const auto& read : reads) {
      CUDF_CUDA_TRY(
          cudaMemcpyAsync(
              read.destination,
              read.source->data.data(),
              read.bytes,
              cudaMemcpyHostToDevice,
              stream.get()));
    }
    stream.sync();
    completed = true;
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(deviceColumns.size());
  for (auto& column : deviceColumns) {
    columns.push_back(std::move(column).releaseColumn());
  }
  return std::make_shared<CudfVector>(
      impl_->vectorPool.get(),
      impl_->type,
      impl_->numRows,
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
}

std::unique_ptr<cudf::table> SpilledCudfVector::tryRestoreConcatenated(
    const std::vector<const SpilledCudfVector*>& sources,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (sources.empty()) {
    return nullptr;
  }
  for (const auto* source : sources) {
    source->impl_->waitCopied();
  }
  const auto& first = *sources.front()->impl_;
  const auto numColumns = first.columns.size();
  uint64_t totalRows = 0;
  for (const auto* source : sources) {
    const auto& impl = *source->impl_;
    if (impl.columns.size() != numColumns) {
      return nullptr;
    }
    for (size_t c = 0; c < numColumns; ++c) {
      const auto& column = impl.columns[c];
      if (column.type != first.columns[c].type ||
          !cudf::is_fixed_width(column.type) || !column.children.empty() ||
          column.nullMask.bytes != 0 || column.nullCount != 0 ||
          column.data.bytes !=
              static_cast<uint64_t>(column.size) * cudf::size_of(column.type)) {
        return nullptr;
      }
    }
    totalRows += impl.numRows;
  }
  if (totalRows >
      static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max())) {
    return nullptr;
  }

  std::vector<ColumnDataBufferPtr> buffers;
  buffers.reserve(numColumns);
  for (size_t c = 0; c < numColumns; ++c) {
    buffers.push_back(
        std::make_unique<typename ColumnDataBufferPtr::element_type>(
            totalRows * cudf::size_of(first.columns[c].type), stream, mr));
  }
  {
    bool completed = false;
    SCOPE_EXIT {
      if (!completed) {
        try {
          stream.sync();
        } catch (...) {
        }
      }
    };
    for (size_t c = 0; c < numColumns; ++c) {
      auto* base = static_cast<uint8_t*>(buffers[c]->data());
      uint64_t offset = 0;
      for (const auto* source : sources) {
        const auto& column = source->impl_->columns[c];
        if (column.data.bytes != 0) {
          CUDF_CUDA_TRY(
              cudaMemcpyAsync(
                  base + offset,
                  column.data.data.data(),
                  column.data.bytes,
                  cudaMemcpyHostToDevice,
                  stream.get()));
        }
        offset += column.data.bytes;
      }
    }
    stream.sync();
    completed = true;
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numColumns);
  for (size_t c = 0; c < numColumns; ++c) {
    columns.push_back(
        std::make_unique<cudf::column>(
            first.columns[c].type,
            static_cast<cudf::size_type>(totalRows),
            std::move(*buffers[c]),
            cudf::create_null_mask(
                0, cudf::mask_state::UNALLOCATED, stream, mr),
            0));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

vector_size_t SpilledCudfVector::numRows() const {
  VELOX_CHECK_NOT_NULL(impl_);
  return impl_->numRows;
}

uint64_t SpilledCudfVector::deviceBytes() const {
  VELOX_CHECK_NOT_NULL(impl_);
  return impl_->deviceBytes;
}

uint64_t SpilledCudfVector::hostBytes() const {
  VELOX_CHECK_NOT_NULL(impl_);
  uint64_t bytes = 0;
  for (const auto& column : impl_->columns) {
    bytes += column.bytes();
  }
  return bytes;
}

} // namespace facebook::velox::cudf_velox
