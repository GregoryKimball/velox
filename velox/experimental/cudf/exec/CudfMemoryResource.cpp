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
#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Task.h"

#include <folly/Indestructible.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <utility>
#include <variant>

namespace facebook::velox::cudf_velox::detail {

CudfMemoryResourceImpl::CudfMemoryResourceImpl(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::shared_ptr<memory::MemoryPool> pool)
    : pool_(std::move(pool)), upstream_(std::move(upstream)) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK(pool_->isLeaf(), "cuDF memory accounting requires a leaf pool");
}

void* CudfMemoryResourceImpl::allocate_sync(
    std::size_t bytes,
    std::size_t alignment) {
  if (bytes == 0) {
    return upstream_.allocate_sync(bytes, alignment);
  }

  pool_->reportExternalAllocation(bytes);
  try {
    return upstream_.allocate_sync(bytes, alignment);
  } catch (...) {
    pool_->reportExternalFree(bytes);
    throw;
  }
}

void CudfMemoryResourceImpl::deallocate_sync(
    void* pointer,
    std::size_t bytes,
    std::size_t alignment) noexcept {
  upstream_.deallocate_sync(pointer, bytes, alignment);
  if (bytes != 0) {
    pool_->reportExternalFree(bytes);
  }
}

void* CudfMemoryResourceImpl::allocate(
    cuda::stream_ref stream,
    std::size_t bytes,
    std::size_t alignment) {
  if (bytes == 0) {
    return upstream_.allocate(stream, bytes, alignment);
  }

  pool_->reportExternalAllocation(bytes);
  try {
    return upstream_.allocate(stream, bytes, alignment);
  } catch (...) {
    pool_->reportExternalFree(bytes);
    throw;
  }
}

void CudfMemoryResourceImpl::deallocate(
    cuda::stream_ref stream,
    void* pointer,
    std::size_t bytes,
    std::size_t alignment) noexcept {
  upstream_.deallocate(stream, pointer, bytes, alignment);
  if (bytes != 0) {
    // reportExternalFree() cannot fail for a correctly paired allocation. A
    // failure is an accounting invariant violation and terminates through
    // this noexcept path, as the RMM resource contract requires.
    pool_->reportExternalFree(bytes);
  }
}

} // namespace facebook::velox::cudf_velox::detail

namespace facebook::velox::cudf_velox {
namespace {

// The process GPU resource and the capacity it was first requested with.
// Never destroyed: query pools and device buffers that borrow its allocator
// and arbitrator can outlive unregisterCudf() and static teardown.
struct ProcessGpuResource {
  std::mutex mutex;
  std::shared_ptr<memory::CustomMemoryResource> resource;
  std::optional<int64_t> requestedCapacity;
};

ProcessGpuResource& processGpuResource() {
  static folly::Indestructible<ProcessGpuResource> instance;
  return *instance;
}

// Returns Velox's built-in reclaimer for the context: a query reclaimer for
// the query's GPU root, a task reclaimer for the task's GPU pool, and the
// operator-level default for node pools. Arbitration on a GPU root then
// pauses the Task and reclaims only GPU pools.
std::unique_ptr<memory::MemoryReclaimer> makeGpuReclaimer(
    const memory::ReclaimerContext& context) {
  if (const auto* query =
          std::get_if<memory::QueryReclaimerContext>(&context)) {
    return core::QueryCtx::MemoryReclaimer::create(
        query->queryCtx, query->pool);
  }
  if (const auto* task = std::get_if<memory::TaskReclaimerContext>(&context)) {
    return exec::Task::MemoryReclaimer::create(
        task->task, task->priority, std::string(task->resourceTag));
  }
  return exec::MemoryReclaimer::create();
}

} // namespace

CudfMemoryResource::CudfMemoryResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    std::shared_ptr<memory::MemoryPool> pool)
    : SharedBase(
          cuda::mr::make_shared_resource<detail::CudfMemoryResourceImpl>(
              std::move(upstream),
              std::move(pool))) {}

std::shared_ptr<memory::CustomMemoryResource> createCudfCustomMemoryResource(
    int64_t capacity) {
  VELOX_USER_CHECK_GT(capacity, 0, "GPU memory capacity must be positive");

  // RMM allocates the device bytes, which are charged through
  // reportExternalAllocation(). CustomMemoryResource requires an allocator,
  // but this one never backs a charged device allocation.
  memory::MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = capacity;
  auto allocator = std::make_shared<memory::MallocAllocator>(allocatorOptions);

  memory::MemoryArbitrator::Config arbitratorConfig{
      .kind = "SHARED",
      .capacity = capacity,
      .arbitrationStateCheckCb = exec::memoryArbitrationStateCheck,
      .extraConfigs = {
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolInitialCapacity),
           "0B"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kGlobalArbitrationEnabled),
           "true"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinReclaimBytes),
           "1MB"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinReclaimPct),
           "0"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinFreeCapacity),
           "0B"},
          {std::string(
               memory::SharedArbitrator::ExtraConfig::
                   kMemoryPoolMinFreeCapacityPct),
           "0"}}};
  auto arbitrator =
      std::make_shared<memory::SharedArbitrator>(arbitratorConfig);

  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCudfMemoryResourceTag},
      std::move(allocator),
      std::move(arbitrator),
      makeGpuReclaimer,
      capacity);
}

std::shared_ptr<memory::CustomMemoryResource> registerCudfMemoryResource(
    int64_t capacity) {
  auto& process = processGpuResource();
  std::lock_guard lock(process.mutex);
  auto& registry = memory::CustomMemoryResourceRegistry::global();
  auto registered = registry.find(std::string{kCudfMemoryResourceTag});
  if (process.resource) {
    VELOX_USER_CHECK(
        capacity == process.resource->maxCapacity() ||
            (process.requestedCapacity &&
             capacity == *process.requestedCapacity),
        "The cuDF GPU memory capacity is process-lived and cannot be changed");
    VELOX_USER_CHECK(
        !registered || registered == process.resource,
        "Cannot replace the process-lived cuDF GPU resource in the registry");
    if (!registered) {
      registry.insert(std::string{kCudfMemoryResourceTag}, process.resource);
    }
    return process.resource;
  }
  // An application-provided resource is retained just as permanently. Its
  // explicit capacity takes precedence over the computed default.
  auto resource =
      registered ? registered : createCudfCustomMemoryResource(capacity);
  if (!registered) {
    registry.insert(std::string{kCudfMemoryResourceTag}, resource);
  }
  process.requestedCapacity = capacity;
  process.resource = std::move(resource);
  return process.resource;
}

std::shared_ptr<memory::CustomMemoryResource> cudfCustomMemoryResource() {
  auto& process = processGpuResource();
  std::lock_guard lock(process.mutex);
  return process.resource;
}

void attachCudfMemoryPool(core::QueryCtx& queryCtx, int64_t maxCapacity) {
  auto resource = cudfCustomMemoryResource();
  if (resource == nullptr) {
    return;
  }
  if (queryCtx.customPool(resource->tag()) != nullptr) {
    return;
  }
  VELOX_USER_CHECK_GE(maxCapacity, 0);
  const auto capacity = maxCapacity == 0
      ? resource->maxCapacity()
      : std::min(maxCapacity, resource->maxCapacity());
  const auto name = fmt::format(
      "{}.{}",
      core::QueryCtx::generatePoolName(queryCtx.queryId()),
      kCudfMemoryResourceTag);
  // The component overload installs no reclaimer. The query reclaimer needs
  // the root to exist and is set before any Task is created.
  auto root = memory::memoryManager()->addCustomRootPool(
      name, resource->allocator(), resource->arbitrator(), capacity);
  auto reclaimer = resource->newReclaimer(
      memory::QueryReclaimerContext{&queryCtx, root.get()});
  if (reclaimer != nullptr) {
    root->setReclaimer(std::move(reclaimer));
  }
  queryCtx.addCustomPool(resource->tag(), std::move(root));
}

std::shared_ptr<CudfMemoryResourceRegistry> cudfMemoryResourceRegistry(
    core::QueryCtx& queryCtx) {
  if (auto registry = queryCtx.registry<CudfMemoryResourceRegistry>(
          kCudfMemoryResourceRegistryKey)) {
    return registry;
  }

  static std::mutex initMutex;
  std::lock_guard<std::mutex> lock(initMutex);
  if (auto registry = queryCtx.registry<CudfMemoryResourceRegistry>(
          kCudfMemoryResourceRegistryKey)) {
    return registry;
  }
  auto registry = std::make_shared<CudfMemoryResourceRegistry>();
  queryCtx.setRegistry(kCudfMemoryResourceRegistryKey, registry);
  return registry;
}

CudfMemoryResourceRegistry::Resources::Resources(
    cuda::mr::any_resource<cuda::mr::device_accessible> tempUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream,
    std::shared_ptr<memory::MemoryPool> pool)
    : tempUpstream{std::move(tempUpstream)},
      outputUpstream{std::move(outputUpstream)},
      temp{this->tempUpstream, pool} {
  if (this->tempUpstream != this->outputUpstream) {
    output.emplace(this->outputUpstream, std::move(pool));
  }
}

bool CudfMemoryResourceRegistry::Resources::matches(
    const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
    const cuda::mr::any_resource<cuda::mr::device_accessible>& outputUpstream)
    const {
  return this->tempUpstream == tempUpstream &&
      this->outputUpstream == outputUpstream;
}

CudfMemoryResourceRegistry::ResourceRefs
CudfMemoryResourceRegistry::Resources::refs() {
  auto tempRef = rmm::device_async_resource_ref{temp};
  return {
      tempRef,
      output.has_value() ? rmm::device_async_resource_ref{*output} : tempRef};
}

CudfMemoryResourceRegistry::ResourceRefs
CudfMemoryResourceRegistry::resourcesFor(
    const cuda::mr::any_resource<cuda::mr::device_accessible>& tempUpstream,
    const cuda::mr::any_resource<cuda::mr::device_accessible>& outputUpstream,
    std::shared_ptr<memory::MemoryPool> pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* poolKey = pool.get();
  auto it = resources_.find(poolKey);
  if (it != resources_.end()) {
    VELOX_CHECK(
        it->second->matches(tempUpstream, outputUpstream),
        "A cuDF pool cannot be reused with different upstream resources");
    return it->second->refs();
  }

  // Construct before inserting so that a throwing constructor cannot leave a
  // null entry that poisons the next lookup for this pool.
  auto resources = std::make_unique<Resources>(
      tempUpstream, outputUpstream, std::move(pool));
  auto [insertedIt, inserted] =
      resources_.emplace(poolKey, std::move(resources));
  VELOX_CHECK(inserted, "Duplicate cuDF memory resource pool");
  return insertedIt->second->refs();
}

void ensureCudfMemoryReclaimer(memory::MemoryPool& pool) {
  static std::mutex installMutex;
  std::lock_guard lock(installMutex);
  if (pool.reclaimer() == nullptr) {
    pool.setReclaimer(exec::MemoryReclaimer::create());
  }
}

} // namespace facebook::velox::cudf_velox
