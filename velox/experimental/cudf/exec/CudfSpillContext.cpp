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

#include "velox/common/base/Exceptions.h"
#include "velox/core/QueryCtx.h"

#include <mutex>
#include <string_view>

namespace facebook::velox::cudf_velox {
namespace {

constexpr std::string_view kContextKey{"cudf.spillContext"};

std::mutex storageMutex;
// The pool new queries spill to, or nullptr when spilling is off.
std::shared_ptr<PinnedSpillPool> storage;
// The last pool created. Alive for as long as a query context or spilled data
// holds it, also after unregisterCudfSpilling().
std::weak_ptr<PinnedSpillPool> retainedStorage;
std::optional<int> retainedNumaNode;
std::mutex queryInitMutex;

} // namespace

void initializeCudfSpilling(uint64_t hostBytes, std::optional<int> numaNode) {
  std::lock_guard lock(storageMutex);
  if (hostBytes == 0) {
    storage.reset();
    return;
  }
  auto retained = retainedStorage.lock();
  if (retained) {
    VELOX_USER_CHECK(
        retained->budget()->limit() == hostBytes &&
            retainedNumaNode == numaNode,
        "The host spill pool limit and location cannot change while spilled "
        "data is still held");
  } else {
    retained = PinnedSpillPool::create(hostBytes, numaNode);
    retainedStorage = retained;
    retainedNumaNode = numaNode;
  }
  storage = std::move(retained);
}

void unregisterCudfSpilling() {
  std::lock_guard lock(storageMutex);
  storage.reset();
}

std::shared_ptr<CudfSpillContext> cudfSpillContext(core::QueryCtx& queryCtx) {
  if (auto existing = queryCtx.registry<CudfSpillContext>(kContextKey)) {
    return existing;
  }
  std::shared_ptr<PinnedSpillPool> selected;
  {
    std::lock_guard lock(storageMutex);
    selected = storage;
  }
  if (selected == nullptr) {
    return nullptr;
  }
  std::lock_guard lock(queryInitMutex);
  if (auto existing = queryCtx.registry<CudfSpillContext>(kContextKey)) {
    return existing;
  }
  auto context = std::make_shared<CudfSpillContext>();
  context->budget = selected->budget()->createQuery(
      queryCtx.queryId(), selected->budget()->limit());
  context->storage = std::move(selected);
  queryCtx.setRegistry(kContextKey, context);
  return context;
}

} // namespace facebook::velox::cudf_velox
