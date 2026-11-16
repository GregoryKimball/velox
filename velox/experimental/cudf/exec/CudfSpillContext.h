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
 */#pragma once

#include "velox/experimental/cudf/exec/HostSpillBudget.h"
#include "velox/experimental/cudf/exec/PinnedSpillPool.h"

#include <memory>
#include <optional>

namespace facebook::velox::core {
class QueryCtx;
}

namespace facebook::velox::cudf_velox {

/// What one query needs to move device memory to pinned host memory and back.
/// Host memory is the only spill target: there is no disk path.
struct CudfSpillContext {
  /// The process pool of pinned host memory that holds spilled data.
  std::shared_ptr<PinnedSpillPool> storage;

  /// The query's share of the host spill limit. Every spilled byte is
  /// admitted here before it is copied.
  std::shared_ptr<HostSpillBudget> budget;
};

/// Creates the process pool of pinned host memory for spilled data, limited
/// to 'hostBytes', on NUMA node 'numaNode' if given. Zero 'hostBytes' turns
/// spilling off. Called at startup. The limit cannot change while spilled
/// data from an earlier call is still held.
void initializeCudfSpilling(uint64_t hostBytes, std::optional<int> numaNode);

/// Turns spilling off for queries that start afterwards. Queries that already
/// have a spill context keep it and the pool.
void unregisterCudfSpilling();

/// Returns the spill context of 'queryCtx', creating it on first use, or
/// nullptr when spilling is off. The query's budget is limited to the process
/// limit.
std::shared_ptr<CudfSpillContext> cudfSpillContext(core::QueryCtx& queryCtx);

} // namespace facebook::velox::cudf_velox
