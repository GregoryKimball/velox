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

#include <string>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/experimental/supervelox/store/ColumnStore.h"

namespace facebook::velox::supervelox {

/// Replaces each subtree of `plan` that a fused query computes with a
/// FusedQueryNode carrying the literals the subtree was written with, and
/// keeps the rest of the plan as it was. The literals are converted to the
/// encoding of `store`. `log`, when given, receives a line per match.
core::PlanNodePtr rewriteWithFusedQueries(
    const core::PlanNodePtr& plan,
    ColumnStore& store,
    std::vector<std::string>* log = nullptr);

} // namespace facebook::velox::supervelox
