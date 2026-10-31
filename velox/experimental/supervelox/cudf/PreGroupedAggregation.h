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

#include <map>
#include <string>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/experimental/supervelox/cudf/CacheConnector.h"

namespace facebook::velox::supervelox::cudf_backend {

struct PreGrouping {
  core::PlanNodePtr plan;
  /// Per scanned table, the columns its splits must not cut a run of.
  std::map<std::string, std::vector<std::string>> splitKeys;
  std::vector<std::string> log;
};

/// Aggregations whose grouping keys are columns of a table scanned from
/// 'connector' in which rows with equal keys are adjacent become single step
/// aggregations over pre-grouped input. Between the scan and the aggregation
/// only filters and projections passing the keys through are allowed, which
/// keep rows in order. A partial aggregation, local partition and final
/// aggregation over such a scan become one aggregation in the scan pipeline,
/// and the table's splits keep each group whole.
PreGrouping preGroupAggregations(const core::PlanNodePtr& plan, CacheConnector& connector);

} // namespace facebook::velox::supervelox::cudf_backend
