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

struct AggregationRules {
  /// A scan from the cache reads its columns as stored, evaluates its filters
  /// over them in one kernel, compacts the encoded columns of the rows that
  /// pass and decodes them in another. Dictionary strings are tested once
  /// per dictionary entry and the rows by their codes. A fused aggregation
  /// reads its scan's columns encoded.
  bool decode{false};
  /// With 'decode', the filters and projections above the scan are
  /// evaluated in the same two kernels.
  bool project{false};
  /// Aggregations whose grouping keys are columns of a table scanned from the
  /// cache in which rows with equal keys are adjacent become single step
  /// aggregations over pre-grouped input. Between the scan and the
  /// aggregation only filters and projections passing the keys through are
  /// allowed, which keep rows in order, and the table's splits keep each
  /// group whole.
  bool preGroup{true};
  /// Aggregations grouped by keys whose values the store bounds to a range
  /// no larger than the table become one direct aggregation that all drivers
  /// add into, indexed by key. The keys may pass through filters,
  /// projections and joins, which only narrow their values.
  bool direct{true};
  /// Aggregations grouped by keys the store bounds to few groups, or by no
  /// keys, over filters and projections become one reduction that evaluates
  /// the filters and the aggregates' inputs from the columns below them. A
  /// scan from the cache below keeps its filters only to skip zones.
  bool fuse{false};
  /// Semi and anti joins on a key a scanned table is clustered by, filtered
  /// by an inequality between a column of each side, as for exists and not
  /// exists subqueries, become joins with the min and max of the tested
  /// column per key, aggregated over pre-grouped input.
  bool existsAggregation{false};
  /// An inner join of an aggregation on its one grouping key with a side
  /// whose key values come from a scan with filters semi joins the
  /// aggregation's input with that scan, through inner joins on the key to
  /// both of their sides, so that groups the join drops are not aggregated.
  bool keyPushdown{false};
  /// With 'decode', the fused scan below the probe side of an inner or semi
  /// join whose build side keys come from a filtered scan passes only rows
  /// whose key is one of those, tested before the rows are decoded, as a
  /// dynamic filter would.
  bool keyFilter{false};
};

struct AggregationRewrite {
  core::PlanNodePtr plan;
  /// Per scanned table, the columns its splits must not cut a run of.
  std::map<std::string, std::vector<std::string>> splitKeys;
  std::vector<std::string> log;
};

/// Applies 'rules' to the aggregations of 'plan'. A partial aggregation,
/// local partition and final aggregation that a rule applies to become one
/// aggregation in the pipeline of the partial one.
AggregationRewrite rewriteAggregations(
    const core::PlanNodePtr& plan,
    CacheConnector& connector,
    const AggregationRules& rules);

} // namespace facebook::velox::supervelox::cudf_backend
