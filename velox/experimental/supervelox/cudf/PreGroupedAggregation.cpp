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
#include "velox/experimental/supervelox/cudf/PreGroupedAggregation.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "velox/core/Expressions.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

using core::AggregationNode;

std::vector<std::string> names(const std::vector<core::FieldAccessTypedExprPtr>& fields) {
  std::vector<std::string> out;
  for (const auto& field : fields) {
    out.push_back(field->name());
  }
  return out;
}

/// The table scan feeding 'node' through filters and projections that pass
/// 'columns' through unchanged, renaming 'columns' to the table's column
/// names. Null if anything else lies between.
const core::TableScanNode* traceToScan(const core::PlanNodePtr& node, std::vector<std::string>& columns) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node.get())) {
    for (auto& column : columns) {
      const auto found = scan->assignments().find(column);
      if (found == scan->assignments().end()) {
        return nullptr;
      }
      column = found->second->name();
    }
    return scan;
  }
  if (dynamic_cast<const core::FilterNode*>(node.get())) {
    return traceToScan(node->sources()[0], columns);
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    for (auto& column : columns) {
      const auto index = project->outputType()->getChildIdxIfExists(column);
      if (!index.has_value()) {
        return nullptr;
      }
      const auto field = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(project->projections()[*index]);
      if (field == nullptr || !field->isInputColumn()) {
        return nullptr;
      }
      column = field->name();
    }
    return traceToScan(node->sources()[0], columns);
  }
  return nullptr;
}

/// 'node' over 'sources' in place of its own, or null for a kind of node
/// this does not rebuild.
core::PlanNodePtr withSources(const core::PlanNodePtr& node, const std::vector<core::PlanNodePtr>& sources) {
  const auto* raw = node.get();
  if (const auto* filter = dynamic_cast<const core::FilterNode*>(raw)) {
    return core::FilterNode::Builder(*filter).source(sources[0]).build();
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(raw)) {
    return core::ProjectNode::Builder(*project).source(sources[0]).build();
  }
  if (const auto* aggregation = dynamic_cast<const AggregationNode*>(raw)) {
    return AggregationNode::Builder(*aggregation).source(sources[0]).build();
  }
  if (const auto* exchange = dynamic_cast<const core::LocalPartitionNode*>(raw)) {
    return core::LocalPartitionNode::Builder(*exchange).sources(sources).build();
  }
  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(raw)) {
    return core::HashJoinNode::Builder(*join).left(sources[0]).right(sources[1]).build();
  }
  if (const auto* join = dynamic_cast<const core::NestedLoopJoinNode*>(raw)) {
    return core::NestedLoopJoinNode::Builder(*join).left(sources[0]).right(sources[1]).build();
  }
  if (const auto* orderBy = dynamic_cast<const core::OrderByNode*>(raw)) {
    return core::OrderByNode::Builder(*orderBy).source(sources[0]).build();
  }
  if (const auto* topN = dynamic_cast<const core::TopNNode*>(raw)) {
    return core::TopNNode::Builder(*topN).source(sources[0]).build();
  }
  if (const auto* limit = dynamic_cast<const core::LimitNode*>(raw)) {
    return core::LimitNode::Builder(*limit).source(sources[0]).build();
  }
  if (const auto* single = dynamic_cast<const core::EnforceSingleRowNode*>(raw)) {
    return core::EnforceSingleRowNode::Builder(*single).source(sources[0]).build();
  }
  if (const auto* unique = dynamic_cast<const core::AssignUniqueIdNode*>(raw)) {
    return core::AssignUniqueIdNode::Builder(*unique).source(sources[0]).build();
  }
  return nullptr;
}

/// 'node' as a single step aggregation over pre-grouped input, or 'node'.
core::PlanNodePtr preGroup(const core::PlanNodePtr& node, CacheConnector& connector, PreGrouping& out) {
  const auto* aggregation = dynamic_cast<const AggregationNode*>(node.get());
  if (aggregation == nullptr || aggregation->groupingKeys().empty() || aggregation->isPreGrouped() ||
      aggregation->groupId().has_value()) {
    return node;
  }
  const AggregationNode* raw = aggregation;
  if (aggregation->step() == AggregationNode::Step::kFinal) {
    const auto* exchange = dynamic_cast<const core::LocalPartitionNode*>(aggregation->sources()[0].get());
    if (exchange == nullptr || exchange->sources().size() != 1) {
      return node;
    }
    raw = dynamic_cast<const AggregationNode*>(exchange->sources()[0].get());
    if (raw == nullptr || raw->step() != AggregationNode::Step::kPartial ||
        names(raw->groupingKeys()) != names(aggregation->groupingKeys())) {
      return node;
    }
  } else if (aggregation->step() != AggregationNode::Step::kSingle) {
    return node;
  }
  auto columns = names(raw->groupingKeys());
  const auto* scan = traceToScan(raw->sources()[0], columns);
  if (scan == nullptr || scan->tableHandle()->connectorId() != connector.connectorId()) {
    return node;
  }
  const auto& table = scan->tableHandle()->name();
  for (const auto& column : columns) {
    if (!connector.clustered(table, column)) {
      return node;
    }
  }
  // The raw inputs of the partial step with the result types of the final.
  std::vector<AggregationNode::Aggregate> aggregates = raw->aggregates();
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& call = aggregates[i].call;
    aggregates[i].call = std::make_shared<core::CallTypedExpr>(
        aggregation->aggregates()[i].call->type(), call->inputs(), call->name());
  }
  out.splitKeys[table] = columns;
  out.log.push_back(fmt::format(
      "aggregation {} grouped by {} of {}, whose equal values are adjacent, is one step over pre-grouped input",
      aggregation->id(),
      fmt::join(columns, ", "),
      table));
  return std::make_shared<AggregationNode>(
      aggregation->id(),
      AggregationNode::Step::kSingle,
      raw->groupingKeys(),
      raw->groupingKeys(),
      raw->aggregateNames(),
      aggregates,
      raw->ignoreNullKeys(),
      /*noGroupsSpanBatches=*/false,
      std::nullopt,
      raw->sources()[0]);
}

core::PlanNodePtr rewrite(const core::PlanNodePtr& node, CacheConnector& connector, PreGrouping& out) {
  std::vector<core::PlanNodePtr> sources;
  bool changed = false;
  for (const auto& source : node->sources()) {
    sources.push_back(rewrite(source, connector, out));
    changed |= sources.back() != source;
  }
  auto current = node;
  if (changed) {
    if (auto rebuilt = withSources(node, sources)) {
      current = rebuilt;
    } else {
      out.log.push_back(fmt::format("cannot rebuild {} over rewritten inputs", node->toString()));
    }
  }
  return preGroup(current, connector, out);
}

} // namespace

PreGrouping preGroupAggregations(const core::PlanNodePtr& plan, CacheConnector& connector) {
  PreGrouping out;
  out.plan = rewrite(plan, connector, out);
  return out;
}

} // namespace facebook::velox::supervelox::cudf_backend
