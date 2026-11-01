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
#include "velox/experimental/supervelox/cudf/AggregationRules.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "velox/core/Expressions.h"
#include "velox/experimental/cudf/exec/DenseGroupby.h"

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

/// An aggregation that computes its result in one step from 'raw' input,
/// either itself or a final step over a local partition of a partial step.
struct OneStep {
  const AggregationNode* aggregation;
  const AggregationNode* raw;
  /// The raw inputs of 'raw' with the result types of 'aggregation'.
  std::vector<AggregationNode::Aggregate> aggregates;
};

std::optional<OneStep> oneStep(const core::PlanNodePtr& node) {
  const auto* aggregation = dynamic_cast<const AggregationNode*>(node.get());
  if (aggregation == nullptr || aggregation->groupingKeys().empty() || aggregation->isPreGrouped() ||
      aggregation->groupId().has_value()) {
    return std::nullopt;
  }
  const AggregationNode* raw = aggregation;
  if (aggregation->step() == AggregationNode::Step::kFinal) {
    const auto* exchange = dynamic_cast<const core::LocalPartitionNode*>(aggregation->sources()[0].get());
    if (exchange == nullptr || exchange->sources().size() != 1) {
      return std::nullopt;
    }
    raw = dynamic_cast<const AggregationNode*>(exchange->sources()[0].get());
    if (raw == nullptr || raw->step() != AggregationNode::Step::kPartial ||
        names(raw->groupingKeys()) != names(aggregation->groupingKeys())) {
      return std::nullopt;
    }
  } else if (aggregation->step() != AggregationNode::Step::kSingle) {
    return std::nullopt;
  }
  OneStep out{aggregation, raw, raw->aggregates()};
  for (size_t i = 0; i < out.aggregates.size(); ++i) {
    const auto& call = out.aggregates[i].call;
    out.aggregates[i].call = std::make_shared<core::CallTypedExpr>(
        aggregation->aggregates()[i].call->type(), call->inputs(), call->name());
  }
  return out;
}

/// 'step' as a single step aggregation over pre-grouped input, or null.
core::PlanNodePtr preGroup(const OneStep& step, CacheConnector& connector, AggregationRewrite& out) {
  auto columns = names(step.raw->groupingKeys());
  const auto* scan = traceToScan(step.raw->sources()[0], columns);
  if (scan == nullptr || scan->tableHandle()->connectorId() != connector.connectorId()) {
    return nullptr;
  }
  const auto& table = scan->tableHandle()->name();
  for (const auto& column : columns) {
    if (!connector.clustered(table, column)) {
      return nullptr;
    }
  }
  out.splitKeys[table] = columns;
  out.log.push_back(fmt::format(
      "aggregation {} grouped by {} of {}, whose equal values are adjacent, is one step over pre-grouped input",
      step.aggregation->id(),
      fmt::join(columns, ", "),
      table));
  return std::make_shared<AggregationNode>(
      step.aggregation->id(),
      AggregationNode::Step::kSingle,
      step.raw->groupingKeys(),
      step.raw->groupingKeys(),
      step.raw->aggregateNames(),
      step.aggregates,
      step.raw->ignoreNullKeys(),
      /*noGroupsSpanBatches=*/false,
      std::nullopt,
      step.raw->sources()[0]);
}

struct KeySource {
  std::string table;
  std::string column;
};

/// The stored column that 'column' of 'node' takes its values from, through
/// filters, projections passing it through and joins, none of which add
/// values.
std::optional<KeySource> traceValues(const core::PlanNodePtr& node, const std::string& column) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node.get())) {
    const auto found = scan->assignments().find(column);
    if (found == scan->assignments().end()) {
      return std::nullopt;
    }
    return KeySource{scan->tableHandle()->name(), found->second->name()};
  }
  if (dynamic_cast<const core::FilterNode*>(node.get())) {
    return traceValues(node->sources()[0], column);
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    const auto index = project->outputType()->getChildIdxIfExists(column);
    if (!index.has_value()) {
      return std::nullopt;
    }
    const auto field = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(project->projections()[*index]);
    if (field == nullptr || !field->isInputColumn()) {
      return std::nullopt;
    }
    return traceValues(node->sources()[0], field->name());
  }
  if (const auto* join = dynamic_cast<const core::AbstractJoinNode*>(node.get())) {
    if (join->isLeftJoin() || join->isRightJoin() || join->isFullJoin()) {
      return std::nullopt;
    }
    for (const auto& source : join->sources()) {
      if (source->outputType()->containsChild(column)) {
        return traceValues(source, column);
      }
    }
  }
  return std::nullopt;
}

/// 'step' as a dense aggregation, or null.
core::PlanNodePtr dense(const OneStep& step, CacheConnector& connector, AggregationRewrite& out) {
  constexpr int64_t kMaxSlots = int64_t{1} << 28;
  if (step.raw->groupingKeys().size() != 1 || step.aggregates.empty()) {
    return nullptr;
  }
  for (const auto& aggregate : step.aggregates) {
    if (!cudf_velox::isDenseAggregate(aggregate)) {
      return nullptr;
    }
  }
  const auto& key = step.raw->groupingKeys()[0];
  if (!(key->type()->isBigint() || key->type()->isInteger() || key->type()->isDate())) {
    return nullptr;
  }
  const auto source = traceValues(step.raw->sources()[0], key->name());
  if (!source.has_value()) {
    return nullptr;
  }
  const Column& column = connector.store().metadata(source->table, source->column);
  const int64_t slots = column.max - column.min + 1;
  if ((column.kind != Kind::kInt && column.kind != Kind::kDate) || slots > kMaxSlots ||
      slots > connector.store().rows(source->table)) {
    return nullptr;
  }
  out.log.push_back(fmt::format(
      "aggregation {} grouped by {}, which takes the values {}..{} of {}.{}, adds into {} slots indexed by key",
      step.aggregation->id(),
      key->name(),
      column.min,
      column.max,
      source->table,
      source->column,
      slots));
  return std::make_shared<cudf_velox::DenseAggregationNode>(
      step.aggregation->id(),
      step.raw->groupingKeys(),
      step.raw->aggregateNames(),
      step.aggregates,
      step.raw->sources()[0],
      column.min,
      column.max);
}

core::PlanNodePtr rewrite(
    const core::PlanNodePtr& node,
    CacheConnector& connector,
    const AggregationRules& rules,
    AggregationRewrite& out) {
  std::vector<core::PlanNodePtr> sources;
  bool changed = false;
  for (const auto& source : node->sources()) {
    sources.push_back(rewrite(source, connector, rules, out));
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
  const auto step = oneStep(current);
  if (!step.has_value()) {
    return current;
  }
  if (rules.preGroup) {
    if (auto grouped = preGroup(*step, connector, out)) {
      return grouped;
    }
  }
  if (rules.dense) {
    if (auto indexed = dense(*step, connector, out)) {
      return indexed;
    }
  }
  return current;
}

} // namespace

AggregationRewrite rewriteAggregations(
    const core::PlanNodePtr& plan,
    CacheConnector& connector,
    const AggregationRules& rules) {
  AggregationRewrite out;
  out.plan = rewrite(plan, connector, rules, out);
  return out;
}

} // namespace facebook::velox::supervelox::cudf_backend
