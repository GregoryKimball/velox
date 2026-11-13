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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAggregationNode.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/RoundRobinPartitionFunction.h"

#include <algorithm>

namespace facebook::velox::cudf_velox {
namespace {

using core::AggregationNode;
using core::PlanNodePtr;

bool sameType(const RowTypePtr& lhs, const RowTypePtr& rhs) {
  return *lhs == *rhs;
}

void checkType(const RowTypePtr& actual, const RowTypePtr& expected) {
  VELOX_USER_CHECK(
      sameType(actual, expected),
      "Incompatible compact aggregation state schemas: {} vs {}",
      actual->toString(),
      expected->toString());
}

// HashPartitionFunctionSpec has no schema/key accessors. Rebuild its schema
// through its serde, rejecting state-column keys and constant-key specs.
core::PartitionFunctionSpecPtr rebindPartition(
    const core::PartitionFunctionSpec& spec,
    const RowTypePtr& oldType,
    const RowTypePtr& newType) {
  if (dynamic_cast<const core::GatherPartitionFunctionSpec*>(&spec)) {
    return std::make_shared<core::GatherPartitionFunctionSpec>();
  }
  VELOX_USER_CHECK(
      dynamic_cast<const exec::HashPartitionFunctionSpec*>(&spec),
      "Compact aggregation requires hash or gather partitioning");
  auto serialized = spec.serialize();
  VELOX_USER_CHECK(
      serialized["constants"].empty(),
      "Compact aggregation does not support constant partition keys");
  std::vector<column_index_t> channels;
  for (const auto& key : serialized["keyChannels"]) {
    const auto channel = key.asInt();
    VELOX_USER_CHECK_GE(channel, 0);
    VELOX_USER_CHECK_LT(channel, oldType->size());
    VELOX_USER_CHECK(
        *oldType->childAt(channel) == *newType->childAt(channel),
        "Cannot partition on a compact aggregation state column");
    channels.push_back(channel);
  }
  return std::make_shared<exec::HashPartitionFunctionSpec>(
      newType, std::move(channels));
}

void checkFields(const core::TypedExprPtr& expr, const RowTypePtr& inputType) {
  if (const auto* field =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get());
      field && field->isInputColumn()) {
    VELOX_USER_CHECK(
        *field->type() == *inputType->findChild(field->name()),
        "Expression references an incompatible compact state column: {}",
        field->name());
  }
  for (const auto& input : expr->inputs()) {
    checkFields(input, inputType);
  }
}

// Prestissimo puts a Project that only renames and reorders columns on each
// source of a local exchange.
const core::ProjectNode* asPassThroughProject(const PlanNodePtr& node) {
  const auto* project = dynamic_cast<const core::ProjectNode*>(node.get());
  if (!project || node->name() != "Project") {
    return nullptr;
  }
  for (const auto& projection : project->projections()) {
    const auto* field =
        dynamic_cast<const core::FieldAccessTypedExpr*>(projection.get());
    if (!field || !field->isInputColumn()) {
      return nullptr;
    }
  }
  return project;
}

// Returns whether 'output' partitions its rows with a connector's function
// (Hive buckets) rather than one of the engine's. A materialized exchange does,
// and a later fragment writes its rows to a table; but so does an exchange that
// Presto aligns with a bucketed table's buckets. State that crosses the latter
// keeps VARBINARY here while its consumer reads it from an exchange and
// expects compact state; the compact SUM paths then fail on the input type.
// TODO: Decide from the query's exchange materialization setting instead.
bool partitionsForTable(const core::PartitionedOutputNode& output) {
  const auto* spec = output.partitionFunctionSpecPtr().get();
  return spec != nullptr &&
      !dynamic_cast<const exec::HashPartitionFunctionSpec*>(spec) &&
      !dynamic_cast<const exec::RoundRobinPartitionFunctionSpec*>(spec) &&
      !dynamic_cast<const core::GatherPartitionFunctionSpec*>(spec);
}

// Returns whether the aggregation state reaching a FINAL or INTERMEDIATE
// aggregation through 'node' is read back from a table, the reading end of a
// materialized exchange.
bool readsStateFromTable(const PlanNodePtr& node) {
  if (dynamic_cast<const core::TableScanNode*>(node.get())) {
    return true;
  }
  if (!dynamic_cast<const core::LocalPartitionNode*>(node.get()) &&
      !dynamic_cast<const core::ProjectNode*>(node.get()) &&
      !dynamic_cast<const core::FilterNode*>(node.get())) {
    return false;
  }
  const auto& sources = node->sources();
  return !sources.empty() &&
      std::all_of(sources.begin(), sources.end(), [](const auto& source) {
        return readsStateFromTable(source);
      });
}

bool hasCompactOutput(const PlanNodePtr& node) {
  if (const auto* aggregation =
          dynamic_cast<const CudfAggregationNode*>(node.get())) {
    return aggregation->step() == AggregationNode::Step::kPartial ||
        aggregation->step() == AggregationNode::Step::kIntermediate;
  }
  if (dynamic_cast<const core::LocalPartitionNode*>(node.get())) {
    for (const auto& source : node->sources()) {
      if (hasCompactOutput(source)) {
        return true;
      }
    }
  }
  if (asPassThroughProject(node)) {
    return hasCompactOutput(node->sources()[0]);
  }
  return false;
}

// 'keepState': the state of PARTIAL aggregations reaching 'node''s output keeps
// its VARBINARY type, because it leaves the task through a materialized
// exchange (partitionsForTable()).
PlanNodePtr rewriteNode(
    const PlanNodePtr& node,
    const RowTypePtr& expected,
    bool compactInput = false,
    bool keepState = false);

// Rewrites a pass-through Project on a source of a local exchange. Compact
// state passes through it: each output column has the type of the source
// column it forwards. `expected` applies to the Project's output.
PlanNodePtr rewritePassThroughProject(
    const PlanNodePtr& node,
    const core::ProjectNode& project,
    const RowTypePtr& expected,
    bool compactInput,
    bool keepState) {
  const auto& source = project.sources()[0];
  const auto& sourceType = source->outputType();
  auto fieldName = [](const core::TypedExprPtr& projection) {
    return static_cast<const core::FieldAccessTypedExpr&>(*projection).name();
  };

  RowTypePtr sourceExpected;
  if (expected) {
    auto types = sourceType->children();
    std::vector<bool> assigned(types.size(), false);
    for (size_t i = 0; i < project.projections().size(); ++i) {
      const auto name = fieldName(project.projections()[i]);
      const auto channel = sourceType->getChildIdx(name);
      VELOX_USER_CHECK(
          !assigned[channel] || *types[channel] == *expected->childAt(i),
          "Conflicting compact state requirements for {}",
          name);
      types[channel] = expected->childAt(i);
      assigned[channel] = true;
    }
    sourceExpected = ROW(sourceType->names(), std::move(types));
  }

  auto rewritten = rewriteNode(source, sourceExpected, compactInput, keepState);
  if (rewritten == source) {
    return node;
  }
  std::vector<core::TypedExprPtr> projections;
  for (const auto& projection : project.projections()) {
    const auto name = fieldName(projection);
    projections.push_back(std::make_shared<core::FieldAccessTypedExpr>(
        rewritten->outputType()->findChild(name), name));
  }
  return core::ProjectNode::Builder(project)
      .projections(std::move(projections))
      .source(rewritten)
      .build();
}

// expected is supplied by a consumer, never inferred from a VARBINARY exchange
// column. A producer instead declares its state using original rawInputTypes.
PlanNodePtr rewriteNode(
    const PlanNodePtr& node,
    const RowTypePtr& expected,
    bool compactInput,
    bool keepState) {
  if (compactInput) {
    VELOX_USER_CHECK(
        dynamic_cast<const AggregationNode*>(node.get()) ||
            dynamic_cast<const core::LocalPartitionNode*>(node.get()) ||
            dynamic_cast<const core::ExchangeNode*>(node.get()),
        "Unsupported compact aggregation state source: {} ({})",
        node->name(),
        node->id());
  }
  auto finish = [&](PlanNodePtr result) {
    if (expected) {
      checkType(result->outputType(), expected);
    }
    return result;
  };

  if (const auto* aggregation =
          dynamic_cast<const AggregationNode*>(node.get())) {
    if (compactInput) {
      VELOX_USER_CHECK(
          aggregation->step() == AggregationNode::Step::kPartial ||
              aggregation->step() == AggregationNode::Step::kIntermediate,
          "Compact state must come from a PARTIAL or INTERMEDIATE aggregation");
    }
    if (const auto* cudfNode =
            dynamic_cast<const CudfAggregationNode*>(node.get())) {
      auto source = rewriteNode(
          node->sources()[0],
          node->sources()[0]->outputType(),
          aggregation->step() != AggregationNode::Step::kPartial);
      if (source == node->sources()[0]) {
        return finish(node);
      }
      std::vector<TypePtr> intermediateTypes;
      for (size_t i = 0; i < aggregation->aggregates().size(); ++i) {
        intermediateTypes.push_back(cudfNode->intermediateType(i));
      }
      auto rewritten =
          AggregationNode::Builder(*aggregation).source(source).build();
      return finish(
          std::make_shared<CudfAggregationNode>(
              *rewritten, std::move(intermediateTypes)));
    }
    auto aggregates = aggregation->aggregates();
    std::vector<TypePtr> intermediateTypes(aggregates.size());
    auto inputType = aggregation->sources()[0]->outputType();
    auto inputTypes = inputType->children();
    bool compact = false;
    const auto step = aggregation->step();
    const bool rawInput = step == AggregationNode::Step::kPartial ||
        step == AggregationNode::Step::kSingle;
    // A materialized exchange writes the state into a table through a fragment
    // that cannot tell state columns from others, so the state crossing it
    // keeps its VARBINARY type at both ends. An INTERMEDIATE aggregation has
    // one state type for its input and its output and cannot be one of them.
    const bool stateFromTable =
        !rawInput && readsStateFromTable(aggregation->sources()[0]);
    VELOX_USER_CHECK(
        step != AggregationNode::Step::kIntermediate ||
            !(keepState || stateFromTable),
        "Compact aggregation state cannot cross a materialized exchange at an INTERMEDIATE aggregation: {}",
        node->id());
    for (size_t i = 0; i < aggregates.size(); ++i) {
      if (step == AggregationNode::Step::kSingle || stateFromTable ||
          (keepState && step == AggregationNode::Step::kPartial)) {
        continue;
      }
      auto& aggregate = aggregates[i];
      auto stateType = CudfAggregationNode::compactIntermediateType(aggregate);
      if (!stateType) {
        continue;
      }
      compact = true;
      intermediateTypes[i] = stateType;
      const auto& field = static_cast<const core::FieldAccessTypedExpr&>(
          *aggregate.call->inputs()[0]);
      const auto channel = inputType->getChildIdx(field.name());
      const auto originalInput =
          rawInput ? aggregate.rawInputTypes[0] : VARBINARY();
      VELOX_USER_CHECK(
          *field.type() == *originalInput ||
              (!rawInput && *field.type() == *stateType),
          "Unexpected Decimal64 SUM input type");
      VELOX_USER_CHECK(
          *inputType->childAt(channel) == *field.type(),
          "Decimal64 SUM input does not match its source");
      const bool partialOutput = step == AggregationNode::Step::kPartial ||
          step == AggregationNode::Step::kIntermediate;
      // A final step keeps its own decimal result type, which equals a
      // DECIMAL(38, s) state but not a 64-bit one.
      VELOX_USER_CHECK(
          partialOutput ? *aggregate.call->type() == *VARBINARY() ||
                  *aggregate.call->type() == *stateType
                        : aggregate.call->type()->isDecimal() &&
                  getDecimalPrecisionScale(*aggregate.call->type()).second ==
                      getDecimalPrecisionScale(*stateType).second,
          "Unexpected Decimal64 SUM result type");
      if (!rawInput) {
        VELOX_USER_CHECK(
            *inputTypes[channel] == *inputType->childAt(channel) ||
                *inputTypes[channel] == *stateType,
            "Conflicting compact state requirements for {}",
            field.name());
        inputTypes[channel] = stateType;
      }
      aggregate.call = std::make_shared<core::CallTypedExpr>(
          partialOutput ? stateType : aggregate.call->type(),
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(
                  rawInput ? aggregate.rawInputTypes[0] : stateType,
                  field.name())},
          aggregate.call->name());
    }
    auto source = rewriteNode(
        aggregation->sources()[0],
        ROW(inputType->names(), std::move(inputTypes)),
        compact && !rawInput);
    for (const auto& aggregate : aggregates) {
      checkFields(aggregate.call, source->outputType());
      if (aggregate.mask) {
        checkFields(aggregate.mask, source->outputType());
      }
      for (const auto& key : aggregate.sortingKeys) {
        checkFields(key, source->outputType());
      }
    }
    for (const auto& key : aggregation->groupingKeys()) {
      checkFields(key, source->outputType());
    }
    if (!compact && source == aggregation->sources()[0]) {
      return finish(node);
    }
    auto rewritten = AggregationNode::Builder(*aggregation)
                         .aggregates(std::move(aggregates))
                         .source(source)
                         .build();
    if (!compact) {
      return finish(rewritten);
    }
    return finish(
        std::make_shared<CudfAggregationNode>(
            *rewritten, std::move(intermediateTypes)));
  }

  if (const auto* exchange =
          dynamic_cast<const core::ExchangeNode*>(node.get())) {
    if (compactInput || (expected && !sameType(node->outputType(), expected))) {
      VELOX_USER_CHECK(
          !dynamic_cast<const core::MergeExchangeNode*>(node.get()) &&
              exchange->transportKind() == core::TransportKind::kUcx,
          "Compact aggregation state requires an unordered UCX exchange");
    }
    if (!expected || sameType(node->outputType(), expected)) {
      return node;
    }
    return core::ExchangeNode::Builder(*exchange).outputType(expected).build();
  }

  if (const auto* local =
          dynamic_cast<const core::LocalPartitionNode*>(node.get())) {
    std::vector<PlanNodePtr> sources;
    bool changed = false;
    for (const auto& source : local->sources()) {
      const auto* project = asPassThroughProject(source);
      auto rewritten = project
          ? rewritePassThroughProject(
                source, *project, expected, compactInput, keepState)
          : rewriteNode(source, expected, compactInput, keepState);
      changed |= rewritten != source;
      sources.push_back(std::move(rewritten));
    }
    if (!changed) {
      return finish(node);
    }
    auto builder = core::LocalPartitionNode::Builder(*local);
    if (!sameType(node->outputType(), sources[0]->outputType())) {
      VELOX_USER_CHECK(!local->scaleWriter());
      builder.partitionFunctionSpec(rebindPartition(
          local->partitionFunctionSpec(),
          node->outputType(),
          sources[0]->outputType()));
    }
    return finish(builder.sources(std::move(sources)).build());
  }

  if (const auto* output =
          dynamic_cast<const core::PartitionedOutputNode*>(node.get())) {
    auto source = rewriteNode(
        output->sources()[0], nullptr, false, partitionsForTable(*output));
    VELOX_USER_CHECK(
        !hasCompactOutput(source) ||
            output->transportKind() == core::TransportKind::kUcx,
        "Compact aggregation state cannot leave a task over non-UCX output");
    if (source == output->sources()[0]) {
      for (size_t i = 0; i < output->outputType()->size(); ++i) {
        VELOX_USER_CHECK(
            *output->outputType()->childAt(i) ==
                *source->outputType()->findChild(
                    output->outputType()->nameOf(i)),
            "Partitioned output does not match its source schema");
      }
      return finish(node);
    }
    auto builder = core::PartitionedOutputNode::Builder(*output).source(source);
    if (!sameType(output->inputType(), source->outputType())) {
      VELOX_USER_CHECK(
          output->transportKind() == core::TransportKind::kUcx,
          "Compact aggregation state cannot leave a task over non-UCX output");
      for (const auto& key : output->keys()) {
        checkFields(key, source->outputType());
      }
      auto types = output->outputType()->children();
      for (size_t i = 0; i < types.size(); ++i) {
        types[i] =
            source->outputType()->findChild(output->outputType()->nameOf(i));
      }
      builder.outputType(ROW(output->outputType()->names(), std::move(types)));
      if (output->partitionFunctionSpecPtr()) {
        builder.partitionFunctionSpec(rebindPartition(
            output->partitionFunctionSpec(),
            output->inputType(),
            source->outputType()));
      }
    }
    return finish(builder.build());
  }

  // These wrappers may enclose a completed FINAL, but are not state bridges.
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get());
      project && node->name() == "Project") {
    auto source = rewriteNode(
        project->sources()[0],
        project->sources()[0]->outputType(),
        false,
        keepState);
    VELOX_USER_CHECK(
        !hasCompactOutput(source),
        "Unsupported compact aggregation state source: Project ({})",
        node->id());
    if (source == project->sources()[0]) {
      return finish(node);
    }
    return finish(core::ProjectNode::Builder(*project).source(source).build());
  }
  if (const auto* filter = dynamic_cast<const core::FilterNode*>(node.get())) {
    auto source = rewriteNode(
        filter->sources()[0],
        filter->sources()[0]->outputType(),
        false,
        keepState);
    VELOX_USER_CHECK(
        !hasCompactOutput(source),
        "Unsupported compact aggregation state source: Filter ({})",
        node->id());
    if (source == filter->sources()[0]) {
      return finish(node);
    }
    return finish(core::FilterNode::Builder(*filter).source(source).build());
  }

  std::vector<PlanNodePtr> sources;
  bool changed = false;
  for (const auto& source : node->sources()) {
    auto rewritten = rewriteNode(source, source->outputType());
    VELOX_USER_CHECK(
        !hasCompactOutput(rewritten),
        "Unsupported compact aggregation state source: {} ({})",
        node->name(),
        node->id());
    changed |= rewritten != source;
    sources.push_back(std::move(rewritten));
  }
  if (!changed) {
    return finish(node);
  }

  // These nodes may surround a completed aggregation, but cannot consume a
  // changed intermediate schema. Each source was checked against its old type.
  if (const auto* topN = dynamic_cast<const core::TopNNode*>(node.get())) {
    return finish(core::TopNNode::Builder(*topN).source(sources[0]).build());
  }
  if (const auto* orderBy =
          dynamic_cast<const core::OrderByNode*>(node.get())) {
    return finish(
        core::OrderByNode::Builder(*orderBy).source(sources[0]).build());
  }
  if (const auto* limit = dynamic_cast<const core::LimitNode*>(node.get())) {
    return finish(core::LimitNode::Builder(*limit).source(sources[0]).build());
  }
  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(node.get())) {
    return finish(
        core::HashJoinNode::Builder(*join)
            .left(sources[0])
            .right(sources[1])
            .build());
  }
  if (const auto* join = dynamic_cast<const core::MergeJoinNode*>(node.get())) {
    return finish(
        core::MergeJoinNode::Builder(*join)
            .left(sources[0])
            .right(sources[1])
            .build());
  }
  if (const auto* join =
          dynamic_cast<const core::NestedLoopJoinNode*>(node.get())) {
    return finish(
        core::NestedLoopJoinNode::Builder(*join)
            .left(sources[0])
            .right(sources[1])
            .build());
  }
  if (const auto* merge =
          dynamic_cast<const core::LocalMergeNode*>(node.get())) {
    return finish(
        core::LocalMergeNode::Builder(*merge).sources(sources).build());
  }
  VELOX_USER_FAIL(
      "Unsupported node above compact aggregation: {} ({})",
      node->name(),
      node->id());
}

} // namespace

core::PlanNodePtr CudfPlanRewriter::rewrite(const core::PlanNodePtr& root) {
  VELOX_USER_CHECK_NOT_NULL(root);
  if (!CudfConfig::getInstance().compactAggregationStateEnabled) {
    return root;
  }
  auto rewritten = rewriteNode(root, nullptr);
  // An unconsumed PARTIAL is not a valid fragment result. Only an explicit UCX
  // output may change the external schema of a fragment.
  if (!dynamic_cast<const core::PartitionedOutputNode*>(root.get())) {
    VELOX_USER_CHECK(
        !hasCompactOutput(rewritten),
        "Compact aggregation state requires UCX partitioned output");
    checkType(rewritten->outputType(), root->outputType());
  }
  return rewritten;
}

} // namespace facebook::velox::cudf_velox
