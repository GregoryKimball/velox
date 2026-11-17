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
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/HashPartitionFunction.h"

#include <typeinfo>

namespace facebook::velox::cudf_velox {
namespace {

using core::AggregationNode;

// Returns true if 'node' is a Node and not a class derived from it. A copy
// made with Node's builder would drop what a derived class adds.
template <typename Node>
bool isExactly(const core::PlanNode& node) {
  return typeid(node) == typeid(Node);
}

// Returns the single input column of 'call', or nullptr if the call does not
// read exactly one input column.
const core::FieldAccessTypedExpr* inputColumn(const core::CallTypedExpr& call) {
  if (call.inputs().size() != 1) {
    return nullptr;
  }
  const auto* field =
      dynamic_cast<const core::FieldAccessTypedExpr*>(call.inputs()[0].get());
  return field != nullptr && field->isInputColumn() ? field : nullptr;
}

// Returns true if 'type' has a column 'name' of type 'columnType'.
bool hasColumn(
    const RowTypePtr& type,
    const std::string& name,
    const Type& columnType) {
  const auto index = type->getChildIdxIfExists(name);
  return index.has_value() && *type->childAt(index.value()) == columnType;
}

// Returns true if every input column of 'aggregates' has the type that
// 'sourceType' gives the column of that name.
bool readsColumnsOf(
    const std::vector<AggregationNode::Aggregate>& aggregates,
    const RowTypePtr& sourceType) {
  for (const auto& aggregate : aggregates) {
    for (const auto& input : aggregate.call->inputs()) {
      const auto* field =
          dynamic_cast<const core::FieldAccessTypedExpr*>(input.get());
      if (field == nullptr || !field->isInputColumn()) {
        continue;
      }
      if (!hasColumn(sourceType, field->name(), *field->type())) {
        return false;
      }
    }
  }
  return true;
}

// Returns the partition function of a LocalPartition for input of 'newType'
// in place of 'oldType'. Returns nullptr if the function cannot be shown to
// ignore the columns whose type changes: anything but gather, or hash on
// unchanged keys without constants. HashPartitionFunctionSpec exposes its key
// channels only through its serialized form.
core::PartitionFunctionSpecPtr rebindPartitionFunction(
    const core::PartitionFunctionSpec& spec,
    const RowTypePtr& oldType,
    const RowTypePtr& newType) {
  if (typeid(spec) == typeid(core::GatherPartitionFunctionSpec)) {
    return std::make_shared<core::GatherPartitionFunctionSpec>();
  }
  if (typeid(spec) != typeid(exec::HashPartitionFunctionSpec)) {
    return nullptr;
  }
  const auto serialized = spec.serialize();
  if (!serialized["constants"].empty()) {
    return nullptr;
  }
  std::vector<column_index_t> channels;
  for (const auto& key : serialized["keyChannels"]) {
    const auto channel = key.asInt();
    if (channel < 0 || static_cast<uint64_t>(channel) >= oldType->size()) {
      return nullptr;
    }
    const auto keyChannel = static_cast<column_index_t>(channel);
    if (!(*oldType->childAt(keyChannel) == *newType->childAt(keyChannel))) {
      return nullptr;
    }
    channels.push_back(keyChannel);
  }
  return std::make_shared<exec::HashPartitionFunctionSpec>(
      newType, std::move(channels));
}

// Returns a copy of 'node', a Node with one source, that reads 'source'.
template <typename Node>
core::PlanNodePtr copyWithSource(
    const core::PlanNode& node,
    const core::PlanNodePtr& source) {
  typename Node::Builder builder(static_cast<const Node&>(node));
  return builder.source(source).build();
}

// Returns a copy of 'node', a join of type Node, that reads 'sources'.
template <typename Node>
core::PlanNodePtr copyWithLeftAndRight(
    const core::PlanNode& node,
    const std::vector<core::PlanNodePtr>& sources) {
  typename Node::Builder builder(static_cast<const Node&>(node));
  return builder.left(sources[0]).right(sources[1]).build();
}

// Returns a copy of 'node' that reads 'sources' in place of its own sources,
// one for one. Returns nullptr for a node class this function does not copy.
// 'sources' must have the output types of the sources they replace: nothing
// derived from them, such as an output type or a partition function, is
// recomputed.
core::PlanNodePtr copyWithSources(
    const core::PlanNode& node,
    const std::vector<core::PlanNodePtr>& sources) {
  VELOX_CHECK_EQ(sources.size(), node.sources().size());
  if (isExactly<core::FilterNode>(node)) {
    return copyWithSource<core::FilterNode>(node, sources[0]);
  }
  if (isExactly<core::ProjectNode>(node)) {
    return copyWithSource<core::ProjectNode>(node, sources[0]);
  }
  if (isExactly<core::AggregationNode>(node)) {
    return copyWithSource<core::AggregationNode>(node, sources[0]);
  }
  if (isExactly<core::OrderByNode>(node)) {
    return copyWithSource<core::OrderByNode>(node, sources[0]);
  }
  if (isExactly<core::TopNNode>(node)) {
    return copyWithSource<core::TopNNode>(node, sources[0]);
  }
  if (isExactly<core::LimitNode>(node)) {
    return copyWithSource<core::LimitNode>(node, sources[0]);
  }
  if (isExactly<core::WindowNode>(node)) {
    return copyWithSource<core::WindowNode>(node, sources[0]);
  }
  if (isExactly<core::HashJoinNode>(node)) {
    return copyWithLeftAndRight<core::HashJoinNode>(node, sources);
  }
  if (isExactly<core::NestedLoopJoinNode>(node)) {
    return copyWithLeftAndRight<core::NestedLoopJoinNode>(node, sources);
  }
  if (isExactly<core::LocalPartitionNode>(node)) {
    core::LocalPartitionNode::Builder builder(
        static_cast<const core::LocalPartitionNode&>(node));
    return builder.sources(sources).build();
  }
  return nullptr;
}

} // namespace

// static
TypePtr CudfAggregationNode::compactIntermediateType(
    const Aggregate& aggregate) {
  if (aggregate.call->name() !=
      CudfConfig::getInstance().functionNamePrefix + "sum") {
    return nullptr;
  }
  if (aggregate.rawInputTypes.size() != 1 ||
      !aggregate.rawInputTypes[0]->isShortDecimal()) {
    return nullptr;
  }
  if (aggregate.distinct || aggregate.mask != nullptr ||
      !aggregate.sortingKeys.empty() || !aggregate.sortingOrders.empty()) {
    return nullptr;
  }
  if (inputColumn(*aggregate.call) == nullptr) {
    return nullptr;
  }
  return DECIMAL(
      38, getDecimalPrecisionScale(*aggregate.rawInputTypes[0]).second);
}

CudfAggregationNode::CudfAggregationNode(
    const core::AggregationNode& node,
    std::vector<TypePtr> compactIntermediateTypes)
    : core::AggregationNode(
          node.id(),
          node.step(),
          node.groupingKeys(),
          node.preGroupedKeys(),
          node.aggregateNames(),
          node.aggregates(),
          node.globalGroupingSets(),
          node.groupId(),
          node.ignoreNullKeys(),
          node.noGroupsSpanBatches(),
          node.mayRetainInput(),
          node.sources()[0]),
      compactIntermediateTypes_(std::move(compactIntermediateTypes)) {
  VELOX_USER_CHECK(
      step() != Step::kSingle,
      "A single-step aggregation has no intermediate state to make compact");
  VELOX_USER_CHECK(
      !groupingKeys().empty(),
      "The compact decimal SUM state is not supported for a global aggregation");
  VELOX_USER_CHECK_EQ(compactIntermediateTypes_.size(), aggregates().size());
  bool hasCompactState{false};
  for (size_t i = 0; i < aggregates().size(); ++i) {
    const auto& stateType = intermediateType(i);
    if (stateType == nullptr) {
      continue;
    }
    hasCompactState = true;
    const auto& aggregate = aggregates()[i];
    const auto expected = compactIntermediateType(aggregate);
    VELOX_USER_CHECK_NOT_NULL(
        expected,
        "Aggregate is not eligible for the compact decimal SUM state: {}",
        aggregate.call->toString());
    VELOX_USER_CHECK(
        *stateType == *expected,
        "Compact decimal SUM state must be DECIMAL(38, scale of the input): {}",
        stateType->toString());
    const auto* field = inputColumn(*aggregate.call);
    // A PARTIAL step reads the raw input and every other step the state.
    const auto& inputType =
        step() == Step::kPartial ? aggregate.rawInputTypes[0] : stateType;
    VELOX_USER_CHECK(
        *field->type() == *inputType,
        "Compact decimal SUM input must have the type of its step: {}",
        field->type()->toString());
    VELOX_USER_CHECK(
        *aggregate.call->type() == *stateType,
        "Compact decimal SUM result must have the type of the state: {}",
        aggregate.call->type()->toString());
    VELOX_USER_CHECK(
        hasColumn(sources()[0]->outputType(), field->name(), *inputType),
        "Compact decimal SUM input must match its source column: {}",
        field->name());
  }
  VELOX_USER_CHECK(
      hasCompactState,
      "CudfAggregationNode must have an aggregate with the compact state");
}

bool CudfAggregationNode::canSpill(
    const core::QueryConfig& /*queryConfig*/) const {
  return false;
}

folly::dynamic CudfAggregationNode::serialize() const {
  VELOX_UNSUPPORTED(
      "An aggregation with the compact decimal SUM state cannot be serialized");
}

// static
core::PlanNodePtr CudfAggregationNode::rewriteFinalAggregation(
    const core::PlanNodePtr& finalAggregation) {
  // Without the cuDF driver adapter nothing keeps a compact aggregation off
  // the CPU.
  if (!cudfIsRegistered() || !CudfConfig::getInstance().enabled) {
    return finalAggregation;
  }
  if (finalAggregation == nullptr ||
      !isExactly<AggregationNode>(*finalAggregation)) {
    return finalAggregation;
  }
  const auto& finalNode =
      static_cast<const AggregationNode&>(*finalAggregation);
  if (finalNode.step() != Step::kFinal || finalNode.groupingKeys().empty() ||
      !finalNode.globalGroupingSets().empty() ||
      finalNode.groupId().has_value()) {
    return finalAggregation;
  }

  // The PARTIAL aggregation feeds the FINAL one directly or through one
  // LocalPartition.
  const core::LocalPartitionNode* local{nullptr};
  const core::PlanNode* partialSource = finalNode.sources()[0].get();
  if (isExactly<core::LocalPartitionNode>(*partialSource)) {
    local = static_cast<const core::LocalPartitionNode*>(partialSource);
    if (local->sources().size() != 1 || local->scaleWriter()) {
      return finalAggregation;
    }
    partialSource = local->sources()[0].get();
  }
  if (!isExactly<AggregationNode>(*partialSource)) {
    return finalAggregation;
  }
  const auto& partial = static_cast<const AggregationNode&>(*partialSource);
  if (partial.step() != Step::kPartial ||
      !partial.globalGroupingSets().empty() || partial.groupId().has_value() ||
      partial.aggregates().size() != finalNode.aggregates().size() ||
      partial.groupingKeys().size() != finalNode.groupingKeys().size()) {
    return finalAggregation;
  }

  const auto& rawType = partial.sources()[0]->outputType();
  auto partialAggregates = partial.aggregates();
  auto finalAggregates = finalNode.aggregates();
  std::vector<TypePtr> stateTypes(partialAggregates.size());
  bool hasCompactState{false};
  for (size_t i = 0; i < partialAggregates.size(); ++i) {
    const auto& partialAggregate = partialAggregates[i];
    const auto& finalAggregate = finalAggregates[i];
    const auto stateType = compactIntermediateType(partialAggregate);
    if (stateType == nullptr) {
      continue;
    }
    // The FINAL aggregate must be the plain merge of this PARTIAL aggregate:
    // the same function over the same raw input type, reading the PARTIAL
    // output column of the same position, with the VARBINARY state on both
    // sides.
    const auto* rawField = inputColumn(*partialAggregate.call);
    const auto* stateField = inputColumn(*finalAggregate.call);
    if (stateField == nullptr ||
        stateField->name() != partial.aggregateNames()[i] ||
        finalAggregate.call->name() != partialAggregate.call->name() ||
        finalAggregate.distinct || finalAggregate.mask != nullptr ||
        !finalAggregate.sortingKeys.empty() ||
        !finalAggregate.sortingOrders.empty() ||
        finalAggregate.rawInputTypes.size() != 1 ||
        !(*finalAggregate.rawInputTypes[0] ==
          *partialAggregate.rawInputTypes[0])) {
      continue;
    }
    if (!(*rawField->type() == *partialAggregate.rawInputTypes[0]) ||
        !hasColumn(rawType, rawField->name(), *rawField->type()) ||
        !(*partialAggregate.call->type() == *VARBINARY()) ||
        !(*stateField->type() == *VARBINARY()) ||
        !(*finalAggregate.call->type() == *stateType)) {
      continue;
    }
    hasCompactState = true;
    stateTypes[i] = stateType;
    const auto partialName = partialAggregate.call->name();
    const auto finalName = finalAggregate.call->name();
    const auto stateName = stateField->name();
    partialAggregates[i].call = std::make_shared<core::CallTypedExpr>(
        stateType, partialAggregate.call->inputs(), partialName);
    finalAggregates[i].call = std::make_shared<core::CallTypedExpr>(
        stateType,
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(stateType, stateName)},
        finalName);
  }
  if (!hasCompactState) {
    return finalAggregation;
  }

  const std::shared_ptr<const AggregationNode> rewrittenPartial =
      AggregationNode::Builder(partial)
          .aggregates(std::move(partialAggregates))
          .build();
  // A column whose type changed must be read by the FINAL aggregate that was
  // rewritten with it and by nothing else.
  if (!readsColumnsOf(finalAggregates, rewrittenPartial->outputType())) {
    return finalAggregation;
  }
  core::PartitionFunctionSpecPtr partitionFunction;
  if (local != nullptr) {
    partitionFunction = rebindPartitionFunction(
        local->partitionFunctionSpec(),
        partial.outputType(),
        rewrittenPartial->outputType());
    if (partitionFunction == nullptr) {
      return finalAggregation;
    }
  }
  core::PlanNodePtr compactSource =
      std::make_shared<CudfAggregationNode>(*rewrittenPartial, stateTypes);
  if (local != nullptr) {
    compactSource = core::LocalPartitionNode::Builder(*local)
                        .partitionFunctionSpec(std::move(partitionFunction))
                        .sources(std::vector<core::PlanNodePtr>{compactSource})
                        .build();
  }
  const std::shared_ptr<const AggregationNode> rewrittenFinal =
      AggregationNode::Builder(finalNode)
          .aggregates(std::move(finalAggregates))
          .source(compactSource)
          .build();
  core::PlanNodePtr compactFinal = std::make_shared<CudfAggregationNode>(
      *rewrittenFinal, std::move(stateTypes));
  VELOX_CHECK(
      *compactFinal->outputType() == *finalAggregation->outputType(),
      "The compact decimal SUM rewrite must not change the FINAL output type: {}",
      compactFinal->outputType()->toString());
  return compactFinal;
}

// static
core::PlanNodePtr CudfAggregationNode::rewritePlan(
    const core::PlanNodePtr& plan) {
  VELOX_CHECK_NOT_NULL(plan);
  std::vector<core::PlanNodePtr> sources;
  sources.reserve(plan->sources().size());
  bool sourcesChanged{false};
  for (const auto& source : plan->sources()) {
    sources.push_back(rewritePlan(source));
    sourcesChanged |= sources.back() != source;
  }
  if (!sourcesChanged) {
    return rewriteFinalAggregation(plan);
  }
  const auto copy = copyWithSources(*plan, sources);
  if (copy == nullptr) {
    // The chains below a node of a class that is not copied keep their
    // ordinary state.
    return plan;
  }
  VELOX_CHECK(
      *copy->outputType() == *plan->outputType(),
      "Reading a compact aggregation must not change a node's output type: {}",
      plan->id());
  return rewriteFinalAggregation(copy);
}

} // namespace facebook::velox::cudf_velox
