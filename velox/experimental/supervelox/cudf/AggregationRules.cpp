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

#include <set>

#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/experimental/cudf/exec/DenseGroupby.h"
#include "velox/experimental/cudf/exec/FusedAggregation.h"
#include "velox/expression/Expr.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

using core::AggregationNode;

std::string_view baseName(const std::string& name) {
  const auto dot = name.rfind('.');
  return dot == std::string::npos ? std::string_view(name) : std::string_view(name).substr(dot + 1);
}

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
  if (dynamic_cast<const cudf_velox::FusedAggregationNode*>(raw)) {
    return nullptr;
  }
  if (const auto* direct = dynamic_cast<const cudf_velox::DenseAggregationNode*>(raw)) {
    return std::make_shared<cudf_velox::DenseAggregationNode>(
        direct->id(), direct->groupingKeys(), direct->aggregateNames(), direct->aggregates(), sources[0], direct->keys());
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

std::optional<OneStep> oneStep(const core::PlanNodePtr& node, bool global = false) {
  const auto* aggregation = dynamic_cast<const AggregationNode*>(node.get());
  if (aggregation == nullptr || (!global && aggregation->groupingKeys().empty()) || aggregation->isPreGrouped() ||
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

bool references(const core::TypedExprPtr& expr, const std::set<std::string>& columns) {
  if (const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (field->isInputColumn() && columns.count(field->name())) {
      return true;
    }
  }
  for (const auto& input : expr->inputs()) {
    if (references(input, columns)) {
      return true;
    }
  }
  return false;
}

/// 'node' with dictionary 'columns' of its output read as their integer codes
/// from the scan up, or null unless only filters not reading them and
/// projections passing them through lie between.
core::PlanNodePtr withCodes(const core::PlanNodePtr& node, const std::set<std::string>& columns) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node.get())) {
    std::set<std::string> stored;
    for (const auto& column : columns) {
      stored.insert(scan->assignments().at(column)->name());
    }
    if (const auto* hive = dynamic_cast<const connector::hive::HiveTableHandle*>(scan->tableHandle().get())) {
      for (const auto& [subfield, filter] : hive->subfieldFilters()) {
        if (stored.count(subfield.baseName())) {
          return nullptr;
        }
      }
      if (hive->remainingFilter() != nullptr && references(hive->remainingFilter(), stored)) {
        return nullptr;
      }
    }
    auto types = scan->outputType()->children();
    for (const auto& column : columns) {
      types[scan->outputType()->getChildIdx(column)] = INTEGER();
    }
    return std::make_shared<core::TableScanNode>(
        scan->id(), ROW(std::vector<std::string>(scan->outputType()->names()), std::move(types)),
        scan->tableHandle(), scan->assignments());
  }
  if (const auto* filter = dynamic_cast<const core::FilterNode*>(node.get())) {
    if (references(filter->filter(), columns)) {
      return nullptr;
    }
    auto source = withCodes(node->sources()[0], columns);
    return source ? std::make_shared<core::FilterNode>(filter->id(), filter->filter(), source) : nullptr;
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    std::set<std::string> below;
    auto projections = project->projections();
    for (size_t i = 0; i < projections.size(); ++i) {
      if (columns.count(project->names()[i])) {
        below.insert(std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(projections[i])->name());
      }
    }
    for (size_t i = 0; i < projections.size(); ++i) {
      if (columns.count(project->names()[i])) {
        const auto& name = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(projections[i])->name();
        projections[i] = std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), name);
      } else if (references(projections[i], below)) {
        return nullptr;
      }
    }
    auto source = withCodes(node->sources()[0], below);
    return source ? std::make_shared<core::ProjectNode>(project->id(), project->names(), projections, source)
                  : nullptr;
  }
  return nullptr;
}

/// The ranges the store bounds the grouping keys of 'step' to, the keys read
/// as dictionary codes, and the slots they index, or nullopt if a key is
/// unbounded or the slots exceed 'maxSlots' or the rows of the keys' tables.
struct KeyRanges {
  std::vector<cudf_velox::DenseKey> ranges;
  std::set<std::string> coded;
  std::vector<std::string> described;
  int64_t slots{1};
};

std::optional<KeyRanges> keyRanges(const OneStep& step, CacheConnector& connector, int64_t maxSlots) {
  KeyRanges out;
  auto& [ranges, coded, described, slots] = out;
  int64_t rows = 0;
  for (const auto& key : step.raw->groupingKeys()) {
    const auto source = traceValues(step.raw->sources()[0], key->name());
    if (!source.has_value()) {
      return std::nullopt;
    }
    const Column& column = connector.store().metadata(source->table, source->column);
    const auto& type = key->type();
    if ((type->isBigint() || type->isInteger() || type->isDate()) &&
        (column.kind == Kind::kInt || column.kind == Kind::kDate)) {
      ranges.push_back({column.min, column.max, {}});
    } else if (type->isVarchar() && column.kind == Kind::kDictionary && !column.dictionary.empty()) {
      ranges.push_back({0, static_cast<int64_t>(column.dictionary.size()) - 1, column.dictionary});
      coded.insert(key->name());
    } else {
      return std::nullopt;
    }
    slots *= ranges.back().high - ranges.back().low + 1;
    if (slots > maxSlots) {
      return std::nullopt;
    }
    rows = std::max(rows, connector.store().rows(source->table));
    described.push_back(fmt::format(
        "{} ({}.{}{})",
        key->name(),
        source->table,
        source->column,
        coded.count(key->name()) ? fmt::format(", {} dictionary codes", column.dictionary.size())
                                 : fmt::format(" in {}..{}", column.min, column.max)));
  }
  if (!step.raw->groupingKeys().empty() && slots > rows) {
    return std::nullopt;
  }
  return out;
}

/// 'step' as a dense aggregation, or null.
core::PlanNodePtr dense(const OneStep& step, CacheConnector& connector, AggregationRewrite& out) {
  constexpr int64_t kMaxSlots = int64_t{1} << 28;
  if (step.aggregates.empty()) {
    return nullptr;
  }
  for (const auto& aggregate : step.aggregates) {
    if (!cudf_velox::isDenseAggregate(aggregate)) {
      return nullptr;
    }
  }
  auto bounded = keyRanges(step, connector, kMaxSlots);
  if (!bounded.has_value()) {
    return nullptr;
  }
  auto& [ranges, coded, described, slots] = *bounded;
  auto source = step.raw->sources()[0];
  auto keys = step.raw->groupingKeys();
  if (!coded.empty()) {
    source = withCodes(source, coded);
    if (source == nullptr) {
      return nullptr;
    }
    for (auto& key : keys) {
      if (coded.count(key->name())) {
        key = std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), key->name());
      }
    }
  }
  out.log.push_back(fmt::format(
      "aggregation {} grouped by {} adds into {} slots indexed by key",
      step.aggregation->id(),
      fmt::join(described, ", "),
      slots));
  return std::make_shared<cudf_velox::DenseAggregationNode>(
      step.aggregation->id(), keys, step.raw->aggregateNames(), step.aggregates, source, std::move(ranges));
}

/// 'expr' with its input columns replaced by their expressions in 'mapping',
/// or null for a kind of expression a fused aggregation does not evaluate.
core::TypedExprPtr substitute(
    const core::TypedExprPtr& expr,
    const std::unordered_map<std::string, core::TypedExprPtr>& mapping) {
  if (const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (!field->isInputColumn()) {
      return nullptr;
    }
    const auto it = mapping.find(field->name());
    return it != mapping.end() ? it->second : expr;
  }
  if (dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
    return expr;
  }
  std::vector<core::TypedExprPtr> inputs;
  for (const auto& input : expr->inputs()) {
    inputs.push_back(substitute(input, mapping));
    if (inputs.back() == nullptr) {
      return nullptr;
    }
  }
  if (const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    return std::make_shared<core::CallTypedExpr>(call->type(), std::move(inputs), call->name());
  }
  if (const auto* cast = dynamic_cast<const core::CastTypedExpr*>(expr.get())) {
    return std::make_shared<core::CastTypedExpr>(cast->type(), inputs[0], cast->isTryCast());
  }
  return nullptr;
}

/// The node below the filters and projections from 'node' down, with
/// 'exprs' over the output of 'node' rewritten over its output, and the
/// filters' conditions, rewritten likewise, added to 'filters'. Null if an
/// expression cannot be rewritten.
core::PlanNodePtr belowChain(
    core::PlanNodePtr node,
    std::vector<core::TypedExprPtr>& exprs,
    std::vector<core::TypedExprPtr>& filters) {
  for (;;) {
    if (const auto* filter = dynamic_cast<const core::FilterNode*>(node.get())) {
      filters.push_back(filter->filter());
    } else if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
      std::unordered_map<std::string, core::TypedExprPtr> mapping;
      for (size_t i = 0; i < project->names().size(); ++i) {
        mapping[project->names()[i]] = project->projections()[i];
      }
      for (auto& expr : exprs) {
        if (expr != nullptr && (expr = substitute(expr, mapping)) == nullptr) {
          return nullptr;
        }
      }
      for (auto& filter : filters) {
        if ((filter = substitute(filter, mapping)) == nullptr) {
          return nullptr;
        }
      }
    } else {
      return node;
    }
    node = node->sources()[0];
  }
}

void inputColumns(const core::TypedExprPtr& expr, std::set<std::string>& out) {
  if (const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (field->isInputColumn()) {
      out.insert(field->name());
    }
  }
  for (const auto& input : expr->inputs()) {
    inputColumns(input, out);
  }
}

struct LiftedScan {
  core::PlanNodePtr scan;
  std::vector<core::TypedExprPtr> filters;
  std::vector<cudf_velox::ColumnFilter> columnFilters;
};

/// 'scan' with its filters only skipping zones, the columns they read added
/// to its output, and the filters over that output, or nullopt if a filter
/// cannot be compiled. 'scan' itself if it has no filters.
std::optional<LiftedScan> liftFilters(const core::TableScanNode& scan, std::string& why) {
  const auto* hive = dynamic_cast<const connector::hive::HiveTableHandle*>(scan.tableHandle().get());
  LiftedScan out;
  if (hive == nullptr || (hive->subfieldFilters().empty() && hive->remainingFilter() == nullptr)) {
    out.scan = std::make_shared<core::TableScanNode>(scan.id(), scan.outputType(), scan.tableHandle(), scan.assignments());
    return out;
  }
  auto names = scan.outputType()->names();
  auto types = scan.outputType()->children();
  auto assignments = scan.assignments();
  std::map<std::string, std::string> outputOf;
  for (const auto& [name, handle] : assignments) {
    outputOf.emplace(handle->name(), name);
  }
  auto output = [&](const std::string& column) -> std::optional<std::string> {
    if (const auto it = outputOf.find(column); it != outputOf.end()) {
      return it->second;
    }
    const auto type = hive->dataColumns() ? hive->dataColumns()->findChild(column) : nullptr;
    if (type == nullptr) {
      return std::nullopt;
    }
    std::string name = column;
    while (std::find(names.begin(), names.end(), name) != names.end()) {
      name += "_";
    }
    names.push_back(name);
    types.push_back(type);
    assignments.emplace(
        name,
        std::make_shared<connector::hive::HiveColumnHandle>(
            column, connector::hive::HiveColumnHandle::ColumnType::kRegular, type, type));
    outputOf.emplace(column, name);
    return name;
  };
  common::SubfieldFilters kept;
  for (const auto& [subfield, filter] : hive->subfieldFilters()) {
    const auto name = subfield.path().size() == 1 ? output(subfield.baseName()) : std::nullopt;
    if (!name.has_value()) {
      why = fmt::format("filter on {}", subfield.toString());
      return std::nullopt;
    }
    const auto& type = types[std::find(names.begin(), names.end(), *name) - names.begin()];
    if (auto reason = cudf_velox::unfusable(*filter, type)) {
      why = *reason;
      return std::nullopt;
    }
    out.columnFilters.push_back({*name, std::shared_ptr<const common::Filter>(filter->clone())});
    kept.emplace(subfield.clone(), filter->clone());
  }
  if (const auto& remaining = hive->remainingFilter()) {
    std::set<std::string> columns;
    inputColumns(remaining, columns);
    std::unordered_map<std::string, core::TypedExprPtr> mapping;
    for (const auto& column : columns) {
      const auto name = output(column);
      if (!name.has_value()) {
        why = fmt::format("remaining filter on {}", column);
        return std::nullopt;
      }
      const auto& type = types[std::find(names.begin(), names.end(), *name) - names.begin()];
      mapping[column] = std::make_shared<core::FieldAccessTypedExpr>(type, *name);
    }
    auto filter = substitute(remaining, mapping);
    if (filter == nullptr) {
      why = "an expression in the remaining filter";
      return std::nullopt;
    }
    out.filters.push_back(std::move(filter));
  }
  auto parameters = hive->tableParameters();
  parameters[CacheConnector::kZoneOnly] = "true";
  auto handle = std::make_shared<connector::hive::HiveTableHandle>(
      hive->connectorId(),
      hive->tableName(),
      std::move(kept),
      nullptr,
      hive->dataColumns(),
      hive->indexColumns(),
      parameters,
      hive->hiveFilterColumnHandles(),
      hive->sampleRate(),
      hive->dbName(),
      hive->dataColumnFieldIds());
  out.scan = std::make_shared<core::TableScanNode>(
      scan.id(), ROW(std::move(names), std::move(types)), handle, assignments);
  return out;
}

/// Per entry of the dictionary of 'column', whether 'expr', boolean over
/// VARCHAR input column 'name' alone, is true. Nullopt if it cannot be
/// evaluated.
std::optional<std::vector<bool>> testDictionary(
    const core::TypedExprPtr& expr,
    const std::string& name,
    const Column& column) {
  static auto* pool = new std::shared_ptr<memory::MemoryPool>(memory::memoryManager()->addLeafPool("dictionary_tests"));
  const auto size = static_cast<vector_size_t>(column.dictionary.size());
  try {
    auto values = BaseVector::create<FlatVector<StringView>>(VARCHAR(), size, pool->get());
    for (vector_size_t i = 0; i < size; ++i) {
      values->set(i, StringView(column.dictionary[i]));
    }
    auto input = std::make_shared<RowVector>(
        pool->get(), ROW({name}, {VARCHAR()}), nullptr, size, std::vector<VectorPtr>{values});
    auto queryCtx = core::QueryCtx::create();
    core::ExecCtx execCtx(pool->get(), queryCtx.get());
    exec::ExprSet exprs({expr}, &execCtx);
    exec::EvalCtx evalCtx(&execCtx, &exprs, input.get());
    SelectivityVector rows(size);
    std::vector<VectorPtr> result(1);
    exprs.eval(rows, evalCtx, result);
    DecodedVector decoded(*result[0], rows);
    std::vector<bool> out(size);
    for (vector_size_t i = 0; i < size; ++i) {
      out[i] = !decoded.isNullAt(i) && decoded.valueAt<bool>(i);
    }
    return out;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

/// Whether the code in input column 'name' is one of those 'matches' is true
/// for, as equalities with the fewer codes.
core::TypedExprPtr codeTest(const std::string& name, const std::vector<bool>& matches) {
  std::vector<int32_t> in;
  std::vector<int32_t> out;
  for (size_t code = 0; code < matches.size(); ++code) {
    (matches[code] ? in : out).push_back(static_cast<int32_t>(code));
  }
  if (in.empty() || out.empty()) {
    return std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), variant(out.empty()));
  }
  const bool negate = in.size() > out.size();
  const auto field = std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), name);
  std::vector<core::TypedExprPtr> terms;
  for (const auto code : negate ? out : in) {
    terms.push_back(std::make_shared<core::CallTypedExpr>(
        BOOLEAN(),
        std::vector<core::TypedExprPtr>{field, std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(code))},
        "eq"));
  }
  core::TypedExprPtr any = terms.size() == 1 ? terms[0] : std::make_shared<core::CallTypedExpr>(BOOLEAN(), terms, "or");
  return negate ? std::make_shared<core::CallTypedExpr>(BOOLEAN(), std::vector<core::TypedExprPtr>{any}, "not") : any;
}

using Dictionaries = std::unordered_map<std::string, const Column*>;

/// 'expr' with each boolean subexpression that reads one column of
/// 'dictionaries' and no other column replaced by a test of the column's
/// codes, or null if it reads such a column otherwise.
core::TypedExprPtr withCodeTests(const core::TypedExprPtr& expr, const Dictionaries& dictionaries) {
  std::set<std::string> columns;
  inputColumns(expr, columns);
  if (std::none_of(columns.begin(), columns.end(), [&](const auto& c) { return dictionaries.count(c) > 0; })) {
    return expr;
  }
  if (expr->type()->isBoolean() && columns.size() == 1 &&
      dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get()) == nullptr) {
    const auto& name = *columns.begin();
    const auto matches = testDictionary(expr, name, *dictionaries.at(name));
    return matches.has_value() ? codeTest(name, *matches) : nullptr;
  }
  std::vector<core::TypedExprPtr> inputs;
  for (const auto& input : expr->inputs()) {
    inputs.push_back(withCodeTests(input, dictionaries));
    if (inputs.back() == nullptr) {
      return nullptr;
    }
  }
  if (const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    return std::make_shared<core::CallTypedExpr>(call->type(), std::move(inputs), call->name());
  }
  if (const auto* cast = dynamic_cast<const core::CastTypedExpr*>(expr.get())) {
    return std::make_shared<core::CastTypedExpr>(cast->type(), inputs[0], cast->isTryCast());
  }
  return nullptr;
}

/// 'filter' on the strings of dictionary 'column' as a filter on its codes.
std::shared_ptr<const common::Filter> codeFilter(const common::Filter& filter, const Column& column) {
  std::vector<int64_t> codes;
  for (size_t code = 0; code < column.dictionary.size(); ++code) {
    const auto& value = column.dictionary[code];
    if (filter.testBytes(value.data(), static_cast<int32_t>(value.size()))) {
      codes.push_back(static_cast<int64_t>(code));
    }
  }
  if (codes.empty()) {
    return std::make_shared<common::AlwaysFalse>();
  }
  if (codes.size() == 1) {
    return std::make_shared<common::BigintRange>(codes[0], codes[0], false);
  }
  return common::createBigintValues(codes, false);
}

/// Adds to 'encodings' how input 'name', stored as 'column', decodes to
/// 'type'. Why it cannot, or nullopt. Strings and dictionary codes are read
/// as stored.
std::optional<std::string>
encode(const Column& column, const TypePtr& type, const std::string& name, cudf_velox::Encodings& encodings) {
  auto mismatch = [&] { return fmt::format("{}.{} read as {}", column.table, column.name, type->toString()); };
  switch (column.kind) {
    case Kind::kString:
      return type->isVarchar() ? std::nullopt : std::optional(mismatch());
    case Kind::kDictionary:
      return type->isVarchar() || type->isInteger() ? std::nullopt : std::optional(mismatch());
    case Kind::kDate:
      if (!type->isDate()) {
        return mismatch();
      }
      break;
    case Kind::kDecimal:
      if (!type->isDouble() && !(type->isDecimal() && getDecimalPrecisionScale(*type).second == column.scale)) {
        return mismatch();
      }
      break;
    case Kind::kInt:
      if (type->isDecimal() || !(type->isBigint() || type->isInteger() || type->isSmallint() || type->isTinyint())) {
        return mismatch();
      }
      break;
  }
  encodings[name] = {type, column.scale};
  return std::nullopt;
}

common::SubfieldFilters clone(const common::SubfieldFilters& filters) {
  common::SubfieldFilters out;
  for (const auto& [subfield, filter] : filters) {
    out.emplace(subfield.clone(), filter->clone());
  }
  return out;
}

void fieldTypes(const core::TypedExprPtr& expr, std::map<std::string, TypePtr>& out) {
  if (const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (field->isInputColumn()) {
      out.emplace(field->name(), field->type());
    }
  }
  for (const auto& input : expr->inputs()) {
    fieldTypes(input, out);
  }
}

/// A scan from the cache with the filters and projections above it that it
/// absorbs, as filters and expressions over its stored columns.
class ScanFusion {
 public:
  ScanFusion(const core::TableScanNode& scan, const connector::hive::HiveTableHandle& hive, CacheConnector& connector)
      : scan_(scan), hive_(hive), connector_(connector), names_(scan.outputType()->names()) {
    for (size_t i = 0; i < names_.size(); ++i) {
      exprs_.push_back(std::make_shared<core::FieldAccessTypedExpr>(
          scan.outputType()->childAt(i), scan.assignments().at(names_[i])->name()));
    }
    for (const auto& [subfield, filter] : hive.subfieldFilters()) {
      if (subfield.path().size() != 1) {
        valid_ = false;
        return;
      }
      columnFilters_.push_back({subfield.baseName(), std::shared_ptr<const common::Filter>(filter->clone())});
    }
    if (hive.remainingFilter() != nullptr) {
      filters_.push_back(hive.remainingFilter());
    }
  }

  bool valid() const {
    return valid_;
  }

  /// Absorbs 'node', a filter or projection over the rows so far. False,
  /// changing nothing, if an expression cannot be rewritten.
  bool absorb(const core::PlanNode& node) {
    std::unordered_map<std::string, core::TypedExprPtr> mapping;
    for (size_t i = 0; i < names_.size(); ++i) {
      mapping[names_[i]] = exprs_[i];
    }
    if (const auto* filter = dynamic_cast<const core::FilterNode*>(&node)) {
      auto condition = substitute(filter->filter(), mapping);
      if (condition == nullptr) {
        return false;
      }
      filters_.push_back(std::move(condition));
      return true;
    }
    const auto* project = dynamic_cast<const core::ProjectNode*>(&node);
    if (project == nullptr) {
      return false;
    }
    std::vector<core::TypedExprPtr> exprs;
    for (const auto& projection : project->projections()) {
      exprs.push_back(substitute(projection, mapping));
      if (exprs.back() == nullptr) {
        return false;
      }
    }
    names_ = project->names();
    exprs_ = std::move(exprs);
    return true;
  }

  /// The scan returning the rows so far, or null with 'why' set.
  core::PlanNodePtr build(std::string& why) const {
    const auto& table = scan_.tableHandle()->name();
    std::map<std::string, TypePtr> declared;
    for (const auto& expr : exprs_) {
      fieldTypes(expr, declared);
    }
    for (const auto& filter : filters_) {
      fieldTypes(filter, declared);
    }
    for (const auto& filter : columnFilters_) {
      if (!declared.count(filter.column)) {
        const auto type = hive_.dataColumns() ? hive_.dataColumns()->findChild(filter.column) : nullptr;
        if (type == nullptr) {
          why = fmt::format("no type for filter column {}", filter.column);
          return nullptr;
        }
        declared.emplace(filter.column, type);
      }
    }
    cudf_velox::Encodings encodings;
    Dictionaries dictionaries;
    std::vector<std::string> inputNames;
    std::vector<TypePtr> inputTypes;
    for (const auto& [name, type] : declared) {
      const Column& column = connector_.store().metadata(table, name);
      if (auto reason = encode(column, type, name, encodings)) {
        why = *reason;
        return nullptr;
      }
      if (column.kind == Kind::kDictionary && type->isVarchar()) {
        dictionaries.emplace(name, &column);
      }
      inputNames.push_back(name);
      inputTypes.push_back(CacheConnector::storedVeloxType(column));
    }
    std::vector<FusedScanHandle::Output> outputs;
    std::vector<TypePtr> outputTypes;
    std::vector<core::TypedExprPtr> projections;
    for (size_t i = 0; i < exprs_.size(); ++i) {
      const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(exprs_[i].get());
      if (field != nullptr && field->type()->isVarchar()) {
        outputs.push_back({FusedScanHandle::Source::kString, field->name(), nullptr});
        outputTypes.push_back(VARCHAR());
        continue;
      }
      auto expr = withCodeTests(exprs_[i], dictionaries);
      if (expr == nullptr) {
        why = fmt::format("{} reads a dictionary string", names_[i]);
        return nullptr;
      }
      outputs.push_back({FusedScanHandle::Source::kExpression, "", expr});
      outputTypes.push_back(expr->type());
      projections.push_back(std::move(expr));
    }
    if (outputs.empty()) {
      why = "no output columns";
      return nullptr;
    }
    std::vector<core::TypedExprPtr> filters;
    for (const auto& filter : filters_) {
      filters.push_back(withCodeTests(filter, dictionaries));
      if (filters.back() == nullptr) {
        why = fmt::format("filter {} reads a dictionary string", filter->toString());
        return nullptr;
      }
    }
    std::vector<cudf_velox::ColumnFilter> columnFilters;
    for (const auto& filter : columnFilters_) {
      const auto dictionary = dictionaries.find(filter.column);
      columnFilters.push_back(
          dictionary == dictionaries.end() ? filter
                                           : cudf_velox::ColumnFilter{filter.column, codeFilter(*filter.filter, *dictionary->second)});
    }
    const auto input = ROW(std::move(inputNames), std::move(inputTypes));
    if (auto reason = cudf_velox::FusedTransform::unfusable(input, encodings, filters, columnFilters, projections)) {
      why = *reason;
      return nullptr;
    }
    connector::ColumnHandleMap assignments;
    for (size_t i = 0; i < names_.size(); ++i) {
      assignments.emplace(
          names_[i],
          std::make_shared<connector::hive::HiveColumnHandle>(
              names_[i], connector::hive::HiveColumnHandle::ColumnType::kRegular, outputTypes[i], outputTypes[i]));
    }
    auto handle = std::make_shared<FusedScanHandle>(
        hive_,
        clone(hive_.subfieldFilters()),
        input,
        std::move(encodings),
        std::move(filters),
        std::move(columnFilters),
        std::move(outputs));
    return std::make_shared<core::TableScanNode>(
        scan_.id(), ROW(std::vector<std::string>(names_), std::move(outputTypes)), handle, assignments);
  }

 private:
  const core::TableScanNode& scan_;
  const connector::hive::HiveTableHandle& hive_;
  CacheConnector& connector_;
  std::vector<std::string> names_;
  /// Per output, its value over the stored columns.
  std::vector<core::TypedExprPtr> exprs_;
  std::vector<core::TypedExprPtr> filters_;
  std::vector<cudf_velox::ColumnFilter> columnFilters_;
  bool valid_{true};
};

const connector::hive::HiveTableHandle* cacheScan(const core::PlanNode* node, CacheConnector& connector) {
  const auto* scan = dynamic_cast<const core::TableScanNode*>(node);
  if (scan == nullptr || scan->tableHandle()->connectorId() != connector.connectorId() ||
      dynamic_cast<const FusedScanHandle*>(scan->tableHandle().get())) {
    return nullptr;
  }
  return dynamic_cast<const connector::hive::HiveTableHandle*>(scan->tableHandle().get());
}

/// 'node' with each scan from the cache, and with 'project' the filters and
/// projections above it, made one fused scan.
core::PlanNodePtr fuseScans(
    const core::PlanNodePtr& node,
    CacheConnector& connector,
    bool project,
    AggregationRewrite& out) {
  if (dynamic_cast<const cudf_velox::FusedAggregationNode*>(node.get())) {
    return node;
  }
  std::vector<core::PlanNodePtr> chain;
  auto below = node;
  while (dynamic_cast<const core::FilterNode*>(below.get()) ||
         (project && dynamic_cast<const core::ProjectNode*>(below.get()))) {
    chain.push_back(below);
    below = below->sources()[0];
  }
  const auto* hive = cacheScan(below.get(), connector);
  if (hive != nullptr && !hive->tableParameters().count(CacheConnector::kZoneOnly)) {
    const auto& scan = static_cast<const core::TableScanNode&>(*below);
    ScanFusion fusion(scan, *hive, connector);
    std::string why = "a filter on a nested field";
    auto best = fusion.valid() ? fusion.build(why) : nullptr;
    if (best == nullptr) {
      out.log.push_back(fmt::format("scan {} of {} is not fused: {}", scan.id(), hive->tableName(), why));
      return node;
    }
    size_t absorbed = 0;
    for (size_t k = 0; k < chain.size() && fusion.absorb(*chain[chain.size() - 1 - k]); ++k) {
      std::string reason;
      if (auto built = fusion.build(reason)) {
        best = std::move(built);
        absorbed = k + 1;
      }
    }
    auto result = best;
    for (size_t j = chain.size() - absorbed; j-- > 0;) {
      result = withSources(chain[j], {result});
    }
    out.log.push_back(fmt::format(
        "scan {} of {} filters its encoded columns in one kernel and compacts and decodes the rows that pass in "
        "another{}",
        scan.id(),
        hive->tableName(),
        absorbed > 0 ? fmt::format(", absorbing the {} filter and project nodes above it", absorbed) : ""));
    return result;
  }
  std::vector<core::PlanNodePtr> sources;
  bool changed = false;
  for (const auto& source : node->sources()) {
    sources.push_back(fuseScans(source, connector, project, out));
    changed |= sources.back() != source;
  }
  if (!changed) {
    return node;
  }
  if (auto rebuilt = withSources(node, sources)) {
    return rebuilt;
  }
  out.log.push_back(fmt::format("cannot rebuild {} over fused scans", node->toString()));
  return node;
}

/// 'source', a scan from the cache whose filters only skip zones, returning
/// its columns as stored, with 'filters', 'inputs' and 'columnFilters' over
/// its output rewritten to read dictionary strings as codes and 'encodings'
/// set. 'source' itself if it is not such a scan; null with 'why' set if
/// they cannot be rewritten.
core::PlanNodePtr encodedScan(
    const core::PlanNodePtr& source,
    CacheConnector& connector,
    std::vector<core::TypedExprPtr>& filters,
    std::vector<core::TypedExprPtr>& inputs,
    std::vector<cudf_velox::ColumnFilter>& columnFilters,
    cudf_velox::Encodings& encodings,
    std::string& why) {
  const auto* hive = cacheScan(source.get(), connector);
  if (hive == nullptr ||
      (!hive->tableParameters().count(CacheConnector::kZoneOnly) &&
       (!hive->subfieldFilters().empty() || hive->remainingFilter() != nullptr))) {
    return source;
  }
  const auto& scan = static_cast<const core::TableScanNode&>(*source);
  const auto& type = scan.outputType();
  cudf_velox::Encodings encoded;
  Dictionaries dictionaries;
  std::vector<FusedScanHandle::Output> outputs;
  std::vector<TypePtr> outputTypes;
  std::vector<std::string> inputNames;
  std::vector<TypePtr> inputTypes;
  connector::ColumnHandleMap assignments;
  for (size_t i = 0; i < type->size(); ++i) {
    const auto& name = type->nameOf(i);
    const auto& stored = scan.assignments().at(name)->name();
    const Column& column = connector.store().metadata(hive->tableName(), stored);
    if (auto reason = encode(column, type->childAt(i), name, encoded)) {
      why = *reason;
      return nullptr;
    }
    const bool string = column.kind == Kind::kString;
    if (column.kind == Kind::kDictionary && type->childAt(i)->isVarchar()) {
      dictionaries.emplace(name, &column);
    }
    outputs.push_back({string ? FusedScanHandle::Source::kString : FusedScanHandle::Source::kStored, stored, nullptr});
    outputTypes.push_back(CacheConnector::storedVeloxType(column));
    if (std::find(inputNames.begin(), inputNames.end(), stored) == inputNames.end()) {
      inputNames.push_back(stored);
      inputTypes.push_back(outputTypes.back());
    }
    assignments.emplace(
        name,
        std::make_shared<connector::hive::HiveColumnHandle>(
            name, connector::hive::HiveColumnHandle::ColumnType::kRegular, outputTypes.back(), outputTypes.back()));
  }
  auto rewritten = [&](std::vector<core::TypedExprPtr> exprs) -> std::optional<std::vector<core::TypedExprPtr>> {
    for (auto& expr : exprs) {
      if (expr != nullptr && (expr = withCodeTests(expr, dictionaries)) == nullptr) {
        return std::nullopt;
      }
    }
    return exprs;
  };
  auto newFilters = rewritten(filters);
  auto newInputs = rewritten(inputs);
  if (!newFilters.has_value() || !newInputs.has_value()) {
    why = "an expression reads a dictionary string other than in a condition";
    return nullptr;
  }
  for (auto& filter : columnFilters) {
    if (const auto dictionary = dictionaries.find(filter.column); dictionary != dictionaries.end()) {
      filter.filter = codeFilter(*filter.filter, *dictionary->second);
    }
  }
  filters = std::move(*newFilters);
  inputs = std::move(*newInputs);
  encodings = std::move(encoded);
  auto handle = std::make_shared<FusedScanHandle>(
      *hive,
      clone(hive->subfieldFilters()),
      ROW(std::move(inputNames), std::move(inputTypes)),
      cudf_velox::Encodings{},
      std::vector<core::TypedExprPtr>{},
      std::vector<cudf_velox::ColumnFilter>{},
      std::move(outputs));
  return std::make_shared<core::TableScanNode>(
      scan.id(), ROW(std::vector<std::string>(type->names()), std::move(outputTypes)), handle, assignments);
}

/// 'step' as a fused aggregation over the source below its filters and
/// projections, or null. With 'decode', a scan from the cache below returns
/// its columns as stored.
core::PlanNodePtr fuse(const OneStep& step, CacheConnector& connector, bool decode, AggregationRewrite& out) {
  constexpr int64_t kMaxStateBytes = 4096;
  constexpr int64_t kMaxSlots = 256;
  const auto& keys = step.raw->groupingKeys();
  auto bounded = keyRanges(step, connector, kMaxSlots);
  if (!bounded.has_value() || step.aggregates.empty()) {
    return nullptr;
  }
  auto& [ranges, coded, described, slots] = *bounded;
  std::vector<core::TypedExprPtr> exprs(keys.begin(), keys.end());
  for (const auto& aggregate : step.aggregates) {
    const auto& inputs = aggregate.call->inputs();
    const auto* constant = inputs.size() == 1 ? dynamic_cast<const core::ConstantTypedExpr*>(inputs[0].get()) : nullptr;
    const bool rows = inputs.empty() || (constant != nullptr && !constant->isNull() && baseName(aggregate.call->name()) == "count");
    if (!rows && inputs.size() != 1) {
      return nullptr;
    }
    exprs.push_back(rows ? nullptr : inputs[0]);
  }
  std::vector<core::TypedExprPtr> filters;
  auto source = belowChain(step.raw->sources()[0], exprs, filters);
  if (source == nullptr) {
    return nullptr;
  }
  const bool chain = source != step.raw->sources()[0];
  std::vector<cudf_velox::ColumnFilter> columnFilters;
  std::string why;
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(source.get())) {
    if (scan->tableHandle()->connectorId() == connector.connectorId()) {
      auto lifted = liftFilters(*scan, why);
      if (!lifted.has_value()) {
        out.log.push_back(fmt::format("aggregation {} is not fused: {}", step.aggregation->id(), why));
        return nullptr;
      }
      source = lifted->scan;
      filters.insert(filters.end(), lifted->filters.begin(), lifted->filters.end());
      columnFilters = std::move(lifted->columnFilters);
    }
  }
  if (!chain && filters.empty() && columnFilters.empty()) {
    return nullptr;
  }
  auto reject = [&](const std::string& reason) -> core::PlanNodePtr {
    out.log.push_back(fmt::format("aggregation {} is not fused: {}", step.aggregation->id(), reason));
    return nullptr;
  };

  // The keys as source columns, dictionary keys read as their codes.
  std::vector<core::FieldAccessTypedExprPtr> sourceKeys;
  std::set<std::string> codedColumns;
  for (size_t k = 0; k < keys.size(); ++k) {
    const auto field = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(exprs[k]);
    if (field == nullptr || !field->isInputColumn()) {
      return reject(fmt::format("key {} is computed", keys[k]->name()));
    }
    const bool isCoded = coded.count(keys[k]->name()) > 0;
    if (isCoded) {
      codedColumns.insert(field->name());
    }
    sourceKeys.push_back(std::make_shared<core::FieldAccessTypedExpr>(isCoded ? INTEGER() : field->type(), field->name()));
  }
  std::vector<core::TypedExprPtr> inputs(exprs.begin() + keys.size(), exprs.end());
  if (!codedColumns.empty()) {
    for (const auto& expr : inputs) {
      if (expr != nullptr && references(expr, codedColumns)) {
        return reject("an aggregate reads a dictionary key");
      }
    }
    for (const auto& filter : filters) {
      if (references(filter, codedColumns)) {
        return reject("a filter reads a dictionary key");
      }
    }
    for (const auto& filter : columnFilters) {
      if (codedColumns.count(filter.column)) {
        return reject("a filter reads a dictionary key");
      }
    }
    source = withCodes(source, codedColumns);
    if (source == nullptr) {
      return reject("dictionary keys cannot be read as codes");
    }
  }
  cudf_velox::Encodings encodings;
  bool encoded = false;
  if (decode) {
    if (auto scan = encodedScan(source, connector, filters, inputs, columnFilters, encodings, why)) {
      encoded = scan != source;
      source = std::move(scan);
    } else {
      out.log.push_back(fmt::format("aggregation {} reads decoded columns: {}", step.aggregation->id(), why));
    }
  }
  const auto& input = source->outputType();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] != nullptr) {
      if (auto reason = cudf_velox::unfusable(inputs[i], input, encodings)) {
        return reject(*reason);
      }
    }
    if (auto reason = cudf_velox::unfusable(step.aggregates[i], inputs[i] ? inputs[i]->type() : nullptr)) {
      return reject(*reason);
    }
  }
  for (const auto& filter : filters) {
    if (auto reason = cudf_velox::unfusable(filter, input, encodings)) {
      return reject(*reason);
    }
  }
  for (const auto& filter : columnFilters) {
    const auto encoding = encodings.find(filter.column);
    if (auto reason = cudf_velox::unfusable(
            *filter.filter,
            input->findChild(filter.column),
            encoding == encodings.end() ? std::nullopt : std::optional(encoding->second))) {
      return reject(*reason);
    }
  }
  const auto bytes = cudf_velox::fusedStateBytes(slots, step.aggregates, inputs);
  if (bytes > kMaxStateBytes) {
    return reject(fmt::format("{} bytes of state", bytes));
  }
  out.log.push_back(fmt::format(
      "aggregation {}{}{} is fused with {} filters and its inputs' expressions over {}{} into one reduction of {} "
      "bytes of state",
      step.aggregation->id(),
      keys.empty() ? "" : " grouped by ",
      fmt::join(described, ", "),
      filters.size() + columnFilters.size(),
      source->name(),
      encoded ? " read encoded" : "",
      bytes));
  return std::make_shared<cudf_velox::FusedAggregationNode>(
      step.aggregation->id(),
      sourceKeys,
      step.raw->aggregateNames(),
      step.aggregates,
      source,
      std::move(ranges),
      step.aggregation->outputType(),
      std::move(inputs),
      std::move(filters),
      std::move(columnFilters),
      std::move(encodings));
}

core::TypedExprPtr call(std::string name, std::vector<core::TypedExprPtr> inputs) {
  return std::make_shared<core::CallTypedExpr>(BOOLEAN(), std::move(inputs), std::move(name));
}

/// 'join', a semi or anti join on one key filtered by 'a <> b', 'a' from the
/// side tested for a match and 'b' from the other, as a join of the other
/// side with the min and max of 'a' per key: some 'a' differs from 'b'
/// exactly when the min or the max does. The tested side must be a scan from
/// the cache clustered by the key, so that the min and max come from one
/// step over pre-grouped input, and 'b' a stored column, never null. Null if
/// 'join' is not such a join.
core::PlanNodePtr existsAggregation(const core::HashJoinNode& join, CacheConnector& connector, AggregationRewrite& out) {
  const bool anti = join.isAntiJoin();
  const bool testedLeft = join.isRightSemiFilterJoin();
  if (!(anti || testedLeft || join.isLeftSemiFilterJoin()) || join.isNullAware() || join.leftKeys().size() != 1 ||
      join.filter() == nullptr) {
    return nullptr;
  }
  const auto& tested = join.sources()[testedLeft ? 0 : 1];
  const auto& other = join.sources()[testedLeft ? 1 : 0];
  const auto& testedKey = testedLeft ? join.leftKeys()[0] : join.rightKeys()[0];
  const auto& otherKey = testedLeft ? join.rightKeys()[0] : join.leftKeys()[0];
  const auto* neq = dynamic_cast<const core::CallTypedExpr*>(join.filter().get());
  if (neq == nullptr || baseName(neq->name()) != "neq" || neq->inputs().size() != 2) {
    return nullptr;
  }
  auto a = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(neq->inputs()[0]);
  auto b = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(neq->inputs()[1]);
  if (a == nullptr || b == nullptr || !a->isInputColumn() || !b->isInputColumn()) {
    return nullptr;
  }
  if (!tested->outputType()->containsChild(a->name())) {
    std::swap(a, b);
  }
  if (!tested->outputType()->containsChild(a->name()) || !other->outputType()->containsChild(b->name())) {
    return nullptr;
  }
  std::vector<std::string> columns{testedKey->name(), a->name()};
  const auto* scan = traceToScan(tested, columns);
  if (scan == nullptr || scan->tableHandle()->connectorId() != connector.connectorId() ||
      !traceValues(other, b->name()).has_value()) {
    return nullptr;
  }
  const auto& table = scan->tableHandle()->name();
  const std::vector<std::string> splitKeys{columns[0]};
  const auto existing = out.splitKeys.find(table);
  if (!connector.clustered(table, columns[0]) || (existing != out.splitKeys.end() && existing->second != splitKeys)) {
    return nullptr;
  }
  auto unique = [&](std::string name) {
    while (other->outputType()->containsChild(name) || tested->outputType()->containsChild(name)) {
      name += "_";
    }
    return name;
  };
  const auto low = unique(a->name() + "_min");
  const auto high = unique(a->name() + "_max");
  std::vector<AggregationNode::Aggregate> aggregates;
  for (const auto* function : {"min", "max"}) {
    aggregates.push_back(
        {std::make_shared<core::CallTypedExpr>(a->type(), std::vector<core::TypedExprPtr>{a}, function), {a->type()}});
  }
  auto bounds = std::make_shared<AggregationNode>(
      join.id() + ".bounds",
      AggregationNode::Step::kSingle,
      std::vector<core::FieldAccessTypedExprPtr>{testedKey},
      std::vector<core::FieldAccessTypedExprPtr>{testedKey},
      std::vector<std::string>{low, high},
      aggregates,
      /*ignoreNullKeys=*/false,
      /*noGroupsSpanBatches=*/false,
      std::nullopt,
      tested);
  out.splitKeys[table] = splitKeys;
  const auto lowField = std::make_shared<core::FieldAccessTypedExpr>(a->type(), low);
  const auto highField = std::make_shared<core::FieldAccessTypedExpr>(a->type(), high);
  out.log.push_back(fmt::format(
      "{} join {} on {} with {} becomes a join with the min and max of {}.{} per {}.{}, aggregated in one step over "
      "pre-grouped input",
      anti ? "anti" : "semi",
      join.id(),
      otherKey->name(),
      join.filter()->toString(),
      table,
      columns[1],
      table,
      columns[0]));
  if (!anti) {
    return std::make_shared<core::HashJoinNode>(
        join.id(),
        core::JoinType::kInner,
        false,
        std::vector<core::FieldAccessTypedExprPtr>{testedKey},
        std::vector<core::FieldAccessTypedExprPtr>{otherKey},
        call("or", {call("neq", {lowField, b}), call("neq", {highField, b})}),
        bounds,
        other,
        join.outputType());
  }
  auto names = join.outputType()->names();
  auto types = join.outputType()->children();
  if (!join.outputType()->containsChild(b->name())) {
    names.push_back(b->name());
    types.push_back(b->type());
  }
  names.insert(names.end(), {low, high});
  types.insert(types.end(), {a->type(), a->type()});
  auto matched = std::make_shared<core::HashJoinNode>(
      join.id(),
      core::JoinType::kRight,
      false,
      std::vector<core::FieldAccessTypedExprPtr>{testedKey},
      std::vector<core::FieldAccessTypedExprPtr>{otherKey},
      nullptr,
      bounds,
      other,
      ROW(std::move(names), std::move(types)));
  auto kept = std::make_shared<core::FilterNode>(
      join.id() + ".none_differ",
      call("or", {call("is_null", {lowField}), call("and", {call("eq", {lowField, b}), call("eq", {highField, b})})}),
      matched);
  std::vector<core::TypedExprPtr> projections;
  for (size_t i = 0; i < join.outputType()->size(); ++i) {
    projections.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(join.outputType()->childAt(i), join.outputType()->nameOf(i)));
  }
  return std::make_shared<core::ProjectNode>(join.id() + ".output", join.outputType()->names(), projections, kept);
}

/// A scan from the cache with filters whose rows 'node''s column 'key' takes
/// its values from, through filters, projections and inner joins, with the
/// key's name in its output.
struct KeyOrigin {
  std::shared_ptr<const core::TableScanNode> scan;
  std::string key;
};

std::optional<KeyOrigin> keyOrigin(const core::PlanNodePtr& node, const std::string& key, CacheConnector& connector) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node.get());
      scan != nullptr && scan->tableHandle()->connectorId() == connector.connectorId()) {
    bool filtered = false;
    if (const auto* fused = dynamic_cast<const FusedScanHandle*>(scan->tableHandle().get())) {
      filtered = !fused->filters().empty() || !fused->columnFilters().empty() || !fused->keyFilters().empty();
    } else if (const auto* hive = dynamic_cast<const connector::hive::HiveTableHandle*>(scan->tableHandle().get())) {
      filtered = (!hive->subfieldFilters().empty() || hive->remainingFilter() != nullptr) &&
          !hive->tableParameters().count(CacheConnector::kZoneOnly);
    }
    if (!filtered) {
      return std::nullopt;
    }
    return KeyOrigin{std::static_pointer_cast<const core::TableScanNode>(node), key};
  }
  if (dynamic_cast<const core::FilterNode*>(node.get())) {
    return keyOrigin(node->sources()[0], key, connector);
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    const auto index = project->outputType()->getChildIdxIfExists(key);
    const auto field = index.has_value()
        ? std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(project->projections()[*index])
        : nullptr;
    return field != nullptr && field->isInputColumn() ? keyOrigin(node->sources()[0], field->name(), connector)
                                                      : std::nullopt;
  }
  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(node.get())) {
    if (!join->isInnerJoin()) {
      return std::nullopt;
    }
    for (size_t side = 0; side < 2; ++side) {
      const auto& keys = side == 0 ? join->leftKeys() : join->rightKeys();
      const auto& otherKeys = side == 0 ? join->rightKeys() : join->leftKeys();
      if (!join->sources()[side]->outputType()->containsChild(key)) {
        continue;
      }
      if (auto found = keyOrigin(join->sources()[side], key, connector)) {
        return found;
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i]->name() == key) {
          return keyOrigin(join->sources()[1 - side], otherKeys[i]->name(), connector);
        }
      }
    }
  }
  return std::nullopt;
}

/// Rewrites the input of an aggregation to the rows whose key the other side
/// of a join can match.
class KeyPushdown {
 public:
  explicit KeyPushdown(KeyOrigin origin) : origin_(std::move(origin)) {}

  /// 'node', an aggregation grouped by 'key' alone, possibly under filters and
  /// projections passing the key through, over the rows of its input whose
  /// key is in the origin's. Null if 'node' is no such aggregation.
  core::PlanNodePtr aggregation(const core::PlanNodePtr& node, const std::string& key) {
    const auto* raw = node.get();
    std::string below = key;
    if (dynamic_cast<const core::FilterNode*>(raw) ||
        (dynamic_cast<const core::ProjectNode*>(raw) &&
         passesThrough(*static_cast<const core::ProjectNode*>(raw), key, below))) {
      auto source = aggregation(node->sources()[0], below);
      return source ? withSources(node, {source}) : nullptr;
    }
    if (dynamic_cast<const cudf_velox::FusedAggregationNode*>(raw)) {
      return nullptr;
    }
    const auto* direct = dynamic_cast<const cudf_velox::DenseAggregationNode*>(raw);
    if (direct != nullptr) {
      return groupedBy(*direct, key) ? restricted(node, {restrict(node->sources()[0], key, false)}) : nullptr;
    }
    const auto step = oneStep(node);
    if (!step.has_value() || !groupedBy(*step->raw, key)) {
      return nullptr;
    }
    if (step->raw == step->aggregation) {
      return restricted(node, {restrict(node->sources()[0], key, false)});
    }
    const auto& rawNode = node->sources()[0]->sources()[0];
    auto partial = restricted(rawNode, {restrict(rawNode->sources()[0], key, false)});
    if (partial == nullptr) {
      return nullptr;
    }
    auto exchange = withSources(node->sources()[0], {partial});
    return withSources(node, {exchange});
  }

  int32_t semiJoins() const {
    return count_;
  }

 private:
  static bool groupedBy(const AggregationNode& aggregation, const std::string& key) {
    return aggregation.groupingKeys().size() == 1 && aggregation.groupingKeys()[0]->name() == key &&
        !aggregation.isPreGrouped();
  }

  /// 'node' over 'sources', or null if no semi join was added.
  core::PlanNodePtr restricted(const core::PlanNodePtr& node, const std::vector<core::PlanNodePtr>& sources) const {
    return count_ > 0 ? withSources(node, sources) : nullptr;
  }

  static bool passesThrough(const core::ProjectNode& project, const std::string& key, std::string& below) {
    const auto index = project.outputType()->getChildIdxIfExists(key);
    const auto field = index.has_value()
        ? std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(project.projections()[*index])
        : nullptr;
    if (field == nullptr || !field->isInputColumn()) {
      return false;
    }
    below = field->name();
    return true;
  }

  /// 'node''s rows whose column 'key' is in the origin's keys, keeping only
  /// those past inner joins on the key on both of their sides. Only below
  /// such a join, where it saves the join work, is 'node' semi joined;
  /// otherwise it is returned as is.
  core::PlanNodePtr restrict(const core::PlanNodePtr& node, const std::string& key, bool belowJoin) {
    if (dynamic_cast<const core::FilterNode*>(node.get())) {
      return withSources(node, {restrict(node->sources()[0], key, belowJoin)});
    }
    if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
      std::string below;
      if (passesThrough(*project, key, below)) {
        return withSources(node, {restrict(node->sources()[0], below, belowJoin)});
      }
    }
    if (const auto* join = dynamic_cast<const core::HashJoinNode*>(node.get()); join != nullptr && join->isInnerJoin()) {
      for (size_t i = 0; i < join->leftKeys().size(); ++i) {
        if (join->leftKeys()[i]->name() == key || join->rightKeys()[i]->name() == key) {
          return withSources(
              node,
              {restrict(join->sources()[0], join->leftKeys()[i]->name(), true),
               restrict(join->sources()[1], join->rightKeys()[i]->name(), true)});
        }
      }
    }
    if (!belowJoin) {
      return node;
    }
    const auto& scan = *origin_.scan;
    ++count_;
    // The filters may read any column the scan returns, so it returns them all.
    std::vector<std::string> names;
    connector::ColumnHandleMap assignments;
    std::string name;
    for (const auto& column : scan.outputType()->names()) {
      std::string renamed = fmt::format("{}_keys{}", column, count_);
      while (node->outputType()->containsChild(renamed)) {
        renamed += "_";
      }
      if (column == origin_.key) {
        name = renamed;
      }
      names.push_back(renamed);
      assignments.emplace(renamed, scan.assignments().at(column));
    }
    const auto keyType = scan.outputType()->findChild(origin_.key);
    auto keys = std::make_shared<core::TableScanNode>(
        fmt::format("{}.keys{}", scan.id(), count_),
        ROW(std::move(names), std::vector<TypePtr>(scan.outputType()->children())),
        scan.tableHandle(),
        assignments);
    return std::make_shared<core::HashJoinNode>(
        fmt::format("{}.semi{}", node->id(), count_),
        core::JoinType::kLeftSemiFilter,
        false,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(node->outputType()->findChild(key), key)},
        std::vector<core::FieldAccessTypedExprPtr>{std::make_shared<core::FieldAccessTypedExpr>(keyType, name)},
        nullptr,
        node,
        keys,
        node->outputType());
  }

  const KeyOrigin origin_;
  int32_t count_{0};
};

/// 'join', an inner join of an aggregation on its one grouping key with a
/// side whose key values come from a filtered scan, with the aggregation's
/// input semi joined with that scan below it: groups the join drops are not
/// aggregated. Null if 'join' is not such a join.
core::PlanNodePtr keyPushdown(const core::HashJoinNode& join, CacheConnector& connector, AggregationRewrite& out) {
  if (!join.isInnerJoin() || join.leftKeys().size() != 1) {
    return nullptr;
  }
  for (size_t side = 0; side < 2; ++side) {
    const auto& key = side == 0 ? join.leftKeys()[0] : join.rightKeys()[0];
    const auto& otherKey = side == 0 ? join.rightKeys()[0] : join.leftKeys()[0];
    auto origin = keyOrigin(join.sources()[1 - side], otherKey->name(), connector);
    if (!origin.has_value()) {
      continue;
    }
    KeyPushdown pushdown(*origin);
    auto restricted = pushdown.aggregation(join.sources()[side], key->name());
    if (restricted == nullptr) {
      continue;
    }
    out.log.push_back(fmt::format(
        "join {} keeps only the keys of filtered scan {} of {}, so the aggregation grouped by {} below it reads its "
        "input through {} semi joins with that scan",
        join.id(),
        origin->scan->id(),
        origin->scan->tableHandle()->name(),
        key->name(),
        pushdown.semiJoins()));
    std::vector<core::PlanNodePtr> sources = join.sources();
    sources[side] = restricted;
    return core::HashJoinNode::Builder(join).left(sources[0]).right(sources[1]).build();
  }
  return nullptr;
}

/// 'node' with the fused scan its column 'key' comes from, through filters,
/// projections, inner joins and aggregations grouped by it, passing only
/// rows whose key is one of 'origin''s. Null if there is no such scan, or if
/// 'origin' scans a table at least as large as the scan's: getting its keys
/// then costs about as much as the scan they would save.
core::PlanNodePtr pushKeyFilter(
    const core::PlanNodePtr& node,
    const std::string& key,
    const KeyOrigin& origin,
    CacheConnector& connector) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node.get())) {
    const auto* fused = dynamic_cast<const FusedScanHandle*>(scan->tableHandle().get());
    const auto index = scan->outputType()->getChildIdxIfExists(key);
    if (fused == nullptr || !index.has_value() || scan->id() == origin.scan->id() ||
        connector.store().rows(origin.scan->tableHandle()->name()) >= connector.store().rows(fused->tableName())) {
      return nullptr;
    }
    const auto& output = fused->outputs()[*index];
    std::string column = output.column;
    if (output.source == FusedScanHandle::Source::kExpression) {
      const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(output.expression.get());
      if (field == nullptr || !field->isInputColumn()) {
        return nullptr;
      }
      column = field->name();
    } else if (output.source != FusedScanHandle::Source::kStored) {
      return nullptr;
    }
    return std::make_shared<core::TableScanNode>(
        scan->id(),
        scan->outputType(),
        fused->withKeyFilter({column, origin.scan, origin.key}),
        scan->assignments());
  }
  const auto* aggregation = dynamic_cast<const AggregationNode*>(node.get());
  const bool grouped = aggregation != nullptr && !aggregation->groupId().has_value() &&
      std::any_of(aggregation->groupingKeys().begin(), aggregation->groupingKeys().end(), [&](const auto& k) {
                         return k->name() == key;
                       });
  if (dynamic_cast<const core::FilterNode*>(node.get()) || grouped ||
      (dynamic_cast<const core::LocalPartitionNode*>(node.get()) && node->sources().size() == 1)) {
    auto source = pushKeyFilter(node->sources()[0], key, origin, connector);
    return source ? withSources(node, {source}) : nullptr;
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    const auto index = project->outputType()->getChildIdxIfExists(key);
    const auto field = index.has_value()
        ? std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(project->projections()[*index])
        : nullptr;
    if (field == nullptr || !field->isInputColumn()) {
      return nullptr;
    }
    auto source = pushKeyFilter(node->sources()[0], field->name(), origin, connector);
    return source ? withSources(node, {source}) : nullptr;
  }
  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(node.get()); join != nullptr && join->isInnerJoin()) {
    for (size_t side = 0; side < 2; ++side) {
      if (!join->sources()[side]->outputType()->containsChild(key)) {
        continue;
      }
      auto source = pushKeyFilter(join->sources()[side], key, origin, connector);
      if (source == nullptr) {
        return nullptr;
      }
      auto sources = join->sources();
      sources[side] = source;
      return withSources(node, sources);
    }
  }
  return nullptr;
}

/// 'node' with the probe side scans of inner and semi joins whose build
/// side keys come from a filtered scan passing only the rows with those keys.
core::PlanNodePtr keyFilters(const core::PlanNodePtr& node, CacheConnector& connector, AggregationRewrite& out) {
  std::vector<core::PlanNodePtr> sources;
  bool changed = false;
  for (const auto& source : node->sources()) {
    sources.push_back(keyFilters(source, connector, out));
    changed |= sources.back() != source;
  }
  auto current = node;
  if (changed) {
    if (auto rebuilt = withSources(node, sources)) {
      current = rebuilt;
    } else {
      out.log.push_back(fmt::format("cannot rebuild {} over key filtered scans", node->toString()));
    }
  }
  const auto* join = dynamic_cast<const core::HashJoinNode*>(current.get());
  if (join == nullptr || join->isNullAware() ||
      !(join->isInnerJoin() || join->isLeftSemiFilterJoin() || join->isRightSemiFilterJoin())) {
    return current;
  }
  for (size_t i = 0; i < join->leftKeys().size(); ++i) {
    const auto origin = keyOrigin(join->sources()[1], join->rightKeys()[i]->name(), connector);
    if (!origin.has_value()) {
      continue;
    }
    if (auto probe = pushKeyFilter(join->sources()[0], join->leftKeys()[i]->name(), *origin, connector)) {
      out.log.push_back(fmt::format(
          "join {} passes the keys of filtered scan {} of {} to the scan below its probe side {}",
          join->id(),
          origin->scan->id(),
          origin->scan->tableHandle()->name(),
          join->leftKeys()[i]->name()));
      return core::HashJoinNode::Builder(*join).left(probe).build();
    }
  }
  return current;
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
  if (rules.keyPushdown) {
    if (const auto* join = dynamic_cast<const core::HashJoinNode*>(current.get())) {
      if (auto rewritten = keyPushdown(*join, connector, out)) {
        current = rewritten;
      }
    }
  }
  if (rules.existsAggregation) {
    if (const auto* join = dynamic_cast<const core::HashJoinNode*>(current.get())) {
      if (auto rewritten = existsAggregation(*join, connector, out)) {
        return rewritten;
      }
    }
  }
  if (rules.fuse) {
    if (const auto step = oneStep(current, /*global=*/true)) {
      if (auto fused = fuse(*step, connector, rules.decode || rules.project, out)) {
        return fused;
      }
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
  if (rules.direct) {
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
  if (rules.decode || rules.project) {
    out.plan = fuseScans(out.plan, connector, rules.project, out);
    if (rules.keyFilter) {
      out.plan = keyFilters(out.plan, connector, out);
    }
  }
  return out;
}

} // namespace facebook::velox::supervelox::cudf_backend
