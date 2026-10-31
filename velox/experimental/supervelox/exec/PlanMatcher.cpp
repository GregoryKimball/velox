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
#include "velox/experimental/supervelox/exec/PlanMatcher.h"

#include <cmath>
#include <limits>
#include <optional>
#include <set>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/vector/ComplexVector.h"
#include "velox/experimental/supervelox/exec/FusedQuery.h"
#include "velox/experimental/supervelox/queries/Derived.h"

namespace facebook::velox::supervelox {
namespace {

constexpr long double kInfinity = std::numeric_limits<long double>::infinity();

/// The values a filter admits for one column, in the column's logical units:
/// days for a date, the decimal value for a decimal.
struct Range {
  long double min{-kInfinity};
  long double max{kInfinity};
  bool minExclusive{false};
  bool maxExclusive{false};

  void atLeast(long double value, bool exclusive) {
    if (value > min || (value == min && exclusive)) {
      min = value;
      minExclusive = exclusive;
    }
  }

  void atMost(long double value, bool exclusive) {
    if (value < max || (value == max && exclusive)) {
      max = value;
      maxExclusive = exclusive;
    }
  }

  bool hasMin() const {
    return std::isfinite(min);
  }

  bool hasMax() const {
    return std::isfinite(max);
  }
};

/// What a tree of scans, projections, filters and joins computes: each
/// output column as a canonical expression over table columns, and the
/// filters as ranges and value sets of table columns plus the conjuncts that
/// are neither. TPC-H column names are unique across tables, so a table
/// column is named by its own name.
struct Relation {
  std::set<std::string> tables;
  std::map<std::string, std::string> columns;
  std::map<std::string, Range> ranges;
  /// Columns restricted to a list of string values.
  std::map<std::string, std::set<std::string>> values;
  /// Joins as `kind(left=right)`, and deduplications as `distinct(keys)`.
  std::vector<std::string> joins;
  /// Canonical text of the conjuncts that are not ranges, value sets or
  /// disjunctions.
  std::vector<std::string> residuals;
  /// Conjuncts that are an OR of branches, each branch only ranges and value
  /// sets.
  std::vector<std::vector<Relation>> disjunctions;
};

/// An aggregation over a relation, its partial and final steps taken as one
/// and the exchange between them skipped.
struct Aggregation {
  std::vector<std::string> keys;
  std::vector<std::string> aggregates;
  /// The columns the matched subtree produces: the keys and aggregates, or
  /// a projection over them that the match takes in too.
  std::vector<std::string> outputs;
  Relation input;
};

std::string functionName(const std::string& name) {
  const auto dot = name.rfind('.');
  return dot == std::string::npos ? name : name.substr(dot + 1);
}

const core::TypedExprPtr& stripCasts(const core::TypedExprPtr& expr) {
  if (dynamic_cast<const core::CastTypedExpr*>(expr.get())) {
    return stripCasts(expr->inputs()[0]);
  }
  return expr;
}

/// Days since the epoch of a 'YYYY-MM-DD' date.
std::optional<int64_t> parseDate(const std::string& text) {
  int year, month, day;
  if (std::sscanf(text.c_str(), "%d-%d-%d", &year, &month, &day) != 3) {
    return std::nullopt;
  }
  year -= month <= 2;
  const int era = (year >= 0 ? year : year - 399) / 400;
  const int yearOfEra = year - era * 400;
  const int dayOfYear = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
  return static_cast<int64_t>(era) * 146097 + dayOfEra - 719468;
}

/// The value of a numeric or date literal, in logical units.
std::optional<long double> literalValue(const core::TypedExprPtr& expr) {
  const auto* cast = dynamic_cast<const core::CastTypedExpr*>(expr.get());
  const auto& inner = stripCasts(expr);
  const auto* constant = dynamic_cast<const core::ConstantTypedExpr*>(inner.get());
  if (!constant || constant->hasValueVector() || constant->value().isNull()) {
    return std::nullopt;
  }
  const auto& type = constant->type();
  const auto& value = constant->value();
  if (cast && cast->type()->isDate() && type->isVarchar()) {
    return parseDate(value.value<std::string>());
  }
  if (type->isDecimal()) {
    const auto scale = getDecimalPrecisionScale(*type).second;
    const long double unscaled = type->isShortDecimal() ? static_cast<long double>(value.value<int64_t>())
                                                        : static_cast<long double>(value.value<int128_t>());
    return unscaled / std::pow(10.0L, scale);
  }
  switch (type->kind()) {
    case TypeKind::TINYINT:
      return value.value<int8_t>();
    case TypeKind::SMALLINT:
      return value.value<int16_t>();
    case TypeKind::INTEGER:
      return value.value<int32_t>();
    case TypeKind::BIGINT:
      return value.value<int64_t>();
    case TypeKind::REAL:
      return value.value<float>();
    case TypeKind::DOUBLE:
      return value.value<double>();
    default:
      return std::nullopt;
  }
}

std::optional<std::string> stringLiteral(const core::TypedExprPtr& expr) {
  const auto* constant = dynamic_cast<const core::ConstantTypedExpr*>(stripCasts(expr).get());
  if (!constant || !constant->type()->isVarchar() || constant->hasValueVector() || constant->value().isNull()) {
    return std::nullopt;
  }
  return constant->value().value<std::string>();
}

/// The strings of an array literal, as an IN list is given.
std::optional<std::vector<std::string>> stringList(const core::TypedExprPtr& expr) {
  if (const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    if (functionName(call->name()) != "array_constructor") {
      return std::nullopt;
    }
    std::vector<std::string> strings;
    for (const auto& input : call->inputs()) {
      auto text = stringLiteral(input);
      if (!text) {
        return std::nullopt;
      }
      strings.push_back(std::move(*text));
    }
    return strings;
  }
  const auto* constant = dynamic_cast<const core::ConstantTypedExpr*>(expr.get());
  if (!constant || !constant->hasValueVector()) {
    return std::nullopt;
  }
  const auto* array = constant->valueVector()->wrappedVector()->as<ArrayVector>();
  if (!array || !array->elements()->type()->isVarchar()) {
    return std::nullopt;
  }
  const auto index = constant->valueVector()->wrappedIndex(0);
  auto* elements = array->elements()->as<SimpleVector<StringView>>();
  std::vector<std::string> strings;
  for (auto i = 0; i < array->sizeAt(index); ++i) {
    strings.push_back(elements->valueAt(array->offsetAt(index) + i).str());
  }
  return strings;
}

/// `expr` as text that does not depend on how the plan named intermediate
/// columns or where it placed casts: columns are replaced by what `columns`
/// says they compute, casts are dropped and numbers are printed by value.
std::string canonical(const core::TypedExprPtr& expr, const std::map<std::string, std::string>& columns) {
  if (const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (field->isInputColumn()) {
      auto it = columns.find(field->name());
      return it == columns.end() ? field->name() : it->second;
    }
    return canonical(field->inputs()[0], columns) + "." + field->name();
  }
  if (dynamic_cast<const core::CastTypedExpr*>(expr.get())) {
    if (auto value = literalValue(expr)) {
      return fmt::format("{}", static_cast<double>(*value));
    }
    return canonical(expr->inputs()[0], columns);
  }
  if (const auto* constant = dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
    if (auto value = literalValue(expr)) {
      return fmt::format("{}", static_cast<double>(*value));
    }
    if (auto text = stringLiteral(expr)) {
      return "'" + *text + "'";
    }
    return constant->toString();
  }
  if (const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    const std::string name = functionName(call->name());
    bool literalsOnly = true;
    for (const auto& input : call->inputs()) {
      literalsOnly &= dynamic_cast<const core::ConstantTypedExpr*>(input.get()) != nullptr;
    }
    if (name == "count" && literalsOnly) {
      return "count()";
    }
    std::string text = name + "(";
    for (size_t i = 0; i < call->inputs().size(); ++i) {
      text += (i ? "," : "") + canonical(call->inputs()[i], columns);
    }
    return text + ")";
  }
  return expr->toString();
}

/// The table column `expr` reads unchanged, if it is one.
std::optional<std::string> tableColumn(const core::TypedExprPtr& expr, const Relation& relation) {
  const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(stripCasts(expr).get());
  if (!field || !field->isInputColumn()) {
    return std::nullopt;
  }
  auto it = relation.columns.find(field->name());
  if (it == relation.columns.end() || it->second.find('(') != std::string::npos) {
    return std::nullopt;
  }
  return it->second;
}

/// A BIGINT or short decimal constant at the int64 limits, which is how a
/// range filter turned back into an expression writes an open side.
bool isInt64Limit(const core::TypedExprPtr& expr) {
  const auto* constant = dynamic_cast<const core::ConstantTypedExpr*>(expr.get());
  if (!constant || constant->hasValueVector() || constant->value().isNull() ||
      constant->type()->kind() != TypeKind::BIGINT) {
    return false;
  }
  const auto value = constant->value().value<int64_t>();
  return value <= std::numeric_limits<int64_t>::min() + 1 || value >= std::numeric_limits<int64_t>::max() - 1;
}

/// The values set `column` keeps, intersected with `values`.
void restrictValues(Relation& relation, const std::string& column, const std::vector<std::string>& values) {
  std::set<std::string> kept(values.begin(), values.end());
  auto it = relation.values.find(column);
  if (it != relation.values.end()) {
    std::set<std::string> both;
    for (const auto& value : kept) {
      if (it->second.count(value)) {
        both.insert(value);
      }
    }
    kept = std::move(both);
  }
  relation.values[column] = std::move(kept);
}

/// A conjunct that compares a column with a literal, added as a range or a
/// value set. False if `expr` is not one.
bool addComparison(const core::TypedExprPtr& expr, Relation& relation) {
  const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get());
  if (!call) {
    return false;
  }
  const std::string name = functionName(call->name());
  const auto& inputs = call->inputs();
  if (name == "between" && inputs.size() == 3) {
    auto column = tableColumn(inputs[0], relation);
    auto low = literalValue(inputs[1]);
    auto high = literalValue(inputs[2]);
    if (!column || !low || !high) {
      return false;
    }
    if (!isInt64Limit(stripCasts(inputs[1]))) {
      relation.ranges[*column].atLeast(*low, false);
    }
    if (!isInt64Limit(stripCasts(inputs[2]))) {
      relation.ranges[*column].atMost(*high, false);
    }
    return true;
  }
  if (name == "in" && inputs.size() == 2) {
    auto column = tableColumn(inputs[0], relation);
    auto strings = stringList(inputs[1]);
    if (!column || !strings) {
      return false;
    }
    restrictValues(relation, *column, *strings);
    return true;
  }
  if (inputs.size() != 2) {
    return false;
  }
  auto column = tableColumn(inputs[0], relation);
  auto literal = inputs[1];
  std::string op = name;
  if (!column) {
    column = tableColumn(inputs[1], relation);
    literal = inputs[0];
    static const std::map<std::string, std::string> kFlipped = {
        {"lt", "gt"}, {"lte", "gte"}, {"gt", "lt"}, {"gte", "lte"}, {"eq", "eq"}};
    auto it = kFlipped.find(name);
    if (it == kFlipped.end()) {
      return false;
    }
    op = it->second;
  }
  if (!column) {
    return false;
  }
  if (auto text = stringLiteral(literal); text && op == "eq") {
    restrictValues(relation, *column, {*text});
    return true;
  }
  auto value = literalValue(literal);
  if (!value) {
    return false;
  }
  if (isInt64Limit(stripCasts(literal)) && op != "eq") {
    return true;
  }
  Range& range = relation.ranges[*column];
  if (op == "lt" || op == "lte") {
    range.atMost(*value, op == "lt");
  } else if (op == "gt" || op == "gte") {
    range.atLeast(*value, op == "gt");
  } else if (op == "eq") {
    range.atLeast(*value, false);
    range.atMost(*value, false);
  } else {
    return false;
  }
  return true;
}

/// Adds the conjuncts of `expr`: comparisons of a column with literals as
/// ranges and value sets, and anything else as canonical text.
void addConjuncts(const core::TypedExprPtr& expr, Relation& relation);

void addDisjuncts(const core::TypedExprPtr& expr, std::vector<core::TypedExprPtr>& disjuncts) {
  const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get());
  if (call && functionName(call->name()) == "or") {
    for (const auto& input : call->inputs()) {
      addDisjuncts(input, disjuncts);
    }
    return;
  }
  disjuncts.push_back(expr);
}

/// `expr` as branches of ranges and value sets, if it is an OR of those.
std::optional<std::vector<Relation>> readDisjunction(const core::TypedExprPtr& expr, const Relation& relation) {
  std::vector<core::TypedExprPtr> disjuncts;
  addDisjuncts(expr, disjuncts);
  if (disjuncts.size() < 2) {
    return std::nullopt;
  }
  std::vector<Relation> branches;
  for (const auto& disjunct : disjuncts) {
    Relation branch;
    branch.columns = relation.columns;
    addConjuncts(disjunct, branch);
    if (!branch.residuals.empty() || !branch.disjunctions.empty()) {
      return std::nullopt;
    }
    branch.columns.clear();
    branches.push_back(std::move(branch));
  }
  return branches;
}

void addConjuncts(const core::TypedExprPtr& expr, Relation& relation) {
  const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get());
  if (call && functionName(call->name()) == "and") {
    for (const auto& input : call->inputs()) {
      addConjuncts(input, relation);
    }
    return;
  }
  if (addComparison(expr, relation)) {
    return;
  }
  if (auto branches = readDisjunction(expr, relation)) {
    relation.disjunctions.push_back(std::move(*branches));
    return;
  }
  relation.residuals.push_back(canonical(expr, relation.columns));
}

/// Adds the ranges a connector's pushed-down filters put on table columns.
bool addSubfieldFilters(const connector::hive::FileTableHandle& handle, const RowTypePtr& types, Relation& relation) {
  for (const auto& [subfield, filter] : handle.subfieldFilters()) {
    const std::string column = subfield.toString();
    const auto index = types->getChildIdxIfExists(column);
    if (!index) {
      return false;
    }
    const auto& type = types->childAt(*index);
    if (filter->kind() == common::FilterKind::kBytesValues) {
      const auto& values = static_cast<const common::BytesValues*>(filter.get())->values();
      restrictValues(relation, column, std::vector<std::string>(values.begin(), values.end()));
      continue;
    }
    if (filter->kind() == common::FilterKind::kBytesRange) {
      const auto* bytes = static_cast<const common::BytesRange*>(filter.get());
      if (!bytes->isSingleValue()) {
        return false;
      }
      restrictValues(relation, column, {bytes->lower()});
      continue;
    }
    const long double unit = type->isDecimal() ? std::pow(10.0L, getDecimalPrecisionScale(*type).second) : 1.0L;
    Range& range = relation.ranges[column];
    switch (filter->kind()) {
      case common::FilterKind::kBigintRange: {
        const auto* bigint = static_cast<const common::BigintRange*>(filter.get());
        if (bigint->lower() != std::numeric_limits<int64_t>::min()) {
          range.atLeast(bigint->lower() / unit, false);
        }
        if (bigint->upper() != std::numeric_limits<int64_t>::max()) {
          range.atMost(bigint->upper() / unit, false);
        }
        break;
      }
      case common::FilterKind::kHugeintRange: {
        const auto* hugeint = static_cast<const common::HugeintRange*>(filter.get());
        range.atLeast(static_cast<long double>(hugeint->lower()) / unit, false);
        range.atMost(static_cast<long double>(hugeint->upper()) / unit, false);
        break;
      }
      case common::FilterKind::kDoubleRange: {
        const auto* real = static_cast<const common::DoubleRange*>(filter.get());
        if (!real->lowerUnbounded()) {
          range.atLeast(real->lower(), real->lowerExclusive());
        }
        if (!real->upperUnbounded()) {
          range.atMost(real->upper(), real->upperExclusive());
        }
        break;
      }
      default:
        return false;
    }
  }
  return true;
}

const core::PlanNode* skipLocalExchanges(const core::PlanNode* node) {
  while ((dynamic_cast<const core::LocalPartitionNode*>(node) || dynamic_cast<const core::LocalMergeNode*>(node)) &&
         node->sources().size() == 1) {
    node = node->sources()[0].get();
  }
  return node;
}

std::optional<Relation> readRelation(const core::PlanNode* node);

/// The aggregation step that reads the input, from the final step of a
/// partial and final pair or a single step.
const core::AggregationNode* inputAggregation(const core::AggregationNode* top) {
  if (!top->globalGroupingSets().empty() || top->groupId()) {
    return nullptr;
  }
  if (top->step() == core::AggregationNode::Step::kSingle) {
    return top;
  }
  if (top->step() != core::AggregationNode::Step::kFinal) {
    return nullptr;
  }
  const auto* raw = dynamic_cast<const core::AggregationNode*>(skipLocalExchanges(top->sources()[0].get()));
  return raw && raw->step() == core::AggregationNode::Step::kPartial ? raw : nullptr;
}

const char* joinKind(core::JoinType type) {
  switch (type) {
    case core::JoinType::kInner:
      return "inner";
    case core::JoinType::kLeftSemiFilter:
      return "left_semi";
    case core::JoinType::kRightSemiFilter:
      return "right_semi";
    default:
      return nullptr;
  }
}

/// Both sides of a join as one relation, restricted to the join's output.
std::optional<Relation> readJoin(const core::HashJoinNode& join) {
  const char* kind = joinKind(join.joinType());
  auto left = readRelation(join.sources()[0].get());
  auto right = readRelation(join.sources()[1].get());
  if (!kind || !left || !right) {
    return std::nullopt;
  }
  Relation relation = std::move(*left);
  relation.tables.insert(right->tables.begin(), right->tables.end());
  relation.ranges.insert(right->ranges.begin(), right->ranges.end());
  relation.values.insert(right->values.begin(), right->values.end());
  relation.joins.insert(relation.joins.end(), right->joins.begin(), right->joins.end());
  relation.residuals.insert(relation.residuals.end(), right->residuals.begin(), right->residuals.end());
  relation.disjunctions.insert(relation.disjunctions.end(), right->disjunctions.begin(), right->disjunctions.end());
  std::map<std::string, std::string> both = relation.columns;
  both.insert(right->columns.begin(), right->columns.end());
  std::string keys;
  for (size_t i = 0; i < join.leftKeys().size(); ++i) {
    keys += (i ? "," : "") + canonical(join.leftKeys()[i], both) + "=" + canonical(join.rightKeys()[i], both);
  }
  relation.joins.push_back(std::string(kind) + "(" + keys + ")");
  relation.columns = both;
  if (join.filter()) {
    addConjuncts(join.filter(), relation);
  }
  relation.columns.clear();
  for (const auto& name : join.outputType()->names()) {
    relation.columns[name] = both.at(name);
  }
  return relation;
}

std::optional<Relation> readRelation(const core::PlanNode* node) {
  node = skipLocalExchanges(node);
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node)) {
    const auto* file = dynamic_cast<const connector::hive::FileTableHandle*>(scan->tableHandle().get());
    if (!file) {
      return std::nullopt;
    }
    const auto& handle = *file;
    Relation relation;
    relation.tables.insert(handle.name());
    for (const auto& [name, column] : scan->assignments()) {
      relation.columns[name] = column->name();
    }
    const RowTypePtr types = handle.dataColumns() ? handle.dataColumns() : scan->outputType();
    if (!addSubfieldFilters(handle, types, relation)) {
      return std::nullopt;
    }
    if (const auto& remaining = handle.remainingFilter()) {
      addConjuncts(remaining, relation);
    }
    return relation;
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node)) {
    auto relation = readRelation(project->sources()[0].get());
    if (!relation) {
      return std::nullopt;
    }
    std::map<std::string, std::string> columns;
    for (size_t i = 0; i < project->names().size(); ++i) {
      columns[project->names()[i]] = canonical(project->projections()[i], relation->columns);
    }
    relation->columns = std::move(columns);
    return relation;
  }
  if (const auto* filter = dynamic_cast<const core::FilterNode*>(node)) {
    auto relation = readRelation(filter->sources()[0].get());
    if (!relation) {
      return std::nullopt;
    }
    addConjuncts(filter->filter(), *relation);
    return relation;
  }
  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(node)) {
    return readJoin(*join);
  }
  if (const auto* top = dynamic_cast<const core::AggregationNode*>(node)) {
    // Grouping with no aggregates keeps one row per key: a deduplication.
    const auto* raw = inputAggregation(top);
    if (!raw || !raw->aggregates().empty()) {
      return std::nullopt;
    }
    auto relation = readRelation(raw->sources()[0].get());
    if (!relation) {
      return std::nullopt;
    }
    std::map<std::string, std::string> columns;
    std::string keys;
    for (const auto& key : raw->groupingKeys()) {
      columns[key->name()] = canonical(key, relation->columns);
      keys += (keys.empty() ? "" : ",") + columns[key->name()];
    }
    relation->joins.push_back("distinct(" + keys + ")");
    relation->columns = std::move(columns);
    return relation;
  }
  return std::nullopt;
}

/// The aggregation at `node`, and when `project` is given, the projection
/// over it as the subtree's output.
std::optional<Aggregation> readAggregation(const core::PlanNode* node, const core::ProjectNode* project = nullptr) {
  const auto* top = dynamic_cast<const core::AggregationNode*>(node);
  const auto* raw = top ? inputAggregation(top) : nullptr;
  if (!raw) {
    return std::nullopt;
  }
  auto input = readRelation(raw->sources()[0].get());
  if (!input) {
    return std::nullopt;
  }
  Aggregation aggregation;
  std::map<std::string, std::string> named;
  for (const auto& key : raw->groupingKeys()) {
    aggregation.keys.push_back(canonical(key, input->columns));
    named[key->name()] = aggregation.keys.back();
    aggregation.outputs.push_back(aggregation.keys.back());
  }
  for (size_t i = 0; i < raw->aggregates().size(); ++i) {
    const auto& aggregate = raw->aggregates()[i];
    if (aggregate.mask || aggregate.distinct || !aggregate.sortingKeys.empty()) {
      return std::nullopt;
    }
    aggregation.aggregates.push_back(canonical(aggregate.call, input->columns));
    named[raw->aggregateNames()[i]] = aggregation.aggregates.back();
    aggregation.outputs.push_back(aggregation.aggregates.back());
  }
  if (project) {
    aggregation.outputs.clear();
    for (const auto& projection : project->projections()) {
      aggregation.outputs.push_back(canonical(projection, named));
    }
  }
  aggregation.input = std::move(*input);
  return aggregation;
}

/// The least stored value a range admits, with the store's `scale`.
int64_t storedMin(const Range& range, int32_t scale, int64_t unbounded) {
  if (!range.hasMin()) {
    return unbounded;
  }
  const long double value = range.min * std::pow(10.0L, scale);
  const long double nearest = std::round(value);
  if (std::fabs(value - nearest) < 1e-6L) {
    return static_cast<int64_t>(nearest) + (range.minExclusive ? 1 : 0);
  }
  return static_cast<int64_t>(std::ceil(value));
}

/// The greatest stored value a range admits, with the store's `scale`.
int64_t storedMax(const Range& range, int32_t scale, int64_t unbounded) {
  if (!range.hasMax()) {
    return unbounded;
  }
  const long double value = range.max * std::pow(10.0L, scale);
  const long double nearest = std::round(value);
  if (std::fabs(value - nearest) < 1e-6L) {
    return static_cast<int64_t>(nearest) - (range.maxExclusive ? 1 : 0);
  }
  return static_cast<int64_t>(std::floor(value));
}

constexpr int64_t kUnbounded = 1LL << 30;

bool rangesOnly(const Aggregation& aggregation, const std::set<std::string>& columns) {
  const Relation& input = aggregation.input;
  std::vector<std::string> plain = aggregation.keys;
  plain.insert(plain.end(), aggregation.aggregates.begin(), aggregation.aggregates.end());
  if (!input.values.empty() || !input.joins.empty() || !input.residuals.empty() || !input.disjunctions.empty() ||
      aggregation.outputs != plain) {
    return false;
  }
  for (const auto& [column, range] : aggregation.input.ranges) {
    if (!columns.count(column)) {
      return false;
    }
  }
  return true;
}

Range rangeOf(const Aggregation& aggregation, const std::string& column) {
  auto it = aggregation.input.ranges.find(column);
  return it == aggregation.input.ranges.end() ? Range{} : it->second;
}

std::optional<QueryParams> matchQ1(const Aggregation& aggregation, ColumnStore& /*store*/) {
  static const std::vector<std::string> kKeys = {"l_returnflag", "l_linestatus"};
  static const std::vector<std::string> kAggregates = {
      "sum(l_quantity)",
      "sum(l_extendedprice)",
      "sum(multiply(l_extendedprice,minus(1,l_discount)))",
      "sum(multiply(multiply(l_extendedprice,minus(1,l_discount)),plus(1,l_tax)))",
      "avg(l_quantity)",
      "avg(l_extendedprice)",
      "avg(l_discount)",
      "count()"};
  if (aggregation.input.tables != std::set<std::string>{"lineitem"} || aggregation.keys != kKeys || aggregation.aggregates != kAggregates ||
      !rangesOnly(aggregation, {"l_shipdate"})) {
    return std::nullopt;
  }
  const Range shipdate = rangeOf(aggregation, "l_shipdate");
  if (shipdate.hasMin()) {
    return std::nullopt;
  }
  QueryParams params;
  params.values["shipdate_max"] = storedMax(shipdate, 0, kUnbounded);
  return params;
}

std::optional<QueryParams> matchQ6(const Aggregation& aggregation, ColumnStore& store) {
  static const std::vector<std::string> kAggregates = {"sum(multiply(l_extendedprice,l_discount))"};
  if (aggregation.input.tables != std::set<std::string>{"lineitem"} || !aggregation.keys.empty() || aggregation.aggregates != kAggregates ||
      !rangesOnly(aggregation, {"l_shipdate", "l_discount", "l_quantity"})) {
    return std::nullopt;
  }
  const Range quantity = rangeOf(aggregation, "l_quantity");
  if (quantity.hasMin()) {
    return std::nullopt;
  }
  const Range shipdate = rangeOf(aggregation, "l_shipdate");
  const Range discount = rangeOf(aggregation, "l_discount");
  const int32_t discountScale = store.metadata("lineitem", "l_discount").scale;
  const int32_t quantityScale = store.metadata("lineitem", "l_quantity").scale;
  QueryParams params;
  params.values["shipdate_min"] = storedMin(shipdate, 0, -kUnbounded);
  params.values["shipdate_max"] = storedMax(shipdate, 0, kUnbounded);
  params.values["discount_min"] = storedMin(discount, discountScale, -kUnbounded);
  params.values["discount_max"] = storedMax(discount, discountScale, kUnbounded);
  params.values["quantity_max"] = storedMax(quantity, quantityScale, kUnbounded);
  return params;
}

/// Matches `text` against `pattern`, where each `?` in the pattern stands
/// for one literal: a quoted string, given without its quotes, or a number.
std::optional<std::vector<std::string>> matchPattern(const std::string& text, const std::string& pattern) {
  std::vector<std::string> literals;
  size_t t = 0;
  for (size_t p = 0; p < pattern.size(); ++p) {
    if (pattern[p] != '?') {
      if (t >= text.size() || text[t] != pattern[p]) {
        return std::nullopt;
      }
      ++t;
      continue;
    }
    if (t < text.size() && text[t] == '\'') {
      const size_t close = text.find('\'', t + 1);
      if (close == std::string::npos) {
        return std::nullopt;
      }
      literals.push_back(text.substr(t + 1, close - t - 1));
      t = close + 1;
    } else {
      const size_t end = text.find_first_not_of("0123456789.-e", t);
      if (end == t) {
        return std::nullopt;
      }
      literals.push_back(text.substr(t, end - t));
      t = end == std::string::npos ? text.size() : end;
    }
  }
  if (t != text.size()) {
    return std::nullopt;
  }
  return literals;
}

/// SQL LIKE with % and _, and no escapes.
bool likeMatches(const char* value, const char* pattern) {
  if (*pattern == '\0') {
    return *value == '\0';
  }
  if (*pattern == '%') {
    for (const char* rest = value;; ++rest) {
      if (likeMatches(rest, pattern + 1)) {
        return true;
      }
      if (*rest == '\0') {
        return false;
      }
    }
  }
  return *value != '\0' && (*pattern == '_' || *pattern == *value) && likeMatches(value + 1, pattern + 1);
}

/// The dictionary codes of table.column whose values `keep` accepts, or
/// nothing when the column has no dictionary small enough for a CodeSet.
template <typename Keep>
std::optional<CodeSet> codesWhere(ColumnStore& store, const std::string& table, const std::string& column, Keep keep) {
  const Column& meta = store.metadata(table, column);
  if (meta.dictionary.empty() || meta.dictionary.size() > 256) {
    return std::nullopt;
  }
  CodeSet codes;
  for (size_t i = 0; i < meta.dictionary.size(); ++i) {
    if (keep(meta.dictionary[i])) {
      codes.add(i);
    }
  }
  return codes;
}

std::optional<CodeSet>
codesOf(ColumnStore& store, const std::string& table, const std::string& column, const std::set<std::string>& values) {
  return codesWhere(store, table, column, [&](const std::string& value) { return values.count(value) > 0; });
}

/// The one value a value set admits.
std::optional<std::string> onlyValue(const Relation& relation, const std::string& column) {
  auto it = relation.values.find(column);
  if (it == relation.values.end() || it->second.size() != 1) {
    return std::nullopt;
  }
  return *it->second.begin();
}

template <typename T>
std::set<std::string> keysOf(const std::map<std::string, T>& map) {
  std::set<std::string> keys;
  for (const auto& entry : map) {
    keys.insert(entry.first);
  }
  return keys;
}

const Range* rangeIn(const Relation& relation, const std::string& column) {
  auto it = relation.ranges.find(column);
  return it == relation.ranges.end() ? nullptr : &it->second;
}

std::optional<QueryParams> matchQ4(const Aggregation& aggregation, ColumnStore& /*store*/) {
  const Relation& input = aggregation.input;
  if (input.tables != std::set<std::string>{"lineitem", "orders"} ||
      aggregation.keys != std::vector<std::string>{"o_orderpriority"} ||
      aggregation.aggregates != std::vector<std::string>{"count()"} ||
      aggregation.outputs != std::vector<std::string>{"o_orderpriority", "count()"} ||
      input.joins != std::vector<std::string>{"distinct(l_orderkey)", "right_semi(l_orderkey=o_orderkey)"} ||
      input.residuals != std::vector<std::string>{"lt(l_commitdate,l_receiptdate)"} || !input.values.empty() ||
      !input.disjunctions.empty() || keysOf(input.ranges) != std::set<std::string>{"o_orderdate"}) {
    return std::nullopt;
  }
  const Range& orderdate = input.ranges.at("o_orderdate");
  QueryParams params;
  params.values["orderdate_min"] = storedMin(orderdate, 0, -kUnbounded);
  params.values["orderdate_max"] = storedMax(orderdate, 0, kUnbounded);
  return params;
}

std::optional<QueryParams> matchQ12(const Aggregation& aggregation, ColumnStore& store) {
  const Relation& input = aggregation.input;
  if (input.tables != std::set<std::string>{"lineitem", "orders"} ||
      aggregation.keys != std::vector<std::string>{"l_shipmode"} || aggregation.aggregates.size() != 2 ||
      aggregation.outputs.size() != 3 || input.joins != std::vector<std::string>{"inner(o_orderkey=l_orderkey)"} ||
      std::set<std::string>(input.residuals.begin(), input.residuals.end()) !=
          std::set<std::string>{"lt(l_commitdate,l_receiptdate)", "lt(l_shipdate,l_commitdate)"} ||
      keysOf(input.values) != std::set<std::string>{"l_shipmode"} || !input.disjunctions.empty()) {
    return std::nullopt;
  }
  auto high = matchPattern(
      aggregation.aggregates[0], "sum(if(or(eq(o_orderpriority,?),eq(o_orderpriority,?)),1,0))");
  auto low = matchPattern(
      aggregation.aggregates[1], "sum(if(and(neq(o_orderpriority,?),neq(o_orderpriority,?)),1,0))");
  if (!high || !low || std::set<std::string>(high->begin(), high->end()) != std::set<std::string>(low->begin(), low->end())) {
    return std::nullopt;
  }
  const Range* receipt = rangeIn(input, "l_receiptdate");
  if (!receipt) {
    return std::nullopt;
  }
  const int64_t receiptMax = storedMax(*receipt, 0, kUnbounded);
  // Lines also need ship < commit < receipt, so bounds on the ship and commit
  // dates that those and the receipt bound imply change nothing.
  for (const auto& [column, implied] : std::map<std::string, int64_t>{
           {"l_receiptdate", receiptMax}, {"l_commitdate", receiptMax - 1}, {"l_shipdate", receiptMax - 2}}) {
    const Range* range = rangeIn(input, column);
    if (range && column != "l_receiptdate" && (range->hasMin() || storedMax(*range, 0, kUnbounded) < implied)) {
      return std::nullopt;
    }
  }
  if (keysOf(input.ranges).size() > 3) {
    return std::nullopt;
  }
  for (const auto& column : keysOf(input.ranges)) {
    if (column != "l_receiptdate" && column != "l_commitdate" && column != "l_shipdate") {
      return std::nullopt;
    }
  }
  auto modes = codesOf(store, "lineitem", "l_shipmode", input.values.at("l_shipmode"));
  auto priorities = codesOf(store, "orders", "o_orderpriority", {high->begin(), high->end()});
  if (!modes || !priorities) {
    return std::nullopt;
  }
  QueryParams params;
  params.values["receiptdate_min"] = storedMin(*receipt, 0, -kUnbounded);
  params.values["receiptdate_max"] = receiptMax;
  putCodes(params, "shipmodes", *modes);
  putCodes(params, "high_priorities", *priorities);
  return params;
}

std::optional<QueryParams> matchQ14(const Aggregation& aggregation, ColumnStore& store) {
  static const std::string kRevenue = "multiply(l_extendedprice,minus(1,l_discount))";
  const Relation& input = aggregation.input;
  if (input.tables != std::set<std::string>{"lineitem", "part"} || !aggregation.keys.empty() ||
      aggregation.aggregates.size() != 2 || aggregation.aggregates[0] != "sum(" + kRevenue + ")" ||
      input.joins != std::vector<std::string>{"inner(l_partkey=p_partkey)"} || !input.residuals.empty() ||
      !input.values.empty() || !input.disjunctions.empty() ||
      keysOf(input.ranges) != std::set<std::string>{"l_shipdate"} || aggregation.outputs.size() != 1) {
    return std::nullopt;
  }
  auto like = matchPattern(aggregation.aggregates[1], "sum(if(like(p_type,?)," + kRevenue + ",0))");
  if (!like ||
      aggregation.outputs[0] != "divide(multiply(100," + aggregation.aggregates[1] + ")," + aggregation.aggregates[0] + ")") {
    return std::nullopt;
  }
  const std::string pattern = (*like)[0];
  auto promo = codesWhere(store, "part", "p_type", [&](const std::string& value) {
    return likeMatches(value.c_str(), pattern.c_str());
  });
  if (!promo) {
    return std::nullopt;
  }
  const Range& shipdate = input.ranges.at("l_shipdate");
  QueryParams params;
  params.values["shipdate_min"] = storedMin(shipdate, 0, -kUnbounded);
  params.values["shipdate_max"] = storedMax(shipdate, 0, kUnbounded);
  putCodes(params, "promo_types", *promo);
  return params;
}

std::optional<QueryParams> matchQ19(const Aggregation& aggregation, ColumnStore& store) {
  const Relation& input = aggregation.input;
  if (input.tables != std::set<std::string>{"lineitem", "part"} || !aggregation.keys.empty() ||
      aggregation.aggregates != std::vector<std::string>{"sum(multiply(l_extendedprice,minus(1,l_discount)))"} ||
      aggregation.outputs != aggregation.aggregates ||
      input.joins != std::vector<std::string>{"inner(l_partkey=p_partkey)"} || !input.residuals.empty() ||
      !input.ranges.empty() || keysOf(input.values) != std::set<std::string>{"l_shipinstruct", "l_shipmode"} ||
      input.disjunctions.size() != 1 || input.disjunctions[0].size() != 3) {
    return std::nullopt;
  }
  auto instruct = onlyValue(input, "l_shipinstruct");
  auto modes = codesOf(store, "lineitem", "l_shipmode", input.values.at("l_shipmode"));
  if (!instruct || !modes) {
    return std::nullopt;
  }
  const int32_t quantityScale = store.metadata("lineitem", "l_quantity").scale;
  QueryParams params;
  params.values["shipinstruct"] = store.metadata("lineitem", "l_shipinstruct").code(*instruct);
  putCodes(params, "shipmodes", *modes);
  for (size_t b = 0; b < 3; ++b) {
    const Relation& branch = input.disjunctions[0][b];
    const Range* quantity = rangeIn(branch, "l_quantity");
    const Range* size = rangeIn(branch, "p_size");
    auto brand = onlyValue(branch, "p_brand");
    if (keysOf(branch.ranges) != std::set<std::string>{"l_quantity", "p_size"} ||
        keysOf(branch.values) != std::set<std::string>{"p_brand", "p_container"} || !brand) {
      return std::nullopt;
    }
    auto containers = codesOf(store, "part", "p_container", branch.values.at("p_container"));
    if (!containers) {
      return std::nullopt;
    }
    const std::string n = std::to_string(b);
    params.values["brand" + n] = store.metadata("part", "p_brand").code(*brand);
    putCodes(params, "containers" + n, *containers);
    params.values["quantity_min" + n] = storedMin(*quantity, quantityScale, -kUnbounded);
    params.values["quantity_max" + n] = storedMax(*quantity, quantityScale, kUnbounded);
    params.values["size_min" + n] = storedMin(*size, 0, -kUnbounded);
    params.values["size_max" + n] = storedMax(*size, 0, kUnbounded);
  }
  return params;
}

struct QueryMatcher {
  int32_t query;
  std::optional<QueryParams> (*match)(const Aggregation&, ColumnStore&);
};

const QueryMatcher kMatchers[] = {
    {1, matchQ1}, {4, matchQ4}, {6, matchQ6}, {12, matchQ12}, {14, matchQ14}, {19, matchQ19}};

/// `node` with its sources replaced, for the node types a fused query is
/// placed under; null for any other.
core::PlanNodePtr withSources(const core::PlanNodePtr& node, const std::vector<core::PlanNodePtr>& sources) {
  if (const auto* orderBy = dynamic_cast<const core::OrderByNode*>(node.get())) {
    return std::make_shared<core::OrderByNode>(
        orderBy->id(), orderBy->sortingKeys(), orderBy->sortingOrders(), orderBy->isPartial(), sources[0]);
  }
  if (const auto* topN = dynamic_cast<const core::TopNNode*>(node.get())) {
    return std::make_shared<core::TopNNode>(
        topN->id(), topN->sortingKeys(), topN->sortingOrders(), topN->count(), topN->isPartial(), sources[0]);
  }
  if (const auto* limit = dynamic_cast<const core::LimitNode*>(node.get())) {
    return std::make_shared<core::LimitNode>(
        limit->id(), limit->offset(), limit->count(), limit->isPartial(), sources[0]);
  }
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    return std::make_shared<core::ProjectNode>(project->id(), project->names(), project->projections(), sources[0]);
  }
  if (const auto* filter = dynamic_cast<const core::FilterNode*>(node.get())) {
    return std::make_shared<core::FilterNode>(filter->id(), filter->filter(), sources[0]);
  }
  return nullptr;
}

std::string describeFilters(const Relation& input, const std::string& indent) {
  std::string text;
  for (const auto& [column, range] : input.ranges) {
    text += fmt::format(
        "{}range: {} {}{}, {}{}\n",
        indent,
        column,
        range.minExclusive ? "(" : "[",
        range.hasMin() ? fmt::format("{}", static_cast<double>(range.min)) : "-inf",
        range.hasMax() ? fmt::format("{}", static_cast<double>(range.max)) : "inf",
        range.maxExclusive ? ")" : "]");
  }
  for (const auto& [column, values] : input.values) {
    text += fmt::format("{}values: {} {{{}}}\n", indent, column, fmt::join(values, ", "));
  }
  for (const auto& residual : input.residuals) {
    text += fmt::format("{}residual: {}\n", indent, residual);
  }
  for (const auto& branches : input.disjunctions) {
    text += indent + "or:\n";
    for (const auto& branch : branches) {
      text += indent + "  branch:\n" + describeFilters(branch, indent + "    ");
    }
  }
  return text;
}

std::string describe(const Aggregation& aggregation) {
  const Relation& input = aggregation.input;
  return fmt::format(
             "tables: {}\nkeys: {}\naggregates: {}\noutputs: {}\njoins: {}\n",
             fmt::join(input.tables, " "),
             fmt::join(aggregation.keys, " | "),
             fmt::join(aggregation.aggregates, " | "),
             fmt::join(aggregation.outputs, " | "),
             fmt::join(input.joins, " ")) +
      describeFilters(input, "");
}

std::optional<std::pair<int32_t, QueryParams>> match(const Aggregation& aggregation, ColumnStore& store) {
  for (const auto& matcher : kMatchers) {
    if (auto params = matcher.match(aggregation, store)) {
      if (std::getenv("SV_EXPLAIN_MATCH")) {
        LOG(INFO) << "q" << matcher.query << " params: " << fmt::format("{}", fmt::join(params->values, " "));
      }
      return std::make_pair(matcher.query, std::move(*params));
    }
  }
  if (std::getenv("SV_EXPLAIN_MATCH")) {
    LOG(INFO) << "Unmatched aggregation:\n" << describe(aggregation);
  }
  return std::nullopt;
}

core::PlanNodePtr rewrite(const core::PlanNodePtr& node, ColumnStore& store, std::vector<std::string>* log) {
  std::optional<Aggregation> aggregation;
  if (const auto* project = dynamic_cast<const core::ProjectNode*>(node.get())) {
    aggregation = readAggregation(skipLocalExchanges(project->sources()[0].get()), project);
  }
  if (!aggregation) {
    aggregation = readAggregation(node.get());
  }
  if (aggregation) {
    if (auto matched = match(*aggregation, store)) {
      if (log) {
        log->push_back(fmt::format("plan node {} ({}) becomes fused q{}", node->id(), node->name(), matched->first));
      }
      return std::make_shared<FusedQueryNode>(node->id(), matched->first, std::move(matched->second), node->outputType());
    }
  }
  std::vector<core::PlanNodePtr> sources;
  bool changed = false;
  for (const auto& source : node->sources()) {
    sources.push_back(rewrite(source, store, log));
    changed |= sources.back() != source;
  }
  if (!changed) {
    return node;
  }
  if (auto rebuilt = withSources(node, sources)) {
    return rebuilt;
  }
  if (log) {
    log->push_back(
        fmt::format("plan node {} ({}) cannot take rewritten inputs, so it is left as it was", node->id(), node->name()));
  }
  return node;
}

} // namespace

core::PlanNodePtr
rewriteWithFusedQueries(const core::PlanNodePtr& plan, ColumnStore& store, std::vector<std::string>* log) {
  return rewrite(plan, store, log);
}

} // namespace facebook::velox::supervelox
