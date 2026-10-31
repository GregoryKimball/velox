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

#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/experimental/supervelox/exec/FusedQuery.h"

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

/// What a chain of projections and filters over a table scan computes: each
/// output column as a canonical expression over table columns, and the
/// filters as ranges of table columns.
struct Relation {
  std::string table;
  std::map<std::string, std::string> columns;
  std::map<std::string, Range> ranges;
};

/// An aggregation over a relation, its partial and final steps taken as one
/// and the exchange between them skipped.
struct Aggregation {
  std::vector<std::string> keys;
  std::vector<std::string> aggregates;
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

/// Adds the ranges `expr` puts on table columns. False when some part of it
/// is not a comparison of a column with a literal.
bool addConjuncts(const core::TypedExprPtr& expr, Relation& relation) {
  const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get());
  if (!call) {
    return false;
  }
  const std::string name = functionName(call->name());
  const auto& inputs = call->inputs();
  if (name == "and") {
    for (const auto& input : inputs) {
      if (!addConjuncts(input, relation)) {
        return false;
      }
    }
    return true;
  }
  if (name == "between" && inputs.size() == 3) {
    auto column = tableColumn(inputs[0], relation);
    auto low = literalValue(inputs[1]);
    auto high = literalValue(inputs[2]);
    if (!column || !low || !high) {
      return false;
    }
    relation.ranges[*column].atLeast(*low, false);
    relation.ranges[*column].atMost(*high, false);
    return true;
  }
  if (inputs.size() != 2) {
    return false;
  }
  auto column = tableColumn(inputs[0], relation);
  auto value = literalValue(inputs[1]);
  std::string op = name;
  if (!column) {
    column = tableColumn(inputs[1], relation);
    value = literalValue(inputs[0]);
    static const std::map<std::string, std::string> kFlipped = {
        {"lt", "gt"}, {"lte", "gte"}, {"gt", "lt"}, {"gte", "lte"}, {"eq", "eq"}};
    auto it = kFlipped.find(name);
    if (it == kFlipped.end()) {
      return false;
    }
    op = it->second;
  }
  if (!column || !value) {
    return false;
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

/// Adds the ranges a connector's pushed-down filters put on table columns.
bool addSubfieldFilters(const connector::hive::FileTableHandle& handle, const RowTypePtr& types, Relation& relation) {
  for (const auto& [subfield, filter] : handle.subfieldFilters()) {
    const std::string column = subfield.toString();
    const auto index = types->getChildIdxIfExists(column);
    if (!index) {
      return false;
    }
    const auto& type = types->childAt(*index);
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

std::optional<Relation> readRelation(const core::PlanNode* node) {
  node = skipLocalExchanges(node);
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(node)) {
    const auto* file = dynamic_cast<const connector::hive::FileTableHandle*>(scan->tableHandle().get());
    if (!file) {
      return std::nullopt;
    }
    const auto& handle = *file;
    Relation relation;
    relation.table = handle.name();
    for (const auto& [name, column] : scan->assignments()) {
      relation.columns[name] = column->name();
    }
    const RowTypePtr types = handle.dataColumns() ? handle.dataColumns() : scan->outputType();
    if (!addSubfieldFilters(handle, types, relation)) {
      return std::nullopt;
    }
    if (const auto& remaining = handle.remainingFilter(); remaining && !addConjuncts(remaining, relation)) {
      return std::nullopt;
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
    if (!relation || !addConjuncts(filter->filter(), *relation)) {
      return std::nullopt;
    }
    return relation;
  }
  return std::nullopt;
}

std::optional<Aggregation> readAggregation(const core::PlanNode* node) {
  const auto* top = dynamic_cast<const core::AggregationNode*>(node);
  if (!top || !top->globalGroupingSets().empty() || top->groupId()) {
    return std::nullopt;
  }
  const core::AggregationNode* raw = top;
  if (top->step() == core::AggregationNode::Step::kFinal) {
    raw = dynamic_cast<const core::AggregationNode*>(skipLocalExchanges(top->sources()[0].get()));
    if (!raw || raw->step() != core::AggregationNode::Step::kPartial) {
      return std::nullopt;
    }
  } else if (top->step() != core::AggregationNode::Step::kSingle) {
    return std::nullopt;
  }
  auto input = readRelation(raw->sources()[0].get());
  if (!input) {
    return std::nullopt;
  }
  Aggregation aggregation;
  for (const auto& key : raw->groupingKeys()) {
    aggregation.keys.push_back(canonical(key, input->columns));
  }
  for (const auto& aggregate : raw->aggregates()) {
    if (aggregate.mask || aggregate.distinct || !aggregate.sortingKeys.empty()) {
      return std::nullopt;
    }
    aggregation.aggregates.push_back(canonical(aggregate.call, input->columns));
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
  if (aggregation.input.table != "lineitem" || aggregation.keys != kKeys || aggregation.aggregates != kAggregates ||
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
  if (aggregation.input.table != "lineitem" || !aggregation.keys.empty() || aggregation.aggregates != kAggregates ||
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

struct QueryMatcher {
  int32_t query;
  std::optional<QueryParams> (*match)(const Aggregation&, ColumnStore&);
};

const QueryMatcher kMatchers[] = {{1, matchQ1}, {6, matchQ6}};

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

core::PlanNodePtr rewrite(const core::PlanNodePtr& node, ColumnStore& store, std::vector<std::string>* log) {
  if (auto aggregation = readAggregation(node.get())) {
    for (const auto& matcher : kMatchers) {
      if (auto params = matcher.match(*aggregation, store)) {
        if (log) {
          log->push_back(fmt::format("plan node {} ({}) becomes fused q{}", node->id(), node->name(), matcher.query));
        }
        return std::make_shared<FusedQueryNode>(node->id(), matcher.query, std::move(*params), node->outputType());
      }
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
