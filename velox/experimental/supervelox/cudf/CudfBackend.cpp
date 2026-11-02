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
#include "velox/experimental/supervelox/cudf/CudfBackend.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>

#include "velox/experimental/supervelox/common/Cuda.h"

namespace facebook::velox::supervelox::cudf_backend {

Result q1(QueryContext& ctx);
Result q2(QueryContext& ctx);
Result q3(QueryContext& ctx);
Result q4(QueryContext& ctx);
Result q5(QueryContext& ctx);
Result q6(QueryContext& ctx);
Result q7(QueryContext& ctx);
Result q8(QueryContext& ctx);
Result q9(QueryContext& ctx);
Result q10(QueryContext& ctx);
Result q11(QueryContext& ctx);
Result q12(QueryContext& ctx);
Result q13(QueryContext& ctx);
Result q14(QueryContext& ctx);
Result q15(QueryContext& ctx);
Result q16(QueryContext& ctx);
Result q17(QueryContext& ctx);
Result q18(QueryContext& ctx);
Result q19(QueryContext& ctx);
Result q20(QueryContext& ctx);
Result q21(QueryContext& ctx);
Result q22(QueryContext& ctx);

namespace {

cudf::type_id typeId(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return cudf::type_id::INT8;
    case DType::kInt16:
      return cudf::type_id::INT16;
    case DType::kInt32:
      return cudf::type_id::INT32;
    case DType::kInt64:
      return cudf::type_id::INT64;
  }
  throw std::logic_error("unknown dtype");
}

const void* storedColumn(QueryContext& ctx, const std::string& table, const std::string& name, DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return ctx.col<int8_t>(table, name);
    case DType::kInt16:
      return ctx.col<int16_t>(table, name);
    case DType::kInt32:
      return ctx.col<int32_t>(table, name);
    case DType::kInt64:
      return ctx.col<int64_t>(table, name);
  }
  throw std::logic_error("unknown dtype");
}

/// Host copies of zone maps, by device address and zone count.
const std::vector<int2>& hostZones(QueryContext& ctx, const std::string& table, const std::string& name) {
  static std::mutex mutex;
  static std::map<std::pair<const int2*, int64_t>, std::vector<int2>> copies;
  const int2* device = ctx.zones(table, name);
  const int64_t count = (ctx.rows(table) + QueryContext::kZoneRows - 1) / QueryContext::kZoneRows;
  std::lock_guard<std::mutex> lock(mutex);
  auto& copy = copies[{device, count}];
  if (copy.empty() && count > 0) {
    copy = ctx.download(device, count);
  }
  return copy;
}

bool jitEnabled = true;

/// Declarations of `arrays` in generated code, read from `user`, a device
/// copy of `pointers`, to which their addresses are appended.
std::string
declareArrays(QueryContext& ctx, const std::vector<ArrayInput>& arrays, std::vector<const void*>& pointers);

/// A device copy of `pointers`, for transform_lto's or reduce_lto's user data.
rmm::device_buffer devicePointers(QueryContext& ctx, const std::vector<const void*>& pointers) {
  rmm::device_buffer buffer(std::max<size_t>(1, pointers.size()) * sizeof(void*), ctx.stream());
  if (!pointers.empty()) {
    SV_CUDA(cudaMemcpyAsync(
        buffer.data(), pointers.data(), pointers.size() * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream()));
  }
  return buffer;
}

/// The C subset of the queries' device expressions as a cuDF AST, which
/// compute_column evaluates with its precompiled interpreter. Integers are
/// evaluated as int64, since AST operands must share a type; `c ? a : b`,
/// which the AST lacks, is int64(c) * a + int64(!c) * b, and the bit test
/// `(maskU >> x) & 1` is x equal to one of the bits of the mask.
class AstExpression {
 public:
  /// `inputs` declares one parameter per column of `table`; `prelude` holds
  /// `const <type> <name> = <expression>;` definitions. The literals are
  /// made and freed on `stream`, which evaluates the expression.
  AstExpression(
      const cudf::table_view& table,
      const std::string& inputs,
      const std::string& prelude,
      cudaStream_t stream)
      : stream_(stream) {
    std::stringstream declared(inputs);
    std::string declaration;
    cudf::size_type index = 0;
    while (std::getline(declared, declaration, ',')) {
      const std::string name = declaration.substr(declaration.find_last_of(" \t") + 1);
      const auto& reference = tree_.push(cudf::ast::column_reference(index));
      if (table.column(index).type().id() == cudf::type_id::BOOL8) {
        names_[name] = Value{&reference, true};
      } else if (table.column(index).type().id() == cudf::type_id::INT64) {
        names_[name] = Value{&reference, false};
      } else {
        names_[name] = Value{&operation(cudf::ast::ast_operator::CAST_TO_INT64, reference), false};
      }
      ++index;
    }
    std::stringstream statements(prelude);
    std::string statement;
    while (std::getline(statements, statement, ';')) {
      const size_t equals = statement.find('=');
      if (equals == std::string::npos) {
        continue;
      }
      const bool adds = equals > 0 && statement[equals - 1] == '+';
      std::string left = statement.substr(0, adds ? equals - 1 : equals);
      left = left.substr(0, left.find_last_not_of(" \t\n") + 1);
      const std::string name = left.substr(left.find_last_of(" \t\n") + 1);
      Value value = parse(statement.substr(equals + 1));
      if (adds) {
        const Value sum = names_.at(name);
        value = sum.constant && value.constant ? Value{.constant = *sum.constant + *value.constant}
                                               : arithmetic(cudf::ast::ast_operator::ADD, sum, value);
      }
      names_[name] = value;
    }
  }

  /// `expression` as a column of `type`, an integer type or BOOL8.
  std::unique_ptr<cudf::column>
  compute(QueryContext& ctx, const cudf::table_view& table, const std::string& expression, cudf::type_id type) {
    const auto stream = ctx.stream();
    const Value value = parse(expression);
    const bool wantBool = type == cudf::type_id::BOOL8;
    if (value.constant) {
      if (wantBool) {
        const cudf::numeric_scalar<bool> fill(*value.constant != 0, true, stream);
        return cudf::make_column_from_scalar(fill, table.num_rows(), stream);
      }
      const cudf::numeric_scalar<int64_t> fill(*value.constant, true, stream);
      auto column = cudf::make_column_from_scalar(fill, table.num_rows(), stream);
      return type == cudf::type_id::INT64 ? std::move(column)
                                          : cudf::cast(column->view(), cudf::data_type{type}, stream);
    }
    const auto& root =
        operation(cudf::ast::ast_operator::IDENTITY, wantBool ? boolean(value) : integer(value));
    auto column = cudf::compute_column(table, root, stream);
    return column->type().id() == type ? std::move(column)
                                       : cudf::cast(column->view(), cudf::data_type{type}, stream);
  }

 private:
  /// An int64 or boolean expression, or a constant not yet made a literal.
  /// `mask` is set for `maskU >> x`, whose x is `expression`.
  struct Value {
    const cudf::ast::expression* expression{nullptr};
    bool isBool{false};
    std::optional<int64_t> constant;
    std::optional<uint64_t> mask;
  };

  const cudf::ast::expression& operation(cudf::ast::ast_operator op, const cudf::ast::expression& input) {
    return tree_.push(cudf::ast::operation(op, input));
  }

  const cudf::ast::expression&
  operation(cudf::ast::ast_operator op, const cudf::ast::expression& left, const cudf::ast::expression& right) {
    return tree_.push(cudf::ast::operation(op, left, right));
  }

  const cudf::ast::expression& integer(const Value& value) {
    if (value.mask) {
      throw std::invalid_argument("a shifted mask is only tested with & 1");
    }
    if (value.constant) {
      integers_.emplace_back(*value.constant, true, stream_);
      return tree_.push(cudf::ast::literal(integers_.back()));
    }
    return value.isBool ? operation(cudf::ast::ast_operator::CAST_TO_INT64, *value.expression) : *value.expression;
  }

  const cudf::ast::expression& boolean(const Value& value) {
    if (value.mask) {
      throw std::invalid_argument("a shifted mask is only tested with & 1");
    }
    if (value.constant) {
      booleans_.emplace_back(*value.constant != 0, true, stream_);
      return tree_.push(cudf::ast::literal(booleans_.back()));
    }
    if (value.isBool) {
      return *value.expression;
    }
    return operation(cudf::ast::ast_operator::NOT_EQUAL, *value.expression, integer(Value{.constant = 0}));
  }

  Value parse(const std::string& text) {
    text_ = text;
    at_ = 0;
    Value value = ternary();
    skipSpace();
    if (at_ != text_.size()) {
      throw std::invalid_argument("cannot translate to a cuDF AST: " + text_);
    }
    return value;
  }

  void skipSpace() {
    while (at_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[at_]))) {
      ++at_;
    }
  }

  bool accept(const char* token) {
    skipSpace();
    const size_t length = std::strlen(token);
    if (text_.compare(at_, length, token) != 0) {
      return false;
    }
    // Keeps & from matching &&, > from >> and >=, and so on.
    const char next = at_ + length < text_.size() ? text_[at_ + length] : '\0';
    if (length == 1 && ((next == token[0] && std::strchr("&|<>=", next)) || (next == '=' && std::strchr("<>!=", token[0])))) {
      return false;
    }
    if (length == 2 && token[0] == '>' && token[1] == '>' && next == '=') {
      return false;
    }
    at_ += length;
    return true;
  }

  void expect(const char* token) {
    if (!accept(token)) {
      throw std::invalid_argument(std::string("expected ") + token + " in " + text_);
    }
  }

  Value boolValue(const cudf::ast::expression& expression) {
    return Value{&expression, true};
  }

  Value intValue(const cudf::ast::expression& expression) {
    return Value{&expression, false};
  }

  Value ternary() {
    Value condition = logicalOr();
    if (!accept("?")) {
      return condition;
    }
    Value whenTrue = ternary();
    expect(":");
    Value whenFalse = ternary();
    if (condition.constant) {
      return *condition.constant ? whenTrue : whenFalse;
    }
    const auto& test = boolean(condition);
    if (whenTrue.isBool && whenFalse.isBool) {
      return boolValue(operation(
          cudf::ast::ast_operator::LOGICAL_OR,
          operation(cudf::ast::ast_operator::LOGICAL_AND, test, boolean(whenTrue)),
          operation(
              cudf::ast::ast_operator::LOGICAL_AND,
              operation(cudf::ast::ast_operator::NOT, test),
              boolean(whenFalse))));
    }
    auto chosen = [&](const cudf::ast::expression& selector, const Value& value) -> const cudf::ast::expression* {
      if (value.constant && *value.constant == 0) {
        return nullptr;
      }
      return &operation(
          cudf::ast::ast_operator::MUL,
          operation(cudf::ast::ast_operator::CAST_TO_INT64, selector),
          integer(value));
    };
    const auto* first = chosen(test, whenTrue);
    const auto* second = chosen(operation(cudf::ast::ast_operator::NOT, test), whenFalse);
    if (!first && !second) {
      return Value{.constant = 0};
    }
    if (!first || !second) {
      return intValue(first ? *first : *second);
    }
    return intValue(operation(cudf::ast::ast_operator::ADD, *first, *second));
  }

  Value logicalOr() {
    Value left = logicalAnd();
    while (accept("||")) {
      Value right = logicalAnd();
      left = boolValue(operation(cudf::ast::ast_operator::LOGICAL_OR, boolean(left), boolean(right)));
    }
    return left;
  }

  Value logicalAnd() {
    Value left = bitAnd();
    while (accept("&&")) {
      Value right = bitAnd();
      left = boolValue(operation(cudf::ast::ast_operator::LOGICAL_AND, boolean(left), boolean(right)));
    }
    return left;
  }

  Value bitAnd() {
    Value left = equality();
    while (accept("&")) {
      Value right = equality();
      if (left.mask && right.constant && *right.constant == 1) {
        left = bitTest(*left.mask, *left.expression);
      } else if (left.constant && right.constant) {
        left = Value{.constant = *left.constant & *right.constant};
      } else {
        left = intValue(operation(cudf::ast::ast_operator::BITWISE_AND, integer(left), integer(right)));
      }
    }
    return left;
  }

  Value bitTest(uint64_t mask, const cudf::ast::expression& index) {
    const cudf::ast::expression* test = nullptr;
    for (int64_t bit = 0; bit < 64; ++bit) {
      if (mask >> bit & 1) {
        const auto& equal = operation(cudf::ast::ast_operator::EQUAL, index, integer(Value{.constant = bit}));
        test = test ? &operation(cudf::ast::ast_operator::LOGICAL_OR, *test, equal) : &equal;
      }
    }
    return test ? boolValue(*test) : Value{.constant = 0};
  }

  Value compare(cudf::ast::ast_operator op, const Value& left, const Value& right) {
    return boolValue(operation(op, integer(left), integer(right)));
  }

  Value equality() {
    Value left = relational();
    for (;;) {
      if (accept("==")) {
        left = compare(cudf::ast::ast_operator::EQUAL, left, relational());
      } else if (accept("!=")) {
        left = compare(cudf::ast::ast_operator::NOT_EQUAL, left, relational());
      } else {
        return left;
      }
    }
  }

  Value relational() {
    Value left = shift();
    for (;;) {
      if (accept("<=")) {
        left = compare(cudf::ast::ast_operator::LESS_EQUAL, left, shift());
      } else if (accept(">=")) {
        left = compare(cudf::ast::ast_operator::GREATER_EQUAL, left, shift());
      } else if (accept("<")) {
        left = compare(cudf::ast::ast_operator::LESS, left, shift());
      } else if (accept(">")) {
        left = compare(cudf::ast::ast_operator::GREATER, left, shift());
      } else {
        return left;
      }
    }
  }

  Value shift() {
    Value left = additive();
    while (accept(">>")) {
      Value right = additive();
      if (!left.constant || right.constant) {
        throw std::invalid_argument("only a constant mask shifts by a column: " + text_);
      }
      left = Value{.expression = &integer(right), .mask = static_cast<uint64_t>(*left.constant)};
    }
    return left;
  }

  Value arithmetic(cudf::ast::ast_operator op, const Value& left, const Value& right) {
    return intValue(operation(op, integer(left), integer(right)));
  }

  Value additive() {
    Value left = multiplicative();
    for (;;) {
      if (accept("+")) {
        left = arithmetic(cudf::ast::ast_operator::ADD, left, multiplicative());
      } else if (accept("-")) {
        left = arithmetic(cudf::ast::ast_operator::SUB, left, multiplicative());
      } else {
        return left;
      }
    }
  }

  Value multiplicative() {
    Value left = unary();
    for (;;) {
      if (accept("*")) {
        left = arithmetic(cudf::ast::ast_operator::MUL, left, unary());
      } else if (accept("/")) {
        left = arithmetic(cudf::ast::ast_operator::DIV, left, unary());
      } else if (accept("%")) {
        left = arithmetic(cudf::ast::ast_operator::MOD, left, unary());
      } else {
        return left;
      }
    }
  }

  Value unary() {
    if (accept("!")) {
      Value operand = unary();
      if (operand.constant) {
        return Value{.constant = *operand.constant == 0};
      }
      return boolValue(operation(cudf::ast::ast_operator::NOT, boolean(operand)));
    }
    if (accept("-")) {
      Value operand = unary();
      if (operand.constant) {
        return Value{.constant = -*operand.constant};
      }
      return arithmetic(cudf::ast::ast_operator::SUB, Value{.constant = 0}, operand);
    }
    return primary();
  }

  Value primary() {
    skipSpace();
    if (accept("(")) {
      Value inner = ternary();
      expect(")");
      return inner;
    }
    if (at_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at_]))) {
      size_t length = 0;
      const int64_t number = std::stoll(text_.substr(at_), &length);
      at_ += length;
      while (at_ < text_.size() && std::strchr("uUlL", text_[at_])) {
        ++at_;
      }
      return Value{.constant = number};
    }
    const size_t start = at_;
    while (at_ < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[at_])) || text_[at_] == '_')) {
      ++at_;
    }
    const std::string word = text_.substr(start, at_ - start);
    if (word == "true" || word == "false") {
      return Value{.constant = word == "true"};
    }
    static const std::set<std::string> kTypes = {"int8_t", "int16_t", "int32_t", "int64_t", "int", "uint32_t"};
    if (kTypes.contains(word)) {
      // Every value fits the types the expressions convert to.
      const bool braced = accept("{");
      if (!braced) {
        expect("(");
      }
      Value inner = ternary();
      expect(braced ? "}" : ")");
      return inner;
    }
    const auto it = names_.find(word);
    if (word.empty() || it == names_.end()) {
      throw std::invalid_argument("cannot translate '" + word + "' to a cuDF AST in " + text_);
    }
    return it->second;
  }

  cudaStream_t stream_;
  cudf::ast::tree tree_;
  std::deque<cudf::numeric_scalar<int64_t>> integers_;
  std::deque<cudf::numeric_scalar<bool>> booleans_;
  std::map<std::string, Value> names_;
  std::string text_;
  size_t at_{0};
};

/// Rows the stock libcudf version of sumGroups filters and sums at a time,
/// bounding the value columns it makes.
constexpr int64_t kStockRows = 128 << 20;

/// sumGroups in stock libcudf: per chunk of rows, a boolean column of the
/// filter, the rows that pass, a column per value and of the group, then a
/// reduction or a groupby summing each value.
std::vector<int64_t>
stockSumGroups(QueryContext& ctx, const GroupedSums& sums, const std::vector<cudf::column_view>& columns) {
  const auto stream = ctx.stream();
  const auto values = static_cast<int32_t>(sums.values.size());
  std::vector<int64_t> totals(sums.groups * values, 0);
  const int64_t rows = columns[0].size();
  for (int64_t begin = 0; begin < rows; begin += kStockRows) {
    const std::vector<cudf::size_type> bounds{
        static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(std::min(rows, begin + kStockRows))};
    const auto slice = cudf::slice(cudf::table_view(columns), bounds)[0];
    const std::vector<cudf::column_view> chunk(slice.begin(), slice.end());
    const auto keep = computeColumn(ctx, sums.inputs, sums.filter, cudf::type_id::BOOL8, chunk, sums.prelude);
    const auto kept = cudf::apply_boolean_mask(slice, keep->view(), stream);
    if (kept->num_rows() == 0) {
      continue;
    }
    std::vector<cudf::column_view> passed;
    for (cudf::size_type i = 0; i < kept->num_columns(); ++i) {
      passed.push_back(kept->get_column(i).view());
    }
    std::vector<std::unique_ptr<cudf::column>> computed;
    for (const auto& value : sums.values) {
      computed.push_back(computeColumn(ctx, sums.inputs, value, cudf::type_id::INT64, passed, sums.prelude));
    }
    if (sums.groups == 1) {
      for (int32_t k = 0; k < values; ++k) {
        const auto sum = cudf::reduce(
            computed[k]->view(),
            *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::INT64},
            stream);
        totals[k] += static_cast<const cudf::numeric_scalar<int64_t>&>(*sum).value(stream);
      }
      continue;
    }
    const auto group = computeColumn(ctx, sums.inputs, sums.group, cudf::type_id::INT32, passed, sums.prelude);
    std::vector<cudf::groupby::aggregation_request> requests(values);
    for (int32_t k = 0; k < values; ++k) {
      requests[k].values = computed[k]->view();
      requests[k].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    }
    cudf::groupby::groupby groupby(cudf::table_view({group->view()}));
    auto [keys, results] = groupby.aggregate(requests, stream);
    const auto ids = ctx.download(keys->get_column(0).view().data<int32_t>(), keys->num_rows());
    for (int32_t k = 0; k < values; ++k) {
      const auto sumsOfGroups = ctx.download(results[k].results[0]->view().data<int64_t>(), ids.size());
      for (size_t g = 0; g < ids.size(); ++g) {
        totals[ids[g] * values + k] += sumsOfGroups[g];
      }
    }
  }
  return totals;
}

} // namespace

void setUseJit(bool use) {
  jitEnabled = use;
}

bool useJit() {
  return jitEnabled;
}

std::unique_ptr<cudf::table> transformColumns(
    QueryContext& ctx,
    const std::string& source,
    std::span<const cudf::transform_input> inputs,
    std::span<const cudf::transform_output> outputs) {
  return cudf::transform_lto(
      compileFragment(source).view(),
      cudf::lto_binary_type::LTO_IR,
      cudf::null_aware::NO,
      std::nullopt,
      inputs,
      outputs,
      {},
      std::nullopt,
      ctx.stream());
}

cudf::column_view fixedView(QueryContext& ctx, const std::string& table, const std::string& name, RowRange rows) {
  const Column& column = ctx.meta(table, name);
  if (column.kind == Kind::kString) {
    throw std::invalid_argument(table + "." + name + " is not fixed-width");
  }
  if (rows.end > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("cuDF columns hold at most 2^31 - 1 rows");
  }
  const auto* data = static_cast<const char*>(storedColumn(ctx, table, name, column.dtype));
  return cudf::column_view(
      cudf::data_type{typeId(column.dtype)},
      static_cast<cudf::size_type>(rows.size()),
      data + rows.begin * dtypeSize(column.dtype),
      nullptr,
      0);
}

ZoneSplit
splitZones(QueryContext& ctx, const std::string& table, const std::string& name, int64_t low, int64_t high) {
  const auto& zones = hostZones(ctx, table, name);
  const int64_t rows = ctx.rows(table);
  const int64_t zoneRows = QueryContext::kZoneRows;
  int64_t first = -1;
  int64_t last = -1;
  int64_t runStart = -1;
  int64_t bestStart = 0;
  int64_t bestEnd = 0;
  for (int64_t zone = 0; zone < static_cast<int64_t>(zones.size()); ++zone) {
    const int2 range = zones[zone];
    const bool any = range.y >= low && range.x <= high;
    const bool all = range.x >= low && range.y <= high;
    if (any) {
      first = first < 0 ? zone : first;
      last = zone;
    }
    if (all) {
      runStart = runStart < 0 ? zone : runStart;
      if (zone + 1 - runStart > bestEnd - bestStart) {
        bestStart = runStart;
        bestEnd = zone + 1;
      }
    } else {
      runStart = -1;
    }
  }
  ZoneSplit split;
  if (first < 0) {
    return split;
  }
  auto rowOf = [&](int64_t zone) { return std::min(rows, zone * zoneRows); };
  split.some = {rowOf(first), rowOf(last + 1)};
  split.all = {rowOf(bestStart), rowOf(bestEnd)};
  return split;
}

cudf::column_view deviceView(cudf::type_id type, const void* data, int64_t rows) {
  if (rows > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("cuDF columns hold at most 2^31 - 1 rows");
  }
  return cudf::column_view(cudf::data_type{type}, static_cast<cudf::size_type>(rows), data, nullptr, 0);
}

std::string codeTest(const std::string& code, const CodeSet& codes) {
  std::string test;
  for (int32_t c = 0; c < 256; ++c) {
    if (codes.contains(c)) {
      test += (test.empty() ? "" : " || ") + code + " == " + std::to_string(c);
    }
  }
  return test.empty() ? "false" : "(" + test + ")";
}

std::vector<int64_t>
sumGroups(QueryContext& ctx, const GroupedSums& sums, const std::vector<cudf::column_view>& columns) {
  const int32_t fields = sums.groups * static_cast<int32_t>(sums.values.size());
  auto list = [&](const std::string& type, const std::string& prefix) {
    std::string text;
    for (int32_t i = 0; i < fields; ++i) {
      text += (i ? ", " : "") + type + " " + prefix + std::to_string(i);
    }
    return text;
  };
  std::string source = R"(
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cuda/std/cstdint>

using cudf::errc;
using cudf::size_type;
using namespace cuda::std;

extern "C" {
)";
  // Arrays come last among the inputs, through the user data.
  std::vector<const void*> pointers;
  const std::string inputs = sums.arrays.empty() ? sums.inputs : sums.inputs + ", void* user";
  const std::string arrays = sums.arrays.empty() ? "" : declareArrays(ctx, sums.arrays, pointers);
  source += "__device__ errc sums_filter(size_type row, bool* keep, " + inputs + ") {\n" +
      (sums.filter == "true" ? "" : arrays + sums.prelude) + "  *keep = " + sums.filter +
      ";\n  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_init(" + list("int64_t*", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *s" + std::to_string(i) + " = 0;\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_update(size_type row, " + list("int64_t*", "s") + ", " + inputs + ") {\n" + arrays +
      sums.prelude + "  const int group = " + sums.group + ";\n";
  for (size_t k = 0; k < sums.values.size(); ++k) {
    source += "  const int64_t v" + std::to_string(k) + " = " + sums.values[k] + ";\n";
  }
  for (int32_t g = 0; g < sums.groups; ++g) {
    source += sums.groups == 1 ? "  {\n" : "  if (group == " + std::to_string(g) + ") {\n";
    for (size_t k = 0; k < sums.values.size(); ++k) {
      source += "    *s" + std::to_string(g * sums.values.size() + k) + " += v" + std::to_string(k) + ";\n";
    }
    source += "  }\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_merge(" + list("int64_t*", "d") + ", " + list("int64_t", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *d" + std::to_string(i) + " += s" + std::to_string(i) + ";\n";
  }
  source += "  return errc::SUCCESS;\n}\n";
  source += "__device__ errc sums_finalize(" + list("int64_t*", "o") + ", " + list("int64_t", "s") + ") {\n";
  for (int32_t i = 0; i < fields; ++i) {
    source += "  *o" + std::to_string(i) + " = s" + std::to_string(i) + ";\n";
  }
  source += "  return errc::SUCCESS;\n}\n}\n";

  std::vector<int64_t> totals(fields, 0);
  if (columns.empty() || columns[0].size() == 0) {
    return totals;
  }
  if (!useJit()) {
    if (!sums.arrays.empty()) {
      throw std::invalid_argument("sumGroups reads arrays only with JIT");
    }
    return stockSumGroups(ctx, sums, columns);
  }
  const auto& fragment = compileFragment(source);
  rmm::device_buffer user = devicePointers(ctx, pointers);
  std::vector<cudf::reduce_input> reduceInputs(columns.begin(), columns.end());
  if (!sums.arrays.empty()) {
    reduceInputs.push_back(cudf::user_data{user.data()});
  }
  const cudf::lto_fragment fragments[] = {{cudf::lto_binary_type::LTO_IR, fragment.view()}};
  const cudf::reduce_udf udf{fragments, "sums_init", "sums_update", "sums_merge", "sums_finalize", "sums_filter"};
  const std::vector<cudf::data_type> types(fields, cudf::data_type{cudf::type_id::INT64});
  auto results = cudf::reduce_lto(udf, cudf::null_aware::NO, reduceInputs, types, types, ctx.stream());
  for (int32_t i = 0; i < fields; ++i) {
    SV_CUDA(cudaMemcpyAsync(
        &totals[i], results[i]->view().head(), sizeof(int64_t), cudaMemcpyDeviceToHost, ctx.stream()));
  }
  SV_CUDA(cudaStreamSynchronize(ctx.stream()));
  return totals;
}

const cudf::compiled_fragment& compileFragment(const std::string& source) {
  static std::mutex mutex;
  static std::map<std::string, cudf::compiled_fragment> fragments;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = fragments.find(source);
  if (it == fragments.end()) {
    it = fragments.emplace(source, cudf::compile_cuda_to_lto_ir(source.c_str(), true, {}, {}, {})).first;
  }
  return it->second;
}

const char* deviceType(cudf::type_id type) {
  switch (type) {
    case cudf::type_id::BOOL8:
      return "bool";
    case cudf::type_id::INT8:
      return "int8_t";
    case cudf::type_id::INT16:
      return "int16_t";
    case cudf::type_id::INT32:
      return "int32_t";
    case cudf::type_id::INT64:
      return "int64_t";
    case cudf::type_id::UINT32:
      return "uint32_t";
    case cudf::type_id::FLOAT64:
      return "double";
    default:
      throw std::invalid_argument("no device type for this cuDF type");
  }
}

std::unique_ptr<cudf::column> computeColumn(
    QueryContext& ctx,
    const std::string& inputs,
    const std::string& expression,
    cudf::type_id type,
    const std::vector<cudf::column_view>& columns,
    const std::string& prelude) {
  if (!useJit()) {
    const cudf::table_view table(columns);
    return AstExpression(table, inputs, prelude, ctx.stream()).compute(ctx, table, expression, type);
  }
  const std::string source = std::string(R"(
#include <cuda/std/cstdint>

using namespace cuda::std;

extern "C" __device__ int transform()") +
      deviceType(type) + "* out, " + inputs + ") {\n" + prelude + "  *out = " + expression + ";\n  return 0;\n}\n";
  const std::vector<cudf::transform_input> in(columns.begin(), columns.end());
  const cudf::transform_output out[] = {{cudf::data_type{type}, cudf::output_nullability::ALL_VALID}};
  return std::move(transformColumns(ctx, source, in, out)->release()[0]);
}

std::unique_ptr<cudf::column> keyRows(QueryContext& ctx, const cudf::column_view& keys) {
  const cudf::numeric_scalar<int32_t> one(1, true, ctx.stream());
  return cudf::binary_operation(keys, one, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::INT32}, ctx.stream());
}

std::unique_ptr<cudf::table> gatherColumns(
    QueryContext& ctx,
    const std::string& table,
    const std::vector<std::string>& names,
    const cudf::column_view& rows) {
  std::vector<cudf::column_view> columns;
  for (const auto& name : names) {
    columns.push_back(fixedView(ctx, table, name, {0, ctx.rows(table)}));
  }
  return cudf::gather(cudf::table_view(columns), rows, cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
}

OrderLines orderLines(QueryContext& ctx, int64_t begin, int64_t end) {
  OrderLines lines;
  lines.begin = begin;
  lines.end = end;
  if (end <= begin) {
    return lines;
  }
  const OrderBatch batch = orderBatch(ctx, begin, end);
  lines.lines = batch.lines;
  const auto size = static_cast<cudf::size_type>(end - begin);
  lines.counts = cudf::binary_operation(
      cudf::slice(batch.starts, {1, size + 1})[0],
      cudf::slice(batch.starts, {0, size})[0],
      cudf::binary_operator::SUB,
      cudf::data_type{cudf::type_id::INT32},
      ctx.stream());
  const cudf::numeric_scalar<int64_t> base(lines.lines.begin - batch.base, true, ctx.stream());
  lines.offsets = cudf::binary_operation(
      batch.starts, base, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::INT32}, ctx.stream());
  lines.groups = std::make_unique<cudf::groupby::groupby>(
      cudf::table_view({fixedView(ctx, "lineitem", "l_orderkey", lines.lines)}),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES);
  return lines;
}

namespace {

std::string
declareArrays(QueryContext& ctx, const std::vector<ArrayInput>& arrays, std::vector<const void*>& pointers) {
  std::string source = "  const void* const* pointers = static_cast<const void* const*>(user);\n";
  for (const auto& array : arrays) {
    cudf::type_id type = array.view.type().id();
    if (array.table.empty()) {
      pointers.push_back(array.view.head());
    } else {
      const Column& column = ctx.meta(array.table, array.column);
      pointers.push_back(
          static_cast<const char*>(storedColumn(ctx, array.table, array.column, column.dtype)) +
          array.first * dtypeSize(column.dtype));
      type = typeId(column.dtype);
    }
    source += fmt::format(
        "  const {0}* {1} = static_cast<const {0}*>(pointers[{2}]);\n", deviceType(type), array.name, pointers.size() - 1);
  }
  return source;
}

} // namespace

std::unique_ptr<cudf::table> transformRows(
    QueryContext& ctx,
    int64_t rows,
    const std::string& inputs,
    const std::vector<cudf::column_view>& columns,
    const std::vector<ArrayInput>& arrays,
    const std::vector<std::pair<cudf::type_id, std::string>>& outputs,
    const std::string& body) {
  std::vector<const void*> pointers;
  std::string source = R"(
#include <cuda/std/cstdint>

using namespace cuda::std;

extern "C" __device__ int transform(void* user, int32_t row)";
  for (const auto& [type, name] : outputs) {
    source += std::string(", ") + deviceType(type) + "* " + name;
  }
  source += (inputs.empty() ? "" : ", " + inputs) + ") {\n";
  source += declareArrays(ctx, arrays, pointers);
  source += body + "\n  return 0;\n}\n";
  rmm::device_buffer user = devicePointers(ctx, pointers);
  const std::vector<cudf::transform_input> in(columns.begin(), columns.end());
  std::vector<cudf::transform_output> out;
  for (const auto& output : outputs) {
    out.push_back({cudf::data_type{output.first}, cudf::output_nullability::ALL_VALID});
  }
  return cudf::transform_lto(
      compileFragment(source).view(),
      cudf::lto_binary_type::LTO_IR,
      cudf::null_aware::NO,
      user.data(),
      in,
      out,
      {},
      static_cast<cudf::size_type>(rows),
      ctx.stream());
}

std::unique_ptr<cudf::table> perOrderColumns(
    QueryContext& ctx,
    int64_t begin,
    int64_t end,
    const std::string& orderInputs,
    const std::vector<cudf::column_view>& orderColumns,
    const std::vector<ArrayInput>& arrays,
    const std::vector<std::pair<cudf::type_id, std::string>>& outputs,
    const std::string& init,
    const std::string& line,
    const std::string& finish) {
  const OrderBatch batch = orderBatch(ctx, begin, end);
  return transformRows(
      ctx,
      end - begin,
      orderInputs,
      orderColumns,
      withLines(batch, arrays),
      outputs,
      init + "\n  for (int64_t l = starts[row]; l < starts[row + 1]; ++l) {\n" + line + "\n  }\n" + finish);
}

namespace {

/// linesOfParts over one batch, with the parts as bitmask words.
std::unique_ptr<cudf::table> linesOfPartsIn(
    QueryContext& ctx,
    const cudf::column_view& wanted,
    RowRange range,
    const std::vector<ArrayInput>& arrays,
    const std::string& condition,
    const std::vector<std::string>& columns) {
  // Words cover 32 lines from a multiple of 32, so that whole words inside
  // the range read the part keys 16 bytes at a time.
  // Lineitem arrays are read from `base`, so that batches of the same size
  // generate the same source.
  const int64_t base = range.begin / 32 * 32;
  const int64_t words = (range.end - base + 31) / 32;
  std::vector<ArrayInput> bitArrays = arrays;
  bitArrays.push_back({"lineitem", "l_partkey", "partkey"});
  for (auto& array : bitArrays) {
    if (array.table == "lineitem") {
      array.first = base;
    }
  }
  bitArrays.push_back({.name = "wanted", .view = wanted});
  const bool vector = typeId(ctx.meta("lineitem", "l_partkey").dtype) == cudf::type_id::INT32;
  auto bits = transformRows(
      ctx,
      words,
      "",
      {},
      bitArrays,
      {{cudf::type_id::UINT32, "word"}, {cudf::type_id::INT32, "count"}},
      fmt::format(
          "  uint32_t bits = 0;\n"
          "  const int64_t first = {0} + int64_t{{row}} * 32;\n"
          "  auto test = [&](int64_t key, int k) {{\n"
          "    const int64_t part = key - 1;\n"
          "    const int64_t l = first + k;\n"
          "    if (((wanted[part >> 5] >> (part & 31)) & 1u) && ({3})) bits |= 1u << k;\n"
          "  }};\n"
          "  if ({4} && first >= {1} && first + 32 <= {2}) {{\n"
          "    const int4* keys = reinterpret_cast<const int4*>(partkey + first);\n"
          "#pragma unroll\n"
          "    for (int v = 0; v < 8; ++v) {{\n"
          "      const int4 four = keys[v];\n"
          "      test(four.x, 4 * v); test(four.y, 4 * v + 1); test(four.z, 4 * v + 2); test(four.w, 4 * v + 3);\n"
          "    }}\n"
          "  }} else {{\n"
          "    for (int k = 0; k < 32; ++k) {{\n"
          "      if (first + k >= {1} && first + k < {2}) test(partkey[first + k], k);\n"
          "    }}\n"
          "  }}\n"
          "  *word = bits;\n"
          "  *count = __popc(bits);",
          0,
          range.begin - base,
          range.end - base,
          condition.empty() ? "true" : condition,
          vector ? "true" : "false"));
  auto ends = cudf::scan(
      bits->get_column(1).view(),
      *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
      cudf::scan_type::INCLUSIVE,
      cudf::null_policy::EXCLUDE,
      ctx.stream());
  const int64_t total =
      words == 0 ? 0 : ctx.download(static_cast<const int32_t*>(ends->view().head()) + words - 1, 1)[0];
  std::vector<ArrayInput> rowArrays = {
      {.name = "words", .view = bits->get_column(0).view()}, {.name = "ends", .view = ends->view()}};
  std::vector<std::pair<cudf::type_id, std::string>> outputs;
  std::string copies;
  for (size_t i = 0; i < columns.size(); ++i) {
    rowArrays.push_back({"lineitem", columns[i], "in" + std::to_string(i), {}, base});
    outputs.emplace_back(typeId(ctx.meta("lineitem", columns[i]).dtype), "out" + std::to_string(i));
    copies += fmt::format("  *out{0} = in{0}[line];\n", i);
  }
  if (total == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (const auto& [type, name] : outputs) {
      empty.push_back(cudf::make_empty_column(cudf::data_type{type}));
    }
    return std::make_unique<cudf::table>(std::move(empty));
  }
  // The kept row of rank `row` is in the first word whose inclusive count
  // exceeds it.
  return transformRows(
      ctx,
      total,
      "",
      {},
      rowArrays,
      outputs,
      fmt::format(
          "  int64_t low = 0, high = {1} - 1;\n"
          "  while (low < high) {{\n"
          "    const int64_t middle = (low + high) / 2;\n"
          "    if (ends[middle] > row) high = middle; else low = middle + 1;\n"
          "  }}\n"
          "  uint32_t bits = words[low];\n"
          "  for (int skip = row - (ends[low] - __popc(bits)); skip > 0; --skip) bits &= bits - 1;\n"
          "  const int64_t line = {0} + low * 32 + __ffs(bits) - 1;\n",
          0,
          words) +
          copies);
}

} // namespace

std::unique_ptr<cudf::table> linesOfParts(
    QueryContext& ctx,
    const cudf::column_view& parts,
    RowRange range,
    const std::vector<ArrayInput>& arrays,
    const std::string& condition,
    const std::vector<std::string>& columns) {
  auto [partMask, partNulls] = cudf::bools_to_mask(parts, ctx.stream());
  const cudf::column_view wanted =
      deviceView(cudf::type_id::UINT32, partMask->data(), (parts.size() + 31) / 32);
  std::vector<std::unique_ptr<cudf::table>> pieces;
  for (const RowRange batch : rowBatches(range, range.size(), 1, 32)) {
    pieces.push_back(linesOfPartsIn(ctx, wanted, batch, arrays, condition, columns));
  }
  if (pieces.size() == 1) {
    return std::move(pieces[0]);
  }
  return concatenateTables(ctx, pieces);
}

std::vector<std::pair<int64_t, int64_t>>
orderChunks(QueryContext& ctx, int64_t begin, int64_t end, int64_t maxLines) {
  std::vector<std::pair<int64_t, int64_t>> chunks;
  if (end <= begin) {
    return chunks;
  }
  if (batched()) {
    // Batches count four lines an order, as the hand-written kernels do, and
    // end at multiples of their size, where lineStart needs no search.
    const int64_t step = std::max<int64_t>(1, batchRows() / 4);
    for (int64_t first = begin; first < end;) {
      const int64_t last = std::min(end, (first / step + 1) * step);
      chunks.emplace_back(first, last);
      first = last;
    }
    return chunks;
  }
  // Orders hold at most seven lines, so a run of maxLines / 8 orders fits.
  for (const RowRange run : rowBatches({begin, end}, std::max<int64_t>(1, maxLines / 8))) {
    chunks.emplace_back(run.begin, run.end);
  }
  return chunks;
}

bool batched() {
  return batchRows() > 0;
}

std::vector<RowRange> rowBatches(RowRange range, int64_t whole, int64_t rowsPerItem, int64_t align) {
  std::vector<RowRange> pieces;
  if (range.size() <= 0) {
    return pieces;
  }
  int64_t step = std::max<int64_t>(1, whole);
  if (batched()) {
    const int64_t items = std::max<int64_t>(1, batchRows() / rowsPerItem);
    step = (items + align - 1) / align * align;
  }
  for (int64_t first = range.begin; first < range.end; first += step) {
    pieces.push_back({first, std::min(range.end, first + step)});
  }
  return pieces;
}

OrderBatch orderBatch(QueryContext& ctx, int64_t first, int64_t last) {
  OrderBatch batch;
  batch.first = first;
  batch.last = last;
  const auto stream = ctx.stream();
  if (!batched()) {
    const uint32_t* starts = orderLineStarts(ctx);
    batch.starts = deviceView(cudf::type_id::UINT32, starts + first, last - first + 1);
    batch.lines = {ctx.download(starts + first, 1)[0], ctx.download(starts + last, 1)[0]};
    return batch;
  }
  // An order starts where l_orderkey changes from the line before.
  batch.lines = {lineStart(ctx, first), lineStart(ctx, last)};
  batch.base = batch.lines.begin;
  const int64_t lines = batch.lines.size();
  const cudf::numeric_scalar<int32_t> zero(0, true, stream);
  const cudf::numeric_scalar<int32_t> end(static_cast<int32_t>(lines), true, stream);
  std::vector<std::unique_ptr<cudf::column>> parts;
  parts.push_back(cudf::make_column_from_scalar(zero, last > first ? 1 : 0, stream));
  if (lines > 1) {
    const cudf::column_view keys = fixedView(ctx, "lineitem", "l_orderkey", batch.lines);
    const auto next = cudf::slice(keys, {1, static_cast<cudf::size_type>(lines)})[0];
    const auto previous = cudf::slice(keys, {0, static_cast<cudf::size_type>(lines - 1)})[0];
    auto starts = cudf::binary_operation(
        next, previous, cudf::binary_operator::NOT_EQUAL, cudf::data_type{cudf::type_id::BOOL8}, stream);
    auto rows = sequence(ctx, lines - 1, 1);
    auto selected = selectRows(ctx, cudf::table_view({rows->view()}), starts->view());
    parts.push_back(std::move(selected->release()[0]));
  }
  parts.push_back(cudf::make_column_from_scalar(end, 1, stream));
  std::vector<cudf::column_view> views;
  for (const auto& part : parts) {
    views.push_back(part->view());
  }
  batch.owned = cudf::concatenate(views, stream);
  if (batch.owned->size() != last - first + 1) {
    throw std::runtime_error(
        "lines " + std::to_string(batch.lines.begin) + ".." + std::to_string(batch.lines.end) + " hold " +
        std::to_string(batch.owned->size() - 1) + " orders, not " + std::to_string(last - first));
  }
  batch.starts = batch.owned->view();
  return batch;
}

std::vector<ArrayInput> withLines(const OrderBatch& batch, std::vector<ArrayInput> arrays) {
  for (auto& array : arrays) {
    if (array.table == "lineitem") {
      array.first = batch.base;
    }
  }
  arrays.push_back({.name = "starts", .view = batch.starts});
  return arrays;
}

std::unique_ptr<cudf::table>
perLine(QueryContext& ctx, const OrderLines& lines, const std::vector<cudf::column_view>& columns) {
  return cudf::repeat(cudf::table_view(columns), lines.counts->view(), ctx.stream());
}

namespace {

/// 'aggregation' of 'values' over each order's lines. A sorted groupby over
/// l_orderkey finds the orders' boundaries once per OrderLines; segmented
/// reductions spend a thread block on each order's few lines.
std::unique_ptr<cudf::column> perOrder(
    QueryContext& ctx,
    const cudf::column_view& values,
    const OrderLines& lines,
    std::unique_ptr<cudf::groupby_aggregation> aggregation) {
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(std::move(aggregation));
  auto [keys, results] = lines.groups->aggregate(requests, ctx.stream());
  if (keys->num_rows() != lines.end - lines.begin) {
    throw std::runtime_error("orders without lines");
  }
  return std::move(results[0].results[0]);
}

} // namespace

std::unique_ptr<cudf::column> sumPerOrder(QueryContext& ctx, const cudf::column_view& values, const OrderLines& lines) {
  auto sums = perOrder(ctx, values, lines, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  return sums->type().id() == cudf::type_id::INT64 ? std::move(sums)
                                                     : cudf::cast(sums->view(), cudf::data_type{cudf::type_id::INT64}, ctx.stream());
}

std::unique_ptr<cudf::column>
extremePerOrder(QueryContext& ctx, const cudf::column_view& values, const OrderLines& lines, bool maximum) {
  return perOrder(
      ctx,
      values,
      lines,
      maximum ? cudf::make_max_aggregation<cudf::groupby_aggregation>()
              : cudf::make_min_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<cudf::column>
partsuppRows(QueryContext& ctx, const cudf::column_view& partkeys, const cudf::column_view& suppkeys) {
  const std::string part = std::string(deviceType(partkeys.type().id())) + " part";
  std::vector<std::unique_ptr<cudf::table>> slots;
  for (int slot = 0; slot < 4; ++slot) {
    auto rows = computeColumn(
        ctx, part, "(int32_t{part} - 1) * 4 + " + std::to_string(slot), cudf::type_id::INT32, {partkeys});
    slots.push_back(gatherColumns(ctx, "partsupp", {"ps_suppkey"}, rows->view()));
  }
  const std::string supplier = deviceType(slots[0]->get_column(0).type().id());
  auto rows = computeColumn(
      ctx,
      part + ", " + deviceType(suppkeys.type().id()) + " supplier, " + supplier + " s0, " + supplier + " s1, " + supplier +
          " s2, " + supplier + " s3",
      "supplier == s0 ? base : supplier == s1 ? base + 1 : supplier == s2 ? base + 2 : supplier == s3 ? base + 3 : -1",
      cudf::type_id::INT32,
      {partkeys,
       suppkeys,
       slots[0]->get_column(0).view(),
       slots[1]->get_column(0).view(),
       slots[2]->get_column(0).view(),
       slots[3]->get_column(0).view()},
      "  const int32_t base = (int32_t{part} - 1) * 4;\n");
  if (rows->size() > 0) {
    auto lowest =
        cudf::reduce(rows->view(), *cudf::make_min_aggregation<cudf::reduce_aggregation>(), rows->type(), ctx.stream());
    if (static_cast<cudf::numeric_scalar<int32_t>&>(*lowest).value(ctx.stream()) < 0) {
      throw std::runtime_error("lineitem (part, supplier) missing from the partsupp layout");
    }
  }
  return rows;
}

std::unique_ptr<cudf::table> concatenateTables(QueryContext& ctx, const std::vector<std::unique_ptr<cudf::table>>& pieces) {
  std::vector<cudf::table_view> views;
  for (const auto& piece : pieces) {
    views.push_back(piece->view());
  }
  return cudf::concatenate(views, ctx.stream());
}

std::unique_ptr<cudf::column>
likeColumn(QueryContext& ctx, const std::string& table, const std::string& column, const std::string& pattern, RowRange rows) {
  // %a%b%...% with LTO: a transform that finds each literal after the last.
  const bool literals = pattern.size() >= 2 && pattern.front() == '%' && pattern.back() == '%' &&
      pattern.find_first_of("_\\") == std::string::npos;
  if (useJit() && literals) {
    std::string source = R"(
#include <cudf/strings/string_view.cuh>
#include <cuda/std/cstdint>

/// Where `word` starts in [from, size) of `text`, or -1.
__device__ inline int find(const char* text, int from, int size, const char* word, int length) {
  for (int i = from; i + length <= size; ++i) {
    if (text[i] == word[0]) {
      int j = 1;
      while (j < length && text[i + j] == word[j]) {
        ++j;
      }
      if (j == length) {
        return i;
      }
    }
  }
  return -1;
}

extern "C" __device__ int transform(bool* matches, cudf::string_view value) {
  const char* text = value.data();
  const int size = value.size_bytes();
  int at = 0;
)";
    std::stringstream words(pattern);
    std::string word;
    while (std::getline(words, word, '%')) {
      if (!word.empty()) {
        source += fmt::format(
            "  at = find(text, at, size, \"{0}\", {1});\n  if (at < 0) {{\n    *matches = false;\n    return 0;\n  }}\n"
            "  at += {1};\n",
            word,
            word.size());
      }
    }
    source += "  *matches = true;\n  return 0;\n}\n";
    const cudf::transform_input in[] = {stringView(ctx, table, column, rows)};
    const cudf::transform_output out[] = {{cudf::data_type{cudf::type_id::BOOL8}, cudf::output_nullability::ALL_VALID}};
    return std::move(transformColumns(ctx, source, in, out)->release()[0]);
  }
  return cudf::strings::like(
      cudf::strings_column_view(stringView(ctx, table, column, rows)), pattern, "", ctx.stream());
}

cudf::column_view stringView(QueryContext& ctx, const std::string& table, const std::string& name, RowRange rows) {
  if (rows.size() > std::numeric_limits<cudf::size_type>::max()) {
    throw std::invalid_argument("cuDF columns hold at most 2^31 - 1 rows");
  }
  const int64_t* offsets = ctx.offsets(table, name);
  const cudf::column_view offsetView(
      cudf::data_type{cudf::type_id::INT64}, static_cast<cudf::size_type>(rows.size() + 1), offsets + rows.begin, nullptr, 0);
  return cudf::column_view(
      cudf::data_type{cudf::type_id::STRING},
      static_cast<cudf::size_type>(rows.size()),
      ctx.chars(table, name),
      nullptr,
      0,
      0,
      {offsetView});
}

std::vector<int32_t> firstSorted(
    QueryContext& ctx,
    const cudf::table_view& keys,
    const std::vector<cudf::order>& orders,
    int64_t limit) {
  auto order = cudf::sorted_order(keys, orders, {}, ctx.stream());
  const auto count = std::min<int64_t>(limit, order->size());
  return download<int32_t>(ctx, cudf::slice(order->view(), {0, static_cast<cudf::size_type>(count)})[0]);
}

std::unique_ptr<cudf::table> firstRows(
    QueryContext& ctx,
    const cudf::table_view& values,
    const cudf::table_view& keys,
    const std::vector<cudf::order>& orders,
    int64_t limit) {
  auto order = cudf::sorted_order(keys, orders, {}, ctx.stream());
  const auto count = static_cast<cudf::size_type>(std::min<int64_t>(limit, order->size()));
  return cudf::gather(values, cudf::slice(order->view(), {0, count})[0], cudf::out_of_bounds_policy::DONT_CHECK, ctx.stream());
}

std::unique_ptr<cudf::column> sequence(QueryContext& ctx, int64_t count, int32_t first, int32_t step) {
  const cudf::numeric_scalar<int32_t> init(first, true, ctx.stream());
  const cudf::numeric_scalar<int32_t> stride(step, true, ctx.stream());
  return cudf::sequence(static_cast<cudf::size_type>(count), init, stride, ctx.stream());
}

std::unique_ptr<cudf::table> selectRows(QueryContext& ctx, const cudf::table_view& table, const cudf::column_view& mask) {
  return cudf::apply_boolean_mask(table, mask, ctx.stream());
}

std::vector<int64_t> downloadInts(QueryContext& ctx, const cudf::column_view& column) {
  if (column.type().id() == cudf::type_id::INT64) {
    return download<int64_t>(ctx, column);
  }
  auto wide = cudf::cast(column, cudf::data_type{cudf::type_id::INT64}, ctx.stream());
  return download<int64_t>(ctx, wide->view());
}

Grouped groupBy(
    QueryContext& ctx,
    const cudf::table_view& keys,
    const std::vector<std::pair<cudf::column_view, Aggregate>>& aggregates) {
  std::vector<cudf::groupby::aggregation_request> requests(aggregates.size());
  for (size_t i = 0; i < aggregates.size(); ++i) {
    requests[i].values = aggregates[i].first;
    switch (aggregates[i].second) {
      case Aggregate::kSum:
        requests[i].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kMin:
        requests[i].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kMax:
        requests[i].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kCount:
        requests[i].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        break;
      case Aggregate::kDistinct:
        requests[i].aggregations.push_back(cudf::make_nunique_aggregation<cudf::groupby_aggregation>());
        break;
    }
  }
  cudf::groupby::groupby grouping(keys);
  auto [groupKeys, results] = grouping.aggregate(requests, ctx.stream());
  Grouped grouped;
  grouped.keys = std::move(groupKeys);
  for (auto& result : results) {
    grouped.values.push_back(std::move(result.results[0]));
  }
  return grouped;
}

StreamingGroups::StreamingGroups(const std::vector<Aggregate>& aggregates, int64_t maxKeys) : maxKeys_(maxKeys) {
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    std::unique_ptr<cudf::groupby_aggregation> aggregation;
    switch (aggregates[i]) {
      case Aggregate::kSum:
        aggregation = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
        break;
      case Aggregate::kMin:
        aggregation = cudf::make_min_aggregation<cudf::groupby_aggregation>();
        break;
      case Aggregate::kMax:
        aggregation = cudf::make_max_aggregation<cudf::groupby_aggregation>();
        break;
      case Aggregate::kCount:
        aggregation = cudf::make_count_aggregation<cudf::groupby_aggregation>();
        break;
      case Aggregate::kDistinct:
        throw std::invalid_argument("streaming groups do not count distinct values");
    }
    requests.push_back({static_cast<cudf::size_type>(i + 1), std::move(aggregation)});
  }
  const std::vector<cudf::size_type> keys = {0};
  groups_ = std::make_unique<cudf::groupby::streaming_groupby>(
      keys, requests, static_cast<cudf::size_type>(std::max<int64_t>(1, maxKeys)));
}

StreamingGroups::~StreamingGroups() = default;

void StreamingGroups::add(QueryContext& ctx, const cudf::table_view& batch) {
  // A batch may not have more rows than the state has keys.
  for (int64_t first = 0; first < batch.num_rows(); first += maxKeys_) {
    const auto last = static_cast<cudf::size_type>(std::min<int64_t>(batch.num_rows(), first + maxKeys_));
    groups_->aggregate(cudf::slice(batch, {static_cast<cudf::size_type>(first), last})[0], ctx.stream());
    added_ = true;
  }
}

Grouped StreamingGroups::finish(QueryContext& ctx) {
  Grouped grouped;
  if (!added_) {
    return grouped;
  }
  auto [keys, results] = groups_->finalize(ctx.stream());
  grouped.keys = std::move(keys);
  for (auto& result : results) {
    grouped.values.push_back(std::move(result.results[0]));
  }
  return grouped;
}

void registerCudfBackend() {
  cudaMemPool_t pool;
  SV_CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
  rmm::mr::set_current_device_resource(rmm::mr::cuda_async_view_memory_resource(pool));
  registerImplementation(Backend::kCudf, 1, q1);
  registerImplementation(Backend::kCudf, 2, q2);
  registerImplementation(Backend::kCudf, 3, q3);
  registerImplementation(Backend::kCudf, 4, q4);
  registerImplementation(Backend::kCudf, 5, q5);
  registerImplementation(Backend::kCudf, 6, q6);
  registerImplementation(Backend::kCudf, 7, q7);
  registerImplementation(Backend::kCudf, 8, q8);
  registerImplementation(Backend::kCudf, 9, q9);
  registerImplementation(Backend::kCudf, 10, q10);
  registerImplementation(Backend::kCudf, 11, q11);
  registerImplementation(Backend::kCudf, 12, q12);
  registerImplementation(Backend::kCudf, 13, q13);
  registerImplementation(Backend::kCudf, 14, q14);
  registerImplementation(Backend::kCudf, 15, q15);
  registerImplementation(Backend::kCudf, 16, q16);
  registerImplementation(Backend::kCudf, 17, q17);
  registerImplementation(Backend::kCudf, 18, q18);
  registerImplementation(Backend::kCudf, 19, q19);
  registerImplementation(Backend::kCudf, 20, q20);
  registerImplementation(Backend::kCudf, 21, q21);
  registerImplementation(Backend::kCudf, 22, q22);
}

} // namespace facebook::velox::supervelox::cudf_backend
