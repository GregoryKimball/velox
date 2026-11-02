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

#include "velox/experimental/cudf/exec/FusedAggregation.h"

#include "velox/core/Expressions.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include "velox/experimental/cudf/CudfNoDefaults.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <set>

namespace facebook::velox::cudf_velox {
namespace {

using Kind = dense::Kind;
using MergeOp = dense::MergeOp;

constexpr size_t kMaxListValues = 256;

std::string_view baseName(const std::string& name) {
  const auto dot = name.rfind('.');
  return dot == std::string::npos ? std::string_view(name)
                                  : std::string_view(name).substr(dot + 1);
}

/// How a generated expression holds its value: int64_t, an unscaled __int128
/// with a scale, double, bool or cudf::string_view.
enum class Repr { kInteger, kDecimal, kDouble, kBool, kString };

struct Code {
  std::string text;
  Repr repr;
  int32_t scale{0};
};

std::string int64Literal(int64_t value) {
  if (value == std::numeric_limits<int64_t>::min()) {
    return "(-9223372036854775807LL - 1)";
  }
  return fmt::format("{}LL", value);
}

std::string int128Literal(int128_t value) {
  if (value >= std::numeric_limits<int64_t>::min() &&
      value <= std::numeric_limits<int64_t>::max()) {
    return fmt::format("((__int128){})", int64Literal(static_cast<int64_t>(value)));
  }
  const auto high = static_cast<int64_t>(value >> 64);
  const auto low = static_cast<uint64_t>(value);
  return fmt::format(
      "((__int128)(((unsigned __int128)(unsigned long long){} << 64) | (unsigned __int128){}ULL))",
      int64Literal(high),
      low);
}

std::string pow10(int32_t exponent) {
  int128_t value = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    value *= 10;
  }
  return int128Literal(value);
}

std::optional<Repr> reprOf(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return Repr::kBool;
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
      return Repr::kInteger;
    case TypeKind::BIGINT:
      return type->isShortDecimal() ? Repr::kDecimal : Repr::kInteger;
    case TypeKind::HUGEINT:
      return type->isLongDecimal() ? std::optional(Repr::kDecimal) : std::nullopt;
    case TypeKind::DOUBLE:
      return Repr::kDouble;
    case TypeKind::VARCHAR:
      return Repr::kString;
    default:
      return std::nullopt;
  }
}

int32_t scaleOf(const TypePtr& type) {
  return type->isDecimal() ? getDecimalPrecisionScale(*type).second : 0;
}

/// The device parameter type of an input column of 'type', whose cuDF column
/// is passed retyped to the integer of its width for dates and decimals.
std::optional<std::string> parameterType(const TypePtr& type) {
  if (type->isDate()) {
    return "int32_t";
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return "bool";
    case TypeKind::TINYINT:
      return "int8_t";
    case TypeKind::SMALLINT:
      return "int16_t";
    case TypeKind::INTEGER:
      return "int32_t";
    case TypeKind::BIGINT:
      return "int64_t";
    case TypeKind::DOUBLE:
      return "double";
    case TypeKind::VARCHAR:
      return "cudf::string_view";
    default:
      return std::nullopt;
  }
}

/// Translates Velox expressions over the columns of 'input' into CUDA
/// expressions over the callback parameters c0, c1, ... Records the
/// parameters and constants the expressions use. Fails with 'error' set.
class Generator {
 public:
  explicit Generator(RowTypePtr input, const Encodings* encodings = nullptr)
      : input_(std::move(input)), encodings_(encodings) {}

  /// 'expr' in the representation of its type.
  std::optional<Code> typed(const core::TypedExprPtr& expr) {
    auto code = this->expr(expr);
    return code.has_value() ? checked(std::move(*code), expr->type()) : std::nullopt;
  }

  std::optional<Code> expr(const core::TypedExprPtr& expr) {
    if (const auto* field = dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
      if (!field->isInputColumn()) {
        return fail("access to a field of a struct");
      }
      return column(field->name());
    }
    if (const auto* constant = dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
      return this->constant(*constant);
    }
    if (const auto* cast = dynamic_cast<const core::CastTypedExpr*>(expr.get())) {
      return this->cast(*cast);
    }
    if (const auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
      return this->call(*call);
    }
    return fail(fmt::format("expression {}", expr->toString()));
  }

  std::optional<Code> column(const std::string& name) {
    const auto index = input_->getChildIdxIfExists(name);
    if (!index.has_value()) {
      return fail(fmt::format("no input column {}", name));
    }
    const auto& stored = input_->childAt(*index);
    const Encoding* encoding = nullptr;
    if (encodings_ != nullptr) {
      if (const auto it = encodings_->find(name); it != encodings_->end()) {
        encoding = &it->second;
      }
    }
    const auto& type = encoding != nullptr ? encoding->type : stored;
    const auto parameter = parameterType(stored);
    const auto repr = reprOf(type);
    if (!parameter.has_value() || !repr.has_value()) {
      return fail(fmt::format("input column {} of type {}", name, type->toString()));
    }
    if (encoding != nullptr && !stored->isTinyint() && !stored->isSmallint() && !stored->isInteger() &&
        !stored->isBigint()) {
      return fail(fmt::format("input column {} encoded as {}", name, stored->toString()));
    }
    auto [it, added] = parameters_.emplace(*index, static_cast<int32_t>(channels_.size()));
    if (added) {
      channels_.push_back(static_cast<int32_t>(*index));
      types_.push_back(*parameter);
    }
    const auto text = fmt::format("c{}", it->second);
    switch (*repr) {
      case Repr::kInteger:
        return Code{fmt::format("((int64_t){})", text), *repr};
      case Repr::kDecimal:
        return Code{fmt::format("((__int128){})", text), *repr, scaleOf(type)};
      case Repr::kDouble:
        if (encoding != nullptr) {
          return Code{fmt::format("((double){} / {:.1f})", text, std::pow(10.0, encoding->scale)), *repr};
        }
        return Code{text, *repr};
      default:
        if (encoding != nullptr) {
          return fail(fmt::format("input column {} of type {} encoded", name, type->toString()));
        }
        return Code{text, *repr};
    }
  }

  /// 'filter' applied to input column 'name'.
  std::optional<std::string> filter(const std::string& name, const common::Filter& filter) {
    const auto value = column(name);
    if (!value.has_value()) {
      return std::nullopt;
    }
    return this->filter(*value, filter);
  }

  std::optional<std::string> filter(const Code& value, const common::Filter& filter) {
    using common::FilterKind;
    const bool integer = value.repr == Repr::kInteger || value.repr == Repr::kDecimal;
    auto range = [&](int64_t lower, int64_t upper) {
      std::vector<std::string> terms;
      if (lower != std::numeric_limits<int64_t>::min()) {
        terms.push_back(fmt::format("{} >= {}", value.text, int64Literal(lower)));
      }
      if (upper != std::numeric_limits<int64_t>::max()) {
        terms.push_back(fmt::format("{} <= {}", value.text, int64Literal(upper)));
      }
      return terms.empty() ? std::string("true") : fmt::format("({})", fmt::join(terms, " && "));
    };
    auto list = [&](const std::vector<int64_t>& values) -> std::optional<std::string> {
      if (values.size() > kMaxListValues) {
        failText(fmt::format("a list of {} values", values.size()));
        return std::nullopt;
      }
      std::vector<std::string> terms;
      for (const auto v : values) {
        terms.push_back(fmt::format("{} == {}", value.text, int64Literal(v)));
      }
      return fmt::format("({})", fmt::join(terms, " || "));
    };
    auto strings = [&](const auto& values) -> std::optional<std::string> {
      if (values.size() > kMaxListValues) {
        failText(fmt::format("a list of {} strings", values.size()));
        return std::nullopt;
      }
      std::vector<std::string> sorted(values.begin(), values.end());
      std::sort(sorted.begin(), sorted.end());
      std::vector<std::string> terms;
      for (const auto& v : sorted) {
        terms.push_back(fmt::format("{} == {}", value.text, string(v)));
      }
      return fmt::format("({})", fmt::join(terms, " || "));
    };
    switch (filter.kind()) {
      case FilterKind::kAlwaysTrue:
      case FilterKind::kIsNotNull:
        return std::string("true");
      case FilterKind::kAlwaysFalse:
      case FilterKind::kIsNull:
        return std::string("false");
      case FilterKind::kBoolValue:
        if (value.repr != Repr::kBool) {
          break;
        }
        return filter.testBool(true) ? value.text : fmt::format("!{}", value.text);
      case FilterKind::kBigintRange: {
        if (!integer) {
          break;
        }
        const auto& f = static_cast<const common::BigintRange&>(filter);
        return range(f.lower(), f.upper());
      }
      case FilterKind::kNegatedBigintRange: {
        if (!integer) {
          break;
        }
        const auto& f = static_cast<const common::NegatedBigintRange&>(filter);
        return fmt::format("!{}", range(f.lower(), f.upper()));
      }
      case FilterKind::kBigintValuesUsingHashTable:
        if (!integer) {
          break;
        }
        return list(static_cast<const common::BigintValuesUsingHashTable&>(filter).values());
      case FilterKind::kBigintValuesUsingBitmask:
        if (!integer) {
          break;
        }
        return list(static_cast<const common::BigintValuesUsingBitmask&>(filter).values());
      case FilterKind::kBigintMultiRange: {
        if (!integer) {
          break;
        }
        std::vector<std::string> terms;
        for (const auto& r : static_cast<const common::BigintMultiRange&>(filter).ranges()) {
          terms.push_back(range(r->lower(), r->upper()));
        }
        return fmt::format("({})", fmt::join(terms, " || "));
      }
      case FilterKind::kDoubleRange: {
        if (value.repr != Repr::kDouble) {
          break;
        }
        const auto& f = static_cast<const common::DoubleRange&>(filter);
        std::vector<std::string> terms;
        if (!f.lowerUnbounded()) {
          terms.push_back(fmt::format(
              "{} {} {}", value.text, f.lowerExclusive() ? ">" : ">=", doubleLiteral(f.lower())));
        }
        if (!f.upperUnbounded()) {
          terms.push_back(fmt::format(
              "{} {} {}", value.text, f.upperExclusive() ? "<" : "<=", doubleLiteral(f.upper())));
        }
        return terms.empty() ? std::string("true") : fmt::format("({})", fmt::join(terms, " && "));
      }
      case FilterKind::kBytesValues:
        if (value.repr != Repr::kString) {
          break;
        }
        return strings(static_cast<const common::BytesValues&>(filter).values());
      case FilterKind::kNegatedBytesValues: {
        if (value.repr != Repr::kString) {
          break;
        }
        auto in = strings(static_cast<const common::NegatedBytesValues&>(filter).values());
        return in.has_value() ? std::optional(fmt::format("!{}", *in)) : std::nullopt;
      }
      case FilterKind::kBytesRange: {
        if (value.repr != Repr::kString) {
          break;
        }
        const auto& f = static_cast<const common::BytesRange&>(filter);
        if (f.isSingleValue()) {
          return fmt::format("({} == {})", value.text, string(f.lower()));
        }
        std::vector<std::string> terms;
        if (!f.lowerUnbounded()) {
          terms.push_back(fmt::format(
              "{}.compare({}) {} 0", value.text, string(f.lower()), f.lowerExclusive() ? ">" : ">="));
        }
        if (!f.upperUnbounded()) {
          terms.push_back(fmt::format(
              "{}.compare({}) {} 0", value.text, string(f.upper()), f.upperExclusive() ? "<" : "<="));
        }
        return terms.empty() ? std::string("true") : fmt::format("({})", fmt::join(terms, " && "));
      }
      case FilterKind::kMultiRange: {
        std::vector<std::string> terms;
        for (const auto& f : static_cast<const common::MultiRange&>(filter).filters()) {
          auto term = this->filter(value, *f);
          if (!term.has_value()) {
            return std::nullopt;
          }
          terms.push_back(*term);
        }
        return fmt::format("({})", fmt::join(terms, " || "));
      }
      default:
        break;
    }
    failText(fmt::format("filter {}", filter.toString()));
    return std::nullopt;
  }

  const std::vector<int32_t>& channels() const {
    return channels_;
  }

  /// The parameter list "type c0, type c1, ...", empty without inputs.
  std::string parameters() const {
    std::vector<std::string> out;
    for (size_t i = 0; i < types_.size(); ++i) {
      out.push_back(fmt::format("{} c{}", types_[i], i));
    }
    return fmt::format("{}", fmt::join(out, ", "));
  }

  /// Definitions the expressions refer to.
  std::string prelude() const {
    return constants_;
  }

  bool usesStrings() const {
    return usesStrings_;
  }

  bool usesLike() const {
    return usesLike_;
  }

  const std::string& error() const {
    return error_;
  }

 private:
  std::nullopt_t fail(const std::string& what) {
    failText(what);
    return std::nullopt;
  }

  void failText(const std::string& what) {
    if (error_.empty()) {
      error_ = what;
    }
  }

  static std::string doubleLiteral(double value) {
    if (std::isinf(value)) {
      return value > 0 ? "__longlong_as_double(0x7ff0000000000000LL)"
                       : "__longlong_as_double((long long)0xfff0000000000000ULL)";
    }
    return fmt::format("{:.17g}", value);
  }

  /// A device constant holding 'value' and a string view of it.
  std::string string(const std::string& value) {
    usesStrings_ = true;
    auto [it, added] = strings_.emplace(value, static_cast<int32_t>(strings_.size()));
    if (added) {
      std::vector<std::string> bytes;
      for (const auto c : value) {
        bytes.push_back(fmt::format("{}", static_cast<int>(static_cast<unsigned char>(c))));
      }
      bytes.push_back("0");
      constants_ += fmt::format(
          "__device__ const char k{}[] = {{{}}};\n", it->second, fmt::join(bytes, ", "));
    }
    return fmt::format("cudf::string_view(k{}, {})", it->second, value.size());
  }

  template <typename T>
  static T scalar(const core::ConstantTypedExpr& constant) {
    if (constant.hasValueVector()) {
      return constant.valueVector()->as<SimpleVector<T>>()->valueAt(0);
    }
    return constant.value().value<T>();
  }

  std::optional<Code> constant(const core::ConstantTypedExpr& constant) {
    if (constant.isNull()) {
      return fail("a null constant");
    }
    const auto& type = constant.type();
    if (type->isDate()) {
      return Code{int64Literal(scalar<int32_t>(constant)), Repr::kInteger};
    }
    switch (type->kind()) {
      case TypeKind::BOOLEAN:
        return Code{scalar<bool>(constant) ? "true" : "false", Repr::kBool};
      case TypeKind::TINYINT:
        return Code{int64Literal(scalar<int8_t>(constant)), Repr::kInteger};
      case TypeKind::SMALLINT:
        return Code{int64Literal(scalar<int16_t>(constant)), Repr::kInteger};
      case TypeKind::INTEGER:
        return Code{int64Literal(scalar<int32_t>(constant)), Repr::kInteger};
      case TypeKind::BIGINT: {
        const auto value = scalar<int64_t>(constant);
        if (type->isShortDecimal()) {
          return Code{int128Literal(value), Repr::kDecimal, scaleOf(type)};
        }
        return Code{int64Literal(value), Repr::kInteger};
      }
      case TypeKind::HUGEINT:
        if (!type->isLongDecimal()) {
          break;
        }
        return Code{int128Literal(scalar<int128_t>(constant)), Repr::kDecimal, scaleOf(type)};
      case TypeKind::DOUBLE: {
        const auto value = scalar<double>(constant);
        if (std::isnan(value)) {
          break;
        }
        return Code{doubleLiteral(value), Repr::kDouble};
      }
      case TypeKind::VARCHAR: {
        if (constant.hasValueVector()) {
          return Code{
              string(std::string(constant.valueVector()->as<SimpleVector<StringView>>()->valueAt(0))),
              Repr::kString};
        }
        return Code{string(constant.value().value<std::string>()), Repr::kString};
      }
      default:
        break;
    }
    return fail(fmt::format("a constant of type {}", type->toString()));
  }

  /// 'code', an unscaled decimal of 'from' digits after the point, with 'to'.
  static Code rescale(const Code& code, int32_t to) {
    if (code.scale == to) {
      return code;
    }
    return Code{fmt::format("({} * {})", code.text, pow10(to - code.scale)), Repr::kDecimal, to};
  }

  /// 'left' and 'right' in one representation, decimals at the larger scale.
  std::optional<std::pair<Code, Code>> unify(const Code& left, const Code& right) {
    if (left.repr != right.repr) {
      return fail("operands of different kinds");
    }
    if (left.repr == Repr::kDecimal) {
      const auto scale = std::max(left.scale, right.scale);
      return std::pair{rescale(left, scale), rescale(right, scale)};
    }
    return std::pair{left, right};
  }

  bool matches(const Code& code, const TypePtr& type) {
    const auto repr = reprOf(type);
    if (type->isDate()) {
      return code.repr == Repr::kInteger;
    }
    if (!repr.has_value() || *repr != code.repr || (code.repr == Repr::kDecimal && code.scale != scaleOf(type))) {
      failText(fmt::format("a result that is not {}", type->toString()));
      return false;
    }
    return true;
  }

  std::optional<Code> checked(Code code, const TypePtr& type) {
    if (!matches(code, type)) {
      return std::nullopt;
    }
    return code;
  }

  std::optional<std::vector<Code>> operands(const core::CallTypedExpr& call) {
    std::vector<Code> out;
    for (const auto& input : call.inputs()) {
      auto code = expr(input);
      if (!code.has_value()) {
        return std::nullopt;
      }
      out.push_back(std::move(*code));
    }
    return out;
  }

  std::optional<std::string> compare(const Code& left, const Code& right, const std::string& op) {
    auto pair = unify(left, right);
    if (!pair.has_value()) {
      return std::nullopt;
    }
    const auto& [l, r] = *pair;
    if (l.repr == Repr::kString) {
      if (op == "==") {
        return fmt::format("({} == {})", l.text, r.text);
      }
      if (op == "!=") {
        return fmt::format("!({} == {})", l.text, r.text);
      }
      return fmt::format("({}.compare({}) {} 0)", l.text, r.text, op);
    }
    return fmt::format("({} {} {})", l.text, op, r.text);
  }

  std::optional<Code> call(const core::CallTypedExpr& call) {
    const auto name = baseName(call.name());
    const auto& type = call.type();
    static const std::map<std::string_view, std::string> kComparisons = {
        {"eq", "=="}, {"neq", "!="}, {"lt", "<"}, {"lte", "<="}, {"gt", ">"}, {"gte", ">="}};
    if (name == "in") {
      return in(call);
    }
    if (name == "like") {
      return like(call);
    }
    auto args = operands(call);
    if (!args.has_value()) {
      return std::nullopt;
    }
    auto& a = *args;
    if (name == "and" || name == "or") {
      std::vector<std::string> terms;
      for (const auto& arg : a) {
        if (arg.repr != Repr::kBool) {
          return fail("a logical operand that is not boolean");
        }
        terms.push_back(arg.text);
      }
      return Code{fmt::format("({})", fmt::join(terms, name == "and" ? " && " : " || ")), Repr::kBool};
    }
    if (name == "not" && a.size() == 1 && a[0].repr == Repr::kBool) {
      return Code{fmt::format("(!{})", a[0].text), Repr::kBool};
    }
    if (const auto op = kComparisons.find(name); op != kComparisons.end() && a.size() == 2) {
      auto text = compare(a[0], a[1], op->second);
      return text.has_value() ? std::optional(Code{*text, Repr::kBool}) : std::nullopt;
    }
    if (name == "between" && a.size() == 3) {
      auto low = compare(a[0], a[1], ">=");
      auto high = compare(a[0], a[2], "<=");
      if (!low.has_value() || !high.has_value()) {
        return std::nullopt;
      }
      return Code{fmt::format("({} && {})", *low, *high), Repr::kBool};
    }
    if ((name == "plus" || name == "minus") && a.size() == 2) {
      auto pair = unify(a[0], a[1]);
      if (!pair.has_value() || pair->first.repr == Repr::kBool || pair->first.repr == Repr::kString) {
        return fail(fmt::format("{} of these operands", name));
      }
      return checked(
          Code{fmt::format("({} {} {})", pair->first.text, name == "plus" ? "+" : "-", pair->second.text),
               pair->first.repr,
               pair->first.scale},
          type);
    }
    if (name == "multiply" && a.size() == 2) {
      if (a[0].repr != a[1].repr || a[0].repr == Repr::kBool || a[0].repr == Repr::kString) {
        return fail("multiply of these operands");
      }
      return checked(
          Code{fmt::format("({} * {})", a[0].text, a[1].text), a[0].repr, a[0].scale + a[1].scale}, type);
    }
    if (name == "divide" && a.size() == 2 && a[0].repr == Repr::kDouble && a[1].repr == Repr::kDouble) {
      return checked(Code{fmt::format("({} / {})", a[0].text, a[1].text), Repr::kDouble}, type);
    }
    if (name == "negate" && a.size() == 1 &&
        (a[0].repr == Repr::kInteger || a[0].repr == Repr::kDecimal || a[0].repr == Repr::kDouble)) {
      return checked(Code{fmt::format("(-{})", a[0].text), a[0].repr, a[0].scale}, type);
    }
    if ((name == "if" || name == "switch") && a.size() >= 3 && a.size() % 2 == 1) {
      // switch(c1, v1, c2, v2, ..., otherwise).
      const auto& otherwise = a.back();
      std::string text = otherwise.text;
      for (int64_t i = static_cast<int64_t>(a.size()) - 3; i >= 0; i -= 2) {
        if (a[i].repr != Repr::kBool || a[i + 1].repr != otherwise.repr ||
            a[i + 1].scale != otherwise.scale) {
          return fail("branches of different kinds");
        }
        text = fmt::format("({} ? {} : {})", a[i].text, a[i + 1].text, text);
      }
      return checked(Code{text, otherwise.repr, otherwise.scale}, type);
    }
    return fail(fmt::format("function {}", call.name()));
  }

  std::optional<Code> in(const core::CallTypedExpr& call) {
    if (call.inputs().size() < 2) {
      return fail("in without values");
    }
    auto value = expr(call.inputs()[0]);
    if (!value.has_value()) {
      return std::nullopt;
    }
    std::vector<Code> values;
    const auto* list = dynamic_cast<const core::ConstantTypedExpr*>(call.inputs()[1].get());
    if (call.inputs().size() == 2 && list != nullptr && list->type()->isArray()) {
      if (!list->hasValueVector()) {
        return fail("an in list without a vector");
      }
      const auto* array = list->valueVector()->wrappedVector()->as<ArrayVector>();
      const auto index = list->valueVector()->wrappedIndex(0);
      const auto& elements = array->elements();
      if (array->sizeAt(index) > static_cast<vector_size_t>(kMaxListValues)) {
        return fail("a long in list");
      }
      for (vector_size_t i = 0; i < array->sizeAt(index); ++i) {
        const auto row = array->offsetAt(index) + i;
        if (elements->isNullAt(row)) {
          return fail("a null in an in list");
        }
        auto element = expr(std::make_shared<core::ConstantTypedExpr>(BaseVector::wrapInConstant(1, row, elements)));
        if (!element.has_value()) {
          return std::nullopt;
        }
        values.push_back(std::move(*element));
      }
    } else {
      for (size_t i = 1; i < call.inputs().size(); ++i) {
        if (dynamic_cast<const core::ConstantTypedExpr*>(call.inputs()[i].get()) == nullptr) {
          return fail("in over values that are not constants");
        }
        auto element = expr(call.inputs()[i]);
        if (!element.has_value()) {
          return std::nullopt;
        }
        values.push_back(std::move(*element));
      }
    }
    std::vector<std::string> terms;
    for (const auto& v : values) {
      auto term = compare(*value, v, "==");
      if (!term.has_value()) {
        return std::nullopt;
      }
      terms.push_back(*term);
    }
    return Code{fmt::format("({})", fmt::join(terms, " || ")), Repr::kBool};
  }

  std::optional<Code> like(const core::CallTypedExpr& call) {
    if (call.inputs().size() != 2) {
      return fail("like with an escape");
    }
    const auto* pattern = dynamic_cast<const core::ConstantTypedExpr*>(call.inputs()[1].get());
    if (pattern == nullptr || pattern->isNull() || !pattern->type()->isVarchar()) {
      return fail("like with a pattern that is not a constant");
    }
    auto value = expr(call.inputs()[0]);
    if (!value.has_value()) {
      return std::nullopt;
    }
    if (value->repr != Repr::kString) {
      return fail("like of a value that is not a string");
    }
    const auto text = pattern->hasValueVector()
        ? std::string(pattern->valueVector()->as<SimpleVector<StringView>>()->valueAt(0))
        : pattern->value().value<std::string>();
    usesLike_ = true;
    return Code{fmt::format("like({}, {})", value->text, string(text)), Repr::kBool};
  }

  std::optional<Code> cast(const core::CastTypedExpr& cast) {
    if (cast.isTryCast()) {
      return fail("try_cast");
    }
    auto value = expr(cast.inputs()[0]);
    if (!value.has_value()) {
      return std::nullopt;
    }
    const auto& from = cast.inputs()[0]->type();
    const auto& to = cast.type();
    if (from->equivalent(*to)) {
      return value;
    }
    const auto repr = reprOf(to);
    if (to->isDate() || from->isDate() || !repr.has_value()) {
      return fail(fmt::format("cast from {} to {}", from->toString(), to->toString()));
    }
    if (*repr == Repr::kDouble && (value->repr == Repr::kInteger || value->repr == Repr::kDouble)) {
      return Code{fmt::format("((double){})", value->text), Repr::kDouble};
    }
    if (*repr == Repr::kDouble && value->repr == Repr::kDecimal) {
      return Code{fmt::format("((double){} / {:.1f})", value->text, std::pow(10.0, value->scale)), Repr::kDouble};
    }
    if (*repr == Repr::kInteger && value->repr == Repr::kInteger && to->cppSizeInBytes() >= from->cppSizeInBytes()) {
      return value;
    }
    if (*repr == Repr::kDecimal && value->repr == Repr::kInteger) {
      return Code{fmt::format("((__int128){} * {})", value->text, pow10(scaleOf(to))), Repr::kDecimal, scaleOf(to)};
    }
    if (*repr == Repr::kDecimal && value->repr == Repr::kDecimal && scaleOf(to) >= value->scale) {
      return rescale(*value, scaleOf(to));
    }
    return fail(fmt::format("cast from {} to {}", from->toString(), to->toString()));
  }

  const RowTypePtr input_;
  const Encodings* const encodings_;
  std::map<column_index_t, int32_t> parameters_;
  std::vector<int32_t> channels_;
  std::vector<std::string> types_;
  std::map<std::string, int32_t> strings_;
  std::string constants_;
  bool usesStrings_{false};
  bool usesLike_{false};
  std::string error_;
};

std::optional<Kind> kindOf(const core::AggregationNode::Aggregate& aggregate, bool hasInput) {
  const auto name = baseName(aggregate.call->name());
  if (name == "count") {
    return hasInput ? Kind::kCount : Kind::kCountRows;
  }
  if (!hasInput) {
    return std::nullopt;
  }
  if (name == "sum") {
    return Kind::kSum;
  }
  if (name == "avg") {
    return Kind::kAvg;
  }
  if (name == "min") {
    return Kind::kMin;
  }
  if (name == "max") {
    return Kind::kMax;
  }
  return std::nullopt;
}

/// The partial state an aggregate of 'kind' over 'type' keeps beside the
/// slot's row count, or nullopt for counts, which equal the row count since
/// rows with null inputs are skipped.
std::optional<MergeOp> mergeOp(Kind kind, const TypePtr& type) {
  const bool floating = type != nullptr && type->isDouble();
  switch (kind) {
    case Kind::kSum:
    case Kind::kAvg:
      return floating ? MergeOp::kAddDouble : MergeOp::kAddInt128;
    case Kind::kMin:
      return floating ? MergeOp::kMinDouble : MergeOp::kMinInteger;
    case Kind::kMax:
      return floating ? MergeOp::kMaxDouble : MergeOp::kMaxInteger;
    default:
      return std::nullopt;
  }
}

int32_t fieldCount(MergeOp op) {
  return op == MergeOp::kAddInt128 ? 2 : 1;
}

bool floatingField(MergeOp op) {
  return op == MergeOp::kAddDouble || op == MergeOp::kMinDouble || op == MergeOp::kMaxDouble;
}

/// Per aggregate, the merge op and the expression text identifying its
/// value, shared by aggregates of the same value, as (op, input).
using ValueKey = std::pair<MergeOp, std::string>;

std::vector<std::optional<ValueKey>> valueKeys(
    const std::vector<core::AggregationNode::Aggregate>& aggregates,
    const std::vector<core::TypedExprPtr>& inputs) {
  std::vector<std::optional<ValueKey>> out;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto kind = kindOf(aggregates[i], inputs[i] != nullptr);
    const auto op = kind.has_value() && inputs[i] ? mergeOp(*kind, inputs[i]->type()) : std::nullopt;
    out.push_back(op.has_value() ? std::optional(ValueKey{*op, inputs[i]->toString()}) : std::nullopt);
  }
  return out;
}

/// Matches % and _ in 'pattern' against 'value', with _ one UTF-8 character.
constexpr const char* kLike = R"(
__device__ int characterBytes(unsigned char lead) {
  return lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
}

__device__ bool like(cudf::string_view value, cudf::string_view pattern) {
  const char* s = value.data();
  const char* p = pattern.data();
  const int sn = value.size_bytes();
  const int pn = pattern.size_bytes();
  int si = 0;
  int pi = 0;
  int star = -1;
  int mark = 0;
  while (si < sn) {
    if (pi < pn && p[pi] == '_') {
      si += characterBytes(s[si]);
      ++pi;
    } else if (pi < pn && p[pi] == '%') {
      star = pi++;
      mark = si;
    } else if (pi < pn && p[pi] == s[si]) {
      ++si;
      ++pi;
    } else if (star >= 0) {
      pi = star + 1;
      mark += characterBytes(s[mark]);
      si = mark;
    } else {
      return false;
    }
  }
  while (pi < pn && p[pi] == '%') {
    ++pi;
  }
  return pi == pn && si == sn;
}
)";

constexpr const char* kAdd128 = R"(
__device__ void add128(int64_t* low, int64_t* high, __int128 value) {
  const unsigned long long old = (unsigned long long)*low;
  const unsigned long long sum = old + (unsigned long long)value;
  *low = (int64_t)sum;
  *high += (int64_t)(value >> 64) + (sum < old ? 1 : 0);
}
)";

const cudf::compiled_fragment& compile(const std::string& source) {
  static std::mutex mutex;
  static std::map<std::string, std::unique_ptr<cudf::compiled_fragment>> fragments;
  std::lock_guard<std::mutex> lock(mutex);
  auto& fragment = fragments[source];
  if (fragment == nullptr) {
    fragment = std::make_unique<cudf::compiled_fragment>(
        cudf::compile_cuda_to_lto_ir(source.c_str(), true, {}, {}, {}));
  }
  return *fragment;
}

/// 'column' as the integer of its width for dates and decimals, the types
/// the generated parameters take.
cudf::column_view asParameter(const cudf::column_view& column) {
  switch (column.type().id()) {
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
      return cudf::column_view(
          cudf::data_type{cudf::type_id::INT32},
          column.size(),
          column.head(),
          column.null_mask(),
          column.null_count(),
          column.offset());
    case cudf::type_id::DECIMAL64:
      return cudf::column_view(
          cudf::data_type{cudf::type_id::INT64},
          column.size(),
          column.head(),
          column.null_mask(),
          column.null_count(),
          column.offset());
    default:
      return column;
  }
}

std::vector<core::AggregationNode::Aggregate> withConstantInputs(
    std::vector<core::AggregationNode::Aggregate> aggregates) {
  for (auto& aggregate : aggregates) {
    std::vector<core::TypedExprPtr> inputs;
    for (const auto& type : aggregate.rawInputTypes) {
      inputs.push_back(std::make_shared<core::ConstantTypedExpr>(type, Variant::null(type->kind())));
    }
    aggregate.call = std::make_shared<core::CallTypedExpr>(
        aggregate.call->type(), std::move(inputs), aggregate.call->name());
  }
  return aggregates;
}

} // namespace

FusedAggregationNode::FusedAggregationNode(
    const core::PlanNodeId& id,
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const std::vector<std::string>& aggregateNames,
    const std::vector<Aggregate>& aggregates,
    core::PlanNodePtr source,
    std::vector<DenseKey> keys,
    RowTypePtr outputType,
    std::vector<core::TypedExprPtr> inputs,
    std::vector<core::TypedExprPtr> filters,
    std::vector<ColumnFilter> columnFilters,
    Encodings encodings)
    : DenseAggregationNode(
          id,
          groupingKeys,
          aggregateNames,
          withConstantInputs(aggregates),
          std::move(source),
          std::move(keys)),
      outputType_(std::move(outputType)),
      inputs_(std::move(inputs)),
      filters_(std::move(filters)),
      columnFilters_(std::move(columnFilters)),
      encodings_(std::move(encodings)) {
  VELOX_CHECK_EQ(inputs_.size(), aggregates.size());
}

std::optional<std::string> unfusable(
    const core::TypedExprPtr& expr,
    const RowTypePtr& input,
    const Encodings& encodings) {
  Generator generator(input, &encodings);
  if (!generator.expr(expr).has_value()) {
    return generator.error();
  }
  return std::nullopt;
}

std::optional<std::string> unfusable(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<Encoding>& encoding) {
  Encodings encodings;
  if (encoding.has_value()) {
    encodings.emplace("c", *encoding);
  }
  Generator generator(ROW({"c"}, {type}), &encodings);
  if (!generator.filter("c", filter).has_value()) {
    return generator.error();
  }
  return std::nullopt;
}

std::optional<std::string> unfusable(
    const core::AggregationNode::Aggregate& aggregate,
    const TypePtr& type) {
  if (aggregate.mask != nullptr || aggregate.distinct || !aggregate.sortingKeys.empty()) {
    return "a masked, distinct or ordered aggregate";
  }
  const auto kind = kindOf(aggregate, type != nullptr);
  if (!kind.has_value()) {
    return fmt::format("aggregate {}", aggregate.call->name());
  }
  if (type == nullptr || *kind == Kind::kCount) {
    return std::nullopt;
  }
  if (type->isDouble() || type->isShortDecimal() || type->isTinyint() || type->isSmallint() ||
      type->isInteger() || type->isBigint() || type->isDate()) {
    if (type->isDate() && (*kind == Kind::kSum || *kind == Kind::kAvg)) {
      return "a sum of dates";
    }
    return std::nullopt;
  }
  if (type->isLongDecimal() && (*kind == Kind::kSum || *kind == Kind::kAvg)) {
    return std::nullopt;
  }
  return fmt::format("{} of {}", aggregate.call->name(), type->toString());
}

int64_t fusedStateBytes(
    int64_t slots,
    const std::vector<core::AggregationNode::Aggregate>& aggregates,
    const std::vector<core::TypedExprPtr>& inputs) {
  std::set<ValueKey> distinct;
  for (const auto& key : valueKeys(aggregates, inputs)) {
    if (key.has_value()) {
      distinct.insert(*key);
    }
  }
  int64_t fields = 1;
  for (const auto& key : distinct) {
    fields += fieldCount(key.first);
  }
  return slots * fields * static_cast<int64_t>(sizeof(int64_t));
}

FusedReduction::FusedReduction(const FusedAggregationNode& node, const RowTypePtr& input) {
  Generator generator(input, &node.encodings());
  const auto& aggregates = node.aggregates();
  const auto& inputs = node.inputs();
  const auto keys = valueKeys(aggregates, inputs);
  const auto groupingKeys = node.groupingKeys().size();

  // Values first: their parameters, then the keys', then the filters'.
  std::map<ValueKey, int32_t> fields;
  std::vector<std::pair<MergeOp, std::string>> values;
  fieldsPerSlot_ = 1;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto kind = kindOf(aggregates[i], inputs[i] != nullptr);
    VELOX_CHECK(kind.has_value(), "Not a fused aggregate: {}", aggregates[i].call->toString());
    aggregates_.push_back(DenseGroupby::Aggregate{
        *kind,
        -1,
        inputs[i] ? inputs[i]->type() : nullptr,
        node.outputType()->childAt(groupingKeys + i)});
    if (!keys[i].has_value()) {
      values_.push_back(std::nullopt);
      continue;
    }
    auto [it, added] = fields.emplace(*keys[i], fieldsPerSlot_);
    if (added) {
      auto code = generator.expr(inputs[i]);
      VELOX_CHECK(code.has_value(), "Cannot fuse {}: {}", inputs[i]->toString(), generator.error());
      std::string text = code->text;
      if (code->repr == Repr::kDouble || keys[i]->first == MergeOp::kAddInt128) {
        text = fmt::format("{}({})", code->repr == Repr::kDouble ? "(double)" : "(__int128)", text);
      } else {
        text = fmt::format("(int64_t)({})", text);
      }
      values.emplace_back(keys[i]->first, text);
      fieldsPerSlot_ += fieldCount(keys[i]->first);
    }
    values_.push_back(Value{keys[i]->first, it->second});
  }

  // The slot from the keys.
  std::string slot = "  int64_t slot = 0;\n";
  slots_ = 1;
  for (const auto& range : node.keys()) {
    slots_ *= range.high - range.low + 1;
  }
  {
    int64_t stride = 1;
    std::vector<std::string> terms(groupingKeys);
    for (int64_t k = static_cast<int64_t>(groupingKeys) - 1; k >= 0; --k) {
      const auto& range = node.keys()[k];
      const auto code = generator.column(node.groupingKeys()[k]->name());
      VELOX_CHECK(code.has_value() && code->repr == Repr::kInteger, "Key {} is not an integer", node.groupingKeys()[k]->name());
      const auto size = range.high - range.low + 1;
      terms[k] = fmt::format(
          "  {{\n    const int64_t offset = {} - {};\n"
          "    if (offset < 0 || offset >= {}) {{\n      return errc::USER_ERROR_0;\n    }}\n"
          "    slot += offset * {};\n  }}\n",
          code->text,
          int64Literal(range.low),
          size,
          stride);
      stride *= size;
    }
    for (const auto& term : terms) {
      slot += term;
    }
  }

  std::vector<std::string> conditions;
  for (const auto& filter : node.filters()) {
    auto code = generator.expr(filter);
    VELOX_CHECK(code.has_value() && code->repr == Repr::kBool, "Cannot fuse filter {}: {}", filter->toString(), generator.error());
    conditions.push_back(code->text);
  }
  for (const auto& filter : node.columnFilters()) {
    auto code = generator.filter(filter.column, *filter.filter);
    VELOX_CHECK(code.has_value(), "Cannot fuse filter on {}: {}", filter.column, generator.error());
    conditions.push_back(*code);
  }
  filtered_ = !conditions.empty();
  // A reduction without column inputs sees no rows, so count(*) alone takes
  // one column it does not read.
  if (generator.channels().empty()) {
    VELOX_CHECK_GT(input->size(), 0);
    VELOX_CHECK(generator.column(input->nameOf(0)).has_value(), "Cannot pass {}: {}", input->nameOf(0), generator.error());
  }
  channels_ = generator.channels();

  // State field types and initial values, slot by slot.
  std::vector<std::string> fieldTypes;
  std::vector<std::string> initial;
  for (int64_t s = 0; s < slots_; ++s) {
    stateTypes_.push_back(cudf::data_type{cudf::type_id::INT64});
    fieldTypes.push_back("int64_t");
    initial.push_back("0");
    for (const auto& [op, text] : values) {
      for (int32_t f = 0; f < fieldCount(op); ++f) {
        const bool floating = floatingField(op);
        stateTypes_.push_back(cudf::data_type{floating ? cudf::type_id::FLOAT64 : cudf::type_id::INT64});
        fieldTypes.push_back(floating ? "double" : "int64_t");
        switch (op) {
          case MergeOp::kMinInteger:
            initial.push_back(int64Literal(std::numeric_limits<int64_t>::max()));
            break;
          case MergeOp::kMaxInteger:
            initial.push_back(int64Literal(std::numeric_limits<int64_t>::min()));
            break;
          case MergeOp::kMinDouble:
            initial.push_back("__longlong_as_double(0x7ff0000000000000LL)");
            break;
          case MergeOp::kMaxDouble:
            initial.push_back("__longlong_as_double((long long)0xfff0000000000000ULL)");
            break;
          default:
            initial.push_back(floating ? "0.0" : "0");
        }
      }
    }
  }
  const auto stateCount = stateTypes_.size();
  auto list = [&](const std::string& prefix, bool pointers) {
    std::vector<std::string> out;
    for (size_t i = 0; i < stateCount; ++i) {
      out.push_back(fmt::format("{}{} {}{}", fieldTypes[i], pointers ? "*" : "", prefix, i));
    }
    return fmt::format("{}", fmt::join(out, ", "));
  };
  const auto parameters = generator.parameters();
  const auto comma = parameters.empty() ? "" : ", ";

  std::string source =
      "#include <cudf/errc.hpp>\n#include <cudf/types.hpp>\n#include <cuda/std/cstdint>\n";
  if (generator.usesStrings()) {
    source += "#include <cudf/strings/string_view.cuh>\n";
  }
  source += "\nusing cudf::errc;\nusing cudf::size_type;\nusing namespace cuda::std;\n\n";
  source += generator.prelude();
  if (generator.usesLike()) {
    source += kLike;
  }
  source += kAdd128;
  source += "\nextern \"C\" {\n";
  if (filtered_) {
    source += fmt::format(
        "__device__ errc fused_filter(size_type, bool* keep{}{}) {{\n  *keep = {};\n  return errc::SUCCESS;\n}}\n",
        comma,
        parameters,
        fmt::join(conditions, " && "));
  }
  source += fmt::format("__device__ errc fused_init({}) {{\n", list("s", true));
  for (size_t i = 0; i < stateCount; ++i) {
    source += fmt::format("  *s{} = {};\n", i, initial[i]);
  }
  source += "  return errc::SUCCESS;\n}\n";

  source += fmt::format("__device__ errc fused_update(size_type, {}{}{}) {{\n", list("s", true), comma, parameters);
  source += slot;
  for (size_t v = 0; v < values.size(); ++v) {
    source += fmt::format(
        "  const {} v{} = {};\n",
        values[v].first == MergeOp::kAddInt128 ? "__int128" : floatingField(values[v].first) ? "double" : "int64_t",
        v,
        values[v].second);
  }
  auto update = [&](int64_t s) {
    std::string body;
    int32_t field = static_cast<int32_t>(s * fieldsPerSlot_);
    body += fmt::format("      *s{} += 1;\n", field++);
    for (size_t v = 0; v < values.size(); ++v) {
      switch (values[v].first) {
        case MergeOp::kAddInt128:
          body += fmt::format("      add128(s{}, s{}, v{});\n", field, field + 1, v);
          break;
        case MergeOp::kAddDouble:
          body += fmt::format("      *s{} += v{};\n", field, v);
          break;
        case MergeOp::kMinInteger:
        case MergeOp::kMinDouble:
          body += fmt::format("      *s{0} = v{1} < *s{0} ? v{1} : *s{0};\n", field, v);
          break;
        case MergeOp::kMaxInteger:
        case MergeOp::kMaxDouble:
          body += fmt::format("      *s{0} = v{1} > *s{0} ? v{1} : *s{0};\n", field, v);
          break;
        default:
          break;
      }
      field += fieldCount(values[v].first);
    }
    return body;
  };
  if (slots_ == 1) {
    source += "  {\n" + update(0) + "  }\n";
  } else {
    source += "  switch (slot) {\n";
    for (int64_t s = 0; s < slots_; ++s) {
      source += fmt::format("    case {}: {{\n{}      break;\n    }}\n", s, update(s));
    }
    source += "  }\n";
  }
  source += "  return errc::SUCCESS;\n}\n";

  source += fmt::format("__device__ errc fused_merge({}, {}) {{\n", list("d", true), list("s", false));
  for (int64_t s = 0; s < slots_; ++s) {
    int32_t field = static_cast<int32_t>(s * fieldsPerSlot_);
    source += fmt::format("  *d{0} += s{0};\n", field++);
    for (const auto& [op, text] : values) {
      switch (op) {
        case MergeOp::kAddInt128:
          source += fmt::format(
              "  add128(d{0}, d{1}, (__int128)(((unsigned __int128)(unsigned long long)s{1} << 64) | (unsigned long long)s{0}));\n",
              field,
              field + 1);
          break;
        case MergeOp::kAddDouble:
          source += fmt::format("  *d{0} += s{0};\n", field);
          break;
        case MergeOp::kMinInteger:
        case MergeOp::kMinDouble:
          source += fmt::format("  *d{0} = s{0} < *d{0} ? s{0} : *d{0};\n", field);
          break;
        case MergeOp::kMaxInteger:
        case MergeOp::kMaxDouble:
          source += fmt::format("  *d{0} = s{0} > *d{0} ? s{0} : *d{0};\n", field);
          break;
        default:
          break;
      }
      field += fieldCount(op);
    }
  }
  source += "  return errc::SUCCESS;\n}\n";

  source += fmt::format("__device__ errc fused_finalize({}, {}) {{\n", list("o", true), list("s", false));
  for (size_t i = 0; i < stateCount; ++i) {
    source += fmt::format("  *o{0} = s{0};\n", i);
  }
  source += "  return errc::SUCCESS;\n}\n}\n";
  source_ = std::move(source);
  fragment_ = &compile(source_);
}

void FusedReduction::add(
    const std::vector<cudf::column_view>& input,
    DenseGroupby& into,
    cuda::stream_ref stream) {
  if (input.empty() || input[0].size() == 0) {
    return;
  }
  std::vector<cudf::reduce_input> inputs;
  for (const auto channel : channels_) {
    const auto& column = input[channel];
    VELOX_CHECK_EQ(column.null_count(), 0, "A fused aggregation takes no nulls");
    inputs.emplace_back(asParameter(column));
  }
  const cudf::lto_fragment fragments[] = {{cudf::lto_binary_type::LTO_IR, fragment_->view()}};
  const cudf::reduce_udf udf{
      fragments, "fused_init", "fused_update", "fused_merge", "fused_finalize", filtered_ ? "fused_filter" : ""};
  auto results = cudf::reduce_lto(
      udf,
      cudf::null_aware::NO,
      inputs,
      stateTypes_,
      stateTypes_,
      stream,
      cudf::memory_resources(get_temp_mr(), get_temp_mr()));
  std::vector<dense::Merge> merges;
  auto field = [&](int64_t slot, int32_t offset) {
    return results[slot * fieldsPerSlot_ + offset]->view().head();
  };
  for (int64_t s = 0; s < slots_; ++s) {
    const void* count = field(s, 0);
    merges.push_back({MergeOp::kAddCount, count, nullptr, into.rowsAt(s)});
    for (size_t a = 0; a < aggregates_.size(); ++a) {
      if (auto* target = into.countAt(a, s)) {
        merges.push_back({MergeOp::kAddCount, count, nullptr, target});
      }
      if (values_[a].has_value()) {
        const auto& value = *values_[a];
        merges.push_back({
            value.op,
            field(s, value.field),
            value.op == MergeOp::kAddInt128 ? field(s, value.field + 1) : nullptr,
            into.valueAt(a, s)});
      }
    }
  }
  into.merge(merges, stream);
}

namespace {

/// The cuDF type of a transform output of 'type' and the device type the
/// generated transform writes it as.
std::optional<std::pair<cudf::data_type, std::string>> outputOf(const TypePtr& type) {
  using cudf::type_id;
  if (type->isDate()) {
    return std::pair{cudf::data_type{type_id::TIMESTAMP_DAYS}, "int32_t"};
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return std::pair{cudf::data_type{type_id::BOOL8}, "bool"};
    case TypeKind::TINYINT:
      return std::pair{cudf::data_type{type_id::INT8}, "int8_t"};
    case TypeKind::SMALLINT:
      return std::pair{cudf::data_type{type_id::INT16}, "int16_t"};
    case TypeKind::INTEGER:
      return std::pair{cudf::data_type{type_id::INT32}, "int32_t"};
    case TypeKind::BIGINT:
      if (type->isShortDecimal()) {
        return std::pair{cudf::data_type{type_id::DECIMAL64, -scaleOf(type)}, "int64_t"};
      }
      return std::pair{cudf::data_type{type_id::INT64}, "int64_t"};
    case TypeKind::HUGEINT:
      if (type->isLongDecimal()) {
        return std::pair{cudf::data_type{type_id::DECIMAL128, -scaleOf(type)}, "__int128"};
      }
      return std::nullopt;
    case TypeKind::DOUBLE:
      return std::pair{cudf::data_type{type_id::FLOAT64}, "double"};
    default:
      return std::nullopt;
  }
}

std::string transformHeader(const Generator& generator) {
  std::string source = "#include <cuda/std/cstdint>\n";
  if (generator.usesStrings()) {
    source += "#include <cudf/strings/string_view.cuh>\n";
  }
  source += "\nusing namespace cuda::std;\n\n" + generator.prelude();
  if (generator.usesLike()) {
    source += kLike;
  }
  return source;
}

/// The mask kernel's source and input channels, or the reason there is none.
struct Generated {
  std::string source;
  std::vector<int32_t> channels;
  std::vector<cudf::data_type> outputs;
  std::string error;
};

Generated maskSource(
    const RowTypePtr& input,
    const Encodings& encodings,
    const std::vector<core::TypedExprPtr>& filters,
    const std::vector<ColumnFilter>& columnFilters) {
  Generator generator(input, &encodings);
  std::vector<std::string> conditions;
  for (const auto& filter : filters) {
    auto code = generator.expr(filter);
    if (!code.has_value()) {
      return {.error = generator.error()};
    }
    if (code->repr != Repr::kBool) {
      return {.error = fmt::format("filter {} is not boolean", filter->toString())};
    }
    conditions.push_back(code->text);
  }
  for (const auto& filter : columnFilters) {
    auto code = generator.filter(filter.column, *filter.filter);
    if (!code.has_value()) {
      return {.error = generator.error()};
    }
    conditions.push_back(*code);
  }
  if (generator.channels().empty() && !generator.column(input->nameOf(0)).has_value()) {
    return {.error = generator.error()};
  }
  auto source = transformHeader(generator) +
      fmt::format(
                    "extern \"C\" __device__ int transform(bool* keep, {}) {{\n  *keep = {};\n  return 0;\n}}\n",
                    generator.parameters(),
                    fmt::join(conditions, " && "));
  return {std::move(source), generator.channels(), {cudf::data_type{cudf::type_id::BOOL8}}, ""};
}

Generated projectSource(
    const RowTypePtr& input,
    const Encodings& encodings,
    const std::vector<core::TypedExprPtr>& projections) {
  Generator generator(input, &encodings);
  std::vector<std::string> outputs;
  std::vector<std::string> body;
  std::vector<cudf::data_type> types;
  for (size_t i = 0; i < projections.size(); ++i) {
    const auto& type = projections[i]->type();
    const auto output = outputOf(type);
    if (!output.has_value()) {
      return {.error = fmt::format("a projection of type {}", type->toString())};
    }
    auto code = generator.typed(projections[i]);
    if (!code.has_value()) {
      return {.error = generator.error()};
    }
    types.push_back(output->first);
    outputs.push_back(fmt::format("{}* o{}", output->second, i));
    body.push_back(fmt::format("  *o{} = ({})({});\n", i, output->second, code->text));
  }
  if (generator.channels().empty() && !generator.column(input->nameOf(0)).has_value()) {
    return {.error = generator.error()};
  }
  auto source = transformHeader(generator) +
      fmt::format(
                    "extern \"C\" __device__ int transform({}, {}) {{\n{}  return 0;\n}}\n",
                    fmt::join(outputs, ", "),
                    generator.parameters(),
                    fmt::join(body, ""));
  return {std::move(source), generator.channels(), std::move(types), ""};
}

std::unique_ptr<cudf::table> runTransform(
    const cudf::compiled_fragment& fragment,
    const std::vector<int32_t>& channels,
    const std::vector<cudf::data_type>& types,
    const std::vector<cudf::column_view>& input,
    cudf::size_type rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::transform_input> inputs;
  for (const auto channel : channels) {
    VELOX_CHECK_EQ(input[channel].null_count(), 0, "A fused transform takes no nulls");
    inputs.emplace_back(asParameter(input[channel]));
  }
  std::vector<cudf::transform_output> outputs;
  for (const auto& type : types) {
    outputs.push_back({type, cudf::output_nullability::ALL_VALID});
  }
  return cudf::transform_lto(
      fragment.view(),
      cudf::lto_binary_type::LTO_IR,
      cudf::null_aware::NO,
      std::nullopt,
      inputs,
      outputs,
      {},
      rows,
      stream,
      mr);
}

} // namespace

std::optional<std::string> FusedTransform::unfusable(
    const RowTypePtr& input,
    const Encodings& encodings,
    const std::vector<core::TypedExprPtr>& filters,
    const std::vector<ColumnFilter>& columnFilters,
    const std::vector<core::TypedExprPtr>& projections) {
  if (!filters.empty() || !columnFilters.empty()) {
    auto mask = maskSource(input, encodings, filters, columnFilters);
    if (!mask.error.empty()) {
      return mask.error;
    }
  }
  if (!projections.empty()) {
    auto project = projectSource(input, encodings, projections);
    if (!project.error.empty()) {
      return project.error;
    }
  }
  return std::nullopt;
}

FusedTransform::FusedTransform(
    const RowTypePtr& input,
    const Encodings& encodings,
    const std::vector<core::TypedExprPtr>& filters,
    const std::vector<ColumnFilter>& columnFilters,
    const std::vector<core::TypedExprPtr>& projections) {
  if (!filters.empty() || !columnFilters.empty()) {
    auto mask = maskSource(input, encodings, filters, columnFilters);
    VELOX_CHECK(mask.error.empty(), "Cannot fuse the filters: {}", mask.error);
    maskSource_ = std::move(mask.source);
    maskChannels_ = std::move(mask.channels);
    maskFragment_ = &compile(maskSource_);
  }
  if (!projections.empty()) {
    auto project = projectSource(input, encodings, projections);
    VELOX_CHECK(project.error.empty(), "Cannot fuse the projections: {}", project.error);
    projectSource_ = std::move(project.source);
    projectChannels_ = std::move(project.channels);
    outputTypes_ = std::move(project.outputs);
    projectFragment_ = &compile(projectSource_);
  }
}

std::unique_ptr<cudf::column> FusedTransform::mask(
    const std::vector<cudf::column_view>& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_NOT_NULL(maskFragment_);
  const auto rows = input[maskChannels_[0]].size();
  auto table = runTransform(
      *maskFragment_, maskChannels_, {cudf::data_type{cudf::type_id::BOOL8}}, input, rows, stream, mr);
  return std::move(table->release()[0]);
}

std::vector<std::unique_ptr<cudf::column>> FusedTransform::project(
    const std::vector<cudf::column_view>& input,
    cudf::size_type rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  if (projectFragment_ == nullptr) {
    return {};
  }
  return runTransform(*projectFragment_, projectChannels_, outputTypes_, input, rows, stream, mr)->release();
}

} // namespace facebook::velox::cudf_velox
