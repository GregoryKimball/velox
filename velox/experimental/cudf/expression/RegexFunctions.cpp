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

#include "velox/experimental/cudf/expression/RegexFunctions.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/Re2Functions.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include "velox/experimental/cudf/CudfNoDefaults.h"

#include <fmt/format.h>
#include <re2/re2.h>

#include <memory>

namespace facebook::velox::cudf_velox {
namespace {

using cudf::strings::regex_flags;

// The largest repeat count libcudf accepts in '{n,m}'.
constexpr int32_t kMaxRepeat = 999;

// Rewrites an RE2 pattern into libcudf syntax by recursive descent over the
// RE2 grammar. Every construct is either translated to libcudf syntax with
// the same meaning or rejected, so that a translated pattern matches the same
// text as RE2:
// - '$' becomes '\Z'. In libcudf '$' also matches before a final '\n', but in
//   RE2 it only matches at the end of the text.
// - '\d', '\w' and '\s' become explicit ASCII classes. RE2's '\s' is
//   [\t\n\f\r ], unlike libcudf's.
// - Literal punctuation is written as '\xHH', which libcudf always reads as a
//   literal, so RE2 and libcudf need not agree on which characters need a
//   backslash.
// - Range ends inside a class are written raw because libcudf does not decode
//   escapes there.
class Re2ToCudfTranslator {
 public:
  Re2ToCudfTranslator(
      std::string_view pattern,
      RegexCapture capture,
      int32_t captureGroup)
      : pattern_(pattern), capture_(capture), captureGroup_(captureGroup) {}

  std::optional<CudfRegexPattern> translate() {
    auto flags = regex_flags::ASCII;
    bool hasModifiers = false;
    // libcudf has no inline mode modifiers, but modifiers at the start of the
    // pattern apply to all of it, the same as libcudf's regex flags.
    while (pattern_.substr(pos_, 2) == "(?") {
      size_t end = pos_ + 2;
      auto modifierFlags = flags;
      while (end < pattern_.size() &&
             (pattern_[end] == 'i' || pattern_[end] == 's')) {
        modifierFlags = modifierFlags |
            (pattern_[end] == 'i' ? regex_flags::IGNORECASE
                                  : regex_flags::DOTALL);
        ++end;
      }
      if (end == pos_ + 2 || end >= pattern_.size() || pattern_[end] != ')') {
        break;
      }
      flags = modifierFlags;
      hasModifiers = true;
      pos_ = end + 1;
    }
    ignoreCase_ = cudf::strings::is_ignorecase(flags);

    auto node = parseAlternation();
    if (!node.has_value() || pos_ != pattern_.size()) {
      return std::nullopt;
    }
    // RE2 folds case with Unicode rules: under (?i), 'k' also matches the
    // Kelvin sign and 's' the long s. libcudf only swaps ASCII case.
    if (ignoreCase_ && foldSensitive_) {
      return std::nullopt;
    }
    if (capture_ == RegexCapture::kSingle && captureGroup_ == 0) {
      node->text = "(" + node->text + ")";
    }
    return CudfRegexPattern{
        .pattern = std::move(node->text),
        .flags = flags,
        .numGroups = numGroups_,
        .canMatchEmpty = node->nullable,
        .hasModifiers = hasModifiers,
        .hasWordBoundary = hasWordBoundary_,
    };
  }

 private:
  struct Node {
    std::string text;
    // Can match the empty string.
    bool nullable;
    // Matches only the empty string, as an anchor does.
    bool zeroWidth;
  };

  bool atEnd() const {
    return pos_ >= pattern_.size();
  }

  char peek(size_t offset = 0) const {
    return pos_ + offset < pattern_.size() ? pattern_[pos_ + offset] : '\0';
  }

  static bool isAsciiAlnum(char32_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z');
  }

  void noteFoldSensitive(char32_t c) {
    if (c >= 0x80 || c == 'k' || c == 'K' || c == 's' || c == 'S') {
      foldSensitive_ = true;
    }
  }

  static void appendUtf8(std::string& out, char32_t c) {
    if (c < 0x80) {
      out += static_cast<char>(c);
    } else if (c < 0x800) {
      out += static_cast<char>(0xC0 | (c >> 6));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      out += static_cast<char>(0xE0 | (c >> 12));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (c >> 18));
      out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    }
  }

  // Writes one character as a libcudf literal, inside or outside a class.
  void appendLiteral(std::string& out, char32_t c) {
    noteFoldSensitive(c);
    if (c >= 0x80 || isAsciiAlnum(c)) {
      appendUtf8(out, c);
    } else {
      out += fmt::format("\\x{:02x}", static_cast<uint32_t>(c));
    }
  }

  // Reads one UTF-8 character of the pattern.
  std::optional<char32_t> readUtf8() {
    const auto lead = static_cast<unsigned char>(pattern_[pos_]);
    int32_t length;
    char32_t c;
    if (lead < 0x80) {
      length = 1;
      c = lead;
    } else if ((lead & 0xE0) == 0xC0) {
      length = 2;
      c = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
      length = 3;
      c = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
      length = 4;
      c = lead & 0x07;
    } else {
      return std::nullopt;
    }
    if (pos_ + length > pattern_.size()) {
      return std::nullopt;
    }
    for (int32_t i = 1; i < length; ++i) {
      const auto next = static_cast<unsigned char>(pattern_[pos_ + i]);
      if ((next & 0xC0) != 0x80) {
        return std::nullopt;
      }
      c = (c << 6) | (next & 0x3F);
    }
    pos_ += length;
    return c;
  }

  // Reads the code point of '\xHH' or '\x{H...}' with pos_ past the 'x'.
  std::optional<char32_t> readHexEscape() {
    auto hexValue = [](char c) -> int32_t {
      if (c >= '0' && c <= '9') {
        return c - '0';
      }
      if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
      }
      if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
      }
      return -1;
    };
    char32_t value = 0;
    if (peek() == '{') {
      ++pos_;
      int32_t digits = 0;
      while (!atEnd() && peek() != '}') {
        const auto digit = hexValue(peek());
        if (digit < 0 || ++digits > 6) {
          return std::nullopt;
        }
        value = value * 16 + digit;
        ++pos_;
      }
      if (atEnd() || digits == 0) {
        return std::nullopt;
      }
      ++pos_;
    } else {
      for (int32_t i = 0; i < 2; ++i) {
        const auto digit = hexValue(peek());
        if (digit < 0) {
          return std::nullopt;
        }
        value = value * 16 + digit;
        ++pos_;
      }
    }
    // libcudf ends a pattern at a NUL character.
    if (value == 0 || value > 0x10FFFF) {
      return std::nullopt;
    }
    return value;
  }

  // Reads an escape that stands for one character, with pos_ past the
  // backslash. Returns std::nullopt for anything else.
  std::optional<char32_t> readCharacterEscape() {
    if (atEnd()) {
      return std::nullopt;
    }
    const char c = pattern_[pos_++];
    switch (c) {
      case 'n':
        return '\n';
      case 't':
        return '\t';
      case 'r':
        return '\r';
      case 'f':
        return '\f';
      case 'a':
        return '\a';
      case 'v':
        return '\v';
      case 'x':
        return readHexEscape();
      default:
        break;
    }
    const auto u = static_cast<unsigned char>(c);
    if (u > 0x20 && u < 0x7F && !isAsciiAlnum(u)) {
      return static_cast<char32_t>(u);
    }
    return std::nullopt;
  }

  std::optional<Node> parseAlternation() {
    auto first = parseConcat();
    if (!first.has_value()) {
      return std::nullopt;
    }
    Node node = std::move(*first);
    while (peek() == '|') {
      ++pos_;
      auto next = parseConcat();
      if (!next.has_value()) {
        return std::nullopt;
      }
      node.text += "|" + next->text;
      node.nullable = node.nullable || next->nullable;
      node.zeroWidth = node.zeroWidth && next->zeroWidth;
    }
    return node;
  }

  // Parses a non-empty sequence. libcudf's behavior for empty alternatives
  // and groups is undefined, so 'a|', '(|a)' and '()' are rejected.
  std::optional<Node> parseConcat() {
    Node node{"", true, true};
    bool empty = true;
    while (!atEnd() && peek() != '|' && peek() != ')') {
      auto item = parseRepeat();
      if (!item.has_value()) {
        return std::nullopt;
      }
      node.text += item->text;
      node.nullable = node.nullable && item->nullable;
      node.zeroWidth = node.zeroWidth && item->zeroWidth;
      empty = false;
    }
    if (empty) {
      return std::nullopt;
    }
    return node;
  }

  // Parses '{n}', '{n,}' or '{n,m}'. Returns false, leaving pos_ alone, if
  // the brace does not start a valid repeat, in which case RE2 reads it as a
  // literal.
  bool parseBraces(int32_t& min, std::optional<int32_t>& max) {
    size_t p = pos_ + 1;
    auto readNumber = [&](int32_t& value) {
      const size_t start = p;
      value = 0;
      while (p < pattern_.size() && pattern_[p] >= '0' && pattern_[p] <= '9') {
        value = std::min(value * 10 + (pattern_[p] - '0'), 100000);
        ++p;
      }
      return p > start;
    };
    if (!readNumber(min)) {
      return false;
    }
    if (p < pattern_.size() && pattern_[p] == ',') {
      ++p;
      int32_t value;
      if (readNumber(value)) {
        max = value;
      } else {
        max = std::nullopt;
      }
    } else {
      max = min;
    }
    if (p >= pattern_.size() || pattern_[p] != '}') {
      return false;
    }
    pos_ = p + 1;
    return true;
  }

  bool atRepeat() const {
    const char c = peek();
    if (c == '*' || c == '+' || c == '?') {
      return true;
    }
    if (c != '{') {
      return false;
    }
    auto copy = *this;
    int32_t min;
    std::optional<int32_t> max;
    return copy.parseBraces(min, max);
  }

  std::optional<Node> parseRepeat() {
    auto atom = parseAtom();
    if (!atom.has_value() || !atRepeat()) {
      return atom;
    }
    // libcudf's behavior for repeating an anchor, or a group that can match
    // the empty string, is not known to agree with RE2.
    if (atom->zeroWidth || atom->nullable) {
      return std::nullopt;
    }
    Node node = std::move(*atom);
    const char c = peek();
    if (c == '{') {
      int32_t min;
      std::optional<int32_t> max;
      parseBraces(min, max);
      if (min > kMaxRepeat || (max.has_value() && *max > kMaxRepeat) ||
          (max.has_value() && *max < min)) {
        return std::nullopt;
      }
      node.text += max == min
          ? fmt::format("{{{}}}", min)
          : (max.has_value() ? fmt::format("{{{},{}}}", min, *max)
                             : fmt::format("{{{},}}", min));
      node.nullable = min == 0;
    } else {
      ++pos_;
      node.text += c;
      node.nullable = c != '+';
    }
    if (peek() == '?') {
      ++pos_;
      node.text += '?';
    }
    if (atRepeat()) {
      return std::nullopt;
    }
    return node;
  }

  std::optional<Node> parseAtom() {
    const char c = peek();
    switch (c) {
      case '(':
        return parseGroup();
      case '[':
        return parseClass();
      case '.':
        ++pos_;
        return Node{".", false, false};
      case '^':
        ++pos_;
        return Node{"\\A", true, true};
      case '$':
        ++pos_;
        return Node{"\\Z", true, true};
      case '\\':
        ++pos_;
        return parseEscape();
      case '*':
      case '+':
      case '?':
        return std::nullopt;
      default:
        break;
    }
    auto literal = readUtf8();
    if (!literal.has_value() || *literal == 0) {
      return std::nullopt;
    }
    Node node{"", false, false};
    appendLiteral(node.text, *literal);
    return node;
  }

  std::optional<Node> parseGroup() {
    ++pos_;
    std::string open;
    if (peek() == '?') {
      // Only '(?:' is supported. Named groups, lookarounds, and mode
      // modifiers after the start of the pattern stay on the CPU.
      if (peek(1) != ':') {
        return std::nullopt;
      }
      pos_ += 2;
      open = "(?:";
    } else {
      const int32_t group = ++numGroups_;
      const bool keep = capture_ == RegexCapture::kAll ||
          (capture_ == RegexCapture::kSingle && captureGroup_ == group);
      open = keep ? "(" : "(?:";
    }
    auto body = parseAlternation();
    if (!body.has_value() || peek() != ')') {
      return std::nullopt;
    }
    ++pos_;
    body->text = open + body->text + ")";
    return body;
  }

  std::optional<Node> parseEscape() {
    if (atEnd()) {
      return std::nullopt;
    }
    const char c = peek();
    static constexpr std::string_view kDigit = "[0-9]";
    static constexpr std::string_view kNotDigit = "[^0-9]";
    static constexpr std::string_view kWord = "[0-9A-Za-z_]";
    static constexpr std::string_view kNotWord = "[^0-9A-Za-z_]";
    static constexpr std::string_view kSpace = "[\\x09\\x0a\\x0c\\x0d\\x20]";
    static constexpr std::string_view kNotSpace =
        "[^\\x09\\x0a\\x0c\\x0d\\x20]";
    switch (c) {
      case 'd':
        ++pos_;
        return Node{std::string(kDigit), false, false};
      case 'D':
        ++pos_;
        return Node{std::string(kNotDigit), false, false};
      case 'w':
        ++pos_;
        foldSensitive_ = true;
        return Node{std::string(kWord), false, false};
      case 'W':
        ++pos_;
        foldSensitive_ = true;
        return Node{std::string(kNotWord), false, false};
      case 's':
        ++pos_;
        return Node{std::string(kSpace), false, false};
      case 'S':
        ++pos_;
        return Node{std::string(kNotSpace), false, false};
      case 'b':
      case 'B':
        ++pos_;
        hasWordBoundary_ = true;
        foldSensitive_ = true;
        return Node{std::string("\\") + c, true, true};
      case 'A':
        ++pos_;
        return Node{"\\A", true, true};
      case 'z':
        ++pos_;
        return Node{"\\Z", true, true};
      default:
        break;
    }
    auto literal = readCharacterEscape();
    if (!literal.has_value()) {
      return std::nullopt;
    }
    Node node{"", false, false};
    appendLiteral(node.text, *literal);
    return node;
  }

  // Reads one class member: a character, or for a shorthand, its ranges.
  // Returns false for anything libcudf cannot express.
  bool readClassMember(std::optional<char32_t>& character, std::string& set) {
    character.reset();
    if (atEnd()) {
      return false;
    }
    const char c = peek();
    // RE2 reads '[:alpha:]' as a POSIX class; libcudf, and Spark's
    // java.util.regex, read nested brackets differently.
    if (c == '[') {
      return false;
    }
    if (c != '\\') {
      auto literal = readUtf8();
      if (!literal.has_value() || *literal == 0) {
        return false;
      }
      character = *literal;
      return true;
    }
    ++pos_;
    switch (peek()) {
      case 'd':
        ++pos_;
        set = "0-9";
        return true;
      case 'w':
        ++pos_;
        foldSensitive_ = true;
        set = "0-9A-Za-z_";
        return true;
      case 's':
        ++pos_;
        set = "\\x09\\x0a\\x0c\\x0d\\x20";
        return true;
      default:
        break;
    }
    character = readCharacterEscape();
    return character.has_value();
  }

  std::optional<Node> parseClass() {
    ++pos_;
    std::string text = "[";
    if (peek() == '^') {
      ++pos_;
      text += '^';
    }
    // RE2 reads a leading ']' as a literal; libcudf's behavior differs.
    if (peek() == ']') {
      return std::nullopt;
    }
    while (true) {
      if (atEnd()) {
        return std::nullopt;
      }
      if (peek() == ']') {
        ++pos_;
        break;
      }
      std::optional<char32_t> low;
      std::string set;
      if (!readClassMember(low, set)) {
        return std::nullopt;
      }
      if (!low.has_value()) {
        text += set;
        continue;
      }
      // As in RE2, a '-' between two characters forms a range, and is a
      // literal before ']'.
      if (peek() == '-' && peek(1) != ']' && pos_ + 1 < pattern_.size()) {
        ++pos_;
        std::optional<char32_t> high;
        std::string highSet;
        if (!readClassMember(high, highSet) || !high.has_value() ||
            *high < *low) {
          return std::nullopt;
        }
        // libcudf reads a range end raw, without decoding escapes, and a raw
        // ']' or '\' would end the class or start an escape.
        if (*high == ']' || *high == '\\' || *high < 0x20) {
          return std::nullopt;
        }
        foldSensitive_ = true;
        appendLiteral(text, *low);
        text += '-';
        appendUtf8(text, *high);
        continue;
      }
      appendLiteral(text, *low);
    }
    if (text == "[" || text == "[^") {
      return std::nullopt;
    }
    text += ']';
    return Node{std::move(text), false, false};
  }

  const std::string_view pattern_;
  const RegexCapture capture_;
  const int32_t captureGroup_;
  size_t pos_{0};
  int32_t numGroups_{0};
  bool ignoreCase_{false};
  bool foldSensitive_{false};
  bool hasWordBoundary_{false};
};

std::optional<int64_t> constantIntegerValue(const core::TypedExprPtr& expr) {
  if (!expr->isConstantKind()) {
    return std::nullopt;
  }
  const auto kind = expr->type()->kind();
  if (kind != TypeKind::INTEGER && kind != TypeKind::BIGINT) {
    return std::nullopt;
  }
  const auto* constExpr = expr->asUnchecked<core::ConstantTypedExpr>();
  if (constExpr->hasValueVector()) {
    const auto& vector = constExpr->valueVector();
    if (vector->isNullAt(0)) {
      return std::nullopt;
    }
    return kind == TypeKind::INTEGER
        ? vector->as<SimpleVector<int32_t>>()->valueAt(0)
        : vector->as<SimpleVector<int64_t>>()->valueAt(0);
  }
  const auto& value = constExpr->value();
  if (value.isNull()) {
    return std::nullopt;
  }
  return kind == TypeKind::INTEGER ? value.value<TypeKind::INTEGER>()
                                   : value.value<TypeKind::BIGINT>();
}

enum class RegexOp {
  kLike,
  kExtract,
  kExtractAll,
  kReplace,
  kSplit,
  kInstr,
};

// Everything a GPU regex function needs from its call, derived once at plan
// time by canEvaluate and again by the function's constructor.
struct RegexCall {
  CudfRegexPattern regex;
  int64_t group{0};
  bool hasBackrefs{false};
  // The replacement in libcudf form: literal text for replace_re(), or with
  // '${n}' group references for replace_with_backrefs().
  std::string replacement;
};

// Converts a replacement that Velox has rewritten into RE2 form, where '\n' is
// group n and '\\' a backslash, into libcudf form.
bool convertReplacement(
    std::string_view rewrite,
    int32_t numGroups,
    RegexCall& call) {
  std::string literal;
  std::string withBackrefs;
  for (size_t i = 0; i < rewrite.size(); ++i) {
    const char c = rewrite[i];
    if (c != '\\') {
      literal += c;
      withBackrefs += c;
      continue;
    }
    if (++i == rewrite.size()) {
      return false;
    }
    const char next = rewrite[i];
    if (next == '\\') {
      literal += '\\';
      withBackrefs += '\\';
    } else if (next >= '0' && next <= '9') {
      const int32_t group = next - '0';
      // RE2::GlobalReplace leaves the string unchanged when the rewrite
      // names a group the pattern lacks.
      if (group > numGroups) {
        return false;
      }
      withBackrefs += fmt::format("${{{}}}", group);
      call.hasBackrefs = true;
    } else {
      return false;
    }
  }
  if (!call.hasBackrefs) {
    call.replacement = std::move(literal);
    return true;
  }
  // libcudf reads '\<digit>' and '${<digit>}' in the replacement as group
  // references, with no way to escape them, so literal text must contain
  // neither.
  if (literal.find('\\') != std::string::npos ||
      literal.find("${") != std::string::npos) {
    return false;
  }
  call.replacement = std::move(withBackrefs);
  return true;
}

std::optional<RegexCall> analyzeRegexCall(
    const core::TypedExprPtr& expr,
    RegexOp op) {
  const auto& inputs = expr->inputs();
  if (inputs.size() < 2 || inputs[0]->isConstantKind() ||
      !inputs[0]->type()->isVarchar()) {
    return std::nullopt;
  }
  const auto patternValue = constantVarcharValue(inputs[1]);
  if (!patternValue.has_value() || patternValue->size() == 0) {
    return std::nullopt;
  }
  const std::string patternString(patternValue->data(), patternValue->size());

  RegexCall call;
  if (op == RegexOp::kExtract || op == RegexOp::kExtractAll) {
    if (inputs.size() > 3) {
      return std::nullopt;
    }
    if (inputs.size() == 3) {
      const auto group = constantIntegerValue(inputs[2]);
      if (!group.has_value()) {
        return std::nullopt;
      }
      call.group = *group;
    }
  }

  // Compile with the options of Velox's RE2 functions. A pattern RE2 rejects
  // fails on the CPU with the error the user expects.
  RE2 re(patternString, RE2::Quiet);
  if (!re.ok()) {
    return std::nullopt;
  }
  const int32_t numGroups = re.NumberOfCapturingGroups();
  if (call.group < 0 || call.group > numGroups) {
    return std::nullopt;
  }

  if (op == RegexOp::kReplace) {
    std::string replacement;
    if (inputs.size() >= 3) {
      const auto value = constantVarcharValue(inputs[2]);
      if (!value.has_value()) {
        return std::nullopt;
      }
      replacement = std::string(value->data(), value->size());
    }
    // Spark's optional position: only replacing from the first character is
    // supported.
    if (inputs.size() == 4) {
      const auto position = constantIntegerValue(inputs[3]);
      if (!position.has_value() || *position != 1) {
        return std::nullopt;
      }
    }
    if (inputs.size() > 4) {
      return std::nullopt;
    }
    std::string rewrite;
    try {
      rewrite = functions::prepareRegexpReplaceReplacement(
          re, StringView(replacement));
    } catch (const VeloxException&) {
      return std::nullopt;
    }
    if (!convertReplacement(rewrite, numGroups, call)) {
      return std::nullopt;
    }
  }

  RegexCapture capture = RegexCapture::kNone;
  if (op == RegexOp::kExtract ||
      (op == RegexOp::kExtractAll && call.group > 0)) {
    capture = RegexCapture::kSingle;
  } else if (op == RegexOp::kReplace && call.hasBackrefs) {
    capture = RegexCapture::kAll;
  }
  auto regex = translateRe2Pattern(
      patternString, capture, static_cast<int32_t>(call.group));
  if (!regex.has_value() || regex->numGroups != numGroups) {
    return std::nullopt;
  }
  // Velox steps past an empty match by one byte, and RE2::GlobalReplace by
  // its own rules, so patterns that can match empty stay on the CPU for the
  // functions that find every match.
  if ((op == RegexOp::kExtractAll || op == RegexOp::kReplace ||
       op == RegexOp::kSplit) &&
      regex->canMatchEmpty) {
    return std::nullopt;
  }
  // split_record_re() ignores the regex flags, including ASCII, which \b
  // depends on.
  if (op == RegexOp::kSplit &&
      (regex->hasModifiers || regex->hasWordBoundary)) {
    return std::nullopt;
  }
  try {
    cudf::strings::regex_program::create(regex->pattern, regex->flags);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  call.regex = std::move(*regex);
  return call;
}

class RegexFunction : public CudfFunction {
 protected:
  RegexFunction(const core::TypedExprPtr& expr, RegexOp op) {
    auto call = analyzeRegexCall(expr, op);
    VELOX_CHECK(
        call.has_value(),
        "Regex call is not supported by the cuDF evaluator: {}",
        expr->toString());
    call_ = std::move(*call);
    program_ = cudf::strings::regex_program::create(
        call_.regex.pattern, call_.regex.flags);
  }

  RegexCall call_;
  std::unique_ptr<cudf::strings::regex_program> program_;
};

// Presto regexp_like and Spark rlike.
class RegexLikeFunction : public RegexFunction {
 public:
  explicit RegexLikeFunction(const core::TypedExprPtr& expr)
      : RegexFunction(expr, RegexOp::kLike) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::contains_re(
        asView(inputColumns[0]), *program_, stream, mr);
  }
};

// regexp_extract. Presto returns NULL when nothing matches or the group does
// not take part in the match; Spark returns an empty string.
class RegexExtractFunction : public RegexFunction {
 public:
  RegexExtractFunction(const core::TypedExprPtr& expr, bool emptyNoMatch)
      : RegexFunction(expr, RegexOp::kExtract), emptyNoMatch_(emptyNoMatch) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    auto extracted =
        cudf::strings::extract_single(input, *program_, 0, stream, mr);
    if (!emptyNoMatch_) {
      return extracted;
    }
    cudf::string_scalar empty("", true, stream, mr);
    auto result = cudf::replace_nulls(extracted->view(), empty, stream, mr);
    if (input.nullable()) {
      result->set_null_mask(
          cudf::copy_bitmask(
              input, stream, cudf::memory_resources{mr, get_temp_mr()}),
          input.null_count());
    }
    return result;
  }

 private:
  const bool emptyNoMatch_;
};

// regexp_extract_all: every non-overlapping match, or the given group of each.
class RegexExtractAllFunction : public RegexFunction {
 public:
  explicit RegexExtractAllFunction(const core::TypedExprPtr& expr)
      : RegexFunction(expr, RegexOp::kExtractAll) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    if (call_.group == 0) {
      return cudf::strings::findall(input, *program_, stream, mr);
    }
    // extract_all_record() returns NULL for a row without a match, where
    // Velox returns an empty array. Such rows hold no elements, so the input's
    // null mask makes them empty arrays.
    auto result =
        cudf::strings::extract_all_record(input, *program_, stream, mr);
    if (input.nullable()) {
      result->set_null_mask(
          cudf::copy_bitmask(
              input, stream, cudf::memory_resources{mr, get_temp_mr()}),
          input.null_count());
    } else {
      result->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
    }
    return result;
  }
};

// regexp_replace, with or without group references in the replacement.
class RegexReplaceFunction : public RegexFunction {
 public:
  explicit RegexReplaceFunction(const core::TypedExprPtr& expr)
      : RegexFunction(expr, RegexOp::kReplace) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    if (call_.hasBackrefs) {
      return cudf::strings::replace_with_backrefs(
          input, *program_, call_.replacement, stream, mr);
    }
    cudf::string_scalar replacement(call_.replacement, true, stream, mr);
    return cudf::strings::replace_re(
        input, *program_, replacement, std::nullopt, stream, mr);
  }
};

// Presto regexp_split.
class RegexSplitFunction : public RegexFunction {
 public:
  explicit RegexSplitFunction(const core::TypedExprPtr& expr)
      : RegexFunction(expr, RegexOp::kSplit) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    auto result =
        cudf::strings::split_record_re(input, *program_, -1, stream, mr);
    // split_record_re() splits an empty string into an empty list, where
    // Velox returns [""]. The pattern cannot match an empty string, so the
    // rows with no elements are exactly the empty strings.
    auto counts = cudf::lists::count_elements(
        cudf::lists_column_view(result->view()), stream, mr);
    cudf::numeric_scalar<int32_t> zero(0, true, stream, mr);
    auto isEmpty = cudf::binary_operation(
        counts->view(),
        zero,
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto anyAgg = cudf::make_any_aggregation<cudf::reduce_aggregation>();
    auto anyEmpty = cudf::reduce(
        isEmpty->view(),
        *anyAgg,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    const auto& anyEmptyScalar =
        static_cast<cudf::numeric_scalar<bool> const&>(*anyEmpty);
    if (!anyEmptyScalar.is_valid(stream) || !anyEmptyScalar.value(stream)) {
      return result;
    }
    auto emptyString = cudf::make_column_from_scalar(
        cudf::string_scalar("", true, stream, mr), 1, stream, mr);
    cudf::list_scalar emptyStringList(emptyString->view(), true, stream, mr);
    return cudf::copy_if_else(
        emptyStringList, result->view(), isEmpty->view(), stream, mr);
  }
};

// Spark regexp_instr: the 1-based character position of the first match, or
// 0 when there is none.
class RegexInstrFunction : public RegexFunction {
 public:
  explicit RegexInstrFunction(const core::TypedExprPtr& expr)
      : RegexFunction(expr, RegexOp::kInstr) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    auto positions = cudf::strings::find_re(
        asView(inputColumns[0]), *program_, stream, mr);
    cudf::numeric_scalar<int32_t> one(1, true, stream, mr);
    return cudf::binary_operation(
        positions->view(),
        one,
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_id::INT32},
        stream,
        mr);
  }
};

CudfCanEvaluate canEvaluateRegex(RegexOp op) {
  return [op](const core::TypedExprPtr& expr, const core::QueryConfig&) {
    return analyzeRegexCall(expr, op).has_value();
  };
}

template <typename F>
CudfFunctionFactory regexFactory(F make) {
  return [make](
             const std::string&,
             const core::TypedExprPtr& expr,
             memory::MemoryPool*,
             const core::QueryConfig&) -> std::shared_ptr<CudfFunction> {
    return make(expr);
  };
}

std::vector<exec::FunctionSignaturePtr> regexSignatures(
    const std::string& returnType,
    const std::vector<std::vector<std::string>>& constantArgumentLists) {
  std::vector<exec::FunctionSignaturePtr> signatures;
  for (const auto& constantArguments : constantArgumentLists) {
    exec::FunctionSignatureBuilder builder;
    builder.returnType(returnType).argumentType("varchar");
    for (const auto& type : constantArguments) {
      builder.constantArgumentType(type);
    }
    signatures.push_back(builder.build());
  }
  return signatures;
}

void registerRegexLike(const std::string& name) {
  registerCudfFunction(
      name,
      regexFactory([](const core::TypedExprPtr& expr) {
        return std::make_shared<RegexLikeFunction>(expr);
      }),
      regexSignatures("boolean", {{"varchar"}}),
      true,
      canEvaluateRegex(RegexOp::kLike));
}

void registerRegexExtract(const std::string& name, bool emptyNoMatch) {
  registerCudfFunction(
      name,
      regexFactory([emptyNoMatch](const core::TypedExprPtr& expr) {
        return std::make_shared<RegexExtractFunction>(expr, emptyNoMatch);
      }),
      regexSignatures(
          "varchar",
          {{"varchar"}, {"varchar", "bigint"}, {"varchar", "integer"}}),
      true,
      canEvaluateRegex(RegexOp::kExtract));
}

void registerRegexExtractAll(const std::string& name) {
  registerCudfFunction(
      name,
      regexFactory([](const core::TypedExprPtr& expr) {
        return std::make_shared<RegexExtractAllFunction>(expr);
      }),
      regexSignatures(
          "array(varchar)",
          {{"varchar"}, {"varchar", "bigint"}, {"varchar", "integer"}}),
      true,
      canEvaluateRegex(RegexOp::kExtractAll));
}

void registerRegexReplace(
    const std::string& name,
    const std::vector<std::vector<std::string>>& constantArgumentLists) {
  registerCudfFunction(
      name,
      regexFactory([](const core::TypedExprPtr& expr) {
        return std::make_shared<RegexReplaceFunction>(expr);
      }),
      regexSignatures("varchar", constantArgumentLists),
      true,
      canEvaluateRegex(RegexOp::kReplace));
}

} // namespace

std::optional<CudfRegexPattern> translateRe2Pattern(
    std::string_view re2Pattern,
    RegexCapture capture,
    int32_t captureGroup) {
  RE2 re(re2::StringPiece(re2Pattern.data(), re2Pattern.size()), RE2::Quiet);
  if (!re.ok() || re2Pattern.empty()) {
    return std::nullopt;
  }
  return Re2ToCudfTranslator(re2Pattern, capture, captureGroup).translate();
}

void registerPrestoRegexFunctions(const std::string& prefix) {
  registerRegexLike(prefix + "regexp_like");
  registerRegexExtract(prefix + "regexp_extract", /*emptyNoMatch=*/false);
  registerRegexExtractAll(prefix + "regexp_extract_all");
  registerRegexReplace(
      prefix + "regexp_replace", {{"varchar"}, {"varchar", "varchar"}});
  registerCudfFunction(
      prefix + "regexp_split",
      regexFactory([](const core::TypedExprPtr& expr) {
        return std::make_shared<RegexSplitFunction>(expr);
      }),
      regexSignatures("array(varchar)", {{"varchar"}}),
      true,
      canEvaluateRegex(RegexOp::kSplit));
}

void registerSparkRegexFunctions(const std::string& prefix) {
  registerRegexLike(prefix + "rlike");
  registerRegexExtract(prefix + "regexp_extract", /*emptyNoMatch=*/true);
  registerRegexExtractAll(prefix + "regexp_extract_all");
  registerRegexReplace(
      prefix + "regexp_replace",
      {{"varchar", "varchar"}, {"varchar", "varchar", "integer"}});
  registerCudfFunction(
      prefix + "regexp_instr",
      regexFactory([](const core::TypedExprPtr& expr) {
        return std::make_shared<RegexInstrFunction>(expr);
      }),
      regexSignatures("integer", {{"varchar"}}),
      true,
      canEvaluateRegex(RegexOp::kInstr));
}

} // namespace facebook::velox::cudf_velox
