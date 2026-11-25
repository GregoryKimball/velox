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
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/RegexFunctions.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace facebook::velox;

namespace facebook::velox::cudf_velox {
namespace {

std::string sqlString(std::string_view value) {
  std::string quoted = "'";
  for (char c : value) {
    quoted += c;
    if (c == '\'') {
      quoted += '\'';
    }
  }
  return quoted + "'";
}

// Hand-picked strings covering anchors, a trailing newline, punctuation,
// multi-byte characters, and empty and null rows.
std::vector<std::optional<std::string>> edgeCaseStrings() {
  return {
      "abc",
      "aaa bbb ccc",
      "ab",
      "a",
      "b",
      "",
      std::nullopt,
      "abc\n",
      "a\nb\nc",
      "line one\nline two\n",
      "foo bar foo",
      "foobar",
      "x foo, y foo; z",
      "john.doe@example.com",
      "alice@test.org, bob@example.com",
      "call 555-1234 or 555-9876",
      "price: $42.50, tax: $3.10",
      "a,b,,c,",
      ",leading,comma",
      "tabs\tand  spaces\t",
      "colour color colr",
      "xx xxx xxxx x",
      "(parens) [brackets] {braces}",
      "back\\slash",
      "C:\\path\\to\\file.txt",
      "https://www.example.com/a/b?c=1&d=2",
      "2026-11-25T10:30:00",
      "ID-00042-XYZ",
      "café crème",
      "日本語のテキスト",
      "naïve résumé",
      "emoji 😀 here 😀",
      "MiXeD CaSe AbC",
      "ABC abc Abc",
      "under_score and-dash",
      "a1b2c3",
      "   ",
      "aaaa",
      "abababab",
      "the end.",
  };
}

// Pseudo-random strings over a small alphabet so that short patterns match
// often, with every 17th row null.
std::vector<std::optional<std::string>> randomStrings(int32_t count) {
  static const std::vector<std::string> kAlphabet = {
      "a", "b", "c", "x", "1", "2", " ", "_", "-", ".", "@", ",", "\n", "é",
      "日", "A", "B"};
  std::mt19937 rng(20261125);
  std::uniform_int_distribution<int32_t> length(0, 24);
  std::uniform_int_distribution<size_t> pick(0, kAlphabet.size() - 1);
  std::vector<std::optional<std::string>> strings;
  strings.reserve(count);
  for (int32_t i = 0; i < count; ++i) {
    if (i % 17 == 16) {
      strings.push_back(std::nullopt);
      continue;
    }
    std::string s;
    const auto n = length(rng);
    for (int32_t j = 0; j < n; ++j) {
      s += kAlphabet[pick(rng)];
    }
    strings.push_back(std::move(s));
  }
  return strings;
}

// Patterns that cannot match an empty string and that every function should
// run on the GPU.
const std::vector<std::string>& gpuPatterns() {
  static const std::vector<std::string> kPatterns = {
      "a",
      "abc",
      "foo",
      "[a-c]+",
      "[^a-c ]+",
      "\\d+",
      "\\D+",
      "\\w+",
      "\\W",
      "\\s+",
      "\\S+",
      "a.c",
      "a.b",
      "^a",
      "c$",
      "^abc$",
      "\\Aab",
      "c\\z",
      "a|ab",
      "ab|a",
      "(a|b)+c",
      "(?:ab)+",
      "x{2}",
      "x{2,3}",
      "x{2,}",
      "x{2,3}?",
      "a+?",
      "a*?b",
      "colou?r",
      "\\d{3}-\\d{4}",
      "\\$\\d+\\.\\d\\d",
      "[.]",
      "\\.",
      "[\\-+]",
      "[-a]",
      "[a-]",
      "[\\]x]",
      "[\\^b]",
      "[,;]",
      "\\(\\w+\\)",
      "\\[|\\]",
      "\\{|\\}",
      "\\\\",
      "\\x41",
      "\\x{e9}",
      "é+",
      "日本",
      "[à-ü]+",
      "[\\x{3040}-\\x{30ff}]+",
      "😀",
      "\\n",
      "\\t+",
      "[\\w.]+@[\\w.]+",
      "https?://[^/\\s]+",
      "\\d{4}-\\d{2}-\\d{2}",
      "[A-Z]+",
      "(?i)abc",
      "(?i)hello|world",
      "(?s)a.b",
      "(?is)A.B",
      "a{,2}",
      "a{",
      "[a\\sb]",
      "[\\d_]+",
  };
  return kPatterns;
}

// Patterns that can match an empty string. regexp_like, regexp_extract and
// regexp_instr run them on the GPU; the functions that find every match do
// not.
const std::vector<std::string>& nullablePatterns() {
  static const std::vector<std::string> kPatterns = {
      "a*",
      "x?",
      "^",
      "$",
      "(?:ab)*",
      "(a|b){0,2}",
  };
  return kPatterns;
}

// Patterns that no function runs on the GPU: RE2 syntax that libcudf lacks or
// reads differently, and invalid patterns, which fail on the CPU.
const std::vector<std::string>& cpuOnlyPatterns() {
  static const std::vector<std::string> kPatterns = {
      "\\pL",
      "\\p{Greek}",
      "\\PN",
      "[[:alpha:]]+",
      "(?P<name>a)",
      "(?<name>a)",
      "(?m)^a",
      "a(?i)b",
      "(?U)a+",
      "(?i)k",
      "(?i)[a-z]",
      "(?i)\\w",
      "\\b",
      "\\bfoo\\b",
      "foo\\B",
      "\\Qa.b\\E",
      "\\C",
      "a|",
      "|a",
      "()",
      "(a*)*",
      "(a?)+",
      "[]a]",
      "[^]a]",
      "\\x00",
      "a{1000}",
      "a(",
      "[a",
      "*a",
      "\\1",
      "",
  };
  return kPatterns;
}

class CudfRegexTest : public CudfFunctionBaseTest {
 protected:
  void setInput(const std::vector<std::optional<std::string>>& strings) {
    input_ = makeRowVector({makeNullableFlatVector<std::string>(strings)});
    rowType_ = asRowType(input_->type());
  }

  bool runsOnGpu(const std::string& expr) {
    return canExprRunOnGpu(
        makeTypedExpr(expr, rowType_), execCtx_.queryCtx(), pool_.get());
  }

  void expectGpuMatchesCpu(const std::string& expr) {
    SCOPED_TRACE(expr);
    ASSERT_TRUE(runsOnGpu(expr)) << "expected GPU support";
    assertExpressionMatchesCpu(expr, input_, rowType_);
  }

  void expectCpuOnly(const std::string& expr) {
    SCOPED_TRACE(expr);
    EXPECT_FALSE(runsOnGpu(expr)) << "expected CPU fallback";
  }

  void expectAllInputs(const std::function<void()>& check) {
    {
      SCOPED_TRACE("edge cases");
      setInput(edgeCaseStrings());
      check();
    }
    {
      SCOPED_TRACE("random strings");
      setInput(randomStrings(4'000));
      check();
    }
  }

  RowVectorPtr input_;
  RowTypePtr rowType_;
};

class CudfPrestoRegexTest : public CudfRegexTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
    registerPrestoFunctions("");
  }

  static void TearDownTestCase() {
    unregisterFunctions();
    unregisterCudf();
  }
};

TEST(RegexTranslationTest, rewritesRe2Syntax) {
  auto translate = [](std::string_view pattern) {
    auto result = translateRe2Pattern(pattern);
    return result.has_value() ? result->pattern : std::string("<cpu>");
  };
  EXPECT_EQ(translate("abc"), "abc");
  EXPECT_EQ(translate("^a.c$"), "\\Aa.c\\Z");
  EXPECT_EQ(translate("a\\z"), "a\\Z");
  EXPECT_EQ(translate("\\d+"), "[0-9]+");
  EXPECT_EQ(translate("\\s"), "[\\x09\\x0a\\x0c\\x0d\\x20]");
  EXPECT_EQ(translate("[\\w-]"), "[0-9A-Za-z_\\x2d]");
  EXPECT_EQ(translate("\\$1\\.00"), "\\x241\\x2e00");
  EXPECT_EQ(translate("a.b"), "a.b");
  EXPECT_EQ(translate("x{2,3}?"), "x{2,3}?");
  EXPECT_EQ(translate("a{,2}"), "a\\x7b\\x2c2\\x7d");
  EXPECT_EQ(translate("[!-/]"), "[\\x21-/]");
  EXPECT_EQ(translate("\\x{e9}"), "é");
  EXPECT_EQ(translate("(a)(?:b)"), "(?:a)(?:b)");
  EXPECT_EQ(translate("\\pL"), "<cpu>");
  EXPECT_EQ(translate("[[:digit:]]"), "<cpu>");
  EXPECT_EQ(translate("a|"), "<cpu>");
}

TEST(RegexTranslationTest, capturesOnlyTheRequestedGroup) {
  auto translate = [](std::string_view pattern,
                      RegexCapture capture,
                      int32_t group) {
    auto result = translateRe2Pattern(pattern, capture, group);
    return result.has_value() ? result->pattern : std::string("<cpu>");
  };
  EXPECT_EQ(translate("(a)(b)", RegexCapture::kAll, 0), "(a)(b)");
  EXPECT_EQ(translate("(a)(b)", RegexCapture::kSingle, 2), "(?:a)(b)");
  EXPECT_EQ(translate("(a)(b)", RegexCapture::kSingle, 0), "((?:a)(?:b))");
  EXPECT_EQ(translate("((a)b)", RegexCapture::kSingle, 2), "(?:(a)b)");
}

TEST(RegexTranslationTest, reportsPatternProperties) {
  auto a = translateRe2Pattern("(?i)(x)(y)?");
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(a->numGroups, 2);
  EXPECT_TRUE(a->hasModifiers);
  EXPECT_TRUE(cudf::strings::is_ignorecase(a->flags));
  EXPECT_FALSE(a->canMatchEmpty);
  EXPECT_TRUE(translateRe2Pattern("a*")->canMatchEmpty);
  EXPECT_TRUE(translateRe2Pattern("a|b*")->canMatchEmpty);
}

TEST_F(CudfPrestoRegexTest, regexpLike) {
  expectAllInputs([&] {
    for (const auto& pattern : gpuPatterns()) {
      expectGpuMatchesCpu(fmt::format("regexp_like(c0, {})", sqlString(pattern)));
    }
    for (const auto& pattern : nullablePatterns()) {
      expectGpuMatchesCpu(fmt::format("regexp_like(c0, {})", sqlString(pattern)));
    }
  });
}

TEST_F(CudfPrestoRegexTest, regexpExtract) {
  expectAllInputs([&] {
    for (const auto& pattern : gpuPatterns()) {
      expectGpuMatchesCpu(
          fmt::format("regexp_extract(c0, {})", sqlString(pattern)));
    }
    for (const auto& pattern : nullablePatterns()) {
      expectGpuMatchesCpu(
          fmt::format("regexp_extract(c0, {})", sqlString(pattern)));
    }
  });
}

TEST_F(CudfPrestoRegexTest, regexpExtractGroup) {
  const std::vector<std::pair<std::string, int32_t>> cases = {
      {"(\\w+)@(\\w+)", 0},
      {"(\\w+)@(\\w+)", 1},
      {"(\\w+)@(\\w+)", 2},
      {"(a)|(b)", 1},
      {"(a)|(b)", 2},
      {"(a)?b", 1},
      {"(a|b)+", 1},
      {"((a)(b))c", 2},
      {"((a)(b))c", 3},
      {"(\\d+)-(\\d+)", 2},
      {"(?i)(abc)", 1},
      {"(\\$)(\\d+)", 2},
      {"(é)(.)", 2},
  };
  expectAllInputs([&] {
    for (const auto& [pattern, group] : cases) {
      expectGpuMatchesCpu(fmt::format(
          "regexp_extract(c0, {}, {})", sqlString(pattern), group));
      expectGpuMatchesCpu(fmt::format(
          "regexp_extract_all(c0, {}, {})", sqlString(pattern), group));
    }
  });
}

TEST_F(CudfPrestoRegexTest, regexpExtractAll) {
  expectAllInputs([&] {
    for (const auto& pattern : gpuPatterns()) {
      expectGpuMatchesCpu(
          fmt::format("regexp_extract_all(c0, {})", sqlString(pattern)));
    }
  });
}

TEST_F(CudfPrestoRegexTest, regexpReplace) {
  expectAllInputs([&] {
    for (const auto& pattern : gpuPatterns()) {
      expectGpuMatchesCpu(
          fmt::format("regexp_replace(c0, {})", sqlString(pattern)));
      expectGpuMatchesCpu(
          fmt::format("regexp_replace(c0, {}, '<$0>')", sqlString(pattern)));
    }
  });
}

TEST_F(CudfPrestoRegexTest, regexpReplaceBackrefs) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"(\\w+)@(\\w+)", "$2 at $1"},
      {"(\\w+)@(\\w+)", "[$0]"},
      {"(a)|(b)", "<$1|$2>"},
      {"(\\d)(\\d)", "$2$1"},
      {"(\\d+)", "$1$1"},
      {"(x+)", "\\$1"},
      {"a", "\\\\"},
      {"a", "$"},
      {"a", "50%"},
      {"(é)", "e$1"},
  };
  expectAllInputs([&] {
    for (const auto& [pattern, replacement] : cases) {
      expectGpuMatchesCpu(fmt::format(
          "regexp_replace(c0, {}, {})",
          sqlString(pattern),
          sqlString(replacement)));
    }
  });
}

TEST_F(CudfPrestoRegexTest, regexpSplit) {
  const std::vector<std::string> patterns = {
      ",", "\\s+", "[,;]", "\\d", "foo", "a|b", "\\n", "x{2}", "é"};
  expectAllInputs([&] {
    for (const auto& pattern : patterns) {
      expectGpuMatchesCpu(
          fmt::format("regexp_split(c0, {})", sqlString(pattern)));
    }
  });
}

TEST_F(CudfPrestoRegexTest, filterAndProject) {
  setInput(edgeCaseStrings());
  expectGpuMatchesCpu(
      "if(regexp_like(c0, '\\d+'), regexp_extract(c0, '\\d+'), 'none')");
  expectGpuMatchesCpu("regexp_extract_all(lower(c0), '[a-z]+')");
  expectGpuMatchesCpu("upper(regexp_replace(c0, '[aeiou]', '*'))");
}

TEST_F(CudfPrestoRegexTest, supportCheck) {
  setInput(edgeCaseStrings());
  for (const auto& pattern : cpuOnlyPatterns()) {
    const auto quoted = sqlString(pattern);
    expectCpuOnly(fmt::format("regexp_like(c0, {})", quoted));
    expectCpuOnly(fmt::format("regexp_extract(c0, {})", quoted));
    expectCpuOnly(fmt::format("regexp_extract_all(c0, {})", quoted));
    expectCpuOnly(fmt::format("regexp_replace(c0, {}, 'x')", quoted));
    expectCpuOnly(fmt::format("regexp_split(c0, {})", quoted));
  }
  for (const auto& pattern : nullablePatterns()) {
    const auto quoted = sqlString(pattern);
    expectCpuOnly(fmt::format("regexp_extract_all(c0, {})", quoted));
    expectCpuOnly(fmt::format("regexp_replace(c0, {}, 'x')", quoted));
    expectCpuOnly(fmt::format("regexp_split(c0, {})", quoted));
  }
  // The pattern, group and replacement must be constants.
  expectCpuOnly("regexp_like(c0, c0)");
  expectCpuOnly("regexp_replace(c0, 'a', c0)");
  // A group the pattern lacks fails on the CPU.
  expectCpuOnly("regexp_extract(c0, '(a)', 2)");
  expectCpuOnly("regexp_extract(c0, '(a)', -1)");
  // libcudf cannot escape a literal backslash next to a group reference.
  expectCpuOnly("regexp_replace(c0, '(a)', '$1\\\\')");
  // RE2 leaves the string unchanged when a replacement names a missing group.
  expectCpuOnly("regexp_replace(c0, 'a', '$1')");
  // split_record_re ignores regex flags.
  expectCpuOnly("regexp_split(c0, '(?i)a')");
  expectCpuOnly("regexp_split(c0, '\\bfoo')");
}

class CudfSparkRegexTest : public CudfRegexTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions("");
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
    registerSparkFunctions("");
  }

  static void TearDownTestCase() {
    unregisterFunctions();
    unregisterCudf();
  }

  CudfSparkRegexTest() {
    options_.parseIntegerAsBigint = false;
  }
};

TEST_F(CudfSparkRegexTest, rlikeAndInstr) {
  expectAllInputs([&] {
    for (const auto& pattern : gpuPatterns()) {
      expectGpuMatchesCpu(fmt::format("rlike(c0, {})", sqlString(pattern)));
      expectGpuMatchesCpu(
          fmt::format("regexp_instr(c0, {})", sqlString(pattern)));
    }
  });
}

TEST_F(CudfSparkRegexTest, regexpExtract) {
  const std::vector<std::pair<std::string, int32_t>> cases = {
      {"\\d+", 0},
      {"(\\w+)@(\\w+)", 1},
      {"(\\w+)@(\\w+)", 2},
      {"(a)|(b)", 2},
      {"(a)?b", 1},
      {"x*", 0},
  };
  expectAllInputs([&] {
    for (const auto& [pattern, group] : cases) {
      expectGpuMatchesCpu(fmt::format(
          "regexp_extract(c0, {}, {})", sqlString(pattern), group));
      if (pattern != "x*") {
        expectGpuMatchesCpu(fmt::format(
            "regexp_extract_all(c0, {}, {})", sqlString(pattern), group));
      }
    }
  });
}

TEST_F(CudfSparkRegexTest, regexpReplace) {
  expectAllInputs([&] {
    expectGpuMatchesCpu("regexp_replace(c0, '\\d', '#')");
    expectGpuMatchesCpu("regexp_replace(c0, '(\\w+)@(\\w+)', '$2/$1')");
    expectGpuMatchesCpu("regexp_replace(c0, '[aeiou]', '', 1)");
  });
  setInput(edgeCaseStrings());
  expectCpuOnly("regexp_replace(c0, 'a', 'b', 2)");
  expectCpuOnly("rlike(c0, '[a[b]]')");
}

} // namespace
} // namespace facebook::velox::cudf_velox
