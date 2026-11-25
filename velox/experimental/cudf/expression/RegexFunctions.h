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

#pragma once

#include <cudf/strings/regex/flags.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// Which capturing groups of an RE2 pattern stay capturing in its libcudf
/// translation. The others become non-capturing groups, so a libcudf program
/// only tracks the group the function reads.
enum class RegexCapture {
  // Every group stays capturing, as regexp_replace back-references need.
  kAll,
  // No group captures.
  kNone,
  // Only the group passed as 'captureGroup' captures; group 0 wraps the whole
  // pattern in a new group.
  kSingle,
};

/// An RE2 pattern rewritten into a libcudf pattern that matches the same
/// strings at the same positions.
struct CudfRegexPattern {
  std::string pattern;
  cudf::strings::regex_flags flags;
  // Capturing groups in the RE2 pattern.
  int32_t numGroups;
  // True if the pattern can match an empty string somewhere, e.g. 'a*' or
  // '^'. The functions that find every match step over empty matches in ways
  // libcudf does not reproduce, so they stay on the CPU for these patterns.
  bool canMatchEmpty;
  // True if the pattern uses leading mode modifiers, which become 'flags'.
  bool hasModifiers;
  // True if the pattern uses \b or \B.
  bool hasWordBoundary;
};

/// Translates an RE2 pattern, as Velox's RE2-based regex functions compile it,
/// into a libcudf pattern with the same semantics. Returns std::nullopt if RE2
/// rejects the pattern or it uses syntax that libcudf cannot match exactly,
/// such as Unicode classes, named groups, inline mode modifiers after the
/// start, or empty alternatives.
std::optional<CudfRegexPattern> translateRe2Pattern(
    std::string_view re2Pattern,
    RegexCapture capture = RegexCapture::kNone,
    int32_t captureGroup = 0);

void registerPrestoRegexFunctions(const std::string& prefix);

void registerSparkRegexFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox
