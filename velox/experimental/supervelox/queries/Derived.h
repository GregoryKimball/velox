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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "velox/experimental/supervelox/queries/Query.h"

namespace facebook::velox::supervelox {

/// Index of the first lineitem row of each order row, plus a final entry
/// holding the lineitem row count. Lineitem follows the order of orders, so
/// the lines of order row o are [starts[o], starts[o + 1]).
const uint32_t* orderLineStarts(QueryContext& ctx);

/// The lines of order rows [first, last): those of order row o are
/// [base + starts[o], base + starts[o + 1]), with 'starts' valid at
/// [first, last]. In one batch the whole orderLineStarts with base 0 and
/// [firstLine, lastLine) all of lineitem; batch by batch, offsets from
/// 'base' = 'firstLine' built for these orders only.
struct LineRange {
  const uint32_t* starts;
  int64_t base;
  int64_t firstLine;
  int64_t lastLine;
};
LineRange lineRange(QueryContext& ctx, int64_t first, int64_t last);

/// {min, max} over all lines of lineitem.'column' minus the o_orderdate of the
/// line's order. Lets lineitem date predicates bound the range of orders.
std::pair<int32_t, int32_t> lineDateLag(QueryContext& ctx, const std::string& column);

/// Rows [begin, end) of 'table' whose values of the ascending 'column' are in
/// [low, high), found by reading single values.
std::pair<int64_t, int64_t>
sortedRange(QueryContext& ctx, const std::string& table, const std::string& column, int64_t low, int64_t high);

/// Bit n set for each nation key n in the named region.
uint32_t nationsInRegion(QueryContext& ctx, const std::string& region);

/// Key of the named nation.
int32_t nationKey(QueryContext& ctx, const std::string& nation);

std::string nationName(QueryContext& ctx, int32_t key);

enum class Match { kContains, kPrefix, kSuffix };

/// Device bitmap with bit r set when string row r of table.column matches
/// 'pattern' under 'match'. With two patterns and kContains, matches
/// '%first%second%'.
const uint32_t* likeBitmap(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    Match match,
    const std::string& pattern,
    const std::string& second = "");

/// Like likeBitmap for rows [first, last) of 'strings', setting bit r - first
/// of 'bits' for a match at row r.
void likeRows(
    QueryContext& ctx,
    QueryContext::StringRows strings,
    int64_t first,
    int64_t last,
    Match match,
    const std::string& pattern,
    const std::string& second,
    uint32_t* bits);

/// Dense numbering of the set bits of a bitmap over 'rows' rows: the rank of
/// row r is ranks[r / 32] + popcount of the lower bits of its word.
struct BitmapRanks {
  const uint32_t* bits;
  const uint32_t* ranks;
  int64_t count;
};

BitmapRanks rankBitmap(QueryContext& ctx, const uint32_t* bits, int64_t rows);

#ifdef __CUDACC__
__device__ __forceinline__ int64_t bitRank(const BitmapRanks& r, int64_t row) {
  const uint32_t word = __ldg(r.bits + (row >> 5));
  return __ldg(r.ranks + (row >> 5)) + __popc(word & ((1u << (row & 31)) - 1));
}
#endif

/// A set of dictionary codes below 256, passed to kernels by value.
struct CodeSet {
  uint32_t words[8] = {};

  void add(int32_t code) {
    words[code >> 5] |= 1u << (code & 31);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
      bool
      contains(int32_t code) const {
    return (words[(code >> 5) & 7] >> (code & 31)) & 1;
  }
};

/// A code set travels in QueryParams as the words 'name.0' to 'name.7'.
inline void putCodes(QueryParams& params, const std::string& name, const CodeSet& codes) {
  for (int i = 0; i < 8; ++i) {
    params.values[name + "." + std::to_string(i)] = codes.words[i];
  }
}

inline CodeSet codesParam(const QueryContext& ctx, const std::string& name, const CodeSet& fallback) {
  if (ctx.param(name + ".0", -1) < 0) {
    return fallback;
  }
  CodeSet codes;
  for (int i = 0; i < 8; ++i) {
    codes.words[i] = static_cast<uint32_t>(ctx.param(name + "." + std::to_string(i), 0));
  }
  return codes;
}

/// Codes of the dictionary column table.column whose value matches.
CodeSet dictionaryCodes(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    Match match,
    const std::string& pattern);

/// Codes of table.column whose value is one of 'values'.
CodeSet dictionaryCodes(
    QueryContext& ctx,
    const std::string& table,
    const std::string& column,
    const std::vector<std::string>& values);

struct Candidate {
  int64_t index;
  int64_t value;
};

/// Rows of 'values' that can be among the 'k' largest: every row whose value
/// is at least the k-th largest, possibly a few more. Rows with value <=
/// 'floor' never qualify.
std::vector<Candidate>
topCandidates(QueryContext& ctx, const int64_t* values, int64_t count, int32_t k, int64_t floor = 0);

/// Device bitmap test.
#ifdef __CUDACC__
__device__ __forceinline__ bool testBit(const uint32_t* bits, int64_t index) {
  return (__ldg(bits + (index >> 5)) >> (index & 31)) & 1;
}
#endif

} // namespace facebook::velox::supervelox
