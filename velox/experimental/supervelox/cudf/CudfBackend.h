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

#include <cudf/column/column_view.hpp>
#include <cudf/udf.hpp>

#include <cstdint>
#include <string>

#include "velox/experimental/supervelox/queries/Query.h"

// Versions of the fused queries built only from cuDF operators: the JIT
// transform and reduction APIs, groupby and the column utilities. They read
// the same HBM cache as the hand-written kernels, through zero-copy views, and
// take the same layout hints: zone maps, dictionary codes and decimals as
// scaled integers.

namespace facebook::velox::supervelox::cudf_backend {

/// Rows [begin, end) of a table.
struct RowRange {
  int64_t begin{0};
  int64_t end{0};

  int64_t size() const {
    return end - begin;
  }
};

/// A zero-copy view of rows of a cached fixed-width column, at its stored
/// width: dictionary codes are integer columns, decimals unscaled integers.
cudf::column_view fixedView(QueryContext& ctx, const std::string& table, const std::string& name, RowRange rows);

/// Splits the zones of an integer column by an inclusive bound [low, high]:
/// `all` covers zones whose values all lie in it, `some` the zones from the
/// first to the last that may hold a value in it. Rows are whole zones.
struct ZoneSplit {
  RowRange some;
  /// The longest run of zones, within `some`, that pass entirely.
  RowRange all;
};

ZoneSplit
splitZones(QueryContext& ctx, const std::string& table, const std::string& name, int64_t low, int64_t high);

/// LTO-IR of CUDA source using the bundled cuDF headers, compiled once per
/// distinct source.
const cudf::compiled_fragment& compileFragment(const std::string& source);

/// Registers the cuDF versions of the queries and makes cuDF allocate from
/// the CUDA default memory pool the engine pre-grows.
void registerCudfBackend();

} // namespace facebook::velox::supervelox::cudf_backend
