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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/udf.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox/experimental/supervelox/common/Cuda.h"
#include "velox/experimental/supervelox/queries/Derived.h"
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

/// A zero-copy view of `rows` values of type `type` at `data`.
cudf::column_view deviceView(cudf::type_id type, const void* data, int64_t rows);

/// LTO-IR of CUDA source using the bundled cuDF headers, compiled once per
/// distinct source.
const cudf::compiled_fragment& compileFragment(const std::string& source);

/// A device expression true when `code` is one of `codes`.
std::string codeTest(const std::string& code, const CodeSet& codes);

/// A reduction of rows to int64 sums in a dense table of groups, generated
/// as one reduce_lto. `inputs` declares one device parameter per input
/// column; `filter`, `group` and `values` are device expressions over them,
/// after the statements of `prelude`.
struct GroupedSums {
  std::string inputs;
  std::string filter;
  std::string prelude;
  std::string group = "0";
  int32_t groups = 1;
  std::vector<std::string> values;
};

/// The sums, `groups` times `values.size()`, group-major.
std::vector<int64_t>
sumGroups(QueryContext& ctx, const GroupedSums& sums, const std::vector<cudf::column_view>& columns);

/// One column computed by `expression`, after the statements of `prelude`,
/// over `columns`, as one transform_lto. `inputs` declares one device
/// parameter per column.
std::unique_ptr<cudf::column> computeColumn(
    QueryContext& ctx,
    const std::string& inputs,
    const std::string& expression,
    cudf::type_id type,
    const std::vector<cudf::column_view>& columns,
    const std::string& prelude = "");

/// The rows holding `keys` in a table stored by key, at row key - 1.
std::unique_ptr<cudf::column> keyRows(QueryContext& ctx, const cudf::column_view& keys);

/// Columns `names` of `table` at `rows`.
std::unique_ptr<cudf::table> gatherColumns(
    QueryContext& ctx,
    const std::string& table,
    const std::vector<std::string>& names,
    const cudf::column_view& rows);

/// The lines of orders [begin, end), which are contiguous, with each order's
/// line count and where its lines start among them.
struct OrderLines {
  int64_t begin{0};
  int64_t end{0};
  RowRange lines;
  std::unique_ptr<cudf::column> counts;
  std::unique_ptr<cudf::column> offsets;
};

OrderLines orderLines(QueryContext& ctx, int64_t begin, int64_t end);

/// Orders [begin, end) cut into runs whose lines are at most `maxLines`, so
/// that what is made per line fits beside the cache.
std::vector<std::pair<int64_t, int64_t>>
orderChunks(QueryContext& ctx, int64_t begin, int64_t end, int64_t maxLines = int64_t{1} << 28);

/// Each order's value of `columns` repeated over its lines.
std::unique_ptr<cudf::table>
perLine(QueryContext& ctx, const OrderLines& lines, const std::vector<cudf::column_view>& columns);

/// The sum of the INT64 `values` over each order's lines.
std::unique_ptr<cudf::column> sumPerOrder(QueryContext& ctx, const cudf::column_view& values, const OrderLines& lines);

/// A zero-copy view of rows of a cached string column.
cudf::column_view stringView(QueryContext& ctx, const std::string& table, const std::string& name, RowRange rows);

/// The first `limit` row indices of `keys` in the order `orders` gives.
std::vector<int32_t> firstSorted(
    QueryContext& ctx,
    const cudf::table_view& keys,
    const std::vector<cudf::order>& orders,
    int64_t limit);

/// Aggregations `groupBy` computes per group.
enum class Aggregate { kSum, kMin, kMax, kCount, kDistinct };

struct Grouped {
  std::unique_ptr<cudf::table> keys;
  std::vector<std::unique_ptr<cudf::column>> values;
};

/// One row per distinct row of `keys`, with each of `aggregates` over its
/// column of values.
Grouped groupBy(
    QueryContext& ctx,
    const cudf::table_view& keys,
    const std::vector<std::pair<cudf::column_view, Aggregate>>& aggregates);

/// The host copy of a fixed-width column.
template <typename T>
std::vector<T> download(QueryContext& ctx, const cudf::column_view& column) {
  std::vector<T> host(column.size());
  if (!host.empty()) {
    SV_CUDA(cudaMemcpyAsync(
        host.data(), column.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost, ctx.stream()));
    SV_CUDA(cudaStreamSynchronize(ctx.stream()));
  }
  return host;
}

/// Registers the cuDF versions of the queries and makes cuDF allocate from
/// the CUDA default memory pool the engine pre-grows.
void registerCudfBackend();

} // namespace facebook::velox::supervelox::cudf_backend
