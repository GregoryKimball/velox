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
#include <cudf/column/column.hpp>
#include <cudf/reduction.hpp>

#include <fmt/format.h>

#include "velox/experimental/supervelox/cudf/CudfBackend.h"

namespace facebook::velox::supervelox::cudf_backend {
namespace {

/// Inputs in order: shipdate, discount, quantity, extendedprice. The state
/// is the unscaled revenue at scale 4.
constexpr const char* kSource = R"(
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cuda/std/cstdint>

using cudf::errc;
using cudf::size_type;
using namespace cuda::std;

extern "C" {{

__device__ errc q6_filter(size_type, bool* keep, int16_t shipdate, int8_t discount, int16_t quantity, int32_t) {{
  *keep = shipdate >= {} && shipdate <= {} && discount >= {} && discount <= {} && quantity <= {};
  return errc::SUCCESS;
}}

__device__ errc q6_init(int64_t* revenue) {{
  *revenue = 0;
  return errc::SUCCESS;
}}

__device__ errc q6_update(size_type, int64_t* revenue, int16_t, int8_t discount, int16_t, int32_t price) {{
  *revenue += int64_t{{price}} * discount;
  return errc::SUCCESS;
}}

__device__ errc q6_merge(int64_t* revenue, int64_t other) {{
  *revenue += other;
  return errc::SUCCESS;
}}

__device__ errc q6_finalize(int64_t* out, int64_t revenue) {{
  *out = revenue;
  return errc::SUCCESS;
}}

}}
)";

const std::vector<ResultField> kSchema = {{"revenue", ResultType::kDecimal, 38, 4}};

} // namespace

Result q6(QueryContext& ctx) {
  const int64_t dateLow = ctx.param("shipdate_min", days(1994, 1, 1));
  const int64_t dateHigh = ctx.param("shipdate_max", days(1994, 12, 31));
  const auto& fragment = compileFragment(fmt::format(
      kSource,
      dateLow,
      dateHigh,
      ctx.param("discount_min", 5),
      ctx.param("discount_max", 7),
      ctx.param("quantity_max", 2399)));

  const ZoneSplit split = splitZones(ctx, "lineitem", "l_shipdate", dateLow, dateHigh);
  Result result(kSchema);
  if (split.some.size() == 0) {
    result.decimals[0].push_back(0);
    return result;
  }
  const cudf::reduce_input inputs[] = {
      fixedView(ctx, "lineitem", "l_shipdate", split.some),
      fixedView(ctx, "lineitem", "l_discount", split.some),
      fixedView(ctx, "lineitem", "l_quantity", split.some),
      fixedView(ctx, "lineitem", "l_extendedprice", split.some),
  };
  const cudf::lto_fragment fragments[] = {{cudf::lto_binary_type::LTO_IR, fragment.view()}};
  const cudf::reduce_udf udf{fragments, "q6_init", "q6_update", "q6_merge", "q6_finalize", "q6_filter"};
  const cudf::data_type types[] = {cudf::data_type{cudf::type_id::INT64}};
  auto columns = cudf::reduce_lto(udf, cudf::null_aware::NO, inputs, types, types, ctx.stream());
  int64_t revenue = 0;
  cudaMemcpyAsync(&revenue, columns[0]->view().head(), sizeof(revenue), cudaMemcpyDeviceToHost, ctx.stream());
  cudaStreamSynchronize(ctx.stream());
  result.decimals[0].push_back(revenue);
  return result;
}

} // namespace facebook::velox::supervelox::cudf_backend
