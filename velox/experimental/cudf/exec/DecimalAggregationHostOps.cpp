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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {

void validateIntermediateColumnType(cudf::column_view const& column) {
  // fmt does not understand cudf::type_id enum class
  auto const colType = static_cast<int>(column.type().id());
  VELOX_CHECK_EQ(
      colType,
      static_cast<int>(cudf::type_id::STRING),
      "Expected serialized decimal aggregation state: Velox VARBINARY represented as cuDF STRING (got type {})",
      colType);
}

cudf::column_view castDecimal64InputToDecimal128(
    cudf::column_view inputCol,
    std::unique_ptr<cudf::column>& holder,
    cuda::stream_ref stream) {
  if (inputCol.type().id() != cudf::type_id::DECIMAL64) {
    return inputCol;
  }
  holder = cudf::cast(
      inputCol,
      cudf::data_type{cudf::type_id::DECIMAL128, inputCol.type().scale()},
      stream,
      get_temp_mr());
  return holder->view();
}

std::unique_ptr<cudf::column> narrowDecimal128SumState(
    std::unique_ptr<cudf::column> sum,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(sum->type().id() == cudf::type_id::DECIMAL128);
  if (sum->null_count() < sum->size()) {
    using Scalar = cudf::fixed_point_scalar<numeric::decimal128>;
    auto const minAgg = cudf::make_min_aggregation<cudf::reduce_aggregation>();
    auto const maxAgg = cudf::make_max_aggregation<cudf::reduce_aggregation>();
    auto const minimum = cudf::reduce(
        sum->view(), *minAgg, sum->type(), stream, get_temp_mr());
    auto const maximum = cudf::reduce(
        sum->view(), *maxAgg, sum->type(), stream, get_temp_mr());
    VELOX_USER_CHECK(
        static_cast<const Scalar&>(*minimum).value(stream) >=
                std::numeric_limits<int64_t>::min() &&
            static_cast<const Scalar&>(*maximum).value(stream) <=
                std::numeric_limits<int64_t>::max(),
        "A decimal SUM state exceeds 64 bits; disable {}",
        CudfConfig::kCudfNarrowDecimalSumState);
  }
  return cudf::cast(
      sum->view(),
      cudf::data_type{cudf::type_id::DECIMAL64, sum->type().scale()},
      stream,
      mr);
}

std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    cuda::stream_ref stream) {
  if (count->type().id() != cudf::type_id::INT64) {
    count = cudf::cast(
        *count, cudf::data_type{cudf::type_id::INT64}, stream, get_temp_mr());
  }
  return count;
}

std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  return serializeDecimalSumState(
      sum->view(),
      count->view(),
      stream,
      mr,
      CudfConfig::getInstance().compactDecimalIntermediateState);
}

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  auto avgCol = computeDecimalAverage(sum->view(), count->view(), stream, mr);
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (avgCol->type() != cudfOutType) {
    avgCol = cudf::cast(avgCol->view(), cudfOutType, stream, mr);
  }
  return avgCol;
}

} // namespace facebook::velox::cudf_velox
