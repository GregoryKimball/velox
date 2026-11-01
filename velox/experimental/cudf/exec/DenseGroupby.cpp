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

#include "velox/experimental/cudf/exec/DenseGroupby.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/core/Expressions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <cstring>
#include <limits>
#include <map>

namespace facebook::velox::cudf_velox {
namespace {

using Kind = DenseGroupby::Kind;

std::string_view baseName(const std::string& name) {
  const auto dot = name.rfind('.');
  return dot == std::string::npos ? std::string_view(name)
                                  : std::string_view(name).substr(dot + 1);
}

bool isFloating(const TypePtr& type) {
  return type->isDouble() || type->isReal();
}

bool isDenseInput(const TypePtr& type) {
  return type->isTinyint() || type->isSmallint() || type->isInteger() ||
      type->isBigint() || type->isShortDecimal() || type->isDate() ||
      isFloating(type);
}

std::optional<Kind> kindOf(const core::AggregationNode::Aggregate& aggregate) {
  const auto name = baseName(aggregate.call->name());
  const bool noInput = aggregate.call->inputs().empty();
  if (name == "count") {
    return noInput ? Kind::kCountRows : Kind::kCount;
  }
  if (noInput) {
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

std::mutex registryMutex;
std::map<std::pair<const void*, core::PlanNodeId>, std::shared_ptr<DenseGroupby>>
    registry;

} // namespace

bool isDenseAggregate(const core::AggregationNode::Aggregate& aggregate) {
  if (aggregate.mask != nullptr || aggregate.distinct ||
      !aggregate.sortingKeys.empty() || !kindOf(aggregate).has_value() ||
      aggregate.call->inputs().size() > 1) {
    return false;
  }
  if (aggregate.call->inputs().empty()) {
    return true;
  }
  return std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
             aggregate.call->inputs()[0]) != nullptr &&
      aggregate.rawInputTypes.size() == 1 &&
      isDenseInput(aggregate.rawInputTypes[0]);
}

std::vector<DenseGroupby::Aggregate> DenseGroupby::aggregates(
    const core::AggregationNode& node,
    const RowTypePtr& inputType) {
  std::vector<Aggregate> out;
  for (size_t i = 0; i < node.aggregates().size(); ++i) {
    const auto& aggregate = node.aggregates()[i];
    VELOX_CHECK(
        isDenseAggregate(aggregate),
        "Not a dense aggregate: {}",
        aggregate.call->toString());
    Aggregate dense{
        *kindOf(aggregate),
        -1,
        nullptr,
        node.outputType()->childAt(node.groupingKeys().size() + i)};
    if (!aggregate.call->inputs().empty()) {
      const auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
              aggregate.call->inputs()[0]);
      dense.channel = inputType->getChildIdx(field->name());
      dense.inputType = inputType->childAt(dense.channel);
    }
    out.push_back(std::move(dense));
  }
  return out;
}

DenseGroupby::DenseGroupby(
    int64_t low,
    int64_t high,
    TypePtr keyType,
    std::vector<Aggregate> aggregates,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : low_(low),
      slots_(high - low + 1),
      keyType_(std::move(keyType)),
      rows_(slots_ * sizeof(int64_t), stream, mr),
      outOfRange_(sizeof(int), stream, mr) {
  VELOX_CHECK_LE(slots_, std::numeric_limits<cudf::size_type>::max());
  CUDF_CUDA_TRY(cudaMemsetAsync(rows_.data(), 0, rows_.size(), stream.get()));
  CUDF_CUDA_TRY(
      cudaMemsetAsync(outOfRange_.data(), 0, outOfRange_.size(), stream.get()));
  for (auto& aggregate : aggregates) {
    const bool floating =
        aggregate.inputType != nullptr && isFloating(aggregate.inputType);
    const bool counted = aggregate.kind != Kind::kCountRows;
    State state{
        aggregate,
        rmm::device_buffer(0, stream, mr),
        rmm::device_buffer(counted ? slots_ * sizeof(int64_t) : 0, stream, mr),
        floating};
    if (counted) {
      CUDF_CUDA_TRY(cudaMemsetAsync(
          state.counts.data(), 0, state.counts.size(), stream.get()));
    }
    switch (aggregate.kind) {
      case Kind::kSum:
      case Kind::kAvg:
        state.values = rmm::device_buffer(
            slots_ * (floating ? sizeof(double) : 2 * sizeof(int64_t)),
            stream,
            mr);
        CUDF_CUDA_TRY(cudaMemsetAsync(
            state.values.data(), 0, state.values.size(), stream.get()));
        break;
      case Kind::kMin:
      case Kind::kMax: {
        state.values = rmm::device_buffer(slots_ * sizeof(int64_t), stream, mr);
        const bool min = aggregate.kind == Kind::kMin;
        int64_t start;
        if (floating) {
          const double bound = min ? std::numeric_limits<double>::infinity()
                                   : -std::numeric_limits<double>::infinity();
          std::memcpy(&start, &bound, sizeof(start));
        } else {
          start = min ? std::numeric_limits<int64_t>::max()
                      : std::numeric_limits<int64_t>::min();
        }
        dense::fill(
            static_cast<int64_t*>(state.values.data()), slots_, start, stream);
        break;
      }
      default:
        break;
    }
    states_.push_back(std::move(state));
  }
  stream.sync();
}

void DenseGroupby::add(
    const std::vector<cudf::column_view>& input,
    int32_t keyChannel,
    cuda::stream_ref stream) {
  const auto& key = input[keyChannel];
  if (key.size() == 0) {
    return;
  }
  VELOX_CHECK_EQ(key.null_count(), 0, "Dense aggregation keys cannot be null");
  dense::addRows(
      key,
      low_,
      slots_,
      static_cast<unsigned long long*>(rows_.data()),
      static_cast<int*>(outOfRange_.data()),
      stream);
  for (auto& state : states_) {
    if (state.aggregate.kind == Kind::kCountRows) {
      continue;
    }
    dense::add(
        key,
        low_,
        slots_,
        input[state.aggregate.channel],
        state.aggregate.kind,
        state.floating,
        state.values.data(),
        static_cast<unsigned long long*>(state.counts.data()),
        stream);
  }
}

std::vector<std::unique_ptr<cudf::column>> DenseGroupby::release(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  int outOfRange = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &outOfRange,
      outOfRange_.data(),
      sizeof(int),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  VELOX_CHECK_EQ(
      outOfRange,
      0,
      "A dense aggregation key is outside [{}, {}]",
      low_,
      low_ + slots_ - 1);

  const auto* rows = static_cast<const unsigned long long*>(rows_.data());
  const auto slots = dense::occupied(rows, slots_, stream, mr);
  const auto groups = static_cast<cudf::size_type>(slots.size());

  std::vector<std::unique_ptr<cudf::column>> columns;
  auto keys = cudf::make_fixed_width_column(
      veloxToCudfDataType(keyType_),
      groups,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  dense::keys(slots, low_, keys->mutable_view(), stream);
  columns.push_back(std::move(keys));

  for (auto& state : states_) {
    auto column = cudf::make_fixed_width_column(
        veloxToCudfDataType(state.aggregate.resultType),
        groups,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    if (groups > 0) {
      rmm::device_uvector<bool> valid(groups, stream, mr);
      const bool rowCount = state.aggregate.kind == Kind::kCountRows;
      dense::results(
          slots,
          state.aggregate.kind,
          state.floating,
          state.values.data(),
          rowCount
              ? rows
              : static_cast<const unsigned long long*>(state.counts.data()),
          column->mutable_view(),
          valid.data(),
          stream);
      const cudf::column_view validView(
          cudf::data_type{cudf::type_id::BOOL8},
          groups,
          valid.data(),
          nullptr,
          0);
      auto [mask, nulls] = cudf::bools_to_mask(validView, stream, mr);
      if (nulls > 0) {
        column->set_null_mask(std::move(*mask), nulls);
      }
    }
    columns.push_back(std::move(column));
  }
  stream.sync();
  return columns;
}

std::shared_ptr<DenseGroupby> DenseGroupby::shared(
    const void* task,
    const core::PlanNodeId& node,
    const std::function<std::shared_ptr<DenseGroupby>()>& make) {
  std::lock_guard<std::mutex> lock(registryMutex);
  auto& slot = registry[{task, node}];
  if (slot == nullptr) {
    slot = make();
  }
  return slot;
}

void DenseGroupby::forget(const void* task, const core::PlanNodeId& node) {
  std::lock_guard<std::mutex> lock(registryMutex);
  registry.erase({task, node});
}

} // namespace facebook::velox::cudf_velox
