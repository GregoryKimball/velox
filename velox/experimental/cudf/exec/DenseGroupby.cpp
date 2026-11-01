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
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
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

/// Whether 'aggregate' takes no input or, as count(0), a non-null constant.
bool countsRows(const core::AggregationNode::Aggregate& aggregate) {
  const auto& inputs = aggregate.call->inputs();
  if (inputs.empty()) {
    return true;
  }
  const auto* constant =
      dynamic_cast<const core::ConstantTypedExpr*>(inputs[0].get());
  return inputs.size() == 1 && constant != nullptr && !constant->isNull();
}

std::optional<Kind> kindOf(const core::AggregationNode::Aggregate& aggregate) {
  const auto name = baseName(aggregate.call->name());
  const bool noInput = countsRows(aggregate);
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

RowTypePtr withDictionaryKeys(const RowTypePtr& type, const std::vector<DenseKey>& keys) {
  auto names = type->names();
  auto types = type->children();
  for (size_t k = 0; k < keys.size(); ++k) {
    if (!keys[k].dictionary.empty()) {
      types[k] = VARCHAR();
    }
  }
  return ROW(std::move(names), std::move(types));
}

/// A strings column of 'values'.
std::unique_ptr<cudf::column> strings(
    const std::vector<std::string>& values,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<int32_t> offsets{0};
  std::string chars;
  for (const auto& value : values) {
    chars += value;
    offsets.push_back(static_cast<int32_t>(chars.size()));
  }
  const auto count = static_cast<cudf::size_type>(values.size());
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      count + 1,
      rmm::device_buffer(
          offsets.data(), offsets.size() * sizeof(int32_t), stream, mr),
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
      0);
  auto out = cudf::make_strings_column(
      count,
      std::move(offsetsColumn),
      rmm::device_buffer(chars.data(), chars.size(), stream, mr),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  stream.sync();
  return out;
}

} // namespace

DenseAggregationNode::DenseAggregationNode(
    const core::PlanNodeId& id,
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const std::vector<std::string>& aggregateNames,
    const std::vector<Aggregate>& aggregates,
    core::PlanNodePtr source,
    std::vector<DenseKey> keys)
    : AggregationNode(
          id,
          Step::kSingle,
          groupingKeys,
          {},
          aggregateNames,
          aggregates,
          /*ignoreNullKeys=*/false,
          /*noGroupsSpanBatches=*/false,
          std::nullopt,
          std::move(source)),
      keys_(std::move(keys)),
      outputType_(withDictionaryKeys(AggregationNode::outputType(), keys_)) {
  VELOX_CHECK_EQ(keys_.size(), groupingKeys.size());
}

bool isDenseAggregate(const core::AggregationNode::Aggregate& aggregate) {
  if (aggregate.mask != nullptr || aggregate.distinct ||
      !aggregate.sortingKeys.empty() || !kindOf(aggregate).has_value() ||
      aggregate.call->inputs().size() > 1) {
    return false;
  }
  if (countsRows(aggregate)) {
    return true;
  }
  if (std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
          aggregate.call->inputs()[0]) == nullptr ||
      aggregate.rawInputTypes.size() != 1) {
    return false;
  }
  const auto& type = aggregate.rawInputTypes[0];
  // 128-bit decimals only add into the int128 sums.
  const auto kind = *kindOf(aggregate);
  return isDenseInput(type) ||
      (type->isLongDecimal() &&
       (kind == Kind::kSum || kind == Kind::kAvg || kind == Kind::kCount));
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
    if (!countsRows(aggregate)) {
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

namespace {

int64_t slotCount(const std::vector<DenseGroupby::Key>& keys) {
  int64_t slots = 1;
  for (const auto& key : keys) {
    slots *= key.range.high - key.range.low + 1;
    VELOX_CHECK_LE(slots, std::numeric_limits<cudf::size_type>::max());
  }
  return slots;
}

} // namespace

DenseGroupby::DenseGroupby(
    std::vector<Key> keys,
    std::vector<Aggregate> aggregates,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
    : keys_(std::move(keys)),
      slots_(slotCount(keys_)),
      rows_(slots_ * sizeof(int64_t), stream, mr),
      outOfRange_(sizeof(int), stream, mr) {
  VELOX_CHECK(!keys_.empty());
  strides_.resize(keys_.size());
  int64_t stride = 1;
  for (int64_t k = keys_.size() - 1; k >= 0; --k) {
    strides_[k] = stride;
    stride *= keys_[k].range.high - keys_[k].range.low + 1;
  }
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
    cuda::stream_ref stream) {
  const auto rowCount = input[keys_[0].channel].size();
  if (rowCount == 0) {
    return;
  }
  for (const auto& key : keys_) {
    VELOX_CHECK_EQ(
        input[key.channel].null_count(),
        0,
        "Dense aggregation keys cannot be null");
  }
  // One key indexes the slots directly; several combine into a slot column.
  cudf::column_view key = input[keys_[0].channel];
  int64_t low = keys_[0].range.low;
  std::optional<rmm::device_uvector<int32_t>> combined;
  if (keys_.size() > 1) {
    std::vector<cudf::column_view> columns;
    std::vector<int64_t> lows;
    std::vector<int64_t> sizes;
    for (const auto& k : keys_) {
      columns.push_back(input[k.channel]);
      lows.push_back(k.range.low);
      sizes.push_back(k.range.high - k.range.low + 1);
    }
    combined.emplace(rowCount, stream);
    dense::combine(
        columns, lows, sizes, strides_, slots_, combined->data(), stream);
    key = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32},
        rowCount,
        combined->data(),
        nullptr,
        0);
    low = 0;
  }
  if (slots_ <= dense::sharedSlots(states_.size())) {
    std::vector<dense::Target> targets;
    for (auto& state : states_) {
      targets.push_back(dense::Target{
          state.aggregate.kind == Kind::kCountRows
              ? cudf::column_view{}
              : input[state.aggregate.channel],
          state.aggregate.kind,
          state.floating,
          state.values.data(),
          static_cast<unsigned long long*>(state.counts.data())});
    }
    dense::addShared(
        key,
        low,
        slots_,
        targets,
        static_cast<unsigned long long*>(rows_.data()),
        static_cast<int*>(outOfRange_.data()),
        stream);
    // 'combined' is freed on return.
    stream.sync();
    return;
  }
  dense::addRows(
      key,
      low,
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
        low,
        slots_,
        input[state.aggregate.channel],
        state.aggregate.kind,
        state.floating,
        state.values.data(),
        static_cast<unsigned long long*>(state.counts.data()),
        stream);
  }
  if (combined.has_value()) {
    stream.sync();
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
      "A dense aggregation key is outside its range");

  const auto* rows = static_cast<const unsigned long long*>(rows_.data());
  const auto slots = dense::occupied(rows, slots_, stream, mr);
  const auto groups = static_cast<cudf::size_type>(slots.size());

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (size_t k = 0; k < keys_.size(); ++k) {
    const auto& key = keys_[k];
    const bool coded = !key.range.dictionary.empty();
    auto values = cudf::make_fixed_width_column(
        coded ? cudf::data_type{cudf::type_id::INT32}
              : veloxToCudfDataType(key.type),
        groups,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    dense::keys(
        slots,
        key.range.low,
        strides_[k],
        key.range.high - key.range.low + 1,
        values->mutable_view(),
        stream);
    if (coded) {
      auto dictionary = strings(key.range.dictionary, stream, mr);
      auto gathered = cudf::gather(
          cudf::table_view({dictionary->view()}),
          values->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      values = std::move(gathered->release()[0]);
    }
    columns.push_back(std::move(values));
  }

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
