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
#include "velox/experimental/supervelox/cudf/CacheConnector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <set>
#include <thread>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::supervelox::cudf_backend {

namespace {

cudf::data_type storedType(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return cudf::data_type{cudf::type_id::INT8};
    case DType::kInt16:
      return cudf::data_type{cudf::type_id::INT16};
    case DType::kInt32:
      return cudf::data_type{cudf::type_id::INT32};
    default:
      return cudf::data_type{cudf::type_id::INT64};
  }
}

/// The same values as 'column', which must be of 'type''s width, typed as 'type'.
std::unique_ptr<cudf::column> retype(std::unique_ptr<cudf::column> column, cudf::data_type type) {
  const auto rows = column->size();
  auto contents = column->release();
  return std::make_unique<cudf::column>(type, rows, std::move(*contents.data), std::move(*contents.null_mask), 0);
}

void collectFields(const core::TypedExprPtr& expr, std::vector<std::string>& names) {
  if (auto field = std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr); field && field->isInputColumn()) {
    names.push_back(field->name());
    return;
  }
  for (const auto& input : expr->inputs()) {
    collectFields(input, names);
  }
}

/// Whether 'filter' can be tested against a zone's integer {min, max}.
bool testsRanges(const common::Filter& filter) {
  switch (filter.kind()) {
    case common::FilterKind::kBigintRange:
    case common::FilterKind::kBigintMultiRange:
    case common::FilterKind::kBigintValuesUsingBitmask:
    case common::FilterKind::kBigintValuesUsingHashTable:
      return true;
    default:
      return false;
  }
}

using ZoneFilters = std::vector<std::pair<const Column*, const common::Filter*>>;

/// The rows of 'split' from the first to the last zone that every range
/// filter may match, as the row group statistics of a file would give.
std::pair<int64_t, int64_t> matchingRows(CacheConnector& connector, const ZoneFilters& filters, const CacheSplit& split) {
  constexpr int64_t kZone = CacheConnector::kZoneRows;
  if (filters.empty()) {
    return {split.begin, split.end};
  }
  int64_t first = split.end;
  int64_t last = split.begin;
  std::vector<const std::vector<std::pair<int64_t, int64_t>>*> zones;
  for (const auto& [column, _] : filters) {
    zones.push_back(&connector.zones(*column));
  }
  for (int64_t zone = split.begin / kZone; zone * kZone < split.end; ++zone) {
    bool match = true;
    for (size_t i = 0; i < filters.size() && match; ++i) {
      const auto& [low, high] = (*zones[i])[zone];
      match = filters[i].second->testInt64Range(low, high, false);
    }
    if (match) {
      first = std::min(first, std::max(split.begin, zone * kZone));
      last = std::max(last, std::min(split.end, (zone + 1) * kZone));
    }
  }
  return {first, last};
}

/// Rows [begin, begin + rows) of 'column' as stored, without copying.
cudf::column_view storedView(
    CacheConnector& connector,
    const Column& column,
    int64_t begin,
    int64_t rows,
    cudaStream_t stream) {
  if (column.kind == Kind::kString) {
    const cudf::column_view offsets(
        cudf::data_type{cudf::type_id::INT64},
        static_cast<cudf::size_type>(column.rows + 1),
        connector.offsets(column, stream),
        nullptr,
        0);
    const cudf::column_view all(
        cudf::data_type{cudf::type_id::STRING},
        static_cast<cudf::size_type>(column.rows),
        connector.data(column, stream),
        nullptr,
        0,
        0,
        {offsets});
    return cudf::slice(all, {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(begin + rows)}, stream)[0];
  }
  return cudf::column_view(
      storedType(column.dtype),
      static_cast<cudf::size_type>(rows),
      static_cast<const char*>(connector.data(column, stream)) + begin * dtypeSize(column.dtype),
      nullptr,
      0);
}

/// Serves a FusedScanHandle: views the stored columns of each split's
/// matching zones, finds the rows that pass in one kernel, compacts the
/// columns the outputs read, still encoded, and computes the expression
/// outputs from them in another kernel.
class FusedDataSource : public connector::DataSource {
 public:
  FusedDataSource(
      CacheConnector& connector,
      RowTypePtr outputType,
      const FusedScanHandle& handle,
      connector::ConnectorQueryCtx* connectorQueryCtx,
      bool forKeys = false)
      : connector_(connector),
        outputType_(std::move(outputType)),
        table_(handle.tableName()),
        input_(handle.input()),
        outputs_(handle.outputs()),
        transform_(handle.input(), handle.encodings(), handle.filters(), handle.columnFilters(), handle.expressions()),
        keyFilters_(
            forKeys || connector.store().rows(handle.tableName()) >= CacheConnector::kMinKeyFilterRows
                ? handle.keyFilters()
                : std::vector<FusedScanHandle::KeyFilter>{}),
        queryCtx_(connectorQueryCtx),
        forKeys_(forKeys),
        pool_(connectorQueryCtx->memoryPool()) {
    VELOX_CHECK_EQ(outputs_.size(), outputType_->size());
    for (const auto& name : input_->names()) {
      columns_.push_back(&connector_.store().metadata(table_, name));
    }
    for (size_t i = 0; i < keyFilters_.size(); ++i) {
      keyChannels_.push_back(static_cast<int32_t>(input_->getChildIdx(keyFilters_[i].column)));
      probeOrder_.push_back(i);
    }
    keys_.resize(keyFilters_.size());
    loaded_.resize(keyFilters_.size());
    std::stable_sort(probeOrder_.begin(), probeOrder_.end(), [&](size_t a, size_t b) {
      return connector_.store().rows(keyFilters_[a].keys->tableHandle()->name()) <
          connector_.store().rows(keyFilters_[b].keys->tableHandle()->name());
    });
    for (const auto& [subfield, filter] : handle.subfieldFilters()) {
      const Column& column = connector_.store().metadata(table_, subfield.baseName());
      if (testsRanges(*filter) && column.kind != Kind::kString && column.kind != Kind::kDictionary) {
        const auto [it, added] = filters_.emplace(subfield.clone(), filter->clone());
        zoneFilters_.emplace_back(&column, it->second.get());
      }
    }
    std::set<int32_t> kept(transform_.projectChannels().begin(), transform_.projectChannels().end());
    for (const auto& output : outputs_) {
      if (output.source != FusedScanHandle::Source::kExpression) {
        const auto channel = static_cast<int32_t>(input_->getChildIdx(output.column));
        channels_.push_back(channel);
        kept.insert(channel);
        ++uses_[channel];
      } else {
        channels_.push_back(-1);
      }
    }
    kept_.assign(kept.begin(), kept.end());
  }

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    split_ = std::dynamic_pointer_cast<CacheSplit>(split);
    VELOX_CHECK_NOT_NULL(split_, "Not a cache split: {}", split->toString());
    VELOX_CHECK_EQ(split_->table, table_);
  }

  std::optional<RowVectorPtr> next(uint64_t /*size*/, ContinueFuture& /*future*/) override {
    if (!split_) {
      return nullptr;
    }
    const auto split = std::move(split_);
    const auto [begin, end] = matchingRows(connector_, zoneFilters_, *split);
    if (begin >= end) {
      return nullptr;
    }
    cudf_velox::ensureCudaContextForThread();
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    std::vector<cudf::column_view> views;
    for (const auto* column : columns_) {
      views.push_back(storedView(connector_, *column, begin, end - begin, stream.get()));
    }
    auto rows = static_cast<cudf::size_type>(end - begin);
    std::vector<std::unique_ptr<cudf::column>> owned(views.size());
    std::unique_ptr<cudf::column> mask = transform_.filters() ? transform_.mask(views, stream, mr) : nullptr;
    if (!keyFilters_.empty()) {
      mask = keyMask(views, rows, std::move(mask), stream, mr);
    }
    if (mask != nullptr) {
      std::vector<cudf::column_view> read;
      for (const auto channel : kept_) {
        read.push_back(views[channel]);
      }
      auto passed = cudf::apply_retention_mask(cudf::table_view(read), mask->view(), stream, mr)->release();
      for (size_t i = 0; i < kept_.size(); ++i) {
        views[kept_[i]] = passed[i]->view();
        owned[kept_[i]] = std::move(passed[i]);
      }
      rows = kept_.empty()
          ? cudf::apply_retention_mask(cudf::table_view({mask->view()}), mask->view(), stream, mr)->num_rows()
          : views[kept_[0]].size();
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    if (rows == 0) {
      for (const auto& type : outputType_->children()) {
        columns.push_back(cudf::make_empty_column(cudf_velox::veloxToCudfDataType(type)));
      }
    } else {
      auto computed = transform_.project(views, rows, stream, mr);
      size_t next = 0;
      auto uses = uses_;
      for (size_t i = 0; i < outputs_.size(); ++i) {
        const auto channel = channels_[i];
        if (channel < 0) {
          columns.push_back(std::move(computed[next++]));
          continue;
        }
        const Column& column = *columns_[channel];
        if (outputs_[i].source == FusedScanHandle::Source::kString && column.kind == Kind::kDictionary) {
          auto gathered = cudf::gather(
              cudf::table_view({connector_.dictionary(column).view()}),
              views[channel],
              cudf::out_of_bounds_policy::DONT_CHECK,
              stream,
              mr);
          columns.push_back(std::move(gathered->release()[0]));
        } else if (owned[channel] && --uses[channel] == 0) {
          columns.push_back(std::move(owned[channel]));
        } else {
          columns.push_back(std::make_unique<cudf::column>(views[channel], stream, mr));
        }
      }
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    stream.sync();
    completedRows_ += end - begin;
    const auto size = table->num_rows();
    return std::make_shared<cudf_velox::CudfVector>(pool_, outputType_, size, std::move(table), stream);
  }

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override {
    return {};
  }

  void addDynamicFilter(column_index_t /*outputChannel*/, const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_UNSUPPORTED("The cache connector takes no dynamic filters");
  }

 private:
  /// The keys of key filter 'i', got on first use, or null if they are more
  /// than one in kMaxKeyFraction of the rows of the table they come from. As
  /// a foreign key is about as likely to have any of those rows' keys, that
  /// many pass too many rows to pay for probing, unless this scan gets keys
  /// for another, which then probes fewer.
  const CacheConnector::JoinKeys* keys(size_t i, const std::vector<cudf::column_view>& views) {
    constexpr int64_t kMaxKeyFraction = 6;
    if (!loaded_[i]) {
      loaded_[i] = true;
      const auto& filter = keyFilters_[i];
      auto keys = connector_.joinKeys(queryCtx_->queryId(), filter, views[keyChannels_[i]].type(), queryCtx_);
      if (forKeys_ ||
          keys->values->size() * kMaxKeyFraction <= connector_.store().rows(filter.keys->tableHandle()->name())) {
        keys_[i] = std::move(keys);
      }
    }
    return keys_[i].get();
  }

  /// 'mask', or all of 'rows' if null, narrowed to the rows whose keys the
  /// key filters have. The filters over the smallest key tables go first and
  /// each probes only the rows the ones before it passed. Once at most one
  /// in kSelective rows remain, the rest are not worth getting keys for.
  std::unique_ptr<cudf::column> keyMask(
      const std::vector<cudf::column_view>& views,
      cudf::size_type rows,
      std::unique_ptr<cudf::column> mask,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) {
    // The rows that pass so far, or null for all of them.
    std::unique_ptr<cudf::column> candidates;
    if (mask != nullptr) {
      const cudf::numeric_scalar<cudf::size_type> zero(0, true, stream);
      auto all = cudf::sequence(rows, zero, stream, mr);
      candidates = std::move(
          cudf::apply_retention_mask(cudf::table_view({all->view()}), mask->view(), stream, mr)->release()[0]);
    }
    constexpr int64_t kSelective = 8;
    bool probed = false;
    for (const auto i : probeOrder_) {
      if (candidates != nullptr && static_cast<int64_t>(candidates->size()) * kSelective <= rows) {
        break;
      }
      const auto* keys = this->keys(i, views);
      if (keys == nullptr) {
        continue;
      }
      probed = true;
      if (candidates == nullptr) {
        auto passing = keys->set->semi_join(cudf::table_view({views[keyChannels_[i]]}), stream, mr);
        candidates = std::make_unique<cudf::column>(
            std::move(*passing), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr), 0);
        continue;
      }
      auto probe = cudf::gather(
          cudf::table_view({views[keyChannels_[i]]}), candidates->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
      const auto passing = keys->set->semi_join(probe->view(), stream, mr);
      const cudf::column_view positions(
          cudf::data_type{cudf::type_id::INT32}, static_cast<cudf::size_type>(passing->size()), passing->data(), nullptr, 0);
      candidates = std::move(cudf::gather(
                                 cudf::table_view({candidates->view()}),
                                 positions,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr)
                                 ->release()[0]);
    }
    if (!probed) {
      return mask;
    }
    const cudf::numeric_scalar<bool> no(false, true, stream);
    const cudf::numeric_scalar<bool> yes(true, true, stream);
    auto none = cudf::make_column_from_scalar(no, rows, stream, mr);
    return std::move(
        cudf::scatter({std::cref<const cudf::scalar>(yes)}, candidates->view(), cudf::table_view({none->view()}), stream, mr)
            ->release()[0]);
  }

  CacheConnector& connector_;
  const RowTypePtr outputType_;
  const std::string table_;
  const RowTypePtr input_;
  const std::vector<FusedScanHandle::Output> outputs_;
  const cudf_velox::FusedTransform transform_;
  const std::vector<FusedScanHandle::KeyFilter> keyFilters_;
  connector::ConnectorQueryCtx* const queryCtx_;
  const bool forKeys_;
  memory::MemoryPool* const pool_;
  std::vector<int32_t> keyChannels_;
  std::vector<std::shared_ptr<const CacheConnector::JoinKeys>> keys_;
  std::vector<bool> loaded_;
  std::vector<size_t> probeOrder_;
  std::vector<const Column*> columns_;
  common::SubfieldFilters filters_;
  ZoneFilters zoneFilters_;
  /// Per output, its input channel, or -1 for an expression.
  std::vector<int32_t> channels_;
  /// The input channels the outputs read, compacted after filtering.
  std::vector<int32_t> kept_;
  std::map<int32_t, int32_t> uses_;
  std::shared_ptr<CacheSplit> split_;
  uint64_t completedRows_{0};
};

/// Reads the scanned columns and any others the filters use, keeps the
/// zones of each split that range filters may match, evaluates the filters on
/// the GPU and returns the scanned columns of the rows that pass.
class CacheDataSource : public connector::DataSource {
 public:
  CacheDataSource(
      CacheConnector& connector,
      RowTypePtr outputType,
      std::string table,
      std::vector<std::string> columns,
      const connector::hive::HiveTableHandle* hive,
      memory::MemoryPool* pool)
      : connector_(connector),
        outputType_(std::move(outputType)),
        table_(std::move(table)),
        readNames_(std::move(columns)),
        readTypes_(outputType_->children()),
        pool_(pool) {
    if (hive == nullptr) {
      return;
    }
    if (hive->tableParameters().count(CacheConnector::kZoneOnly)) {
      VELOX_CHECK_NULL(hive->remainingFilter(), "A zone only scan has no remaining filter");
      for (const auto& [subfield, filter] : hive->subfieldFilters()) {
        const Column& column = connector_.store().metadata(table_, subfield.baseName());
        if (testsRanges(*filter) && column.kind != Kind::kString && column.kind != Kind::kDictionary) {
          const auto [it, added] = filters_.emplace(subfield.clone(), filter->clone());
          zoneFilters_.emplace_back(&column, it->second.get());
        }
      }
      return;
    }
    std::vector<std::string> filterColumns;
    for (const auto& [subfield, filter] : hive->subfieldFilters()) {
      VELOX_CHECK_EQ(subfield.path().size(), 1, "Filter on a nested field {}", subfield.toString());
      filterColumns.push_back(subfield.baseName());
      filters_.emplace(subfield.clone(), filter->clone());
    }
    if (hive->remainingFilter()) {
      collectFields(hive->remainingFilter(), filterColumns);
    }
    for (const auto& name : filterColumns) {
      if (std::find(readNames_.begin(), readNames_.end(), name) == readNames_.end()) {
        VELOX_CHECK_NOT_NULL(hive->dataColumns(), "No type for filter column {}", name);
        readNames_.push_back(name);
        readTypes_.push_back(hive->dataColumns()->findChild(name));
      }
    }
    const auto readType = ROW(std::vector<std::string>(readNames_), std::vector<TypePtr>(readTypes_));
    if (!filters_.empty()) {
      filterAst_ = &cudf_velox::createAstFromSubfieldFilters(filters_, tree_, scalars_, readType);
      for (const auto& [subfield, filter] : filters_) {
        const Column& column = connector_.store().metadata(table_, subfield.baseName());
        if (testsRanges(*filter) && column.kind != Kind::kString && column.kind != Kind::kDictionary &&
            !readType->findChild(subfield.baseName())->isDouble()) {
          zoneFilters_.emplace_back(&column, filter.get());
        }
      }
    }
    if (hive->remainingFilter()) {
      queryCtx_ = core::QueryCtx::create();
      remainingFilter_ = cudf_velox::createCudfExpression(
          expression::optimize(hive->remainingFilter(), queryCtx_.get(), pool_),
          readType,
          pool_,
          queryCtx_->queryConfig());
    }
  }

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    split_ = std::dynamic_pointer_cast<CacheSplit>(split);
    VELOX_CHECK_NOT_NULL(split_, "Not a cache split: {}", split->toString());
    VELOX_CHECK_EQ(split_->table, table_);
  }

  std::optional<RowVectorPtr> next(uint64_t /*size*/, ContinueFuture& /*future*/) override {
    if (!split_) {
      return nullptr;
    }
    const auto split = std::move(split_);
    const auto [begin, end] = matchingRows(connector_, zoneFilters_, *split);
    if (begin >= end) {
      return nullptr;
    }
    cudf_velox::ensureCudaContextForThread();
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (size_t i = 0; i < readNames_.size(); ++i) {
      columns.push_back(decode(connector_.store().column(table_, readNames_[i]), readTypes_[i], begin, end - begin, stream));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    if (filterAst_) {
      auto mask = cudf::compute_column(table->view(), *filterAst_, stream, mr);
      table = cudf::apply_boolean_mask(table->view(), mask->view(), stream, mr);
    }
    if (remainingFilter_) {
      std::vector<cudf::column_view> views;
      for (const auto& column : table->view()) {
        views.push_back(column);
      }
      auto mask = remainingFilter_->eval(views, stream, mr);
      table = cudf::apply_boolean_mask(table->view(), cudf_velox::asView(mask), stream, mr);
    }
    if (static_cast<size_t>(table->num_columns()) > outputType_->size()) {
      auto all = table->release();
      all.resize(outputType_->size());
      table = std::make_unique<cudf::table>(std::move(all));
    }
    stream.sync();
    completedRows_ += end - begin;
    const auto rows = table->num_rows();
    return std::make_shared<cudf_velox::CudfVector>(pool_, outputType_, rows, std::move(table), stream);
  }

  uint64_t getCompletedBytes() override {
    return 0;
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override {
    return {};
  }

  void addDynamicFilter(column_index_t /*outputChannel*/, const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_UNSUPPORTED("The cache connector takes no dynamic filters");
  }

 private:
  /// Rows [begin, begin + rows) of 'column' as 'type'.
  std::unique_ptr<cudf::column> decode(
      const Column& column,
      const TypePtr& type,
      int64_t begin,
      int64_t rows,
      cuda::stream_ref stream) {
    const auto mr = cudf_velox::get_output_mr();
    const auto size = static_cast<cudf::size_type>(rows);
    if (column.kind == Kind::kString) {
      VELOX_CHECK(type->isVarchar(), "{}.{} is a string, not {}", table_, column.name, type->toString());
      const cudf::column_view offsets(
          cudf::data_type{cudf::type_id::INT64},
          static_cast<cudf::size_type>(column.rows + 1),
          connector_.offsets(column, stream.get()),
          nullptr,
          0);
      const cudf::column_view all(
          cudf::data_type{cudf::type_id::STRING},
          static_cast<cudf::size_type>(column.rows),
          connector_.data(column, stream.get()),
          nullptr,
          0,
          0,
          {offsets});
      const auto piece = cudf::slice(all, {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(begin + rows)}, stream)[0];
      return std::make_unique<cudf::column>(piece, stream, mr);
    }
    const auto width = dtypeSize(column.dtype);
    const cudf::column_view stored(
        storedType(column.dtype),
        size,
        static_cast<const char*>(connector_.data(column, stream.get())) + begin * width,
        nullptr,
        0);
    if (column.kind == Kind::kDictionary && type->isInteger()) {
      return cudf::cast(stored, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    }
    if (column.kind == Kind::kDictionary) {
      VELOX_CHECK(type->isVarchar(), "{}.{} is a dictionary, not {}", table_, column.name, type->toString());
      auto gathered = cudf::gather(
          cudf::table_view({connector_.dictionary(column).view()}),
          stored,
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      return std::move(gathered->release()[0]);
    }
    const auto target = cudf_velox::veloxToCudfDataType(type);
    if (column.kind == Kind::kDecimal) {
      if (type->kind() == TypeKind::DOUBLE) {
        auto values = cudf::cast(stored, target, stream, mr);
        const cudf::numeric_scalar<double> divisor(std::pow(10.0, column.scale), true, stream);
        return cudf::binary_operation(*values, divisor, cudf::binary_operator::TRUE_DIV, target, stream, mr);
      }
      VELOX_CHECK(type->isDecimal(), "{}.{} is a decimal, not {}", table_, column.name, type->toString());
      VELOX_CHECK_EQ(-target.scale(), column.scale, "{}.{} has another scale than {}", table_, column.name, type->toString());
      auto wide = retype(cudf::cast(stored, cudf::data_type{cudf::type_id::INT64}, stream, mr), cudf::data_type{cudf::type_id::DECIMAL64, target.scale()});
      return target.id() == cudf::type_id::DECIMAL64 ? std::move(wide) : cudf::cast(*wide, target, stream, mr);
    }
    if (column.kind == Kind::kDate) {
      VELOX_CHECK(type->isDate(), "{}.{} is a date, not {}", table_, column.name, type->toString());
      return retype(cudf::cast(stored, cudf::data_type{cudf::type_id::INT32}, stream, mr), target);
    }
    return cudf::cast(stored, target, stream, mr);
  }

  CacheConnector& connector_;
  const RowTypePtr outputType_;
  const std::string table_;
  /// The scanned columns, then the other columns the filters use.
  std::vector<std::string> readNames_;
  std::vector<TypePtr> readTypes_;
  memory::MemoryPool* const pool_;
  common::SubfieldFilters filters_;
  cudf::ast::tree tree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  const cudf::ast::expression* filterAst_{nullptr};
  std::vector<std::pair<const Column*, const common::Filter*>> zoneFilters_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::shared_ptr<cudf_velox::CudfExpression> remainingFilter_;
  std::shared_ptr<CacheSplit> split_;
  uint64_t completedRows_{0};
};

} // namespace

std::string CacheSplit::toString() const {
  return fmt::format("[cache split: {} rows {} to {}]", table, begin, end);
}

CacheConnector::CacheConnector(const std::string& id, ColumnStore& store, DeviceCache& cache, int64_t batchRows)
    : Connector(id), store_(store), cache_(cache), batchRows_(batchRows) {}

FusedScanHandle::FusedScanHandle(
    const connector::hive::HiveTableHandle& scan,
    common::SubfieldFilters zoneFilters,
    RowTypePtr input,
    cudf_velox::Encodings encodings,
    std::vector<core::TypedExprPtr> filters,
    std::vector<cudf_velox::ColumnFilter> columnFilters,
    std::vector<Output> outputs)
    : HiveTableHandle(
          scan.connectorId(),
          scan.tableName(),
          std::move(zoneFilters),
          nullptr,
          scan.dataColumns(),
          scan.indexColumns(),
          scan.tableParameters(),
          scan.hiveFilterColumnHandles(),
          scan.sampleRate(),
          scan.dbName(),
          scan.dataColumnFieldIds()),
      input_(std::move(input)),
      encodings_(std::move(encodings)),
      filters_(std::move(filters)),
      columnFilters_(std::move(columnFilters)),
      outputs_(std::move(outputs)) {}

std::shared_ptr<const FusedScanHandle> FusedScanHandle::withKeyFilter(KeyFilter filter) const {
  common::SubfieldFilters zoneFilters;
  for (const auto& [subfield, zoneFilter] : subfieldFilters()) {
    zoneFilters.emplace(subfield.clone(), zoneFilter->clone());
  }
  auto out = std::make_shared<FusedScanHandle>(
      *this, std::move(zoneFilters), input_, encodings_, filters_, columnFilters_, outputs_);
  out->keyFilters_ = keyFilters_;
  out->keyFilters_.push_back(std::move(filter));
  return out;
}

std::vector<core::TypedExprPtr> FusedScanHandle::expressions() const {
  std::vector<core::TypedExprPtr> out;
  for (const auto& output : outputs_) {
    if (output.source == Source::kExpression) {
      out.push_back(output.expression);
    }
  }
  return out;
}

std::string FusedScanHandle::toString() const {
  std::vector<std::string> outputs;
  for (const auto& output : outputs_) {
    outputs.push_back(output.source == Source::kExpression ? output.expression->toString()
                          : output.source == Source::kStored ? output.column + " as stored"
                                                             : output.column);
  }
  std::vector<std::string> filters;
  for (const auto& filter : filters_) {
    filters.push_back(filter->toString());
  }
  for (const auto& filter : columnFilters_) {
    filters.push_back(fmt::format("{} {}", filter.column, filter.filter->toString()));
  }
  for (const auto& filter : keyFilters_) {
    filters.push_back(fmt::format("{} in {} of scan {}", filter.column, filter.key, filter.keys->id()));
  }
  return fmt::format(
      "fused scan of {} reading {}, filters [{}], outputs [{}]",
      tableName(),
      input_->toString(),
      fmt::join(filters, ", "),
      fmt::join(outputs, ", "));
}

std::shared_ptr<const CacheConnector::JoinKeys> CacheConnector::joinKeys(
    const std::string& queryId,
    const FusedScanHandle::KeyFilter& filter,
    cudf::data_type type,
    connector::ConnectorQueryCtx* connectorQueryCtx) {
  std::lock_guard<std::recursive_mutex> l(keysMutex_);
  if (queryId != keysQuery_) {
    joinKeys_.clear();
    keysQuery_ = queryId;
  }
  const auto name = fmt::format("{} {} {}", filter.keys->id(), filter.key, static_cast<int32_t>(type.id()));
  if (const auto it = joinKeys_.find(name); it != joinKeys_.end()) {
    return it->second;
  }
  const auto& scan = *filter.keys;
  const auto* fused = dynamic_cast<const FusedScanHandle*>(scan.tableHandle().get());
  std::unique_ptr<connector::DataSource> source = fused != nullptr
      ? std::make_unique<FusedDataSource>(*this, scan.outputType(), *fused, connectorQueryCtx, true)
      : createDataSource(scan.outputType(), scan.tableHandle(), scan.assignments(), connectorQueryCtx);
  const auto channel = scan.outputType()->getChildIdx(filter.key);
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  std::vector<std::unique_ptr<cudf::column>> pieces;
  ContinueFuture future;
  for (const auto& split : splits(scan.tableHandle()->name())) {
    source->addSplit(split);
    while (auto batch = source->next(0, future)) {
      if (*batch == nullptr) {
        break;
      }
      auto* vector = dynamic_cast<cudf_velox::CudfVector*>(batch->get());
      VELOX_CHECK_NOT_NULL(vector, "Join keys from scan {} are not on the GPU", scan.id());
      pieces.push_back(cudf::cast(vector->getTableView().column(channel), type, stream, mr));
    }
  }
  std::vector<cudf::column_view> views;
  for (const auto& piece : pieces) {
    views.push_back(piece->view());
  }
  auto keys = std::make_shared<JoinKeys>();
  keys->values = views.empty() ? cudf::make_empty_column(type) : cudf::concatenate(views, stream, mr);
  keys->set = std::make_unique<cudf::filtered_join>(
      cudf::table_view({keys->values->view()}), cudf::null_equality::UNEQUAL, stream, mr);
  stream.sync();
  joinKeys_.emplace(name, keys);
  return keys;
}

TypePtr CacheConnector::storedVeloxType(const Column& column) {
  if (column.kind == Kind::kString) {
    return VARCHAR();
  }
  switch (column.dtype) {
    case DType::kInt8:
      return TINYINT();
    case DType::kInt16:
      return SMALLINT();
    case DType::kInt32:
      return INTEGER();
    default:
      return BIGINT();
  }
}

std::unique_ptr<connector::DataSource> CacheConnector::createDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    connector::ConnectorQueryCtx* connectorQueryCtx) {
  if (const auto* fused = dynamic_cast<const FusedScanHandle*>(tableHandle.get())) {
    return std::make_unique<FusedDataSource>(*this, outputType, *fused, connectorQueryCtx);
  }
  auto hive = std::dynamic_pointer_cast<const connector::hive::HiveTableHandle>(tableHandle);
  std::vector<std::string> columns;
  for (const auto& name : outputType->names()) {
    columns.push_back(columnHandles.at(name)->name());
  }
  return std::make_unique<CacheDataSource>(
      *this, outputType, tableHandle->name(), std::move(columns), hive.get(), connectorQueryCtx->memoryPool());
}

std::vector<std::shared_ptr<connector::ConnectorSplit>> CacheConnector::splits(
    const std::string& table,
    const std::vector<std::string>& keys) {
  const int64_t rows = store_.rows(table);
  std::vector<const Column*> columns;
  for (const auto& key : keys) {
    columns.push_back(&store_.column(table, key));
  }
  auto sameKeys = [&](int64_t row) {
    for (const auto* column : columns) {
      if (column->value(row) != column->value(row - 1)) {
        return false;
      }
    }
    return true;
  };
  std::vector<std::shared_ptr<connector::ConnectorSplit>> out;
  for (int64_t begin = 0; begin < rows;) {
    int64_t end = std::min(rows, begin + batchRows_);
    while (!columns.empty() && end < rows && sameKeys(end)) {
      ++end;
    }
    out.push_back(std::make_shared<CacheSplit>(connectorId(), table, begin, end));
    begin = end;
  }
  return out;
}

bool CacheConnector::clustered(const std::string& table, const std::string& name) {
  const Column& metadata = store_.metadata(table, name);
  // One bit per value from min to max, set where a run of the value starts.
  constexpr int64_t kMaxBits = int64_t{1} << 34;
  if (metadata.kind == Kind::kString || metadata.kind == Kind::kDictionary || metadata.max - metadata.min >= kMaxBits) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = clustered_.find(table + "." + name);
  if (found != clustered_.end()) {
    return found->second;
  }
  const Column& column = store_.column(table, name);
  std::vector<std::atomic<uint64_t>> seen((column.max - column.min) / 64 + 1);
  constexpr int64_t kThreads = 32;
  const int64_t perThread = (column.rows + kThreads - 1) / kThreads;
  std::atomic<bool> adjacent{true};
  std::vector<std::thread> threads;
  for (int64_t first = 0; first < column.rows; first += perThread) {
    threads.emplace_back([&, first] {
      const int64_t last = std::min(column.rows, first + perThread);
      for (int64_t row = first; row < last && adjacent.load(std::memory_order_relaxed); ++row) {
        const int64_t value = column.value(row);
        if (row > 0 && column.value(row - 1) == value) {
          continue;
        }
        const int64_t bit = value - column.min;
        const uint64_t mask = uint64_t{1} << (bit % 64);
        if (seen[bit / 64].fetch_or(mask, std::memory_order_relaxed) & mask) {
          adjacent = false;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  clustered_[table + "." + name] = adjacent;
  return adjacent;
}

const void* CacheConnector::data(const Column& column, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_.get(column.table + "." + column.name, column.data, column.bytes, stream);
}

const int64_t* CacheConnector::offsets(const Column& column, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<const int64_t*>(cache_.get(
      column.table + "." + column.name + ".offsets", column.offsets, (column.rows + 1) * sizeof(int64_t), stream));
}

const std::vector<std::pair<int64_t, int64_t>>& CacheConnector::zones(const Column& column) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& slot = zones_[column.table + "." + column.name];
  if (slot.empty()) {
    const Column& loaded = store_.column(column.table, column.name);
    const int64_t count = (loaded.rows + kZoneRows - 1) / kZoneRows;
    slot.resize(count);
    std::vector<std::thread> threads;
    const int64_t perThread = (count + 31) / 32;
    for (int64_t first = 0; first < count; first += perThread) {
      threads.emplace_back([&, first] {
        for (int64_t zone = first; zone < std::min(count, first + perThread); ++zone) {
          int64_t low = std::numeric_limits<int64_t>::max();
          int64_t high = std::numeric_limits<int64_t>::min();
          for (int64_t row = zone * kZoneRows; row < std::min(loaded.rows, (zone + 1) * kZoneRows); ++row) {
            const int64_t value = loaded.value(row);
            low = std::min(low, value);
            high = std::max(high, value);
          }
          slot[zone] = {low, high};
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  return slot;
}

const cudf::column& CacheConnector::dictionary(const Column& column) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& slot = dictionaries_[column.table + "." + column.name];
  if (!slot) {
    std::vector<int32_t> offsets{0};
    std::string chars;
    for (const auto& value : column.dictionary) {
      chars += value;
      offsets.push_back(static_cast<int32_t>(chars.size()));
    }
    auto stream = cudf_velox::getDefaultStreamForCurrentThread();
    const auto count = static_cast<cudf::size_type>(column.dictionary.size());
    auto offsetsColumn = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        count + 1,
        rmm::device_buffer(offsets.data(), offsets.size() * sizeof(int32_t), stream),
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream),
        0);
    slot = cudf::make_strings_column(
        count,
        std::move(offsetsColumn),
        rmm::device_buffer(chars.data(), chars.size(), stream),
        0,
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
    stream.sync();
  }
  return *slot;
}

} // namespace facebook::velox::supervelox::cudf_backend
