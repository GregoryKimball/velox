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
#include <thread>

#include <fmt/format.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
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
    const auto [begin, end] = matchingRows(*split);
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
  /// The rows of 'split' from the first to the last zone that every range
  /// filter may match, as the row group statistics of a file would give.
  std::pair<int64_t, int64_t> matchingRows(const CacheSplit& split) {
    constexpr int64_t kZone = CacheConnector::kZoneRows;
    int64_t first = split.end;
    int64_t last = split.begin;
    if (zoneFilters_.empty()) {
      return {split.begin, split.end};
    }
    std::vector<const std::vector<std::pair<int64_t, int64_t>>*> zones;
    for (const auto& [column, _] : zoneFilters_) {
      zones.push_back(&connector_.zones(*column));
    }
    for (int64_t zone = split.begin / kZone; zone * kZone < split.end; ++zone) {
      bool match = true;
      for (size_t i = 0; i < zoneFilters_.size() && match; ++i) {
        const auto& [low, high] = (*zones[i])[zone];
        match = zoneFilters_[i].second->testInt64Range(low, high, false);
      }
      if (match) {
        first = std::min(first, std::max(split.begin, zone * kZone));
        last = std::max(last, std::min(split.end, (zone + 1) * kZone));
      }
    }
    return {first, last};
  }

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

std::unique_ptr<connector::DataSource> CacheConnector::createDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    connector::ConnectorQueryCtx* connectorQueryCtx) {
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
