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

#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"

#include "velox/type/Filter.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::cudf_velox::connector::encoded {

using namespace facebook::velox::connector;

namespace {

// Rows per batch when the caller asks for fewer.
constexpr uint64_t kMinBatchRows = 16384;

inline int64_t
unpack(const uint64_t* words, const SegmentParams& p, int64_t index) {
  if (p.bitWidth == 0) {
    return p.base;
  }
  const uint64_t bit = p.bitOffset + index * p.bitWidth;
  const uint64_t word = bit >> 6;
  const uint32_t shift = bit & 63;
  uint64_t value = words[word] >> shift;
  if (shift + p.bitWidth > 64) {
    value |= words[word + 1] << (64 - shift);
  }
  const uint64_t mask = p.bitWidth == 64 ? ~0ULL : ((1ULL << p.bitWidth) - 1);
  return p.base + static_cast<int64_t>(value & mask);
}

// The host string data outlives every query, so views of it release nothing.
struct NoopReleaser {
  void addRef() const {}
  void release() const {}
};

StringView stringAt(const EncodedColumn& column, int64_t index) {
  const auto begin = column.hostOffsets[index];
  return StringView(
      column.hostChars.data() + begin,
      static_cast<int32_t>(column.hostOffsets[index + 1] - begin));
}

// Keeps the rows of 'rows' for which 'keep' returns true, in order.
template <typename Keep>
void compact(std::vector<int32_t>& rows, Keep keep) {
  size_t kept = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto row = rows[i];
    if (keep(row)) {
      rows[kept++] = row;
    }
  }
  rows.resize(kept);
}

} // namespace

EncodedCpuDataSource::EncodedCpuDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx)
    : outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      spec_(outputType, tableHandle, columnHandles, connectorQueryCtx) {
  for (const auto& [subfield, filter] : spec_.subfieldFilters) {
    const auto& column = spec_.table->column(subfield.baseName());
    VELOX_CHECK(
        column.codec != Codec::kForBitpack || !column.hostPacked.empty() ||
            spec_.table->numRows == 0,
        "Table {} has no host copy; call EncodedColumnCache::copyToHost()",
        spec_.table->name);
    ColumnFilter columnFilter{&column, filter.get(), {}};
    if (column.codec == Codec::kDictionary) {
      const auto numValues = column.hostOffsets.size() - 1;
      columnFilter.codePasses.resize(numValues);
      for (size_t code = 0; code < numValues; ++code) {
        const auto value = stringAt(column, code);
        columnFilter.codePasses[code] =
            filter->testBytes(value.data(), value.size());
      }
    }
    columnFilters_.push_back(std::move(columnFilter));
  }
  if (spec_.remainingFilter) {
    remainingFilterExprSet_ =
        expressionEvaluator_->compile(spec_.remainingFilter);
  }
}

void EncodedCpuDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  split_ = std::dynamic_pointer_cast<EncodedConnectorSplit>(split);
  VELOX_CHECK_NOT_NULL(split_, "Expected an EncodedConnectorSplit");
  VELOX_CHECK_EQ(split_->table, spec_.table->name);
  segments_ = spec_.rejectsAllRows
      ? std::vector<int32_t>{}
      : spec_.pruneSegments(split_->segmentBegin, split_->segmentEnd);
  segmentsScanned_ += segments_.size();
  segmentsPruned_ +=
      (split_->segmentEnd - split_->segmentBegin) - segments_.size();
  segmentIndex_ = 0;
  rowInSegment_ = 0;
}

const VectorPtr& EncodedCpuDataSource::dictionaryValues(
    const EncodedColumn& column) {
  auto it = dictionaryValues_.find(&column);
  if (it != dictionaryValues_.end()) {
    return it->second;
  }
  const auto numValues =
      static_cast<vector_size_t>(column.hostOffsets.size() - 1);
  auto values =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), numValues, pool_);
  values->addStringBuffer(BufferView<NoopReleaser>::create(
      reinterpret_cast<const uint8_t*>(column.hostChars.data()),
      column.hostChars.size(),
      NoopReleaser{}));
  for (vector_size_t i = 0; i < numValues; ++i) {
    values->setNoCopy(i, stringAt(column, i));
  }
  return dictionaryValues_.emplace(&column, std::move(values)).first->second;
}

VectorPtr EncodedCpuDataSource::decode(
    const EncodedColumn& column,
    const TypePtr& type,
    int32_t segment,
    const std::vector<int32_t>& rows) {
  const auto size = static_cast<vector_size_t>(rows.size());
  switch (column.codec) {
    case Codec::kForBitpack: {
      const auto& params = column.hostSegmentParams[segment];
      const auto* words = column.hostPacked.data();
      auto fill = [&](auto* out) {
        for (vector_size_t i = 0; i < size; ++i) {
          out[i] = static_cast<std::remove_reference_t<decltype(out[0])>>(
              unpack(words, params, rows[i]));
        }
      };
      if (type->kind() == TypeKind::BIGINT) {
        auto vector = BaseVector::create<FlatVector<int64_t>>(type, size, pool_);
        fill(vector->mutableRawValues());
        return vector;
      }
      VELOX_CHECK_EQ(
          type->kind(),
          TypeKind::INTEGER,
          "Unsupported type for a bit-packed column: {}",
          type->toString());
      auto vector = BaseVector::create<FlatVector<int32_t>>(type, size, pool_);
      fill(vector->mutableRawValues());
      return vector;
    }
    case Codec::kDictionary: {
      const auto& params = column.hostSegmentParams[segment];
      const auto* words = column.hostPacked.data();
      auto indices = allocateIndices(size, pool_);
      auto* rawIndices = indices->asMutable<vector_size_t>();
      for (vector_size_t i = 0; i < size; ++i) {
        rawIndices[i] =
            static_cast<vector_size_t>(unpack(words, params, rows[i]));
      }
      return BaseVector::wrapInDictionary(
          nullptr, std::move(indices), size, dictionaryValues(column));
    }
    case Codec::kPlain: {
      auto vector = BaseVector::create<FlatVector<StringView>>(type, size, pool_);
      auto& buffer = charBuffers_[&column];
      if (buffer == nullptr) {
        buffer = BufferView<NoopReleaser>::create(
            reinterpret_cast<const uint8_t*>(column.hostChars.data()),
            column.hostChars.size(),
            NoopReleaser{});
      }
      vector->addStringBuffer(buffer);
      const int64_t base =
          static_cast<int64_t>(segment) * spec_.table->segmentRows;
      for (vector_size_t i = 0; i < size; ++i) {
        vector->setNoCopy(i, stringAt(column, base + rows[i]));
      }
      return vector;
    }
  }
  VELOX_UNREACHABLE();
}

RowVectorPtr EncodedCpuDataSource::nextBatch(uint64_t size) {
  const auto batchRows = std::max(size, kMinBatchRows);
  const auto segmentRows = spec_.table->segmentRows;
  std::vector<int32_t> rows;
  while (segmentIndex_ < segments_.size()) {
    const auto segment = segments_[segmentIndex_];
    const auto begin = rowInSegment_;
    const auto end = static_cast<int32_t>(std::min<int64_t>(
        spec_.table->segmentSize(segment), begin + batchRows));
    if (end >= spec_.table->segmentSize(segment)) {
      ++segmentIndex_;
      rowInSegment_ = 0;
    } else {
      rowInSegment_ = end;
    }
    rows.resize(end - begin);
    std::iota(rows.begin(), rows.end(), begin);
    rowsDecodedForFilter_ += rows.size();

    for (const auto& columnFilter : columnFilters_) {
      const auto& column = *columnFilter.column;
      const auto* filter = columnFilter.filter;
      switch (column.codec) {
        case Codec::kForBitpack: {
          const auto& params = column.hostSegmentParams[segment];
          const auto* words = column.hostPacked.data();
          if (const auto* range =
                  dynamic_cast<const common::BigintRange*>(filter)) {
            const auto lower = range->lower();
            const auto upper = range->upper();
            compact(rows, [&](int32_t row) {
              const auto value = unpack(words, params, row);
              return value >= lower && value <= upper;
            });
          } else {
            compact(rows, [&](int32_t row) {
              return filter->testInt64(unpack(words, params, row));
            });
          }
          break;
        }
        case Codec::kDictionary: {
          const auto& params = column.hostSegmentParams[segment];
          const auto* words = column.hostPacked.data();
          const auto& passes = columnFilter.codePasses;
          compact(rows, [&](int32_t row) {
            return passes[unpack(words, params, row)];
          });
          break;
        }
        case Codec::kPlain: {
          const int64_t base = static_cast<int64_t>(segment) * segmentRows;
          compact(rows, [&](int32_t row) {
            const auto value = stringAt(column, base + row);
            return filter->testBytes(value.data(), value.size());
          });
          break;
        }
      }
      if (rows.empty()) {
        break;
      }
    }
    if (rows.empty()) {
      continue;
    }

    if (remainingFilterExprSet_) {
      std::vector<VectorPtr> children;
      for (size_t i = 0; i < spec_.filterColumnNames.size(); ++i) {
        children.push_back(decode(
            spec_.table->column(spec_.filterColumnNames[i]),
            spec_.filterType->childAt(i),
            segment,
            rows));
      }
      auto input = std::make_shared<RowVector>(
          pool_,
          spec_.filterType,
          nullptr,
          static_cast<vector_size_t>(rows.size()),
          std::move(children));
      SelectivityVector allRows(rows.size());
      VectorPtr result;
      expressionEvaluator_->evaluate(
          remainingFilterExprSet_.get(), allRows, *input, result);
      const auto numPassed =
          exec::processFilterResults(result, allRows, filterEvalCtx_, pool_);
      if (numPassed == 0) {
        rows.clear();
        continue;
      }
      if (numPassed < static_cast<vector_size_t>(rows.size())) {
        const auto* selected =
            filterEvalCtx_.selectedIndices->as<vector_size_t>();
        for (vector_size_t i = 0; i < numPassed; ++i) {
          rows[i] = rows[selected[i]];
        }
        rows.resize(numPassed);
      }
    }

    rowsPassed_ += rows.size();
    std::vector<VectorPtr> children;
    for (size_t i = 0; i < spec_.outputColumnNames.size(); ++i) {
      children.push_back(decode(
          spec_.table->column(spec_.outputColumnNames[i]),
          outputType_->childAt(i),
          segment,
          rows));
    }
    completedRows_ += rows.size();
    return std::make_shared<RowVector>(
        pool_,
        outputType_,
        nullptr,
        static_cast<vector_size_t>(rows.size()),
        std::move(children));
  }
  return nullptr;
}

std::optional<RowVectorPtr> EncodedCpuDataSource::next(
    uint64_t size,
    ContinueFuture& /*future*/) {
  if (split_ == nullptr) {
    return nullptr;
  }
  auto batch = nextBatch(size);
  if (batch == nullptr) {
    split_.reset();
  }
  return batch;
}

std::unordered_map<std::string, RuntimeMetric>
EncodedCpuDataSource::getRuntimeStats() {
  return {
      {"encodedSegmentsScanned",
       RuntimeMetric(static_cast<int64_t>(segmentsScanned_))},
      {"encodedSegmentsPruned",
       RuntimeMetric(static_cast<int64_t>(segmentsPruned_))},
      {"encodedRowsDecodedForFilter",
       RuntimeMetric(static_cast<int64_t>(rowsDecodedForFilter_))},
      {"encodedRowsPassed", RuntimeMetric(static_cast<int64_t>(rowsPassed_))},
  };
}

} // namespace facebook::velox::cudf_velox::connector::encoded
