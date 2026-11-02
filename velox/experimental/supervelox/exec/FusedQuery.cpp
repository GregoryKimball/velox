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
#include "velox/experimental/supervelox/exec/FusedQuery.h"

#include <chrono>

#include "velox/experimental/supervelox/common/Cuda.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::supervelox {
namespace {

Engine* gEngine = nullptr;

TypePtr fieldType(const ResultField& field) {
  switch (field.type) {
    case ResultType::kBigint:
      return BIGINT();
    case ResultType::kInteger:
      return INTEGER();
    case ResultType::kDouble:
      return DOUBLE();
    case ResultType::kVarchar:
      return VARCHAR();
    case ResultType::kDate:
      return DATE();
    case ResultType::kDecimal:
      return DECIMAL(field.precision, field.scale);
  }
  VELOX_UNREACHABLE();
}

template <typename T, typename Source>
VectorPtr makeFlat(const TypePtr& type, const std::vector<Source>& values, memory::MemoryPool* pool) {
  auto vector = BaseVector::create<FlatVector<T>>(type, values.size(), pool);
  for (size_t i = 0; i < values.size(); ++i) {
    vector->set(i, static_cast<T>(values[i]));
  }
  return vector;
}

__int128 pow10(int32_t exponent) {
  __int128 value = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    value *= 10;
  }
  return value;
}

/// Rescales an unscaled decimal, rounding half away from zero as Presto does.
__int128 rescale(__int128 value, int32_t from, int32_t to) {
  if (to >= from) {
    return value * pow10(to - from);
  }
  const __int128 divisor = pow10(from - to);
  const __int128 half = divisor / 2;
  return value >= 0 ? (value + half) / divisor : -((-value + half) / divisor);
}

/// The value of a numeric result field as an unscaled decimal at `scale`.
__int128 asDecimal(const Result& result, size_t index, size_t row, int32_t scale) {
  const ResultField& field = result.fields[index];
  switch (field.type) {
    case ResultType::kDecimal:
      return rescale(result.decimals[index][row], field.scale, scale);
    case ResultType::kDouble:
      return static_cast<__int128>(std::llround(result.doubles[index][row] * static_cast<double>(pow10(scale))));
    case ResultType::kBigint:
    case ResultType::kInteger:
    case ResultType::kDate:
      return static_cast<__int128>(result.ints[index][row]) * pow10(scale);
    case ResultType::kVarchar:
      break;
  }
  VELOX_FAIL("result field {} is not numeric", field.name);
}

double asDouble(const Result& result, size_t index, size_t row) {
  const ResultField& field = result.fields[index];
  switch (field.type) {
    case ResultType::kDecimal:
      return static_cast<double>(result.decimals[index][row]) / static_cast<double>(pow10(field.scale));
    case ResultType::kDouble:
      return result.doubles[index][row];
    case ResultType::kBigint:
    case ResultType::kInteger:
    case ResultType::kDate:
      return static_cast<double>(result.ints[index][row]);
    case ResultType::kVarchar:
      break;
  }
  VELOX_FAIL("result field {} is not numeric", field.name);
}

/// A result column as a vector of `type`, which is either the field's own
/// type or that of the plan subtree the fused query replaced.
VectorPtr convertColumn(const Result& result, size_t index, const TypePtr& type, memory::MemoryPool* pool) {
  const size_t rows = result.rows();
  auto build = [&]<typename T>(auto value) -> VectorPtr {
    auto vector = BaseVector::create<FlatVector<T>>(type, rows, pool);
    for (size_t row = 0; row < rows; ++row) {
      vector->set(row, value(row));
    }
    return vector;
  };
  switch (type->kind()) {
    case TypeKind::BIGINT:
      if (type->isShortDecimal()) {
        const auto scale = getDecimalPrecisionScale(*type).second;
        return build.template operator()<int64_t>(
            [&](size_t row) { return static_cast<int64_t>(asDecimal(result, index, row, scale)); });
      }
      return build.template operator()<int64_t>(
          [&](size_t row) { return static_cast<int64_t>(asDecimal(result, index, row, 0)); });
    case TypeKind::HUGEINT: {
      const auto scale = getDecimalPrecisionScale(*type).second;
      return build.template operator()<int128_t>([&](size_t row) { return asDecimal(result, index, row, scale); });
    }
    case TypeKind::INTEGER:
      return build.template operator()<int32_t>(
          [&](size_t row) { return static_cast<int32_t>(asDecimal(result, index, row, 0)); });
    case TypeKind::DOUBLE:
      return build.template operator()<double>([&](size_t row) { return asDouble(result, index, row); });
    default:
      VELOX_FAIL("cannot convert result field {} to {}", result.fields[index].name, type->toString());
  }
}

VectorPtr makeColumn(const Result& result, size_t index, memory::MemoryPool* pool) {
  const ResultField& field = result.fields[index];
  const TypePtr type = fieldType(field);
  switch (field.type) {
    case ResultType::kBigint:
      return makeFlat<int64_t>(type, result.ints[index], pool);
    case ResultType::kInteger:
    case ResultType::kDate:
      return makeFlat<int32_t>(type, result.ints[index], pool);
    case ResultType::kDouble:
      return makeFlat<double>(type, result.doubles[index], pool);
    case ResultType::kDecimal:
      if (type->isShortDecimal()) {
        return makeFlat<int64_t>(type, result.decimals[index], pool);
      }
      return makeFlat<int128_t>(type, result.decimals[index], pool);
    case ResultType::kVarchar: {
      if (const auto* dictionary = result.dictionaries[index]) {
        const auto& codes = result.codes[index];
        auto base = BaseVector::create<FlatVector<StringView>>(type, dictionary->size(), pool);
        for (size_t i = 0; i < dictionary->size(); ++i) {
          base->set(i, StringView((*dictionary)[i]));
        }
        auto indices = allocateIndices(codes.size(), pool);
        std::copy(codes.begin(), codes.end(), indices->asMutable<vector_size_t>());
        return BaseVector::wrapInDictionary(nullptr, std::move(indices), codes.size(), std::move(base));
      }
      const auto& strings = result.strings[index];
      auto vector = BaseVector::create<FlatVector<StringView>>(type, strings.size(), pool);
      for (size_t i = 0; i < strings.size(); ++i) {
        vector->set(i, StringView(strings[i]));
      }
      return vector;
    }
  }
  VELOX_UNREACHABLE();
}

} // namespace

Engine::Engine(const std::string& dataDirectory, size_t cacheBytes, size_t stageBytes)
    : store_(dataDirectory), cache_(cacheBytes, stageBytes) {
  SV_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  cudaMemPool_t pool;
  SV_CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t threshold = UINT64_MAX;
  SV_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  // Grow the scratch pool now so queries do not pay for mapping HBM pages.
  constexpr size_t kScratchBytes = 8ULL << 30;
  void* warm = nullptr;
  SV_CUDA(cudaMallocAsync(&warm, kScratchBytes, stream_));
  SV_CUDA(cudaFreeAsync(warm, stream_));
  SV_CUDA(cudaStreamSynchronize(stream_));
}

Engine::~Engine() {
  cache_.clear();
  cudaStreamDestroy(stream_);
}

void Engine::setInstance(Engine* engine) {
  gEngine = engine;
}

Engine& Engine::instance() {
  VELOX_CHECK_NOT_NULL(gEngine, "supervelox Engine is not set");
  return *gEngine;
}

Result Engine::run(const QuerySpec& spec, const QueryParams& params) {
  const auto start = std::chrono::steady_clock::now();
  cache_.beginQuery();
  cache_.resetStats();
  lastRun_.backend = backend(spec.number);
  QueryFunction function = implementation(spec, lastRun_.backend);
  if (function == nullptr) {
    lastRun_.backend = Backend::kHandwritten;
    function = spec.run;
  }
  Result result = [&] {
    QueryContext ctx(store_, cache_, stream_, params);
    return function(ctx);
  }();
  SV_CUDA(cudaStreamSynchronize(stream_));
  SV_CUDA(cudaGetLastError());
  lastRun_.gpuMs =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  lastRun_.cache = cache_.stats();
  return result;
}

RowTypePtr resultRowType(const std::vector<ResultField>& schema) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (const auto& field : schema) {
    names.push_back(field.name);
    types.push_back(fieldType(field));
  }
  return ROW(std::move(names), std::move(types));
}

FusedQueryNode::FusedQueryNode(
    const core::PlanNodeId& id,
    int32_t query,
    QueryParams params,
    RowTypePtr outputType)
    : PlanNode(id), spec_(findQuery(query)), params_(std::move(params)), outputType_([&] {
        VELOX_CHECK_NOT_NULL(spec_, "no fused query {}", query);
        if (!outputType) {
          return resultRowType(spec_->schema);
        }
        VELOX_CHECK_EQ(outputType->size(), spec_->schema.size(), "q{} output has the wrong width", query);
        return outputType;
      }()) {}

void FusedQueryNode::addDetails(std::stringstream& stream) const {
  stream << "q" << spec_->number;
  for (const auto& [name, value] : params_.values) {
    stream << " " << name << "=" << value;
  }
}

FusedQueryOperator::FusedQueryOperator(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const FusedQueryNode>& node)
    : SourceOperator(driverCtx, node->outputType(), operatorId, node->id(), "FusedQuery"),
      spec_(node->spec()),
      params_(node->params()) {}

RowVectorPtr FusedQueryOperator::getOutput() {
  if (finished_) {
    return nullptr;
  }
  finished_ = true;
  Result result = Engine::instance().run(spec_, params_);
  std::vector<VectorPtr> children;
  for (size_t i = 0; i < result.fields.size(); ++i) {
    const auto& type = outputType_->childAt(i);
    if (type->equivalent(*fieldType(result.fields[i]))) {
      children.push_back(makeColumn(result, i, pool()));
    } else {
      children.push_back(convertColumn(result, i, type, pool()));
    }
  }
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, result.rows(), std::move(children));
}

std::unique_ptr<exec::Operator> FusedQueryTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto fused = std::dynamic_pointer_cast<const FusedQueryNode>(node)) {
    return std::make_unique<FusedQueryOperator>(id, ctx, fused);
  }
  return nullptr;
}

std::optional<uint32_t> FusedQueryTranslator::maxDrivers(const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const FusedQueryNode>(node)) {
    return 1;
  }
  return std::nullopt;
}

void registerFusedQuery() {
  exec::Operator::registerOperator(std::make_unique<FusedQueryTranslator>());
}

} // namespace facebook::velox::supervelox
