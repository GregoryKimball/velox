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

Result Engine::run(const QuerySpec& spec) {
  const auto start = std::chrono::steady_clock::now();
  cache_.beginQuery();
  cache_.resetStats();
  Result result = [&] {
    QueryContext ctx(store_, cache_, stream_);
    return spec.run(ctx);
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

FusedQueryNode::FusedQueryNode(const core::PlanNodeId& id, int32_t query)
    : PlanNode(id), spec_(findQuery(query)), outputType_([&] {
        VELOX_CHECK_NOT_NULL(spec_, "no fused query {}", query);
        return resultRowType(spec_->schema);
      }()) {}

void FusedQueryNode::addDetails(std::stringstream& stream) const {
  stream << "q" << spec_->number;
}

FusedQueryOperator::FusedQueryOperator(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const FusedQueryNode>& node)
    : SourceOperator(driverCtx, node->outputType(), operatorId, node->id(), "FusedQuery"),
      spec_(node->spec()) {}

RowVectorPtr FusedQueryOperator::getOutput() {
  if (finished_) {
    return nullptr;
  }
  finished_ = true;
  Result result = Engine::instance().run(spec_);
  std::vector<VectorPtr> children;
  for (size_t i = 0; i < result.fields.size(); ++i) {
    children.push_back(makeColumn(result, i, pool()));
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
