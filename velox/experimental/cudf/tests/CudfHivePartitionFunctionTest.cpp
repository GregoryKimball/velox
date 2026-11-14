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
#include "velox/experimental/cudf/exec/CudfHivePartitionFunction.h"

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <fmt/format.h>

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

#include <limits>
#include <numeric>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

class CudfHivePartitionFunctionTest : public testing::Test,
                                      public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  static rmm::device_async_resource_ref memoryResource() {
    return cudf::get_current_device_resource_ref();
  }

  // Partitions 'input' on 'keyChannels' with the CPU HivePartitionFunction and
  // with CudfHivePartitionFunction and expects the same partition per row.
  void expectSamePartitions(
      const RowVectorPtr& input,
      int numBuckets,
      const std::vector<int>& bucketToPartition,
      const std::vector<column_index_t>& keyChannels) {
    connector::hive::HivePartitionFunction cpuFunction(
        numBuckets, bucketToPartition, keyChannels);
    std::vector<uint32_t> expected;
    if (auto single = cpuFunction.partition(*input, expected)) {
      expected.assign(input->size(), *single);
    }

    const CudfHivePartitionFunction gpuFunction(
        numBuckets,
        bucketToPartition,
        std::vector<cudf::size_type>(keyChannels.begin(), keyChannels.end()));
    auto table = with_arrow::toCudfTable(
        input, pool(), stream_, memoryResource());
    auto partitions =
        gpuFunction.partitions(table->view(), stream_, memoryResource());
    ASSERT_EQ(partitions->size(), input->size());
    ASSERT_EQ(partitions->null_count(), 0);
    std::vector<int32_t> actual(input->size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        actual.data(),
        partitions->view().data<int32_t>(),
        actual.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream_.value()));
    stream_.synchronize();
    EXPECT_EQ(std::vector<uint32_t>(actual.begin(), actual.end()), expected);
  }

  // A map of 'numBuckets' buckets onto 'numPartitions' partitions that is not
  // the identity or a plain modulo.
  static std::vector<int> scatteredBuckets(int numBuckets, int numPartitions) {
    std::vector<int> bucketToPartition(numBuckets);
    for (int bucket = 0; bucket < numBuckets; ++bucket) {
      bucketToPartition[bucket] = (bucket * 7 + bucket / 3) % numPartitions;
    }
    return bucketToPartition;
  }

  VectorFuzzer makeFuzzer() {
    VectorFuzzer::Options options;
    options.vectorSize = 10'000;
    options.nullRatio = 0.1;
    options.stringLength = 40;
    options.stringVariableLength = true;
    // Non-ASCII characters exercise bytes that hash as negative values.
    options.charEncodings = {
        fuzzer::ASCII,
        fuzzer::UNICODE_CASE_SENSITIVE,
        fuzzer::EXTENDED_UNICODE,
    };
    return VectorFuzzer(options, pool());
  }

  rmm::cuda_stream stream_;
};

TEST_F(CudfHivePartitionFunctionTest, matchesCpuForEveryKeyType) {
  auto fuzzer = makeFuzzer();
  const std::vector<TypePtr> types{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      DATE(),
      DECIMAL(12, 2),
      VARCHAR(),
  };
  for (const auto& type : types) {
    SCOPED_TRACE(type->toString());
    const auto input = makeRowVector({fuzzer.fuzzFlat(type)});
    expectSamePartitions(input, 256, {}, {0});
    expectSamePartitions(input, 256, scatteredBuckets(256, 5), {0});
    // With a power of two buckets, the bucket depends only on the hash's low
    // bits, which hide differences in the high bits.
    expectSamePartitions(input, 13, {}, {0});
    expectSamePartitions(input, 1, {}, {0});
  }
}

TEST_F(CudfHivePartitionFunctionTest, matchesCpuForSeveralKeys) {
  auto fuzzer = makeFuzzer();
  const auto input = makeRowVector({
      fuzzer.fuzzFlat(BIGINT()),
      fuzzer.fuzzFlat(VARCHAR()),
      fuzzer.fuzzFlat(INTEGER()),
      fuzzer.fuzzFlat(DATE()),
  });
  expectSamePartitions(input, 7, scatteredBuckets(7, 3), {0, 1, 2, 3});
  expectSamePartitions(input, 256, {}, {3, 1});
}

TEST_F(CudfHivePartitionFunctionTest, matchesCpuForExtremeKeys) {
  const auto input = makeRowVector({
      makeFlatVector<int64_t>(
          {std::numeric_limits<int64_t>::min(),
           -1,
           0,
           1,
           std::numeric_limits<int64_t>::max(),
           6'000'000'000}),
      makeFlatVector<std::string>(
          {"", "\xff", "\x80\x7f", "abc", std::string(1'000, '\xc3'), "z"}),
  });
  expectSamePartitions(input, 256, {}, {0});
  expectSamePartitions(input, 256, {}, {1});
  expectSamePartitions(input, 3, {}, {1, 0});
}

TEST_F(CudfHivePartitionFunctionTest, partitionGroupsRowsByPartition) {
  auto fuzzer = makeFuzzer();
  const auto keys = fuzzer.fuzzFlat(BIGINT());
  const auto input = makeRowVector(
      {keys,
       makeFlatVector<int32_t>(keys->size(), [](auto row) { return row; })});
  auto table =
      with_arrow::toCudfTable(input, pool(), stream_, memoryResource());
  constexpr int kNumBuckets = 16;
  // cuDF partitions by a mask for a power of two partitions, by a remainder
  // otherwise.
  for (const int numPartitions : {4, 3}) {
    SCOPED_TRACE(numPartitions);
    const auto bucketToPartition = scatteredBuckets(kNumBuckets, numPartitions);
    connector::hive::HivePartitionFunction cpuFunction(
        kNumBuckets, bucketToPartition, {0});
    std::vector<uint32_t> expected;
    cpuFunction.partition(*input, expected);
    const CudfHivePartitionFunction gpuFunction(
        kNumBuckets, bucketToPartition, {0});
    auto [grouped, offsets] = gpuFunction.partition(
        table->view(), numPartitions, stream_, memoryResource());
    ASSERT_EQ(offsets.size(), numPartitions + 1);
    ASSERT_EQ(offsets.front(), 0);
    ASSERT_EQ(offsets.back(), input->size());
    ASSERT_EQ(grouped->num_columns(), 2);
    std::vector<int32_t> rows(input->size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        rows.data(),
        grouped->view().column(1).data<int32_t>(),
        rows.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream_.value()));
    stream_.synchronize();
    for (int partition = 0; partition < numPartitions; ++partition) {
      for (auto i = offsets[partition]; i < offsets[partition + 1]; ++i) {
        ASSERT_EQ(expected[rows[i]], partition) << "grouped row " << i;
      }
    }
  }
}

TEST_F(CudfHivePartitionFunctionTest, fromSpecMatchesTheFunctionTheSpecCreates) {
  auto fuzzer = makeFuzzer();
  // The operator's output drops a column and reorders the others: the keys,
  // channels of the input, are located in the output by name.
  const auto inputType =
      ROW({"a", "dropped", "b"}, {BIGINT(), DOUBLE(), VARCHAR()});
  const auto outputType = ROW({"b", "a"}, {VARCHAR(), BIGINT()});
  const auto input = makeRowVector(
      {"a", "dropped", "b"},
      {fuzzer.fuzzFlat(BIGINT()),
       fuzzer.fuzzFlat(DOUBLE()),
       fuzzer.fuzzFlat(VARCHAR())});
  const auto output =
      makeRowVector({"b", "a"}, {input->childAt(2), input->childAt(0)});
  auto table =
      with_arrow::toCudfTable(output, pool(), stream_, memoryResource());
  const std::vector<column_index_t> keys{2, 0};
  constexpr int kNumBuckets = 16;
  constexpr int kNumPartitions = 3;
  const connector::hive::HivePartitionFunctionSpec withoutMap(
      kNumBuckets, keys, {});
  const connector::hive::HivePartitionFunctionSpec withMap(
      kNumBuckets, scatteredBuckets(kNumBuckets, kNumPartitions), keys, {});
  for (const auto* spec : {&withoutMap, &withMap}) {
    for (const bool localExchange : {false, true}) {
      SCOPED_TRACE(
          fmt::format("{}, local {}", spec->toString(), localExchange));
      std::vector<uint32_t> expected;
      spec->create(kNumPartitions, localExchange)->partition(*input, expected);
      const auto gpuFunction = CudfHivePartitionFunction::fromSpec(
          *spec, kNumPartitions, localExchange, inputType, outputType);
      auto partitions =
          gpuFunction.partitions(table->view(), stream_, memoryResource());
      std::vector<int32_t> actual(input->size());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          actual.data(),
          partitions->view().data<int32_t>(),
          actual.size() * sizeof(int32_t),
          cudaMemcpyDeviceToHost,
          stream_.value()));
      stream_.synchronize();
      EXPECT_EQ(std::vector<uint32_t>(actual.begin(), actual.end()), expected);
    }
  }
  EXPECT_TRUE(CudfHivePartitionFunction::supports(withMap, inputType));
  EXPECT_FALSE(CudfHivePartitionFunction::supports(
      connector::hive::HivePartitionFunctionSpec(kNumBuckets, {1}, {}),
      ROW({"a", "b"}, {BIGINT(), TIMESTAMP()})));
}

TEST_F(CudfHivePartitionFunctionTest, rejectsUnsupportedKeyType) {
  EXPECT_FALSE(
      CudfHivePartitionFunction::supports(
          cudf::data_type{cudf::type_id::DECIMAL128, -2}));
  const auto input = makeRowVector({makeFlatVector<int128_t>(
      {1, 2, 3}, DECIMAL(38, 2))});
  auto table =
      with_arrow::toCudfTable(input, pool(), stream_, memoryResource());
  const CudfHivePartitionFunction function(4, {}, {0});
  EXPECT_THROW(
      function.partitions(table->view(), stream_, memoryResource()),
      cudf::logic_error);
}

} // namespace
} // namespace facebook::velox::cudf_velox
