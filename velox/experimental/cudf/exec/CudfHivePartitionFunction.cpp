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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/connectors/hive/HiveConnector.h"

#include <algorithm>
#include <random>

namespace facebook::velox::cudf_velox {

namespace {

// The key channels of 'spec' in the rows it partitions.
std::vector<column_index_t> keyChannels(
    const velox::connector::hive::HivePartitionFunctionSpec& spec) {
  return ISerializable::deserialize<std::vector<column_index_t>>(
      spec.serialize()["keys"]);
}

} // namespace

bool CudfHivePartitionFunction::supports(
    const velox::connector::hive::HivePartitionFunctionSpec& spec,
    const RowTypePtr& inputType) {
  if (!spec.serialize()["constants"].empty()) {
    return false;
  }
  const auto channels = keyChannels(spec);
  return std::all_of(channels.begin(), channels.end(), [&](auto channel) {
    return supports(veloxToCudfDataType(inputType->childAt(channel)));
  });
}

CudfHivePartitionFunction CudfHivePartitionFunction::fromSpec(
    const velox::connector::hive::HivePartitionFunctionSpec& spec,
    int numPartitions,
    bool localExchange,
    const RowTypePtr& inputType,
    const RowTypePtr& outputType) {
  VELOX_CHECK(
      supports(spec, inputType),
      "The GPU cannot compute the Hive partitions: {}",
      spec.toString());
  const auto serialized = spec.serialize();
  const auto numBuckets = static_cast<int>(serialized["numBuckets"].asInt());
  auto bucketToPartition = ISerializable::deserialize<std::vector<int>>(
      serialized["bucketToPartition"]);
  if (bucketToPartition.empty()) {
    // The map HivePartitionFunctionSpec::create() makes.
    for (int bucket = 0; bucket < numBuckets; ++bucket) {
      bucketToPartition.push_back(bucket % numPartitions);
    }
    if (localExchange) {
      std::shuffle(
          bucketToPartition.begin(), bucketToPartition.end(), std::mt19937{0});
    }
  }
  std::vector<cudf::size_type> keyColumns;
  for (const auto channel : keyChannels(spec)) {
    keyColumns.push_back(
        static_cast<cudf::size_type>(
            outputType->getChildIdx(inputType->nameOf(channel))));
  }
  return CudfHivePartitionFunction(
      numBuckets, std::move(bucketToPartition), std::move(keyColumns));
}

} // namespace facebook::velox::cudf_velox
