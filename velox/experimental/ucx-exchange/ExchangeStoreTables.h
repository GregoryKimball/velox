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
#pragma once

#include "velox/vector/TypeAliases.h"

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::core {
class PlanNode;
class QueryCtx;
} // namespace facebook::velox::core

namespace facebook::velox::ucx_exchange {

/// Keeps Presto's temporary tables, which hold materialized exchanges, in
/// ExchangeStore::instance() instead of files, with a device tier of
/// 'deviceBytes' and a host tier of 'hostBytes' (0 for no limit). Each file of
/// a temporary table becomes a small placeholder file, so that the coordinator
/// lists and commits the table as usual, and the data of the file is kept in
/// the store under storeKey() of its path.
void registerExchangeStoreTables(uint64_t deviceBytes, uint64_t hostBytes);

/// Returns temporary tables to files.
void unregisterExchangeStoreTables();

/// Returns the table and file under which ExchangeStore keeps the data of the
/// temporary table file at 'path': the name of its directory, and the bucket
/// number of a bucket file (the name of any other file). Both are unique per
/// query and are the same in the writer's path and in a split's path, which
/// may differ in scheme and, for a split made before the file was written, in
/// the rest of the file name.
std::pair<std::string, std::string> storeKey(std::string_view path);

/// Returns the store key of the temporary table whose writer, in 'root',
/// receives the rows of the UCX exchange 'exchangeId' unchanged, when the
/// exchange store holds temporary tables and the query reads them by bucket
/// (Hive session property stream_temporary_tables or
/// list_temporary_tables_by_bucket, so that a bucket needs no file): the
/// writer writes every column it reads, in order, through Projects
/// that keep each column in its position and local exchanges. The exchange's
/// sources then keep each bucket's payload in the store as it arrives
/// (UcxExchangeQueue::setStoreTable()), and the writer gets no rows.
std::optional<std::string> receivingTableInStore(
    const core::PlanNode& root,
    std::string_view exchangeId,
    const core::QueryCtx& queryCtx);

/// Adds a payload that a source of the exchange receivingTableInStore() names
/// received with a Hive bucket tag (untagPayloadBucket()) to the bucket's file
/// of the store table 'table', as it arrived: with the exchange codec, without
/// the store's dictionary and difference encodings. 'metadata' is without the
/// tag. The table is complete once the receiving task 'taskId' ends
/// (ExchangeStore::endWrites()).
void addReceivedBucketPayload(
    std::string_view table,
    int32_t bucket,
    std::string_view taskId,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<rmm::device_buffer> data,
    vector_size_t numRows,
    cuda::stream_ref stream);

/// Keys of a temporary table split's custom split info. A coordinator that
/// reads a bucketed table as fewer buckets than it has, each the union of the
/// table buckets with the same remainder (Hive's read bucket count), sets both
/// on the splits of a table that is read while it is written. The split of a
/// table bucket then reads its whole read bucket: the first split of a read
/// bucket streams every table bucket of it, and the others find theirs read.
inline constexpr std::string_view kReadBucketCountSplitInfo{
    "read_bucket_count"};
inline constexpr std::string_view kTableBucketCountSplitInfo{
    "table_bucket_count"};

} // namespace facebook::velox::ucx_exchange
