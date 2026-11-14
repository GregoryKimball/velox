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

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

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

} // namespace facebook::velox::ucx_exchange
