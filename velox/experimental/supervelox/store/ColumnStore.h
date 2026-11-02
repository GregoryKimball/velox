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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace facebook::velox::supervelox {

enum class Kind { kInt, kDecimal, kDate, kDictionary, kString };

enum class DType { kInt8, kInt16, kInt32, kInt64 };

size_t dtypeSize(DType dtype);

/// One column of a table converted by prep.py, held in pinned host memory that
/// kernels can also read directly.
struct Column {
  std::string table;
  std::string name;
  Kind kind;
  DType dtype{DType::kInt64};
  int32_t scale{0};
  int64_t min{0};
  int64_t max{0};
  int64_t rows{0};
  std::vector<std::string> dictionary;

  /// Values, or the characters of a string column.
  void* data{nullptr};
  size_t bytes{0};
  /// rows + 1 offsets into 'data' for string columns.
  int64_t* offsets{nullptr};

  /// Returns the code of 'value' in the dictionary, or -1 if absent.
  int32_t code(const std::string& value) const;

  /// Value at 'row' of a fixed-width column, as stored (unscaled, codes).
  int64_t value(int64_t row) const;

  /// Value at 'row' of a string column, or the decoded dictionary value.
  std::string string(int64_t row) const;
};

/// The tables of one scale factor, loaded lazily into pinned host memory.
class ColumnStore {
 public:
  explicit ColumnStore(std::string directory);
  ~ColumnStore();

  const Column& column(const std::string& table, const std::string& name);

  /// The column's description without its data, which stays unloaded.
  const Column& metadata(const std::string& table, const std::string& name);

  /// Loads the named columns in parallel. Each entry is "table.column".
  void preload(const std::vector<std::string>& columns, int32_t threads = 32);

  int64_t rows(const std::string& table);

  /// Copies rows [first, first + count) of a fixed-width column as stored, or
  /// for a string column its offsets [first, first + count], to host 'out'.
  /// Reads the column's file unless the column is loaded.
  void read(const std::string& table, const std::string& name, int64_t first, int64_t count, void* out);

  /// Copies characters [begin, end) of a string column to host 'out'.
  void readChars(const std::string& table, const std::string& name, int64_t begin, int64_t end, void* out);

  /// The value at 'row' of a fixed-width column as stored (unscaled, codes),
  /// read from the file unless the column is loaded.
  int64_t value(const std::string& table, const std::string& name, int64_t row);

  /// The value at 'row' of a string column, or the decoded dictionary value.
  std::string string(const std::string& table, const std::string& name, int64_t row);

  /// "table.column" for every column of the named tables.
  std::vector<std::string> columns(const std::vector<std::string>& tables);

  const std::string& directory() const {
    return directory_;
  }

  size_t pinnedBytes() const {
    return pinnedBytes_;
  }

 private:
  struct Table {
    std::map<std::string, std::unique_ptr<Column>> columns;
  };

  Table& table(const std::string& name);
  Column& describe(const std::string& table, const std::string& name);
  void load(Column& column);

  const std::string directory_;
  std::mutex mutex_;
  std::map<std::string, Table> tables_;
  std::vector<void*> allocations_;
  size_t pinnedBytes_{0};
};

} // namespace facebook::velox::supervelox
