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
#include "velox/experimental/supervelox/store/ColumnStore.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include <folly/json.h>

#include "velox/experimental/supervelox/common/Cuda.h"

namespace facebook::velox::supervelox {

size_t dtypeSize(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return 1;
    case DType::kInt16:
      return 2;
    case DType::kInt32:
      return 4;
    case DType::kInt64:
      return 8;
  }
  return 0;
}

int32_t Column::code(const std::string& value) const {
  for (size_t i = 0; i < dictionary.size(); ++i) {
    if (dictionary[i] == value) {
      return static_cast<int32_t>(i);
    }
  }
  return -1;
}

int64_t Column::value(int64_t row) const {
  switch (dtype) {
    case DType::kInt8:
      return static_cast<const int8_t*>(data)[row];
    case DType::kInt16:
      return static_cast<const int16_t*>(data)[row];
    case DType::kInt32:
      return static_cast<const int32_t*>(data)[row];
    case DType::kInt64:
      return static_cast<const int64_t*>(data)[row];
  }
  return 0;
}

std::string Column::string(int64_t row) const {
  if (kind == Kind::kDictionary) {
    return dictionary.at(value(row));
  }
  const char* chars = static_cast<const char*>(data);
  return std::string(chars + offsets[row], offsets[row + 1] - offsets[row]);
}

namespace {

DType parseDType(const std::string& name) {
  if (name == "int8") {
    return DType::kInt8;
  }
  if (name == "int16") {
    return DType::kInt16;
  }
  if (name == "int32") {
    return DType::kInt32;
  }
  if (name == "int64") {
    return DType::kInt64;
  }
  throw std::runtime_error("unknown dtype " + name);
}

Kind parseKind(const std::string& name) {
  if (name == "int") {
    return Kind::kInt;
  }
  if (name == "decimal") {
    return Kind::kDecimal;
  }
  if (name == "date") {
    return Kind::kDate;
  }
  if (name == "dictionary") {
    return Kind::kDictionary;
  }
  if (name == "string") {
    return Kind::kString;
  }
  throw std::runtime_error("unknown kind " + name);
}

size_t fileSize(const std::string& path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    throw std::runtime_error("cannot stat " + path);
  }
  return info.st_size;
}

void preadAll(const std::string& path, size_t offset, size_t bytes, void* out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("cannot open " + path);
  }
  size_t done = 0;
  while (done < bytes) {
    const ssize_t n = pread(fd, static_cast<char*>(out) + done, bytes - done, offset + done);
    if (n <= 0) {
      close(fd);
      throw std::runtime_error("short read of " + path);
    }
    done += n;
  }
  close(fd);
}

/// Reads 'path' into pinned memory using several threads for large files.
void* readPinned(const std::string& path, size_t bytes) {
  void* buffer = nullptr;
  SV_CUDA(cudaHostAlloc(&buffer, std::max<size_t>(bytes, 1), cudaHostAllocMapped | cudaHostAllocPortable));
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("cannot open " + path);
  }
  constexpr size_t kChunk = 256 << 20;
  const size_t chunks = (bytes + kChunk - 1) / kChunk;
  const size_t threads = std::min<size_t>(chunks, 8);
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  auto work = [&]() {
    for (size_t chunk = next++; chunk < chunks; chunk = next++) {
      size_t offset = chunk * kChunk;
      size_t end = std::min(bytes, offset + kChunk);
      while (offset < end) {
        ssize_t n = pread(fd, static_cast<char*>(buffer) + offset, end - offset, offset);
        if (n <= 0) {
          failed = true;
          return;
        }
        offset += n;
      }
    }
  };
  std::vector<std::thread> workers;
  for (size_t i = 1; i < threads; ++i) {
    workers.emplace_back(work);
  }
  work();
  for (auto& worker : workers) {
    worker.join();
  }
  close(fd);
  if (failed) {
    throw std::runtime_error("short read of " + path);
  }
  return buffer;
}

} // namespace

ColumnStore::ColumnStore(std::string directory) : directory_(std::move(directory)) {}

ColumnStore::~ColumnStore() {
  for (void* allocation : allocations_) {
    cudaFreeHost(allocation);
  }
}

ColumnStore::Table& ColumnStore::table(const std::string& name) {
  auto it = tables_.find(name);
  if (it != tables_.end()) {
    return it->second;
  }
  std::ifstream in(directory_ + "/" + name + "/meta.json");
  if (!in) {
    throw std::runtime_error("no table " + name + " in " + directory_);
  }
  std::stringstream text;
  text << in.rdbuf();
  auto meta = folly::parseJson(text.str());
  Table& table = tables_[name];
  for (auto& [key, value] : meta["columns"].items()) {
    auto column = std::make_unique<Column>();
    column->table = name;
    column->name = key.asString();
    column->kind = parseKind(value["kind"].asString());
    column->rows = value["rows"].asInt();
    if (column->kind != Kind::kString) {
      column->dtype = parseDType(value["dtype"].asString());
    }
    if (auto* scale = value.get_ptr("scale"); scale && scale->isInt()) {
      column->scale = scale->asInt();
    }
    if (auto* min = value.get_ptr("min"); min && min->isInt()) {
      column->min = min->asInt();
      column->max = value["max"].asInt();
    }
    if (auto* dictionary = value.get_ptr("dictionary")) {
      for (auto& entry : *dictionary) {
        column->dictionary.push_back(entry.asString());
      }
      column->min = 0;
      column->max = static_cast<int64_t>(column->dictionary.size()) - 1;
    }
    table.columns[column->name] = std::move(column);
  }
  return table;
}

Column& ColumnStore::describe(const std::string& tableName, const std::string& name) {
  auto& columns = table(tableName).columns;
  auto it = columns.find(name);
  if (it == columns.end()) {
    throw std::runtime_error("no column " + tableName + "." + name);
  }
  return *it->second;
}

void ColumnStore::load(Column& column) {
  const std::string base = directory_ + "/" + column.table + "/" + column.name;
  void* offsets = nullptr;
  size_t offsetBytes = 0;
  if (column.kind == Kind::kString) {
    offsetBytes = fileSize(base + ".offsets");
    offsets = readPinned(base + ".offsets", offsetBytes);
    column.bytes = fileSize(base + ".chars");
    column.data = readPinned(base + ".chars", column.bytes);
  } else {
    column.bytes = fileSize(base + ".bin");
    if (column.bytes != column.rows * dtypeSize(column.dtype)) {
      throw std::runtime_error("size mismatch for " + base);
    }
    column.data = readPinned(base + ".bin", column.bytes);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  allocations_.push_back(column.data);
  pinnedBytes_ += column.bytes + offsetBytes;
  if (offsets) {
    allocations_.push_back(offsets);
    column.offsets = static_cast<int64_t*>(offsets);
  }
}

const Column& ColumnStore::metadata(const std::string& tableName, const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  return describe(tableName, name);
}

const Column& ColumnStore::column(const std::string& tableName, const std::string& name) {
  Column* column;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    column = &describe(tableName, name);
  }
  if (column->data == nullptr) {
    load(*column);
  }
  return *column;
}

void ColumnStore::preload(const std::vector<std::string>& names, int32_t threads) {
  std::vector<Column*> pending;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& name : names) {
      auto dot = name.find('.');
      Column& column = describe(name.substr(0, dot), name.substr(dot + 1));
      if (column.data == nullptr) {
        pending.push_back(&column);
      }
    }
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> workers;
  for (int32_t i = 0; i < threads; ++i) {
    workers.emplace_back([&]() {
      for (size_t index = next++; index < pending.size(); index = next++) {
        load(*pending[index]);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

std::vector<std::string> ColumnStore::columns(const std::vector<std::string>& tables) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  for (const auto& name : tables) {
    for (const auto& [column, unused] : table(name).columns) {
      names.push_back(name + "." + column);
    }
  }
  return names;
}

void ColumnStore::read(const std::string& tableName, const std::string& name, int64_t first, int64_t count, void* out) {
  const Column& column = metadata(tableName, name);
  const std::string base = directory_ + "/" + tableName + "/" + name;
  if (column.kind == Kind::kString) {
    const size_t bytes = (count + 1) * sizeof(int64_t);
    if (column.offsets != nullptr) {
      std::memcpy(out, column.offsets + first, bytes);
    } else {
      preadAll(base + ".offsets", first * sizeof(int64_t), bytes, out);
    }
    return;
  }
  const size_t width = dtypeSize(column.dtype);
  if (column.data != nullptr) {
    std::memcpy(out, static_cast<const char*>(column.data) + first * width, count * width);
  } else {
    preadAll(base + ".bin", first * width, count * width, out);
  }
}

void ColumnStore::readChars(
    const std::string& tableName,
    const std::string& name,
    int64_t begin,
    int64_t end,
    void* out) {
  const Column& column = metadata(tableName, name);
  if (column.data != nullptr) {
    std::memcpy(out, static_cast<const char*>(column.data) + begin, end - begin);
  } else {
    preadAll(directory_ + "/" + tableName + "/" + name + ".chars", begin, end - begin, out);
  }
}

int64_t ColumnStore::value(const std::string& tableName, const std::string& name, int64_t row) {
  const Column& column = metadata(tableName, name);
  if (column.data != nullptr) {
    return column.value(row);
  }
  int64_t raw = 0;
  read(tableName, name, row, 1, &raw);
  switch (column.dtype) {
    case DType::kInt8:
      return static_cast<int8_t>(raw);
    case DType::kInt16:
      return static_cast<int16_t>(raw);
    case DType::kInt32:
      return static_cast<int32_t>(raw);
    case DType::kInt64:
      return raw;
  }
  return 0;
}

std::string ColumnStore::string(const std::string& tableName, const std::string& name, int64_t row) {
  const Column& column = metadata(tableName, name);
  if (column.kind == Kind::kDictionary) {
    return column.dictionary.at(value(tableName, name, row));
  }
  if (column.data != nullptr) {
    return column.string(row);
  }
  int64_t offsets[2];
  read(tableName, name, row, 1, offsets);
  std::string out(offsets[1] - offsets[0], '\0');
  readChars(tableName, name, offsets[0], offsets[1], out.data());
  return out;
}

int64_t ColumnStore::rows(const std::string& tableName) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& columns = table(tableName).columns;
  return columns.begin()->second->rows;
}

} // namespace facebook::velox::supervelox
