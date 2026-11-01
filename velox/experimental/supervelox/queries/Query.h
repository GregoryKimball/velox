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

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "velox/experimental/supervelox/store/ColumnStore.h"
#include "velox/experimental/supervelox/store/DeviceCache.h"

// Shared by the CUDA query files and the Velox operator, so this header must
// not include Velox or folly.

namespace facebook::velox::supervelox {

/// Days since 1970-01-01 for a proleptic Gregorian date.
constexpr int32_t days(int32_t year, int32_t month, int32_t day) {
  year -= month <= 2;
  const int32_t era = (year >= 0 ? year : year - 399) / 400;
  const int32_t yoe = year - era * 400;
  const int32_t doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}

enum class ResultType { kBigint, kInteger, kDouble, kVarchar, kDate, kDecimal };

struct ResultField {
  std::string name;
  ResultType type;
  int32_t precision{0};
  int32_t scale{0};
};

/// Host-side output of a fused query: small, column-major, ready to become a
/// Velox RowVector.
struct Result {
  std::vector<ResultField> fields;
  /// Per field, one of these is filled according to the field type. Decimals
  /// are unscaled; dates are days since the epoch.
  std::vector<std::vector<int64_t>> ints;
  std::vector<std::vector<__int128>> decimals;
  std::vector<std::vector<double>> doubles;
  std::vector<std::vector<std::string>> strings;
  /// A varchar field may instead be codes into a dictionary that outlives the
  /// result; it then becomes a Velox dictionary vector.
  std::vector<std::vector<int32_t>> codes;
  std::vector<const std::vector<std::string>*> dictionaries;

  explicit Result(std::vector<ResultField> schema)
      : fields(std::move(schema)),
        ints(fields.size()),
        decimals(fields.size()),
        doubles(fields.size()),
        strings(fields.size()),
        codes(fields.size()),
        dictionaries(fields.size(), nullptr) {}

  size_t rows() const;
};

/// Literals taken from a matched plan, in the store's encoding: dates as days,
/// decimals unscaled at the column's scale, strings as dictionary codes. A
/// query uses the TPC-H validation value for any literal it is not given.
struct QueryParams {
  std::map<std::string, int64_t> values;

  int64_t get(const std::string& name, int64_t fallback) const {
    auto it = values.find(name);
    return it == values.end() ? fallback : it->second;
  }
};

/// Everything a fused query needs while it runs: device-readable columns from
/// the HBM cache, derived structures, and scratch memory freed at the end.
class QueryContext {
 public:
  QueryContext(ColumnStore& store, DeviceCache& cache, cudaStream_t stream, const QueryParams& params);
  ~QueryContext();

  int64_t param(const std::string& name, int64_t fallback) const {
    return params_.get(name, fallback);
  }

  const Column& meta(const std::string& table, const std::string& name) {
    return store_.column(table, name);
  }

  /// The description and dictionary of table.name, without loading its values.
  const Column& info(const std::string& table, const std::string& name) {
    return store_.metadata(table, name);
  }

  int64_t rows(const std::string& table) {
    return store_.rows(table);
  }

  /// Device-readable values of a fixed-width column as T, widening a
  /// narrower stored type through a cached derived copy.
  template <typename T>
  const T* col(const std::string& table, const std::string& name) {
    return static_cast<const T*>(column(table, name, sizeof(T)));
  }

  const int64_t* offsets(const std::string& table, const std::string& name);
  const char* chars(const std::string& table, const std::string& name);

  /// Device-readable rows [first, last) of a fixed-width column as T, at
  /// p + first through p + last - 1 of the returned p. In one batch the whole
  /// cached column; run batch by batch, only these rows, copied into memory
  /// that endBatch frees.
  template <typename T>
  const T* rows(const std::string& table, const std::string& name, int64_t first, int64_t last) {
    return static_cast<const T*>(range(table, name, first, last, sizeof(T)));
  }

  /// Rows [first, last) of a string column: offsets valid at [first, last]
  /// and characters at the absolute positions those offsets give.
  struct StringRows {
    const int64_t* offsets;
    const char* chars;
  };
  StringRows strings(const std::string& table, const std::string& name, int64_t first, int64_t last);

  /// Device memory that endBatch frees.
  void* batchBytes(size_t bytes);

  template <typename T>
  T* batchScratch(size_t count) {
    return static_cast<T*>(batchBytes(count * sizeof(T)));
  }

  /// Frees the memory of the batch that just finished.
  void endBatch();

  /// A value of table.name as stored, read from the store without loading
  /// the column.
  int64_t value(const std::string& table, const std::string& name, int64_t row) {
    return store_.value(table, name, row);
  }

  std::string string(const std::string& table, const std::string& name, int64_t row) {
    return store_.string(table, name, row);
  }

  /// Per-zone {min, max} of an integer column, kZoneRows rows per zone.
  const int2* zones(const std::string& table, const std::string& name);

  void* scratchBytes(size_t bytes);

  template <typename T>
  T* scratch(size_t count) {
    return static_cast<T*>(scratchBytes(count * sizeof(T)));
  }

  /// Zeroed scratch.
  template <typename T>
  T* zeroed(size_t count) {
    T* buffer = scratch<T>(count);
    cudaMemsetAsync(buffer, 0, count * sizeof(T), stream_);
    return buffer;
  }

  template <typename T>
  std::vector<T> download(const T* device, size_t count) {
    std::vector<T> host(count);
    if (count) {
      cudaMemcpyAsync(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost, stream_);
    }
    cudaStreamSynchronize(stream_);
    return host;
  }

  /// With SV_TRACE set, waits for the stream and prints the time since the
  /// previous mark.
  void mark(const char* phase);

  DeviceCache& cache() {
    return cache_;
  }

  ColumnStore& store() {
    return store_;
  }

  cudaStream_t stream() const {
    return stream_;
  }

  static constexpr int32_t kZoneRows = 8192;

 private:
  const void* column(const std::string& table, const std::string& name, size_t width);
  const void* range(const std::string& table, const std::string& name, int64_t first, int64_t last, size_t width);
  /// Copies 'bytes' of host 'from' to device 'to', through pinned staging
  /// unless 'from' is pinned.
  void upload(void* to, const void* from, size_t bytes, bool pinned);
  void* staging(size_t bytes);

  ColumnStore& store_;
  DeviceCache& cache_;
  cudaStream_t stream_;
  const QueryParams& params_;
  std::vector<void*> scratch_;
  std::vector<void*> batch_;
  void* staging_{nullptr};
  size_t stagingBytes_{0};
  const bool trace_;
  double lastMark_;
};

/// Lineitem-equivalent rows per batch when the hand-written queries run batch
/// by batch, as a batched engine would; 0 (the default) runs each kernel once
/// over all of its rows.
void setBatchRows(int64_t rows);
int64_t batchRows();

using QueryFunction = Result (*)(QueryContext&);

struct QuerySpec {
  int32_t number;
  QueryFunction run;
  std::vector<ResultField> schema;
};

/// The fused queries that exist, by TPC-H number.
const std::vector<QuerySpec>& querySpecs();
const QuerySpec* findQuery(int32_t number);

/// Who implements a query: the hand-written kernels of QuerySpec::run, or a
/// version composed from a library's operators over the same cached columns.
enum class Backend { kHandwritten, kCudf, kMeld };

const char* backendName(Backend backend);
Backend parseBackend(const std::string& name);

/// Adds a library version of a query, which must produce the same result
/// schema as the hand-written one.
void registerImplementation(Backend backend, int32_t query, QueryFunction function);

/// The version of `spec` that `backend` provides, or null if it has none.
QueryFunction implementation(const QuerySpec& spec, Backend backend);

} // namespace facebook::velox::supervelox
