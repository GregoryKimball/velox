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
#include <string>
#include <vector>

#include "velox/experimental/supervelox/store/ColumnStore.h"
#include "velox/experimental/supervelox/store/DeviceCache.h"

// Shared by the CUDA query files and the Velox operator, so this header must
// not include Velox or folly.

namespace facebook::velox::supervelox {

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

/// Everything a fused query needs while it runs: device-readable columns from
/// the HBM cache, derived structures, and scratch memory freed at the end.
class QueryContext {
 public:
  QueryContext(ColumnStore& store, DeviceCache& cache, cudaStream_t stream);
  ~QueryContext();

  const Column& meta(const std::string& table, const std::string& name) {
    return store_.column(table, name);
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

  ColumnStore& store_;
  DeviceCache& cache_;
  cudaStream_t stream_;
  std::vector<void*> scratch_;
  const bool trace_;
  double lastMark_;
};

using QueryFunction = Result (*)(QueryContext&);

struct QuerySpec {
  int32_t number;
  QueryFunction run;
  std::vector<ResultField> schema;
};

/// The fused queries that exist, by TPC-H number.
const std::vector<QuerySpec>& querySpecs();
const QuerySpec* findQuery(int32_t number);

} // namespace facebook::velox::supervelox
