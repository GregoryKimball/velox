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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::encoded {

/// How a column's values are stored. Follows DBCompiler's codec set, with the
/// frame of reference chosen per segment rather than per column.
enum class Codec : uint8_t {
  /// The cuDF column as read, used for high-cardinality strings.
  kPlain,
  /// Unsigned offsets from the segment minimum, bit-packed at the segment's
  /// width. Used for integers, dates and short decimals.
  kForBitpack,
  /// Sorted distinct values plus one code per row, the codes stored as
  /// kForBitpack. Used for low-cardinality strings.
  kDictionary,
};

const char* codecName(Codec codec);

/// Decode parameters of one segment of a bit-packed stream.
struct SegmentParams {
  int64_t base;
  uint64_t bitOffset;
  uint32_t bitWidth;
  uint32_t pad;
};

/// One column of an encoded table. Every column is split at the same segment
/// boundaries, every 'segmentRows' rows.
struct EncodedColumn {
  std::string name;
  /// The type the column decodes to.
  cudf::data_type type;
  Codec codec;

  /// kForBitpack values or kDictionary codes: packed 64-bit words, segments
  /// concatenated, each starting on a word boundary, plus one padding word.
  rmm::device_uvector<uint64_t> packed;
  rmm::device_uvector<SegmentParams> segmentParams;
  /// Per-segment minimum and maximum of the decoded integer value, for
  /// pruning. Empty for strings.
  std::vector<int64_t> segmentMin;
  std::vector<int64_t> segmentMax;
  std::vector<uint32_t> segmentBitWidth;

  /// kPlain: the whole column. kDictionary: the sorted distinct values.
  std::unique_ptr<cudf::column> values;

  /// Host copies of the streams above, filled by
  /// EncodedColumnCache::copyToHost() for CPU scans. 'hostOffsets' and
  /// 'hostChars' hold the strings of 'values': numValues + 1 offsets and the
  /// characters.
  std::vector<uint64_t> hostPacked;
  std::vector<SegmentParams> hostSegmentParams;
  std::vector<int64_t> hostOffsets;
  std::vector<char> hostChars;

  EncodedColumn(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /// Device bytes held by the column.
  int64_t encodedBytes() const;

  /// Host bytes held by the column.
  int64_t hostBytes() const;
};

/// A table held in device memory in encoded form.
struct EncodedTable {
  std::string name;
  int64_t numRows{0};
  int32_t segmentRows{0};
  int32_t numSegments{0};
  /// Column the rows were sorted by when loaded, or empty.
  std::string clusterKey;
  /// Set by EncodedColumnCache::moveTo(): the packed streams and kPlain value
  /// columns are in host memory, and scans stage them with stageSegments().
  /// Dictionaries and segment params stay in device memory, and
  /// EncodedColumn::hostSegmentParams holds a host copy of the params.
  bool hostResident{false};
  std::vector<std::unique_ptr<EncodedColumn>> columns;
  std::unordered_map<std::string, int32_t> columnIndex;

  const EncodedColumn& column(const std::string& name) const;

  /// Number of rows in 'segment'; only the last segment is short.
  int32_t segmentSize(int32_t segment) const;
};

/// Decodes 'column' for the rows of 'segments', in ascending segment order.
/// With 'rows', decodes only those rows, given as offsets into the
/// concatenation of 'segments'. 'segments' is a device copy of segment ids.
std::unique_ptr<cudf::column> decodeColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Copies 'columnNames' of a host-resident 'table' for 'segments' into device
/// memory from 'mr' with bulk copies on 'stream', merging adjacent segments
/// into one copy. 'segments' must be ascending with only the last one possibly
/// short. The result holds those segments in order as segments 0 to n-1, so
/// decodeColumn() reads it with segment ids 0 to n-1. Copies issued from
/// pinned host memory are asynchronous; keep the result alive until work on
/// 'stream' that reads it has completed.
std::unique_ptr<EncodedTable> stageSegments(
    const EncodedTable& table,
    const std::vector<std::string>& columnNames,
    std::span<const int32_t> segments,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Process-wide store of encoded tables, resident in device memory.
class EncodedColumnCache {
 public:
  static EncodedColumnCache& instance();

  /// Reads the Parquet file(s) of a table, sorts it by 'clusterKey' if
  /// non-empty, and encodes it in segments of 'segmentRows' rows.
  void loadTable(
      const std::string& name,
      const std::vector<std::string>& parquetFiles,
      const std::string& clusterKey,
      int32_t segmentRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  /// Loads like loadTable() without a cluster key, but reads one row group at
  /// a time, so tables larger than device memory load. String columns take
  /// the dictionary codec on the same distinct-count thresholds; each row
  /// group is read once, its strings coded against its own distinct values and
  /// recoded to the table's dictionary at the end. The packed streams, those
  /// interim codes and kPlain value columns are
  /// stored in memory from 'storageMr'. With 'hostResident' that is pinned
  /// host memory, and the table is marked EncodedTable::hostResident as by
  /// moveTo(). Dictionaries and segment params use 'mr'.
  void loadTableByRowGroups(
      const std::string& name,
      const std::vector<std::string>& parquetFiles,
      int32_t segmentRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      rmm::device_async_resource_ref storageMr,
      bool hostResident);

  /// Returns the table, or null if it is not loaded.
  const EncodedTable* findTable(const std::string& name) const;

  const EncodedTable& table(const std::string& name) const;

  /// Copies every loaded table to host memory for CPU scans. With
  /// 'releaseDevice', frees the device copies afterwards.
  void copyToHost(cuda::stream_ref stream, bool releaseDevice);

  /// Moves the packed streams and kPlain value columns of every loaded table
  /// into pinned host memory from 'mr' and marks the tables hostResident.
  /// Dictionaries and segment params stay in device memory. 'mr' must outlive
  /// the cache contents.
  void moveTo(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /// CSV size report: rows, codec, device and host bytes per column.
  std::string sizeReport() const;

  void clear();

 private:
  std::unordered_map<std::string, std::unique_ptr<EncodedTable>> tables_;
};

} // namespace facebook::velox::cudf_velox::connector::encoded
