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

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfMemoryLimitBytes{
      "cudf.memory_limit_bytes"};
  static constexpr const char* kCudfSpillHostBytes{
      "cudf.spill_host_memory_bytes"};
  static constexpr const char* kCudfSpillHostNumaNode{
      "cudf.spill_host_numa_node"};
  static constexpr const char* kCudfSpillPartitionFanout{
      "cudf.spill_partition_fanout"};
  static constexpr const char* kCudfSpillGroupbyMaxRowsPerLeaf{
      "cudf.spill_groupby_max_rows_per_leaf"};
  static constexpr const char* kCudfSpillJoinPartitionBytes{
      "cudf.spill_join_partition_bytes"};
  static constexpr const char* kCudfGroupbySelfSpillPercent{
      "cudf.groupby_self_spill_percent"};
  static constexpr const char* kCudfGroupbySelfSpillTargetPercent{
      "cudf.groupby_self_spill_target_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  static constexpr const char* kCudfBatchSizeMinBytes{
      "cudf.batch_size_min_bytes"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  /// Hash table occupancy for cudf::hash_join build tables.
  static constexpr const char* kCudfHashJoinLoadFactor{
      "cudf.hash_join_load_factor"};
  static constexpr const char* kCudfDistinctSpillMb{
      "cudf.distinct_spill_mb"};
  static constexpr const char* kCudfDistinctSpillPartitions{
      "cudf.distinct_spill_partitions"};
  static constexpr const char* kCudfCompactDecimalSumState{
      "cudf.compact_decimal_sum_state"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  static constexpr const char* kCudfStreamingGroupbyEnabled{
      "cudf.streaming_groupby_enabled"};
  static constexpr const char* kCudfStreamingGroupbyCapacityMultiplier{
      "cudf.streaming_groupby_capacity_multiplier"};
  static constexpr const char* kCudfStreamingGroupbyTables{
      "cudf.streaming_groupby_tables"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  static constexpr const char* kUcxExchange{"cudf.exchange"};
  static constexpr const char* kUcxxErrorHandling{"ucxx.error_handling"};
  static constexpr const char* kUcxIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kUcxxBlockingProgress{"ucxx.blocking_progress"};
  static constexpr const char* kUcxExchangeLogLevel{"cudf.exchange_log_level"};
  static constexpr const char* kUcxPartitionedOutputBatchRows{
      "cudf.partitioned_output_batch_rows"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Enable GPU exchange operators (UcxExchange / UcxPartitionedOutput). This
  /// is a capability, not a request: it is read once at registerCudf() to
  /// decide whether the UCX transports are registered in this process at all.
  /// Which transport a given edge uses is named per node in the plan, so
  /// different edges of one plan may differ. Naming a transport nothing
  /// registered -- kUcx while this is false, or on a worker built without the
  /// UCX exchange -- is a user error from exec::Task, never a silent fallback.
  bool exchange{false};

  /// Whether to enable error handling in UCXX endpoints.
  bool ucxxErrorHandling{true};

  /// Whether intra-node exchange optimization is enabled.
  bool intraNodeExchange{false};

  /// Whether the UCX worker is set up for UCXX blocking progress mode: the
  /// wakeup feature on the context plus an epoll file descriptor on the worker,
  /// which is also what lets enqueueing work signal it. False creates a
  /// tag/active-message-only worker that the progress loop polls.
  ///
  /// Requesting a feature the fabric cannot provide removes that transport from
  /// UCX's selection instead of failing, so setting this on a fabric without
  /// wakeup support silently costs RDMA. Tested with this and
  /// kUcxxErrorHandling both true on InfiniBand / A100, and both false on AWS
  /// SRD, which supports neither and otherwise stops using RDMA.
  bool ucxxBlockingProgress{true};

  /// VLOG level for ucx-exchange source files.
  int32_t exchangeLogLevel{0};

  /// Minimum number of rows to accumulate in UCX partitioned output before
  /// flushing. Small inputs are buffered and concatenated when this threshold
  /// is reached, avoiding pathologically small exchange chunks. Set to 0 to
  /// disable accumulation.
  int64_t partitionedOutputBatchRows{10'000};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Capacity in bytes of the process GPU memory resource that query GPU pools
  /// are arbitrated under. Zero takes memoryPercent of the device memory free
  /// at registration. A nonzero value must not exceed that free memory. Only
  /// queries that were given a GPU pool with attachCudfMemoryPool() are
  /// accounted against it.
  uint64_t memoryLimitBytes{0};

  /// Limit in bytes of the pinned host memory that spilled device data may
  /// occupy, a multiple of 256. Zero turns spilling off: an operator that does
  /// not fit its query's GPU pool then fails. The limit is never derived from
  /// the machine's memory. There is no disk spilling.
  uint64_t spillHostBytes{0};

  /// NUMA node of the pinned host memory for spilled data. Negative leaves the
  /// placement to CUDA.
  int32_t spillHostNumaNode{-1};

  /// The least number of hash partitions a spillable group-by makes when one
  /// of its leaves outgrows spillGroupbyMaxRowsPerLeaf, from 2 to 1024. A
  /// wider split partitions each buffered row fewer times.
  int32_t spillPartitionFanout{64};

  /// The most rows one leaf of a spillable final or single group-by holds,
  /// which must be positive. A leaf that outgrows it is split into hash
  /// partitions. A leaf moves to host memory and is aggregated on the device
  /// as a whole, so this bounds the device memory either takes. Only the
  /// spillable state reads it. It does not depend on batchSizeMaxThreshold,
  /// which caps the tables other operators concatenate and may be unset.
  int32_t spillGroupbyMaxRowsPerLeaf{1 << 24};

  /// A hash join build whose estimated device memory exceeds this many bytes
  /// is split into hash partitions that are joined one after the other. Zero,
  /// the default, uses the room the query's GPU pool has when the build
  /// starts. A smaller value makes smaller partitions, which leaves more room
  /// for the join's output at the price of more probe rows going through host
  /// memory.
  uint64_t spillJoinPartitionBytes{0};

  /// A spillable final group-by moves its largest leaves to host memory by
  /// itself, from its driver thread, when the query's GPU pool is fuller than
  /// this percentage of its capacity after an input. That is cheaper than
  /// reaching the limit, which pauses the whole task to reclaim. Zero turns
  /// this off.
  int32_t groupbySelfSpillPercent{92};

  /// How far a self-spill goes: down to this percentage of the query's GPU
  /// capacity, which must not exceed groupbySelfSpillPercent. The gap leaves
  /// room for the transient allocations of other operators between two
  /// inputs.
  int32_t groupbySelfSpillTargetPercent{86};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat operators before supported Cudf
  /// operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators. Inputs are collected until
  /// batchSizeMinBytes is reached, or batchSizeMinThreshold otherwise.
  /// batchSizeMaxThreshold limits the rows in a concatenated batch.
  bool concatOptimizationEnabled{false};

  /// Use libcudf's persistent streaming_groupby for eligible final grouped
  /// aggregations. This is opt-in while it supports only a subset of the
  /// aggregation combinations supported by the regular cuDF groupby path.
  bool streamingGroupbyEnabled{false};

  /// Multiplier used to derive streaming_groupby's initial logical capacity
  /// from the first batch and to grow capacity when it is exhausted.
  double streamingGroupbyCapacityMultiplier{2.0};

  /// Number of tables a final streaming aggregation starts with: 1, 2, 4, 8,
  /// 16, 32 or 64. Each key lives in exactly one table, picked by a hash of
  /// the key. Only one table grows or produces output at a time, so more
  /// tables lower the peak GPU memory. When a table holds too many keys, the
  /// tables double, up to 64.
  int32_t streamingGroupbyTables{8};

  /// Minimum rows to accumulate before GPU-side concatenation (default 100k).
  /// Applies when batchSizeMinBytes is unset, and always to zero-column
  /// vectors, which have no GPU buffers to measure.
  int32_t batchSizeMinThreshold{100000};

  /// Estimated GPU bytes (CudfVector::estimateFlatSize()) to accumulate before
  /// concatenation. Overrides batchSizeMinThreshold when set.
  ///
  /// This is a flush threshold, not a bound: inputs are buffered whole, so the
  /// buffer can exceed it by up to one input. Peak GPU usage is about twice the
  /// buffered bytes while cudf::concatenate builds the output, plus any
  /// batchSizeMaxThreshold splits not yet drained downstream.
  ///
  /// With string columns, keep this well under 2 GiB unless
  /// LIBCUDF_LARGE_STRINGS_ENABLED is set; cuDF rejects strings columns whose
  /// character data overflows 32-bit offsets.
  std::optional<uint64_t> batchSizeMinBytes;

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;

  /// Desired occupancy of the cudf::hash_join hash table in (0, 1]. The table
  /// stores one 8-byte (hash, row index) slot per capacity entry, so the build
  /// side of a join with N rows costs N / loadFactor * 8 bytes on top of the
  /// build table itself. libcudf's default is 0.5 (fastest probes); 0.8 halves
  /// the table for large builds at a small probe cost, which is what lets a
  /// 1.5 B-row build fit next to its own data on a 48 GiB GPU.
  double hashJoinLoadFactor{0.5};

  /// A final-step distinct buffers its input on the GPU and deduplicates it
  /// once all of it has arrived. When this is positive, every time the buffered
  /// input reaches this many MiB it is hash-partitioned on the grouping keys
  /// into distinctSpillPartitions parts and moved to host memory; the output is
  /// then produced one partition at a time, so GPU memory holds about one
  /// partition instead of the whole input. 0 disables. Independent of the
  /// spill_host_memory_bytes pool, which does not cover distinct.
  int64_t distinctSpillMb{0};
  int32_t distinctSpillPartitions{16};

  /// Grouped partial and final aggregations exchange a SUM over a short
  /// decimal column as one DECIMAL(38, scale) column instead of the 32-byte
  /// VARBINARY state, without a CudfAggregationNode in the plan. Both steps
  /// must run in cuDF, as when all workers are GPU workers.
  bool compactDecimalSumState{false};

  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
};

} // namespace facebook::velox::cudf_velox
