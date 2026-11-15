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
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  static constexpr const char* kCudfStreamingGroupbyEnabled{
      "cudf.streaming_groupby_enabled"};
  static constexpr const char* kCudfStreamingGroupbyCapacityMultiplier{
      "cudf.streaming_groupby_capacity_multiplier"};
  static constexpr const char* kCudfStreamingGroupbyPartitions{
      "cudf.streaming_groupby_partitions"};
  static constexpr const char* kCudfCompactAggregationStateEnabled{
      "cudf.compact_aggregation_state_enabled"};
  static constexpr const char* kCudfDecimalGroupbyNarrowAccumulation{
      "cudf.decimal_groupby_narrow_accumulation"};
  static constexpr const char* kCudfCompactDecimalIntermediateState{
      "cudf.compact_decimal_intermediate_state"};
  static constexpr const char* kCudfPartialAggregationAbandonment{
      "cudf.partial_aggregation_abandonment"};
  static constexpr const char* kCudfNarrowDecimalSumState{
      "cudf.narrow_decimal_sum_state"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  static constexpr const char* kUcxExchange{"cudf.exchange"};
  static constexpr const char* kUcxxErrorHandling{"ucxx.error_handling"};
  static constexpr const char* kUcxIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kUcxxBlockingProgress{"ucxx.blocking_progress"};
  static constexpr const char* kUcxExchangeLogLevel{"cudf.exchange_log_level"};
  static constexpr const char* kUcxPartitionedOutputBatchRows{
      "cudf.partitioned_output_batch_rows"};
  static constexpr const char* kUcxExchangeMaxQueuedBytes{
      "cudf.exchange_max_queued_bytes"};
  static constexpr const char* kExchangeStore{"cudf.exchange_store"};
  static constexpr const char* kExchangeStoreDeviceBytes{
      "cudf.exchange_store_device_bytes"};
  static constexpr const char* kExchangeStoreHostBytes{
      "cudf.exchange_store_host_bytes"};
  static constexpr const char* kSplitGroupAdmissionFraction{
      "cudf.split_group_admission_fraction"};
  static constexpr const char* kCudfKvikioCacheCopyThreads{
      "cudf.kvikio_cache_copy_threads"};
  /// Query configuration key for selecting UCX exchange compression.
  static constexpr const char* kUcxExchangeCompression{
      "cudf.exchange_compression"};
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

  /// Bytes of received, not yet consumed GPU data one UCX exchange client may
  /// queue before its sources stop posting receives; they resume once the
  /// queue drains to half of it. Applies in addition to the bound on queued
  /// tables (UcxExchangeSource::kBackpressureHighWaterMark). Zero disables it.
  int64_t exchangeMaxQueuedBytes{0};

  /// Keeps Presto's temporary tables, which hold materialized exchanges, in
  /// the exchange store in device and pinned host memory instead of files.
  /// Needs the UCX exchange build.
  bool exchangeStore{false};

  /// Bytes of temporary table data the exchange store keeps in device memory;
  /// the rest goes to pinned host memory.
  int64_t exchangeStoreDeviceBytes{0};

  /// Bytes of temporary table data the exchange store keeps in pinned host
  /// memory; a query that needs more fails. 0 means no limit.
  int64_t exchangeStoreHostBytes{0};

  /// Share of the device memory available at startup that a task's split
  /// groups (grouped execution) may fill before the task starts another split
  /// group next to running ones (admitSplitGroup()). 0 starts them up to the
  /// task's concurrent split group limit.
  double splitGroupAdmissionFraction{0};

  /// Threads that copy KvikIO data from the async data cache to the GPU. One
  /// pool for the whole process, created by the first cached GPU read.
  int32_t kvikioCacheCopyThreads{16};

  /// Compression mode used by UCX exchange output unless a query overrides it
  /// with kUcxExchangeCompression.
  std::string exchangeCompression{"none"};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

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

  /// Number of streaming_groupby tables per final aggregation, split by a hash
  /// of the grouping keys. Total state is unchanged, but growing rebuilds one
  /// table and output materializes one table at a time, so their transient
  /// memory is about 1/N of a single table's. One keeps a single table.
  int32_t streamingGroupbyPartitions{1};

  /// Rewrite Decimal64 SUM partial and intermediate states to a compact
  /// DECIMAL(38, s) column instead of the VARBINARY state
  /// (CudfPlanRewriter). Every worker of a query must use the same setting.
  bool compactAggregationStateEnabled{false};

  /// Narrow raw decimal groupby SUM/AVG accumulation only after proving every
  /// possible group sum fits INT64. Intermediate and final merges stay INT128.
  bool decimalGroupbyNarrowAccumulation{false};

  /// Emit the marked 16-byte intermediate state for eligible DECIMAL64 SUM
  /// and AVG partial results. The historical 32-byte state remains the default
  /// and DECIMAL128 results always retain that representation.
  bool compactDecimalIntermediateState{false};

  /// Stop hashing in a partial groupby that does not reduce its input, by the
  /// rule of Velox CPU's HashAggregation (query configs
  /// abandon_partial_aggregation_min_rows and abandon_partial_aggregation_min_pct):
  /// later inputs are converted to intermediate rows one to one and emitted.
  bool partialAggregationAbandonment{false};

  /// Keep the compact Decimal64 SUM state (compactAggregationStateEnabled) at
  /// 64 bits, DECIMAL(18, s) instead of DECIMAL(38, s), from the partial
  /// aggregation through the exchange into the final one. A partial or
  /// intermediate sum enters the state only when it fits 64 bits (proven
  /// before summing or checked after a 128-bit sum); the final aggregation
  /// sums the states in 64 bits under a bound on all its input and fails
  /// rather than overflow. Every worker of a query must use the same setting.
  bool narrowDecimalSumState{false};

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
  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
};

} // namespace facebook::velox::cudf_velox
