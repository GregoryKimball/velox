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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/DenseGroupbyDevice.h"

#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

namespace facebook::velox::cudf_velox::dense {
namespace {

constexpr int kBlock = 256;

int blocks(int64_t rows) {
  return static_cast<int>(std::min<int64_t>((rows + kBlock - 1) / kBlock, 65536));
}




/// A fixed-width input column as integers or doubles.
struct Values {
  const char* data;
  const cudf::bitmask_type* nulls;
  cudf::size_type offset;
  cudf::type_id type;

  __device__ bool valid(cudf::size_type row) const {
    return nulls == nullptr || cudf::bit_is_set(nulls, offset + row);
  }

  __device__ int64_t integer(cudf::size_type row) const {
    switch (type) {
      case cudf::type_id::INT8:
        return reinterpret_cast<const int8_t*>(data)[row];
      case cudf::type_id::INT16:
        return reinterpret_cast<const int16_t*>(data)[row];
      case cudf::type_id::INT32:
      case cudf::type_id::DECIMAL32:
      case cudf::type_id::TIMESTAMP_DAYS:
        return reinterpret_cast<const int32_t*>(data)[row];
      default:
        return reinterpret_cast<const int64_t*>(data)[row];
    }
  }

  __device__ double floating(cudf::size_type row) const {
    return type == cudf::type_id::FLOAT32
        ? reinterpret_cast<const float*>(data)[row]
        : reinterpret_cast<const double*>(data)[row];
  }
};

Values values(const cudf::column_view& column) {
  CUDF_EXPECTS(
      cudf::is_fixed_width(column.type()),
      "Dense aggregation needs fixed width columns");
  return Values{
      static_cast<const char*>(column.head()) +
          static_cast<int64_t>(column.offset()) * cudf::size_of(column.type()),
      column.nullable() ? column.null_mask() : nullptr,
      column.offset(),
      column.type().id()};
}

/// Adds the sign-extended 'value' to the int128 at 'slot', as two 64-bit
/// halves, low first.
__device__ void addInt128(unsigned long long* slot, int64_t value) {
  const auto low = static_cast<unsigned long long>(value);
  const unsigned long long old = atomicAdd(slot, low);
  const unsigned long long carry = old + low < old ? 1ULL : 0ULL;
  const unsigned long long high = (value < 0 ? ~0ULL : 0ULL) + carry;
  if (high != 0) {
    atomicAdd(slot + 1, high);
  }
}

__device__ void minDouble(double* slot, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(slot);
  unsigned long long old = *bits;
  while (__longlong_as_double(static_cast<long long>(old)) > value) {
    const unsigned long long seen = atomicCAS(
        bits, old, static_cast<unsigned long long>(__double_as_longlong(value)));
    if (seen == old) {
      return;
    }
    old = seen;
  }
}

__device__ void maxDouble(double* slot, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(slot);
  unsigned long long old = *bits;
  while (__longlong_as_double(static_cast<long long>(old)) < value) {
    const unsigned long long seen = atomicCAS(
        bits, old, static_cast<unsigned long long>(__double_as_longlong(value)));
    if (seen == old) {
      return;
    }
    old = seen;
  }
}

/// Adds the int128 (low, high) to the int128 at 'slot'.
__device__ void addInt128Parts(unsigned long long* slot, unsigned long long low, unsigned long long high) {
  const unsigned long long old = atomicAdd(slot, low);
  high += old + low < old ? 1ULL : 0ULL;
  if (high != 0) {
    atomicAdd(slot + 1, high);
  }
}

/// Adds row 'row' of 'value', an integer or 128-bit decimal, to the int128 at
/// 'slot'.
__device__ void addInteger(unsigned long long* slot, const Values& value, cudf::size_type row) {
  if (value.type == cudf::type_id::DECIMAL128) {
    const __int128_t wide = reinterpret_cast<const __int128_t*>(value.data)[row];
    addInt128Parts(slot, static_cast<unsigned long long>(wide), static_cast<unsigned long long>(wide >> 64));
  } else {
    addInt128(slot, value.integer(row));
  }
}

constexpr int kMaxKeys = 8;

struct Keys {
  Values key[kMaxKeys];
  int64_t low[kMaxKeys];
  int64_t size[kMaxKeys];
  int64_t stride[kMaxKeys];
  int count;
};

__global__ void combineKernel(Keys keys, cudf::size_type rows, int64_t slots, int32_t* out) {
  for (cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    int64_t slot = 0;
    bool inRange = true;
    for (int k = 0; k < keys.count; ++k) {
      const int64_t offset = keys.key[k].integer(row) - keys.low[k];
      inRange &= offset >= 0 && offset < keys.size[k];
      slot += offset * keys.stride[k];
    }
    out[row] = static_cast<int32_t>(inRange ? slot : slots);
  }
}

constexpr int kMaxTargets = 16;
constexpr int64_t kSharedBytes = 48 * 1024;
constexpr int kSharedBlocks = 1024;

struct SharedTarget {
  Values value;
  Kind kind;
  bool floating;
  void* accumulators;
  unsigned long long* counts;
};

struct Targets {
  SharedTarget target[kMaxTargets];
  int count;
};

/// Shared memory: rows per slot, then per target counts per slot and two
/// words of value per slot.
__global__ void addSharedKernel(
    Values key,
    int64_t low,
    int64_t slots,
    cudf::size_type rows,
    Targets targets,
    unsigned long long* globalRows,
    int* outOfRange) {
  extern __shared__ unsigned long long shared[];
  unsigned long long* blockRows = shared;
  auto counts = [&](int t) { return shared + slots * (1 + 3 * t); };
  auto values = [&](int t) { return shared + slots * (2 + 3 * t); };
  for (int64_t i = threadIdx.x; i < slots; i += blockDim.x) {
    blockRows[i] = 0;
    for (int t = 0; t < targets.count; ++t) {
      const auto& target = targets.target[t];
      counts(t)[i] = 0;
      int64_t start = 0;
      if (target.kind == Kind::kMin || target.kind == Kind::kMax) {
        if (target.floating) {
          // Bits of +infinity and -infinity.
          start = target.kind == Kind::kMin ? 0x7ff0000000000000LL : static_cast<int64_t>(0xfff0000000000000ULL);
        } else {
          start = target.kind == Kind::kMin ? INT64_MAX : INT64_MIN;
        }
      }
      values(t)[2 * i] = static_cast<unsigned long long>(start);
      values(t)[2 * i + 1] = 0;
    }
  }
  __syncthreads();
  for (cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    const int64_t slot = key.integer(row) - low;
    if (slot < 0 || slot >= slots) {
      *outOfRange = 1;
      continue;
    }
    atomicAdd(&blockRows[slot], 1ULL);
    for (int t = 0; t < targets.count; ++t) {
      const auto& target = targets.target[t];
      if (target.kind == Kind::kCountRows || !target.value.valid(row)) {
        continue;
      }
      atomicAdd(&counts(t)[slot], 1ULL);
      unsigned long long* value = values(t) + 2 * slot;
      switch (target.kind) {
        case Kind::kSum:
        case Kind::kAvg:
          if (target.floating) {
            atomicAdd(reinterpret_cast<double*>(value), target.value.floating(row));
          } else {
            addInteger(value, target.value, row);
          }
          break;
        case Kind::kMin:
          if (target.floating) {
            minDouble(reinterpret_cast<double*>(value), target.value.floating(row));
          } else {
            atomicMin(reinterpret_cast<long long*>(value), static_cast<long long>(target.value.integer(row)));
          }
          break;
        case Kind::kMax:
          if (target.floating) {
            maxDouble(reinterpret_cast<double*>(value), target.value.floating(row));
          } else {
            atomicMax(reinterpret_cast<long long*>(value), static_cast<long long>(target.value.integer(row)));
          }
          break;
        default:
          break;
      }
    }
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < slots; i += blockDim.x) {
    if (blockRows[i] == 0) {
      continue;
    }
    atomicAdd(&globalRows[i], blockRows[i]);
    for (int t = 0; t < targets.count; ++t) {
      const auto& target = targets.target[t];
      if (target.kind == Kind::kCountRows || counts(t)[i] == 0) {
        continue;
      }
      atomicAdd(&target.counts[i], counts(t)[i]);
      const unsigned long long* value = values(t) + 2 * i;
      switch (target.kind) {
        case Kind::kSum:
        case Kind::kAvg:
          if (target.floating) {
            atomicAdd(static_cast<double*>(target.accumulators) + i, __longlong_as_double(static_cast<long long>(value[0])));
          } else {
            addInt128Parts(static_cast<unsigned long long*>(target.accumulators) + 2 * i, value[0], value[1]);
          }
          break;
        case Kind::kMin:
          if (target.floating) {
            minDouble(static_cast<double*>(target.accumulators) + i, __longlong_as_double(static_cast<long long>(value[0])));
          } else {
            atomicMin(static_cast<long long*>(target.accumulators) + i, static_cast<long long>(value[0]));
          }
          break;
        case Kind::kMax:
          if (target.floating) {
            maxDouble(static_cast<double*>(target.accumulators) + i, __longlong_as_double(static_cast<long long>(value[0])));
          } else {
            atomicMax(static_cast<long long*>(target.accumulators) + i, static_cast<long long>(value[0]));
          }
          break;
        default:
          break;
      }
    }
  }
}

__global__ void fillKernel(int64_t* out, int64_t count, int64_t value) {
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < count;
       i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = value;
  }
}

__global__ void addRowsKernel(
    Values key,
    int64_t low,
    int64_t slots,
    cudf::size_type rows,
    unsigned long long* counts,
    int* outOfRange) {
  for (cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    const int64_t slot = key.integer(row) - low;
    if (slot < 0 || slot >= slots) {
      *outOfRange = 1;
      continue;
    }
    atomicAdd(&counts[slot], 1ULL);
  }
}

__global__ void addKernel(
    Values key,
    int64_t low,
    int64_t slots,
    cudf::size_type rows,
    Values value,
    Kind kind,
    bool floating,
    void* accumulators,
    unsigned long long* counts) {
  for (cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    const int64_t slot = key.integer(row) - low;
    if (slot < 0 || slot >= slots || !value.valid(row)) {
      continue;
    }
    atomicAdd(&counts[slot], 1ULL);
    switch (kind) {
      case Kind::kSum:
      case Kind::kAvg:
        if (floating) {
          atomicAdd(static_cast<double*>(accumulators) + slot, value.floating(row));
        } else {
          addInteger(
              static_cast<unsigned long long*>(accumulators) + 2 * slot,
              value,
              row);
        }
        break;
      case Kind::kMin:
        if (floating) {
          minDouble(static_cast<double*>(accumulators) + slot, value.floating(row));
        } else {
          atomicMin(
              static_cast<long long*>(accumulators) + slot,
              static_cast<long long>(value.integer(row)));
        }
        break;
      case Kind::kMax:
        if (floating) {
          maxDouble(static_cast<double*>(accumulators) + slot, value.floating(row));
        } else {
          atomicMax(
              static_cast<long long*>(accumulators) + slot,
              static_cast<long long>(value.integer(row)));
        }
        break;
      default:
        break;
    }
  }
}

__global__ void keysKernel(
    const cudf::size_type* slots,
    cudf::size_type groups,
    int64_t low,
    int64_t stride,
    int64_t size,
    cudf::type_id type,
    void* out) {
  for (cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < groups;
       i += gridDim.x * blockDim.x) {
    const int64_t key = low + slots[i] / stride % size;
    switch (type) {
      case cudf::type_id::INT8:
        static_cast<int8_t*>(out)[i] = static_cast<int8_t>(key);
        break;
      case cudf::type_id::INT16:
        static_cast<int16_t*>(out)[i] = static_cast<int16_t>(key);
        break;
      case cudf::type_id::INT32:
      case cudf::type_id::TIMESTAMP_DAYS:
        static_cast<int32_t*>(out)[i] = static_cast<int32_t>(key);
        break;
      default:
        static_cast<int64_t*>(out)[i] = key;
    }
  }
}

/// The result of each group as an int128 or a double, written as 'type'.
__global__ void resultsKernel(
    const cudf::size_type* slots,
    cudf::size_type groups,
    Kind kind,
    bool floating,
    const void* accumulators,
    const unsigned long long* counts,
    cudf::type_id type,
    void* out,
    bool* valid) {
  for (cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < groups;
       i += gridDim.x * blockDim.x) {
    const int64_t slot = slots[i];
    const auto count = static_cast<int64_t>(counts[slot]);
    valid[i] = count > 0 || kind == Kind::kCount || kind == Kind::kCountRows;
    __int128_t integer = 0;
    double real = 0;
    bool isReal = false;
    switch (kind) {
      case Kind::kCountRows:
      case Kind::kCount:
        integer = count;
        break;
      case Kind::kSum:
      case Kind::kMin:
      case Kind::kMax:
        if (floating) {
          real = static_cast<const double*>(accumulators)[slot];
          isReal = true;
        } else if (kind == Kind::kSum) {
          integer = static_cast<const __int128_t*>(accumulators)[slot];
        } else {
          integer = static_cast<const int64_t*>(accumulators)[slot];
        }
        break;
      case Kind::kAvg:
        if (count == 0) {
          break;
        }
        if (floating) {
          real = static_cast<const double*>(accumulators)[slot] / count;
          isReal = true;
        } else {
          const __int128_t sum = static_cast<const __int128_t*>(accumulators)[slot];
          if (type == cudf::type_id::FLOAT64 || type == cudf::type_id::FLOAT32) {
            real = static_cast<double>(sum) / count;
            isReal = true;
          } else {
            // Rounds half away from zero, as the CPU's decimal average.
            integer = sum / count;
            const __int128_t remainder = sum % count;
            const __int128_t twice = remainder < 0 ? -2 * remainder : 2 * remainder;
            if (twice >= count) {
              integer += sum < 0 ? -1 : 1;
            }
          }
        }
        break;
    }
    switch (type) {
      case cudf::type_id::FLOAT64:
        static_cast<double*>(out)[i] = isReal ? real : static_cast<double>(integer);
        break;
      case cudf::type_id::FLOAT32:
        static_cast<float*>(out)[i] =
            static_cast<float>(isReal ? real : static_cast<double>(integer));
        break;
      case cudf::type_id::DECIMAL128:
        static_cast<__int128_t*>(out)[i] = integer;
        break;
      case cudf::type_id::INT32:
      case cudf::type_id::DECIMAL32:
      case cudf::type_id::TIMESTAMP_DAYS:
        static_cast<int32_t*>(out)[i] = static_cast<int32_t>(integer);
        break;
      case cudf::type_id::INT16:
        static_cast<int16_t*>(out)[i] = static_cast<int16_t>(integer);
        break;
      case cudf::type_id::INT8:
        static_cast<int8_t*>(out)[i] = static_cast<int8_t>(integer);
        break;
      default:
        static_cast<int64_t*>(out)[i] = static_cast<int64_t>(integer);
    }
  }
}

struct Occupied {
  const unsigned long long* rows;

  __device__ bool operator()(cudf::size_type slot) const {
    return rows[slot] > 0;
  }
};

} // namespace

void fill(int64_t* out, int64_t count, int64_t value, cuda::stream_ref stream) {
  fillKernel<<<blocks(count), kBlock, 0, stream.get()>>>(out, count, value);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void combine(
    const std::vector<cudf::column_view>& keys,
    const std::vector<int64_t>& lows,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t slots,
    int32_t* out,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(keys.size() <= kMaxKeys, "Too many dense aggregation keys");
  Keys args{};
  args.count = static_cast<int>(keys.size());
  for (size_t k = 0; k < keys.size(); ++k) {
    args.key[k] = values(keys[k]);
    args.low[k] = lows[k];
    args.size[k] = sizes[k];
    args.stride[k] = strides[k];
  }
  const auto rows = keys[0].size();
  combineKernel<<<blocks(rows), kBlock, 0, stream.get()>>>(args, rows, slots, out);
  CUDF_CUDA_TRY(cudaGetLastError());
}

int64_t sharedSlots(size_t targets) {
  if (targets > kMaxTargets) {
    return 0;
  }
  return kSharedBytes / static_cast<int64_t>(sizeof(unsigned long long) * (1 + 3 * targets));
}

void addShared(
    const cudf::column_view& key,
    int64_t low,
    int64_t slots,
    const std::vector<Target>& targets,
    unsigned long long* rows,
    int* outOfRange,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(slots <= sharedSlots(targets.size()), "Too many slots for shared accumulators");
  Targets args{};
  args.count = static_cast<int>(targets.size());
  for (size_t t = 0; t < targets.size(); ++t) {
    const auto& target = targets[t];
    args.target[t] = SharedTarget{
        target.kind == Kind::kCountRows ? Values{} : values(target.value),
        target.kind,
        target.floating,
        target.accumulators,
        target.counts};
  }
  const size_t bytes = slots * sizeof(unsigned long long) * (1 + 3 * targets.size());
  addSharedKernel<<<std::min(blocks(key.size()), kSharedBlocks), kBlock, bytes, stream.get()>>>(
      values(key), low, slots, key.size(), args, rows, outOfRange);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void addRows(
    const cudf::column_view& key,
    int64_t low,
    int64_t slots,
    unsigned long long* rows,
    int* outOfRange,
    cuda::stream_ref stream) {
  addRowsKernel<<<blocks(key.size()), kBlock, 0, stream.get()>>>(
      values(key), low, slots, key.size(), rows, outOfRange);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void add(
    const cudf::column_view& key,
    int64_t low,
    int64_t slots,
    const cudf::column_view& value,
    Kind kind,
    bool floating,
    void* accumulators,
    unsigned long long* counts,
    cuda::stream_ref stream) {
  addKernel<<<blocks(key.size()), kBlock, 0, stream.get()>>>(
      values(key),
      low,
      slots,
      key.size(),
      values(value),
      kind,
      floating,
      accumulators,
      counts);
  CUDF_CUDA_TRY(cudaGetLastError());
}

rmm::device_uvector<cudf::size_type> occupied(
    const unsigned long long* rows,
    int64_t slots,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  rmm::device_uvector<cudf::size_type> out(slots, stream, mr);
  const auto end = thrust::copy_if(
      rmm::exec_policy_nosync(stream, mr),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(static_cast<cudf::size_type>(slots)),
      out.begin(),
      Occupied{rows});
  out.resize(end - out.begin(), stream);
  return out;
}

void keys(
    const rmm::device_uvector<cudf::size_type>& slots,
    int64_t low,
    int64_t stride,
    int64_t size,
    cudf::mutable_column_view out,
    cuda::stream_ref stream) {
  const auto groups = static_cast<cudf::size_type>(slots.size());
  if (groups == 0) {
    return;
  }
  keysKernel<<<blocks(groups), kBlock, 0, stream.get()>>>(
      slots.data(), groups, low, stride, size, out.type().id(), out.head());
  CUDF_CUDA_TRY(cudaGetLastError());
}

void results(
    const rmm::device_uvector<cudf::size_type>& slots,
    Kind kind,
    bool floating,
    const void* accumulators,
    const unsigned long long* counts,
    cudf::mutable_column_view out,
    bool* valid,
    cuda::stream_ref stream) {
  const auto groups = static_cast<cudf::size_type>(slots.size());
  if (groups == 0) {
    return;
  }
  resultsKernel<<<blocks(groups), kBlock, 0, stream.get()>>>(
      slots.data(), groups, kind, floating, accumulators, counts, out.type().id(), out.head(), valid);
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace facebook::velox::cudf_velox::dense
