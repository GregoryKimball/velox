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
#include "velox/experimental/supervelox/meld/MeldBackend.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "meldvelox/MeldVelox.h"
#include "velox/experimental/supervelox/queries/Derived.h"
#include "velox/experimental/supervelox/queries/Query.h"

namespace facebook::velox::supervelox::meld_backend {
namespace {

struct Library {
  decltype(&meld_velox_abi_version) abiVersion;
  decltype(&meld_velox_run) run;
  decltype(&meld_velox_run_tables) runTables;
  decltype(&meld_velox_adopt_device) adoptDevice;
  decltype(&meld_velox_free) free;
};

Library gMeld;

/// Rows [begin, end) of a table.
struct RowRange {
  int64_t begin{0};
  int64_t end{0};
};

const char* arrowFormat(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return "c";
    case DType::kInt16:
      return "s";
    case DType::kInt32:
      return "i";
    case DType::kInt64:
      return "l";
  }
  throw std::logic_error("unknown dtype");
}

/// One batch of a table's columns as the Arrow C data interface describes
/// it, pointing at the pinned host buffers. It owns only the structs.
class Batch {
 public:
  Batch(QueryContext& ctx, const std::string& table, const std::vector<std::string>& names, RowRange rows) {
    auto* schema = new SchemaHolder;
    auto* array = new ArrayHolder;
    const size_t count = names.size();
    schema->names = names;
    schema->children.resize(count);
    array->children.resize(count);
    array->buffers.resize(count);
    for (size_t i = 0; i < count; ++i) {
      const Column& column = ctx.meta(table, names[i]);
      const size_t width = dtypeSize(column.dtype);
      const void* device = deviceColumn(ctx, table, names[i], column.dtype);
      char* error = nullptr;
      if (gMeld.adoptDevice(column.data, column.bytes, reinterpret_cast<uint64_t>(device), &error)) {
        std::string message = error ? error : "unknown";
        gMeld.free(error);
        throw std::runtime_error("Meld cannot read the HBM cache: " + message);
      }
      schema->children[i] = ArrowSchema{
          arrowFormat(column.dtype), schema->names[i].c_str(), nullptr, 0, 0, nullptr, nullptr, &releaseChild, nullptr};
      array->buffers[i] = {nullptr, static_cast<const char*>(column.data) + rows.begin * width};
      array->children[i] = ArrowArray{
          rows.end - rows.begin, 0, 0, 2, 0, array->buffers[i].data(), nullptr, nullptr, &releaseChild, nullptr};
    }
    for (size_t i = 0; i < count; ++i) {
      schema->pointers.push_back(&schema->children[i]);
      array->pointers.push_back(&array->children[i]);
    }
    schema_ = ArrowSchema{
        "+s", "", nullptr, 0, static_cast<int64_t>(count), schema->pointers.data(), nullptr, &releaseSchema, schema};
    array_ = ArrowArray{
        rows.end - rows.begin,
        0,
        0,
        1,
        static_cast<int64_t>(count),
        array->rootBuffers,
        array->pointers.data(),
        nullptr,
        &releaseArray,
        array};
  }

  ArrowSchema schema_;
  ArrowArray array_;

 private:
  struct SchemaHolder {
    std::vector<std::string> names;
    std::vector<ArrowSchema> children;
    std::vector<ArrowSchema*> pointers;
  };

  struct ArrayHolder {
    const void* rootBuffers[1] = {nullptr};
    std::vector<std::array<const void*, 2>> buffers;
    std::vector<ArrowArray> children;
    std::vector<ArrowArray*> pointers;
  };

  static const void* deviceColumn(QueryContext& ctx, const std::string& table, const std::string& name, DType dtype) {
    switch (dtype) {
      case DType::kInt8:
        return ctx.col<int8_t>(table, name);
      case DType::kInt16:
        return ctx.col<int16_t>(table, name);
      case DType::kInt32:
        return ctx.col<int32_t>(table, name);
      case DType::kInt64:
        return ctx.col<int64_t>(table, name);
    }
    throw std::logic_error("unknown dtype");
  }

  template <typename T>
  static void releaseChild(T* child) {
    child->release = nullptr;
  }

  static void releaseSchema(ArrowSchema* schema) {
    delete static_cast<SchemaHolder*>(schema->private_data);
    schema->release = nullptr;
  }

  static void releaseArray(ArrowArray* array) {
    delete static_cast<ArrayHolder*>(array->private_data);
    array->release = nullptr;
  }
};

/// What a plan answered with: one struct array, read column by column.
class Answer {
 public:
  ~Answer() {
    if (array_.release) {
      array_.release(&array_);
    }
    if (schema_.release) {
      schema_.release(&schema_);
    }
  }

  int64_t rows() const {
    return array_.length;
  }

  /// Row `row` of column `column` as an integer, or nullopt if NULL.
  std::optional<int64_t> value(int32_t column, int64_t row) const {
    const ArrowArray* child = array_.children[column];
    const char format = schema_.children[column]->format[0];
    const int64_t index = child->offset + row;
    const auto* validity = static_cast<const uint8_t*>(child->buffers[0]);
    if (validity && !(validity[index / 8] >> (index % 8) & 1)) {
      return std::nullopt;
    }
    const void* data = child->buffers[1];
    switch (format) {
      case 'c':
        return static_cast<const int8_t*>(data)[index];
      case 's':
        return static_cast<const int16_t*>(data)[index];
      case 'i':
        return static_cast<const int32_t*>(data)[index];
      case 'l':
        return static_cast<const int64_t*>(data)[index];
      default:
        throw std::runtime_error(std::string("Meld answered with Arrow format ") + schema_.children[column]->format);
    }
  }

  ArrowSchema schema_{};
  ArrowArray array_{};
};

/// `text` with every `$name` replaced by its value in `values`.
std::string substitute(std::string text, const std::map<std::string, int64_t>& values) {
  for (const auto& [name, value] : values) {
    const std::string key = "$" + name;
    for (size_t at = text.find(key); at != std::string::npos; at = text.find(key, at)) {
      text.replace(at, key.size(), std::to_string(value));
    }
  }
  return text;
}

void run(QueryContext& ctx, const std::string& plan, Batch& batch, Answer& answer) {
  cudaStreamSynchronize(ctx.stream());
  MeldVeloxOptions options{1, MELD_VELOX_TARGET_GPU};
  char* error = nullptr;
  if (gMeld.run(
          plan.data(), plan.size(), &options, &batch.schema_, &batch.array_, 1, &answer.schema_, &answer.array_, &error)) {
    std::string message = error ? error : "unknown";
    gMeld.free(error);
    throw std::runtime_error("Meld could not run the plan: " + message);
  }
}

/// Runs a plan whose root reads several tables, each sent as one batch.
void run(QueryContext& ctx, const std::string& plan, const std::vector<std::pair<const char*, Batch*>>& batches, Answer& answer) {
  cudaStreamSynchronize(ctx.stream());
  MeldVeloxOptions options{1, MELD_VELOX_TARGET_GPU};
  std::vector<MeldVeloxTable> tables;
  for (const auto& [name, batch] : batches) {
    tables.push_back({name, &batch->schema_, &batch->array_, 1});
  }
  char* error = nullptr;
  if (gMeld.runTables(
          plan.data(), plan.size(), &options, tables.data(), tables.size(), &answer.schema_, &answer.array_, &error)) {
    std::string message = error ? error : "unknown";
    gMeld.free(error);
    throw std::runtime_error("Meld could not run the plan: " + message);
  }
}

/// Statements setting `%result` to whether `%field`, of type `type`, is one
/// of `codes`. A code of -1, which no row holds, stands for none.
std::string codeTest(const std::string& result, const std::string& field, const std::string& type, const CodeSet& codes) {
  std::vector<int32_t> members;
  for (int32_t code = 0; code < 256; ++code) {
    if (codes.contains(code)) {
      members.push_back(code);
    }
  }
  if (members.empty()) {
    members.push_back(-1);
  }
  std::string text;
  std::string tests;
  std::string types;
  for (const int32_t code : members) {
    const std::string name = fmt::format("{}_{}", result, code < 0 ? std::string("none") : std::to_string(code));
    text += fmt::format(
        "      %{0}_code = meld.literal {1} : {2}\n"
        "      %{0} = meld.call @equal_{2}(%{3}, %{0}_code) : ({2}, {2}) -> i1\n",
        name,
        code,
        type,
        field);
    tests += (tests.empty() ? "%" : ", %") + name;
    types += types.empty() ? "i1" : ", i1";
  }
  return text + fmt::format("      %{} = meld.call @or({}) : ({}) -> i1\n", result, tests, types);
}

/// The rows of lineitem's zones that may hold a shipdate in [low, high].
RowRange shipdateRows(QueryContext& ctx, int64_t low, int64_t high) {
  const int64_t rows = ctx.rows("lineitem");
  const int64_t zoneRows = QueryContext::kZoneRows;
  const int2* device = ctx.zones("lineitem", "l_shipdate");
  const auto zones = ctx.download(device, (rows + zoneRows - 1) / zoneRows);
  int64_t first = -1;
  int64_t last = -1;
  for (int64_t zone = 0; zone < static_cast<int64_t>(zones.size()); ++zone) {
    if (zones[zone].y >= low && zones[zone].x <= high) {
      first = first < 0 ? zone : first;
      last = zone;
    }
  }
  if (first < 0) {
    return {0, 0};
  }
  return {first * zoneRows, std::min(rows, (last + 1) * zoneRows)};
}

constexpr const char* kQ6 = R"(
meld.plan version "0.99.0" {
  meld.extension_urn @comparison "urn:comparison"
  meld.extension_urn @boolean "urn:boolean"
  meld.extension_urn @arithmetic "urn:arithmetic"
  meld.function @gte_i16 in @comparison "gte:i16_i16"
  meld.function @lte_i16 in @comparison "lte:i16_i16"
  meld.function @gte_i8 in @comparison "gte:i8_i8"
  meld.function @lte_i8 in @comparison "lte:i8_i8"
  meld.function @and in @boolean "and:bool"
  meld.function @multiply in @arithmetic "multiply:i64_i64"
  meld.function @sum in @arithmetic "sum:i64"

  meld.root names ["revenue"] {
    %lineitem = meld.named_table ["tpch", "lineitem"]
        columns ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"]
        : !meld.relation<i16, i8, i16, i32>

    %kept = meld.filter %lineitem {
    ^bb0(%shipdate: i16, %discount: i8, %quantity: i16, %price: i32):
      %date_low = meld.literal $date_low : i16
      %date_high = meld.literal $date_high : i16
      %discount_low = meld.literal $discount_low : i8
      %discount_high = meld.literal $discount_high : i8
      %quantity_high = meld.literal $quantity_high : i16
      %after = meld.call @gte_i16(%shipdate, %date_low) : (i16, i16) -> i1
      %before = meld.call @lte_i16(%shipdate, %date_high) : (i16, i16) -> i1
      %least = meld.call @gte_i8(%discount, %discount_low) : (i8, i8) -> i1
      %most = meld.call @lte_i8(%discount, %discount_high) : (i8, i8) -> i1
      %small = meld.call @lte_i16(%quantity, %quantity_high) : (i16, i16) -> i1
      %dates = meld.call @and(%after, %before) : (i1, i1) -> i1
      %discounts = meld.call @and(%least, %most) : (i1, i1) -> i1
      %rest = meld.call @and(%discounts, %small) : (i1, i1) -> i1
      %all = meld.call @and(%dates, %rest) : (i1, i1) -> i1
      meld.yield %all : i1
    }

    %amounts = meld.project %kept {
    ^bb0(%shipdate: i16, %discount: i8, %quantity: i16, %price: i32):
      %wide_price = meld.cast %price : i32 to i64 failure error
      %wide_discount = meld.cast %discount : i8 to i64 failure error
      %amount = meld.call @multiply(%wide_price, %wide_discount) : (i64, i64) -> i64
      meld.yield %amount : i64
    } : !meld.relation<i64>

    %revenue = meld.aggregate %amounts keys {
    ^bb0(%amount: i64):
      meld.yield
    } measures {
    ^bb0(%amount: i64):
      %total = meld.measure all @sum(%amount) : (i64) -> !meld.nullable<i64>
      meld.yield %total : !meld.nullable<i64>
    } : !meld.relation<!meld.nullable<i64>>

    meld.yield %revenue : !meld.relation<!meld.nullable<i64>>
  }
}
)";

/// The charge is summed as 100 * discounted + discounted * tax, since Meld
/// sums in 64 bits and the charge itself needs 128 at SF300.
constexpr const char* kQ1 = R"(
meld.plan version "0.99.0" {
  meld.extension_urn @comparison "urn:comparison"
  meld.extension_urn @arithmetic "urn:arithmetic"
  meld.function @lte_i16 in @comparison "lte:i16_i16"
  meld.function @subtract in @arithmetic "subtract:i64_i64"
  meld.function @multiply in @arithmetic "multiply:i64_i64"
  meld.function @sum in @arithmetic "sum:i64"
  meld.function @count in @arithmetic "count:i64"

  meld.root names ["flag", "status", "quantity", "price", "discounted", "taxed", "discount", "lines"] {
    %lineitem = meld.named_table ["tpch", "lineitem"]
        columns ["l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_shipdate"]
        : !meld.relation<i8, i8, i16, i32, i8, i8, i16>

    %kept = meld.filter %lineitem {
    ^bb0(%flag: i8, %status: i8, %quantity: i16, %price: i32, %discount: i8, %tax: i8, %shipdate: i16):
      %limit = meld.literal $date_high : i16
      %shipped = meld.call @lte_i16(%shipdate, %limit) : (i16, i16) -> i1
      meld.yield %shipped : i1
    }

    %rows = meld.project %kept {
    ^bb0(%flag: i8, %status: i8, %quantity: i16, %price: i32, %discount: i8, %tax: i8, %shipdate: i16):
      %wide_quantity = meld.cast %quantity : i16 to i64 failure error
      %wide_price = meld.cast %price : i32 to i64 failure error
      %wide_discount = meld.cast %discount : i8 to i64 failure error
      %wide_tax = meld.cast %tax : i8 to i64 failure error
      %hundred = meld.literal 100 : i64
      %kept_share = meld.call @subtract(%hundred, %wide_discount) : (i64, i64) -> i64
      %discounted = meld.call @multiply(%wide_price, %kept_share) : (i64, i64) -> i64
      %taxed = meld.call @multiply(%discounted, %wide_tax) : (i64, i64) -> i64
      meld.yield %flag, %status, %wide_quantity, %wide_price, %discounted, %taxed, %wide_discount
          : i8, i8, i64, i64, i64, i64, i64
    } : !meld.relation<i8, i8, i64, i64, i64, i64, i64>

    %grouped = meld.aggregate %rows keys {
    ^bb0(%flag: i8, %status: i8, %quantity: i64, %price: i64, %discounted: i64, %taxed: i64, %discount: i64):
      meld.yield %flag, %status : i8, i8
    } measures {
    ^bb0(%flag: i8, %status: i8, %quantity: i64, %price: i64, %discounted: i64, %taxed: i64, %discount: i64):
      %sum_quantity = meld.measure all @sum(%quantity) : (i64) -> !meld.nullable<i64>
      %sum_price = meld.measure all @sum(%price) : (i64) -> !meld.nullable<i64>
      %sum_discounted = meld.measure all @sum(%discounted) : (i64) -> !meld.nullable<i64>
      %sum_taxed = meld.measure all @sum(%taxed) : (i64) -> !meld.nullable<i64>
      %sum_discount = meld.measure all @sum(%discount) : (i64) -> !meld.nullable<i64>
      %lines = meld.measure all @count(%quantity) : (i64) -> i64
      meld.yield %sum_quantity, %sum_price, %sum_discounted, %sum_taxed, %sum_discount, %lines
          : !meld.nullable<i64>, !meld.nullable<i64>, !meld.nullable<i64>, !meld.nullable<i64>,
            !meld.nullable<i64>, i64
    } : !meld.relation<i8, i8, !meld.nullable<i64>, !meld.nullable<i64>, !meld.nullable<i64>,
                       !meld.nullable<i64>, !meld.nullable<i64>, i64>

    meld.yield %grouped : !meld.relation<i8, i8, !meld.nullable<i64>, !meld.nullable<i64>,
                                         !meld.nullable<i64>, !meld.nullable<i64>, !meld.nullable<i64>, i64>
  }
}
)";

/// What the join plans call.
constexpr const char* kJoinFunctions = R"(
  meld.extension_urn @comparison "urn:comparison"
  meld.extension_urn @boolean "urn:boolean"
  meld.extension_urn @arithmetic "urn:arithmetic"
  meld.function @equal_i8 in @comparison "equal:i8_i8"
  meld.function @equal_i16 in @comparison "equal:i16_i16"
  meld.function @equal_i32 in @comparison "equal:i32_i32"
  meld.function @lt_i16 in @comparison "lt:i16_i16"
  meld.function @gte_i16 in @comparison "gte:i16_i16"
  meld.function @lte_i16 in @comparison "lte:i16_i16"
  meld.function @gte_i8 in @comparison "gte:i8_i8"
  meld.function @lte_i8 in @comparison "lte:i8_i8"
  meld.function @and in @boolean "and:bool"
  meld.function @or in @boolean "or:bool"
  meld.function @subtract in @arithmetic "subtract:i64_i64"
  meld.function @multiply in @arithmetic "multiply:i64_i64"
  meld.function @sum in @arithmetic "sum:i64"
  meld.function @count in @arithmetic "count:i32"
)";

/// `text` with every `{{name}}` replaced by its statements in `slots`.
std::string fill(std::string text, const std::map<std::string, std::string>& slots) {
  for (const auto& [name, value] : slots) {
    const std::string key = "{{" + name + "}}";
    for (size_t at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size())) {
      text.replace(at, key.size(), value);
    }
  }
  return text;
}

/// Meld has neither a distinct aggregate nor a second aggregation in a
/// root, so the late orders come back one group each and the host counts
/// them by priority.
constexpr const char* kQ4 = R"(
meld.plan version "0.99.0" {
{{functions}}
  meld.root names ["orderkey", "priority", "lines"] {
    %lineitem = meld.named_table ["tpch", "lineitem"]
        columns ["l_orderkey", "l_commitdate", "l_receiptdate"] : !meld.relation<i32, i16, i16>
    %late = meld.filter %lineitem {
    ^bb0(%key: i32, %commit: i16, %receipt: i16):
      %is_late = meld.call @lt_i16(%commit, %receipt) : (i16, i16) -> i1
      meld.yield %is_late : i1
    }
    %orders = meld.named_table ["tpch", "orders"]
        columns ["o_orderkey", "o_orderpriority"] : !meld.relation<i32, i8>
    %paired = meld.cross_join %late, %orders : !meld.relation<i32, i16, i16>, !meld.relation<i32, i8>
    %matched = meld.filter %paired {
    ^bb0(%key: i32, %commit: i16, %receipt: i16, %orderkey: i32, %priority: i8):
      %same = meld.call @equal_i32(%key, %orderkey) : (i32, i32) -> i1
      meld.yield %same : i1
    }
    %grouped = meld.aggregate %matched keys {
    ^bb0(%key: i32, %commit: i16, %receipt: i16, %orderkey: i32, %priority: i8):
      meld.yield %orderkey, %priority : i32, i8
    } measures {
    ^bb0(%key: i32, %commit: i16, %receipt: i16, %orderkey: i32, %priority: i8):
      %lines = meld.measure all @count(%key) : (i32) -> i64
      meld.yield %lines : i64
    } : !meld.relation<i32, i8, i64>
    meld.yield %grouped : !meld.relation<i32, i8, i64>
  }
}
)";

constexpr const char* kQ12 = R"(
meld.plan version "0.99.0" {
{{functions}}
  meld.root names ["mode", "high", "low"] {
    %lineitem = meld.named_table ["tpch", "lineitem"]
        columns ["l_orderkey", "l_shipmode", "l_shipdate", "l_commitdate", "l_receiptdate"]
        : !meld.relation<i32, i8, i16, i16, i16>
    %kept = meld.filter %lineitem {
    ^bb0(%key: i32, %mode: i8, %ship: i16, %commit: i16, %receipt: i16):
{{modes}}
      %date_low = meld.literal $date_low : i16
      %date_high = meld.literal $date_high : i16
      %after = meld.call @gte_i16(%receipt, %date_low) : (i16, i16) -> i1
      %before = meld.call @lte_i16(%receipt, %date_high) : (i16, i16) -> i1
      %committed = meld.call @lt_i16(%commit, %receipt) : (i16, i16) -> i1
      %shipped = meld.call @lt_i16(%ship, %commit) : (i16, i16) -> i1
      %all = meld.call @and(%mode_ok, %after, %before, %committed, %shipped) : (i1, i1, i1, i1, i1) -> i1
      meld.yield %all : i1
    }
    %orders = meld.named_table ["tpch", "orders"]
        columns ["o_orderkey", "o_orderpriority"] : !meld.relation<i32, i8>
    %paired = meld.cross_join %kept, %orders
        : !meld.relation<i32, i8, i16, i16, i16>, !meld.relation<i32, i8>
    %matched = meld.filter %paired {
    ^bb0(%key: i32, %mode: i8, %ship: i16, %commit: i16, %receipt: i16, %orderkey: i32, %priority: i8):
      %same = meld.call @equal_i32(%key, %orderkey) : (i32, i32) -> i1
      meld.yield %same : i1
    }
    %flagged = meld.project %matched {
    ^bb0(%key: i32, %mode: i8, %ship: i16, %commit: i16, %receipt: i16, %orderkey: i32, %priority: i8):
{{high}}
      %one = meld.literal 1 : i64
      %zero = meld.literal 0 : i64
      %high = meld.if_then {
        meld.yield %is_high : i1
      } then {
        meld.yield %one : i64
      } else {
        meld.yield %zero : i64
      }
      %low = meld.if_then {
        meld.yield %is_high : i1
      } then {
        meld.yield %zero : i64
      } else {
        meld.yield %one : i64
      }
      meld.yield %mode, %high, %low : i8, i64, i64
    } : !meld.relation<i8, i64, i64>
    %grouped = meld.aggregate %flagged keys {
    ^bb0(%mode: i8, %high: i64, %low: i64):
      meld.yield %mode : i8
    } measures {
    ^bb0(%mode: i8, %high: i64, %low: i64):
      %highs = meld.measure all @sum(%high) : (i64) -> !meld.nullable<i64>
      %lows = meld.measure all @sum(%low) : (i64) -> !meld.nullable<i64>
      meld.yield %highs, %lows : !meld.nullable<i64>, !meld.nullable<i64>
    } : !meld.relation<i8, !meld.nullable<i64>, !meld.nullable<i64>>
    meld.yield %grouped : !meld.relation<i8, !meld.nullable<i64>, !meld.nullable<i64>>
  }
}
)";

constexpr const char* kQ14 = R"(
meld.plan version "0.99.0" {
{{functions}}
  meld.root names ["total", "promo"] {
    %lineitem = meld.named_table ["tpch", "lineitem"]
        columns ["l_partkey", "l_shipdate", "l_extendedprice", "l_discount"]
        : !meld.relation<i32, i16, i32, i8>
    %kept = meld.filter %lineitem {
    ^bb0(%key: i32, %shipdate: i16, %price: i32, %discount: i8):
      %date_low = meld.literal $date_low : i16
      %date_high = meld.literal $date_high : i16
      %after = meld.call @gte_i16(%shipdate, %date_low) : (i16, i16) -> i1
      %before = meld.call @lte_i16(%shipdate, %date_high) : (i16, i16) -> i1
      %both = meld.call @and(%after, %before) : (i1, i1) -> i1
      meld.yield %both : i1
    }
    %part = meld.named_table ["tpch", "part"]
        columns ["p_partkey", "p_type"] : !meld.relation<i32, i16>
    %paired = meld.cross_join %kept, %part
        : !meld.relation<i32, i16, i32, i8>, !meld.relation<i32, i16>
    %matched = meld.filter %paired {
    ^bb0(%key: i32, %shipdate: i16, %price: i32, %discount: i8, %partkey: i32, %type: i16):
      %same = meld.call @equal_i32(%key, %partkey) : (i32, i32) -> i1
      meld.yield %same : i1
    }
    %amounts = meld.project %matched {
    ^bb0(%key: i32, %shipdate: i16, %price: i32, %discount: i8, %partkey: i32, %type: i16):
{{promo}}
      %wide_price = meld.cast %price : i32 to i64 failure error
      %wide_discount = meld.cast %discount : i8 to i64 failure error
      %hundred = meld.literal 100 : i64
      %kept_share = meld.call @subtract(%hundred, %wide_discount) : (i64, i64) -> i64
      %volume = meld.call @multiply(%wide_price, %kept_share) : (i64, i64) -> i64
      %zero = meld.literal 0 : i64
      %promo_volume = meld.if_then {
        meld.yield %is_promo : i1
      } then {
        meld.yield %volume : i64
      } else {
        meld.yield %zero : i64
      }
      meld.yield %volume, %promo_volume : i64, i64
    } : !meld.relation<i64, i64>
    %totals = meld.aggregate %amounts keys {
    ^bb0(%volume: i64, %promo_volume: i64):
      meld.yield
    } measures {
    ^bb0(%volume: i64, %promo_volume: i64):
      %total = meld.measure all @sum(%volume) : (i64) -> !meld.nullable<i64>
      %promo = meld.measure all @sum(%promo_volume) : (i64) -> !meld.nullable<i64>
      meld.yield %total, %promo : !meld.nullable<i64>, !meld.nullable<i64>
    } : !meld.relation<!meld.nullable<i64>, !meld.nullable<i64>>
    meld.yield %totals : !meld.relation<!meld.nullable<i64>, !meld.nullable<i64>>
  }
}
)";

/// The lineitem-only conjuncts before the join, the three branches over
/// lines and parts after it.
constexpr const char* kQ19 = R"(
meld.plan version "0.99.0" {
{{functions}}
  meld.root names ["revenue"] {
    %lineitem = meld.named_table ["tpch", "lineitem"]
        columns ["l_partkey", "l_shipinstruct", "l_shipmode", "l_quantity", "l_extendedprice", "l_discount"]
        : !meld.relation<i32, i8, i8, i16, i32, i8>
    %kept = meld.filter %lineitem {
    ^bb0(%key: i32, %instruct: i8, %mode: i8, %quantity: i16, %price: i32, %discount: i8):
{{modes}}
      %instruct_code = meld.literal $instruct : i8
      %in_person = meld.call @equal_i8(%instruct, %instruct_code) : (i8, i8) -> i1
      %both = meld.call @and(%in_person, %mode_ok) : (i1, i1) -> i1
      meld.yield %both : i1
    }
    %part = meld.named_table ["tpch", "part"]
        columns ["p_partkey", "p_brand", "p_container", "p_size"] : !meld.relation<i32, i8, i8, i8>
    %paired = meld.cross_join %kept, %part
        : !meld.relation<i32, i8, i8, i16, i32, i8>, !meld.relation<i32, i8, i8, i8>
    %matched = meld.filter %paired {
    ^bb0(%key: i32, %instruct: i8, %mode: i8, %quantity: i16, %price: i32, %discount: i8,
         %partkey: i32, %brand: i8, %container: i8, %size: i8):
      %same = meld.call @equal_i32(%key, %partkey) : (i32, i32) -> i1
      meld.yield %same : i1
    }
    %chosen = meld.filter %matched {
    ^bb0(%key: i32, %instruct: i8, %mode: i8, %quantity: i16, %price: i32, %discount: i8,
         %partkey: i32, %brand: i8, %container: i8, %size: i8):
{{branches}}
      %any = meld.call @or(%branch_0, %branch_1, %branch_2) : (i1, i1, i1) -> i1
      meld.yield %any : i1
    }
    %amounts = meld.project %chosen {
    ^bb0(%key: i32, %instruct: i8, %mode: i8, %quantity: i16, %price: i32, %discount: i8,
         %partkey: i32, %brand: i8, %container: i8, %size: i8):
      %wide_price = meld.cast %price : i32 to i64 failure error
      %wide_discount = meld.cast %discount : i8 to i64 failure error
      %hundred = meld.literal 100 : i64
      %kept_share = meld.call @subtract(%hundred, %wide_discount) : (i64, i64) -> i64
      %amount = meld.call @multiply(%wide_price, %kept_share) : (i64, i64) -> i64
      meld.yield %amount : i64
    } : !meld.relation<i64>
    %revenue = meld.aggregate %amounts keys {
    ^bb0(%amount: i64):
      meld.yield
    } measures {
    ^bb0(%amount: i64):
      %total = meld.measure all @sum(%amount) : (i64) -> !meld.nullable<i64>
      meld.yield %total : !meld.nullable<i64>
    } : !meld.relation<!meld.nullable<i64>>
    meld.yield %revenue : !meld.relation<!meld.nullable<i64>>
  }
}
)";

/// One Q19 branch over a joined line and part, setting `%branch_<b>`.
constexpr const char* kQ19Branch = R"(
      %brand_{0}_code = meld.literal {1} : i8
      %brand_{0} = meld.call @equal_i8(%brand, %brand_{0}_code) : (i8, i8) -> i1
{2}
      %quantity_{0}_low = meld.literal {3} : i16
      %quantity_{0}_high = meld.literal {4} : i16
      %size_{0}_low = meld.literal {5} : i8
      %size_{0}_high = meld.literal {6} : i8
      %quantity_{0}_least = meld.call @gte_i16(%quantity, %quantity_{0}_low) : (i16, i16) -> i1
      %quantity_{0}_most = meld.call @lte_i16(%quantity, %quantity_{0}_high) : (i16, i16) -> i1
      %size_{0}_least = meld.call @gte_i8(%size, %size_{0}_low) : (i8, i8) -> i1
      %size_{0}_most = meld.call @lte_i8(%size, %size_{0}_high) : (i8, i8) -> i1
      %branch_{0} = meld.call @and(%brand_{0}, %container_{0}, %quantity_{0}_least, %quantity_{0}_most,
                                   %size_{0}_least, %size_{0}_most) : (i1, i1, i1, i1, i1, i1) -> i1
)";

Result q6(QueryContext& ctx) {
  const int64_t dateLow = ctx.param("shipdate_min", days(1994, 1, 1));
  const int64_t dateHigh = ctx.param("shipdate_max", days(1994, 12, 31));
  const std::string plan = substitute(
      kQ6,
      {{"date_low", dateLow},
       {"date_high", dateHigh},
       {"discount_low", ctx.param("discount_min", 5)},
       {"discount_high", ctx.param("discount_max", 7)},
       {"quantity_high", ctx.param("quantity_max", 2399)}});
  Result result(findQuery(6)->schema);
  const RowRange rows = shipdateRows(ctx, dateLow, dateHigh);
  if (rows.end == rows.begin) {
    result.decimals[0].push_back(0);
    return result;
  }
  Batch batch(ctx, "lineitem", {"l_shipdate", "l_discount", "l_quantity", "l_extendedprice"}, rows);
  Answer answer;
  run(ctx, plan, batch, answer);
  result.decimals[0].push_back(answer.rows() ? answer.value(0, 0).value_or(0) : 0);
  return result;
}

Result q1(QueryContext& ctx) {
  const int64_t dateHigh = ctx.param("shipdate_max", days(1998, 9, 2));
  const std::string plan = substitute(kQ1, {{"date_high", dateHigh}});
  const RowRange rows = shipdateRows(ctx, std::numeric_limits<int16_t>::min(), dateHigh);
  Batch batch(
      ctx,
      "lineitem",
      {"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_shipdate"},
      rows);
  Answer answer;
  run(ctx, plan, batch, answer);

  std::map<std::pair<int64_t, int64_t>, int64_t> order;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    order[{*answer.value(0, row), *answer.value(1, row)}] = row;
  }
  const auto& flags = ctx.meta("lineitem", "l_returnflag").dictionary;
  const auto& statuses = ctx.meta("lineitem", "l_linestatus").dictionary;
  Result result(findQuery(1)->schema);
  for (const auto& [key, row] : order) {
    const int64_t count = answer.value(7, row).value_or(0);
    if (count == 0) {
      continue;
    }
    auto sum = [&](int32_t column) { return answer.value(column, row).value_or(0); };
    result.strings[0].push_back(flags.at(key.first));
    result.strings[1].push_back(statuses.at(key.second));
    result.decimals[2].push_back(sum(2));
    result.decimals[3].push_back(sum(3));
    result.decimals[4].push_back(sum(4));
    result.decimals[5].push_back(static_cast<__int128>(sum(4)) * 100 + sum(5));
    result.doubles[6].push_back(sum(2) / 100.0 / count);
    result.doubles[7].push_back(sum(3) / 100.0 / count);
    result.doubles[8].push_back(sum(6) / 100.0 / count);
    result.ints[9].push_back(count);
  }
  return result;
}

Result q4(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(
      ctx,
      "orders",
      "o_orderdate",
      ctx.param("orderdate_min", days(1993, 7, 1)),
      ctx.param("orderdate_max", days(1993, 10, 1) - 1) + 1);
  const auto& names = ctx.meta("orders", "o_orderpriority").dictionary;
  Result result(findQuery(4)->schema);
  if (end <= begin) {
    return result;
  }
  const auto bounds = ctx.download(orderLineStarts(ctx) + begin, end - begin + 1);
  Batch lines(ctx, "lineitem", {"l_orderkey", "l_commitdate", "l_receiptdate"}, {bounds.front(), bounds.back()});
  Batch orders(ctx, "orders", {"o_orderkey", "o_orderpriority"}, {begin, end});
  Answer answer;
  run(ctx, fill(kQ4, {{"functions", kJoinFunctions}}), {{"tpch.lineitem", &lines}, {"tpch.orders", &orders}}, answer);

  std::vector<int64_t> counts(names.size());
  for (int64_t row = 0; row < answer.rows(); ++row) {
    ++counts.at(*answer.value(1, row));
  }
  for (size_t i = 0; i < names.size(); ++i) {
    if (counts[i]) {
      result.strings[0].push_back(names[i]);
      result.ints[1].push_back(counts[i]);
    }
  }
  return result;
}

/// The orders that can have lines received in the date range are
/// contiguous, and so are their lines; only those are sent.
Result q12(QueryContext& ctx) {
  const int64_t dateLow = ctx.param("receiptdate_min", days(1994, 1, 1));
  const int64_t dateHigh = ctx.param("receiptdate_max", days(1995, 1, 1) - 1);
  const CodeSet modes =
      codesParam(ctx, "shipmodes", dictionaryCodes(ctx, "lineitem", "l_shipmode", {"MAIL", "SHIP"}));
  const CodeSet high = codesParam(
      ctx, "high_priorities", dictionaryCodes(ctx, "orders", "o_orderpriority", {"1-URGENT", "2-HIGH"}));
  const auto lag = lineDateLag(ctx, "l_receiptdate");
  const auto [begin, end] =
      sortedRange(ctx, "orders", "o_orderdate", dateLow - lag.second, dateHigh + 1 - lag.first);
  const auto& names = ctx.meta("lineitem", "l_shipmode").dictionary;
  Result result(findQuery(12)->schema);
  if (end <= begin) {
    return result;
  }
  const auto bounds = ctx.download(orderLineStarts(ctx) + begin, end - begin + 1);
  const std::string plan = substitute(
      fill(
          kQ12,
          {{"functions", kJoinFunctions},
           {"modes", codeTest("mode_ok", "mode", "i8", modes)},
           {"high", codeTest("is_high", "priority", "i8", high)}}),
      {{"date_low", dateLow}, {"date_high", dateHigh}});
  Batch lines(
      ctx,
      "lineitem",
      {"l_orderkey", "l_shipmode", "l_shipdate", "l_commitdate", "l_receiptdate"},
      {bounds.front(), bounds.back()});
  Batch orders(ctx, "orders", {"o_orderkey", "o_orderpriority"}, {begin, end});
  Answer answer;
  run(ctx, plan, {{"tpch.lineitem", &lines}, {"tpch.orders", &orders}}, answer);

  std::map<int64_t, std::pair<int64_t, int64_t>> counts;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    counts[*answer.value(0, row)] = {answer.value(1, row).value_or(0), answer.value(2, row).value_or(0)};
  }
  for (const auto& [mode, count] : counts) {
    if (count.first + count.second) {
      result.strings[0].push_back(names.at(mode));
      result.ints[1].push_back(count.first);
      result.ints[2].push_back(count.second);
    }
  }
  return result;
}

Result q14(QueryContext& ctx) {
  const int64_t dateLow = ctx.param("shipdate_min", days(1995, 9, 1));
  const int64_t dateHigh = ctx.param("shipdate_max", days(1995, 10, 1) - 1);
  const CodeSet promo =
      codesParam(ctx, "promo_types", dictionaryCodes(ctx, "part", "p_type", Match::kPrefix, "PROMO"));
  Result result(findQuery(14)->schema);
  const RowRange rows = shipdateRows(ctx, dateLow, dateHigh);
  if (rows.end == rows.begin) {
    result.doubles[0].push_back(0.0 / 0.0);
    return result;
  }
  const std::string plan = substitute(
      fill(kQ14, {{"functions", kJoinFunctions}, {"promo", codeTest("is_promo", "type", "i16", promo)}}),
      {{"date_low", dateLow}, {"date_high", dateHigh}});
  Batch lines(ctx, "lineitem", {"l_partkey", "l_shipdate", "l_extendedprice", "l_discount"}, rows);
  Batch parts(ctx, "part", {"p_partkey", "p_type"}, {0, ctx.rows("part")});
  Answer answer;
  run(ctx, plan, {{"tpch.lineitem", &lines}, {"tpch.part", &parts}}, answer);
  const int64_t total = answer.rows() ? answer.value(0, 0).value_or(0) : 0;
  const int64_t promoted = answer.rows() ? answer.value(1, 0).value_or(0) : 0;
  result.doubles[0].push_back(100.0 * static_cast<double>(promoted) / static_cast<double>(total));
  return result;
}

Result q19(QueryContext& ctx) {
  const int64_t instruct = ctx.param("shipinstruct", ctx.meta("lineitem", "l_shipinstruct").code("DELIVER IN PERSON"));
  const CodeSet modes =
      codesParam(ctx, "shipmodes", dictionaryCodes(ctx, "lineitem", "l_shipmode", {"AIR", "AIR REG"}));
  const Column& brand = ctx.meta("part", "p_brand");
  const char* defaultBrands[3] = {"Brand#12", "Brand#23", "Brand#34"};
  const std::vector<std::string> defaultContainers[3] = {
      {"SM CASE", "SM BOX", "SM PACK", "SM PKG"},
      {"MED BAG", "MED BOX", "MED PKG", "MED PACK"},
      {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}};
  const int64_t defaultQuantity[3][2] = {{100, 1100}, {1000, 2000}, {2000, 3000}};
  const int64_t defaultSize[3] = {5, 10, 15};
  std::string branches;
  for (int b = 0; b < 3; ++b) {
    const std::string n = std::to_string(b);
    const CodeSet containers =
        codesParam(ctx, "containers" + n, dictionaryCodes(ctx, "part", "p_container", defaultContainers[b]));
    branches += fmt::format(
        kQ19Branch,
        b,
        ctx.param("brand" + n, brand.code(defaultBrands[b])),
        codeTest("container_" + n, "container", "i8", containers),
        ctx.param("quantity_min" + n, defaultQuantity[b][0]),
        ctx.param("quantity_max" + n, defaultQuantity[b][1]),
        ctx.param("size_min" + n, 1),
        ctx.param("size_max" + n, defaultSize[b]));
  }
  const std::string plan = substitute(
      fill(
          kQ19,
          {{"functions", kJoinFunctions},
           {"modes", codeTest("mode_ok", "mode", "i8", modes)},
           {"branches", branches}}),
      {{"instruct", instruct}});
  Batch lines(
      ctx,
      "lineitem",
      {"l_partkey", "l_shipinstruct", "l_shipmode", "l_quantity", "l_extendedprice", "l_discount"},
      {0, ctx.rows("lineitem")});
  Batch parts(ctx, "part", {"p_partkey", "p_brand", "p_container", "p_size"}, {0, ctx.rows("part")});
  Answer answer;
  run(ctx, plan, {{"tpch.lineitem", &lines}, {"tpch.part", &parts}}, answer);
  Result result(findQuery(19)->schema);
  result.decimals[0].push_back(answer.rows() ? answer.value(0, 0).value_or(0) : 0);
  return result;
}

const char* mlirType(DType dtype) {
  switch (dtype) {
    case DType::kInt8:
      return "i8";
    case DType::kInt16:
      return "i16";
    case DType::kInt32:
      return "i32";
    case DType::kInt64:
      return "i64";
  }
  throw std::logic_error("unknown dtype");
}

/// Writes a plan's root a relation at a time. Fields are named after their
/// columns, which TPC-H keeps distinct across tables, so every block names a
/// joined row's fields the way the statements in it refer to them.
class PlanWriter {
 public:
  struct Field {
    std::string name;
    std::string type;
  };

  struct Relation {
    std::string value;
    std::vector<Field> fields;
  };

  /// `function` is sum or count, over the i64 field `argument`.
  struct Measure {
    std::string name;
    std::string function;
    std::string argument;
  };

  explicit PlanWriter(QueryContext& ctx) : ctx_(ctx) {}

  /// The MLIR type of a stored column.
  std::string type(const std::string& table, const std::string& column) {
    return mlirType(ctx_.meta(table, column).dtype);
  }

  Relation table(const std::string& table, const std::vector<std::string>& columns) {
    Relation relation{next(), {}};
    std::string names;
    for (const auto& column : columns) {
      relation.fields.push_back({column, type(table, column)});
      names += fmt::format("{}\"{}\"", names.empty() ? "" : ", ", column);
    }
    body_ += fmt::format(
        "    %{} = meld.named_table [\"tpch\", \"{}\"] columns [{}] : {}\n",
        relation.value,
        table,
        names,
        relationType(relation.fields));
    return relation;
  }

  /// Keeps the rows for which `condition`, a field set by `statements`, holds.
  Relation filter(const Relation& input, const std::string& statements, const std::string& condition) {
    Relation relation{next(), input.fields};
    body_ += fmt::format(
        "    %{} = meld.filter %{} {{\n{}{}      meld.yield %{} : i1\n    }}\n",
        relation.value,
        input.value,
        block(input.fields),
        statements,
        condition);
    return relation;
  }

  /// The pairs of `left` and the table `right` with equal keys.
  Relation join(const Relation& left, const Relation& right, const std::string& leftKey, const std::string& rightKey) {
    Relation paired{next(), left.fields};
    paired.fields.insert(paired.fields.end(), right.fields.begin(), right.fields.end());
    body_ += fmt::format(
        "    %{} = meld.cross_join %{}, %{} : {}, {}\n",
        paired.value,
        left.value,
        right.value,
        relationType(left.fields),
        relationType(right.fields));
    const std::string keyType = fieldType(paired.fields, leftKey);
    if (keyType != fieldType(paired.fields, rightKey)) {
      throw std::runtime_error("Meld join keys " + leftKey + " and " + rightKey + " differ in type");
    }
    return filter(
        paired,
        fmt::format("      %same = meld.call @equal_{0}(%{1}, %{2}) : ({0}, {0}) -> i1\n", keyType, leftKey, rightKey),
        "same");
  }

  /// The fields `outputs`, set by `statements` or passed through.
  Relation project(const Relation& input, const std::string& statements, const std::vector<Field>& outputs) {
    Relation relation{next(), outputs};
    body_ += fmt::format(
        "    %{} = meld.project %{} {{\n{}{}      meld.yield {} : {}\n    }} : {}\n",
        relation.value,
        input.value,
        block(input.fields),
        statements,
        values(outputs),
        types(outputs),
        relationType(outputs));
    return relation;
  }

  Relation aggregate(const Relation& input, const std::vector<std::string>& keys, const std::vector<Measure>& measures) {
    Relation relation{next(), {}};
    for (const auto& key : keys) {
      relation.fields.push_back({key, fieldType(input.fields, key)});
    }
    const std::string keyYield = keys.empty()
        ? "      meld.yield\n"
        : fmt::format("      meld.yield {} : {}\n", values(relation.fields), types(relation.fields));
    std::string statements;
    std::vector<Field> results;
    for (const auto& measure : measures) {
      const std::string resultType = measure.function == "count" ? "i64" : "!meld.nullable<i64>";
      statements += fmt::format(
          "      %{} = meld.measure all @{}(%{}) : (i64) -> {}\n", measure.name, measure.function, measure.argument, resultType);
      results.push_back({measure.name, resultType});
    }
    relation.fields.insert(relation.fields.end(), results.begin(), results.end());
    body_ += fmt::format(
        "    %{} = meld.aggregate %{} keys {{\n{}{}    }} measures {{\n{}{}      meld.yield {} : {}\n    }} : {}\n",
        relation.value,
        input.value,
        block(input.fields),
        keyYield,
        block(input.fields),
        statements,
        values(results),
        types(results),
        relationType(relation.fields));
    return relation;
  }

  std::string plan(const Relation& root) const {
    std::string names;
    for (const auto& field : root.fields) {
      names += fmt::format("{}\"{}\"", names.empty() ? "" : ", ", field.name);
    }
    return fmt::format(
        "meld.plan version \"0.99.0\" {{\n{}\n  meld.root names [{}] {{\n{}    meld.yield %{} : {}\n  }}\n}}\n",
        functions(),
        names,
        body_,
        root.value,
        relationType(root.fields));
  }

 private:
  std::string next() {
    return "r" + std::to_string(relations_++);
  }

  static std::string fieldType(const std::vector<Field>& fields, const std::string& name) {
    for (const auto& field : fields) {
      if (field.name == name) {
        return field.type;
      }
    }
    throw std::logic_error("no field " + name);
  }

  static std::string values(const std::vector<Field>& fields) {
    std::string text;
    for (const auto& field : fields) {
      text += (text.empty() ? "%" : ", %") + field.name;
    }
    return text;
  }

  static std::string types(const std::vector<Field>& fields) {
    std::string text;
    for (const auto& field : fields) {
      text += (text.empty() ? "" : ", ") + field.type;
    }
    return text;
  }

  static std::string relationType(const std::vector<Field>& fields) {
    return "!meld.relation<" + types(fields) + ">";
  }

  static std::string block(const std::vector<Field>& fields) {
    std::string text;
    for (const auto& field : fields) {
      text += fmt::format("{}%{}: {}", text.empty() ? "" : ", ", field.name, field.type);
    }
    return "    ^bb0(" + text + "):\n";
  }

  static std::string functions() {
    std::string text =
        "  meld.extension_urn @comparison \"urn:comparison\"\n"
        "  meld.extension_urn @boolean \"urn:boolean\"\n"
        "  meld.extension_urn @arithmetic \"urn:arithmetic\"\n";
    for (const char* type : {"i8", "i16", "i32", "i64"}) {
      for (const char* name : {"equal", "not_equal", "lt", "lte", "gt", "gte"}) {
        text += fmt::format("  meld.function @{0}_{1} in @comparison \"{0}:{1}_{1}\"\n", name, type);
      }
    }
    text +=
        "  meld.function @and in @boolean \"and:bool\"\n"
        "  meld.function @or in @boolean \"or:bool\"\n"
        "  meld.function @add in @arithmetic \"add:i64_i64\"\n"
        "  meld.function @subtract in @arithmetic \"subtract:i64_i64\"\n"
        "  meld.function @multiply in @arithmetic \"multiply:i64_i64\"\n"
        "  meld.function @sum in @arithmetic \"sum:i64\"\n"
        "  meld.function @count in @arithmetic \"count:i64\"\n";
    return text;
  }

  QueryContext& ctx_;
  std::string body_;
  int32_t relations_{0};
};

using Field = PlanWriter::Field;

/// One statement of a block, setting `%name`.
std::string literal(const std::string& name, int64_t value, const std::string& type) {
  return fmt::format("      %{} = meld.literal {} : {}\n", name, value, type);
}

/// `%name` = `%a` `comparison` `%b`, both of `type`.
std::string compare(
    const std::string& name,
    const std::string& comparison,
    const std::string& a,
    const std::string& b,
    const std::string& type) {
  return fmt::format("      %{0} = meld.call @{1}_{4}(%{2}, %{3}) : ({4}, {4}) -> i1\n", name, comparison, a, b, type);
}

/// `%name` = `%field` `comparison` `value`.
std::string compareTo(
    const std::string& name,
    const std::string& comparison,
    const std::string& field,
    int64_t value,
    const std::string& type) {
  return literal(name + "_value", value, type) + compare(name, comparison, field, name + "_value", type);
}

/// `%name` = the `connective` (and, or) of the i1 fields `terms`.
std::string connect(const std::string& name, const std::string& connective, const std::vector<std::string>& terms) {
  std::string values;
  std::string types;
  for (const auto& term : terms) {
    values += (values.empty() ? "%" : ", %") + term;
    types += types.empty() ? "i1" : ", i1";
  }
  return fmt::format("      %{} = meld.call @{}({}) : ({}) -> i1\n", name, connective, values, types);
}

std::string widen(const std::string& name, const std::string& field, const std::string& type) {
  return fmt::format("      %{} = meld.cast %{} : {} to i64 failure error\n", name, field, type);
}

/// `%name` = `%a` `operation` (add, subtract, multiply) `%b`, in i64.
std::string arithmetic(const std::string& name, const std::string& operation, const std::string& a, const std::string& b) {
  return fmt::format("      %{} = meld.call @{}(%{}, %{}) : (i64, i64) -> i64\n", name, operation, a, b);
}

/// `%name` = `%condition` ? `%a` : `%b`, all i64.
std::string choose(const std::string& name, const std::string& condition, const std::string& a, const std::string& b) {
  return fmt::format(
      "      %{} = meld.if_then {{\n        meld.yield %{} : i1\n      }} then {{\n        meld.yield %{} : i64\n"
      "      }} else {{\n        meld.yield %{} : i64\n      }}\n",
      name,
      condition,
      a,
      b);
}

/// `%name` = price * (100 - discount) in i64.
std::string volume(const std::string& name, const std::string& price, const std::string& discount, PlanWriter& writer) {
  return widen(name + "_price", price, writer.type("lineitem", "l_extendedprice")) +
      widen(name + "_discount", discount, writer.type("lineitem", "l_discount")) + literal(name + "_hundred", 100, "i64") +
      arithmetic(name + "_share", "subtract", name + "_hundred", name + "_discount") +
      arithmetic(name, "multiply", name + "_price", name + "_share");
}

CodeSet nationSet(uint32_t nations) {
  CodeSet codes;
  for (int32_t n = 0; n < 32; ++n) {
    if ((nations >> n) & 1) {
      codes.add(n);
    }
  }
  return codes;
}

/// The lines of orders [begin, end), which are contiguous.
RowRange linesOf(QueryContext& ctx, int64_t begin, int64_t end) {
  const uint32_t* starts = orderLineStarts(ctx);
  return {ctx.download(starts + begin, 1)[0], ctx.download(starts + end, 1)[0]};
}

/// Runs `plan` over `tables`, each one batch, single tables included.
void runPlan(QueryContext& ctx, const std::string& plan, std::vector<std::pair<const char*, Batch*>> tables, Answer& answer) {
  if (tables.size() == 1) {
    run(ctx, plan, *tables.front().second, answer);
  } else {
    run(ctx, plan, tables, answer);
  }
}

Result q2(QueryContext& ctx) {
  PlanWriter w(ctx);
  const std::string sizeType = w.type("part", "p_size");
  const std::string typeType = w.type("part", "p_type");
  const std::string nationType = w.type("supplier", "s_nationkey");
  auto offers = w.join(
      w.table("partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost"}),
      w.table("part", {"p_partkey", "p_size", "p_type"}),
      "ps_partkey",
      "p_partkey");
  auto wanted = w.filter(
      offers,
      compareTo("sized", "equal", "p_size", 15, sizeType) +
          codeTest("brass", "p_type", typeType, dictionaryCodes(ctx, "part", "p_type", Match::kSuffix, "BRASS")) +
          connect("wanted", "and", {"sized", "brass"}),
      "wanted");
  auto supplied = w.join(wanted, w.table("supplier", {"s_suppkey", "s_nationkey"}), "ps_suppkey", "s_suppkey");
  auto european = w.filter(
      supplied, codeTest("in_region", "s_nationkey", nationType, nationSet(nationsInRegion(ctx, "EUROPE"))), "in_region");
  auto counted = w.project(
      european,
      literal("one", 1, "i64"),
      {{"ps_partkey", w.type("partsupp", "ps_partkey")},
       {"ps_suppkey", w.type("partsupp", "ps_suppkey")},
       {"ps_supplycost", w.type("partsupp", "ps_supplycost")},
       {"one", "i64"}});
  auto grouped = w.aggregate(counted, {"ps_partkey", "ps_suppkey", "ps_supplycost"}, {{"offers", "count", "one"}});
  Batch partsupp(ctx, "partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost"}, {0, ctx.rows("partsupp")});
  Batch part(ctx, "part", {"p_partkey", "p_size", "p_type"}, {0, ctx.rows("part")});
  Batch supplier(ctx, "supplier", {"s_suppkey", "s_nationkey"}, {0, ctx.rows("supplier")});
  Answer answer;
  runPlan(ctx, w.plan(grouped), {{"tpch.partsupp", &partsupp}, {"tpch.part", &part}, {"tpch.supplier", &supplier}}, answer);

  std::map<int64_t, int64_t> cheapest;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    const int64_t partkey = *answer.value(0, row);
    const int64_t cost = *answer.value(2, row);
    auto [it, added] = cheapest.emplace(partkey, cost);
    if (!added) {
      it->second = std::min(it->second, cost);
    }
  }
  const Column& acctbal = ctx.meta("supplier", "s_acctbal");
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& nationOf = ctx.meta("supplier", "s_nationkey");
  struct Row {
    int64_t acctbal;
    std::string nation;
    std::string name;
    int64_t partkey;
    int64_t supplier;
  };
  std::vector<Row> sorted;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    const int64_t partkey = *answer.value(0, row);
    if (*answer.value(2, row) != cheapest[partkey]) {
      continue;
    }
    const int64_t supplier = *answer.value(1, row) - 1;
    sorted.push_back(
        {acctbal.value(supplier), nationName(ctx, nationOf.value(supplier)), name.string(supplier), partkey, supplier});
  }
  std::sort(sorted.begin(), sorted.end(), [](const Row& x, const Row& y) {
    return std::make_tuple(-x.acctbal, std::cref(x.nation), std::cref(x.name), x.partkey) <
        std::make_tuple(-y.acctbal, std::cref(y.nation), std::cref(y.name), y.partkey);
  });
  sorted.resize(std::min<size_t>(sorted.size(), 100));
  const Column& mfgr = ctx.meta("part", "p_mfgr");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  const Column& comment = ctx.meta("supplier", "s_comment");
  Result result(findQuery(2)->schema);
  for (const auto& row : sorted) {
    result.decimals[0].push_back(row.acctbal);
    result.strings[1].push_back(row.name);
    result.strings[2].push_back(row.nation);
    result.ints[3].push_back(row.partkey);
    result.strings[4].push_back(mfgr.string(row.partkey - 1));
    result.strings[5].push_back(address.string(row.supplier));
    result.strings[6].push_back(phone.string(row.supplier));
    result.strings[7].push_back(comment.string(row.supplier));
  }
  return result;
}

Result q3(QueryContext& ctx) {
  const int32_t date = days(1995, 3, 15);
  const auto lag = lineDateLag(ctx, "l_shipdate");
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", date - lag.second + 1, date);
  Result result(findQuery(3)->schema);
  if (end <= begin) {
    return result;
  }
  PlanWriter w(ctx);
  auto lines = w.filter(
      w.table("lineitem", {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"}),
      compareTo("shipped", "gt", "l_shipdate", date, w.type("lineitem", "l_shipdate")),
      "shipped");
  auto ordered = w.join(lines, w.table("orders", {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"}), "l_orderkey", "o_orderkey");
  auto customers = w.join(ordered, w.table("customer", {"c_custkey", "c_mktsegment"}), "o_custkey", "c_custkey");
  auto building = w.filter(
      customers,
      compareTo(
          "building",
          "equal",
          "c_mktsegment",
          ctx.meta("customer", "c_mktsegment").code("BUILDING"),
          w.type("customer", "c_mktsegment")),
      "building");
  auto revenue = w.project(
      building,
      volume("revenue", "l_extendedprice", "l_discount", w),
      {{"o_orderkey", w.type("orders", "o_orderkey")},
       {"o_orderdate", w.type("orders", "o_orderdate")},
       {"o_shippriority", w.type("orders", "o_shippriority")},
       {"revenue", "i64"}});
  auto grouped = w.aggregate(revenue, {"o_orderkey", "o_orderdate", "o_shippriority"}, {{"sum_revenue", "sum", "revenue"}});
  Batch lineitem(ctx, "lineitem", {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"}, linesOf(ctx, begin, end));
  Batch orders(ctx, "orders", {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"}, {begin, end});
  Batch customer(ctx, "customer", {"c_custkey", "c_mktsegment"}, {0, ctx.rows("customer")});
  Answer answer;
  runPlan(ctx, w.plan(grouped), {{"tpch.lineitem", &lineitem}, {"tpch.orders", &orders}, {"tpch.customer", &customer}}, answer);
  std::vector<std::array<int64_t, 4>> rows;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    rows.push_back({*answer.value(0, row), answer.value(3, row).value_or(0), *answer.value(1, row), *answer.value(2, row)});
  }
  std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) {
    return x[1] != y[1] ? x[1] > y[1] : x[2] < y[2];
  });
  for (size_t i = 0; i < rows.size() && i < 10 && rows[i][1] > 0; ++i) {
    result.ints[0].push_back(rows[i][0]);
    result.decimals[1].push_back(rows[i][1]);
    result.ints[2].push_back(rows[i][2]);
    result.ints[3].push_back(rows[i][3]);
  }
  return result;
}

Result q5(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1994, 1, 1), days(1995, 1, 1));
  const uint32_t nations = nationsInRegion(ctx, "ASIA");
  PlanWriter w(ctx);
  const std::string nationType = w.type("supplier", "s_nationkey");
  auto ordered = w.join(
      w.table("lineitem", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"}),
      w.table("orders", {"o_orderkey", "o_custkey"}),
      "l_orderkey",
      "o_orderkey");
  auto customers = w.join(ordered, w.table("customer", {"c_custkey", "c_nationkey"}), "o_custkey", "c_custkey");
  auto suppliers = w.join(customers, w.table("supplier", {"s_suppkey", "s_nationkey"}), "l_suppkey", "s_suppkey");
  auto local = w.filter(
      suppliers,
      compare("same_nation", "equal", "c_nationkey", "s_nationkey", nationType) +
          codeTest("in_region", "s_nationkey", nationType, nationSet(nations)) +
          connect("local", "and", {"same_nation", "in_region"}),
      "local");
  auto revenue = w.project(
      local, volume("revenue", "l_extendedprice", "l_discount", w), {{"s_nationkey", nationType}, {"revenue", "i64"}});
  auto grouped = w.aggregate(revenue, {"s_nationkey"}, {{"sum_revenue", "sum", "revenue"}});
  Result result(findQuery(5)->schema);
  std::vector<int64_t> totals(32, 0);
  if (end > begin) {
    Batch lineitem(ctx, "lineitem", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"}, linesOf(ctx, begin, end));
    Batch orders(ctx, "orders", {"o_orderkey", "o_custkey"}, {begin, end});
    Batch customer(ctx, "customer", {"c_custkey", "c_nationkey"}, {0, ctx.rows("customer")});
    Batch supplier(ctx, "supplier", {"s_suppkey", "s_nationkey"}, {0, ctx.rows("supplier")});
    Answer answer;
    runPlan(
        ctx,
        w.plan(grouped),
        {{"tpch.lineitem", &lineitem}, {"tpch.orders", &orders}, {"tpch.customer", &customer}, {"tpch.supplier", &supplier}},
        answer);
    for (int64_t row = 0; row < answer.rows(); ++row) {
      totals.at(*answer.value(0, row)) = answer.value(1, row).value_or(0);
    }
  }
  for (int32_t n = 0; n < 25; ++n) {
    if ((nations >> n) & 1) {
      result.strings[0].push_back(nationName(ctx, n));
      result.decimals[1].push_back(totals[n]);
    }
  }
  return result;
}

Result q7(QueryContext& ctx) {
  const int32_t dateLow = days(1995, 1, 1);
  const int32_t dateHigh = days(1996, 12, 31);
  const auto lag = lineDateLag(ctx, "l_shipdate");
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", dateLow - lag.second, dateHigh - lag.first + 1);
  const std::string names[2] = {"FRANCE", "GERMANY"};
  CodeSet pair;
  pair.add(nationKey(ctx, names[0]));
  pair.add(nationKey(ctx, names[1]));
  PlanWriter w(ctx);
  const std::string dateType = w.type("lineitem", "l_shipdate");
  const std::string nationType = w.type("supplier", "s_nationkey");
  auto shipped = w.filter(
      w.table("lineitem", {"l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"}),
      compareTo("after", "gte", "l_shipdate", dateLow, dateType) + compareTo("before", "lte", "l_shipdate", dateHigh, dateType) +
          connect("in_years", "and", {"after", "before"}),
      "in_years");
  auto suppliers = w.join(shipped, w.table("supplier", {"s_suppkey", "s_nationkey"}), "l_suppkey", "s_suppkey");
  auto fromPair = w.filter(suppliers, codeTest("from_pair", "s_nationkey", nationType, pair), "from_pair");
  auto ordered = w.join(fromPair, w.table("orders", {"o_orderkey", "o_custkey"}), "l_orderkey", "o_orderkey");
  auto customers = w.join(ordered, w.table("customer", {"c_custkey", "c_nationkey"}), "o_custkey", "c_custkey");
  auto across = w.filter(
      customers,
      codeTest("to_pair", "c_nationkey", nationType, pair) +
          compare("other", "not_equal", "c_nationkey", "s_nationkey", nationType) + connect("across", "and", {"to_pair", "other"}),
      "across");
  auto revenue = w.project(
      across,
      compareTo("second_year", "gte", "l_shipdate", days(1996, 1, 1), dateType) + literal("year_1995", 1995, "i64") +
          literal("year_1996", 1996, "i64") + choose("year", "second_year", "year_1996", "year_1995") +
          volume("revenue", "l_extendedprice", "l_discount", w),
      {{"s_nationkey", nationType}, {"c_nationkey", nationType}, {"year", "i64"}, {"revenue", "i64"}});
  auto grouped = w.aggregate(revenue, {"s_nationkey", "c_nationkey", "year"}, {{"sum_revenue", "sum", "revenue"}});
  Result result(findQuery(7)->schema);
  if (end <= begin) {
    return result;
  }
  Batch lineitem(
      ctx, "lineitem", {"l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"}, linesOf(ctx, begin, end));
  Batch supplier(ctx, "supplier", {"s_suppkey", "s_nationkey"}, {0, ctx.rows("supplier")});
  Batch orders(ctx, "orders", {"o_orderkey", "o_custkey"}, {begin, end});
  Batch customer(ctx, "customer", {"c_custkey", "c_nationkey"}, {0, ctx.rows("customer")});
  Answer answer;
  runPlan(
      ctx,
      w.plan(grouped),
      {{"tpch.lineitem", &lineitem}, {"tpch.supplier", &supplier}, {"tpch.orders", &orders}, {"tpch.customer", &customer}},
      answer);
  std::map<std::tuple<std::string, std::string, int64_t>, int64_t> rows;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    const int64_t revenueValue = answer.value(3, row).value_or(0);
    if (revenueValue) {
      rows[{nationName(ctx, *answer.value(0, row)), nationName(ctx, *answer.value(1, row)), *answer.value(2, row)}] =
          revenueValue;
    }
  }
  for (const auto& [key, revenueValue] : rows) {
    result.strings[0].push_back(std::get<0>(key));
    result.strings[1].push_back(std::get<1>(key));
    result.ints[2].push_back(std::get<2>(key));
    result.decimals[3].push_back(revenueValue);
  }
  return result;
}

Result q8(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1995, 1, 1), days(1996, 12, 31) + 1);
  PlanWriter w(ctx);
  const std::string nationType = w.type("supplier", "s_nationkey");
  auto parts = w.join(
      w.table("lineitem", {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"}),
      w.table("part", {"p_partkey", "p_type"}),
      "l_partkey",
      "p_partkey");
  auto steel = w.filter(
      parts,
      compareTo(
          "steel", "equal", "p_type", ctx.meta("part", "p_type").code("ECONOMY ANODIZED STEEL"), w.type("part", "p_type")),
      "steel");
  auto ordered = w.join(steel, w.table("orders", {"o_orderkey", "o_custkey", "o_orderdate"}), "l_orderkey", "o_orderkey");
  auto customers = w.join(ordered, w.table("customer", {"c_custkey", "c_nationkey"}), "o_custkey", "c_custkey");
  auto american = w.filter(
      customers, codeTest("american", "c_nationkey", nationType, nationSet(nationsInRegion(ctx, "AMERICA"))), "american");
  auto suppliers = w.join(american, w.table("supplier", {"s_suppkey", "s_nationkey"}), "l_suppkey", "s_suppkey");
  auto volumes = w.project(
      suppliers,
      compareTo("second_year", "gte", "o_orderdate", days(1996, 1, 1), w.type("orders", "o_orderdate")) +
          literal("year_1995", 1995, "i64") + literal("year_1996", 1996, "i64") +
          choose("year", "second_year", "year_1996", "year_1995") + volume("volume", "l_extendedprice", "l_discount", w) +
          compareTo("brazilian", "equal", "s_nationkey", nationKey(ctx, "BRAZIL"), nationType) + literal("zero", 0, "i64") +
          choose("brazil", "brazilian", "volume", "zero"),
      {{"year", "i64"}, {"volume", "i64"}, {"brazil", "i64"}});
  auto grouped = w.aggregate(volumes, {"year"}, {{"sum_volume", "sum", "volume"}, {"sum_brazil", "sum", "brazil"}});
  Result result(findQuery(8)->schema);
  if (end <= begin) {
    return result;
  }
  Batch lineitem(
      ctx, "lineitem", {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"}, linesOf(ctx, begin, end));
  Batch part(ctx, "part", {"p_partkey", "p_type"}, {0, ctx.rows("part")});
  Batch orders(ctx, "orders", {"o_orderkey", "o_custkey", "o_orderdate"}, {begin, end});
  Batch customer(ctx, "customer", {"c_custkey", "c_nationkey"}, {0, ctx.rows("customer")});
  Batch supplier(ctx, "supplier", {"s_suppkey", "s_nationkey"}, {0, ctx.rows("supplier")});
  Answer answer;
  runPlan(
      ctx,
      w.plan(grouped),
      {{"tpch.lineitem", &lineitem},
       {"tpch.part", &part},
       {"tpch.orders", &orders},
       {"tpch.customer", &customer},
       {"tpch.supplier", &supplier}},
      answer);
  std::map<int64_t, std::pair<int64_t, int64_t>> years;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    years[*answer.value(0, row)] = {answer.value(1, row).value_or(0), answer.value(2, row).value_or(0)};
  }
  for (const auto& [year, sums] : years) {
    if (sums.first) {
      result.ints[0].push_back(year);
      result.doubles[1].push_back(static_cast<double>(sums.second) / static_cast<double>(sums.first));
    }
  }
  return result;
}

Result q10(QueryContext& ctx) {
  const auto [begin, end] = sortedRange(ctx, "orders", "o_orderdate", days(1993, 10, 1), days(1994, 1, 1));
  Result result(findQuery(10)->schema);
  if (end <= begin) {
    return result;
  }
  PlanWriter w(ctx);
  auto returned = w.filter(
      w.table("lineitem", {"l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"}),
      compareTo(
          "returned",
          "equal",
          "l_returnflag",
          ctx.meta("lineitem", "l_returnflag").code("R"),
          w.type("lineitem", "l_returnflag")),
      "returned");
  auto ordered = w.join(returned, w.table("orders", {"o_orderkey", "o_custkey"}), "l_orderkey", "o_orderkey");
  auto revenue = w.project(
      ordered,
      volume("revenue", "l_extendedprice", "l_discount", w),
      {{"o_custkey", w.type("orders", "o_custkey")}, {"revenue", "i64"}});
  auto grouped = w.aggregate(revenue, {"o_custkey"}, {{"sum_revenue", "sum", "revenue"}});
  Batch lineitem(ctx, "lineitem", {"l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"}, linesOf(ctx, begin, end));
  Batch orders(ctx, "orders", {"o_orderkey", "o_custkey"}, {begin, end});
  Answer answer;
  runPlan(ctx, w.plan(grouped), {{"tpch.lineitem", &lineitem}, {"tpch.orders", &orders}}, answer);
  std::vector<std::pair<int64_t, int64_t>> rows;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    rows.emplace_back(answer.value(1, row).value_or(0), *answer.value(0, row));
  }
  std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) {
    return x.first != y.first ? x.first > y.first : x.second < y.second;
  });
  const Column& name = ctx.meta("customer", "c_name");
  const Column& acctbal = ctx.meta("customer", "c_acctbal");
  const Column& nation = ctx.meta("customer", "c_nationkey");
  const Column& address = ctx.meta("customer", "c_address");
  const Column& phone = ctx.meta("customer", "c_phone");
  const Column& comment = ctx.meta("customer", "c_comment");
  for (size_t i = 0; i < rows.size() && i < 20 && rows[i].first > 0; ++i) {
    const int64_t row = rows[i].second - 1;
    result.ints[0].push_back(rows[i].second);
    result.strings[1].push_back(name.string(row));
    result.decimals[2].push_back(rows[i].first);
    result.decimals[3].push_back(acctbal.value(row));
    result.strings[4].push_back(nationName(ctx, nation.value(row)));
    result.strings[5].push_back(address.string(row));
    result.strings[6].push_back(phone.string(row));
    result.strings[7].push_back(comment.string(row));
  }
  return result;
}

Result q11(QueryContext& ctx) {
  PlanWriter w(ctx);
  auto supplied = w.join(
      w.table("partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"}),
      w.table("supplier", {"s_suppkey", "s_nationkey"}),
      "ps_suppkey",
      "s_suppkey");
  auto german = w.filter(
      supplied,
      compareTo("german", "equal", "s_nationkey", nationKey(ctx, "GERMANY"), w.type("supplier", "s_nationkey")),
      "german");
  auto values = w.project(
      german,
      widen("cost", "ps_supplycost", w.type("partsupp", "ps_supplycost")) +
          widen("available", "ps_availqty", w.type("partsupp", "ps_availqty")) +
          arithmetic("value", "multiply", "cost", "available"),
      {{"ps_partkey", w.type("partsupp", "ps_partkey")}, {"value", "i64"}});
  auto grouped = w.aggregate(values, {"ps_partkey"}, {{"sum_value", "sum", "value"}});
  Batch partsupp(ctx, "partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"}, {0, ctx.rows("partsupp")});
  Batch supplier(ctx, "supplier", {"s_suppkey", "s_nationkey"}, {0, ctx.rows("supplier")});
  Answer answer;
  runPlan(ctx, w.plan(grouped), {{"tpch.partsupp", &partsupp}, {"tpch.supplier", &supplier}}, answer);
  std::vector<std::pair<int64_t, int64_t>> parts;
  __int128 total = 0;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    const int64_t value = answer.value(1, row).value_or(0);
    parts.emplace_back(value, *answer.value(0, row));
    total += value;
  }
  // value > total * 0.0001 / SF with SF = suppliers / 10000.
  const int64_t suppliers = ctx.rows("supplier");
  std::vector<std::pair<int64_t, int64_t>> kept;
  for (const auto& part : parts) {
    if (static_cast<__int128>(part.first) * suppliers > total) {
      kept.push_back(part);
    }
  }
  std::sort(kept.begin(), kept.end(), [](const auto& x, const auto& y) {
    return x.first != y.first ? x.first > y.first : x.second < y.second;
  });
  Result result(findQuery(11)->schema);
  for (const auto& [value, partkey] : kept) {
    result.ints[0].push_back(partkey);
    result.decimals[1].push_back(value);
  }
  return result;
}

Result q15(QueryContext& ctx) {
  const int32_t low = days(1996, 1, 1);
  const int32_t high = days(1996, 4, 1) - 1;
  Result result(findQuery(15)->schema);
  const RowRange rows = shipdateRows(ctx, low, high);
  if (rows.end == rows.begin) {
    return result;
  }
  PlanWriter w(ctx);
  const std::string dateType = w.type("lineitem", "l_shipdate");
  auto quarter = w.filter(
      w.table("lineitem", {"l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"}),
      compareTo("after", "gte", "l_shipdate", low, dateType) + compareTo("before", "lte", "l_shipdate", high, dateType) +
          connect("in_quarter", "and", {"after", "before"}),
      "in_quarter");
  auto revenue = w.project(
      quarter,
      volume("revenue", "l_extendedprice", "l_discount", w),
      {{"l_suppkey", w.type("lineitem", "l_suppkey")}, {"revenue", "i64"}});
  auto grouped = w.aggregate(revenue, {"l_suppkey"}, {{"sum_revenue", "sum", "revenue"}});
  Batch lineitem(ctx, "lineitem", {"l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"}, rows);
  Answer answer;
  runPlan(ctx, w.plan(grouped), {{"tpch.lineitem", &lineitem}}, answer);
  int64_t best = 0;
  std::vector<std::pair<int64_t, int64_t>> suppliers;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    const int64_t value = answer.value(1, row).value_or(0);
    suppliers.emplace_back(*answer.value(0, row), value);
    best = std::max(best, value);
  }
  std::sort(suppliers.begin(), suppliers.end());
  const Column& name = ctx.meta("supplier", "s_name");
  const Column& address = ctx.meta("supplier", "s_address");
  const Column& phone = ctx.meta("supplier", "s_phone");
  for (const auto& [supplier, value] : suppliers) {
    if (best <= 0 || value != best) {
      continue;
    }
    result.ints[0].push_back(supplier);
    result.strings[1].push_back(name.string(supplier - 1));
    result.strings[2].push_back(address.string(supplier - 1));
    result.strings[3].push_back(phone.string(supplier - 1));
    result.decimals[4].push_back(value);
  }
  return result;
}

/// Grouped by (part, quantity), so that the host has each part's average
/// quantity and the price of its lines below a fifth of it from one pass.
Result q17(QueryContext& ctx) {
  PlanWriter w(ctx);
  auto parts = w.join(
      w.table("lineitem", {"l_partkey", "l_quantity", "l_extendedprice"}),
      w.table("part", {"p_partkey", "p_brand", "p_container"}),
      "l_partkey",
      "p_partkey");
  auto wanted = w.filter(
      parts,
      compareTo("brand", "equal", "p_brand", ctx.meta("part", "p_brand").code("Brand#23"), w.type("part", "p_brand")) +
          compareTo(
              "container", "equal", "p_container", ctx.meta("part", "p_container").code("MED BOX"), w.type("part", "p_container")) +
          connect("wanted", "and", {"brand", "container"}),
      "wanted");
  auto prices = w.project(
      wanted,
      widen("price", "l_extendedprice", w.type("lineitem", "l_extendedprice")),
      {{"l_partkey", w.type("lineitem", "l_partkey")}, {"l_quantity", w.type("lineitem", "l_quantity")}, {"price", "i64"}});
  auto grouped =
      w.aggregate(prices, {"l_partkey", "l_quantity"}, {{"sum_price", "sum", "price"}, {"lines", "count", "price"}});
  Batch lineitem(ctx, "lineitem", {"l_partkey", "l_quantity", "l_extendedprice"}, {0, ctx.rows("lineitem")});
  Batch part(ctx, "part", {"p_partkey", "p_brand", "p_container"}, {0, ctx.rows("part")});
  Answer answer;
  runPlan(ctx, w.plan(grouped), {{"tpch.lineitem", &lineitem}, {"tpch.part", &part}}, answer);
  std::map<int64_t, std::pair<int64_t, int64_t>> perPart;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    auto& [quantity, lines] = perPart[*answer.value(0, row)];
    const int64_t count = *answer.value(3, row);
    quantity += *answer.value(1, row) * count;
    lines += count;
  }
  int64_t total = 0;
  for (int64_t row = 0; row < answer.rows(); ++row) {
    const auto& [quantity, lines] = perPart[*answer.value(0, row)];
    // quantity < 0.2 * avg(quantity), kept in integers.
    if (5 * *answer.value(1, row) * lines < quantity) {
      total += answer.value(2, row).value_or(0);
    }
  }
  Result result(findQuery(17)->schema);
  result.doubles[0].push_back(total / 100.0 / 7.0);
  return result;
}

/// Orders in runs whose lines are sent together, so that one call's groups
/// stay a run's orders.
constexpr int64_t kOrdersPerCall = int64_t{1} << 25;

Result q18(QueryContext& ctx) {
  PlanWriter w(ctx);
  auto quantities = w.project(
      w.table("lineitem", {"l_orderkey", "l_quantity"}),
      widen("quantity", "l_quantity", w.type("lineitem", "l_quantity")),
      {{"l_orderkey", w.type("lineitem", "l_orderkey")}, {"quantity", "i64"}});
  const std::string plan = w.plan(w.aggregate(quantities, {"l_orderkey"}, {{"sum_quantity", "sum", "quantity"}}));
  const int64_t orders = ctx.rows("orders");
  const Column& orderkey = ctx.meta("orders", "o_orderkey");
  std::vector<std::pair<int64_t, int64_t>> found;
  for (int64_t begin = 0; begin < orders; begin += kOrdersPerCall) {
    const int64_t end = std::min(orders, begin + kOrdersPerCall);
    Batch lineitem(ctx, "lineitem", {"l_orderkey", "l_quantity"}, linesOf(ctx, begin, end));
    Answer answer;
    runPlan(ctx, plan, {{"tpch.lineitem", &lineitem}}, answer);
    std::map<int64_t, int64_t> large;
    for (int64_t row = 0; row < answer.rows(); ++row) {
      const int64_t sum = answer.value(1, row).value_or(0);
      if (sum > 30000) {
        large[*answer.value(0, row)] = sum;
      }
    }
    for (int64_t row = begin; row < end && !large.empty(); ++row) {
      auto it = large.find(orderkey.value(row));
      if (it != large.end()) {
        found.emplace_back(row, it->second);
        large.erase(it);
      }
    }
  }
  const Column& totalprice = ctx.meta("orders", "o_totalprice");
  const Column& orderdate = ctx.meta("orders", "o_orderdate");
  std::sort(found.begin(), found.end(), [&](const auto& x, const auto& y) {
    const int64_t px = totalprice.value(x.first);
    const int64_t py = totalprice.value(y.first);
    return px != py ? px > py : orderdate.value(x.first) < orderdate.value(y.first);
  });
  found.resize(std::min<size_t>(found.size(), 100));
  const Column& custkey = ctx.meta("orders", "o_custkey");
  const Column& name = ctx.meta("customer", "c_name");
  Result result(findQuery(18)->schema);
  for (const auto& [row, sum] : found) {
    const int64_t customer = custkey.value(row);
    result.strings[0].push_back(name.string(customer - 1));
    result.ints[1].push_back(customer);
    result.ints[2].push_back(orderkey.value(row));
    result.ints[3].push_back(orderdate.value(row));
    result.decimals[4].push_back(totalprice.value(row));
    result.decimals[5].push_back(sum);
  }
  return result;
}

/// Meld has no min or max, so the per-order test that every late line has
/// the same supplier, and that another supplier has a line, is made from
/// sums: n values are all equal exactly when n * sum(s^2) == sum(s)^2.
Result q21(QueryContext& ctx) {
  PlanWriter w(ctx);
  const std::string supplierType = w.type("lineitem", "l_suppkey");
  const std::string dateType = w.type("lineitem", "l_commitdate");
  auto ordered = w.join(
      w.table("lineitem", {"l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"}),
      w.table("orders", {"o_orderkey", "o_orderstatus"}),
      "l_orderkey",
      "o_orderkey");
  auto finished = w.filter(
      ordered,
      compareTo(
          "finished",
          "equal",
          "o_orderstatus",
          ctx.meta("orders", "o_orderstatus").code("F"),
          w.type("orders", "o_orderstatus")),
      "finished");
  auto suppliers = w.join(finished, w.table("supplier", {"s_suppkey", "s_nationkey"}), "l_suppkey", "s_suppkey");
  auto terms = w.project(
      suppliers,
      widen("s", "l_suppkey", supplierType) + arithmetic("s2", "multiply", "s", "s") +
          compare("is_late", "gt", "l_receiptdate", "l_commitdate", dateType) + literal("one", 1, "i64") +
          literal("zero", 0, "i64") + choose("late", "is_late", "one", "zero") + choose("late_s", "is_late", "s", "zero") +
          choose("late_s2", "is_late", "s2", "zero") +
          compareTo("saudi", "equal", "s_nationkey", nationKey(ctx, "SAUDI ARABIA"), w.type("supplier", "s_nationkey")) +
          connect("late_saudi", "and", {"is_late", "saudi"}) + choose("late_in_nation", "late_saudi", "one", "zero"),
      {{"l_orderkey", w.type("lineitem", "l_orderkey")},
       {"s", "i64"},
       {"s2", "i64"},
       {"late", "i64"},
       {"late_s", "i64"},
       {"late_s2", "i64"},
       {"late_in_nation", "i64"}});
  const std::string plan = w.plan(w.aggregate(
      terms,
      {"l_orderkey"},
      {{"lines", "count", "s"},
       {"sum_s", "sum", "s"},
       {"sum_s2", "sum", "s2"},
       {"sum_late", "sum", "late"},
       {"sum_late_s", "sum", "late_s"},
       {"sum_late_s2", "sum", "late_s2"},
       {"sum_late_in_nation", "sum", "late_in_nation"}}));
  const int64_t orders = ctx.rows("orders");
  Batch supplier(ctx, "supplier", {"s_suppkey", "s_nationkey"}, {0, ctx.rows("supplier")});
  std::map<int64_t, int64_t> counts;
  for (int64_t begin = 0; begin < orders; begin += kOrdersPerCall) {
    const int64_t end = std::min(orders, begin + kOrdersPerCall);
    Batch lineitem(ctx, "lineitem", {"l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"}, linesOf(ctx, begin, end));
    Batch ordersBatch(ctx, "orders", {"o_orderkey", "o_orderstatus"}, {begin, end});
    Answer answer;
    runPlan(ctx, plan, {{"tpch.lineitem", &lineitem}, {"tpch.orders", &ordersBatch}, {"tpch.supplier", &supplier}}, answer);
    for (int64_t row = 0; row < answer.rows(); ++row) {
      const int64_t late = answer.value(4, row).value_or(0);
      if (late == 0 || answer.value(7, row).value_or(0) != late) {
        continue;
      }
      const int64_t lateSum = answer.value(5, row).value_or(0);
      if (late * answer.value(6, row).value_or(0) != lateSum * lateSum) {
        continue;
      }
      const int64_t lines = *answer.value(1, row);
      const int64_t sum = answer.value(2, row).value_or(0);
      if (lines * answer.value(3, row).value_or(0) == sum * sum) {
        continue;
      }
      counts[lateSum / late] += late;
    }
  }
  const Column& name = ctx.meta("supplier", "s_name");
  std::vector<std::pair<int64_t, std::string>> rows;
  for (const auto& [supplierKey, count] : counts) {
    rows.emplace_back(count, name.string(supplierKey - 1));
  }
  std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) {
    return x.first != y.first ? x.first > y.first : x.second < y.second;
  });
  rows.resize(std::min<size_t>(rows.size(), 100));
  Result result(findQuery(21)->schema);
  for (auto& [count, supplierName] : rows) {
    result.strings[0].push_back(std::move(supplierName));
    result.ints[1].push_back(count);
  }
  return result;
}

} // namespace

bool registerMeldBackend(const std::string& library, std::string& error) {
  void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    error = dlerror();
    return false;
  }
  auto symbol = [&](const char* name) {
    void* address = dlsym(handle, name);
    if (!address) {
      throw std::runtime_error(std::string("libmeld-velox has no ") + name);
    }
    return address;
  };
  gMeld.abiVersion = reinterpret_cast<decltype(gMeld.abiVersion)>(symbol("meld_velox_abi_version"));
  if (gMeld.abiVersion() != MELD_VELOX_ABI_VERSION) {
    error = "libmeld-velox has ABI version " + std::to_string(gMeld.abiVersion()) + ", expected " +
        std::to_string(MELD_VELOX_ABI_VERSION);
    return false;
  }
  gMeld.run = reinterpret_cast<decltype(gMeld.run)>(symbol("meld_velox_run"));
  gMeld.runTables = reinterpret_cast<decltype(gMeld.runTables)>(symbol("meld_velox_run_tables"));
  gMeld.adoptDevice = reinterpret_cast<decltype(gMeld.adoptDevice)>(symbol("meld_velox_adopt_device"));
  gMeld.free = reinterpret_cast<decltype(gMeld.free)>(symbol("meld_velox_free"));
  // Q9, Q13, Q16, Q20 and Q22 match strings by LIKE or substring, which Meld
  // has only for equality against a dictionary.
  for (const auto& [query, function] : std::vector<std::pair<int32_t, QueryFunction>>{
           {1, q1}, {2, q2}, {3, q3}, {4, q4}, {5, q5}, {6, q6}, {7, q7}, {8, q8}, {10, q10}, {11, q11},
           {12, q12}, {14, q14}, {15, q15}, {17, q17}, {18, q18}, {19, q19}, {21, q21}}) {
    registerImplementation(Backend::kMeld, query, function);
  }
  return true;
}

} // namespace facebook::velox::supervelox::meld_backend
