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

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
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
  registerImplementation(Backend::kMeld, 1, q1);
  registerImplementation(Backend::kMeld, 4, q4);
  registerImplementation(Backend::kMeld, 6, q6);
  registerImplementation(Backend::kMeld, 12, q12);
  registerImplementation(Backend::kMeld, 14, q14);
  registerImplementation(Backend::kMeld, 19, q19);
  return true;
}

} // namespace facebook::velox::supervelox::meld_backend
