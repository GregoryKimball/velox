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

#include "meldvelox/MeldVelox.h"
#include "velox/experimental/supervelox/queries/Query.h"

namespace facebook::velox::supervelox::meld_backend {
namespace {

struct Library {
  decltype(&meld_velox_abi_version) abiVersion;
  decltype(&meld_velox_run) run;
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
  gMeld.adoptDevice = reinterpret_cast<decltype(gMeld.adoptDevice)>(symbol("meld_velox_adopt_device"));
  gMeld.free = reinterpret_cast<decltype(gMeld.free)>(symbol("meld_velox_free"));
  registerImplementation(Backend::kMeld, 1, q1);
  registerImplementation(Backend::kMeld, 6, q6);
  return true;
}

} // namespace facebook::velox::supervelox::meld_backend
