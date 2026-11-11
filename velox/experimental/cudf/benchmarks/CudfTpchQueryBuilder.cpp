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

#include "velox/experimental/cudf/benchmarks/CudfTpchQueryBuilder.h"

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/tpch/gen/TpchGen.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace facebook::velox::cudf_velox {

using velox::exec::test::PlanBuilder;
using velox::exec::test::TpchPlan;

namespace {

constexpr const char* kLineitem = "lineitem";
constexpr const char* kOrders = "orders";
constexpr const char* kCustomer = "customer";
constexpr const char* kPart = "part";
constexpr const char* kPartsupp = "partsupp";
constexpr const char* kSupplier = "supplier";
constexpr const char* kNation = "nation";
constexpr const char* kRegion = "region";

/// Parses literals such as 1.0 as DECIMAL, so that they combine with DECIMAL
/// columns as in SQL.
const parse::ParseOptions& decimalParseOptions() {
  static const parse::ParseOptions options = [] {
    parse::ParseOptions result;
    result.parseDecimalAsDouble = false;
    return result;
  }();
  return options;
}

/// The standard column names of each table, in TPC-H order.
const std::vector<std::pair<std::string, tpch::Table>>& tables() {
  static const std::vector<std::pair<std::string, tpch::Table>> kTables = {
      {kLineitem, tpch::Table::TBL_LINEITEM},
      {kOrders, tpch::Table::TBL_ORDERS},
      {kCustomer, tpch::Table::TBL_CUSTOMER},
      {kPart, tpch::Table::TBL_PART},
      {kPartsupp, tpch::Table::TBL_PARTSUPP},
      {kSupplier, tpch::Table::TBL_SUPPLIER},
      {kNation, tpch::Table::TBL_NATION},
      {kRegion, tpch::Table::TBL_REGION},
  };
  return kTables;
}

} // namespace

CudfTpchQueryBuilder::CudfTpchQueryBuilder(
    dwio::common::FileFormat format,
    double scaleFactor)
    : format_(format),
      q11Fraction_(q11Fraction(scaleFactor)),
      pool_(memory::memoryManager()->addLeafPool()) {}

std::string CudfTpchQueryBuilder::q11Fraction(double scaleFactor) {
  VELOX_USER_CHECK(
      std::isfinite(scaleFactor) && scaleFactor > 0,
      "TPC-H scale factor must be positive: {}",
      scaleFactor);
  const auto fraction = fmt::format("{:.10f}", 0.0001 / scaleFactor);
  VELOX_USER_CHECK_NE(
      fraction,
      "0.0000000000",
      "TPC-H scale factor {} makes Q11's FRACTION round to zero",
      scaleFactor);
  return fraction;
}

void CudfTpchQueryBuilder::readFileSchema(
    const std::string& table,
    const std::string& filePath,
    const std::vector<std::string>& columns) {
  dwio::common::ReaderOptions readerOptions(pool_.get());
  readerOptions.setFileFormat(format_);
  std::shared_ptr<ReadFile> readFile =
      filesystems::getFileSystem(filePath, nullptr)->openFileForRead(filePath);
  auto input = std::make_unique<dwio::common::BufferedInput>(
      readFile, readerOptions.memoryPool());
  auto reader = dwio::common::getReaderFactory(readerOptions.fileFormat())
                    ->createReader(std::move(input), readerOptions);
  const auto& fileType = reader->rowType();
  // There can be extra columns at the end of the file.
  VELOX_CHECK_GE(fileType->size(), columns.size());
  auto& metadata = tableMetadata_[table];
  std::vector<TypePtr> types;
  for (size_t i = 0; i < columns.size(); ++i) {
    metadata.fileColumnNames[columns[i]] = fileType->nameOf(i);
    types.push_back(fileType->childAt(i));
  }
  metadata.type = ROW(std::vector<std::string>(columns), std::move(types));
}

void CudfTpchQueryBuilder::initialize(const std::string& dataPath) {
  for (const auto& [table, tpchTable] : tables()) {
    const auto columns = tpch::getTableSchema(tpchTable)->names();
    const fs::path tablePath{dataPath + "/" + table};
    auto& files = tableMetadata_[table].dataFiles;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator{
             tablePath, std::filesystem::directory_options(), error}) {
      // Skip directories and hidden files.
      if (!entry.is_regular_file() ||
          entry.path().filename().c_str()[0] == '.') {
        continue;
      }
      if (files.empty()) {
        readFileSchema(table, entry.path().string(), columns);
      }
      files.push_back(entry.path().string());
    }
    if (files.empty() && error) {
      // 'tablePath' is a file listing the table's data files.
      std::ifstream list(tablePath);
      for (std::string line; std::getline(list, line);) {
        if (files.empty()) {
          readFileSchema(table, line, columns);
        }
        files.push_back(line);
      }
    }
    VELOX_USER_CHECK(
        !files.empty(), "No data files for table {} in {}", table, dataPath);
  }
}

RowTypePtr CudfTpchQueryBuilder::rowType(
    const std::string& table,
    const std::vector<std::string>& columns) const {
  const auto& type = tableMetadata_.at(table).type;
  std::vector<TypePtr> types;
  types.reserve(columns.size());
  for (const auto& column : columns) {
    types.push_back(type->findChild(column));
  }
  return ROW(std::vector<std::string>(columns), std::move(types));
}

TpchPlan CudfTpchQueryBuilder::newPlan() const {
  TpchPlan plan;
  plan.dataFileFormat = format_;
  return plan;
}

PlanBuilder CudfTpchQueryBuilder::scan(
    const std::shared_ptr<core::PlanNodeIdGenerator>& ids,
    TpchPlan& plan,
    const Scan& spec,
    const parse::ParseOptions& options) const {
  auto readColumns = spec.columns;
  for (const auto& column : spec.filterOnly) {
    VELOX_CHECK(
        std::find(readColumns.begin(), readColumns.end(), column) ==
            readColumns.end(),
        "Filter-only column {} is also returned",
        column);
    readColumns.push_back(column);
  }
  const auto& metadata = tableMetadata_.at(spec.table);
  core::PlanNodeId scanId;
  PlanBuilder builder(ids, pool_.get());
  // No data columns: the Hive connector would take them for the whole table
  // schema and map them to the file's columns by position.
  builder.setParseOptions(options)
      .tableScan(
          spec.table,
          rowType(spec.table, readColumns),
          metadata.fileColumnNames,
          spec.filters,
          spec.remainingFilter)
      .captureScanNodeId(scanId);
  if (!spec.filterOnly.empty()) {
    // The scan returns the columns its filters read; drop those the rest of
    // the plan does not.
    builder.project(spec.columns);
  }
  plan.dataFiles[scanId] = metadata.dataFiles;
  return builder;
}

PlanBuilder CudfTpchQueryBuilder::scan(
    const std::shared_ptr<core::PlanNodeIdGenerator>& ids,
    TpchPlan& plan,
    const Scan& spec) const {
  return scan(ids, plan, spec, decimalParseOptions());
}

std::string CudfTpchQueryBuilder::dateFilter(
    const std::string& table,
    const std::string& column,
    const std::string& lower,
    const std::string& upper) const {
  // DWRF TPC-H files store dates as VARCHAR.
  const auto suffix =
      tableMetadata_.at(table).type->findChild(column)->isVarchar() ? ""
                                                                    : "::DATE";
  if (!lower.empty() && !upper.empty()) {
    return fmt::format(
        "{} between {}{} and {}{}", column, lower, suffix, upper, suffix);
  }
  if (!lower.empty()) {
    return fmt::format("{} > {}{}", column, lower, suffix);
  }
  VELOX_CHECK(!upper.empty(), "A date filter needs a bound");
  return fmt::format("{} < {}{}", column, upper, suffix);
}

TpchPlan CudfTpchQueryBuilder::getQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 2:
      return getQ2Plan();
    case 3:
      return getQ3Plan();
    case 4:
      return getQ4Plan();
    case 5:
      return getQ5Plan();
    case 6:
      return getQ6Plan();
    case 7:
      return getQ7Plan();
    case 8:
      return getQ8Plan();
    case 9:
      return getQ9Plan();
    case 10:
      return getQ10Plan();
    case 11:
      return getQ11Plan();
    case 12:
      return getQ12Plan();
    case 13:
      return getQ13Plan();
    case 14:
      return getQ14Plan();
    case 15:
      return getQ15Plan();
    case 16:
      return getQ16Plan();
    case 17:
      return getQ17Plan();
    case 18:
      return getQ18Plan();
    case 19:
      return getQ19Plan();
    case 20:
      return getQ20Plan();
    case 21:
      return getQ21Plan();
    case 22:
      return getQ22Plan();
    default:
      VELOX_USER_FAIL("TPC-H query {} does not exist", queryId);
  }
}

TpchPlan CudfTpchQueryBuilder::getQ1Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_returnflag",
                "l_linestatus",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax"},
           .filterOnly = {"l_shipdate"},
           // l_shipdate <= '1995-08-19'
           .filters = {dateFilter(
               kLineitem, "l_shipdate", "", "'1995-08-20'")}})
          .project(
              {"l_returnflag",
               "l_linestatus",
               "l_quantity",
               "l_extendedprice",
               "l_extendedprice * (1.0 - l_discount) AS l_sum_disc_price",
               "l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS l_sum_charge",
               "l_discount"})
          .project(
              {"l_returnflag",
               "l_linestatus",
               "l_quantity",
               "l_extendedprice",
               "l_sum_disc_price",
               "l_sum_charge",
               "l_discount",
               "cast(l_quantity as DOUBLE) as qty_double",
               "cast(l_extendedprice as DOUBLE) as price_double",
               "cast(l_discount as DOUBLE) as discount_double"})
          .partialAggregation(
              {"l_returnflag", "l_linestatus"},
              {"sum(l_quantity)",
               "sum(l_extendedprice)",
               "sum(l_sum_disc_price)",
               "sum(l_sum_charge)",
               "avg(qty_double)",
               "avg(price_double)",
               "avg(discount_double)",
               "count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_returnflag", "l_linestatus"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ2Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const std::string europe = "r_name = 'EUROPE'";

  // Suppliers of the region, for the minimum supply cost of each part.
  auto subqueryRegion = scan(
                            ids,
                            plan,
                            {.table = kRegion,
                             .columns = {"r_regionkey"},
                             .filterOnly = {"r_name"},
                             .filters = {europe}})
                            .planNode();
  auto subqueryNations =
      scan(
          ids,
          plan,
          {.table = kNation, .columns = {"n_nationkey", "n_regionkey"}})
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              subqueryRegion,
              "",
              {"n_nationkey"})
          .planNode();
  auto subquerySuppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier, .columns = {"s_suppkey", "s_nationkey"}})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              subqueryNations,
              "",
              {"s_suppkey"})
          .planNode();

  auto parts = scan(
                   ids,
                   plan,
                   {.table = kPart,
                    .columns = {"p_partkey", "p_mfgr", "p_size"},
                    .filterOnly = {"p_type"},
                    .remainingFilter = "p_type like '%NICKEL'"})
                   .filter("p_size = 41")
                   .planNode();
  auto region = scan(
                    ids,
                    plan,
                    {.table = kRegion,
                     .columns = {"r_regionkey"},
                     .filterOnly = {"r_name"},
                     .filters = {europe}})
                    .planNode();
  auto nations = scan(
                     ids,
                     plan,
                     {.table = kNation,
                      .columns = {"n_nationkey", "n_name", "n_regionkey"}})
                     .hashJoin(
                         {"n_regionkey"},
                         {"r_regionkey"},
                         region,
                         "",
                         {"n_nationkey", "n_name"})
                     .planNode();
  auto suppliers = scan(
                       ids,
                       plan,
                       {.table = kSupplier,
                        .columns =
                            {"s_acctbal",
                             "s_name",
                             "s_address",
                             "s_phone",
                             "s_comment",
                             "s_suppkey",
                             "s_nationkey"}})
                       .hashJoin(
                           {"s_nationkey"},
                           {"n_nationkey"},
                           nations,
                           "",
                           {"s_acctbal",
                            "s_name",
                            "s_address",
                            "s_phone",
                            "s_comment",
                            "s_suppkey",
                            "n_name"})
                       .planNode();
  auto partSupply =
      scan(
          ids,
          plan,
          {.table = kPartsupp,
           .columns = {"ps_partkey", "ps_suppkey", "ps_supplycost"}})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"ps_suppkey", "ps_supplycost", "p_partkey", "p_mfgr"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"s_acctbal",
               "s_name",
               "s_address",
               "s_phone",
               "s_comment",
               "ps_supplycost",
               "p_partkey",
               "p_mfgr",
               "n_name"})
          .planNode();

  plan.plan =
      scan(
          ids,
          plan,
          {.table = kPartsupp,
           .columns = {"ps_partkey", "ps_suppkey", "ps_supplycost"}})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              subquerySuppliers,
              "",
              {"ps_supplycost", "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"min(ps_supplycost) AS min_supplycost"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              partSupply,
              "ps_supplycost = min_supplycost",
              {"s_acctbal",
               "s_name",
               "s_address",
               "s_phone",
               "s_comment",
               "p_partkey",
               "p_mfgr",
               "n_name"})
          .orderBy({"s_acctbal DESC", "n_name", "s_name", "p_partkey"}, false)
          .project(
              {"s_acctbal",
               "s_name",
               "n_name",
               "p_partkey",
               "p_mfgr",
               "s_address",
               "s_phone",
               "s_comment"})
          .limit(0, 100, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ3Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto customers = scan(
                       ids,
                       plan,
                       {.table = kCustomer,
                        .columns = {"c_custkey"},
                        .filterOnly = {"c_mktsegment"},
                        .filters = {"c_mktsegment = 'HOUSEHOLD'"}})
                       .planNode();
  auto orders =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns =
               {"o_orderdate", "o_shippriority", "o_custkey", "o_orderkey"},
           .filters = {dateFilter(kOrders, "o_orderdate", "", "'1995-03-25'")}})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"o_orderdate", "o_shippriority", "o_orderkey"})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_orderkey", "l_extendedprice", "l_discount"},
           .filterOnly = {"l_shipdate"},
           .filters = {dateFilter(
               kLineitem, "l_shipdate", "'1995-03-25'", "")}})
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_orderkey", "o_orderdate", "o_shippriority", "part_revenue"})
          .partialAggregation(
              {"l_orderkey", "o_orderdate", "o_shippriority"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
          .orderBy({"revenue DESC", "o_orderdate"}, false)
          .limit(0, 10, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ4Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto orders =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns = {"o_orderkey", "o_orderpriority"},
           .filterOnly = {"o_orderdate"},
           .filters = {dateFilter(
               kOrders, "o_orderdate", "'1996-10-01'", "'1996-12-31'")}})
          .planNode();
  // EXISTS as an inner join to the selected orders followed by distinct
  // (order key, priority) pairs, which counts each order once given a unique
  // o_orderkey.
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_orderkey"},
           .filterOnly = {"l_commitdate", "l_receiptdate"},
           .remainingFilter = "l_commitdate < l_receiptdate"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_orderkey", "o_orderpriority"})
          .partialAggregation({"l_orderkey", "o_orderpriority"}, {})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .partialAggregation({"o_orderpriority"}, {"count(0) as order_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"o_orderpriority"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ5Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const std::string europe = "r_name = 'EUROPE'";
  auto region = scan(
                    ids,
                    plan,
                    {.table = kRegion,
                     .columns = {"r_regionkey"},
                     .filterOnly = {"r_name"},
                     .filters = {europe}})
                    .planNode();
  auto orders =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns = {"o_orderkey", "o_custkey"},
           .filterOnly = {"o_orderdate"},
           .filters = {dateFilter(
               kOrders, "o_orderdate", "'1997-01-01'", "'1997-12-31'")}})
          .planNode();
  auto customers =
      scan(
          ids,
          plan,
          {.table = kCustomer, .columns = {"c_custkey", "c_nationkey"}})
          .planNode();
  auto nations = scan(
                     ids,
                     plan,
                     {.table = kNation,
                      .columns = {"n_nationkey", "n_name", "n_regionkey"}})
                     .hashJoin(
                         {"n_regionkey"},
                         {"r_regionkey"},
                         region,
                         "",
                         {"n_nationkey", "n_name"})
                     .planNode();
  // The region restriction as a semi join: duplicate nation rows multiply
  // the revenue when the names are joined after the aggregation below.
  auto suppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier, .columns = {"s_nationkey", "s_suppkey"}})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nations,
              "",
              {"s_suppkey", "s_nationkey"},
              core::JoinType::kLeftSemiFilter)
          .planNode();
  auto nameRegion = scan(
                        ids,
                        plan,
                        {.table = kRegion,
                         .columns = {"r_regionkey"},
                         .filterOnly = {"r_name"},
                         .filters = {europe}})
                        .planNode();
  auto names = scan(
                   ids,
                   plan,
                   {.table = kNation,
                    .columns = {"n_nationkey", "n_name", "n_regionkey"}})
                   .hashJoin(
                       {"n_regionkey"},
                       {"r_regionkey"},
                       nameRegion,
                       "",
                       {"n_nationkey", "n_name"})
                   .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"}})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"l_extendedprice", "l_discount", "s_nationkey", "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_extendedprice", "l_discount", "s_nationkey", "o_custkey"})
          .hashJoin(
              {"s_nationkey", "o_custkey"},
              {"c_nationkey", "c_custkey"},
              customers,
              "",
              {"s_nationkey", "l_extendedprice", "l_discount"})
          .project(
              {"s_nationkey",
               "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
          .partialAggregation({"s_nationkey"}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              names,
              "",
              {"n_name", "revenue"})
          .partialAggregation({"n_name"}, {"sum(revenue) as revenue_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"n_name", "revenue_sum as revenue"})
          .orderBy({"revenue DESC"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ6Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_extendedprice", "l_discount"},
           .filterOnly = {"l_shipdate", "l_quantity"},
           .filters =
               {dateFilter(
                    kLineitem, "l_shipdate", "'1997-01-01'", "'1997-12-31'"),
                "l_discount between cast(0.02 as DECIMAL(15, 2)) and cast(0.04 as DECIMAL(15, 2))",
                "l_quantity < cast(24 as DECIMAL(15, 2))"}})
          .project({"l_extendedprice * l_discount"})
          .partialAggregation({}, {"sum(p0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ7Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const std::string nationFilter = "n_name IN ('EGYPT', 'UNITED STATES')";
  auto customerNations = scan(
                             ids,
                             plan,
                             {.table = kNation,
                              .columns = {"n_nationkey", "n_name"},
                              .filters = {nationFilter}})
                             .planNode();
  auto customers =
      scan(
          ids,
          plan,
          {.table = kCustomer, .columns = {"c_custkey", "c_nationkey"}})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              customerNations,
              "",
              {"n_name", "c_custkey"})
          .project({"n_name as cust_nation", "c_custkey"})
          .planNode();
  auto orders =
      scan(
          ids, plan, {.table = kOrders, .columns = {"o_custkey", "o_orderkey"}})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"cust_nation", "o_orderkey"})
          .planNode();
  auto supplierNations = scan(
                             ids,
                             plan,
                             {.table = kNation,
                              .columns = {"n_nationkey", "n_name"},
                              .filters = {nationFilter}})
                             .planNode();
  auto suppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier, .columns = {"s_nationkey", "s_suppkey"}})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              supplierNations,
              "",
              {"n_name", "s_suppkey"})
          .project({"n_name as supp_nation", "s_suppkey"})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_shipdate",
                "l_suppkey",
                "l_orderkey",
                "l_discount",
                "l_extendedprice"},
           .filters = {dateFilter(
               kLineitem, "l_shipdate", "'1995-01-01'", "'1996-12-31'")}})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"supp_nation",
               "l_extendedprice",
               "l_discount",
               "l_shipdate",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "(((cust_nation = 'EGYPT') AND (supp_nation = 'UNITED STATES')) OR "
              "((cust_nation = 'UNITED STATES') AND (supp_nation = 'EGYPT')))",
              {"supp_nation",
               "cust_nation",
               "l_extendedprice",
               "l_discount",
               "l_shipdate"})
          .project(
              {"cust_nation",
               "supp_nation",
               "l_extendedprice * (1.0 - l_discount) as part_revenue",
               "year(l_shipdate) as l_year"})
          .partialAggregation(
              {"supp_nation", "cust_nation", "l_year"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supp_nation", "cust_nation", "l_year"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ8Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto supplierNations =
      scan(
          ids,
          plan,
          {.table = kNation,
           .columns = {"n_name", "n_nationkey", "n_regionkey"}})
          .planNode();
  auto region = scan(
                    ids,
                    plan,
                    {.table = kRegion,
                     .columns = {"r_regionkey"},
                     .filterOnly = {"r_name"},
                     .filters = {"r_name = 'MIDDLE EAST'"}})
                    .planNode();
  auto parts = scan(
                   ids,
                   plan,
                   {.table = kPart,
                    .columns = {"p_partkey"},
                    .filterOnly = {"p_type"},
                    .filters = {"p_type = 'PROMO BRUSHED COPPER'"}})
                   .planNode();
  auto customerNations =
      scan(
          ids,
          plan,
          {.table = kNation, .columns = {"n_nationkey", "n_regionkey"}})
          .hashJoin(
              {"n_regionkey"}, {"r_regionkey"}, region, "", {"n_nationkey"})
          .planNode();
  auto customers =
      scan(
          ids,
          plan,
          {.table = kCustomer, .columns = {"c_nationkey", "c_custkey"}})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              customerNations,
              "",
              {"c_custkey"})
          .planNode();
  auto orders =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns = {"o_orderdate", "o_orderkey", "o_custkey"},
           .filters = {dateFilter(
               kOrders, "o_orderdate", "'1995-01-01'", "'1996-12-31'")}})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"o_orderkey", "o_orderdate"})
          .planNode();
  auto suppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier, .columns = {"s_suppkey", "s_nationkey"}})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              supplierNations,
              "",
              {"s_suppkey", "n_name"})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_suppkey",
                "l_orderkey",
                "l_partkey",
                "l_extendedprice",
                "l_discount"}})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_partkey",
               "l_suppkey",
               "o_orderdate",
               "l_extendedprice",
               "l_discount"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"n_name",
               "o_orderdate",
               "l_partkey",
               "l_extendedprice",
               "l_discount"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"n_name", "o_orderdate", "l_extendedprice", "l_discount"})
          .project(
              {"l_extendedprice * (1.0 - l_discount) as volume",
               "n_name",
               "o_orderdate"})
          .project(
              {"volume",
               "(CASE WHEN n_name = 'EGYPT' THEN volume ELSE cast(0 as DECIMAL(31, 4)) END) as nation_volume",
               "year(o_orderdate) AS o_year"})
          .partialAggregation(
              {"o_year"},
              {"sum(nation_volume) as volume_nation",
               "sum(volume) as volume_all"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"o_year"}, false)
          .project(
              {"o_year",
               "(cast(volume_nation as DOUBLE) / cast(volume_all as DOUBLE)) as mkt_share"})
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ9Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto parts = scan(
                   ids,
                   plan,
                   {.table = kPart,
                    .columns = {"p_partkey"},
                    .filterOnly = {"p_name"},
                    .remainingFilter = "p_name like '%yellow%'"})
                   .planNode();
  auto suppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier, .columns = {"s_suppkey", "s_nationkey"}})
          .planNode();
  // The partsupp rows of the selected parts, which lineitem then probes.
  auto partSupply =
      scan(
          ids,
          plan,
          {.table = kPartsupp,
           .columns = {"ps_partkey", "ps_suppkey", "ps_supplycost"}})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"ps_partkey", "ps_suppkey", "ps_supplycost"})
          .planNode();
  auto lines = scan(
                   ids,
                   plan,
                   {.table = kLineitem,
                    .columns =
                        {"l_orderkey",
                         "l_partkey",
                         "l_suppkey",
                         "l_extendedprice",
                         "l_discount",
                         "l_quantity"}})
                   .hashJoin(
                       {"l_partkey", "l_suppkey"},
                       {"ps_partkey", "ps_suppkey"},
                       partSupply,
                       "",
                       {"l_orderkey",
                        "l_suppkey",
                        "l_extendedprice",
                        "l_discount",
                        "l_quantity",
                        "ps_supplycost"})
                   .hashJoin(
                       {"l_suppkey"},
                       {"s_suppkey"},
                       suppliers,
                       "",
                       {"l_orderkey",
                        "s_nationkey",
                        "l_extendedprice",
                        "l_discount",
                        "l_quantity",
                        "ps_supplycost"})
                   .planNode();
  // Profit per nation key and year; the names are joined afterwards.
  auto profit =
      scan(
          ids,
          plan,
          {.table = kOrders, .columns = {"o_orderkey", "o_orderdate"}})
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              lines,
              "",
              {"s_nationkey",
               "o_orderdate",
               "l_extendedprice",
               "l_discount",
               "l_quantity",
               "ps_supplycost"})
          .project(
              {"s_nationkey",
               "year(o_orderdate) as o_year",
               "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity as amount"})
          .partialAggregation(
              {"s_nationkey", "o_year"}, {"sum(amount) as nation_profit"})
          .localPartition({"s_nationkey", "o_year"})
          .finalAggregation()
          .planNode();
  plan.plan =
      scan(ids, plan, {.table = kNation, .columns = {"n_nationkey", "n_name"}})
          .hashJoin(
              {"n_nationkey"},
              {"s_nationkey"},
              profit,
              "",
              {"n_name", "o_year", "nation_profit"})
          .partialAggregation(
              {"n_name", "o_year"}, {"sum(nation_profit) as sum_profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"n_name as nation", "o_year", "sum_profit"})
          .orderBy({"nation", "o_year DESC"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ10Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto orders =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns = {"o_orderkey", "o_custkey"},
           .filterOnly = {"o_orderdate"},
           .filters = {dateFilter(
               kOrders, "o_orderdate", "'1994-03-01'", "'1994-05-31'")}})
          .planNode();
  // Revenue per order customer key, before any customer or nation column is
  // joined. The final aggregation below groups by every output column of the
  // SQL and sums again, so a duplicate customer or nation row contributes as
  // in the original joins.
  auto revenueByCustomer =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_orderkey", "l_extendedprice", "l_discount"},
           .filterOnly = {"l_returnflag"},
           .filters = {"l_returnflag = 'R'"}})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"o_custkey", "l_extendedprice", "l_discount"})
          .project(
              {"o_custkey",
               "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
          .partialAggregation(
              {"o_custkey"}, {"sum(part_revenue) AS customer_revenue"})
          .localPartition({"o_custkey"})
          .finalAggregation()
          .planNode();
  auto nations =
      scan(ids, plan, {.table = kNation, .columns = {"n_nationkey", "n_name"}})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kCustomer,
           .columns =
               {"c_nationkey",
                "c_custkey",
                "c_acctbal",
                "c_name",
                "c_address",
                "c_phone",
                "c_comment"}})
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              revenueByCustomer,
              "",
              {"c_custkey",
               "c_name",
               "c_acctbal",
               "c_address",
               "c_phone",
               "c_comment",
               "c_nationkey",
               "customer_revenue"})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nations,
              "",
              {"c_custkey",
               "c_name",
               "c_acctbal",
               "n_name",
               "c_address",
               "c_phone",
               "c_comment",
               "customer_revenue"})
          .partialAggregation(
              {"c_custkey",
               "c_name",
               "c_acctbal",
               "n_name",
               "c_address",
               "c_phone",
               "c_comment"},
              {"sum(customer_revenue) AS revenue"})
          // c_custkey is a grouping key, so each group is in one partition.
          .localPartition({"c_custkey"})
          .finalAggregation()
          .topN({"revenue DESC"}, 20, true)
          .localPartition(std::vector<std::string>{})
          .topN({"revenue DESC"}, 20, false)
          .project(
              {"c_custkey",
               "c_name",
               "revenue",
               "c_acctbal",
               "n_name",
               "c_address",
               "c_phone",
               "c_comment"})
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ11Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const std::string germany = "n_name = 'GERMANY'";
  auto germanSuppliers = [&]() {
    auto nations = scan(
                       ids,
                       plan,
                       {.table = kNation,
                        .columns = {"n_nationkey"},
                        .filterOnly = {"n_name"},
                        .filters = {germany}})
                       .planNode();
    return scan(
               ids,
               plan,
               {.table = kSupplier, .columns = {"s_suppkey", "s_nationkey"}})
        .hashJoin({"s_nationkey"}, {"n_nationkey"}, nations, "", {"s_suppkey"})
        .planNode();
  };
  // The threshold keeps the exact type of the product, whose scale is
  // sum_cost_qty's plus FRACTION's ten decimal places (DECIMAL(38, 12) for a
  // DECIMAL(15, 2) ps_supplycost), and each part's value is compared at that
  // scale. Rounding the threshold to fewer places would drop parts just
  // above it.
  auto threshold =
      scan(
          ids,
          plan,
          {.table = kPartsupp,
           .columns = {"ps_availqty", "ps_suppkey", "ps_supplycost"}})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              germanSuppliers(),
              "",
              {"ps_availqty", "ps_suppkey", "ps_supplycost"})
          .project(
              {"ps_supplycost * cast(ps_availqty as DECIMAL(10, 0)) AS product_cost_qty"})
          .partialAggregation({}, {"sum(product_cost_qty) AS sum_cost_qty"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({fmt::format(
              "sum_cost_qty * {} AS filter_cost_qty", q11Fraction_)})
          .planNode();
  const auto& thresholdType = threshold->outputType()->childAt(0);
  VELOX_CHECK(
      thresholdType->isDecimal(),
      "Q11's threshold must be DECIMAL: {}",
      thresholdType->toString());
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kPartsupp,
           .columns =
               {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"}})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              germanSuppliers(),
              "",
              {"ps_partkey", "ps_availqty", "ps_supplycost"})
          .project(
              {"ps_supplycost * cast(ps_availqty as DECIMAL(10, 0)) AS product_cost_qty",
               "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"sum(product_cost_qty) AS value"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .nestedLoopJoin(threshold, {"ps_partkey", "value", "filter_cost_qty"})
          .filter(
              fmt::format(
                  "cast(value as {}) > filter_cost_qty",
                  thresholdType->toString()))
          .project({"ps_partkey", "value"})
          .orderBy({"value DESC"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ12Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  // l_shipdate < l_commitdate < l_receiptdate < '1995-01-01' implies the
  // filters on l_shipdate and l_commitdate, which skip data early.
  auto lineitem =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_orderkey", "l_shipmode", "l_shipdate", "l_commitdate"},
           .filterOnly = {"l_receiptdate"},
           .filters =
               {dateFilter(
                    kLineitem, "l_receiptdate", "'1994-01-01'", "'1994-12-31'"),
                "l_shipmode IN ('TRUCK', 'REG AIR')",
                dateFilter(kLineitem, "l_shipdate", "", "'1995-01-01'"),
                dateFilter(kLineitem, "l_commitdate", "", "'1995-01-01'")},
           .remainingFilter = "l_commitdate < l_receiptdate"})
          .filter("l_shipdate < l_commitdate")
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kOrders, .columns = {"o_orderkey", "o_orderpriority"}})
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              lineitem,
              "",
              {"l_shipmode", "o_orderpriority"})
          .project(
              {"l_shipmode",
               "(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count_partial",
               "(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count_partial"})
          .partialAggregation(
              {"l_shipmode"},
              {"sum(high_line_count_partial) as high_line_count",
               "sum(low_line_count_partial) as low_line_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_shipmode"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ13Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  // The matching orders per customer key, counted before the outer join:
  // every customer then contributes one count, zero if unmatched, given a
  // unique c_custkey. o_comment is read for the filter only.
  auto counts =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns = {"o_custkey", "o_orderkey"},
           .filterOnly = {"o_comment"},
           .remainingFilter = "o_comment not like '%special%requests%'"})
          .partialAggregation(
              {"o_custkey"}, {"count(o_orderkey) as order_count"})
          .localPartition({"o_custkey"})
          .finalAggregation()
          .planNode();
  plan.plan =
      scan(ids, plan, {.table = kCustomer, .columns = {"c_custkey"}})
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              counts,
              "",
              {"c_custkey", "order_count"},
              core::JoinType::kLeft)
          .project({"coalesce(order_count, cast(0 as BIGINT)) as c_count"})
          .partialAggregation({"c_count"}, {"count(0) as custdist"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"custdist DESC", "c_count DESC"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ14Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto parts =
      scan(ids, plan, {.table = kPart, .columns = {"p_partkey", "p_type"}})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_partkey", "l_extendedprice", "l_discount"},
           .filterOnly = {"l_shipdate"},
           .remainingFilter = dateFilter(
               kLineitem, "l_shipdate", "'1994-08-01'", "'1994-08-31'")})
          .project(
              {"l_extendedprice * (1.0 - l_discount) as part_revenue",
               "l_partkey"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"part_revenue", "p_type"})
          .project(
              {"(CASE WHEN (p_type LIKE 'PROMO%') THEN part_revenue ELSE cast(0 as DECIMAL(31, 4)) END) as filter_revenue",
               "part_revenue"})
          .partialAggregation(
              {},
              {"sum(part_revenue) as total_revenue",
               "sum(filter_revenue) as total_promo_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"cast(100.00 * total_promo_revenue as DOUBLE)/cast(total_revenue as DOUBLE) as promo_revenue"})
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ15Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  // Revenue per supplier, kept where it equals the maximum over all
  // suppliers, which a window over the gathered revenues provides.
  auto topSuppliers =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_suppkey", "l_extendedprice", "l_discount"},
           .filterOnly = {"l_shipdate"},
           .filters = {dateFilter(
               kLineitem, "l_shipdate", "'1993-05-01'", "'1993-07-31'")}})
          .project(
              {"l_suppkey", "l_extendedprice * (1.0 - l_discount) as revenue"})
          .partialAggregation({"l_suppkey"}, {"sum(revenue) as total_revenue"})
          .localPartition({"l_suppkey"})
          .finalAggregation()
          .localPartition(std::vector<std::string>{})
          .window({"max(total_revenue) over () as max_revenue"})
          .filter("total_revenue = max_revenue")
          .project({"l_suppkey", "total_revenue"})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kSupplier,
           .columns = {"s_suppkey", "s_name", "s_address", "s_phone"}})
          .hashJoin(
              {"s_suppkey"},
              {"l_suppkey"},
              topSuppliers,
              "",
              {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"})
          .localPartition(std::vector<std::string>{})
          .orderBy({"s_suppkey"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ16Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  // Keep the IN-list literals INTEGER, as the p_size column is: BIGINT
  // literals would cast the column and keep the filter from being pushed
  // into the scan.
  auto options = decimalParseOptions();
  options.parseIntegerAsBigint = false;
  auto parts = scan(
                   ids,
                   plan,
                   {.table = kPart,
                    .columns = {"p_brand", "p_type", "p_size", "p_partkey"},
                    .filters = {"p_size in (38, 2, 8, 31, 44, 5, 14, 24)"},
                    .remainingFilter = "p_type NOT LIKE 'MEDIUM PLATED%'"},
                   options)
                   // Parquet scans take no not-equal subfield filter.
                   .filter("p_brand <> 'Brand#21'")
                   .planNode();
  auto complainingSuppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier,
           .columns = {"s_suppkey"},
           .filterOnly = {"s_comment"},
           .remainingFilter = "s_comment LIKE '%Customer%Complaints%'"})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kPartsupp, .columns = {"ps_partkey", "ps_suppkey"}})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              complainingSuppliers,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"},
              core::JoinType::kAnti,
              /*nullAware=*/true)
          // count(distinct ps_suppkey) as a distinct aggregation followed by
          // a count.
          .partialAggregation({"p_brand", "p_type", "p_size", "ps_suppkey"}, {})
          .localPartition({"p_brand", "p_type", "p_size", "ps_suppkey"})
          .finalAggregation()
          .partialAggregation(
              {"p_brand", "p_type", "p_size"},
              {"count(ps_suppkey) as supplier_cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supplier_cnt DESC", "p_brand", "p_type", "p_size"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ17Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto parts =
      scan(
          ids,
          plan,
          {.table = kPart,
           .columns = {"p_partkey"},
           .filterOnly = {"p_brand", "p_container"},
           .filters = {"p_brand = 'Brand#13'", "p_container = 'JUMBO CAN'"}})
          .planNode();
  // The join keeps every lineitem of each selected part, so windows over the
  // part key compute the correlated average. As in the SQL, where AVG of a
  // DECIMAL is DOUBLE, the average and the comparison are in DOUBLE.
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_partkey", "l_quantity", "l_extendedprice"}})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              parts,
              "",
              {"l_partkey", "l_quantity", "l_extendedprice"})
          .project(
              {"l_partkey",
               "l_extendedprice",
               "cast(l_quantity as DOUBLE) as qty_double"})
          .localPartition({"l_partkey"})
          .window(
              {"sum(qty_double) over (partition by l_partkey rows between unbounded preceding and unbounded following) as sum_qty",
               "count(qty_double) over (partition by l_partkey rows between unbounded preceding and unbounded following) as count_qty"})
          .filter(
              "qty_double < cast(0.2 as DOUBLE) * (sum_qty / cast(count_qty as DOUBLE))")
          .project({"l_extendedprice"})
          .partialAggregation({}, {"sum(l_extendedprice) as partial_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"cast(partial_sum as DOUBLE) / cast(7.0 as DOUBLE) as avg_yearly"})
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ18Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  // The DECIMAL quantity of each large order. Given unique o_orderkey and
  // c_custkey, it is also the sum of the outer query.
  auto largeOrders =
      scan(
          ids,
          plan,
          {.table = kLineitem, .columns = {"l_orderkey", "l_quantity"}})
          .partialAggregation({"l_orderkey"}, {"sum(l_quantity) AS quantity"})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .filter("quantity > cast(300 as DECIMAL(38, 2))")
          .planNode();
  auto customers =
      scan(ids, plan, {.table = kCustomer, .columns = {"c_name", "c_custkey"}})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kOrders,
           .columns =
               {"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"}})
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              largeOrders,
              "",
              {"o_orderkey",
               "o_custkey",
               "o_orderdate",
               "o_totalprice",
               "quantity"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"c_name",
               "c_custkey",
               "o_orderkey",
               "o_orderdate",
               "o_totalprice",
               "quantity"})
          .localPartition(std::vector<std::string>{})
          .orderBy({"o_totalprice DESC", "o_orderdate"}, false)
          .limit(0, 100, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ19Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  const std::string partFilter =
      "((p_brand = 'Brand#41')"
      " AND (l_quantity between cast(2 as DECIMAL(15, 2)) and cast(12 as DECIMAL(15, 2)))"
      " AND (p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'))"
      " AND (p_size BETWEEN 1 AND 5))"
      " OR ((p_brand ='Brand#13')"
      " AND (p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'))"
      " AND (l_quantity between cast(14 as DECIMAL(15, 2)) and cast(24 as DECIMAL(15, 2)))"
      " AND (p_size BETWEEN 1 AND 10))"
      " OR ((p_brand = 'Brand#55')"
      " AND (p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))"
      " AND (l_quantity between cast(23 as DECIMAL(15, 2)) and cast(33 as DECIMAL(15, 2)))"
      " AND (p_size BETWEEN 1 AND 15))";
  auto parts =
      scan(
          ids,
          plan,
          {.table = kPart,
           .columns = {"p_partkey", "p_brand", "p_container", "p_size"}})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_partkey", "l_extendedprice", "l_discount", "l_quantity"},
           .filterOnly = {"l_shipmode", "l_shipinstruct"},
           .filters =
               {"l_shipmode IN ('AIR', 'AIR REG')",
                "(l_shipinstruct = 'DELIVER IN PERSON')"}})
          .project(
              {"l_extendedprice * (1.0 - l_discount) as part_revenue",
               "l_partkey",
               "l_quantity"})
          .hashJoin(
              {"l_partkey"}, {"p_partkey"}, parts, partFilter, {"part_revenue"})
          .partialAggregation({}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ20Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto antiqueParts = [&]() {
    return scan(
               ids,
               plan,
               {.table = kPart,
                .columns = {"p_partkey"},
                .filterOnly = {"p_name"},
                .remainingFilter = "p_name like 'antique%'"})
        .planNode();
  };
  auto nations = scan(
                     ids,
                     plan,
                     {.table = kNation,
                      .columns = {"n_nationkey", "n_name"},
                      .filters = {"n_name = 'KENYA'"}})
                     .planNode();
  auto partSupply =
      scan(
          ids,
          plan,
          {.table = kPartsupp,
           .columns = {"ps_availqty", "ps_partkey", "ps_suppkey"}})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              antiqueParts(),
              "",
              {"ps_partkey", "ps_suppkey", "ps_availqty"},
              core::JoinType::kLeftSemiFilter)
          .planNode();
  auto suppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier,
           .columns = {"s_nationkey", "s_address", "s_name", "s_suppkey"}})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nations,
              "",
              {"s_name", "s_address", "s_suppkey"})
          .planNode();
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns = {"l_partkey", "l_suppkey", "l_quantity"},
           .filterOnly = {"l_shipdate"},
           .filters = {dateFilter(
               kLineitem, "l_shipdate", "'1993-01-01'", "'1993-12-31'")}})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              antiqueParts(),
              "",
              {"l_partkey", "l_suppkey", "l_quantity"})
          .partialAggregation(
              {"l_partkey", "l_suppkey"}, {"sum(l_quantity) AS sum_qty"})
          .localPartition({"l_partkey", "l_suppkey"})
          .finalAggregation()
          .project({"l_partkey", "l_suppkey", "0.5 * sum_qty AS filter_qty"})
          .hashJoin(
              {"l_partkey", "l_suppkey"},
              {"ps_partkey", "ps_suppkey"},
              partSupply,
              "cast(ps_availqty as DECIMAL(38, 3)) > filter_qty",
              {"ps_suppkey"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              suppliers,
              "",
              {"s_name", "s_address"},
              core::JoinType::kRightSemiFilter)
          .orderBy({"s_name"}, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ21Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto nations = scan(
                     ids,
                     plan,
                     {.table = kNation,
                      .columns = {"n_nationkey"},
                      .filterOnly = {"n_name"},
                      .filters = {"n_name = 'BRAZIL'"}})
                     .planNode();
  auto suppliers =
      scan(
          ids,
          plan,
          {.table = kSupplier, .columns = {"s_suppkey", "s_nationkey"}})
          .hashJoin(
              {"s_nationkey"}, {"n_nationkey"}, nations, "", {"s_suppkey"})
          .planNode();
  auto lateLines = scan(
                       ids,
                       plan,
                       {.table = kLineitem,
                        .columns = {"l_orderkey", "l_suppkey"},
                        .filterOnly = {"l_commitdate", "l_receiptdate"},
                        .remainingFilter = "l_receiptdate > l_commitdate"})
                       .hashJoin(
                           {"l_suppkey"},
                           {"s_suppkey"},
                           suppliers,
                           "",
                           {"l_orderkey", "l_suppkey"})
                       .project(
                           {"l_orderkey as candidate_order",
                            "l_suppkey as candidate_supplier"})
                       .planNode();
  // (order, supplier) pairs of the late lineitems of the selected suppliers
  // in finished orders, with the number of such lineitems.
  auto candidates = scan(
                        ids,
                        plan,
                        {.table = kOrders,
                         .columns = {"o_orderkey"},
                         .filterOnly = {"o_orderstatus"},
                         .filters = {"o_orderstatus = 'F'"}})
                        .hashJoin(
                            {"o_orderkey"},
                            {"candidate_order"},
                            lateLines,
                            "",
                            {"candidate_order", "candidate_supplier"})
                        .partialAggregation(
                            {"candidate_order", "candidate_supplier"},
                            {"count(0) as late_rows"})
                        .localPartition({"candidate_order"})
                        .finalAggregation()
                        .planNode();
  // The inner join to the order's lineitems of other suppliers establishes
  // EXISTS, and no late lineitem among them NOT EXISTS. late_rows keeps the
  // number of qualifying l1 rows of each pair.
  auto waiting =
      scan(
          ids,
          plan,
          {.table = kLineitem,
           .columns =
               {"l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"}})
          .project(
              {"l_orderkey as other_order",
               "l_suppkey as other_supplier",
               "if(l_receiptdate > l_commitdate, 1, 0) as other_late"})
          .hashJoin(
              {"other_order"},
              {"candidate_order"},
              candidates,
              "",
              {"candidate_order",
               "candidate_supplier",
               "late_rows",
               "other_supplier",
               "other_late"})
          .filter("other_supplier <> candidate_supplier")
          .project(
              {"candidate_order",
               "candidate_supplier",
               "late_rows",
               "other_late"})
          .partialAggregation(
              {"candidate_order", "candidate_supplier", "late_rows"},
              {"max(other_late) as any_late"})
          .localPartition({"candidate_order"})
          .finalAggregation()
          .filter("any_late = 0")
          .partialAggregation(
              {"candidate_supplier"}, {"sum(late_rows) as waiting_rows"})
          .localPartition({"candidate_supplier"})
          .finalAggregation()
          .planNode();
  plan.plan =
      scan(ids, plan, {.table = kSupplier, .columns = {"s_suppkey", "s_name"}})
          .hashJoin(
              {"s_suppkey"},
              {"candidate_supplier"},
              waiting,
              "",
              {"s_name", "waiting_rows"})
          .partialAggregation({"s_name"}, {"sum(waiting_rows) as numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();
  return plan;
}

TpchPlan CudfTpchQueryBuilder::getQ22Plan() const {
  auto plan = newPlan();
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto orders =
      scan(ids, plan, {.table = kOrders, .columns = {"o_custkey"}}).planNode();
  // The average of the positive balances among the customers of the selected
  // country codes, from windows over those customers: an exact DECIMAL sum
  // converted to DOUBLE and divided by the count. The SQL's AVG of a DECIMAL
  // is DOUBLE, so the balances compare with it in DOUBLE.
  plan.plan =
      scan(
          ids,
          plan,
          {.table = kCustomer,
           .columns = {"c_custkey", "c_acctbal", "c_phone"},
           .remainingFilter =
               "substr(c_phone, 1, 2) IN ('24', '31', '11', '16', '21', '20', '34')"})
          .project(
              {"c_custkey",
               "c_acctbal",
               "substr(c_phone, 1, 2) as country_code",
               "if(c_acctbal > cast(0.00 as DECIMAL(15, 2)), c_acctbal, cast(null as DECIMAL(15, 2))) as positive_balance"})
          .localPartition(std::vector<std::string>{})
          .window(
              {"sum(positive_balance) over () as positive_sum",
               "count(positive_balance) over () as positive_count"})
          .filter(
              "cast(c_acctbal as DOUBLE) > if(positive_count > 0, cast(positive_sum as DOUBLE) / cast(positive_count as DOUBLE), cast(null as DOUBLE))")
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              {"c_acctbal", "country_code"},
              core::JoinType::kAnti,
              /*nullAware=*/false)
          .partialAggregation(
              {"country_code"},
              {"count(0) as numcust", "sum(c_acctbal) as totacctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"country_code"}, false)
          .planNode();
  return plan;
}

} // namespace facebook::velox::cudf_velox
