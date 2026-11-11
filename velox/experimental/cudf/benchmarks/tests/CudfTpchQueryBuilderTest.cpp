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

// Checks the plans of CudfTpchQueryBuilder against DuckDB, which runs the
// reference SQL with the same parameters over the same data. The data comes
// from the TPC-H connector, with the column types of the TPC-H Parquet files
// the benchmark reads: INTEGER keys, DECIMAL(15, 2) amounts and DATE dates.
// Each plan runs on the CPU with the Hive connector, and on the GPU with the
// cuDF Hive connector and with the encoded connector, without CPU fallback.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchQueryBuilder.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedColumnCache.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <unordered_map>

using namespace facebook::velox;

namespace {

using cudf_velox::CudfTpchQueryBuilder;
using exec::test::PlanBuilder;
using exec::test::TpchPlan;

namespace encoded = cudf_velox::connector::encoded;

// At scale factor 0.1 the TPC-H connector's data leaves Q8, Q18 and Q20 empty
// with these parameters.
constexpr double kScaleFactor = 1;
constexpr const char* kTpchConnectorId = "test-tpch";
constexpr int32_t kGenerateParts = 8;
constexpr int32_t kNumDrivers = 4;
// Small encoded segments and splits, so that scans prune segments and batch
// several splits.
constexpr int32_t kSegmentRows = 1 << 16;
constexpr int32_t kSegmentsPerSplit = 2;

// The reference SQL, the queries the plans' parameters come from, with Q11's
// FRACTION for 'scaleFactor'.
std::string referenceSql(int queryId, double scaleFactor = kScaleFactor) {
  switch (queryId) {
    case 1:
      return R"(
select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty,
  sum(l_extendedprice) as sum_base_price,
  sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
  avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price,
  avg(l_discount) as avg_disc, count(*) as count_order
from lineitem
where l_shipdate <= date '1995-08-19'
group by l_returnflag, l_linestatus
order by l_returnflag, l_linestatus)";
    case 2:
      return R"(
select s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, s.s_address,
  s.s_phone, s.s_comment
from part p, supplier s, partsupp ps, nation n, region r
where p.p_partkey = ps.ps_partkey and s.s_suppkey = ps.ps_suppkey
  and p.p_size = 41 and p.p_type like '%NICKEL'
  and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey
  and r.r_name = 'EUROPE'
  and ps.ps_supplycost = (
    select min(ps.ps_supplycost)
    from partsupp ps, supplier s, nation n, region r
    where p.p_partkey = ps.ps_partkey and s.s_suppkey = ps.ps_suppkey
      and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey
      and r.r_name = 'EUROPE')
order by s.s_acctbal desc, n.n_name, s.s_name, p.p_partkey
limit 100)";
    case 3:
      return R"(
select l.l_orderkey, sum(l.l_extendedprice * (1 - l.l_discount)) as revenue,
  o.o_orderdate, o.o_shippriority
from customer c, orders o, lineitem l
where c.c_mktsegment = 'HOUSEHOLD' and c.c_custkey = o.o_custkey
  and l.l_orderkey = o.o_orderkey and o.o_orderdate < date '1995-03-25'
  and l.l_shipdate > date '1995-03-25'
group by l.l_orderkey, o.o_orderdate, o.o_shippriority
order by revenue desc, o.o_orderdate
limit 10)";
    case 4:
      return R"(
select o.o_orderpriority, count(*) as order_count
from orders o
where o.o_orderdate >= date '1996-10-01' and o.o_orderdate < date '1997-01-01'
  and exists (
    select * from lineitem l
    where l.l_orderkey = o.o_orderkey and l.l_commitdate < l.l_receiptdate)
group by o.o_orderpriority
order by o.o_orderpriority)";
    case 5:
      return R"(
select n.n_name, sum(l.l_extendedprice * (1 - l.l_discount)) as revenue
from orders o, lineitem l, supplier s, nation n, region r, customer c
where c.c_custkey = o.o_custkey and l.l_orderkey = o.o_orderkey
  and l.l_suppkey = s.s_suppkey and c.c_nationkey = s.s_nationkey
  and s.s_nationkey = n.n_nationkey and n.n_regionkey = r.r_regionkey
  and r.r_name = 'EUROPE' and o.o_orderdate >= date '1997-01-01'
  and o.o_orderdate < date '1998-01-01'
group by n.n_name
order by revenue desc)";
    case 6:
      return R"(
select sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '1997-01-01' and l_shipdate < date '1998-01-01'
  and l_discount between 0.03 - 0.01 and 0.03 + 0.01 and l_quantity < 24)";
    case 7:
      return R"(
select supp_nation, cust_nation, l_year, sum(volume) as revenue
from (
  select n1.n_name as supp_nation, n2.n_name as cust_nation,
    extract(year from l.l_shipdate) as l_year,
    l.l_extendedprice * (1 - l.l_discount) as volume
  from supplier s, lineitem l, orders o, customer c, nation n1, nation n2
  where s.s_suppkey = l.l_suppkey and o.o_orderkey = l.l_orderkey
    and c.c_custkey = o.o_custkey and s.s_nationkey = n1.n_nationkey
    and c.c_nationkey = n2.n_nationkey
    and ((n1.n_name = 'EGYPT' and n2.n_name = 'UNITED STATES')
      or (n1.n_name = 'UNITED STATES' and n2.n_name = 'EGYPT'))
    and l.l_shipdate between date '1995-01-01' and date '1996-12-31'
) as shipping
group by supp_nation, cust_nation, l_year
order by supp_nation, cust_nation, l_year)";
    case 8:
      return R"(
select o_year,
  sum(case when nation = 'EGYPT' then volume else 0 end) / sum(volume)
    as mkt_share
from (
  select extract(year from o.o_orderdate) as o_year,
    l.l_extendedprice * (1 - l.l_discount) as volume, n2.n_name as nation
  from lineitem l, part p, supplier s, orders o, customer c, nation n1,
    nation n2, region r
  where p.p_partkey = l.l_partkey and s.s_suppkey = l.l_suppkey
    and l.l_orderkey = o.o_orderkey and o.o_custkey = c.c_custkey
    and c.c_nationkey = n1.n_nationkey and n1.n_regionkey = r.r_regionkey
    and r.r_name = 'MIDDLE EAST' and s.s_nationkey = n2.n_nationkey
    and o.o_orderdate between date '1995-01-01' and date '1996-12-31'
    and p.p_type = 'PROMO BRUSHED COPPER'
) as all_nations
group by o_year
order by o_year)";
    case 9:
      return R"(
select nation, o_year, sum(amount) as sum_profit
from (
  select n.n_name as nation, extract(year from o.o_orderdate) as o_year,
    l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity
      as amount
  from part p, supplier s, lineitem l, partsupp ps, orders o, nation n
  where s.s_suppkey = l.l_suppkey and ps.ps_suppkey = l.l_suppkey
    and ps.ps_partkey = l.l_partkey and p.p_partkey = l.l_partkey
    and o.o_orderkey = l.l_orderkey and s.s_nationkey = n.n_nationkey
    and p.p_name like '%yellow%'
) as profit
group by nation, o_year
order by nation, o_year desc)";
    case 10:
      return R"(
select c.c_custkey, c.c_name,
  sum(l.l_extendedprice * (1 - l.l_discount)) as revenue, c.c_acctbal,
  n.n_name, c.c_address, c.c_phone, c.c_comment
from customer c, orders o, lineitem l, nation n
where c.c_custkey = o.o_custkey and l.l_orderkey = o.o_orderkey
  and o.o_orderdate >= date '1994-03-01' and o.o_orderdate < date '1994-06-01'
  and l.l_returnflag = 'R' and c.c_nationkey = n.n_nationkey
group by c.c_custkey, c.c_name, c.c_acctbal, c.c_phone, n.n_name,
  c.c_address, c.c_comment
order by revenue desc
limit 20)";
    case 11:
      return fmt::format(
          R"(
select ps.ps_partkey, sum(ps.ps_supplycost * ps.ps_availqty) as value
from partsupp ps, supplier s, nation n
where ps.ps_suppkey = s.s_suppkey and s.s_nationkey = n.n_nationkey
  and n.n_name = 'GERMANY'
group by ps.ps_partkey having
  sum(ps.ps_supplycost * ps.ps_availqty) > (
    select sum(ps.ps_supplycost * ps.ps_availqty) * {}
    from partsupp ps, supplier s, nation n
    where ps.ps_suppkey = s.s_suppkey and s.s_nationkey = n.n_nationkey
      and n.n_name = 'GERMANY')
order by value desc)",
          CudfTpchQueryBuilder::q11Fraction(scaleFactor));
    case 12:
      return R"(
select l.l_shipmode,
  sum(case when o.o_orderpriority = '1-URGENT' or o.o_orderpriority = '2-HIGH'
    then 1 else 0 end) as high_line_count,
  sum(case when o.o_orderpriority <> '1-URGENT'
    and o.o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count
from orders o, lineitem l
where o.o_orderkey = l.l_orderkey and l.l_shipmode in ('TRUCK', 'REG AIR')
  and l.l_commitdate < l.l_receiptdate and l.l_shipdate < l.l_commitdate
  and l.l_receiptdate >= date '1994-01-01'
  and l.l_receiptdate < date '1995-01-01'
group by l.l_shipmode
order by l.l_shipmode)";
    case 13:
      return R"(
select c_count, count(*) as custdist
from (
  select c.c_custkey, count(o.o_orderkey)
  from customer c left outer join orders o
    on c.c_custkey = o.o_custkey
    and o.o_comment not like '%special%requests%'
  group by c.c_custkey
) as orders (c_custkey, c_count)
group by c_count
order by custdist desc, c_count desc)";
    case 14:
      return R"(
select 100.00 * sum(case when p.p_type like 'PROMO%'
    then l.l_extendedprice * (1 - l.l_discount) else 0 end)
  / sum(l.l_extendedprice * (1 - l.l_discount)) as promo_revenue
from lineitem l, part p
where l.l_partkey = p.p_partkey and l.l_shipdate >= date '1994-08-01'
  and l.l_shipdate < date '1994-09-01')";
    case 15:
      return R"(
with revenue_view as (
  select l_suppkey as supplier_no,
    sum(l_extendedprice * (1 - l_discount)) as total_revenue
  from lineitem
  where l_shipdate >= date '1993-05-01' and l_shipdate < date '1993-08-01'
  group by l_suppkey)
select s.s_suppkey, s.s_name, s.s_address, s.s_phone, r.total_revenue
from supplier s, revenue_view r
where s.s_suppkey = r.supplier_no
  and r.total_revenue = (select max(total_revenue) from revenue_view)
order by s.s_suppkey)";
    case 16:
      return R"(
select p.p_brand, p.p_type, p.p_size,
  count(distinct ps.ps_suppkey) as supplier_cnt
from partsupp ps, part p
where p.p_partkey = ps.ps_partkey and p.p_brand <> 'Brand#21'
  and p.p_type not like 'MEDIUM PLATED%'
  and p.p_size in (38, 2, 8, 31, 44, 5, 14, 24)
  and ps.ps_suppkey not in (
    select s.s_suppkey from supplier s
    where s.s_comment like '%Customer%Complaints%')
group by p.p_brand, p.p_type, p.p_size
order by supplier_cnt desc, p.p_brand, p.p_type, p.p_size)";
    case 17:
      return R"(
select sum(l.l_extendedprice) / 7.0 as avg_yearly
from lineitem l, part p
where p.p_partkey = l.l_partkey and p.p_brand = 'Brand#13'
  and p.p_container = 'JUMBO CAN'
  and l.l_quantity < (
    select 0.2 * avg(l2.l_quantity) from lineitem l2
    where l2.l_partkey = p.p_partkey))";
    case 18:
      return R"(
select c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice,
  sum(l.l_quantity)
from customer c, orders o, lineitem l
where o.o_orderkey in (
    select l_orderkey from lineitem
    group by l_orderkey having sum(l_quantity) > 300)
  and c.c_custkey = o.o_custkey and o.o_orderkey = l.l_orderkey
group by c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice
order by o.o_totalprice desc, o.o_orderdate
limit 100)";
    case 19:
      return R"(
select sum(l.l_extendedprice* (1 - l.l_discount)) as revenue
from lineitem l, part p
where (p.p_partkey = l.l_partkey and p.p_brand = 'Brand#41'
    and p.p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    and l.l_quantity >= 2 and l.l_quantity <= 2 + 10
    and p.p_size between 1 and 5 and l.l_shipmode in ('AIR', 'AIR REG')
    and l.l_shipinstruct = 'DELIVER IN PERSON')
  or (p.p_partkey = l.l_partkey and p.p_brand = 'Brand#13'
    and p.p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
    and l.l_quantity >= 14 and l.l_quantity <= 14 + 10
    and p.p_size between 1 and 10 and l.l_shipmode in ('AIR', 'AIR REG')
    and l.l_shipinstruct = 'DELIVER IN PERSON')
  or (p.p_partkey = l.l_partkey and p.p_brand = 'Brand#55'
    and p.p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
    and l.l_quantity >= 23 and l.l_quantity <= 23 + 10
    and p.p_size between 1 and 15 and l.l_shipmode in ('AIR', 'AIR REG')
    and l.l_shipinstruct = 'DELIVER IN PERSON'))";
    case 20:
      return R"(
select s.s_name, s.s_address
from supplier s, nation n
where s.s_suppkey in (
    select ps.ps_suppkey from partsupp ps
    where ps.ps_partkey in (
        select p.p_partkey from part p where p.p_name like 'antique%')
      and ps.ps_availqty > (
        select 0.5 * sum(l.l_quantity) from lineitem l
        where l.l_partkey = ps.ps_partkey and l.l_suppkey = ps.ps_suppkey
          and l.l_shipdate >= date '1993-01-01'
          and l.l_shipdate < date '1994-01-01'))
  and s.s_nationkey = n.n_nationkey and n.n_name = 'KENYA'
order by s.s_name)";
    case 21:
      return R"(
select s.s_name, count(*) as numwait
from supplier s, lineitem l1, orders o, nation n
where s.s_suppkey = l1.l_suppkey and o.o_orderkey = l1.l_orderkey
  and o.o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate
  and exists (
    select * from lineitem l2
    where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey)
  and not exists (
    select * from lineitem l3
    where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey
      and l3.l_receiptdate > l3.l_commitdate)
  and s.s_nationkey = n.n_nationkey and n.n_name = 'BRAZIL'
group by s.s_name
order by numwait desc, s.s_name
limit 100)";
    case 22:
      return R"(
select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal
from (
  select substring(c_phone from 1 for 2) as cntrycode, c_acctbal
  from customer c
  where substring(c_phone from 1 for 2) in
      ('24', '31', '11', '16', '21', '20', '34')
    and c_acctbal > (
      select avg(c_acctbal) from customer
      where c_acctbal > 0.00 and substring(c_phone from 1 for 2) in
        ('24', '31', '11', '16', '21', '20', '34'))
    and not exists (select * from orders o where o.o_custkey = c.c_custkey)
) as custsale
group by cntrycode
order by cntrycode)";
    default:
      VELOX_FAIL("No query {}", queryId);
  }
}

// The positions of the ORDER BY columns in each query's output.
std::optional<std::vector<uint32_t>> sortingKeys(int queryId) {
  using Keys = std::vector<uint32_t>;
  switch (queryId) {
    case 1:
      return Keys{0, 1};
    case 2:
      return Keys{0, 2, 1, 3};
    case 3:
      return Keys{1, 2};
    case 4:
    case 8:
    case 12:
    case 15:
    case 20:
    case 22:
      return Keys{0};
    case 5:
    case 11:
      return Keys{1};
    case 7:
      return Keys{0, 1, 2};
    case 9:
      return Keys{0, 1};
    case 10:
      return Keys{2};
    case 13:
    case 21:
      return Keys{1, 0};
    case 16:
      return Keys{3, 0, 1, 2};
    case 18:
      return Keys{4, 3};
    default:
      return std::nullopt;
  }
}

// The TPC-H tables written as Parquet with the types of the benchmark data,
// and loaded into DuckDB.
class TpchData {
 public:
  static TpchData& instance() {
    static TpchData data;
    return data;
  }

  const std::string& path() const {
    return directory_->getPath();
  }

  exec::test::DuckDbQueryRunner& duckDb() {
    return duckDb_;
  }

 private:
  TpchData() : directory_(common::testutil::TempDirectoryPath::create()) {
    auto rootPool = memory::memoryManager()->addRootPool();
    auto pool = rootPool->addLeafChild("tpch-data");
    connector::tpch::TpchConnectorFactory tpchFactory;
    connector::ConnectorRegistry::global().insert(
        kTpchConnectorId,
        tpchFactory.newConnector(
            kTpchConnectorId,
            std::make_shared<config::ConfigBase>(
                std::unordered_map<std::string, std::string>())));
    for (const auto table : tpch::tables) {
      const auto name = std::string(tpch::toTableName(table));
      const auto schema = tpch::getTableSchema(table);
      // The TPC-H connector's BIGINT keys are INTEGER and its DOUBLE amounts
      // DECIMAL(15, 2) in the benchmark data.
      std::vector<std::string> columns;
      for (auto i = 0; i < schema->size(); ++i) {
        const auto& column = schema->nameOf(i);
        switch (schema->childAt(i)->kind()) {
          case TypeKind::BIGINT:
            columns.push_back(
                fmt::format("cast({} as INTEGER) as {}", column, column));
            break;
          case TypeKind::DOUBLE:
            columns.push_back(
                fmt::format(
                    "cast({} as DECIMAL(15, 2)) as {}", column, column));
            break;
          default:
            columns.push_back(column);
        }
      }
      std::vector<exec::Split> splits;
      for (int32_t part = 0; part < kGenerateParts; ++part) {
        splits.emplace_back(
            std::make_shared<connector::tpch::TpchConnectorSplit>(
                kTpchConnectorId, /*cacheable=*/true, kGenerateParts, part));
      }
      auto rows =
          exec::test::AssertQueryBuilder(
              PlanBuilder()
                  .tpchTableScan(
                      table, schema->names(), kScaleFactor, kTpchConnectorId)
                  .project(columns)
                  .planNode())
              .splits(std::move(splits))
              .maxDrivers(kGenerateParts)
              .copyResults(pool.get());
      duckDb_.createTable(name, {rows});
      exec::test::AssertQueryBuilder(
          PlanBuilder()
              .values({rows})
              .tableWrite(
                  fmt::format("{}/{}", path(), name),
                  dwio::common::FileFormat::PARQUET)
              .planNode())
          .copyResults(pool.get());
    }
    connector::ConnectorRegistry::global().erase(kTpchConnectorId);
  }

  const std::shared_ptr<common::testutil::TempDirectoryPath> directory_;
  exec::test::DuckDbQueryRunner duckDb_;
};

std::vector<std::string> tpchTableNames() {
  std::vector<std::string> names;
  for (const auto table : tpch::tables) {
    names.emplace_back(tpch::toTableName(table));
  }
  return names;
}

std::shared_ptr<connector::Connector> makeHiveConnector() {
  connector::hive::HiveConnectorFactory factory;
  return factory.newConnector(
      exec::test::kHiveConnectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
}

// The cuDF Hive connector, configured as the benchmark configures it.
std::shared_ptr<connector::Connector> makeCudfHiveConnector(
    folly::Executor* ioExecutor) {
  using CudfHiveConfig = cudf_velox::connector::hive::CudfHiveConfig;
  std::unordered_map<std::string, std::string> config = {
      {CudfHiveConfig::kAllowMismatchedCudfHiveSchemas, "true"}};
  return cudf_velox::connector::hive::CudfHiveConnectorFactory().newConnector(
      exec::test::kHiveConnectorId,
      std::make_shared<config::ConfigBase>(std::move(config)),
      ioExecutor);
}

void setHiveConnector(std::shared_ptr<connector::Connector> connector) {
  connector::ConnectorRegistry::global().erase(exec::test::kHiveConnectorId);
  connector::ConnectorRegistry::global().insert(
      exec::test::kHiveConnectorId, std::move(connector));
}

// CudfConfig::allowCpuFallback before registerGpu().
bool savedAllowCpuFallback = true;

// Registers the cuDF operators and functions, as the benchmark does, and
// makes a plan fail rather than fall back to the CPU.
void registerGpu() {
  auto& config = cudf_velox::CudfConfig::getInstance();
  savedAllowCpuFallback = config.allowCpuFallback;
  config.allowCpuFallback = false;
  cudf_velox::registerCudf();
  cudf_velox::registerPrestoFunctions(config.functionNamePrefix);
}

void unregisterGpu() {
  cudf_velox::unregisterCudf();
  cudf_velox::CudfConfig::getInstance().allowCpuFallback =
      savedAllowCpuFallback;
}

// Loads 'tables' of the data under 'path' into the encoded column cache.
// registerGpu() makes the memory resources the cache loads with.
void loadEncodedTables(
    const std::string& path,
    const std::vector<std::string>& tables) {
  cudf_velox::ensureCudaContextForThread();
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  for (const auto& table : tables) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(
             fmt::format("{}/{}", path, table))) {
      if (entry.is_regular_file() &&
          entry.path().filename().string()[0] != '.') {
        files.push_back(entry.path().string());
      }
    }
    std::sort(files.begin(), files.end());
    encoded::EncodedColumnCache::instance().loadTable(
        table, files, kSegmentRows, stream, mr, mr, /*hostResident=*/false);
  }
  stream.sync();
}

using Splits = std::vector<std::shared_ptr<connector::ConnectorSplit>>;
using SplitMaker = std::function<Splits(const std::vector<std::string>&)>;

// One split per file, for the Hive connectors.
Splits hiveSplits(const std::vector<std::string>& files) {
  Splits result;
  for (const auto& file : files) {
    for (auto& split :
         exec::test::HiveConnectorTestBase::makeHiveConnectorSplits(
             file, 1, dwio::common::FileFormat::PARQUET)) {
      result.push_back(std::move(split));
    }
  }
  return result;
}

// The splits of the encoded table of 'files', which are in a directory named
// after the table.
Splits encodedSplits(const std::vector<std::string>& files) {
  return encoded::makeEncodedSplits(
      exec::test::kHiveConnectorId,
      std::filesystem::path(files.front()).parent_path().filename().string(),
      kSegmentsPerSplit);
}

exec::CursorParameters cursorParameters(const TpchPlan& plan) {
  exec::CursorParameters params;
  params.planNode = plan.plan;
  params.maxDrivers = kNumDrivers;
  return params;
}

// Adds the splits of each scan of 'plan', made by 'splits', to a task. 'plan'
// must outlive the returned function.
std::function<void(exec::TaskCursor*)> splitAdder(
    const TpchPlan& plan,
    SplitMaker splits) {
  return [&plan, splits = std::move(splits)](exec::TaskCursor* cursor) {
    if (cursor->noMoreSplits()) {
      return;
    }
    for (const auto& [nodeId, files] : plan.dataFiles) {
      for (auto& split : splits(files)) {
        cursor->task()->addSplit(nodeId, exec::Split(std::move(split)));
      }
      cursor->task()->noMoreSplits(nodeId);
    }
    cursor->setNoMoreSplits();
  };
}

// Runs 'plan', with the splits of each scan from 'splits', and compares the
// result with DuckDB's for 'sql', in order on 'sortingKeys'.
void assertPlan(
    const TpchPlan& plan,
    const SplitMaker& splits,
    const std::string& sql,
    exec::test::DuckDbQueryRunner& duckDb,
    std::optional<std::vector<uint32_t>> sortingKeys) {
  exec::test::assertQuery(
      cursorParameters(plan),
      splitAdder(plan, splits),
      sql,
      duckDb,
      std::move(sortingKeys));
}

// How the plans run: the base class on the CPU.
class CpuTpchTest : public testing::TestWithParam<int> {
 protected:
  static void SetUpTestSuite() {
    TpchData::instance();
    setHiveConnector(makeHiveConnector());
    builder_ = std::make_unique<CudfTpchQueryBuilder>(
        dwio::common::FileFormat::PARQUET, kScaleFactor);
    builder_->initialize(TpchData::instance().path());
  }

  static void TearDownTestSuite() {
    builder_.reset();
  }

  virtual Splits splits(const std::vector<std::string>& files) const {
    return hiveSplits(files);
  }

  void assertQuery(int queryId) const {
    // Every query returns rows on this data, and the global aggregations a
    // value, so no comparison is vacuous.
    const auto reference =
        TpchData::instance().duckDb().execute(referenceSql(queryId));
    std::cout << fmt::format(
                     "Q{} returns {} rows", queryId, reference->RowCount())
              << std::endl;
    ASSERT_GT(reference->RowCount(), 0);
    EXPECT_FALSE(reference->GetValue(0, 0).IsNull());

    assertPlan(
        builder_->getQueryPlan(queryId),
        [this](const std::vector<std::string>& files) { return splits(files); },
        referenceSql(queryId),
        TpchData::instance().duckDb(),
        sortingKeys(queryId));
  }

  inline static std::unique_ptr<CudfTpchQueryBuilder> builder_;
};

// On the GPU with the cuDF Hive connector, as the benchmark runs without
// --encoded.
class CudfParquetTpchTest : public CpuTpchTest {
 protected:
  static void SetUpTestSuite() {
    CpuTpchTest::SetUpTestSuite();
    ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(4);
    setHiveConnector(makeCudfHiveConnector(ioExecutor_.get()));
    registerGpu();
  }

  static void TearDownTestSuite() {
    unregisterGpu();
    setHiveConnector(makeHiveConnector());
    ioExecutor_.reset();
    CpuTpchTest::TearDownTestSuite();
  }

  inline static std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};

// On the GPU with the encoded connector, as the benchmark runs with
// --encoded.
class EncodedTpchTest : public CpuTpchTest {
 protected:
  static void SetUpTestSuite() {
    CpuTpchTest::SetUpTestSuite();
    registerGpu();
    loadEncodedTables(TpchData::instance().path(), tpchTableNames());
    setHiveConnector(
        std::make_shared<encoded::EncodedConnector>(
            exec::test::kHiveConnectorId));
    // After registerCudf(), which resets the operator adapters.
    encoded::registerEncodedTableScanAdapter();
  }

  static void TearDownTestSuite() {
    unregisterGpu();
    setHiveConnector(makeHiveConnector());
    encoded::EncodedColumnCache::instance().clear();
    CpuTpchTest::TearDownTestSuite();
  }

  Splits splits(const std::vector<std::string>& files) const override {
    return encodedSplits(files);
  }
};

TEST_P(CpuTpchTest, matchesReferenceSql) {
  assertQuery(GetParam());
}

TEST_P(CudfParquetTpchTest, matchesReferenceSql) {
  assertQuery(GetParam());
}

TEST_P(EncodedTpchTest, matchesReferenceSql) {
  assertQuery(GetParam());
}

std::string queryName(const testing::TestParamInfo<int>& info) {
  return fmt::format("Q{}", info.param);
}

INSTANTIATE_TEST_SUITE_P(
    CudfTpchQueryBuilderTest,
    CpuTpchTest,
    testing::Range(1, 23),
    queryName);
INSTANTIATE_TEST_SUITE_P(
    CudfTpchQueryBuilderTest,
    CudfParquetTpchTest,
    testing::Range(1, 23),
    queryName);
INSTANTIATE_TEST_SUITE_P(
    CudfTpchQueryBuilderTest,
    EncodedTpchTest,
    testing::Range(1, 23),
    queryName);

// Data on which Q11 at SF100, where FRACTION is 0.0000010000, keeps a part
// just above its threshold. The German supplier's parts are worth 1.00,
// 999998.49 and 0.50, 999999.99 in all, so the threshold is 0.99999999 and
// the parts worth 1.00 and 999998.49 qualify. Rounded to six decimals the
// threshold would be 1.000000, which the part worth 1.00 does not exceed. The
// French supplier's part counts in neither the total nor the result. Tables
// other than nation, supplier and partsupp are TpchData's.
class Q11ThresholdData {
 public:
  static constexpr double kScaleFactor = 100;

  static Q11ThresholdData& instance() {
    static Q11ThresholdData data;
    return data;
  }

  const std::string& path() const {
    return directory_->getPath();
  }

  exec::test::DuckDbQueryRunner& duckDb() {
    return duckDb_;
  }

 private:
  Q11ThresholdData()
      : directory_(common::testutil::TempDirectoryPath::create()) {
    auto rootPool = memory::memoryManager()->addRootPool();
    auto pool = rootPool->addLeafChild("q11-threshold-data");
    facebook::velox::test::VectorMaker maker(pool.get());
    const std::unordered_map<std::string, RowVectorPtr> tables = {
        {"nation",
         maker.rowVector(
             {"n_nationkey", "n_name", "n_regionkey", "n_comment"},
             {maker.flatVector<int32_t>({7, 6}),
              maker.flatVector<std::string>({"GERMANY", "FRANCE"}),
              maker.flatVector<int32_t>({3, 3}),
              maker.flatVector<std::string>({"", ""})})},
        {"supplier",
         maker.rowVector(
             {"s_suppkey",
              "s_name",
              "s_address",
              "s_nationkey",
              "s_phone",
              "s_acctbal",
              "s_comment"},
             {maker.flatVector<int32_t>({1, 2}),
              maker.flatVector<std::string>(
                  {"Supplier#000000001", "Supplier#000000002"}),
              maker.flatVector<std::string>({"", ""}),
              maker.flatVector<int32_t>({7, 6}),
              maker.flatVector<std::string>({"", ""}),
              maker.flatVector<int64_t>({0, 0}, DECIMAL(15, 2)),
              maker.flatVector<std::string>({"", ""})})},
        {"partsupp",
         maker.rowVector(
             {"ps_partkey",
              "ps_suppkey",
              "ps_availqty",
              "ps_supplycost",
              "ps_comment"},
             {maker.flatVector<int32_t>({1, 2, 3, 4}),
              maker.flatVector<int32_t>({1, 1, 1, 2}),
              maker.flatVector<int32_t>({1, 1, 1, 1}),
              // 1.00, 999998.49 and 0.50 from the German supplier,
              // 5000000.00 from the French one.
              maker.flatVector<int64_t>(
                  {100, 99999849, 50, 500000000}, DECIMAL(15, 2)),
              maker.flatVector<std::string>({"", "", "", ""})})},
    };
    for (const auto& name : tpchTableNames()) {
      const auto tableDirectory = fmt::format("{}/{}", path(), name);
      const auto it = tables.find(name);
      if (it == tables.end()) {
        std::filesystem::create_directory_symlink(
            fmt::format("{}/{}", TpchData::instance().path(), name),
            tableDirectory);
        continue;
      }
      duckDb_.createTable(name, {it->second});
      exec::test::AssertQueryBuilder(
          PlanBuilder()
              .values({it->second})
              .tableWrite(tableDirectory, dwio::common::FileFormat::PARQUET)
              .planNode())
          .copyResults(pool.get());
    }
  }

  const std::shared_ptr<common::testutil::TempDirectoryPath> directory_;
  exec::test::DuckDbQueryRunner duckDb_;
};

enum class Mode { kCpu, kCudfParquet, kEncoded };

std::string modeName(const testing::TestParamInfo<Mode>& info) {
  switch (info.param) {
    case Mode::kCpu:
      return "cpu";
    case Mode::kCudfParquet:
      return "cudfParquet";
    case Mode::kEncoded:
      return "encoded";
  }
  VELOX_UNREACHABLE();
}

// Q11 on Q11ThresholdData, on the CPU and on the GPU with each connector.
class Q11ThresholdTest : public testing::TestWithParam<Mode> {
 protected:
  static void SetUpTestSuite() {
    // The data is written through the Hive connector.
    setHiveConnector(makeHiveConnector());
    Q11ThresholdData::instance();
  }

  void SetUp() override {
    switch (GetParam()) {
      case Mode::kCpu:
        break;
      case Mode::kCudfParquet:
        ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(4);
        setHiveConnector(makeCudfHiveConnector(ioExecutor_.get()));
        registerGpu();
        break;
      case Mode::kEncoded:
        registerGpu();
        loadEncodedTables(
            Q11ThresholdData::instance().path(),
            {"nation", "supplier", "partsupp"});
        setHiveConnector(
            std::make_shared<encoded::EncodedConnector>(
                exec::test::kHiveConnectorId));
        encoded::registerEncodedTableScanAdapter();
        break;
    }
  }

  void TearDown() override {
    if (GetParam() != Mode::kCpu) {
      unregisterGpu();
    }
    setHiveConnector(makeHiveConnector());
    ioExecutor_.reset();
    encoded::EncodedColumnCache::instance().clear();
  }

  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};

TEST_P(Q11ThresholdTest, keepsPartsAboveExactThreshold) {
  auto& data = Q11ThresholdData::instance();
  const auto sql = referenceSql(11, Q11ThresholdData::kScaleFactor);
  // The SQL returns both parts above 0.99999999, the one worth 1.00 included.
  const auto reference = data.duckDb().execute(sql);
  ASSERT_EQ(reference->RowCount(), 2);
  EXPECT_EQ(reference->GetValue(0, 0).GetValue<int32_t>(), 2);
  EXPECT_EQ(reference->GetValue(0, 1).GetValue<int32_t>(), 1);

  CudfTpchQueryBuilder builder(
      dwio::common::FileFormat::PARQUET, Q11ThresholdData::kScaleFactor);
  builder.initialize(data.path());
  const auto plan = builder.getQueryPlan(11);
  const auto splits = GetParam() == Mode::kEncoded ? SplitMaker(encodedSplits)
                                                   : SplitMaker(hiveSplits);

  // The plan returns the same parts.
  const auto [cursor, results] =
      exec::test::readCursor(cursorParameters(plan), splitAdder(plan, splits));
  std::vector<int32_t> partKeys;
  for (const auto& batch : results) {
    const auto* keys = batch->childAt(0)->as<SimpleVector<int32_t>>();
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      partKeys.push_back(keys->valueAt(row));
    }
  }
  ASSERT_EQ(partKeys, (std::vector<int32_t>{2, 1}));

  // With the same values, in the same order.
  assertPlan(plan, splits, sql, data.duckDb(), sortingKeys(11));
}

INSTANTIATE_TEST_SUITE_P(
    CudfTpchQueryBuilderTest,
    Q11ThresholdTest,
    testing::Values(Mode::kCpu, Mode::kCudfParquet, Mode::kEncoded),
    modeName);

class CudfTpchQueryBuilderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    builder_ = std::make_unique<CudfTpchQueryBuilder>(
        dwio::common::FileFormat::PARQUET, kScaleFactor);
    builder_->initialize(TpchData::instance().path());
  }

  static void TearDownTestSuite() {
    builder_.reset();
  }

  // Returns the nodes of 'plan' for which 'predicate' holds.
  static std::vector<const core::PlanNode*> findNodes(
      const core::PlanNodePtr& plan,
      const std::function<bool(const core::PlanNode&)>& predicate) {
    std::vector<const core::PlanNode*> nodes;
    std::function<void(const core::PlanNodePtr&)> visit =
        [&](const core::PlanNodePtr& node) {
          if (predicate(*node)) {
            nodes.push_back(node.get());
          }
          for (const auto& source : node->sources()) {
            visit(source);
          }
        };
    visit(plan);
    return nodes;
  }

  inline static std::unique_ptr<CudfTpchQueryBuilder> builder_;
};

TEST_F(CudfTpchQueryBuilderTest, q11Fraction) {
  EXPECT_EQ(CudfTpchQueryBuilder::q11Fraction(100), "0.0000010000");
  EXPECT_EQ(CudfTpchQueryBuilder::q11Fraction(1), "0.0001000000");
  EXPECT_EQ(CudfTpchQueryBuilder::q11Fraction(0.01), "0.0100000000");
  EXPECT_EQ(CudfTpchQueryBuilder::q11Fraction(1000), "0.0000001000");
  VELOX_ASSERT_USER_THROW(
      CudfTpchQueryBuilder::q11Fraction(0),
      "TPC-H scale factor must be positive");
  VELOX_ASSERT_USER_THROW(
      CudfTpchQueryBuilder::q11Fraction(-1),
      "TPC-H scale factor must be positive");
  VELOX_ASSERT_USER_THROW(
      CudfTpchQueryBuilder::q11Fraction(1e7),
      "makes Q11's FRACTION round to zero");
}

TEST_F(CudfTpchQueryBuilderTest, q11ComparesAtExactScale) {
  // The threshold keeps the scale of sum(ps_supplycost * ps_availqty) plus
  // FRACTION's ten decimal places, and the values are compared with it at
  // that scale.
  const auto plan = builder_->getQueryPlan(11).plan;
  const auto filters = findNodes(plan, [](const core::PlanNode& node) {
    return dynamic_cast<const core::FilterNode*>(&node) != nullptr &&
        dynamic_cast<const core::NestedLoopJoinNode*>(
            node.sources()[0].get()) != nullptr;
  });
  ASSERT_EQ(filters.size(), 1);
  const auto& filter = dynamic_cast<const core::FilterNode&>(*filters[0]);
  const auto& threshold = filter.sources()[0]->sources()[1]->outputType();
  EXPECT_EQ(
      threshold->findChild("filter_cost_qty")->toString(), "DECIMAL(38, 12)");
  ASSERT_EQ(filter.filter()->inputs().size(), 2);
  for (const auto& input : filter.filter()->inputs()) {
    EXPECT_EQ(input->type()->toString(), "DECIMAL(38, 12)");
  }
}

TEST_F(CudfTpchQueryBuilderTest, queryIds) {
  VELOX_ASSERT_USER_THROW(
      builder_->getQueryPlan(0), "TPC-H query 0 does not exist");
  VELOX_ASSERT_USER_THROW(
      builder_->getQueryPlan(23), "TPC-H query 23 does not exist");
}

TEST_F(CudfTpchQueryBuilderTest, filterOnlyColumnsAreProjectedAway) {
  // Every scan reads the columns it returns, and a column only a filter
  // reads is dropped right after the scan: Q13 returns o_comment from its
  // orders scan for its filter, and the projection above drops it.
  const auto plan = builder_->getQueryPlan(13).plan;
  const auto projections = findNodes(plan, [](const core::PlanNode& node) {
    const auto* project = dynamic_cast<const core::ProjectNode*>(&node);
    return project != nullptr &&
        dynamic_cast<const core::TableScanNode*>(project->sources()[0].get()) !=
        nullptr;
  });
  ASSERT_EQ(projections.size(), 1);
  const auto& scan = *projections[0]->sources()[0];
  EXPECT_EQ(
      scan.outputType()->names(),
      (std::vector<std::string>{"o_custkey", "o_orderkey", "o_comment"}));
  EXPECT_EQ(
      projections[0]->outputType()->names(),
      (std::vector<std::string>{"o_custkey", "o_orderkey"}));

  for (int queryId = 1; queryId <= 22; ++queryId) {
    SCOPED_TRACE(fmt::format("Q{}", queryId));
    const auto queryPlan = builder_->getQueryPlan(queryId);
    const auto scans =
        findNodes(queryPlan.plan, [](const core::PlanNode& node) {
          return dynamic_cast<const core::TableScanNode*>(&node) != nullptr;
        });
    EXPECT_EQ(scans.size(), queryPlan.dataFiles.size());
    for (const auto* node : scans) {
      EXPECT_EQ(queryPlan.dataFiles.count(node->id()), 1) << node->id();
    }
  }
}

TEST_F(CudfTpchQueryBuilderTest, q22WindowSumsDecimals) {
  // The average balance comes from an exact DECIMAL sum and a count over an
  // unpartitioned window, not from a second scan of customer.
  const auto plan = builder_->getQueryPlan(22);
  const auto windows = findNodes(plan.plan, [](const core::PlanNode& node) {
    return dynamic_cast<const core::WindowNode*>(&node) != nullptr;
  });
  ASSERT_EQ(windows.size(), 1);
  const auto& window = dynamic_cast<const core::WindowNode&>(*windows[0]);
  EXPECT_TRUE(window.partitionKeys().empty());
  ASSERT_EQ(window.windowFunctions().size(), 2);
  const auto& sum = window.windowFunctions()[0].functionCall;
  EXPECT_EQ(sum->name(), "sum");
  EXPECT_EQ(sum->inputs()[0]->type()->toString(), "DECIMAL(15, 2)");
  EXPECT_EQ(sum->type()->toString(), "DECIMAL(38, 2)");
  EXPECT_EQ(window.windowFunctions()[1].functionCall->name(), "count");
  EXPECT_EQ(plan.dataFiles.size(), 2);
}

TEST_F(CudfTpchQueryBuilderTest, q18SumsDecimals) {
  // Q18 sums l_quantity as DECIMAL; no unscaled integer sum.
  const auto plan = builder_->getQueryPlan(18).plan;
  EXPECT_EQ(plan->outputType()->childAt(5)->toString(), "DECIMAL(38, 2)");
}

} // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  filesystems::registerLocalFileSystem();
  dwio::common::registerFileSinks();
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
  // TpchData writes the Parquet files through the Hive connector.
  setHiveConnector(makeHiveConnector());
  return RUN_ALL_TESTS();
}
