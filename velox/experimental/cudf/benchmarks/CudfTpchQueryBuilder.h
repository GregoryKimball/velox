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

#include "velox/exec/tests/utils/TpchQueryBuilder.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Builds the TPC-H plans that velox_cudf_tpch_benchmark runs. The data
/// layout is that of velox::exec::test::TpchQueryBuilder: one directory per
/// table under the data path, holding that table's files.
///
/// Parameters. Every query uses this fixed set of substitution parameters
/// (TPC-H section 2.4), the set of the SF100 reference queries the plans are
/// validated against:
///
///   Q1   l_shipdate <= DATE '1995-08-19'
///   Q2   SIZE 41, TYPE '%NICKEL', REGION 'EUROPE'
///   Q3   SEGMENT 'HOUSEHOLD', DATE '1995-03-25'
///   Q4   DATE '1996-10-01'
///   Q5   REGION 'EUROPE', DATE '1997-01-01'
///   Q6   DATE '1997-01-01', DISCOUNT 0.03, QUANTITY 24
///   Q7   NATION1 'EGYPT', NATION2 'UNITED STATES'
///   Q8   NATION 'EGYPT', REGION 'MIDDLE EAST', TYPE 'PROMO BRUSHED COPPER'
///   Q9   COLOR 'yellow'
///   Q10  DATE '1994-03-01'
///   Q11  NATION 'GERMANY', FRACTION 0.0001 / scale factor
///   Q12  SHIPMODE1 'TRUCK', SHIPMODE2 'REG AIR', DATE '1994-01-01'
///   Q13  WORD1 'special', WORD2 'requests'
///   Q14  DATE '1994-08-01'
///   Q15  DATE '1993-05-01'
///   Q16  BRAND 'Brand#21', TYPE 'MEDIUM PLATED',
///        SIZE 38, 2, 8, 31, 44, 5, 14, 24
///   Q17  BRAND 'Brand#13', CONTAINER 'JUMBO CAN'
///   Q18  QUANTITY 300
///   Q19  QUANTITY 2, 14, 23; BRAND 'Brand#41', 'Brand#13', 'Brand#55'
///   Q20  COLOR 'antique', DATE '1993-01-01', NATION 'KENYA'
///   Q21  NATION 'BRAZIL'
///   Q22  I1-I7 '24', '31', '11', '16', '21', '20', '34'
///
/// Decimal literals are DECIMAL, not DOUBLE, as in SQL. Where the reference
/// SQL takes AVG of a DECIMAL column, which returns DOUBLE (Q1, Q17, Q22),
/// the plans compute and compare that average in DOUBLE as well.
///
/// Scans read only the columns the plan or their filters use. A scan returns
/// every column it reads, including those only its filters read, and a
/// projection right after the scan drops the latter. Leaving them out of the
/// scan's output instead would need changes to the encoded connector.
///
/// Rewrites. Besides the join orders, these plans differ from a direct
/// translation of the SQL:
///
///   Q4   Joins the selected orders to the order keys of the late lineitems
///        and counts distinct (order key, priority) pairs, instead of a semi
///        join.
///   Q5   Sums revenue per supplier nation key, then joins the nation names
///        and sums per name again.
///   Q9   Joins partsupp to the selected parts before probing it with
///        lineitem, sums profit per nation key and year, then joins the
///        nation names and sums per name and year again.
///   Q10  Sums revenue per order customer key before joining the customer
///        and nation columns, then sums again per output group and takes
///        the top 20 with TopN.
///   Q13  Counts the matching orders per customer key before the outer join
///        with customer.
///   Q15  Finds the top supplier revenue with a MAX window instead of a
///        second scan of lineitem.
///   Q17  Computes each part's average quantity with SUM and COUNT windows
///        over the joined lineitems instead of a second scan of lineitem.
///   Q21  Evaluates EXISTS and NOT EXISTS per (order, late supplier) pair
///        with one join to the lineitems of the candidate orders.
///   Q22  Computes the average balance with SUM and COUNT windows over the
///        customers of the selected country codes instead of a second scan
///        of customer. The SUM is an exact DECIMAL sum, converted to DOUBLE
///        and divided by the count, as in the aggregation it replaces.
///
/// Q4, Q13, Q18 and Q21 also rely on TPC-H's primary keys: Q4 on unique
/// o_orderkey, Q13 on unique c_custkey, Q18 on unique o_orderkey and
/// c_custkey (it returns the per-order quantity sum of the IN subquery as the
/// sum of the outer query), and Q21 on unique s_suppkey. On data with
/// duplicate keys these plans can differ from the SQL.
class CudfTpchQueryBuilder {
 public:
  /// @param format The format of the data files.
  /// @param scaleFactor The TPC-H scale factor of the data, which sets Q11's
  /// FRACTION. Must be positive.
  CudfTpchQueryBuilder(dwio::common::FileFormat format, double scaleFactor);

  /// Reads the schema of each table from its first data file and lists each
  /// table's data files.
  /// @param dataPath The directory holding one directory per table, or one
  /// file per table that lists the table's data files, one per line.
  void initialize(const std::string& dataPath);

  /// Returns the plan of TPC-H query 'queryId', 1 to 22.
  velox::exec::test::TpchPlan getQueryPlan(int queryId) const;

  /// Returns Q11's FRACTION, 0.0001 / 'scaleFactor', as a DECIMAL literal
  /// with ten decimal places, e.g. 0.0000010000 for scale factor 100.
  static std::string q11Fraction(double scaleFactor);

 private:
  /// A table scan and the columns the plan uses after it. 'filters' (subfield
  /// filters) and 'remainingFilter' may also read the 'filterOnly' columns,
  /// which a projection after the scan drops.
  struct Scan {
    std::string table;
    std::vector<std::string> columns;
    std::vector<std::string> filterOnly{};
    std::vector<std::string> filters{};
    std::string remainingFilter{};
  };

  /// Starts a plan with the scan 'spec', followed by a projection to
  /// 'spec.columns' when 'spec.filterOnly' is not empty, and adds the scan's
  /// data files to 'plan'. Expressions in the plan parse with 'options'.
  velox::exec::test::PlanBuilder scan(
      const std::shared_ptr<core::PlanNodeIdGenerator>& ids,
      velox::exec::test::TpchPlan& plan,
      const Scan& spec,
      const parse::ParseOptions& options) const;

  /// As above, parsing literals such as 1.0 as DECIMAL.
  velox::exec::test::PlanBuilder scan(
      const std::shared_ptr<core::PlanNodeIdGenerator>& ids,
      velox::exec::test::TpchPlan& plan,
      const Scan& spec) const;

  /// Returns a filter on the date column 'column' of 'table': 'column'
  /// between 'lower' and 'upper' when both are given, else greater than
  /// 'lower' or less than 'upper'. The bounds are quoted 'YYYY-MM-DD' dates.
  std::string dateFilter(
      const std::string& table,
      const std::string& column,
      const std::string& lower,
      const std::string& upper) const;

  RowTypePtr rowType(
      const std::string& table,
      const std::vector<std::string>& columns) const;

  velox::exec::test::TpchPlan newPlan() const;

  void readFileSchema(
      const std::string& table,
      const std::string& filePath,
      const std::vector<std::string>& columns);

  velox::exec::test::TpchPlan getQ1Plan() const;
  velox::exec::test::TpchPlan getQ2Plan() const;
  velox::exec::test::TpchPlan getQ3Plan() const;
  velox::exec::test::TpchPlan getQ4Plan() const;
  velox::exec::test::TpchPlan getQ5Plan() const;
  velox::exec::test::TpchPlan getQ6Plan() const;
  velox::exec::test::TpchPlan getQ7Plan() const;
  velox::exec::test::TpchPlan getQ8Plan() const;
  velox::exec::test::TpchPlan getQ9Plan() const;
  velox::exec::test::TpchPlan getQ10Plan() const;
  velox::exec::test::TpchPlan getQ11Plan() const;
  velox::exec::test::TpchPlan getQ12Plan() const;
  velox::exec::test::TpchPlan getQ13Plan() const;
  velox::exec::test::TpchPlan getQ14Plan() const;
  velox::exec::test::TpchPlan getQ15Plan() const;
  velox::exec::test::TpchPlan getQ16Plan() const;
  velox::exec::test::TpchPlan getQ17Plan() const;
  velox::exec::test::TpchPlan getQ18Plan() const;
  velox::exec::test::TpchPlan getQ19Plan() const;
  velox::exec::test::TpchPlan getQ20Plan() const;
  velox::exec::test::TpchPlan getQ21Plan() const;
  velox::exec::test::TpchPlan getQ22Plan() const;

  const dwio::common::FileFormat format_;
  const std::string q11Fraction_;
  std::unordered_map<std::string, velox::exec::test::TpchTableMetadata>
      tableMetadata_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

} // namespace facebook::velox::cudf_velox
