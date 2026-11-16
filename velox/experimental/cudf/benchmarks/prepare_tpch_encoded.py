#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates decimal TPC-H Parquet laid out for velox_cudf_tpch_benchmark --encoded.

The encoded column cache encodes rows in file order and prunes per segment by
min/max, so the input order decides the cache. This writes lineitem sorted by
l_shipdate and orders by o_orderdate, which clusters the date filters, and
writes them with 4M-row row groups and 1 MiB data pages: the cuDF Parquet
reader decodes in parallel across pages, and DuckDB writes one or two pages
per column per row group. Money columns stay DECIMAL(15,2).

Output: <out>/<table>/<table>.parquet, the layout --data_path expects.

Requires tpchgen-cli, duckdb and pyarrow. The published SF100 results used
tpchgen-cli 4.0.0 and duckdb 1.4.5.

Usage: prepare_tpch_encoded.py --scale-factor 100 --out /data/tpch_sf100
"""

import argparse
import pathlib
import shutil
import subprocess

import duckdb
import pyarrow.parquet as pq

TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
SORT_KEYS = {"lineitem": "l_shipdate", "orders": "o_orderdate"}
ROW_GROUP_ROWS = 4 << 20
PAGE_BYTES = 1 << 20


def rewrite_pages(source, target):
    reader = pq.ParquetFile(source)
    writer = pq.ParquetWriter(target, reader.schema_arrow, compression="snappy", data_page_size=PAGE_BYTES,
                              store_decimal_as_integer=True)
    for group in range(reader.num_row_groups):
        rows = reader.read_row_group(group)
        writer.write_table(rows, row_group_size=rows.num_rows)
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    generated = args.out / "_generated"
    generated.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tpchgen-cli", "parquet", "-s", str(args.scale_factor), "--output-dir", str(generated)],
                   check=True)
    con = duckdb.connect()
    con.sql(f"set temp_directory = '{args.out / '_duckdb_tmp'}'")
    for table in TABLES:
        source = generated / f"{table}.parquet"
        target_dir = args.out / table
        target_dir.mkdir(exist_ok=True)
        target = target_dir / f"{table}.parquet"
        if table not in SORT_KEYS:
            shutil.move(source, target)
            continue
        sorted_file = generated / f"{table}.sorted.parquet"
        con.sql(f"copy (select * from '{source}' order by {SORT_KEYS[table]}) "
                f"to '{sorted_file}' (format parquet, row_group_size {ROW_GROUP_ROWS})")
        rewrite_pages(sorted_file, target)
        sorted_file.unlink()
        source.unlink()
    shutil.rmtree(generated)
    shutil.rmtree(args.out / "_duckdb_tmp", ignore_errors=True)
    print(f"Wrote {', '.join(TABLES)} to {args.out}")


if __name__ == "__main__":
    main()
