# CuDF Benchmarks

Benchmark binaries for TPC-H and TPC-DS queries with optional CuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Mode | Source |
|--------|-----------|------|--------|
| `velox_tpch_benchmark` | TPC-H (Q1-Q22) | CPU | `velox/benchmarks/tpch/` |
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | GPU | `CudfTpchBenchmark.cpp` |
| `velox_tpcds_benchmark` | TPC-DS (Q1-Q99) | CPU | `velox/benchmarks/tpcds/` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | GPU | `CudfTpcdsBenchmark.cpp` |
| `velox_cudf_filter_project_benchmark` | Filter, wide projection list, grouped aggregation | CPU and GPU | `CudfFilterProjectBenchmark.cpp` |

CPU binaries use HiveConnector. GPU binaries use CudfHiveConnector and register
cuDF GPU operator replacements.

## Build

```bash
# GPU binaries (requires CUDA)
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
cd _build/release
ninja velox_cudf_tpch_benchmark velox_cudf_tpcds_benchmark

# CPU-only binaries (no CUDA required)
ninja velox_tpch_benchmark velox_tpcds_benchmark
```

---

## Filter and Projection Benchmark

Runs a filter, a list of polynomial projections and a grouped aggregation over
synthetic data, with no files to generate. Three arms, one per invocation,
since cuDF reads the evaluator choice once when its adapter registers:

| Arm | Meaning |
|-----|---------|
| `--engine=cpu` | No adapter. The correctness reference. |
| `--engine=cudf --evaluator=ast` | A runtime AST interpreter, `compute_column`. |
| `--engine=cudf --evaluator=jit` | A kernel generated and compiled by NVRTC. |

The plan shape is set by flags: the number of columns and projections, the
polynomial degree, how many columns each projection reads, the filter
selectivity, the batch size and the total input size.

```bash
# CPU reference. Note the checksum it prints.
./velox_cudf_filter_project_benchmark --engine=cpu --num_projections=32

# The two evaluators, both verified against that checksum.
./velox_cudf_filter_project_benchmark --engine=cudf --evaluator=jit --num_projections=32 \
  --expect_checksum=<value from the cpu run> --checksum_tolerance=1e-6
./velox_cudf_filter_project_benchmark --engine=cudf --evaluator=ast --num_projections=32 \
  --expect_checksum=<value from the cpu run> --checksum_tolerance=1e-6
```

Correctness is a checksum within a tolerance rather than an exact comparison,
because NVRTC contracts `a*b+c` into an FMA and the CPU does not, so the
results differ in the last ulp by design.

To sweep, write one flag per line in a file and pass it as `--test_flags_file`.
Every combination is run, and the results are printed at the end sorted by time
and labelled with the flag values that produced them:

```bash
printf 'num_projections:8,32,128\nbatch_size:100000,1000000\n' > grid
./velox_cudf_filter_project_benchmark --engine=cudf --evaluator=jit --test_flags_file=grid
```

Sweeping the column count takes two more flags. The projections have to grow
with the schema or they stop reading all of it, which `--num_projections=0`
does by matching the column count, and `--total_bytes` derives the row count so
the input holds the same bytes at every width:

```bash
printf 'num_columns:8,32,128,512\n' > grid
./velox_cudf_filter_project_benchmark --engine=cudf --evaluator=jit --test_flags_file=grid \
  --num_projections=0 --total_bytes=$((8 << 30))
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--engine` | `cpu` or `cudf` |
| `--evaluator` | cuDF expression evaluator: `jit`, `ast`, or `default` |
| `--num_columns` | Number of DOUBLE columns, N |
| `--num_projections` | Number of polynomial projections, M. 0 matches `--num_columns` |
| `--poly_degree` | Polynomial order, evaluated in Horner form |
| `--projection_fanin` | Distinct input columns each projection reads |
| `--total_bytes` | Derive the row count so the input holds this many bytes at any column count |
| `--batch_size` | Rows per input RowVector |
| `--selectivity` | Fraction of rows passing the filter |
| `--timed_iters` | Timed iterations. Their mean is reported, excluding the first |
| `--expect_checksum` | Fail unless the result matches, within `--checksum_tolerance` |

---

## TPC-H Benchmark

### Data

Generate TPC-H parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below), or the standard `dbgen` tool and convert decimal columns to float.

### Run

```bash
# CPU - all queries
./velox_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet

# GPU (CuDF) - all queries
./velox_cudf_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet
```

---

## TPC-DS Benchmark

TPC-DS plans are loaded from pre-dumped Velox plan JSON files (serialized from Presto).

### 1. Get Plan JSON Files

Clone the plans repository:

```bash
git clone https://github.com/karthikeyann/VeloxPlans.git
# Plans are at: VeloxPlans/presto/tpcds/sf100/
```

The directory contains `Q1.json`, `Q2.json`, ..., `Q99.json`.

### 2. Get TPC-DS Data

Generate TPC-DS parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below). The data directory must have one subdirectory per table:

```
/path/to/tpcds/sf100/
  store_sales/
  customer/
  date_dim/
  item/
  ...
```

Each subdirectory contains parquet files for that table.

### 3. Run

**CPU - all queries (folly benchmark mode):**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**CPU - single query with stats:**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

**GPU (CuDF) - all queries:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**GPU (CuDF) - single query with stats:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

### TPC-DS Flags

These flags are shared by both CPU and GPU binaries:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | (required) | Root directory of TPC-DS table data |
| `--plan_path` | (required) | Directory containing Q*.json plan files |
| `--data_format` | `parquet` | Data file format |
| `--run_query_verbose` | `-1` | Run single query with stats (`-1` = run all) |
| `--num_drivers` | `4` | Number of parallel drivers |
| `--include_results` | `false` | Print query results |

### CuDF Flags (GPU binaries only)

These flags apply to `velox_cudf_tpch_benchmark` and `velox_cudf_tpcds_benchmark`:

| Flag | Default | Description |
|------|---------|-------------|
| `--cudf_chunk_read_limit` | `0` | Chunk read limit for cuDF parquet reader |
| `--cudf_pass_read_limit` | `0` | Pass read limit for cuDF parquet reader |
| `--cudf_gpu_batch_size_rows` | `100000` | GPU batch size in rows |
| `--velox_cudf_table_scan` | `true` | Use CuDF table scan |
| `--cudf_properties` | `""` | Path to a CudfConfig properties file (key=value per line). See `CudfConfig.h` for available keys |

---

## Data Generation

Both TPC-H and TPC-DS parquet data can be generated using the
[velox-testing](https://github.com/rapidsai/velox-testing) repository.
Full instructions are also available in the
[VeloxPlans TPC-DS README](https://github.com/karthikeyann/VeloxPlans/tree/main/presto/tpcds/sf100).

### Quick Start

```bash
# 1. Clone velox-testing
git clone https://github.com/rapidsai/velox-testing.git
cd velox-testing

# 2. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark_data_tools/requirements.txt

# 3. Generate TPC-DS data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpcds \
  --data-dir-path /path/to/tpcds/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats

# 4. Generate TPC-H data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpch \
  --data-dir-path /path/to/tpch/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--benchmark-type` | `tpcds` or `tpch` |
| `--data-dir-path` | Output directory for parquet files |
| `--scale-factor` | Scale factor (e.g. `1`, `10`, `100`) |
| `--convert-decimals-to-floats` | Convert decimal columns to double (recommended for Velox) |

The output directory will contain one subdirectory per table, each with `.parquet` files.
For a quick sanity check, use `--scale-factor 1` first.
