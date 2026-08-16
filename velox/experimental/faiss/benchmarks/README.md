# FAISS candidate-retrieval benchmark

`velox_candidate_retrieval_benchmark` is a Folly benchmark for the complete
candidate-retrieval query: Parquet scan, filtering, cluster assignment, index
build or artifact load, and search.

The default workload is intentionally small and self-cleaning:

```text
10,000 candidates; 1,000 queries; 128 dimensions; topK 10; 8 clusters
```

Run all cases:

```bash
./velox_candidate_retrieval_benchmark
```

Select cases with Folly's normal benchmark filter:

```bash
./velox_candidate_retrieval_benchmark \
  --bm_regex='CandidateRetrieval/CPU/HNSW'
```

The CPU cases keep scan, filter, assignment, index work, and search on CPU.
GPU builds additionally register GPU Flat (cuVS brute force) and CAGRA cases
that keep those stages on GPU.

The CAGRA load case is named `LoadCpuGraphArtifact` because FAISS exports
`GpuIndexCagra` into its CPU serialization carrier, `IndexHNSWCagra`, writes
that carrier, and copies it into a new `GpuIndexCagra` after loading. The graph
was built by CAGRA; this is not a separate CPU HNSW build.

Use `--candidates`, `--queries`, `--dimension`, `--topK`, and `--clusters` to
change the shape. Data is generated in batches and written directly to
Parquet. Without `--data_directory`, inputs and artifacts live in a scoped
temporary directory and are deleted when the process exits.

For large runs, always place generated data on RAID:

```bash
./velox_candidate_retrieval_benchmark \
  --data_directory=/raid/$USER/faiss-candidate-benchmark \
  --candidates=1000000 \
  --bm_regex='CandidateRetrieval/GPU/CAGRA/Build'
```

The benchmark regenerates inputs and artifacts on each invocation. A supplied
`--data_directory` is retained after exit; the default temporary directory is
deleted automatically.

Folly controls repetition and output formatting. Operators publish phase
timings through Velox runtime statistics for external consumers, and each
measured benchmark invocation has a case-named NVTX range.
