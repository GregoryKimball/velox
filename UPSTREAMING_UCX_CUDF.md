# Velox-cuDF and UCX upstreaming inventory

This branch documents the focused work required to move the IBM-derived
Velox-cuDF and UCX exchange stack onto current upstream Velox. It is an
inventory branch, not a build-tested integration branch.

## Scope

The inventory includes:

- `velox/experimental/cudf/**`
- `velox/experimental/ucx-exchange/**`
- Core `PlanNode`, exchange transport, exchange client, and plan-builder files
  required by those experimental modules

Unrelated differences between the parentless builder snapshot and current
Velox are intentionally excluded.

## Source provenance

- Current upstream base: `ff16498015ba7dd701fcb88b7bab69ac81897da5f`
- IBM reference:
  [`IBM/velox@ibm-research-preview-2026-09-03`](https://github.com/IBM/velox/tree/ibm-research-preview-2026-09-03),
  head `660efa0c1b78cee12ec553360a601df04325b677`
- Devavret builder base: parentless snapshot `cd3b0e76`, plus dependency-only
  commits through `712c49e`
- Exact final builder snapshot:
  [`GregoryKimball/velox@gkimball/q9-prefetch-managed-async-cuda13`](https://github.com/GregoryKimball/velox/tree/gkimball/q9-prefetch-managed-async-cuda13)

The IBM preview may be slightly newer than the IBM source used by the
parentless devavret snapshot. IBM-to-devavret attribution is therefore based
on focused tree comparison and known pull requests, not complete ancestry.

## Size

The expanded focused tree diff from current main to the pre-PR-18848 builder
base is:

- 142 files
- 11,824 additions
- 4,327 deletions
- 16,151 changed lines

The complete focused patch is stored in
`upstreaming/main-to-devavret-base-focused.patch`. It is retained as source
evidence; it is not intended to be applied wholesale because it includes
same-module drift.

## Major upstreaming workstreams

### Hybrid TableScan and split preloading

Source: [PR 18602](https://github.com/facebookincubator/velox/pull/18602).

- Makes hybrid scan the only Velox-cuDF `SplitReader`
- Enables split preloading in `CudfHiveDataSource`
- Reworks buffered input and split-reader I/O helpers
- Covers Hive and Iceberg readers and tests

PR 18602 remains open. Its final focused change set is represented on this
branch as one documentation commit.

### Adaptive UCX exchange compression

Source: [PR 18746](https://github.com/facebookincubator/velox/pull/18746).

- Column-adaptive compression and codec pipeline
- Compression cost model
- FP64 and ALP codecs
- Compression configuration and tests
- Current-main transport registration fixes

All four PR commits are retained on this branch.

### UCX transport registration and plan integration

Sources include IBM preview commits `921a0844` and `660efa0c`.

- Transport selection on partitioned-output plan nodes
- Output transport registration and preservation through plan serde
- In-memory versus UCX exchange client selection
- cuDF operator-adapter boundaries for UCX input and output

The arbitrary-output implementation from `660efa0c` is retained. Registration
work overlapping PR 18746 is recorded here rather than duplicated as an
additional conflicting implementation commit.

### Partitioned, broadcast, and arbitrary output modes

- Dynamic destination queues
- Broadcast backfill for late consumers
- Arbitrary-mode shared buffering and fair distribution
- End-of-stream and output-consumed semantics
- Queue accounting and mode-specific tests

These changes overlap transport registration and should be upstreamed with
their core plan-node dependencies.

### cuDF operator and vector evolution

The parentless devavret snapshot contains substantial focused changes beyond
the linked PRs, especially in:

- Hash join
- Groupby and aggregation selection
- Nested-loop join
- Local partition
- Velox/cuDF vector conversion
- Decimal aggregation

The focused IBM-to-devavret comparison is 96 files, 8,425 additions, and
2,888 deletions. These files require semantic review before being divided into
independent upstream PRs; tree replacement would reintroduce snapshot drift.

### Streaming aggregation

Devavret identified streaming aggregation as part of the carried stack.
Streaming support has since merged upstream as commit `796130003` in
[PR 16488](https://github.com/facebookincubator/velox/pull/16488). It is
therefore provenance, not remaining implementation work.

### Managed-memory UCX send prefetch

The tested pre-PR-18848 image adds a small Velox-side prefetch before
`tagSend`:

- One file
- 21 inserted lines
- Built and benchmarked in
  `presto-native-worker-gpu:pr412-cuda13.1-rmm2512-712c49e-d183c23-velox-ucx-prefetch-v1`

The change is retained as a separate commit on this branch.

### Batch-size configuration and memory safety

The best results came from existing query configuration, not adaptive UCX
flow control:

- `task.max-drivers-per-task`
- `cudf.batch_size_min_threshold`
- `cudf.partitioned_output_batch_rows`

At four workers, a per-query async composite using D6/100M where safe and
D1/30M for Q9, Q13, Q17, Q18, and Q21 produced a 219.049-second hot total.

## Explicitly excluded

[PR 18848](https://github.com/facebookincubator/velox/pull/18848) is not part
of the proposed stack.

- SF3000 Q9 took 930.158 seconds with the adaptive-flow implementation.
- Disabling adaptive admission while retaining its partition-output rewrite
  still exceeded six minutes.
- The pre-PR image with direct batch tuning completed Q9 in roughly 29–31
  seconds.

PR 18848 remains linked as a documented experiment and negative result, not
as upstreaming code.

## Proposed PR decomposition

1. Hybrid TableScan and split preloading (PR 18602).
2. Adaptive compression core, then FP64/ALP codecs (PR 18746).
3. Core transport registration and plan serde.
4. Broadcast and arbitrary output modes.
5. Managed-buffer send prefetch.
6. Remaining cuDF operator changes, split by operator family.
7. Queue accounting, lifecycle hardening, and focused integration tests.

Each implementation PR should be rebased independently on current main and
build-tested in the CUDA environment. This documentation branch intentionally
does not claim buildability.
