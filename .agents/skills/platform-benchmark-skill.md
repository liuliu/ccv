# Platform Benchmark Skill

## Purpose

Use this workflow for MFA / MPS benchmarking when kernel results are sensitive to:

- thermal state
- run ordering
- selector cliffs
- stale benchmark-only heuristics
- correctness bugs that can look like speedups

The point is not just to produce numbers. The point is to produce numbers that are:

- correctness-gated
- interpretable
- comparable across shapes
- stable enough to guide a selector or a kernel change

This matters especially for attention kernels, where a broken path can look artificially fast because work silently collapses or outputs go to zero.

## Core Principle

Always establish correctness before interpreting performance.

A performance result from a path that has not been correctness-cleared is not a benchmark result. It is only a hypothesis. In this workspace, several “fast” paths later turned out to be invalid because:

- gradients collapsed in magnitude
- a compare harness was reading the wrong layout
- a benchmark selector did not match production
- a path was only good on one shape because of a stale heuristic

The workflow below is designed to prevent those mistakes.

## Benchmark Structure

Use three layers of measurement, in this order:

1. Direct correctness harness
2. Isolated kernel benchmark
3. End-to-end compare benchmark

Each layer answers a different question:

1. Direct correctness harness:
   - Is the kernel numerically correct on a guarded surface?
2. Isolated kernel benchmark:
   - If I force one policy or one kernel shape, what actually changed?
3. End-to-end compare benchmark:
   - With the real selector and full stack, how does this compare against alternatives such as MPSGraph?

Do not skip layer 1 and jump straight to layer 3.

## Correctness-First Methodology

### Why correctness comes first

Performance work often changes:

- tile sizes
- traversal order
- launch order
- staging strategy
- selector rules
- mixed-precision policy

Any of those can create a path that looks faster only because it is wrong. A correctness harness protects against that.

### What a good correctness harness should do

A good correctness harness should compare against a trusted reference on a shape that is known to expose the target failure mode.

In this workspace, for dense backward, the guarded surface was:

- `B=1`
- `R=C=1536`
- `Hq=Hk=24`
- `D=128`
- `fp16`
- `lowPrecisionIntermediates=1`

That shape mattered because it exposed the original low-precision staged backward failure.

### Metrics to check

Do not look at only one metric.

You should look at:

- max absolute difference
- max relative difference
- reference output max absolute magnitude
- test output max absolute magnitude

Why each one matters:

- max absolute difference:
  - catches large local errors directly
- max relative difference:
  - helps when tensor magnitudes vary a lot across shapes or outputs
- reference/test max absolute magnitude:
  - catches collapse-to-zero or explosion, which can look deceptively “fine” under some relative metrics

In this workspace, some failures were only obvious once the harness reported that:

- reference `dV max_abs` was normal
- test `dV max_abs` was `0`

That is a different class of failure from a modest `max_abs_diff`.

### Practical rule for interpreting correctness output

Treat these as separate quantities:

- `max_abs_diff`
- `ref_max_abs`
- `test_max_abs`

Do not confuse tensor magnitude with tensor difference.

If the harness prints overloaded `max_abs` labels, fix or mentally separate them before making kernel decisions.

### Recommended direct correctness tools in this workspace

Dense backward:

```sh
cd /Users/liu/workspace/ccv/bin/mfa
./attention_backward_compare_probe
```

Focused integration:

```sh
cd /Users/liu/workspace/ccv/test/int/nnc
./mpsblas.tests "scaled dot product attention gradient with mps"
```

For quantized backward work, use the matching int8 compare probe and the focused quantized SDPA tests.

### Why the direct harness precedes performance harnesses

Because once the correctness harness is in place, every speed experiment becomes cheaper to evaluate:

- if the path is wrong, stop immediately
- if the path is right, then benchmark it

Without that order, you waste time optimizing invalid kernels.

## Isolated Kernel Benchmark Methodology

### Why isolated benches matter

End-to-end compare benches mix together:

- selector policy
- query cost
- key/value cost
- staging policy
- dispatch geometry
- integration overhead

That makes them poor tools for root-causing a selector cliff.

An isolated bench lets you force:

- bypass vs staged
- query vs key/value behavior
- specific block sizes
- specific simdgroup counts

Use it whenever you need to answer “what actually changed?”

### Recommended isolated bench in this workspace

```sh
cd /Users/liu/workspace/ccv/bin/mfa
./na_attention_backward_bench ...
```

This bench is useful for:

- locating the crossover between bypass and staged
- measuring query and key/value separately
- checking whether a selector cliff is real or benchmark-only

### Example: force bypass vs staged

```sh
cd /Users/liu/workspace/ccv/bin/mfa

./na_attention_backward_bench 2048 2048 128 1 32 32 3 6 16 64 64 16 64 64 6 6 1 1 0 1
./na_attention_backward_bench 2048 2048 128 1 32 32 3 6 16 64 64 16 64 64 6 6 0 0 0 1
```

The final flags are:

- `queryBypassTG`
- `keyvalueBypassTG`
- `computeDCustom`
- `lowPrecisionIntermediates`

### How to read isolated bench output

Pay attention to:

- `query median_ms`
- `keyvalue median_ms`
- total `backward median_ms`

This lets you answer questions like:

- Is the regression query-side or key/value-side?
- Did staged memory actually help the intended kernel?
- Did a selector threshold move for the right reason?

### Why isolated benches should still follow correctness

Even in an isolated bench, a path can benchmark “well” while being invalid. Always use the direct correctness harness first for the same forced path if that path is new or suspicious.

## End-to-End Compare Benchmark Methodology

### Purpose

Use the end-to-end compare bench to compare:

- dense NA backward
- int8 NA backward
- MPSGraph backward

under the real selector and pipeline setup.

In this workspace, the main tool is:

```sh
cd /Users/liu/workspace/ccv/bin/mfa
./sdpa_backward_compare_bench ...
```

### Why this bench needs special care

This bench is sensitive to:

- thermal state
- process order
- stale benchmark-only selector logic

If you run one long in-process sweep in the wrong order, you can create fake cliffs or hide real ones.

### First rule: verify the bench selector matches production

Before trusting a compare bench, confirm that its selector logic matches production for the shape family you care about.

This is important because a compare bench can silently lag behind production policy. In this workspace, a `3072²` dense backward cliff was mostly due to a stale compare-bench bypass threshold, not a real production cliff.

So before a major sweep:

1. inspect the bench selector
2. inspect the production selector
3. reconcile any mismatch

### Recommended run order

Run one shape per process, descending from hottest to coolest:

1. `16384`
2. `8192`
3. `6144`
4. `4096`
5. `3072`
6. `2048`
7. `1536`

Then rerun one anchor shape, typically `4096`, at the end.

Why descending?

- the largest cases heat the machine the most
- if you run them late, they poison every result after them
- running them first gives them the cleanest thermal state

Why the anchor rerun?

- it detects drift
- if the first and last anchor match closely, the sweep is thermally believable

### Recommended cooldown gaps

Cooldown does not need to be perfect. It needs to be systematic.

Example cadence:

- after `16384`: `sleep 20`
- after `8192`: `sleep 15`
- after `6144`: `sleep 10`
- after `4096`: `sleep 8`
- after `3072`: `sleep 6`
- after `2048`: `sleep 5`

### Recommended commands

```sh
cd /Users/liu/workspace/ccv/bin/mfa

./sdpa_backward_compare_bench 16384 16384 128 1 32 32 3 6
sleep 20
./sdpa_backward_compare_bench 8192 8192 128 1 32 32 3 8
sleep 15
./sdpa_backward_compare_bench 6144 6144 128 1 32 32 3 10
sleep 10
./sdpa_backward_compare_bench 4096 4096 128 1 32 32 3 12
sleep 8
./sdpa_backward_compare_bench 3072 3072 128 1 32 32 3 15
sleep 6
./sdpa_backward_compare_bench 2048 2048 128 1 32 32 3 20
sleep 5
./sdpa_backward_compare_bench 1536 1536 128 1 24 24 3 20
sleep 8
./sdpa_backward_compare_bench 4096 4096 128 1 32 32 3 12
```

### Which metric to trust

Prefer:

- `best3_avg_ms`

Use with caution:

- `avg_ms`
- `median_ms`

Why `best3_avg_ms`?

- it reduces noise from occasional scheduler or thermal hiccups
- it is more stable for cross-case comparisons in this workspace

### How to interpret the repeated anchor

If the first and second anchor runs are close, the sweep is stable enough to trust.

If they diverge materially:

- reduce the number of timed iterations on large cases
- add longer cooldowns
- rerun the sweep

## Decision Workflow

When evaluating a kernel or selector change, use this order:

1. Patch the kernel or selector.
2. Rebuild the relevant probe and test binaries.
3. Run direct correctness harness on the guarded surface.
4. Run focused integration test.
5. Use isolated bench to understand the local effect.
6. Use end-to-end compare bench to measure the production-facing effect.

This order prevents “fast but wrong” paths from contaminating the benchmark story.

## Lessons From This Workspace

- A stale compare-bench selector can create a fake performance cliff.
- A path that looks faster can simply be numerically wrong.
- `max_abs_diff` alone is not enough; always check tensor magnitude too.
- For thermally sensitive attention sweeps, one-case-per-process descending order is much more trustworthy than a simple in-process sweep.
- Once a guarded correctness probe exists, performance iteration becomes much faster and more reliable.
