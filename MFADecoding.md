# MFA Decoding Attention Plan

## Context

The current MFA neural-accelerator SDPA implementations, `NAAttention` and
`NAInt8Attention`, are not well shaped for decode-time attention where the query
length is 1 and the key/value length is large. The main issue is parallelism:
the forward kernels parallelize over query rows, heads, and batch, but they do
not split the key/value sequence dimension across threadgroups.

This matters most for `R = 1`. There is only one query-row tile per head and
batch, so the long `C` dimension is traversed serially inside a small amount of
active work.

## Local MFA Diagnosis

The forward `NAAttention` grid is row/head/batch based:

- `lib/nnc/mfa/kernels/NAAttentionKernel.cpp`
  - `NAAttentionKernel::threadgroupsPerGrid(...)`
  - forward grid uses `ceil(R / (blockParallelization * executionSIMDGroups))`
    row groups, heads, and batch.
  - source generation maps `sgid` back into row groups.

For `R = 1`, this means only `sgid = 0` has useful row work. The other SIMD
groups in the threadgroup return after the row-bound check. The long `C`
dimension is still processed by the one active row path.

`NAInt8Attention` has the same high-level grid shape:

- `lib/nnc/mfa/kernels/NAInt8AttentionKernel.cpp`
  - `NAInt8AttentionKernel::threadgroupsPerGrid(...)`
  - forward grid is also based on query-row groups, heads, and batch.

The int8 path also has a centered-value detail that affects the split design.
It accumulates attention over centered values and adds `V_mean` only at final
output store time. A split-KV decode path should therefore combine centered
partial outputs first, then add `V_mean` once in the combine kernel.

## Relevant SmallM Pattern

`NAMatMulSmallM` is the closest local design precedent. It does not solve
attention directly, but it shows the right style for this class of issue:

- specialize a small-M case rather than perturbing the general matmul path;
- create more parallel work along the large reduction dimension;
- write partial results to scratch;
- run a second small reduction kernel.

Relevant files:

- `lib/nnc/mfa/kernels/NAMatMulSmallMDescriptor.cpp`
  - `splitK()` chooses when to split the large K dimension.
  - `scratchOffsets()` reserves scratch for partials.
- `lib/nnc/mfa/kernels/NAMatMulSmallMKernel.cpp`
  - `matmul_small_m_block_view` writes packed partials.
  - `reduce_diagonal` reduces split and packed-lane partials.
- `lib/nnc/mfa/ccv_nnc_mfa_gemm.cpp`
  - `_ccv_nnc_mfa_use_na_matmul_small_m(...)` gates the specialized path.

The decode attention path should follow this pattern: a narrow specialized path
with explicit scratch and a combine step, gated conservatively.

## Flash-Decoding Research Summary

Flash-Decoding addresses the same decode-time problem. With query length 1,
ordinary FlashAttention has too little query-axis parallelism to fill the GPU.
Flash-Decoding adds sequence-parallelism over the KV dimension:

1. Split the key/value sequence into chunks.
2. Run attention independently for each chunk, producing a partial output and a
   per-split log-sum-exp value.
3. Combine partial outputs using the per-split log-sum-exp values.

The combine is:

```text
m = max_i(lse_i)
lse = log(sum_i(exp(lse_i - m))) + m
O = sum_i(exp(lse_i - lse) * O_i)
```

Our MFA kernels currently store LSE in log2 units, because the online softmax
uses `fast::exp2` and stores `L = cM + log2(cL)`. The MFA combine should
therefore use the log2 equivalent:

```text
m = max_i(lse_i)
lse = log2(sum_i(exp2(lse_i - m))) + m
O = sum_i(exp2(lse_i - lse) * O_i)
```

Our CUDA FlashAttention wrapper already enables this concept for `R == 1`:

- `lib/nnc/cmd/scaled_dot_product_attention/gpu/ccv_nnc_scaled_dot_product_attention_flash_attn.cu`
  - chooses `num_splits` only for non-varlen `R == 1`.
- `lib/nnc/gpu/3rdparty/flash_attn/flash_api.cu`
  - `num_splits_heuristic(...)` chooses split count from occupancy efficiency,
    then picks the smallest split count within 85% of the best efficiency.
- `lib/nnc/gpu/3rdparty/flash_attn/src/flash_fwd_launch_template.h`
  - launches split-KV attention, then launches a combine kernel when
    `num_splits > 1`.
- `lib/nnc/gpu/3rdparty/flash_attn/src/flash_fwd_kernel.h`
  - `compute_attn_1rowblock_splitkv(...)` writes per-split output and LSE.
  - `combine_attn_seqk_parallel(...)` performs the LSE-weighted merge.

External references:

- https://princeton-nlp.github.io/flash-decoding/
- https://pytorch.org/blog/flash-decoding/

## Proposed MFA Design

Add decode-specific forward kernels for both `NAAttention` and
`NAInt8Attention`.

Dense `NAAttention` now uses the implemented gate documented below. For
`NAInt8Attention`, use the same conservative shape policy initially:

- small `R`, starting with `R <= 64` only if benchmarks hold for the quantized
  path
- non-varlen
- sufficiently large `C`
- neural accelerators available
- split count greater than 1 from a conservative heuristic
- fall back to the existing kernels otherwise

The partial decode kernel should parallelize over KV splits:

```text
split_id = split_group * executionSIMDGroups + sgid
```

This is the direct fix for the current `R = 1` underutilization. SIMD groups in
one threadgroup work on different KV chunks instead of mostly returning due to
missing query rows.

Each split writes:

```text
partial_O[batch, head, split, D]
partial_LSE[batch, head, split]
```

Then a combine kernel writes final `O` and optional final `L`.

For `NAAttention`:

- Reuse the existing online softmax structure.
- Restrict the C loop to the split's block range.
- Write normalized per-split output and per-split log2 LSE to scratch.
- Combine with log2 LSE weights.

For `NAInt8Attention`:

- Reuse the existing quantize-Q/K/V and `V_mean` preparation.
- The split attention kernel should write centered partial output, before adding
  `V_mean`.
- The combine kernel should merge centered partial outputs, then add `V_mean`
  once when writing final `O`.

## Split Boundaries

Split by C blocks, not raw elements:

```text
num_c_blocks = ceil(C / blockC)
blocks_per_split = ceil(num_c_blocks / num_splits)
c_begin = split_id * blocks_per_split * blockC
c_end = min(C, c_begin + blocks_per_split * blockC)
```

This keeps most split work aligned to the existing traversal block and makes
empty split handling simple. Empty or fully masked splits should write:

```text
partial_LSE = -inf
partial_O = 0
```

## Heuristic

CUDA FlashAttention uses SM count and wave efficiency. MFA does not have the
same simple SM-count signal exposed for the neural accelerator path, so the
initial heuristic should be conservative and benchmark driven.

A reasonable first version:

- do not split below a `C` threshold such as 1024 or 2048;
- compute `num_c_blocks = ceil(C / blockC)`;
- cap `num_splits` to a small value such as 8 or 16 initially;
- choose the smallest split count that gives enough active work for
  `batch * Hq * num_splits`;
- require each split to own enough C blocks to amortize scratch writes and the
  combine kernel.

The exact thresholds should be tuned with `R = 1` decode benchmarks across
`C`, `D`, `Hq`, `Hk`, batch size, and quantized/non-quantized paths.

## Staging

1. Implement `NAAttention` decode split-KV for unmasked non-varlen `R = 1`.
   - Done for dense forward.
2. Add the log2-LSE combine kernel.
   - Done for dense forward.
3. Add correctness tests against CPU reference for causal and non-causal decode.
   - Done for `B=1, R=1, C=2112, H=8, D=256`, half inputs.
4. Add `NAInt8Attention` decode split-KV using centered partial outputs.
5. Add int8 correctness tests.
6. Extend to masked decode after the base path is correct and benchmarked.
7. Tune split heuristics with local benchmarks.

## NAAttention Implementation Status

Dense `NAAttention` now has a Flash-Decoding-style split-KV forward path for
small query lengths.

Selector:

- forward dense attention only;
- unmasked and non-varlen only;
- `R <= 4 * blockR` (`blockR = 16`, so this currently admits `R <= 64`);
- `C >= 2048` for `R = 1`, `C >= 4096` for `R > 1`;
- `batch * Hq * ceil(R / blockR) <= 128`;
- `splitKV = min(executionSIMDGroups, ceil(C / blockC))`.

Implementation detail: `NAAttentionDescriptor::splitKV(...)` makes the selection
while building the `NAAttentionKernelDescriptor`. After that, the selected
`splitKV` value lives on the kernel descriptor / kernel object and is reused for
function constants, scratch sizing, and dispatch geometry. A value `<= 1`
preserves the existing non-split kernel source.

Kernel shape:

```text
grid.x = Hq * ceil(splitKV / executionSIMDGroups)
grid.y = ceil(R / blockR)
grid.z = batch
split_id = split_group * executionSIMDGroups + sgid
```

Scratch layout:

```text
partial_O[batch, head, split, row, D]
partial_LSE[batch, head, split, row]
```

The split kernel processes one row tile and one KV split per SIMD group. It
writes normalized per-split output and log2 LSE for each valid row. The combine
kernel reduces per `(batch, head, row, d)` with log2-LSE weights and writes final
`O` and final `L`.

Causal small-R support is row-specific:

```text
column <= row + C - R
```

This bound is applied inside each split before the per-split row max/sum.

Current intentional exclusions:

- masked decode stays on the existing path;
- varlen stays on the existing path;
- `NAInt8Attention` still needs the same split/combine treatment, with centered
  partial outputs and one final `V_mean` add.

Arbitrary `C` is supported. The split kernel maps `C` to `ceil(C / blockC)`
tiles and the final split handles a trailing partial traversal block by masking
invalid QK columns, packing the valid probabilities at the end of the
threadgroup `P` tile, and using the same dynamic-extent PV multiply pattern as
the existing non-split tail path.

## Validation

Correctness command:

```text
cd test/int/nnc
make mpsblas.tests
./mpsblas.tests "scaled dot product attention with NA mps splitKV decode"
```

Result:

```text
[PASS] scaled dot product attention with NA mps splitKV decode
all test case(s) passed, congratulations!
```

Byte-identity source check:

```text
Current splitKV=1 generated NAAttention forward source is byte-identical to HEAD
for both causal and non-causal source generation.
```

The test covers half-input causal and non-causal decode for:

```text
R=1, C=2112, H=8, D=256
R=1, C=2113, H=8, D=256
R=2, C=4096, H=8, D=128
R=7, C=4096, H=8, D=128
R=7, C=4097, H=8, D=128
R=16, C=4096, H=8, D=128
R=16, C=4096, Hq=32, Hk=8, D=128
R=16, C=4097, Hq=32, Hk=8, D=128
R=32, C=4096, H=4, D=128
```

## Current NAAttention Results

Benchmark commands:

```text
cd bin/nnc
make sdpa_bench
./sdpa_bench --na --small-r-grid
./sdpa_bench --generic --small-r-grid
./sdpa_bench --na --small-r-causal-grid
./sdpa_bench --generic --small-r-causal-grid
```

Shape:

```text
B=1, Hq=32, Hk=32, R in {1,2,3,5,7,8,15,16,32,64}
D in {64,128,256}, C in {2048,4096,8192,16384}
```

The tables below show selected final-run values for `R={1,16,32,64}`. Ratios
are `generic / default`, so values above `1.0x` favor the NA decode path. These
sub-millisecond timings are noisy, but the long-`C` trend is stable.

Plain:

| D | C | R | default | generic | generic/default |
|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 1 | 0.243 ms | 0.410 ms | 1.69x |
| 64 | 4096 | 16 | 0.219 ms | 0.427 ms | 1.95x |
| 64 | 4096 | 32 | 0.276 ms | 0.592 ms | 2.14x |
| 64 | 4096 | 64 | 0.391 ms | 0.491 ms | 1.26x |
| 64 | 16384 | 1 | 0.576 ms | 1.316 ms | 2.28x |
| 64 | 16384 | 16 | 0.624 ms | 1.379 ms | 2.21x |
| 64 | 16384 | 32 | 0.638 ms | 1.339 ms | 2.10x |
| 64 | 16384 | 64 | 0.885 ms | 1.389 ms | 1.57x |
| 128 | 4096 | 1 | 0.477 ms | 0.741 ms | 1.55x |
| 128 | 4096 | 16 | 0.493 ms | 0.830 ms | 1.68x |
| 128 | 4096 | 32 | 0.519 ms | 0.864 ms | 1.66x |
| 128 | 4096 | 64 | 0.572 ms | 0.935 ms | 1.63x |
| 128 | 16384 | 1 | 0.922 ms | 2.163 ms | 2.35x |
| 128 | 16384 | 16 | 0.963 ms | 2.191 ms | 2.27x |
| 128 | 16384 | 32 | 0.930 ms | 2.216 ms | 2.38x |
| 128 | 16384 | 64 | 1.165 ms | 2.388 ms | 2.05x |
| 256 | 4096 | 1 | 0.620 ms | 1.318 ms | 2.13x |
| 256 | 4096 | 16 | 0.614 ms | 1.610 ms | 2.62x |
| 256 | 4096 | 32 | 0.758 ms | 1.615 ms | 2.13x |
| 256 | 4096 | 64 | 1.007 ms | 1.702 ms | 1.69x |
| 256 | 16384 | 1 | 1.448 ms | 4.553 ms | 3.14x |
| 256 | 16384 | 16 | 1.460 ms | 5.378 ms | 3.68x |
| 256 | 16384 | 32 | 1.785 ms | 5.393 ms | 3.02x |
| 256 | 16384 | 64 | 3.401 ms | 5.910 ms | 1.74x |

Causal:

| D | C | R | default | generic | generic/default |
|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 1 | 0.226 ms | 0.488 ms | 2.16x |
| 64 | 4096 | 16 | 0.274 ms | 0.454 ms | 1.66x |
| 64 | 4096 | 32 | 0.244 ms | 0.574 ms | 2.35x |
| 64 | 4096 | 64 | 0.319 ms | 0.679 ms | 2.13x |
| 64 | 16384 | 1 | 0.642 ms | 1.259 ms | 1.96x |
| 64 | 16384 | 16 | 0.546 ms | 1.283 ms | 2.35x |
| 64 | 16384 | 32 | 0.629 ms | 1.327 ms | 2.11x |
| 64 | 16384 | 64 | 0.685 ms | 1.397 ms | 2.04x |
| 128 | 4096 | 1 | 0.370 ms | 0.897 ms | 2.42x |
| 128 | 4096 | 16 | 0.437 ms | 0.804 ms | 1.84x |
| 128 | 4096 | 32 | 0.584 ms | 0.908 ms | 1.55x |
| 128 | 4096 | 64 | 0.597 ms | 0.936 ms | 1.57x |
| 128 | 16384 | 1 | 0.867 ms | 2.167 ms | 2.50x |
| 128 | 16384 | 16 | 1.016 ms | 2.272 ms | 2.24x |
| 128 | 16384 | 32 | 1.015 ms | 2.272 ms | 2.24x |
| 128 | 16384 | 64 | 1.196 ms | 2.452 ms | 2.05x |
| 256 | 4096 | 1 | 0.721 ms | 1.492 ms | 2.07x |
| 256 | 4096 | 16 | 0.554 ms | 1.574 ms | 2.84x |
| 256 | 4096 | 32 | 0.771 ms | 1.591 ms | 2.06x |
| 256 | 4096 | 64 | 1.213 ms | 1.729 ms | 1.43x |
| 256 | 16384 | 1 | 1.496 ms | 4.614 ms | 3.08x |
| 256 | 16384 | 16 | 1.420 ms | 5.442 ms | 3.83x |
| 256 | 16384 | 32 | 1.761 ms | 5.433 ms | 3.09x |
| 256 | 16384 | 64 | 4.184 ms | 5.967 ms | 1.43x |

Takeaways:

- The original `R=2..16` causal plateau is gone. For example, `D=256,C=16384`
  moved from roughly `4 ms` for `R=2..16` to roughly `1.3-1.5 ms`.
- `R <= 16` is the strongest production target and is consistently better than
  the generic baseline in the long-`C` cases.
- `R=32` still looks useful, especially for `D=128/256`.
- `R=64` is still usually better than generic in the selected final grid, but
  the margin narrows, especially for `D=256` causal. Treat `R=64` as the edge of
  the current conservative selector, not as proof that larger `R` should split.
- `C=2048` remains a threshold/overhead case. The selector only enables `R>1`
  split at `C >= 4096`.

## Arbitrary C Tail Results

The old splitKV selector required `C % blockC == 0`, so odd `C` used the
existing non-split NA attention path. The current splitKV kernel uses
`ceil(C / blockC)` and a dynamic-extent PV multiply for the final partial tile.

No-drop divisible smoke cases:

| Shape | before tail support | current | note |
|---|---:|---:|---|
| R=16 C=4096 Hq=32 Hk=32 D=128 plain | 0.623 ms | 0.622 ms | unchanged |
| R=16 C=8192 Hq=32 Hk=8 D=256 plain | 0.776 ms | 0.725 ms | unchanged/no drop |
| R=64 C=4096 Hq=32 Hk=8 D=64 plain | 0.700 ms | 0.662 ms | unchanged/no drop |

Odd-`C` comparison. "old non-split" was measured by temporarily restoring the
old divisibility gate in the selector, rebuilding `sdpa_bench`, then restoring
the final selector.

| Shape | mode | current splitKV | old non-split NA | generic/MPS |
|---|---|---:|---:|---:|
| R=1 C=2113 Hq=8 Hk=8 D=256 | plain | 0.595 ms | 0.990 ms | 1.280 ms |
| R=1 C=2113 Hq=8 Hk=8 D=256 | causal | 0.502 ms | 0.738 ms | 0.902 ms |
| R=7 C=4097 Hq=8 Hk=8 D=128 | plain | 0.551 ms | 0.947 ms | 1.077 ms |
| R=7 C=4097 Hq=8 Hk=8 D=128 | causal | 0.418 ms | 0.726 ms | 0.781 ms |
| R=16 C=4097 Hq=32 Hk=32 D=128 | plain | 0.723 ms | 1.073 ms | 0.969 ms |
| R=16 C=4097 Hq=32 Hk=32 D=128 | causal | 0.639 ms | 0.863 ms | 0.811 ms |
| R=16 C=4097 Hq=32 Hk=8 D=128 | plain | 0.730 ms | 0.873 ms | 0.994 ms |
| R=16 C=4097 Hq=32 Hk=8 D=128 | causal | 0.431 ms | 0.777 ms | 0.748 ms |
| R=16 C=8193 Hq=32 Hk=8 D=256 | plain | 1.087 ms | 2.412 ms | 2.818 ms |
| R=16 C=8193 Hq=32 Hk=8 D=256 | causal | 0.589 ms | 1.917 ms | 2.780 ms |

The odd-`C` path has a real final-tile overhead compared with exact multiples,
but it is still faster than both the old non-split NA path and the generic/MPS
path on these decode shapes. Masked / triangular-mask timings are intentionally
excluded from this table because the splitKV selector does not enable masked
decode yet.

## R=1 Non-NAX AttentionR1 Probe

The llama.cpp Metal `flash_attn_ext_vec` path is the right reference shape for
`R=1`. The important detail is that the active path is not only the
single-workgroup direct-output mode. The current llama.cpp code keeps a direct
`NWG=1` mode in the source, but it is disabled; for real small-query decode it
uses `NWG=32` workgroups over the KV axis, writes partial `O/S/M`, then runs a
small reduce kernel. That is what prevents the dramatic C scaling. The direct
no-temp mode is still useful as a small-C option, but it cannot flatten
`C=2048 -> 8192` by itself.

Local probe:

- `bin/mfa/na_attention_decode1_bench.cpp`
- dense fp16 only, `R=1`, `D <= 256`, `D % 32 == 0`, `Hq % Hk == 0`
- no NAX
- writes O only; it does not produce LSE
- `nwg=1`: direct output, no global temp
- `nwg>1`: llama-style partial `O/S/M` global temp plus reduce

Because this is a standalone Metal probe, the table uses wall-time median from
the probe and the existing `sdpa_bench` wall-time average for current
`NAAttention`. The probe also reports GPU timestamps, but wall time is the more
honest comparison to `sdpa_bench`.

Targeted `Hq/Hk=10` shape, `B=1,R=1,Hq=40,Hk=4`:

| D | C | current NA plain | current NA causal | generic plain | generic causal | AttentionR1 direct median | AttentionR1 split/reduce median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2048 | 0.448 ms | 0.435 ms | 0.867 ms | 0.573 ms | 0.419 ms | 0.423 ms |
| 128 | 8192 | 0.582 ms | 0.624 ms | 1.372 ms | 1.001 ms | 0.524 ms | 0.450 ms |
| 256 | 2048 | 0.544 ms | 0.441 ms | 1.316 ms | 0.789 ms | 0.335 ms | 0.497 ms |
| 256 | 8192 | 0.845 ms | 0.644 ms | 2.293 ms | 2.218 ms | 0.809 ms | 0.569 ms |

Targeted `Hq/Hk=20` shape, `B=1,R=1,Hq=40,Hk=2`:

| D | C | current NA plain | current NA causal | generic plain | generic causal | AttentionR1 direct median | AttentionR1 split/reduce median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2048 | 0.461 ms | 0.372 ms | 0.793 ms | 0.607 ms | 0.318 ms | 0.371 ms |
| 128 | 8192 | 0.585 ms | 0.418 ms | 1.443 ms | 0.991 ms | 0.613 ms | 0.434 ms |
| 256 | 2048 | 0.526 ms | 0.335 ms | 0.837 ms | 0.698 ms | 0.677 ms | 0.493 ms |
| 256 | 8192 | 0.962 ms | 0.648 ms | 2.298 ms | 2.145 ms | 0.679 ms | 0.536 ms |

Immediate takeaways:

- The user-observed llama.cpp behavior comes from KV-axis workgroup splitting.
  The direct no-temp path alone still grows materially with C.
- `AttentionR1*` should be structured like `Gemv*`: a small dedicated kernel
  family with a descriptor-level selector and source-generation constants for
  kernel shape only.
- Production should support both `nwg=1` direct and `nwg=32` split/reduce.
  A reasonable first selector is:
  - `R == 1`
  - dense half `NAAttention` only; do not route `NAInt8Attention` here yet
  - no external mask and no requested L output
  - `D in {128,256}` initially, with `D % 32 == 0`
  - `Hq % Hk == 0`
  - use direct mode when the measured wall-time selector says it wins; otherwise
    use split/reduce with `NWG=32` and tune `NSG` by `D/C`
- The first production version should keep `AttentionR1Descriptor::splitKV`-like
  selection outside the cached kernel object. Source properties belong in the
  kernel descriptor; runtime shape values such as `C`, strides, scale, `Hq`,
  and `Hk` should be function constants or encode-time args according to the
  existing MFA cache rule.
- Because this path writes O only, it is an inference/decode specialization. If
  callers request LSE, fall back to current `NAAttention` until the reduce kernel
  writes compatible LSE.

## Production AttentionR1 First Cut

Production now has a dense `AttentionR1*` kernel family:

- `AttentionR1Descriptor.{hpp,cpp}`
- `AttentionR1Kernel.{hpp,cpp}`
- dispatch hook in `ccv_nnc_mfa_encode_attention(...)`

The current selector is intentionally simple and does not branch on neural
accelerator availability or head ratio:

- `R == 1`
- fp16 or bf16 dense attention; no `NAInt8Attention`
- no external mask and no varlen
- `D in {128,256}`
- `C > 0`
- `Hq % Hk == 0`

The R1 selector does not require neural accelerators. The plain Metal R1 kernel
is preferred for supported `R=1` dense decode regardless of whether NA is
available or disabled.
For `R=1`, causal and non-causal no-mask decode are equivalent and use the same
R1 source path.

R1 follows the usual MFA style: `alpha` is a function constant on the pipeline,
not a per-dispatch args buffer. `C` is loaded at encode time through a `loadC`
buffer, matching the `NAMatMul` `loadM` pattern, and the shader immediately
wraps it in `uniform<uint>` to keep the runtime shape scalar out of regular
thread registers. The selector does not gate on whether LSE output is requested;
the first implementation still writes only `O`.

The descriptor picks direct no-scratch mode for `C < 2048` and split/reduce mode
(`NSG=8,NWG=32`) for larger `C`. This selection is C-only; it does not depend on
head ratio.

Repeated `sdpa_bench` checks after hooking the production path:

| Shape | AttentionR1 | NAAttention splitKV baseline | generic/MPS |
|---|---:|---:|---:|
| `B=1 R=1 C=8192 Hq=40 Hk=2 D=256` | 0.397 ms | 0.424 ms | 2.102 ms |
| `B=1 R=1 C=8192 Hq=40 Hk=4 D=256` | 0.409 ms | 0.462 ms | 2.213 ms |
| `B=1 R=1 C=4096 Hq=40 Hk=4 D=256`, NA disabled | 0.251 ms | n/a | 1.217 ms |

Light C-boundary sweep, fixed `B=1,R=1,Hq=40,Hk=4,D=256`, single pass to avoid
thermal drift:

| C | R1 on plain | R1 off plain | R1 on causal | R1 off causal |
|---:|---:|---:|---:|---:|
| 512 | 0.381 ms | 0.546 ms | 0.318 ms | 0.493 ms |
| 768 | 0.450 ms | 0.448 ms | 0.205 ms | 0.628 ms |
| 1024 | 0.428 ms | 0.539 ms | 0.209 ms | 0.615 ms |
| 1152 | 0.512 ms | 0.498 ms | 0.549 ms | 0.400 ms |
| 1280 | 0.646 ms | 0.566 ms | 0.585 ms | 0.410 ms |
| 3584 | 0.401 ms | 0.368 ms | 0.428 ms | 0.270 ms |
| 3840 | 0.552 ms | 0.416 ms | 0.595 ms | 0.298 ms |
| 3968 | 0.582 ms | 0.515 ms | 0.559 ms | 0.285 ms |
| 4096 | 0.357 ms | 0.506 ms | 0.279 ms | 0.638 ms |
| 4224 | 0.593 ms | 0.517 ms | 0.281 ms | 0.477 ms |
| 4608 | 0.456 ms | 0.546 ms | 0.299 ms | 0.620 ms |

The non-selected rows are noisy because both R1-on and R1-off use the same
baseline path there; do not overfit a single pass. The useful signal is that
the C banding, not head-ratio, is the right selector axis to continue refining.

128-step C sweep command used for the tighter selector:

```sh
./sdpa_bench --na --c-sweep --c-from=128 --c-to=8192 --c-step=128 \
  --sweep-r=1 --sweep-hq=40 --sweep-hk=4 --sweep-d=256
```

For the comparison, R1 was temporarily broadened to all `C`, then temporarily
disabled to collect the NAAttention/MPS baseline. No production environment
toggle was added.

Representative causal no-mask rows from the 128-step sweep:

| C | D=256 R1 | D=256 baseline | D=128 R1 | D=128 baseline |
|---:|---:|---:|---:|---:|
| 128 | 0.389 ms | 0.453 ms | 0.366 ms | 0.331 ms |
| 256 | 0.286 ms | 0.367 ms | 0.262 ms | 0.415 ms |
| 1024 | 0.210 ms | 0.377 ms | 0.196 ms | 0.471 ms |
| 1920 | 0.259 ms | 0.689 ms | 0.218 ms | 0.601 ms |
| 2048 | 0.259 ms | 0.255 ms | 0.238 ms | 0.194 ms |
| 3840 | 0.314 ms | 0.290 ms | 0.461 ms | 0.442 ms |
| 4096 | 0.279 ms | 0.323 ms | 0.261 ms | 0.537 ms |
| 5120 | 0.420 ms | 0.613 ms | 0.266 ms | 0.441 ms |
| 6144 | 0.493 ms | 0.613 ms | 0.260 ms | 0.600 ms |
| 8192 | 0.441 ms | 0.477 ms | 0.315 ms | 0.266 ms |

The 128-step sweep was useful for understanding the shape, but the NA-side
selection was not strong enough to justify a C-band gate. The production rule is
therefore to enable R1 for every supported `R=1,C>0` dense shape. The descriptor
still uses direct mode for `C < 2048` and split/reduce for `C >= 2048`.

Odd `C` is handled by the R1 split loop (`c < C_LEN`) and the reduce loop over
`NWG`; the integration test now includes a long odd-`C` causal decode case
(`C=8193,Hq=40,Hk=4,D=256`) that routes through the production selector.

## Tests

Existing integration coverage now checks dense half-input decode across the
selector for causal and non-causal `R={1,2,7,16,32}`, including one direct R1
case (`R=1,C=1536,Hq=40,Hk=4,D=256`), one long odd-C R1 case
(`R=1,C=8193,Hq=40,Hk=4,D=256`), and one GQA case with `Hq=32,Hk=8`. There is
also BF16 R1 decode coverage at `R=1,C=4097,Hq=40,Hk=4,D=256`, for both causal
and non-causal no-mask mode.
Remaining coverage to add:

- broader GQA correctness cases
- quantized `NAInt8Attention`
- masked decode after masked support is added

Compare against the CPU reference implementation with tolerance-based checks.

Performance tests should compare:

- existing MFA attention path
- new MFA decode split-KV path
- CUDA FlashAttention behavior where comparable

## Risks

- Incorrect log base in combine. MFA should use `exp2` and `log2`, not natural
  `exp` and `log`, because the existing kernels store log2 LSE.
- `NAInt8Attention` must add `V_mean` exactly once after split combine.
- Too many splits can lose to scratch traffic and combine overhead.
- Masked/all-empty splits need explicit `-inf` LSE and zero partial output.
- Descriptor and kernel descriptor fields must stay in sync when source
  generation depends on a value.
