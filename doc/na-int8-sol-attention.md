# INT8 Sol attention

`NAInt8SolAttentionKernel` implements forward Sol attention using the validated
INT8 algorithm from `ccv_sol_attention_final.patch` (SHA256
`c04c5124fff46f0cdae735475b32da842539ee215d1e27080edc52dbe52679d4`).
It targets contiguous FP16 `[N, T, H, 128]` tensors on MPS hardware with neural
matrix accelerators. There is no FP16 Sol execution variant.

## Command contract

```c
ccv_nnc_cmd_t cmd = CMD_SOL_ATTENTION_FORWARD(1.0, 0.5, 64, video_start, video_end);
cmd.info.sol_attention.query_block_size = 64;
ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0,
    TENSOR_LIST(q, k, v), TENSOR_LIST(o), stream_context);
```

- Q/K/V/O have equal contiguous rank-4 NHWC shapes. Output retains separate heads.
- `scale` is the finite logit multiplier; zero and negative values are supported.
- `tau` is a finite routing threshold multiplier. Increasing it selects fewer
  exact remote blocks. Zero does not request dense attention.
- `block_size` is the KV pooling size: 16, 32, or 64 on MPS.
- `query_block_size` is 16, 32, or 64 and cannot exceed `block_size`. Zero defaults
  to `block_size` at the command layer.
- `local_block_radius` must be at least one. The easy macro initializes it to one.
- `[approximation_start, approximation_end)` applies to both queries and keys.
  A complete query group and complete KV block must lie within it before their
  interaction can use a summary. Boundary-crossing groups and blocks stay exact.
- An optional fourth input is a CPU Int32 scalar, `is_dense_attention`. Nonzero
  calls the backend's shared dense SDPA function directly; zero or an omitted
  input executes Sol. The MPS dense path requests FP16/INT8 arithmetic; `flags`
  supplies additional GEMM flags to that path. MPS Sol always uses INT8, even
  when `flags` is zero.
- To exercise the actual all-exact Sol kernel, select Sol and set an empty
  eligible interval or a radius covering every KV block. Native bypass timings
  are not all-exact Sol timings.

The MPS backend respects MFA/attention/neural-accelerator disable flags and
rejects unsupported Sol execution. The native bypass is available before the
Sol hardware gate. There is no backward, CUDA, causal/masked, GQA,
unequal-length cross-attention, or noncontiguous execution implementation.
The backward command ID exists only for the command-pair registry convention.

The independent CPU reference supports contiguous FP32 with general positive
head/block dimensions. It uses double-precision pooling and accumulation,
and computes the Sol approximation without emulating INT8 quantization. Dense
selection uses the ordinary CPU SDPA implementation.

## Implementation

The kernel library is cached by pooling block size. Pipeline keys contain the
stage, N/T/H, and query block size; shapes specialize Metal function constants.
Encoding computes scratch offsets and dispatch geometry from the current call.
The library cache uses a dedicated `NAInt8SolAttentionKernelDescriptor` type.

For pooling blocks of 64 tokens, the production path uses five launches:

1. Single-pass tiled Morton V-mean.
2. Token Q/K/centered-V quantization and pooled Q/K/V.
3. Summary K quantization, V normalization, and independent pooled-key statistics.
4. Routing and packed route bits.
5. Summary then exact attention, sharing FP32 numerator and softmax state in registers.

B16/B32 retain a separate pooling launch, for six total. Pooled K/V consumers
read only valid entries; summary preparation initializes its padded outputs.
Summary QK uses INT8, while summary PV uses normalized FP16 inputs and FP32
accumulation. Block multiplicity enters the numerator and denominator without
shifting the INT8 running maximum. The implementation preserves cached Q tiles,
ascending packed route traversal, explicit FMA rescaling, and every-two-block
synchronization for B64/Q64 at T >= 32768. Padded query SIMD groups participate
in barriers, and output writes are guarded by the actual row count.

`createSource` composes stage generators through CodeWriter. `loopAttention`
emits separate summary/exact traversals; `accumulateAttention` generates their
precision-specific PV operations and four output slices. The kernel interface
provides dispatch geometry from the current descriptor, and pipeline creation
checks both thread count and static threadgroup memory against device limits.

Scratch includes INT8 Q/K/V, pooled tensors, routing maps, and scales. There is
no intermediate token-sized FP32 numerator or softmax-state buffer. The routing
map is quadratic in the number of pooled blocks; no dense token-by-token score
matrix is materialized.

## Validation

Regenerate command registries before building:

```sh
cd lib/nnc/cmd
./build-cmd.rb .
cd ../../../test/int/nnc
make debug -j4
./sol_attention.tests
./mpsdnn.tests
./mpsblas.tests
cd ../../unit/nnc
make sol_attention.tests
./sol_attention.tests
```

The generated command IDs, backend dispatch, easy macro, and source lists
are included with this change, as requested for the commit.

Native Sol tests cover CPU parity, signed/zero scales, all-exact/native parity,
protected boundaries, pooling/query sizes, batches, centered large values,
runtime switching, offset views with output sentinels, and growing/shrinking
shapes including T=20501 and T=32769. An additional comparison against the
original supplied patch checked fourteen dense/sparse configurations, including
batching, B16/B32/B64, Q16/Q32/Q64, offsets, and long tails: all outputs were
byte-identical and offset sentinels were preserved.

## Shape benchmark

```sh
cd bin/mfa
make na_int8_sol_attention_bench
./na_int8_sol_attention_bench 32768 56 6
./na_int8_sol_attention_bench 65536 56 6
./na_int8_sol_attention_bench 103982 56 6 - 1262
# Replay a capture with equal-size FP16 prefix.{q,k,v}.bin files:
./na_int8_sol_attention_bench 103982 56 6 /path/to/prefix 1262 1
```

The benchmark runs current native NAInt8Attention, actual all-exact Sol, and
B64/Q64/tau0.5/radius1 sparse Sol on identical inputs. It excludes three warmup
rounds, alternates execution order, and reports median GPU command times for
complete attention operators including preprocessing. Speedups are medians
of within-round ratios. It also reports the measured exact-block fraction,
all-exact/protected-output errors, and allocated scratch. Routing inspection
and correctness comparisons are outside timed command buffers.

The original real H3 captures were cleaned up. New synthetic benchmarks measure
performance on their tensor shapes, not H3 output quality or generation speed.
Independent random Q/K/V can have substantial approximation error versus dense
attention; the sparse error reported by this benchmark is not a quality target.
The relevant correctness checks are parity against the CPU Sol algorithm,
all-exact/protected parity against native INT8, and parity with the supplied
implementation.

## Local results, 2026-09-22

Apple M5 Max, macOS 26.6.2. Synthetic uniform seed-42 inputs, N=1/H=56/D=128,
B64/Q64, tau=0.5, radius=1. The protected prefix is 470 tokens at 32k/64k
and 1262 tokens at 103,982. Six measured rounds follow three warmup rounds;
all preprocessing is included.

| Tokens | Native median ms | All-exact median ms | Sparse median ms | Paired all-exact speedup | Paired sparse speedup | Exact-block fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1332.3295 | 1267.5913 | 474.6343 | 1.0478x | 2.8233x | 0.333888 |
| 65,536 | 3982.9503 | 3915.0589 | 1319.0116 | 1.0495x | 2.9936x | 0.321285 |
| 103,982 | 7156.7874 | 6996.4204 | 2492.3670 | 1.0219x | 2.8471x | 0.326659 |

All-exact/protected relative L2 against native INT8 was below 2.2e-5 in all
three completed runs (about 5.3e-6 at 103,982 tokens). Absolute timings varied after the CPU-heavy regression run;
use the paired ratios rather than ratios of independently reported medians.
The initial 32k run measured 1.0736x all-exact and 2.9161x sparse.

The 103,982-token result comes from a fresh complete run. The earlier
interrupted run is excluded from these statistics.

Validation completed: native debug build, CPU Sol tests (2/2), MPS Sol tests
(4/4), `mpsdnn.tests` (126/126), and a fresh full `mpsblas.tests` run (283/283,
no skipped cases). Fourteen comparisons against the supplied patch were
byte-identical. Command registries were regenerated and are included with
this change so SOL_ATTENTION is available directly from the checkout.

## B64 pooling-dispatch removal, 2026-09-22

B64 now skips the standalone pooling dispatch, reducing nine dispatches to
eight, including partial tails. The quantization kernel already writes valid
pooled Q/K/V; subsequent readers do not consume the uninitialized pooled tail.
The shader code, scratch layout, and arithmetic are unchanged.

A paired comparison used the same synthetic shapes and parameters above,
three warmups, and six measured rounds. Each round ran native INT8 attention,
old all-exact Sol, old sparse Sol, new all-exact Sol, and new sparse Sol, with
order reversed on alternating rounds. Both Sol versions used the same kernel
cache and scratch; each had separate output buffers. All preprocessing was
timed. Ratios below are medians of within-round ratios, not ratios of medians.

| Tokens | Old/new all-exact | Old/new sparse | Native/new all-exact | Native/new sparse |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1.0892x | 1.0116x | 1.1099x | 2.9384x |
| 65,536 | 0.9908x | 0.9841x | 0.9015x | 2.4511x |
| 103,982 | 1.0071x | 1.0102x | 1.0393x | 2.9744x |

Values above 1 favor the new path. The sparse old/new ratios straddle parity;
individual rounds vary substantially, so these runs do not establish a
consistent throughput improvement from dropping this launch. The removal is
a simplification; summary/exact fusion remains a separate experiment.

Dense and sparse outputs were byte-identical to the nine-dispatch path at
all three shapes. Poisoning scratch before the new path also preserved byte
parity at T257 and T32769/H2. The native build and all four Sol integration
tests passed. See [full paired logs](benchmarks/na-int8-sol-attention-b64-pool-skip-2026-09-22.txt).

## Single-pass V-mean investigation, 2026-09-22

One V-mean dispatch is numerically possible. Native-style direct reductions
and coalesced single-pass tiled reductions all produced bit-identical means.
The tiled variants keep partial sums in threadgroup memory and preserve the
existing accumulation order. The best long-sequence tile covers 32 channels
per threadgroup.

Reduction-only medians on M5 Max, N=1/H=56/D=128, four warmups and sixteen
measured rounds with alternating execution order:

| Tokens | Current two-pass ms | Native-style Morton single-pass ms | Best tiled single-pass ms |
| ---: | ---: | ---: | ---: |
| 32,768 | 0.89104 | 2.57383 | 0.92081 |
| 65,536 | 1.76398 | 6.12333 | 1.90285 |
| 103,982 | 2.79238 | 10.58208 | 3.11646 |

The single-pass tiled version is 3–12% slower for V-mean itself on these shapes;
these percentages are not whole-attention slowdowns. A short T257/N2/H2 probe
favored the simple single-pass version by about 0.0005 ms. At that point production retained
the two-pass reduction; the integration below supersedes that choice. See [probe logs](benchmarks/na-int8-sol-attention-v-mean-2026-09-22.txt).

## Actual native V-mean and Morton follow-up, 2026-09-22

The earlier native-style measurement used a port. This follow-up directly
benchmarked `NAInt8AttentionDescriptor`'s cached `compute_v_mean` pipeline
(`pipelineValue->fifth`) on the same V buffer as the actual Sol reduction.
It also applied native-style Morton ordering to the coalesced tiled candidate
and tested SIMD-first reductions using 512 B or 1 KiB of threadgroup scratch.

Reduction-only median milliseconds, N=1/H=56/D=128; five warmups and twenty
measured rounds, rotating variant order:

| Tokens | Current two-pass | Actual native single-pass | Tiled linear | Tiled Morton | Small-scratch Morton (256 threads) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 0.88519 | 3.10027 | 0.95969 | 0.94731 | 0.93198 |
| 65,536 | 1.76100 | 6.42679 | 1.98746 | 1.89138 | 1.84237 |
| 103,982 | 2.78463 | 12.12288 | 3.26831 | 3.15906 | 2.97056 |

The native and tiled means matched production byte-for-byte. The small-scratch
variants changed summation order; the 256-thread version had mean relative L2
of about 1.0e-7 to 1.4e-7 and max absolute differences below 4e-9 on the long
shapes. This mean-only check is not full attention validation of that change.
A T257/N2/H3 case also passed, covering batching and padded Morton head indices.

Morton ordering and smaller scratch narrow the gap, but none of these
single-pass candidates beat production on the long shapes. The subsequent integration below adopts tiled Morton to remove a launch despite
the small reduction-only regression. See [full results](benchmarks/na-int8-sol-attention-v-mean-native-2026-09-22.txt).


## Single-pass integration and launch-fusion experiments, 2026-09-22

Production now uses one tiled Morton V-mean dispatch, preserving the original
per-thread summation and XOR reduction order. Its fixed `float4[1024]` allocation
is 16 KiB normally and 32 KiB under shader validation, within the M5 Max cap.
Neither H nor sequence length grows that allocation. H and T choose a tile of
1, 2, 4, or 8 vectors and 128 or 256 dispatch threads; dispatch geometry is
derived from the current invocation. D remains fixed at 128 for SOL.
The partial-mean device buffer and its binding/dispatch were removed. Mean
scratch is explicitly aligned to 16 bytes for odd head counts; packed FP16
input vectors preserve support for scalar-aligned input offsets.

The initial single-pass integration had these seven launches:

1. Single-pass V-mean.
2. Q/K/centered-V quantization and pooled Q/K/V.
3. Summary K quantization and V normalization.
4. Pooled-key mean/variance statistics.
5. Routing and packed route bits.
6. Summary attention, saving FP32 numerator and softmax state.
7. Exact attention, completing the same normalization.

The original two-pass versus single-pass full-attention comparison used N=1,
H=56, D=128, B64/Q64, protected prefix 470, three warmups and six alternating
measured rounds. All outputs were byte-identical. Speedup means old/new; values
below one are regressions.

| Tokens | Single-pass sparse speedup | Single-pass all-exact speedup |
| ---: | ---: | ---: |
| 32,768 | 0.99623x | 1.00572x |
| 65,536 | 0.99865x | 1.00996x |
| 103,982 | 0.99832x | 0.95014x |

The largest all-exact case regressed about 5.2% in this run. There was substantial
clock/timing drift across rounds; this measures the complete generated pipeline,
not an isolated causal cost of the mean dispatch. The single-pass organization follows the requested priority even with a small
regression. Reduction-only measurements
from earlier investigations remain above for context.

The isolated mean oracle covers 16 configurations, including both sides of the
20,480 cutoff, odd heads, every tile-width transition, batches, and lengths
32,768 / 65,536 / 103,982. It compares against the exact original reduction
order and an FP64 sum, with output sentinels. Complete SOL comparisons cover
16 dense/sparse configurations, including B16/B32/B64 and Q16/Q32/Q64.

The two experiments combine stages 3–7 into:

- **Three kernels (five total):** summary preparation + key statistics;
  routing; summary + exact attention.
- **Two kernels (four total):** summary preparation + key statistics;
  routing + summary + exact attention.

The summary/statistics launch adds independent per-head statistics jobs to the
summary-preparation grid. Fused attention keeps FP32 numerator/max/sum in
registers and traverses summaries before exact blocks. Its loops are emitted
separately with CodeWriter: an initial runtime phase-loop prototype was slower
and was replaced before the reported long-shape experiments. Fused routing
retains the existing device route buffers and adds no growing threadgroup
allocation. B16/B32 add their separate pooling dispatch to each total.

The [reproduction package](benchmarks/na-int8-sol-stages-20260922/README.md)
contains standalone comparison/validation programs and patches applied only to
temporary copies of the kernels. The five-launch candidate is now the production path, as described below.


Rotating-order comparisons against seven launches, same synthetic inputs and
six measured rounds. These are paired median speedups; above one is faster.

| Tokens | Five launches sparse | Four launches sparse | Five launches all-exact | Four launches all-exact |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 0.99838x | 1.00192x | 1.00211x | 0.99698x |
| 65,536 | 1.01121x | 1.01379x | 0.98096x | 0.98863x |
| 103,982 | 0.95226x | 0.91660x | 0.98871x | 0.96837x |

| Tokens | Seven-launch scratch GB | Five-/four-launch scratch GB |
| ---: | ---: | ---: |
| 32,768 | 1.724 | 0.770 |
| 65,536 | 3.481 | 1.572 |
| 103,982 | 5.587 | 2.559 |

All-exact outputs were bit-identical. Sparse relative L2 was 4.66e-7, 6.04e-7,
and 6.47e-7 respectively; both fusion candidates produced the same comparison
errors. Peak absolute differences were 0.000977, 0.001953, and 0.000977.
The five-/four-launch designs are feasible and halve device scratch, but are
not unqualified speed improvements: at 103,982 tokens, sparse latency increased
about 5.0% / 9.1%. These measurements preceded the decision to productionize five launches.
Raw timing and validation logs are in the reproduction directory.

Historical validation caveat: the separate-stage attention path
intermittently produces non-finite values at T=32769 with shader instrumentation.
This also occurs in the saved two-pass baseline. The matching pooled inputs,
quantized data, means, routes, and saved summary state isolate this from the mean
change; changing the traversal synchronization did not resolve it and was
reverted. Full instrumented separate-stage SOL is **not** claimed to pass.
The isolated mean/resource checks and the fused-output replay checks are separate
checks; replay compares instrumented fusion to uninstrumented reference outputs.


Final validation results: debug build passed; CPU SOL tests 2/2; Metal API
validation integration tests 4/4, including the new odd-head mean-alignment
case; isolated mean oracle 16/16 with API + shader validation and two-byte-aligned
FP16 input offsets; experimental resource caps 51/51. Both fusion candidates
passed all 16 dense/sparse comparisons for ordinary inputs, scalar-aligned input
offsets, negative scales, and zero logits with large finite V. Instrumented
fusion replay against saved uninstrumented outputs also passed all 16 cases for
both candidates. The separate-stage instrumented suite remains 3/4 because of
the T32769 numerical issue described above. Logs preserve the failure as well
as the passing checks.


A final 32k spot check after the scalar-offset/alignment refinements again
produced bit-identical complete outputs: single-pass paired speedup was 1.00139x
sparse and 1.05956x all-exact against the original two-pass implementation. This
reinforces the timing-variation caveat rather than attributing a whole-attention
speedup to the small mean reduction. See `mean-final-32768.txt`.


## Production five-launch integration, 2026-09-22

The encoder now selects the five-launch organization for B64 (six for B16/B32).
Summary preparation includes independent statistics jobs. Summary and exact
attention share register state, removing the global FP32 numerator and max/sum
buffers. Obsolete stages and sparse buffer-binding gaps are removed.
CodeWriter stage/loop/accumulation methods generate the shader, including the
four PV/output slices. Geometry lives in the kernel interface and is evaluated
from the current descriptor; every pipeline checks thread and memory caps.

Validation of the organized production implementation:

- Debug build passed; CPU SOL tests 2/2; MPS SOL tests 4/4 with both Metal API
  and shader validation. The integration test repeats the T32769 all-exact and
  native-bypass cases to cover the previously intermittent failure.
- 43 production pipeline specializations passed resource checks, alongside the
  51 archived experimental checks. V-mean uses a fixed 16 KiB normally and
  32 KiB under shader instrumentation; combined attention has zero static
  threadgroup allocation.
- The isolated V-mean oracle passed all 16 configurations with instrumented
  execution and scalar-aligned input offsets.
- Instrumented production/prototype comparisons covered 16 dense/sparse cases
  with three repeats each for normal inputs, scalar offsets, negative scales,
  and zero logits with large finite V. Another 80 executions stress-tested
  T32769 dense/sparse over two seeds and fresh contexts. Every execution passed
  finite/tolerance checks; final output comparisons were bit-identical to the
  saved five-launch prototype. The same normal cases
  also passed three repeats without instrumentation.

The older separate-stage instrumented failure remains documented above for
historical context. It was not observed in these production five-launch checks.
The archived prototype reconstructed by the reproduction package is byte-for-byte
identical to the reference saved before this reorganization.

Paired production-versus-prototype benchmarks used N=1/H=56/D=128, B64/Q64,
scale=1, tau=0.5, protected prefix 470, three warmups and six measured rounds,
with validation disabled. Timings include all five launches. Output checking
runs outside timed command buffers. Speedups are medians of within-round
prototype/production ratios; values above one favor production.

| Tokens | Prototype sparse ms | Production sparse ms | Sparse speedup | Prototype all-exact ms | Production all-exact ms | All-exact speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 204.938 | 203.593 | 1.00770x | 786.524 | 731.806 | 1.01879x |
| 65,536 | 1279.559 | 1221.526 | 1.00206x | 5250.946 | 5173.323 | 1.00348x |
| 103,982 | 4322.179 | 4285.646 | 1.01054x | 11304.840 | 11257.796 | 0.99612x |

The organized version stayed within approximately -0.4% to +1.9% in paired
speedup. This does not show a meaningful regression against the selected
five-launch prototype; individual timings varied substantially, especially at
32k/65k. Final dense/sparse outputs were bit-identical at all three shapes, and
every execution (including warmups) passed finite/tolerance checks. Scratch
matches the prototype: 769,646,080 / 1,572,236,288 / 2,559,076,896 bytes.
See `production-paired-*.txt` in the reproduction package for raw rounds.

A fresh native-relative spot check at 32,768 tokens used the same shape/input
settings and six measured rounds. Native / all-exact SOL / sparse SOL medians
were 1034.190 / 997.123 / 338.553 ms. Paired speedups were **1.0506x all-exact**
and **3.0721x sparse**, with exact-block fraction 0.333888. All-exact/protected
relative L2 errors were 1.84e-5 / 1.82e-5, below the 1e-4 check. Every warmup
and measured output was finite. See `production-native-32768.txt`.
These remain synthetic shape measurements on M5 Max; the original captures
were cleaned up, so this does not establish model-output quality.
