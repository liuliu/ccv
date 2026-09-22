# ANE H256 overhead after ordinary quantization fusion

This follow-up measures H256 against the faster ordinary-quantization selector
committed in `468b226c`, on Apple M4 Pro / macOS 26.6.2, 2026-09-22. The ordinary
path remains unchanged: fused quantization/transpose for K <= 8192, separate
scale reduction and tiled quantization/transpose for wider rows.

These are steady-state **full GEMM calls**, including packed-weight decode,
activation preparation, weight upload, CoreML/ANE evaluation, output scaling,
and stream completion. They are not whole-model inference measurements.
Both variants use BF16 activations and Q6_K weights packed from the same dense
matrix; the H256 variant rotates weights offline. Compilation, initial tensor
transfers, offline weight packing, and warmup are excluded.

## Results

All 17 measured shapes have median overhead and upper 95% bounds below 1%.
The largest median overhead is **0.719%** (H3 16K attention output); the largest
upper bound is **0.805%** (H3 4K attention output). This establishes the target
for these synthetic, model-shaped BF16 GEMMs on this machine, not for arbitrary
shapes, devices, or whole-model inference.

| Shape | Ordinary median ms | H256 median ms | Paired overhead | 95% interval |
| --- | ---: | ---: | ---: | --- |
| qwen21_ffn_up | 22.2893 | 22.3829 | +0.175% | [-0.517%, +0.548%] |
| qwen21_ffn_down | 28.1532 | 27.4577 | -2.112% | [-2.571%, -1.403%] |
| klein4_single_in | 43.1119 | 43.1938 | +0.163% | [+0.097%, +0.253%] |
| klein4_single_out | 23.7353 | 23.3438 | -1.800% | [-1.969%, -1.344%] |
| klein4_double_ffn_down | 15.2144 | 15.1315 | -0.301% | [-0.658%, -0.073%] |
| klein9_single_in | 70.1398 | 70.0933 | -0.127% | [-0.689%, +0.484%] |
| klein9_single_out | 41.3951 | 40.7326 | -1.607% | [-1.701%, -1.527%] |
| h3_4k_q | 18.4034 | 18.4495 | +0.322% | [-0.120%, +0.514%] |
| h3_4k_attention_out | 18.7269 | 18.8352 | +0.429% | [-0.031%, +0.805%] |
| h3_4k_ffn_down | 41.2687 | 40.2069 | -2.561% | [-2.647%, -2.486%] |
| h3_16k_q | 64.8312 | 65.3092 | +0.670% | [+0.593%, +0.804%] |
| h3_16k_attention_out | 66.8002 | 67.3400 | +0.719% | [+0.603%, +0.782%] |
| h3_16k_ffn_down | 141.6769 | 142.5025 | +0.558% | [+0.388%, +0.775%] |
| krea2_q | 19.1233 | 19.2180 | +0.383% | [+0.012%, +0.625%] |
| krea2_kv | 5.7958 | 5.7710 | -0.481% | [-1.193%, -0.103%] |
| krea2_ffn_up | 48.5232 | 48.6444 | +0.299% | [+0.201%, +0.338%] |
| krea2_ffn_down | 55.4013 | 54.8609 | -0.922% | [-1.063%, -0.797%] |

The [raw paired samples](ane-hadamard256-optimization.csv) include the complete
sweep and all three confirmations: 1,170 paired samples in total. The table uses
the longer confirmation for those three shapes and the initial sweep for the
others. The original 51-pair intervals for the confirmed shapes crossed 1%;
the longer runs resolve that measurement uncertainty without shader changes.

Workload dimensions are in the [shape manifest](ane-quantization-fusion-shapes.csv).
The [earlier report](ane-quantization-fusion-benchmark.md#model-dimensions-and-workload-assumptions)
documents their sources and assumptions: batch 1, 1024 x 1024 images; H3 uses
4K/16K packed diffusion-token budgets, not LLM prefill.

The initial sweep uses 51 alternating AB/BA pairs per shape, with ten calls
per variant in each pair. The three confirmations use 101 pairs of twenty
calls per variant, for klein4_double_ffn_down, h3_16k_ffn_down, and krea2_kv.
The statistic is the median of `100 * (H256 / ordinary - 1)` across pairs.
Confidence intervals are percentile bootstraps of those paired ratios, using
10,000 resamples and Python `random.Random(42)`. Negative overhead means the
full H256 GEMM call measured faster than the ordinary GEMM call.

## Implementation

- The transform uses two SIMD shuffles for each intermediate H4 stage and one
  shuffle of the difference for the final stage. Normalization by 1/16 is folded
  into the row scale: raw transformed maximum / (16 * 127). Quantization uses
  the unrounded raw maximum, so its rounded values are already bounded by 127.
- For K <= 8192, eight rows retain transformed values in registers through the
  row reduction. Stores pack eight adjacent rows. Two alternating 8 KiB
  threadgroup tiles let each iteration omit the barrier after reading a tile:
  the next iteration writes the other tile, and its barrier completes those
  reads before the first tile is reused. This uses no device activation scratch.
- Rows divisible by 768 but not 1024, with K <= 6144, use three SIMD groups per
  row instead of four. In particular, H3's K=5376 fits seven groups per SIMD
  exactly. Dispatch uses 768 threads, derived from the current K at encode time.
- For K > 8192, one dispatch computes transformed row maxima into the existing
  scale buffer's float capacity. A second recomputes each 16 x 256 transformed
  tile and quantizes/transposes it. This avoids retaining a complete wide row
  and improves surface writes without an intermediate activation buffer.
  The output kernel reconstructs and rounds the activation scale to the same
  IO precision used by the original fused implementation.

The single-dispatch entry point remains `quantize_hadamard_activation`.
Wide rows deliberately use two dispatches; reducing dispatch count alone did
not minimize full-GEMM latency. Shader-library cache keys stay independent of
runtime dimensions, while function constants and dispatch geometry carry shape.

## Investigation notes

Continuous GPU microbenchmarks understated the cost after ANE evaluation.
Cold-GPU probes, with 35 ms of CPU idle before each submission, tracked the
remaining overhead more closely. All acceptance measurements above use actual
full GEMM calls, not the isolated shader probes.

The wide-row recomputation, wider stores, cheaper transform arithmetic,
alternating tiles, and three-SIMD-group layout provided the useful gains.
Experiments with concurrent weight decode, matrix-based H256, interleaved rows,
other threadgroup sizes, compiler hints, row-order swizzling, and padded output
surfaces did not provide a convincing absolute latency improvement and were
not retained.

## Validation and reproduction

The independent activation probe passes 189 cases across FP16, BF16 and FP32,
including zero rows, outliers, varied finite magnitudes, batching, source strides,
padded rows, cache reuse, narrow-row layout variants, and K through 65536.
Integration coverage includes the K=8192 boundary, K=8448 with batched scales,
K=14336 wide rows, and M=513 / K=5376 with the three-SIMD-group layout
and ANE split-K evaluation. The latter also compares against explicitly
rotated ordinary ANE GEMM at the existing 1e-3 relative-L2 tolerance.
`make debug -j4` succeeded. The full `mpsblas.tests` run passed 283 cases
with one backend-dependent skip; `mpsdnn.tests` passed all 126 cases.
Both binaries exited successfully under unrestricted Metal execution.

An additional M=17, N=256, K=5376 diagnostic exposed existing small-M ANE
accumulation drift: FP32 output was 0.763% away from CPU GEMM but identical to
ordinary ANE given explicitly rotated/quantized activations. The activation
bytes themselves matched the CPU reference exactly. The model-like integration
case uses M=513, which selects the existing ANE split-K route, and retains the
original CPU tolerances. The independent quantization probe still covers M=17
at K=5376; this change does not modify ANE accumulation or its split-K selector.

```sh
make -C lib -B -j4 lib
make -C bin/nnc ane_hadamard_quant_probe ane_hadamard_bench
bin/nnc/ane_hadamard_quant_probe
python3 - <<'PY'
import csv, subprocess
for shape in csv.DictReader(open('doc/ane-quantization-fusion-shapes.csv')):
    print(shape['name'], flush=True)
    subprocess.run(['bin/nnc/ane_hadamard_bench', shape['M'], shape['N'],
                    shape['K'], '51', '10', '2'], check=True)
# Longer confirmations:
for m, n, k in [(4096, 3072, 9216), (16384, 5376, 14336), (4352, 1536, 6144)]:
    subprocess.run(['bin/nnc/ane_hadamard_bench', str(m), str(n), str(k),
                    '101', '20', '2'], check=True)
PY
make -C test/int/nnc debug -j4
(cd test/int/nnc && ./mpsblas.tests && ./mpsdnn.tests)
```

The earlier work is preserved in checkpoint `468b226c`. Historical performance
claims in `97f908d3` used the older two-dispatch ordinary baseline.
