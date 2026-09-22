# ANE quantization fusion on diffusion-model shapes

This report records checkpoint `468b226c`. See the
[Hadamard optimization follow-up](ane-hadamard256-optimization.md) for subsequent
shader changes and measurements against this ordinary-quantization baseline.

Measured on Apple M4 Pro, macOS 26.6.2, 2026-09-22, using an `-O3` library.
These are steady-state per-GEMM measurements with synthetic inputs and published
model dimensions. Batch size is one.

## Findings and implementation

Ordinary ANE activation preparation can use one dispatch. The old implementation
first reduces each full row to its scale, then quantizes and transposes square
tiles. This separates the full-row reduction from the efficient transpose.
The fused implementation assigns complete rows to each threadgroup, retains
input values in registers through reduction, and writes packed INT8 bytes
directly to the ANE surface. It uses no device activation staging buffer.

Production enables ordinary fusion for **K <= 8192**: eight rows, 128 threads per
row, a 4 KiB INT8 threadgroup tile, and 128 bytes for partial maxima. The inverse
scale uses the scale rounded to the IO dtype, matching the old two-dispatch path.
Shapes and strides remain function constants; dispatch geometry uses the current
M and K, rather than shape values stored in a cached kernel object.

For wider rows, a four-row/256-thread-per-row fused candidate improved GPU-only
time, but did not reliably improve end-to-end GEMM. Production retains two
dispatches for ordinary K > 8192. The candidate remains accessible to the probe
for reproducing the experiment. A rereading variant initially regressed wide
FFN inputs substantially; retaining values and increasing threads per row fixed
the isolated shader regression, but was insufficient to justify a general
end-to-end rollout.

Hadamard still uses a single dispatch with no device activation staging buffer.
Its layout now uses eight rows for K <= 8192 and four rows for larger K, sharing
an 8 KiB INT8 threadgroup tile. CPU-reference checks cover both layouts.

**The previous <1% Hadamard claim is baseline- and shape-specific.** The
[historical report](ane-hadamard256-benchmark.md) measured against ordinary
quantization with two dispatches. Against the new ordinary baseline, the longer
Krea K/V confirmation measured **+2.12%** overhead (95% paired-bootstrap interval
**[+1.29%, +3.08%]**). Qwen FFN-up measured +0.68%, but its interval crosses 1%.
The feature remains correctness-valid; the broader <1% performance target is
not established, and fails on some measured shapes.

## Model dimensions and workload assumptions

`A` is M × K, weights are N × K, and the output is M × N. M includes flattened
batch/token rows. The image resolution is 1024 × 1024. Text lengths below are
explicit benchmark assumptions; they vary with the prompt and implementation.

| Model | Verified projection dimensions and chosen token rows |
| --- | --- |
| Qwen Image 2.1 | Width 4096, FFN width 12288; M=4096 target-image tokens, representing cached-prefix denoising. |
| FLUX.2 klein 4B | Width 3072, FFN width 9216; combined single-block input maps 3072→27648 and output maps 12288→3072. M=4608 assumes 4096 image + 512 text tokens; double-block image FFN uses M=4096. |
| FLUX.2 klein 9B | Width 4096, FFN width 12288; combined input maps 4096→36864 and output maps 16384→4096. M=4608 uses the same text-token assumption. |
| MiniMax H3 | Denoiser width 5376, attention width 56×128=7168, FFN width 14336. M=4096 and 16384 are requested packed diffusion-token budgets, not a mapping to one particular video duration. |
| Krea 2 | Width 6144, K/V width 12×128=1536, FFN width 16384. M=4352 assumes 4096 image + 256 text tokens. |

Sources: Qwen [transformer config](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/transformer/config.json),
[VAE config](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/config.json),
and [projection implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py);
FLUX.2 [official model](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py)
and [autoencoder](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py);
H3 [config](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/transformer/config.json)
and [attention/FFN implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py);
Krea [inference configuration](https://github.com/krea-ai/krea-2/blob/main/inference.py)
and [model implementation](https://github.com/krea-ai/krea-2/blob/main/mmdit.py).
The benchmark shapes are derived from these configurations and the token-count
assumptions above. [Machine-readable shape list](ane-quantization-fusion-shapes.csv).

## Isolated activation preparation

GPU timestamps cover the old scale + transpose dispatches versus one fused
dispatch. Each case uses 31 alternating AB/BA pairs, 10 repetitions per timing.
All 30 FP16/BF16 cases matched the original scales and quantized bytes exactly,
including padded rows. BF16 results are shown here; both dtypes are in the CSV.

| M × K | Two dispatches (ms) | Fused candidate (ms) | Median paired change | Production selection |
| --- | ---: | ---: | ---: | --- |
| 16384 × 14336 | 5.4242 | 4.7672 | -12.08% | Experimental |
| 16384 × 5376 | 2.0156 | 1.3514 | -32.98% | Enabled |
| 16384 × 7168 | 2.7022 | 1.8148 | -32.85% | Enabled |
| 4096 × 12288 | 1.1436 | 0.9952 | -12.96% | Experimental |
| 4096 × 14336 | 1.3491 | 1.1667 | -13.51% | Experimental |
| 4096 × 4096 | 0.3671 | 0.2233 | -39.16% | Enabled |
| 4096 × 5376 | 0.4886 | 0.3138 | -35.76% | Enabled |
| 4096 × 7168 | 0.6620 | 0.4457 | -32.73% | Enabled |
| 4096 × 9216 | 0.8550 | 0.7206 | -15.77% | Experimental |
| 4352 × 16384 | 1.6062 | 1.4114 | -12.10% | Experimental |
| 4352 × 6144 | 0.6015 | 0.4166 | -30.77% | Enabled |
| 4608 × 12288 | 1.2693 | 1.1369 | -10.46% | Experimental |
| 4608 × 16384 | 1.6961 | 1.4869 | -12.38% | Experimental |
| 4608 × 3072 | 0.3199 | 0.1695 | -46.93% | Enabled |
| 4608 × 4096 | 0.4214 | 0.2529 | -40.07% | Enabled |

## End-to-end ordinary GEMM

Q6_K weights, BF16 inputs/output. Both variants use identical packed weights.
A temporary benchmark-only switch chooses the old or fused activation path
within the same process and cached ANE program; it is removed from production.
Timings include weight decode, activation preparation, weight upload, ANE/CoreML
evaluation, output dequantization, and stream completion. Offline packing,
initial transfers, model compilation, and five warmup calls per variant are
excluded. Each case uses 31 alternating pairs, five calls per variant per pair.

The table includes experimental wide-row candidates. Rows marked **No** retain
the old two-dispatch implementation in the final production selector.
Negative change means faster. Intervals are percentile bootstraps of paired
median ratios, with 10,000 resamples and seed 42 per case. They describe these
runs; the confirmation runs show why small differences should not be treated
as universal gains.

| Case | M × N × K | Original (ms) | Fused candidate (ms) | Median paired change | 95% interval | Enabled |
| --- | --- | ---: | ---: | ---: | --- | --- |
| qwen21_ffn_up | 4096 × 12288 × 4096 | 22.6686 | 22.1580 | -1.92% | [-2.88%, -0.46%] | Yes |
| qwen21_ffn_down | 4096 × 4096 × 12288 | 27.7438 | 28.5468 | +1.04% | [-0.50%, +3.60%] | No |
| klein4_single_in | 4608 × 27648 × 3072 | 43.2494 | 42.9668 | -0.64% | [-0.71%, -0.61%] | Yes |
| klein4_single_out | 4608 × 3072 × 12288 | 23.6368 | 23.4906 | -0.62% | [-1.08%, -0.23%] | No |
| klein4_double_ffn_down | 4096 × 3072 × 9216 | 14.9276 | 14.7880 | +0.37% | [-1.58%, +2.44%] | No |
| klein9_single_in | 4608 × 36864 × 4096 | 69.6582 | 69.5992 | +1.00% | [-1.57%, +1.26%] | Yes |
| klein9_single_out | 4608 × 4096 × 16384 | 41.2576 | 41.3192 | +0.14% | [-0.18%, +0.38%] | No |
| h3_4k_q | 4096 × 7168 × 5376 | 18.5110 | 18.5440 | -0.60% | [-1.46%, +0.76%] | Yes |
| h3_4k_attention_out | 4096 × 5376 × 7168 | 19.2214 | 18.8158 | -2.22% | [-2.60%, -1.56%] | Yes |
| h3_4k_ffn_down | 4096 × 5376 × 14336 | 41.1060 | 41.3996 | +0.63% | [+0.39%, +0.80%] | No |
| h3_16k_q | 16384 × 7168 × 5376 | 67.0848 | 64.7390 | -3.48% | [-4.00%, -3.37%] | Yes |
| h3_16k_attention_out | 16384 × 5376 × 7168 | 65.0790 | 64.4980 | -3.60% | [-4.98%, +0.13%] | Yes |
| h3_16k_ffn_down | 16384 × 5376 × 14336 | 142.3786 | 142.0502 | -0.17% | [-0.61%, +0.66%] | No |
| krea2_q | 4352 × 6144 × 6144 | 19.5588 | 19.1344 | -1.81% | [-2.90%, -0.21%] | Yes |
| krea2_kv | 4352 × 1536 × 6144 | 5.6166 | 5.6416 | -1.07% | [-3.94%, +3.96%] | Yes |
| krea2_ffn_up | 4352 × 16384 × 6144 | 48.9316 | 48.5692 | -0.75% | [-0.84%, -0.71%] | Yes |
| krea2_ffn_down | 4352 × 6144 × 16384 | 55.1102 | 55.3648 | +0.45% | [+0.27%, +0.54%] | No |

## Hadamard against the final ordinary baseline

The final ordinary selector uses one dispatch through K=8192 and two above it.
Hadamard uses the revised eight-/four-row layout. Ordinary Q6_K and H256 Q6_K
weights are packed offline from the same dense weights. Each case uses 15
alternating pairs, five calls per variant. The two longer confirmations below
provide stronger evidence for the small Krea K/V and Qwen FFN-up cases.

| Case | M × N × K | Ordinary (ms) | Hadamard (ms) | Median paired overhead | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| qwen21_ffn_up | 4096 × 12288 × 4096 | 22.3440 | 22.4236 | +0.92% | [+0.05%, +1.68%] |
| qwen21_ffn_down | 4096 × 4096 × 12288 | 27.7806 | 28.8976 | +4.20% | [+0.00%, +5.47%] |
| klein4_single_in | 4608 × 27648 × 3072 | 43.1506 | 43.3504 | +0.47% | [+0.24%, +0.54%] |
| klein4_single_out | 4608 × 3072 × 12288 | 23.4670 | 24.5854 | +4.61% | [+1.80%, +12.27%] |
| klein4_double_ffn_down | 4096 × 3072 × 9216 | 15.0436 | 15.5324 | +2.62% | [-1.07%, +7.73%] |
| klein9_single_in | 4608 × 36864 × 4096 | 69.9250 | 69.7546 | -0.47% | [-3.35%, +2.06%] |
| klein9_single_out | 4608 × 4096 × 16384 | 41.2012 | 42.1454 | +2.74% | [+1.03%, +4.60%] |
| h3_4k_q | 4096 × 7168 × 5376 | 18.5270 | 18.7126 | +2.37% | [+0.16%, +3.81%] |
| h3_4k_attention_out | 4096 × 5376 × 7168 | 19.1068 | 18.9650 | -0.66% | [-1.14%, +2.35%] |
| h3_4k_ffn_down | 4096 × 5376 × 14336 | 40.9752 | 42.7924 | +4.55% | [+4.13%, +4.64%] |
| h3_16k_q | 16384 × 7168 × 5376 | 63.2248 | 63.4122 | +0.39% | [-0.69%, +4.09%] |
| h3_16k_attention_out | 16384 × 5376 × 7168 | 66.7412 | 68.8400 | +3.21% | [+2.54%, +3.73%] |
| h3_16k_ffn_down | 16384 × 5376 × 14336 | 142.9092 | 143.6944 | +1.63% | [+0.22%, +3.02%] |
| krea2_q | 4352 × 6144 × 6144 | 19.3598 | 19.2444 | -0.61% | [-0.81%, +0.83%] |
| krea2_kv | 4352 × 1536 × 6144 | 5.6038 | 5.6090 | +1.23% | [-7.14%, +16.88%] |
| krea2_ffn_up | 4352 × 16384 × 6144 | 48.6280 | 49.1534 | +1.11% | [+0.96%, +1.19%] |
| krea2_ffn_down | 4352 × 6144 × 16384 | 50.9726 | 52.2604 | +0.61% | [-0.56%, +2.54%] |

## Longer confirmations

Ordinary fusion: 51 pairs × 10 calls per variant. Hadamard: 101 pairs × 20 calls
for Krea K/V, 101 pairs × 10 calls for Qwen FFN-up. For comparison, the committed
four-row Hadamard kernel initially measured +4.38% on Krea K/V against the fused
ordinary baseline. The adaptive layout reduced this to about +2.12%, still above
the target.

| Comparison / case | M × N × K | Baseline (ms) | Candidate (ms) | Median paired change | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| ordinary_gemm / krea2_ffn_down | 4352 × 6144 × 16384 | 55.4174 | 55.3299 | -0.11% | [-0.18%, +0.04%] |
| ordinary_gemm / h3_4k_ffn_down | 4096 × 5376 × 14336 | 41.3423 | 41.3740 | +0.16% | [+0.08%, +0.21%] |
| hadamard_gemm / krea2_kv | 4352 × 1536 × 6144 | 5.7120 | 5.8361 | +2.12% | [+1.29%, +3.08%] |
| hadamard_gemm / qwen21_ffn_up | 4096 × 12288 × 4096 | 22.2474 | 22.4349 | +0.68% | [+0.34%, +1.33%] |

[All paired samples](ane-quantization-fusion-benchmark.csv) identify the experiment
and run. `primary` Hadamard samples use the final ordinary selector. The earlier
committed/adaptive spot checks and longer Hadamard confirmations have K<=8192,
so they use the same fused ordinary baseline.

## Validation and reproduction

The CPU-reference quantization probe passed 144 cases, covering FP16/BF16/FP32,
model widths, K=1 through 65537, zero rows, outliers, source offsets, nontrivial
batch strides, padding, and switching cached shapes and modes.
Temporary return-path instrumentation also confirmed 204 successful ANE calls
across all 17 shapes (12 per shape, covering both ordinary variants). The
instrumentation was removed before the final debug build and integration tests.
`make debug -j4` passed. The final `mpsblas.tests` run passed 283 tests with
one skipped, and `mpsdnn.tests` passed all 126 tests. Both suites ran with
unrestricted GPU access. `git diff --check` and a clean applicability check
of the benchmark-only patch also passed.

Build an optimized library before timing:

```sh
make -C lib -B -j4 lib
make -C bin/nnc ane_quant_fusion_probe ane_hadamard_quant_probe ane_hadamard_bench
bin/nnc/ane_hadamard_quant_probe
bin/nnc/ane_quant_fusion_probe 4352 6144 121 selected
bin/nnc/ane_quant_fusion_probe 4352 16384 121 selected
# Omit "selected" to sweep experimental tile layouts and retention strategies.
bin/nnc/ane_hadamard_bench 4352 1536 6144 101 20 2
```

The probe's `selected` mode measures the fused shader directly, including the
experimental wider variant. It does not imply that wider fusion is enabled by
the production selector. Precision 121 in the GPU probe, or 2 in the GEMM
benchmark, selects BF16.

To reproduce the old/new paired GEMM comparison, the accompanying
[benchmark-only patch](ane-quantization-fusion-e2e.patch) adds the temporary
switch, a legacy scale pipeline, and the comparison program. It also enables
the experimental wider fused candidate for comparison. Apply it only for the
benchmark, then reverse it and rebuild:

```sh
git apply doc/ane-quantization-fusion-e2e.patch
make -C lib -j4 lib
make -C bin/nnc TARGETS=ane_quant_fusion_e2e ane_quant_fusion_e2e
bin/nnc/ane_quant_fusion_e2e 4352 6144 16384 51 10 2
git apply -R doc/ane-quantization-fusion-e2e.patch
make -C lib -j4 lib
```
