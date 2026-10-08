# Fused H128 Q/K quantization for NAInt8Attention

Normalized Walsh–Hadamard rotation fits inside the existing Q/K quantization launches. On the tested M5 Max, long forward attention incurs about 0–1% total overhead; short attention and decode have a measurable cost. This report records the initial D=128 experiment. The implementation has since been generalized to `NAInt8AttentionDescriptor::qkHadamard`, default false, with forward block transforms for D=8..256 in multiples of eight. See the [other-D and M5 iPad Pro follow-up](na-int8-attention-hadamard-cross-device.md) for current packing, cross-device results, and the SDPA `use_hadamard` option. The frontend honors this option only during supported NA INT8 forward attention; backward ignores it and recomputes in the original basis.

## Implementation

One SIMD group transforms one 128-channel row. Each lane loads four adjacent values, performs two local butterfly stages, then five XOR-shuffle stages across the 32 lanes. FP32 registers retain four Q rows or eight K rows per thread across the tile-wide absmax reduction. This replaces the native quantizer's two source reads with one source read and retains the existing Q16/K64 scaling granularity.

For the unnormalized transform `y = H128 x`, quantization uses `round(y * 127 / max_abs(y))` and stores the dequantization scale `max_abs(y) / (127 * sqrt(128))`. Folding normalization into the scale avoids an extra multiply before rounding. Q and K use the same orthonormal basis, preserving their exact dot product before quantization. V stays in the original basis, so the attention output needs no inverse transform.

No extra device scratch or kernel launches are required. Q/K use the existing 128/256-thread dispatches and 16/32-byte reduction scratch (32/64 bytes under shader instrumentation). The transform flag participates in both the source-kernel key and execution-descriptor key. Runtime sequence lengths remain function constants/runtime bindings. The attention, V quantization, and V-mean source and algorithms are unchanged.

This differs numerically from the older separate-pass experiment: transformed rows remain FP32 until INT8 rounding, rather than being rounded into an intermediate FP16 buffer.

## Measurement

Apple M5 Max, macOS 27.0.1 (26A434), Xcode 27.0 (27A266a), 2026-10-07. The probe builds the production descriptors and pipelines and manually encodes their normal bindings. It measures GPU command-buffer timestamps, excluding CPU encoding, allocation, pipeline compilation, and the CPU reference checks. Full forward includes Q/K quantization, V-mean, V quantization, and attention.

Both variants share input and timing scratch buffers. Unchanged stages use the same baseline pipeline objects in both variants; the attention source suffix is also checked for equality. Each mode warms for at least five rounds and 150 ms of aggregate GPU time, then alternates baseline/rotated execution order for 64 pairs. Q/K-only command buffers repeat the two launches eight times. Full forward runs once per command buffer. Separate medians can drift, so overhead is the median of paired rotated/baseline ratios, rather than the ratio of separate medians. Negative values at this scale should not be treated as reliable speedups.

Inputs are synthetic, seed 42. The long cases below use Gaussian values with one Q/K channel amplified 10x. Short and decode cases use ordinary Gaussian values. These are timing shapes, including two H3 sequence lengths, rather than captured H3 tensors. FP16 uses low-precision intermediates; BF16/FP32 use FP32 intermediates.

| R × C | Hq / Hk | Precision | Q/K baseline → H128 ms | Q/K paired overhead | Full baseline → H128 ms | Full paired overhead |
|---|---|---|---|---|---|---|
| 4096 × 4096 | 24 / 24 | FP16 | 0.14204 → 0.14348 | +1.26% | 2.61998 → 2.62608 | +0.22% |
| 8192 × 8192 | 32 / 32 | FP16 | 0.38906 → 0.39034 | +0.33% | 14.60194 → 14.75735 | +0.02% |
| 9942 × 9942 | 56 / 56 | FP16 | 0.80721 → 0.80334 | −0.22% | 50.83962 → 51.08402 | +0.59% |
| 21782 × 21782 | 56 / 56 | FP16 | 1.72509 → 1.71333 | −0.66% | 241.80758 → 244.30802 | +0.92% |
| 29 × 29, causal | 32 / 32 | FP16 | 0.00397 → 0.00512 | +28.88% | 0.03125 → 0.03596 | +14.98% |
| 1 × 4097, causal | 32 / 8 | FP16 | 0.01762 → 0.02258 | +27.73% | 0.29921 → 0.30696 | +2.63% |
| 4096 × 4096, runtime dimensions | 24 / 24 | BF16 | 0.14383 → 0.14639 | +1.73% | 3.31510 → 3.31273 | +0.26% |
| 4096 × 4096, runtime dimensions | 24 / 24 | FP32 | 0.24323 → 0.24124 | −0.43% | 3.35994 → 3.36004 | −0.01% |

The measurements are consistent with input traffic limiting long Q/K quantization: retaining the transformed rows saves a source read, largely offsetting the butterfly arithmetic. At short shapes that saving is less useful. Full timings also include the effect of changed quantized values on attention, rather than measuring transform arithmetic alone.

## Accuracy and validation

The probe compares sampled quantization tiles for every batch and head with the existing CPU Walsh–Hadamard command, using input values rounded to the requested storage precision. It checks INT8 outputs within one count to allow rounding ties and scale relative error below 1e-5. Observed scale error is about 1e-7. It separately reports original-basis CPU SDPA error at first/middle/last queries and first/last heads. SDPA error is a quality metric, not an asserted global accuracy bound.

| Synthetic case | Native sampled relative L2 | H128 sampled relative L2 |
|---|---|---|
| 4096 square, H24, channel outliers | 3.328% | 2.975% |
| 8192 square, H32, channel outliers | 2.732% | 2.002% |
| 9942 square, H56, channel outliers | 6.065% | 5.734% |
| 21782 square, H56, channel outliers | 5.041% | 4.096% |
| 257 × 513, B2, H8/H2, FP16, channel outliers | 1.999% | 1.759% |
| 65 × 33, B3, H8/H2, causal varlen, channel outliers | 1.424% | 1.101% |

Lower aggregate L2 does not imply lower peak error. In the FP16 257 × 513 tail case, sampled max absolute error rises from 0.00877 to 0.01417. Ordinary Gaussian 4096-square inputs in the initial sweep also slightly regress in sampled L2 (1.206% to 1.236%). Captured model tensors from the previous separate-pass experiment were unavailable, so its model-quality improvement has not been revalidated for this fused NA path.

Validation completed:

- Debug integration build and standalone probe build passed; `git diff --check` passed.
- CPU quantizer checks passed for FP16/BF16/FP32, static/runtime dimensions, partial tiles, grouped heads, batches, unequal variable lengths, causal empty rows, and all-zero inputs. Zero-input attention returns exact zeros.
- Metal API and shader validation passed on the changed Q/K kernels for partial tiles and variable lengths in all three precisions. Complete aligned 256 × 512 forward attention also passed validation.
- Existing quantized forward attention tests passed 11/11; the separate NA varlen test passed 1/1. DNN passed 126/126.
- The full BLAS suite stops in the H256 rowwise-X dense GEMM fallback, where `GEMMKernel` fails to compile a `simdgroup_matrix_storage<float>` method. The quantized backward filter separately fails an MPP `get_destination_cooperative_tensor` template instantiation. Both failures reproduce when relinking with the pre-fusion `NAInt8AttentionKernel.cpp` from HEAD; the GEMM files are unchanged.
- Full tail/varlen shader validation encounters a baseline `int8_attention` tensor-slice bounds error before the rotated variant runs. Tail quantizer validation is therefore performed independently with `CCV_NA_QUANT_ONLY=1`; full tail attention passes normal execution but is not claimed to pass shader validation.

Raw timings, exact shapes, CPU checks, and failure logs are in [the experiment log](benchmarks/na-int8-attention-hadamard128-2026-10-07.txt).

## Reproduction

```sh
make -C test/int/nnc debug -j4
make -C bin/mfa na_int8_attention_hadamard_bench
CCV_NA_SAMPLES=64 bin/mfa/na_int8_attention_hadamard_bench 4096 4096 1 24 24 0 0 0 1 0
```

Arguments are `R C B Hq Hk precision causal dynamic_flags [distribution=0] [varlen=0] [D=128]`. Precision is 0=FP16, 1=BF16, 2=FP32. Runtime-dimension flags are 1=R, 2=C, 3=both. Distribution is 0=Gaussian, 1=channel outliers, 2=uniform, 3=zeros. Variable lengths shorten successive batches by `batch * 13 % length` and use packed Q/K offsets.

For Q/K-only Metal validation:

```sh
CCV_NA_SAMPLES=2 CCV_NA_QUANT_ONLY=1 \
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
bin/mfa/na_int8_attention_hadamard_bench 257 513 2 8 2 0 0 3 1 1
```

Timing with validation enabled is not comparable to the performance table. This probe requires hardware with neural accelerators and Metal access; the restricted sandbox does not expose the device on this workspace.
