# Fused Q/K Hadamard: other head dimensions and M5 iPad Pro

The fused rotation remains inexpensive for long forward attention across the tested head dimensions. For FP16 4096-square attention with 24 heads, full forward overhead is 0.2–0.6% on M5 Max and approximately 0–0.2% on the connected M5 iPad Pro. Decode costs depend more strongly on D: D=192 adds about 9% on the Mac and 3% on the iPad. These timings do not justify enabling every shape by default.

## Supported transform

`NAInt8AttentionDescriptor::qkHadamard`, default false, enables forward-only rotation for multiples of eight between D=8 and D=256. The block size is the largest power-of-two divisor of D. Powers of two use a full-head transform; other dimensions use independent contiguous blocks. The normalized block-diagonal transform is orthogonal, so it preserves unquantized Q/K dot products. Q16/K64 scale tiles, unrotated V, the five forward launches, and device scratch sizes are retained.

| D | Transform | Values per lane per load |
|---|---|---|
| 64 | H64 | 2 |
| 80 | 5 × H16 | 4 |
| 96 | 3 × H32 | 4 |
| 128 | H128 | 4 |
| 192 | 3 × H64 | 2 |
| 256 | H256 | 4 |

The two D=256 chunks undergo independent H128 butterflies and one final cross-chunk butterfly. D=64/192 use two-wide accesses to fill SIMD groups and reduce register pressure. D=80/96 use four-wide accesses, which measured faster than scalar packing. Normalization is folded into the scale as `max_abs / (127 * sqrt(block_size))`, using the block size rather than D for non-power-of-two heads. The probe uses the existing CPU WHT implementation as its reference.

The SDPA command exposes `info.scaled_dot_product_attention.use_hadamard`, default zero. The CNNP constructor takes `use_hadamard` immediately after `flags` and preserves it when copying a model. Only NA INT8 forward attention honors the option, for the supported dimensions above; other implementations, unsupported dimensions, and backward ignore it. This option does not change backend selection or automatically select shapes for rotation.

Backward recomputes Q/K in the original basis. The exact dot product and its derivatives are invariant under the orthogonal transform, so ignoring this hint does not require an inverse rotation of gradients. Quantization makes the recomputed logits differ from the rotated forward logits; backward therefore remains an approximation to that forward result. Improved accuracy is input-dependent.

```c
ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_8I;
cmd.info.scaled_dot_product_attention.use_hadamard = 1;
```

## Protocol

Measurements on 2026-10-07: MacBook with Apple M5 Max, macOS 27.0.1 (26A434); physical iPad Pro 11-inch (M5, iPad17,1), reported device name `Apple M5 GPU`, iPadOS 27.0.1 (24A446). Xcode 27.0 (27A266a). Every reported iPad process begins and ends with nominal thermal state (`0`).

The [probe](../bin/mfa/na_int8_attention_hadamard_bench.cpp) uses production descriptors and GPU pipelines, manually encoding their normal bindings. Both variants share input buffers and timing scratch addresses and use identical pipeline objects for unchanged V-mean, V quantization, and attention stages. GPU timestamps exclude compilation, allocation, CPU encoding, and reference checks. Warmup requires at least five rounds and 150 ms of aggregate GPU time; each main result uses 48 alternating-order pairs. Q/K-only buffers repeat the two launches eight times. Full forward includes all five stages once.

The iPad app builds the same production kernel files with an embedded probe. It links the CPU precision conversions and includes the existing CPU WHT row helper directly, avoiding the full command registry. Its unused ANE rowwise cleanup and command-flag accessors are stubbed; neither affects these GPU descriptors or pipelines.

Overhead is the median of paired rotated/baseline ratios. Separate time medians need not have the same ratio when clocks drift. Values close to zero should be read as approximately unchanged. Inputs are synthetic with seed 42: channel-outlier Gaussian Q/K for square attention and ordinary Gaussian values for decode. Captured model quality remains unverified.

## Full forward results

Square: R=C=4096, B=1, Hq=Hk=24. Decode: R=1, C=4097, B=1, Hq=32, Hk=8, causal with runtime dimensions. All rows are FP16 with low-precision intermediates.

| D | M5 Max square overhead | M5 iPad square overhead | M5 Max decode overhead | M5 iPad decode overhead |
|---|---|---|---|---|
| 64 | +0.23% | +0.16% | +2.52% | +3.03% |
| 80 | +0.26% | +0.07% | +2.95% | +3.03% |
| 96 | +0.30% | −0.01% | +2.46% | +1.64% |
| 128 | +0.17% | +0.01% | +2.50% | +0.81% |
| 192 | +0.61% | +0.11% | +9.23% | +2.91% |
| 256 | +0.53% | +0.01% | +6.06% | +1.25% |

| D | M5 iPad square baseline → rotated ms | M5 iPad decode baseline → rotated ms |
|---|---|---|
| 64 | 5.84900 → 5.85710 | 0.24300 → 0.25065 |
| 80 | 9.04171 → 9.04879 | 0.31521 → 0.32508 |
| 96 | 8.88942 → 8.89650 | 0.36825 → 0.37510 |
| 128 | 10.88704 → 10.88879 | 0.48698 → 0.49065 |
| 192 | 20.05990 → 20.07706 | 0.73944 → 0.75983 |
| 256 | 32.31350 → 32.33898 | 0.87963 → 0.89181 |

Two longer iPad checks (8192 square, B=1, H=24, 24 pairs) support the same conclusion: D=128 is 41.76450 → 41.79825 ms with −0.01% paired overhead; D=256 is 125.79985 → 125.83402 ms with +0.12% overhead. Short 29-token causal attention with H=32, D=128 costs +7.64%, about 2.8 microseconds (0.03606 → 0.03887 ms).

## Q/K-only overhead

| D | M5 Max square | M5 iPad square | M5 Max decode | M5 iPad decode |
|---|---|---|---|---|
| 64 | +17.37% | +1.27% | +57.61% | +19.82% |
| 80 | +11.56% | +1.55% | +38.06% | +15.34% |
| 96 | +5.29% | +0.69% | +19.34% | +10.92% |
| 128 | +1.12% | −0.54% | +27.08% | +5.82% |
| 192 | +13.73% | +0.82% | +112.85% | +21.49% |
| 256 | +3.02% | −2.00% | +46.39% | +5.01% |

The transform itself is not free on every shape. Arithmetic and register pressure are more visible on the Mac, especially D=192 decode. Long attention dilutes the cost. Higher bandwidth/core count and cache behavior are plausible explanations for the device difference, but these measurements do not isolate those causes.

## Validation

- Debug integration build and signed iPad app build passed. `git diff --check` passed.
- CPU quantizer comparisons passed for all 32 multiples of eight from D=8 through D=256, using grouped heads, batches, runtime lengths, tails, and packed unequal varlen sequences. INT8 agreement is within one rounding count and scale relative error is approximately 1e-7.
- On the Mac, Metal API and shader validation passed Q/K-only tail/varlen probes for D=64/80/96/128/192/256 in FP16, BF16, and FP32. The final two-wide D=192 path was revalidated in all three precisions.
- All iPad main cases passed CPU quantizer and finite-output checks. Additional batched runtime-shape BF16/FP32 checks passed at D=80/192/256. Sampled original-basis CPU SDPA error is reported in each log; improved L2 on synthetic outliers is not a global accuracy guarantee. Several ordinary-Gaussian decode cases slightly regress in L2.
- Existing forward quantized attention tests passed 11/11, and the NA varlen test passed 1/1 after the final build.
- A full D=8 varlen baseline probe stalled in `MTLCommandBuffer::waitUntilCompleted` before rotating Q/K; it was stopped. D=8 Q/K-only CPU checks pass. Full attention is not claimed validated for all 32 quantizer dimensions. The initial [D=128 report](na-int8-attention-hadamard128.md) also records the unrelated full-BLAS/backward compile failures and baseline full-tail shader-validation failure.

Raw selected logs, exact CLI arguments, and build/source metadata are in [the benchmark directory](benchmarks/na-int8-attention-hadamard-cross-device-20261007/metadata.json).

Frontend integration was checked on the Mac on 2026-10-08. The debug integration build passed, all 12 forward quantized-attention tests passed, the expanded NA varlen test passed with the option both off and on, CPU attention tests passed 8/8, and MPS DNN passed 126/126. The new public-option test checks activation across the six common supported dimensions, ignored hints at D=130/320 and on CPU / floating-point attention, off/on/off cache reuse, numeric masks, sinks, runtime shapes, and CNNP model copying. FP16/BF16/FP32 forward comparisons also run with the option off and on.

The full MPS BLAS and quantized-backward runs still encounter the compiler failures recorded above. An additional generic MPS grouped-query backward check hits `NAAttentionKernel`'s MPP `get_destination_cooperative_tensor` template error; an isolated test binary with `use_hadamard=0` reproduces the identical error. GPU gradient accuracy after a rotated quantized forward has therefore not been validated by this integration run.

## Generic butterfly refactor and bit identity

On 2026-10-08, the Metal body was changed to derive its vector types, chunks, rows, and shuffle limit from vector width and Hadamard block size. Fully unrolled component and chunk butterflies replace the C++ strings for width-2/4 transforms and the special D=256 step. Hadamard-specific `SetValue` calls decrease from 13 to 6. Normalization, packing selection, quantizer bindings, and scale granularity stay the same.

Direct GPU comparisons on M5 Max found zero differences across 388 cases: 212,708,352 INT8 elements and 65,664 FP32 scales matched bit for bit. The 384 small cases cover all 32 multiples of eight from D=8 through D=256, FP16/BF16/FP32, aligned sequences, tails, packed unequal varlen sequences, runtime dimensions, grouped heads, Gaussian/outlier/uniform inputs, and zeros. Four larger FP16 cases use 4096-square attention with 24 heads at D=64/128/192/256.

The old generated Metal was captured before the refactor. Both shader versions then quantized the same input buffers in one process. Every allocated Q/K INT8 element and every scale's `uint32` bit pattern was compared without tolerance, including zero-initialized unused packed capacity. Existing CPU tolerance checks also passed. This verifies the tested inputs and M5 Max compiler/device combination; the refactor was not rechecked on iPad.

Paired old/new timings use 48 alternating-order pairs and the existing warmup protocol. Here both variants already apply Hadamard; `baseline` in these logs means the pre-refactor implementation, and `hadamard` means the generic implementation.

| D | Generic Q/K time change | Generic full-forward time change |
|---|---|---|
| 64 | +0.43% | −0.01% |
| 128 | −0.04% | +0.14% |
| 192 | −0.27% | +0.03% |
| 256 | +0.55% | +0.29% |

These small differences do not establish a speedup. The debug integration build, public `use_hadamard` test, and expanded NA varlen test passed after the refactor.

[Verification metadata](benchmarks/na-int8-attention-hadamard-generic-20261008/verification.json), raw checks, paired timing logs, a reversible source patch, and the comparator are archived together. The [build script](benchmarks/na-int8-attention-hadamard-generic-20261008/build.py) reconstructs the old C++ source in temporary storage, checks its SHA256, and gives the old generator separate symbols. Three reproduction checks confirm that it emits exactly the captured old Metal and produces bit-identical quantized buffers.

```sh
python3 doc/benchmarks/na-int8-attention-hadamard-generic-20261008/build.py --output /private/tmp/ccv-hadamard-bit-identity
CCV_NA_QUANT_ONLY=1 CCV_NA_COMPARE_ONLY=1 /private/tmp/ccv-hadamard-bit-identity/hadamard_compare 65 129 3 6 2 1 1 3 2 1 256
```

## Reproduction

```sh
make -C test/int/nnc debug -j4
make -C bin/mfa na_int8_attention_hadamard_bench
CCV_NA_SAMPLES=48 bin/mfa/na_int8_attention_hadamard_bench 4096 4096 1 24 24 0 0 0 1 0 256
```

The last optional CLI argument is D; omitted D defaults to 128. For iPad, the saved [build script](benchmarks/na-int8-attention-hadamard-cross-device-20261007/ios/build.py), harness, and project template create a separate app with bundle ID `com.liu.NAHadamardAttentionExperiment`. Signing requires a configured development team; `--team` can override the template team.

```sh
python3 doc/benchmarks/na-int8-attention-hadamard-cross-device-20261007/ios/build.py --output /private/tmp/ccv-attention-hadamard-repro
xcrun devicectl device install app --device <device-id> /private/tmp/ccv-attention-hadamard-repro/derived/Build/Products/Release-iphoneos/NAInt8TuningApp.app
xcrun devicectl device process launch --device <device-id> --terminate-existing --console \
  --environment-variables '{"CCV_NA_SAMPLES":"48"}' \
  com.liu.NAHadamardAttentionExperiment 4096 4096 1 24 24 0 0 0 1 0 256
```

This NA experiment requires the backend's Apple10 neural-accelerator capability. Results above are from the physical M5 iPad Pro, rather than a simulator or another iPad generation.
