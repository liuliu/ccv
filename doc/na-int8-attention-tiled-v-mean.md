# Tiled Morton V-mean for NAInt8Attention

The native `compute_v_mean` is replaced with one tiled Morton reduction for forward attention and backward preprocessing. It still takes one launch and no additional device scratch. Unmasked forward attention still uses five launches.

Adjacent threads load adjacent channel vectors. A threadgroup-local transpose preserves the previous per-thread accumulation order and XOR reduction tree. Small head counts use narrower tiles to retain parallelism; there is no alternate legacy shader or shape-gated fallback. Tile width depends only on existing kernel cache properties (Hk, D, reduction threads). Batch count is supplied when computing the dispatch grid.

The partial sums and final SIMD-group sums reuse one fixed 16 KiB array. Its size does not grow with H, D, batch count, or sequence length. The tested Metal shader validator doubles this allocation to 32 KiB, matching the M5 Max device cap. Pipeline creation checks the compiled static allocation and maximum thread count against the device/pipeline limits. An earlier separate-sum-buffer layout used 16.5 KiB and failed validation at 33 KiB; it is not the committed layout.

## Comparison protocol

Apple M5 Max, macOS 26.6.2 (25G83), Xcode 26.6 (17F113). The benchmark compiles the previous mean shader alongside the production shader, alternates old/new ordering each round, warms each variant four times, and records GPU timestamps. Each table row uses 20 measured rounds except the 103982 square case, which uses 10. Validation is disabled for timing. Raw round timings and exact commands are in [the benchmark log](benchmarks/na-int8-attention-tiled-v-mean-2026-09-22.txt).

Both mean-only and full-attention modes check bitwise equality. Full mode runs all five production pipelines and bindings, changing only the V-mean shader and its dispatch. It uses synthetic uniform V values (seed 42); Q/K are distinct rotations of those values, scale=1, with equal query/KV head counts. It does not measure captured model quality. FP16 uses low-precision intermediates; BF16/FP32 use FP32 intermediates. Reported times are separate medians, while speedup is the median of paired old/new ratios; these need not equal the ratio of the time medians when clocks drift.

## V-mean timings

| N | R | C | H | D | Precision | Previous ms | Tiled ms | Paired speedup |
|---|---|---|---|---|---|---|---|---|
| 1 | — | 128 | 56 | 128 | 16F | 0.00900 | 0.00733 | 1.222× |
| 1 | — | 1024 | 56 | 128 | 16F | 0.04525 | 0.02300 | 1.965× |
| 1 | — | 4096 | 56 | 128 | 16F | 0.18163 | 0.13027 | 1.417× |
| 1 | — | 16384 | 56 | 128 | 16F | 0.97256 | 0.61140 | 1.581× |
| 1 | — | 20480 | 56 | 128 | 16F | 1.71663 | 0.75175 | 2.242× |
| 1 | — | 20481 | 56 | 128 | 16F | 1.69033 | 0.60548 | 2.817× |
| 1 | — | 32768 | 56 | 128 | 16F | 3.04246 | 0.95696 | 3.211× |
| 1 | — | 65536 | 56 | 128 | 16F | 6.38310 | 1.94910 | 3.249× |
| 1 | — | 103982 | 56 | 128 | 16F | 11.82569 | 3.32363 | 3.612× |
| 1 | — | 32768 | 1 | 128 | 16F | 0.05994 | 0.06023 | 0.994× |
| 2 | — | 32768 | 3 | 128 | 16F | 0.21031 | 0.21596 | 0.966× |
| 1 | — | 32768 | 8 | 128 | 16F | 0.25283 | 0.21817 | 1.202× |
| 1 | — | 32768 | 56 | 64 | 16F | 1.17410 | 0.52804 | 2.217× |
| 1 | — | 32768 | 56 | 80 | 16F | 1.92912 | 0.86438 | 2.213× |
| 1 | — | 32768 | 56 | 192 | 16F | 6.08496 | 1.64888 | 3.711× |
| 1 | — | 32768 | 56 | 256 | 16F | 9.56538 | 2.09448 | 4.446× |
| 1 | — | 32768 | 56 | 128 | 16BF | 2.98625 | 0.96071 | 3.146× |
| 1 | — | 32768 | 56 | 128 | 32F | 3.85310 | 1.80015 | 2.130× |
| 2 | — | 257 | 3 | 8 | 16F | 0.00275 | 0.00287 | 0.957× |
| 2 | — | 257 | 3 | 80 | 16BF | 0.00350 | 0.00373 | 0.950× |
| 2 | — | 257 | 3 | 192 | 32F | 0.00481 | 0.00535 | 0.902× |

## Full-attention timings

| N | R | C | H | D | Precision | Previous ms | Tiled ms | Paired speedup |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 128 | 56 | 128 | 16F | 0.03775 | 0.03675 | 1.031× |
| 1 | 1024 | 1024 | 56 | 128 | 16F | 0.71275 | 0.70806 | 1.018× |
| 1 | 4096 | 4096 | 56 | 128 | 16F | 7.60392 | 7.57365 | 1.011× |
| 1 | 32768 | 32768 | 56 | 128 | 16F | 649.83725 | 663.15171 | 0.999× |
| 1 | 65536 | 65536 | 56 | 128 | 16F | 4362.33090 | 4397.94558 | 1.006× |
| 1 | 17 | 32768 | 56 | 128 | 16F | 7.78963 | 5.54267 | 1.367× |
| 2 | 17 | 32768 | 3 | 80 | 16F | 1.61587 | 1.61031 | 1.001× |
| 1 | 4096 | 32768 | 8 | 128 | 32F | 10.55044 | 10.37371 | 1.036× |
| 1 | 103982 | 103982 | 56 | 128 | 16F | 9558.29817 | 9709.85862 | 0.988× |

The large H56 V-mean speedup does not translate to a comparable square-attention speedup: matrix multiplication dominates those cases. The short-query, long-KV case benefits more. Small shapes can incur sub-microsecond mean overhead; this is not a universal latency win.

## Reproduction

Build `lib/libccv.a` with `make -C test/int/nnc debug -j4`, then use the standalone compile command at the top of `bin/mfa/na_int8_v_mean_bench.cpp`. Its arguments are `C H N rounds D precision full R`; omit the final two arguments for mean-only timing.

For validation, enable `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1`. These are the [Apple-documented shader-validation settings](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage). Timing under these settings is not comparable to normal execution.

## Validation

- Debug integration build: passed.
- Metal API + shader validation: 8/8 quantized forward tests and 1/1 variable-length test passed. This includes the added long-KV regression covering both reduction sizes, non-power-of-two heads, grouped queries, batches, different D, and runtime dimension caching.
- Five additional API + shader validation probes passed with bitwise equality. They cover partial channel tiles, the 20480/20481 reduction boundary, FP16/BF16/FP32, batched full attention, and the scalar D=130 mean path. All reported 32768 bytes of instrumented static shared memory against a 32768-byte device cap. See [validation output](benchmarks/na-int8-attention-tiled-v-mean-validation-2026-09-22.txt).
- The broader `quantized` filter reaches an existing backward API-validation failure after the eight forward tests: backward key/value requests 65536 bytes of dynamic threadgroup memory on this 32768-byte-cap device. The allocation function is byte-identical to the parent commit (`128 * 16 * 16 * 2`). This change does not repair that separate backward allocation issue; the entire backward API-validation suite is not claimed to pass.
- The existing backward probe was updated to the new dispatch helpers and current descriptor arguments. Its autoreleased command objects now use retained ownership, and its pool outlives those objects. It builds and exits 0 under API + shader validation at R=C=128, Hq=Hk=56, D=64. These concurrent-validation smoke timings are not performance comparisons.
- Complete normal MPS suites: BLAS 284/284 and DNN 126/126 passed (both exit 0), including forward and backward attention.
