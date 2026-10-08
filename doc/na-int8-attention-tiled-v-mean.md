# NAInt8Attention V-mean reduction

## Chunked reduction (2026-10-08)

`compute_v_mean` now splits the KV sequence into 512-row chunks: one threadgroup per (chunk, KV head, operand, batch). Each SIMD group reads complete head rows, so a cache line is consumed by one SIMD group instead of being split across the channel tiles of different threadgroups. Rows shorter than a SIMD group are packed several to a SIMD group. Each lane issues four rows' loads before accumulating them, then packed rows fold with a fixed XOR shuffle tree and SIMD groups combine in index order. When one chunk covers the sequence, the threadgroup writes the mean directly; longer sequences write per-chunk sums and `finalize_v_mean` adds them in chunk order. The result is deterministic for a given shape, but the summation order differs from the tiled reduction below, so means are no longer bitwise identical to it.

Plain V centering and Hadamard K+V centering use the same shader. Hadamard adds K as a second operand of the same dispatch; with one operand the K arguments are dead code and need no binding. `NAInt8AttentionKernel::encodeVMean` encodes both passes for forward attention, backward preprocessing and the benchmarks.

Costs relative to the tiled reduction: sequences longer than 512 rows add a second small launch and `B * operands * Hk * chunks * D * 4` bytes of scratch. The smallest measured shapes (N=2, C=257, H=3) are 0.25–1.1 µs slower.

### V-mean timings

Apple M5 Max (40 GPU cores), macOS 27.0.1, Xcode 27.0. `bin/mfa/na_int8_v_mean_bench.cpp` built against the parent library (tiled reduction with Hadamard K-centering) and against this change, run alternately per shape. Both now warm up for at least four rounds and 0.5 s of wall time; four fixed rounds left short dispatches at low GPU clocks. Medians of 30 rounds. Against an FP64 CPU mean, the largest chunked error is 5.4e-8 and the original single-pass shader's is 3.6e-8; both are float rounding. H is both Hq and Hk.

| N | C | H | D | Precision | Tiled ms | Chunked ms | Speedup |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 56 | 128 | 16F | 0.00738 | 0.00450 | 1.64× |
| 1 | 512 | 56 | 128 | 16F | 0.01363 | 0.00967 | 1.41× |
| 1 | 513 | 56 | 128 | 16F | 0.01379 | 0.01067 | 1.29× |
| 1 | 1024 | 56 | 128 | 16F | 0.02285 | 0.01988 | 1.15× |
| 1 | 4096 | 56 | 128 | 16F | 0.13117 | 0.09367 | 1.40× |
| 1 | 16384 | 56 | 128 | 16F | 0.59554 | 0.41690 | 1.43× |
| 1 | 20481 | 56 | 128 | 16F | 0.58302 | 0.52383 | 1.11× |
| 1 | 32768 | 56 | 128 | 16F | 0.91329 | 0.83121 | 1.10× |
| 1 | 65536 | 56 | 128 | 16F | 1.80137 | 1.68244 | 1.07× |
| 1 | 32768 | 1 | 128 | 16F | 0.06302 | 0.01404 | 4.49× |
| 2 | 32768 | 3 | 128 | 16F | 0.21219 | 0.06954 | 3.05× |
| 1 | 32768 | 8 | 128 | 16F | 0.21931 | 0.11667 | 1.88× |
| 1 | 16384 | 8 | 128 | 16F | 0.07067 | 0.04440 | 1.59× |
| 1 | 4131 | 12 | 128 | 16F | 0.02517 | 0.01725 | 1.46× |
| 1 | 32768 | 56 | 64 | 16F | 0.53092 | 0.41725 | 1.27× |
| 1 | 32768 | 56 | 80 | 16F | 0.84894 | 0.52565 | 1.62× |
| 1 | 32768 | 56 | 192 | 16F | 1.50048 | 1.25031 | 1.20× |
| 1 | 32768 | 56 | 256 | 16F | 1.89063 | 1.65360 | 1.14× |
| 1 | 32768 | 56 | 128 | 16BF | 0.91283 | 0.83088 | 1.10× |
| 1 | 32768 | 56 | 128 | 32F | 1.72769 | 1.64367 | 1.05× |
| 1 | 32768 | 8 | 130 | 16F | 0.30702 | 0.11292 | 2.72× |
| 2 | 257 | 3 | 8 | 16F | 0.00275 | 0.00296 | 0.93× |
| 2 | 257 | 3 | 80 | 16BF | 0.00375 | 0.00502 | 0.75× |
| 2 | 257 | 3 | 192 | 32F | 0.00535 | 0.00600 | 0.89× |

The benchmark rereads one input buffer, so these are cache-warm reduction times. Production places the mean between quantization passes; the full-pipeline numbers below include that.

### Hadamard K-centering cost

`bin/mfa/na_int8_attention_hadamard_bench` with `center_pair=1`: FP16, B=1, D=128, 30 paired samples, full five-stage pipeline. Uncentered Hadamard has a V-only mean; centered Hadamard reduces K and V in one dispatch.

| R | C | Hq/Hkv | Causal | Centered, tiled ms | Centered, chunked ms | Centering overhead, tiled → chunked |
|---|---|---|---|---|---|---|
| 1 | 16384 | 32/8 | no | 1.088 | 0.973 | 16.3% → 8.8% |
| 128 | 32768 | 32/8 | yes | 2.416 | 2.211 | 12.0% → 7.1% |
| 512 | 8192 | 32/8 | yes | 1.121 | 1.076 | 4.9% → 2.2% |
| 1024 | 1024 | 48/12 | no | 0.387 | 0.381 | 2.3% → 0.7% |
| 4096 | 4096 | 32/8 | yes | 2.086 | 2.061 | 1.4% → 2.1% |
| 4131 | 4131 | 48/12 | no | 5.683 | 5.697 | 0.4% → 0.3% |

Overheads for R ≥ 4096 moved by about one percentage point between runs; there the centered times are within noise of each other. At R=1, C=16384 the remaining 0.08 ms is mostly the extra full read of K in the mean pass (+0.065 ms), partly offset by K being cache-warm for its quantizer (−0.022 ms), plus V being read further ahead of its quantizer (+0.031 ms). Exact centering requires that K read; it is the floor for this design. These numbers are from one M5 Max; the lower-bandwidth iPad was not measured.

### Validation

- Debug integration build passes. All forward NA INT8 attention tests pass, including the extended V-mean reduction-size test (512/513 chunk boundary, scalar D=130, runtime C), the Hadamard K-centering test with multi-chunk C=1543, and the new mean-chunk test.
- The new test runs varlen and dense attention over KV lengths 1100, 37 and 513 (multiple chunks, empty trailing chunks, a one-row tail) with per-sequence K/V offsets and a query component that amplifies K quantization error. Forcing the mean to zero, or dropping chunk 0 in `finalize_v_mean`, raises its maximum error from 0.003 to about 0.11 against a 0.01 tolerance.
- Under Metal API and shader validation, faults are reported only in the existing `int8_attention` tensor-slice bounds issue and the generic `attention` kernel, never in `compute_v_mean` or `finalize_v_mean`. There are no missing-binding errors for the unused K arguments.
- The replayed 9-chunk K mean on Krea capture block 0 (C=4131) matches an FP64 mean within 3.7e-5 on means up to 596.
- The gradient tests and the generic (non-INT8) attention tests stop on pre-existing Metal compile errors in `NAInt8AttentionKernel` backward and `AttentionKernel`, so the backward preprocessing path was not executed.

## Tiled Morton reduction (2026-09-22, superseded)

The native `compute_v_mean` is replaced with one tiled Morton reduction for forward attention and backward preprocessing. It still takes one launch and no additional device scratch. Unmasked forward attention still uses five launches.

Adjacent threads load adjacent channel vectors. A threadgroup-local transpose preserves the previous per-thread accumulation order and XOR reduction tree. Small head counts use narrower tiles to retain parallelism; there is no alternate legacy shader or shape-gated fallback. Tile width depends only on existing kernel cache properties (Hk, D, reduction threads). Batch count is supplied when computing the dispatch grid.

The partial sums and final SIMD-group sums reuse one fixed 16 KiB array. Its size does not grow with H, D, batch count, or sequence length. The tested Metal shader validator doubles this allocation to 32 KiB, matching the M5 Max device cap. Pipeline creation checks the compiled static allocation and maximum thread count against the device/pipeline limits. An earlier separate-sum-buffer layout used 16.5 KiB and failed validation at 33 KiB; it is not the committed layout.

### Comparison protocol

Apple M5 Max, macOS 26.6.2 (25G83), Xcode 26.6 (17F113). The benchmark compiles the previous mean shader alongside the production shader, alternates old/new ordering each round, warms each variant four times, and records GPU timestamps. Each table row uses 20 measured rounds except the 103982 square case, which uses 10. Validation is disabled for timing. Raw round timings and exact commands are in [the benchmark log](benchmarks/na-int8-attention-tiled-v-mean-2026-09-22.txt).

Both mean-only and full-attention modes check bitwise equality. Full mode runs all five production pipelines and bindings, changing only the V-mean shader and its dispatch. It uses synthetic uniform V values (seed 42); Q/K are distinct rotations of those values, scale=1, with equal query/KV head counts. It does not measure captured model quality. FP16 uses low-precision intermediates; BF16/FP32 use FP32 intermediates. Reported times are separate medians, while speedup is the median of paired old/new ratios; these need not equal the ratio of the time medians when clocks drift.

### V-mean timings

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

### Full-attention timings

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

### Reproduction

Build `lib/libccv.a` with `make -C test/int/nnc debug -j4`, then use the standalone compile command at the top of `bin/mfa/na_int8_v_mean_bench.cpp`. Its arguments are `C H N rounds D precision full R`; omit the final two arguments for mean-only timing.

For validation, enable `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1`. These are the [Apple-documented shader-validation settings](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage). Timing under these settings is not comparable to normal execution.

### Validation

- Debug integration build: passed.
- Metal API + shader validation: 8/8 quantized forward tests and 1/1 variable-length test passed. This includes the added long-KV regression covering both reduction sizes, non-power-of-two heads, grouped queries, batches, different D, and runtime dimension caching.
- Five additional API + shader validation probes passed with bitwise equality. They cover partial channel tiles, the 20480/20481 reduction boundary, FP16/BF16/FP32, batched full attention, and the scalar D=130 mean path. All reported 32768 bytes of instrumented static shared memory against a 32768-byte device cap. See [validation output](benchmarks/na-int8-attention-tiled-v-mean-validation-2026-09-22.txt).
- The broader `quantized` filter reaches an existing backward API-validation failure after the eight forward tests: backward key/value requests 65536 bytes of dynamic threadgroup memory on this 32768-byte-cap device. The allocation function is byte-identical to the parent commit (`128 * 16 * 16 * 2`). This change does not repair that separate backward allocation issue; the entire backward API-validation suite is not claimed to pass.
- The existing backward probe was updated to the new dispatch helpers and current descriptor arguments. Its autoreleased command objects now use retained ownership, and its pool outlives those objects. It builds and exits 0 under API + shader validation at R=C=128, Hq=Hk=56, D=64. These concurrent-validation smoke timings are not performance comparisons.
- Complete normal MPS suites: BLAS 284/284 and DNN 126/126 passed (both exit 0), including forward and backward attention.
