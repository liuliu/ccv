# ANE H256 activation fusion

These are historical measurements of commit `97f908d3`, against the original
**two-dispatch ordinary quantization baseline**. They do not establish a <1%
Hadamard overhead bound against the newer fused baseline. The kernel description
below also refers to that commit. See the [model-shape follow-up](ane-quantization-fusion-benchmark.md)
for the current comparison and its limitations.

Measured on Apple M4 Pro, macOS 26.6.2 (25G83), 2026-09-22, with the library
rebuilt at `-O3`. The baseline uses Q6_K weights; H256 uses Q6_K + H256 weights
packed offline from the same dense matrix. Both use ANE with GPU NA and generic
MFA GEMM disabled. All five measured cases met the less-than-1% overhead
target with no activation staging buffer; median paired overhead ranged from
-2.77% to -0.03%. The upper endpoints of all five 95% bootstrap intervals
were below 1%.

| M × N × K | Precision | Baseline median ms | H256 median ms | Median paired overhead | 95% bootstrap interval |
| --- | --- | ---: | ---: | ---: | --- |
| 1024 × 1024 × 1024 | FP16 | 0.38370 | 0.37328 | -2.766% | [-2.838%, -2.652%] |
| 2048 × 2048 × 2048 | FP16 | 1.23010 | 1.21520 | -1.195% | [-1.461%, -0.892%] |
| 4096 × 4096 × 4096 | FP16 | 8.71820 | 8.67805 | -0.520% | [-0.788%, -0.149%] |
| 8192 × 4096 × 4096 | FP16 | 15.69150 | 15.57800 | -0.769% | [-1.006%, -0.480%] |
| 4096 × 4096 × 4096 | FP32 | 8.75095 | 8.74290 | -0.026% | [-0.576%, +0.628%] |

The overhead statistic is the median of `100 * (H256 / baseline - 1)` for
alternating AB/BA pairs, rather than the ratio of independently selected medians.
Negative values mean the H256 path ran faster. H256 uses one fused activation
preparation dispatch; the baseline uses separate scale and quantize/transpose
dispatches.
The interval is a percentile bootstrap over paired ratios (10,000 resamples,
Python `random.seed(42)`). The 1024³ case uses 301 pairs of 50 calls per variant;
the other cases use 101 pairs of 20 calls. See the [raw paired samples](ane-hadamard256-benchmark.csv).

The CSV labels these measurements `direct_primary`. It also retains the earlier
activation-staging implementation's measurements as `staged_primary` and
`staged_confirmation`; those results do not describe the current kernel.

Timing includes packed-weight decoding, activation preparation, weight upload,
CoreML/ANE evaluation, output dequantization, and stream completion. It excludes
offline weight packing, initial tensor transfers, model compilation, and five
warmup calls per variant. The measurements describe steady-state end-to-end
GEMM time per call on this machine; they do not establish a bound for every
shape or Apple device.

H256 rotates 256-feature groups in SIMD registers, computes each row's scale,
and quantizes directly into the transposed ANE surface in one dispatch. Each
threadgroup handles four rows with 256 threads per row. An 8 KiB threadgroup
INT8 tile packs four adjacent rows into four-byte surface stores; the reduction
uses another 128 bytes of threadgroup memory. Padded rows repeat the last valid
input row. There is no intermediate device activation buffer, no global scratch
reservation for activation quantization, and no separate transpose dispatch.
Shapes and strides remain pipeline constants.

The initial direct scalar-store experiment was correct but measured +5.53%
median paired overhead at 4096³ FP16 (31 pairs, 10 calls). The four-row tile
addresses that cost without device scratch. The earlier staged implementation
used `batch * M * K` scratch bytes and a separate transpose; it has been removed.

Reproduce from the repository root:

```sh
# Force an optimized library rebuild without downloading sample models.
make -C lib -B -j4 lib
make -C bin/nnc ane_hadamard_bench ane_hadamard_quant_probe
bin/nnc/ane_hadamard_bench 1024 1024 1024 301 50
bin/nnc/ane_hadamard_bench 2048 2048 2048 101 20
bin/nnc/ane_hadamard_bench 4096 4096 4096 101 20
bin/nnc/ane_hadamard_bench 8192 4096 4096 101 20
bin/nnc/ane_hadamard_bench 4096 4096 4096 101 20 1
bin/nnc/ane_hadamard_quant_probe
```

The direct probe checks 45 cases: FP16/BF16/FP32, K=256 through 65536, zero rows,
outliers, batch strides, source offsets, padding, different M values, and
alternating H256/ordinary cache entries. It compares scales and quantized bytes
to an independent scalar transform, permitting an adjacent INT8 value only at
a floating-point half-integer rounding boundary. The integration test covers
IQ2_XXS/Q6_K, all three precisions, small M, padded M, batching, bias/no bias,
CPU GEMM parity, and explicitly rotated input through ordinary ANE GEMM.

Validation of the direct-write implementation: `make debug -j4` succeeded;
the direct quantization probe passed 45/45 cases; `mpsblas.tests "ANE"` passed
5/5. Before this revision, the full BLAS suite passed 283 cases with 1
hardware-dependent skip after removing the blanket sparse-attention BF16 gate,
and `mpsdnn.tests` passed 126/126. Those full suites were not repeated for the
activation-staging removal.
