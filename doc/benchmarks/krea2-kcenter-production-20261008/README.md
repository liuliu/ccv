# Production Hadamard K-centering

## Result

On the supported NA int8 forward path, `useHadamard` now also centers K over
the sequence before rotation and quantization. The Krea attention shapes stay
within the requested 1% target in the final paired GPU measurements: 0.66% for
R=C=4131 and 0.81% for R=4096, C=4131. This is not a universal bound: the
1024-token case costs about 10 microseconds, or 2.55%.

Follow-up (2026-10-08): the K/V mean reduction was rewritten as a chunked,
row-coalesced kernel shared by plain and Hadamard centering. Long-context
centering overhead roughly halved, for example 16.3% to 8.8% at R=1,
C=16384 and 12.0% to 7.1% at R=128, C=32768. Sequences over 512 rows add a
second small mean dispatch and per-chunk scratch. See
[the V-mean reduction notes](../../na-int8-attention-tiled-v-mean.md).

Replaying the production GPU quantizers on the original 32 Krea captures
reduces first-block mean softmax TV from 0.223635 with uncentered Hadamard to
0.020635, a 90.8% reduction. All four sampled blocks improve. An uninstrumented
generation with the final patch also completes successfully.

Hardware: Apple M5 Max, 128 GB. Patch base: ccv
`6510c96b09ee7ddd1fd981de3682131f9a81c104`. Draw Things:
`8eef98c82f0de45aa8a7681f11a6e3c4f2d12b3d`, retaining the user's existing
Hadamard-enabled model changes and its pinned s4nnc dependency.

## Implementation

For each batch, KV head, and channel, compute `mu_K = mean(K, sequence)` in
FP32. The existing Hadamard K quantizer subtracts this mean in FP32 before
rotation; there is no intermediate FP16 rounding. Independent K and V
threadgroups share the existing V-mean dispatch, so centering adds no GPU
dispatch. Additional scratch for the mean is `B * Hkv * D * sizeof(float)`.

Without sinks or externally saved LSE, the dispatch order is:

```text
Q quantization -> combined K/V mean -> K quantization -> V quantization
-> int8 attention
```

In exact arithmetic, `alpha * Q @ (K - mu_K).T` differs from the original
scores only by the row constant `alpha * Q.dot(mu_K)`. Ordinary softmax does
not need that offset restored. Keeping it out of the main score calculation
also avoids reintroducing the large-offset numerical problem.

Sinks and exported LSE require a correction. For these cases, the mean dispatch
precedes Q quantization; Q's existing quantizer also computes an original-basis
FP32 row correction, scaled by `alpha * log2(e)`. The attention kernel subtracts
it from the sink logit and adds it to the exported base-2 LSE. Correction scratch
is allocated only for these cases: `B * Hq * R * sizeof(float)`.

This also fixes the existing int8 sink/V-centering interaction: a sink has no
V contribution, so V's mean is restored with the non-sink probability mass,
not with weight one. That correction applies with and without Hadamard.
Both descriptor cache layers include the source-generation correction flag.
There is no production toggle or environment-variable switch for centering.
Hadamard remains forward-only; this patch does not introduce a backward path.

## Real-Capture Accuracy

Same captures and generation trajectory as the
[original investigation](../krea2-qk-hadamard-20261008/README.md):
`krea_2_turbo_i4x.ckpt`, 1024x1024, seed 42, eight steps, CFG 1, red-fox prompt.
Blocks 0, 9, 18, and 27 at each step; FP16 inputs, B=1, Hq=48, Hkv=12, D=128.
R=C=4131, except block 27 with R=4096.

The final GPU mean and Hadamard quantizers are replayed against the original
input snapshots. For each head/capture, 128 query rows use all 4131 keys:
812,187,648 scores and 196,608 full softmax rows per candidate. Integer dots
and rescaling are analyzed in FP64 against original-basis FP64 QK. For raw-score
reporting only, the row offset is restored in FP64. It is not restored inside
the shipping GPU softmax.

| Block | Centered RMSE: plain / old Had / centered Had | Mean TV: plain / old Had / centered Had |
| --- | --- | --- |
| 0 | 20.066411 / 13.138385 / 0.588174 | 0.102976 / 0.223635 / 0.020635 |
| 9 | 0.026955 / 0.021277 / 0.019995 | 0.008878 / 0.006940 / 0.006500 |
| 18 | 0.035047 / 0.027547 / 0.024560 | 0.012185 / 0.009500 / 0.008181 |
| 27 | 0.024016 / 0.018702 / 0.016865 | 0.006190 / 0.005160 / 0.004316 |

The old-Hadamard columns come from the original captured GPU quantization,
not a CPU simulation. Across all captures, mean TV is 0.032557 / 0.061309 /
0.009908 and mean KL is 0.057554 / 2.916643 / 0.002873, respectively. Final
centered-Hadamard raw RMSE is 0.294650 and centered RMSE is 0.294634.

The GPU K-mean's largest absolute error versus the FP64 mean is `4.22e-5`.
CPU/GPU quantizer checks pass: at most one int8 unit of rounding difference,
with float scales matching. All 1,536 head/capture RMSE comparisons improve
over plain int8.

These are local Q/K quantization and recomputed FP64 softmax measurements, not
full-attention output or model-quality scores. They exclude online softmax
arithmetic, probability/V quantization, and PV multiplication. Only four
blocks and one prompt/seed are sampled; broader model-quality claims need
additional generations.

## Performance

The same-process harness alternates before/after order for each pair, reuses
identical buffer addresses, and measures command-buffer GPU timestamps. Full
timings include all means, quantizers, and attention in one serial compute
encoder, matching production dispatch ordering. Neither variant exports LSE
or uses sinks. Inputs are deterministic FP16; Hq=48, Hkv=12, D=128, B=1.

The harness-only baseline rebuilds the original uncentered Hadamard K
specialization and uses the original V-only mean. Q, V quantization, and
attention are unchanged. Full output SHA256 hashes match the separately built
pre-patch and final production dispatcher benchmarks exactly for R=C=4131.
The baseline is not plain int8.

| R x C | Pairs | Before median (ms) | After median (ms) | Median paired overhead | Ratio of medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4131 x 4131, after idle | 500 | 6.248375 | 6.265250 | +0.66% | +0.27% |
| 4096 x 4131 | 300 | 5.079229 | 5.122375 | +0.81% | +0.85% |
| 1024 x 1024 | 300 | 0.386646 | 0.396563 | +2.55% | +2.56% |
| 8192 x 8192 | 150 | 20.670313 | 20.911229 | +0.60% | +1.17% |

Paired overhead is `100 * (median(after_i / before_i) - 1)`. The ratio of
independent timing medians is a different statistic; both are shown rather
than selecting whichever is smaller. Median pair deltas are approximately
40, 41, 10, and 115 microseconds, respectively.

An earlier clean 500-pair Krea run measured 7.817/7.864 ms and +0.65% paired
overhead. Absolute baseline times varied substantially during the session,
including about 13 ms in earlier runs. The user confirmed no other GPU
workload; the cause is unresolved. Final measurements ran sequentially with
no other agent-launched GPU work or concurrent build, and the main Krea run
followed a 30-second GPU idle period. These data support a small overhead for
the target shape, not a guaranteed sub-1% bound across shapes/system states.

The final uninstrumented Draw Things generation takes 18.83 seconds including
loading, with mean step time 1.81 seconds and median step time 1.57 seconds.
This is a correctness smoke test, not a before/after end-to-end benchmark.
The original 29.67-second capture-instrumented run is not comparable.

## Validation

- `make debug -j4`: success.
- Quantized NA forward attention subset: 12/12 passed.
- `mpsdnn.tests`: 126/126 passed.
- Focused eight-trial K-centering test with Metal shader validation: passed.
- Existing Hadamard and varlen attention tests: passed, including head widths
  64, 80, 96, 128, 192, and 256 and unsupported-width fallback coverage.
- New regression checks batch 2, GQA, FP16/BF16/FP32, causal and numeric masks,
  sequence tails, sinks, empty rows, dynamic/static specialization, and
  no-LSE versus saved-LSE cache variants. Original-domain LSE is checked
  against an FP64 reference with large row offsets.
- Final production quantizer replay: all 32 captures passed CPU/GPU checks.
- Final Draw Things build and uninstrumented generation: success.

Full BLAS validation is not green: the unrestricted full suite stops in
unchanged generic GEMM source with Metal `thread_elements()` address-space
compilation errors. The broader quantized-NA subset reaches an unchanged
backward BF16/int8 MPP destination-type compilation error after the 12 forward
tests pass. These failures are logged and were not fixed as part of this
forward-only change. A D=8 full-attention probe also stalls in the saved
pre-patch binary; only its quantizer was verified here.

## Reproduction

From `bin/mfa`, build `na_int8_attention_hadamard_bench`,
`na_int8_attention_forward_bench`, and `na_int8_attention_kcenter_replay`.
The paired benchmark uses CLI arguments, not environment variables:

```sh
./na_int8_attention_hadamard_bench 4131 4131 1 48 12 0 0 3 4 0 128 0 500 1
```

Here the final `1` selects the uncentered/centered Hadamard pair; `0` before
`500` requests full attention as well as the quantizer timing. Use `full=1`
lines for production overhead. `full=0` is not a fair full-path comparison:
the centered quantizer preparation also computes V's mean in the fused dispatch.

From the ccv root, using Python with NumPy and the original snapshots:

```sh
python3 bin/mfa/analyze_krea_qk.py /private/tmp/krea-qk-20261008 \
  --query-rows 128 \
  --replay-executable bin/mfa/na_int8_attention_kcenter_replay \
  --output /private/tmp/kcenter-production-final-accuracy
```

The analyzer replays and deletes one temporary capture at a time. Snapshots
remain outside the repository (about 4 GB). `summary.json`, `head_scores.csv`,
`quantization.csv`, and `quantizer_checks.csv` preserve accuracy results here.
`kcenter-paired-idle-krea.log` and `kcenter-paired-final-*.log` preserve every
final timing pair. Earlier parallel-mean run logs are included for transparency,
but are not the headline measurements. Test/build logs and
`kcenter-generation.png` preserve the validation and smoke-generation evidence.
