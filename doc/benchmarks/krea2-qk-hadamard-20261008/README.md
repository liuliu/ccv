# Krea 2 Turbo: QK quantization with and without Hadamard

Production follow-up (2026-10-08): K-centering is now implemented and validated
when Hadamard is enabled. See the
[production patch measurements](../krea2-kcenter-production-20261008/README.md).
The conclusions below describe the original, uncentered implementation.

## Decision

Hadamard meaningfully improves QK quantization in sampled transformer blocks 9,
18, and 27, but **uncentered Hadamard is not an unconditional accuracy win for
this model**. Block 0 has nearly constant, very large Q/K components. Hadamard
reduces aggregate score RMSE while substantially worsening its softmax.

The most promising follow-up is to center K over the sequence before Hadamard
and quantization. An offline experiment reduced block-0 mean softmax TV error
from 0.223635 to 0.020630, versus 0.102976 for plain int8. This has not yet been
implemented or validated in the production GPU kernel. Until then, retain a
plain-int8 or higher-precision first-block fallback rather than enabling
uncentered Hadamard throughout Krea2. Validate the unsampled blocks and more
prompts before choosing a permanent per-block policy.

## Real Generation

- Hardware: Apple M5 Max, 128 GB memory.
- ccv: `6510c96b09ee7ddd1fd981de3682131f9a81c104`.
- Draw Things: `8eef98c82f0de45aa8a7681f11a6e3c4f2d12b3d`, with the user's existing
  `useHadamard: true` changes in Krea2.swift.
- s4nnc: the Draw Things pinned version,
  `55ac9a762f57003a246704f2070c9ee08dd5e483`; do not override it with the older
  local s4nnc checkout, whose SDPA wrapper lacks `useHadamard`.
- Checkpoint: `krea_2_turbo_i4x.ckpt`, 7,572,160,512 bytes; accompanying
  `qwen_3_vl_4b_q8p.ckpt` and `qwen_image_vae_f16.ckpt`.
- Model metadata: the matching Krea2 entry in the user's hard-linked
  `/Users/liu/workspace/draw-things/custom.json`.
- Generation: 1024x1024, seed 42, eight steps, CFG 1, empty negative prompt.
- Prompt: "A photograph of a red fox standing in a snowy pine forest at sunrise,
  detailed fur, soft golden light, natural colors, shallow depth of field."

The CLI completed successfully and produced `generation.png`. Its instrumented
total generation time was 29.67 seconds; this is **not a performance benchmark**
because capture adds extra kernels, copies, allocation, and disk output.

## Measurement

Captured blocks 0, 9, 18, and 27 at every sampling step: 32 attention calls.
Each uses FP16 input, batch 1, 48 query heads, 12 KV heads, and D=128. R=C=4131
(4096 image tokens plus 35 text tokens), except block 27, where R=4096. The
attention scale is `1/sqrt(128)`. Q scales cover 16 rows; K scales cover 64 rows.

Both candidates quantize **the same live, post-RMSNorm/post-RoPE input tensors**:
the existing Hadamard GPU quantizer and the production plain GPU quantizer
selected through an otherwise identical descriptor. The Hadamard candidate
is also the one used by the live generation. Snapshots are encoded immediately
after Q/K quantization, before scratch can be overwritten by later graph nodes.

The analyzer uses FP64 original-basis `alpha * Q @ K.T` as its reference.
Candidate scores use the actual captured GPU int8 tensors and float scales.
Integer dots are evaluated exactly; FP64 rescaling isolates Q/K quantization
error from the subsequent floating-point arithmetic. For every head and
capture, 128 evenly spaced query rows are compared against **all 4131 keys**.
This yields 812,187,648 scores and 196,608 complete softmax rows per candidate.

Metrics:

- RMSE: root mean square error of scaled QK logits.
- Centered RMSE: remove the per-query mean of the logit error before RMS; this
  discounts row-constant shifts that cannot change exact softmax.
- TV: mean `0.5 * sum(abs(p_reference - p_candidate))`, ranging from 0 to 1.
- KL: mean `KL(p_reference || p_candidate)` in nats.

Softmax is recomputed in FP64, not captured from the shipping online kernel.
These results do not include approximate exp2, online reduction/rescaling,
probability or V quantization, PV multiplication, or the final attention output.
They compare local alternatives along one Hadamard-enabled generation trajectory,
not separate plain/Hadamard generated images or an unquantized model trajectory.

## Paired GPU Quantizer Results

Values below are plain int8 -> Hadamard int8. Blocks are zero-based.

| Block | QK RMSE | RMSE reduction | Mean softmax TV | Mean softmax KL |
| --- | --- | --- | --- | --- |
| 0 | 28.71153 -> 23.87817 | 16.8% | 0.102976 -> 0.223635 | 0.229298 -> 11.666015 |
| 9 | 0.028782 -> 0.022453 | 22.0% | 0.008878 -> 0.006940 | 0.000280 -> 0.000168 |
| 18 | 0.038958 -> 0.030533 | 21.6% | 0.012185 -> 0.009500 | 0.000494 -> 0.000293 |
| 27 | 0.026394 -> 0.020163 | 23.6% | 0.006190 -> 0.005160 | 0.000144 -> 0.000097 |

Across all captures, raw RMSE improves 14.35579 -> 11.93910 (16.8%); centered
RMSE improves 10.03324 -> 6.56922 (34.5%). However, mean TV worsens 0.032557 ->
0.061309 (1.88x), and mean KL worsens 0.057554 -> 2.916643. 91.6% of head/capture
pairs improve raw RMSE. The first block dominates aggregate logit energy, so
neither that fraction nor global raw RMSE is a safe criterion on its own.

The preliminary 32-query-row analysis found the same direction of change;
the 128-row results above supersede it.

## Why The First Block Regresses

For example, at step 0, query head 30 uses KV head 7. K channel 6 has sequence
mean -596.47119 and standard deviation 0.32322. Its sequence-mean vector accounts
for 99.999916% of K's energy. Large components also exist in Q, producing logits
near 31,447; most of that value is a shared row offset, not useful discrimination
between image tokens.

The unrotated quantizer can preserve this dominant coordinate with relatively
little false variation among image keys. Hadamard distributes it over all 128
coordinates, and tile-dependent scale/rounding errors then introduce large
spurious differences between keys. This is an inference from the captured
tensors and the centering intervention, not evidence of a broken Hadamard
transform or index mapping.

For step 0/head 30, mean TV is 0.000258 for plain int8 but 0.977679 for Hadamard.
Reference softmax entropy is 8.317766 nats, versus 1.743711 after Hadamard:
an almost uniform image-token distribution becomes highly concentrated.
Step 0/heads 40 and 41 reach TV around 0.99914 with uncentered Hadamard.

CPU reproduction checks cover every Q and K head of all eight first-block
captures, including these heads and partial sequence tails. The maximum int8
difference from the GPU is one quantization unit; float scales match exactly.
The one-unit discrepancies are rounding-boundary differences, not a layout or
normalization discrepancy. The other sampled blocks also pass the selected-head
checks in `quantizer_checks.csv`.

## Offline K-Centering Experiment

Let `mu = mean(K, sequence)`, one vector per KV head. In exact arithmetic,
`Q @ (K - mu).T = Q @ K.T - (Q @ mu)[:, None]`, so each query loses a constant
and its softmax is unchanged. The largest observed FP64 softmax difference
between these references was `7.47e-12`.

For this experiment, K is centered offline and kept in FP32, then quantized by
a CPU reproduction of the respective production quantizer. Q retains its
captured GPU quantization. No GPU centering kernel or production integration
is claimed.

| First-block candidate | Centered logit RMSE | Mean TV | Mean KL |
| --- | --- | --- | --- |
| Plain int8 | 20.066411 | 0.102976 | 0.229298 |
| Hadamard int8 | 13.138385 | 0.223635 | 11.666015 |
| Center K + plain int8, offline | 3.658978 | 0.065478 | 0.459317 |
| Center K + Hadamard int8, offline | 0.588175 | 0.020630 | 0.011047 |

Center K + Hadamard reduces first-block TV by 90.8% relative to uncentered
Hadamard and by 80.0% relative to plain int8. At step 0/head 30 its TV becomes
0.007759 and entropy returns to 8.317572. Centering plain int8 alone is not a
uniform win: mean TV improves but mean KL worsens. Again, compare probability
metrics rather than just reconstruction or score RMSE.

## Reproduction And Artifacts

The temporary instrumentation was removed from production source after the run.
`capture.patch` preserves it for reproduction; it is a research-only patch with
fixed output path and Krea2 head/block assumptions. The normal Draw Things CLI
was rebuilt without the capture define. Existing user model changes remain.

From ccv, apply `capture.patch` if needed and create the capture output directory.
Then, from `/Users/liu/workspace/draw-things`:

```sh
bazel build //Apps:DrawThingsCLI \
  --override_repository=ccv=/Users/liu/workspace/ccv \
  --per_file_copt='.*ccv_nnc_mfa_attention.cpp@-DCCV_KREA_QK_CAPTURE=1'
bazel-bin/Apps/DrawThingsCLI generate \
  --model krea_2_turbo_i4x.ckpt \
  --models-dir /Users/liu/workspace/draw-things \
  --offline --no-download-missing --disable-preview \
  --steps 8 --cfg 1 --width 1024 --height 1024 --seed 42 \
  --negative-prompt '' \
  --prompt 'A photograph of a red fox standing in a snowy pine forest at sunrise, detailed fur, soft golden light, natural colors, shallow depth of field.' \
  --output /private/tmp/krea-qk-20261008/generation.png
```

From ccv, using Python with NumPy:

```sh
python3 bin/mfa/analyze_krea_qk.py /private/tmp/krea-qk-20261008 \
  --query-rows 128 --output /private/tmp/krea-qk-20261008/analysis128
python3 bin/mfa/analyze_krea_qk.py /private/tmp/krea-qk-20261008 \
  --query-rows 128 --diagnose-first-layer \
  --output /private/tmp/krea-qk-20261008/diagnostics
```

`summary.json` contains overall, per-step, and per-block metrics.
`head_scores.csv` contains per-head metrics. `first-layer.json` and
`first-layer-heads.csv` contain the centering experiment and failure diagnostics.
`capture-metadata/` preserves shapes, offsets, and quantizer metadata; the 32
binary snapshots remain at `/private/tmp/krea-qk-20261008` (about 4 GB) rather than
being copied into the repository. Build logs, generation log, output image,
quantizer checks, and input reconstruction metrics are included here.
