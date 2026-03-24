# Learnings

Date: 2026-02-10

## Summary
Implemented end-to-end `swish.beta` support for CPU + MPS + CUDA GPU_REF paths, added non-one-beta coverage in unit/int tests, and integrated beta support into MFA Swish kernels with a special-case for `beta == 1` to keep the original kernel code path unchanged.

## Core API / Command Changes
- Added Swish beta attribute:
  - `lib/nnc/ccv_nnc.h` (`swish.beta` in `ccv_nnc_cmd_param_t`)
- Swish easy macros now take `_beta`:
  - `lib/nnc/cmd/swish/ccv_nnc_swish.c`
  - regenerated `lib/nnc/cmd/ccv_nnc_cmd_easy.h`

## CPU Reference
- Forward/backward now read `cmd.info.swish.beta`:
  - `lib/nnc/cmd/swish/ccv_nnc_swish_cpu_ref.c`

## MPS Path
- `lib/nnc/cmd/swish/mps/ccv_nnc_swish_mps.m`
  - Uses `cmd.info.swish.beta` in forward/backward.
  - Keeps explicit graph fast-path behavior for `beta == 1`.
  - Uses FP32 compute + cast-back when `beta != 1` in MPSGraph fallback.
  - MFA is now allowed for non-one beta (no longer gated to `beta == 1`).

## CUDA GPU_REF Path
- `lib/nnc/cmd/swish/gpu/ccv_nnc_swish_gpu_ref.cu`
  - Uses `cmd.info.swish.beta` in forward/backward dispatch.
  - Keeps original kernels and formulas when `beta == 1`.
  - Adds dedicated beta-aware template kernels for `beta != 1`.
  - New beta-aware kernels compute in FP32 and cast back to output type.

## MFA Swish Integration (like GELU)
- Added beta plumbing through MFA params/descriptors:
  - `lib/nnc/mfa/ccv_nnc_mfa_swish.hpp`
  - `lib/nnc/mfa/ccv_nnc_mfa_swish.cpp`
  - `lib/nnc/mfa/kernels/SwishDescriptor.hpp`
  - `lib/nnc/mfa/kernels/SwishDescriptor.cpp`
- Added beta into Swish kernel generation:
  - `lib/nnc/mfa/kernels/SwishKernel.hpp`
  - `lib/nnc/mfa/kernels/SwishKernel.cpp`
- Special-case for `beta == 1` in MFA kernel source:
  - Emits original formulas / constants for beta=1 (old code path behavior).
  - Emits beta-aware formulas and function constant only when `beta != 1`.

## Tests Added / Updated
- Unit:
  - `test/unit/nnc/swish.tests.c`
  - Added `TEST_CASE("swish with non-one beta")`
  - Added `TEST_CASE("swish gradient with non-one beta")`
- Integration:
  - `test/int/nnc/swish.tests.c`
  - Added `TEST_CASE("mps swish gradient with non-one beta in half precision")`
  - Added `TEST_CASE("swish gradient with non-one beta in half precision")` (GPU_REF)

## Validation Run Here
- `test/unit/nnc`
  - `make swish.tests -j4 && ./swish.tests` -> pass
- `test/int/nnc`
  - Outside sandbox: `./swish.tests` -> pass (`8/8`, with 3 expected skips for MPS on non-macOS).

---

Date: 2026-03-06

## Summary
Implemented forward MPS support for `EWPOW` / `EWSIN` / `EWCOS`, added MFA sigmoid, migrated the remaining non-attention MFAv2 wrappers (`gemv`, `depalettize`, `adam`, `normalization`) to the Descriptor / Kernel model, and then renamed `lib/nnc/mfa/v2` to `lib/nnc/mfa/kernels` with `v2_cache` renamed to `kernel_cache`.

## MPS Elementwise Learnings
- `EWEXP` was the right precedent for new forward-only MPS elementwise ops:
  - If there is no existing MFA kernel family, use `MPSGraph` first.
  - `EWPOW`, `EWSIN`, and `EWCOS` were added that way.
- Integration tests for these ops belong in:
  - `test/int/nnc/mpsblas.tests.c`
  - CPU reference parity is the right validation model.

## Sigmoid Learnings
- Existing MPS sigmoid backend already used `MPSGraph`, not MFA:
  - `lib/nnc/cmd/sigmoid/mps/ccv_nnc_sigmoid_mps.m`
- The MFA sigmoid kernel should match MLX’s numerically stable formulation rather than the naive `1 / (1 + exp(-x))` form.
- The backward MFA sigmoid path uses the forward output:
  - `g * y * (1 - y)`

## MFA Migration Learnings
- The remaining wrappers that were still legacy inline / cache-based before this task were:
  - `gemv`
  - `depalettize`
  - `adam`
  - `normalization`
  - masked `attention` was intentionally left alone
- Migration rule:
  - Keep the wrapper thin.
  - Move codegen / pipeline creation into a Descriptor / Kernel pair.
  - Preserve the existing shader behavior during migration.
- Important example:
  - `depalettize` must keep the old `qbits == 5`, `qbits == 6`, and `qbits == 8` shader behavior exactly.
  - Do not introduce new tail handling or alternate kernels during a pure migration unless there is a separate intentional behavior change.
- `normalization` MFA still only covers `layer_norm` and `rmsnorm`, matching prior behavior.
  - `group_norm` stays on `MPSGraph`.

## Group Norm Learnings
- `group_norm` on MPS was never wired to MFA in the backend.
- The only MFA involvement in the group norm path is depalettizing quantized affine inputs.
- The old normalization MFA implementation also explicitly rejected group norm, so the migration did not regress coverage.

## Rename Learnings
- After the v2 migration was effectively complete, the folder rename was safe:
  - `lib/nnc/mfa/v2` -> `lib/nnc/mfa/kernels`
  - `context->v2_cache` -> `context->kernel_cache`
- Bazel did not need source-list updates because:
  - `lib/BUILD.bazel` already uses `glob(["nnc/mfa/**/*.cpp", "nnc/mfa/**/*.inc"])`
  - and `glob(["nnc/mfa/**/*.hpp"])`
- The rename still required explicit path fixes in:
  - `Package.swift`
  - `bin/nnc/*` kernel generator / utility sources
  - docs such as this file

## Test / Iteration Learnings
- For iteration, focused MPS int runs are faster and already supported:
  - `./mpsblas.tests <substring>`
  - `./mpsdnn.tests <substring>`
- Full suite runs are still required before wrapping the task.
- For migrated MFA code, a good validation order is:
  - focused op test
  - focused related path test
  - full `mpsblas.tests`
  - full `mpsdnn.tests`

## Performance Learnings
- A pure rename from `v2` to `kernels` should be performance-neutral if:
  - descriptor keys are unchanged
  - generated Metal source is unchanged
  - function constants and dispatch geometry are unchanged
  - cache behavior is unchanged
- Functional tests are not enough to prove perf parity.
- Real confidence should come from:
  - cold-cache timing
  - warm-cache timing
  - representative-shape benchmarks for migrated kernels

## Validation Run Here
- `test/int/nnc`
  - `make debug -j4` -> pass
  - `./mpsblas.tests gemv` -> pass
  - `./mpsblas.tests depalettize` -> pass
  - `./mpsdnn.tests` -> pass (`83/83`)
  - `./mpsblas.tests` -> pass (`68/68`)
- Bazel
  - `bazel build //lib:nnc_mfa_compat` did not validate the rename because this checkout currently lacks a resolved `@local_config_ccv` repository.
  - Static inspection showed the Bazel rule already globbed `nnc/mfa/**`, so there was no explicit `v2` path to update there.

## NAAttention Learnings
- The `executionSIMDGroups` tuning direction for `NAAttention` is not worth pursuing as a production optimization from the current data.
- Apparent wins from lowering `executionSIMDGroups` were highly shape-sensitive and also sensitive to run ordering / thermal state, so simple heuristics did not hold up under broader sweeps.
- For `blockC = 48`, low-head and high-head regimes disagree:
  - at `4095 x 7169 x 128`, `H = 8` preferred `8` simdgroups (`5.189 ms` vs `6.519 ms` for `16`);
  - at the same shape, `H = 16` and `H = 24` strongly preferred `16` (`11.034 ms` vs `16.909 ms`, and `14.926 ms` vs `29.809 ms`).
  - Therefore `blockC = 48 -> always 8` is not defensible.
- For `blockC = 64`, there is no stable global rule such as “always 8” or “always 16”:
  - `8` often wins in the midrange around sequence `4096..8192`;
  - larger sequences, especially with `D >= 80`, often move toward `16`;
  - the transition region around `9216..16384` remained noisy for `D = 80`, mixed but mostly `16` for `D = 128`, and clearly `16` for `D = 160`.
- `blockR = 16` remained clearly correct; `blockR = 8` and `blockR = 32` were much worse.
- Practical conclusion:
  - do not land a production `executionSIMDGroups` heuristic for `NAAttention` based on the current sweeps;
  - any real heuristic would need to depend on multiple variables (`blockC`, sequence length, head count, and `D`);
  - the gain is too inconsistent to justify the complexity right now.

---

Date: 2026-03-23

## NAInt8Attention Learnings

### Summary
- A harness-only `NAInt8Attention` scaffold is now in place and useful for correctness + GPU-timestamp benchmarking.
- `NAInt8MatMul` remains an important reference point:
  - tuned `int8 x int8 -> int32` GEMM (`128x128x128`, `simdgroups=8`) was about `1.8x` to `2.0x` faster than FP16 `NAMatMul`.
- That speedup does not transfer automatically to fused attention.
- For the representative attention case `8192 x 8192 x 128`, `B=1`, `Hq=Hk=32`, the best full int8 fused kernel is still slower than FP16 `NAAttention`.

### Harness / Benchmark Notes
- The local harnesses are:
  - `bin/mfa/na_int8_matmul_bench.cpp`
  - `bin/mfa/na_int8_attention_bench.cpp`
- The attention harness measures GPU time with:
  - `commandBuffer.GPUEndTime() - commandBuffer.GPUStartTime()`
- For attention, the representative saturated benchmark shape on this machine should use:
  - `Hq >= 24` (for example `32`)
- A benchmark bug was fixed:
  - for divisible full-mode cases, the harness now truly skips threadgroup scratch allocation instead of still calling `threadgroupMemoryAllocation()` first.
  - this mattered for experimentation with large `blockC`.

### What Helped
- Matching `NAAttention`'s `cO` accumulation pattern helped materially.
  - The old int8 full kernel eagerly zeroed `cO` before the traversal loop.
  - Changing it to match `NAAttention`:
    - zero `cO` only when `c == 0`
    - otherwise apply the correction multiply on later iterations
  - improved the representative full kernel from about `29.7 ms` to about `26.6 ms`
  - that was roughly an `8%` to `12%` improvement depending on shape
- Reusing the int32 score cooperative tensors as float tensors via a typed alias also helped a bit.
  - This reduced one layer of extra live state in the full fused path.

### What Did Not Help
- Removing `Q_scale` / `K_scale` for performance evaluation did not help.
  - Representative `8192 x 8192 x 128`, `H=32`:
    - full int8 with scales: `29.654 ms`
    - full int8 without scales: `30.378 ms`
  - So the scale loads / multiplies are not the main bottleneck.
- Larger traversal / row tiles were consistently bad in the current fused design:
  - `blockC=128`: about `101 ms`
  - `blockC=256`: about `225 ms`
  - `blockR=32`: about `77 ms`
- `blockC` regressions remained bad even after removing threadgroup scratch from the divisible cases.
  - Therefore the issue is not just threadgroup-memory allocation pressure.
  - The real cost is the larger score / reduction tiles in the fused softmax path.

### Important Structural Comparison vs NAAttention
- `NAAttention` does `QK` directly into float cooperative tensors.
  - max-reduce, `exp2`, and sum-reduce all operate directly on that float score representation.
- `NAInt8Attention` does `QK` into int32 cooperative tensors.
  - even after aliasing the same storage as float, the path is still:
    - `int8 QK`
    - `int32` score tile
    - `int32` row max reduction
    - convert / reinterpret to float
    - `exp2`
    - float row-sum reduction
    - convert to half for PV input
- So the remaining cost is not "too many cooperative tensors" in a simple count sense.
- The remaining cost is the int32-to-float softmax bridge and the online correction machinery around it.

### Representative Performance Numbers
- Representative case: `8192 x 8192 x 128`, `B=1`, `Hq=Hk=32`
  - baseline FP16 `NAAttention`: about `24.5 ms`
  - best full int8 fused kernel after the `cO` fix:
    - about `26.6 ms`
    - about `0.91x` of baseline
  - `qk-only int8`:
    - about `8.6 ms`
  - `qk-pv-raw int8`:
    - about `20.7 ms`
    - about `1.18x` faster than baseline

### Most Important Performance Conclusion
- The `qk-pv-raw` number is a lower bound for the current "int8 QK + half V/PV" design.
- On the representative `D=128` case:
  - `qk-pv-raw int8` is still about `20.7 ms`
  - baseline FP16 `NAAttention` is about `24.5 ms`
- Therefore:
  - even if softmax became free, this design would still only be about `1.18x` faster than baseline
  - it cannot reach a `1.5x` full-kernel speedup target by only optimizing the softmax portion
  - bigger / faster QK tiles alone are not enough

### D=256 Experiment
- The oversized-head-tile experiment became meaningful on real `D=256` shapes.
- On `8192 x 8192 x 256`, `B=1`, `H=32`:
  - `blockD=128` full int8: about `248 ms`
  - `blockD=256` full int8: about `147 ms`

### 2026-03-23 Follow-up: Better Scaled Int8 Point
- A later optimization round improved the realistic tile-scaled path again.
- The useful changes were:
  - load QK scores for the int32-to-float conversion directly from `cS_0 / cS_1`, instead of reading the same bits back through the float alias first;
  - keep `executionSIMDGroups = 16`;
  - move back to `blockD = 128` after the score-load cleanup.
- That last point matters:
  - before the score-load fix, `blockD = 64` looked slightly better on the large-square representative case;
  - after the score-load fix, `blockD = 128` became the better configuration.

### Current Best Representative Configuration
- For the current realistic scaled path on this machine:
  - `blockR = 16`
  - `blockC = 64`
  - `blockD = 128`
  - `executionSIMDGroups = 16`
  - `Q/K/V = int8`
  - `Q/K` scales = tile

### Updated Representative Performance
- `4096 x 4096 x 128`, `B=1`, `H=32`
  - baseline FP16: `5.698 ms`
  - GPU quantize + int8: `5.504 ms`
  - end-to-end speedup: `1.035x`
- `8192 x 8192 x 128`, `B=1`, `H=32`
  - baseline FP16: `23.374 ms`
  - GPU quantize + int8: `21.748 ms`
  - end-to-end speedup: `1.075x`
- `12288 x 12288 x 128`, `B=1`, `H=32`
  - baseline FP16: `57.274 ms`
  - GPU quantize + int8: `47.978 ms`
  - end-to-end speedup: `1.194x`
- `16384 x 16384 x 128`, `B=1`, `H=32`
  - baseline FP16: `108.226 ms`
  - GPU quantize + int8: `90.156 ms`
  - end-to-end speedup: `1.200x`
- `8192 x 16384 x 128`, `B=1`, `H=32`
  - baseline FP16: `50.646 ms`
  - GPU quantize + int8: `44.861 ms`
  - end-to-end speedup: `1.129x`
- `16384 x 8192 x 128`, `B=1`, `H=32`
  - baseline FP16: `48.232 ms`
  - GPU quantize + int8: `43.706 ms`
  - end-to-end speedup: `1.104x`
- `24576 x 16384 x 128`, `B=1`, `H=32`
  - baseline FP16: `163.829 ms`
  - GPU quantize + int8: `147.452 ms`
  - end-to-end speedup: `1.111x`

### Updated Tuning Learnings
- `executionSIMDGroups > 16` is not helping on the current full int8 kernel:
  - `20` was slightly worse than `16`;
  - `24` was clearly worse;
  - `32` collapsed badly.
- `blockC = 32` is a clear regression for the current scaled int8 design, even after the later score-load fix.
- The current best path is now positive even at `4096 x 4096`, but the gain is modest there.
- The gain grows with sequence size, and the best large-square end-to-end number seen so far is about `1.20x`, still short of the original `1.5x` target.
  - speedup over FP16 baseline: about `1.4x`
- This is useful diagnostically:
  - when `blockD` matches `D`, `kBlocks` collapses from `2` to `1`
  - that reduces live `cO` state and helps a lot
- But this does not solve the representative `D=128` case because `blockD=128` already has `kBlocks=1` there.

### Practical Next Direction
- If the goal is a truly performance-competitive `D=128` int8 attention kernel, the next work should not focus only on larger QK tiles.
- The representative half-V path is already bounded by `qk-pv-raw`.
- A likely necessary next step is to speed up the PV side too, for example:
  - quantize `V`
  - or redesign the kernel into multiple passes so the large-tile int8 QK path is not constrained by the current fused softmax / PV structure

### Native Int8 Q/K/V Follow-up
- Bench-only native-int8 `Q/K/V` support in `na_int8_attention_bench` was the first configuration that beat the fp16 `NAAttention` baseline on the representative saturated case.
- Representative case: `8192 x 8192 x 128`, `B=1`, `Hq=Hk=32`
  - command:
    - `./na_int8_attention_bench 8192 8192 128 1 32 32 3 10 0 8 64 16 128 full int8 0 0 1`
  - result:
    - baseline fp16 `NAAttention`: `23.303 ms`
    - full native-int8 path: `20.177 ms`
    - speedup: about `1.155x`
- This path uses:
  - `Q/K` as native int8 with `qkScales=0`
  - `V` as native int8
  - fixed-scale `P` quantization in the full kernel:
    - after `exp(score - M)`, `P` is already in `[0, 1]`
    - quantizing `P` with a fixed `127` scale is much cheaper than per-row scaling
    - dequantization after `PV` is just a constant multiply by `1 / 127`
- Accuracy of this native-int8 path on the sampled check was acceptable for a first performance scaffold:
  - `max_abs_o ~= 3.13e-2`
  - `max_rel_o ~= 2.6e-3`
  - `max_abs_l ~= 7.8e-3`
- Important failed variants:
  - `half P x int8 V` was not useful
  - per-row `P` quantization scales were much too expensive
  - `blockC=128` was still worse than `blockC=64`
  - `blockR=32` was still much worse than `blockR=16`
- Attention-side `pv-only` measurements showed why fixed-scale `P -> int8` is the right direction:
  - scaffold `pv-only fp16/fp16`: about `32.0 ms` on `4096 x 4096 x 128`
  - scaffold `pv-only int8/int8`: about `3.0 ms` on the same shape
  - so the potential `PV` win is real, but it only shows up when both sides are int8
- The remaining gap to a `1.5x` full-kernel target appears to be in the softmax / QK side, not the `PV` side anymore.

### Scaled Native Int8 Follow-up
- The next useful structural fix for the scaled int8 path was to convert the two `int32` score tiles into scaled float tiles before row-max reduction, and then keep the rest of the online softmax path in float.
- That removed repeated `int32 -> float` work from both:
  - row-max reduction
  - the later `exp2(score - M)` loop
- With GPU-side `Q/K/V` quantization added to the harness, the representative scaled path (`qkScales=tile`) is no longer obviously a loss.
- Representative `8192 x 8192 x 128`, `B=1`, `H=32`:
  - command:
    - `./na_int8_attention_bench 8192 8192 128 1 32 32 3 10 0 8 64 16 128 full int8 1 0 1`
  - recent result:
    - baseline fp16: `24.286 ms`
    - int8 kernel only: `23.024 ms`
    - GPU quantize only: `0.843 ms`
    - GPU quantize + int8: `24.249 ms`
    - end-to-end speedup: about parity (`1.00x` avg, `0.99x` median on that run)
- Larger sequence amortizes quantization better:
  - `16384 x 16384 x 128`, `B=1`, `H=32`
  - command:
    - `./na_int8_attention_bench 16384 16384 128 1 32 32 1 5 0 8 64 16 128 full int8 1 0 1`
  - result:
    - baseline fp16: `105.268 ms`
    - int8 kernel only: `93.036 ms`
    - GPU quantize + int8: `95.394 ms`
    - end-to-end speedup: about `1.10x`
- Current conclusion:
  - `Q/K/V` quantization launches are not the main blocker
  - on the representative `8192` case they cost only about `0.8 ms`
  - the harder part is still the scaled softmax/QK path inside the fused int8 kernel
- Bench-only experiments that were not worth keeping:
  - per-head `Q/K` scales:
    - much slower quantization (`~9 ms` at `8192 x 8192 x 128`)
    - worse end-to-end than per-tile scales
  - LUT-based `exp2` approximation in the current scaffold:
    - materially slower than the built-in `fast::exp2`

### Realistic Scaled-V Follow-up
- `V` should be treated the same way as `Q/K` for realistic benchmarking.
  - The earlier unscaled `V=int8` path was only a best-case performance estimate.
  - After adding per-`C`-tile / per-head `V_scale`, the realistic `Q/K/V` all-scaled-int8 path remained a real win, but the speedup dropped.
- Current realistic best points for `single-block-full`:
  - `blockR = 16`
  - `blockC = 64`
  - `blockD = 32`
  - `Q/K` scales = tile
  - `V` scales = per `C` tile / head
- Representative end-to-end `GPU quantize + int8 attention` speedups:
  - `4096 x 4096 x 128`: about `1.23x`
  - `8192 x 8192 x 128`: about `1.33x`
  - `12288 x 12288 x 128`: about `1.28x`
  - `16384 x 16384 x 128`: about `1.30x`
  - `8192 x 16384 x 128`: about `1.30x`
  - `16384 x 8192 x 128`: about `1.30x`
- This is the realistic performance range to compare against fp16 `NAAttention` on this machine from the current scaffold.

### Half-P @ Int8-V Completeness Check
- The current winning `PV` path is not `half @ int8`.
  - It is:
    - quantize the block-local numerator tile `P` into int8
    - run `int8 @ int8 -> int32`
    - rescale by `V_scale / 127`
- A bench-only `pvLeft` switch was added to compare:
  - `pvLeft=int8`: current winner (`int8 @ int8`)
  - `pvLeft=fp16`: completeness path (`half @ int8`)
- `half @ int8` improved output accuracy materially, but it was too slow to be competitive.
- Representative results:
  - `4096 x 4096 x 128`
    - `int8 @ int8`: `1.261x` end-to-end
    - `half @ int8`: `0.916x`
    - `max_abs_o`: `0.0156 -> 0.0053`
  - `8192 x 8192 x 128`
    - `int8 @ int8`: `1.364x`
    - `half @ int8`: `0.977x`
    - `max_abs_o`: `0.0155 -> 0.0056`
  - `16384 x 16384 x 128`
    - `int8 @ int8`: `1.303x`
    - `half @ int8`: `0.900x`
    - `max_abs_o`: `0.0157 -> 0.0055`
- Conclusion:
  - `half @ int8` is not worth keeping as the performance path.
  - The accuracy improvement is real, but not enough to justify the throughput loss.

### Harness Cleanup: V Data Distribution
- The harness now always generates the same fractional `V` distribution for the int8 path.
- Why this was changed:
  - the old bench-only `V=fp16` / `V=int8` split generated different `v_values`,
  - which made the earlier unscaled-int8 `V` path look artificially accurate.
- Practical result:
  - the current `max_abs_o` / `max_rel_o` numbers are now measured on a fairer `V` distribution,
  - and the bench-facing `V=fp16` switch is no longer needed to reason about the fast path.

### MPP Guide Re-read: What Mattered
- The Metal Performance Primitives guide reinforces several points that match the int8 attention data:
  - Simdgroup tile size and threadgroup tile size are separate tuning knobs.
    - This fits the attention result:
      - larger math tiles (`blockC=128`, `256`, or `blockR=32`) were bad,
      - but larger reuse/packaging ideas are still plausible if row ownership stays local.
  - Smaller data types may need larger tiles to reach peak throughput.

### Cleanup After Tuning Locked In
- Once the realistic scaled-`V` path settled, the active scaffold was narrowed to the single-pass modes we still trust:
  - `full`
  - `qk-only`
  - `pv-only`
  - `qk-pv-raw`
  - `softmax-stats`
- The retired bench-only modes were removed from the public scaffold surface:
  - `row_max_only`
  - `replay_full`
  - `row_owned_full`
  - the explicit `two-pass` harness path
  - the explicit cooperative-execution bench toggle
- Why these were removed:
  - they were useful for diagnosis, but they were not the winning implementation path
  - they added a large amount of generator / harness branching that no longer represented the current design direction
  - keeping them around made the scaffold look more flexible than it really was
- What the data said:
  - two-pass was correct after bug fixes, but still slow:
    - `8192 x 8192 x 128`, `H=32`
    - baseline about `25.2 ms`
    - cooperative two-pass about `36.9 ms` including quantization
  - the row-owned / wider-threadgroup packaging did not create a new breakthrough:
    - on `16384 x 16384 x 128`, it was only about tied with the best existing single-pass path
    - on some larger rectangular shapes it helped a bit, but not enough to justify staying as a first-class direction
  - `half @ int8` `PV` was also removed as an active path:
    - it improved `max_abs_o` by about `3x`
    - but turned a real speedup into a loss or near-loss
    - representative results:
      - `4096²`: `1.261x -> 0.916x`
      - `8192²`: `1.364x -> 0.977x`
      - `16384²`: `1.303x -> 0.900x`
- Practical cleanup rule:
  - keep diagnostic modes that still bound the current single-pass design (`qk-only`, `qk-pv-raw`, `softmax-stats`)
  - remove modes that were only temporary structural detours once their data is recorded here
  - after removing them from the public harness / descriptor surface, also delete the corresponding dead generator branches from `NAInt8AttentionKernel.cpp`
    - keeping compile-time-false `row_max_only` / `replay_full` / cooperative-execution branches made the source generator much harder to read without preserving any winning behavior
    - the surviving kernel now maps one-to-one to the supported bench modes, which makes future codegen and AIR inspection much easier
- Additional cleanup after the winner stabilized:
  - remove bench-facing per-head `Q/K` scale support
    - why:
      - it was slower than tiled scales and never part of the winning path
    - data:
      - quantization alone was already much slower (`~9 ms` at `8192 x 8192 x 128`)
      - end-to-end was worse than per-tile scales
    - result:
      - the scaffold now keeps only `Q/K` scales = `none` or `tile`
  - remove bench-facing `V=fp16` / `half @ int8` support from the active kernel surface
    - why:
      - it improved accuracy, but not enough to justify the throughput loss
    - data:
      - `4096²`: `1.261x -> 0.916x`
      - `8192²`: `1.364x -> 0.977x`
      - `16384²`: `1.303x -> 0.900x`
    - result:
      - `V` is now treated as scaled int8 in the scaffold, which matches the realistic fast path
  - fix the int8 default block shape to the tuned winner
    - why:
      - the repeated sweeps converged on one robust int8 shape
    - data:
      - `blockR=16` beat `8` and `32`
      - `blockC=64` beat `32`, while `128` was much worse
      - `blockD=32` beat `64/128` on the realistic scaled-`V` path
    - result:
      - the int8 scaffold defaults are now `blockR=16`, `blockC=64`, `blockD=32`
      - manual overrides are still kept for future sweeps
  - keep the fp16 baseline block-shape selection separate from the int8 scaffold defaults
    - why:
      - reusing the int8-tuned `16x64x32` default for fp16 `NAAttention` inflated speedups by slowing the baseline itself
    - result:
      - the harness now has separate baseline and int8 block-shape helpers
      - the cleaned winner still holds with the fair baseline:
        - `8192 x 8192 x 128`: baseline `24.416 ms`, quantize+int8 `15.243 ms`, `1.602x`
        - `16384 x 16384 x 128`: baseline `105.365 ms`, quantize+int8 `65.111 ms`, `1.618x`
- `full` and `single-block-full` were also consolidated.
  - The old two-block `full` generator path was much slower than the tuned single-block path while producing the same quality level of output.
  - Representative comparisons with the same tuned hyperparameters:
    - `8192 x 8192 x 128`
      - old `full`: quantize+int8 `32.486 ms`
      - `single-block-full`: quantize+int8 `17.918 ms`
    - `16384 x 16384 x 128`
      - old `full`: quantize+int8 `114.593 ms`
      - `single-block-full`: quantize+int8 `75.915 ms`
  - Conclusion:
    - keep only the single-block schedule as `full`
    - retain `"single-block-full"` only as a harness compatibility alias
    - deleting the older `full` branch removes a large amount of generator complexity without giving up a viable benchmark path
  - This matched `NAInt8MatMul`.
  - It did *not* automatically carry over to fused attention because the softmax bridge inflated live state.
  - Larger simdgroup tiles can reduce performance once operands stop fitting fast thread-local memory.
    - This is the best guide-level explanation for why larger `blockC` was catastrophic in int8 attention while still making sense for standalone int8 GEMM.
  - Accumulation-loop synchronization is only useful when it improves cache locality enough to repay synchronization overhead.
    - We already saw this on `NAAttention`: `threadBarrierOverC` was consistently worse.
  - Morton-order threadgroup walk mainly helps cache locality for GEMM-like 2D output tilings.
    - This was a real win for `NAMatMul`.
    - It does not translate directly to the `(sequence, head, batch)` structure of `NAAttention`.

### MPP Guide Re-read: Concrete Next Experiments
- The guide did not reveal a hidden "one weird trick" for the current int8 attention scaffold.
- The most defensible next experiments from the guide + current data are:
  - keep `blockR = 16`, `blockC = 64` fixed and attack the softmax bridge, not the tile sizes;
  - replace `reduce_rows(max/sum)` in the single-pass int8 path with custom row-local reductions, because the row-max-only prototype already showed that custom reduction can be faster than the generic reduction path;
  - inspect generated AIR / code size / register allocation on the current best single-pass kernel versus fp16 `NAAttention`, because the remaining performance gap now looks more like codegen / live-state pressure than a tiling problem;
  - after fixing the `V` data-generation bias, re-evaluate whether the current `int8 @ int8` accuracy is acceptable enough for an integration path, or whether a mixed-precision accuracy mode is needed separately from the fast path.

### Bench-Only `threadgroup_barrier(mem_none)` Over `C`
- Re-read of the MPP guide suggested another bench-only experiment:
  - add `threadgroup_barrier(mem_flags::mem_none)` at a tunable cadence over the `for (c += blockC)` traversal loop,
  - then re-sweep `executionSIMDGroups` to see whether the barrier reduces divergence enough to improve cache behavior.
- This was implemented as a bench-only knob:
  - `cBarrierEvery = 0` means no extra barrier,
  - `1` means barrier every `c` iteration,
  - `4` means barrier every fourth `c` iteration.
- This is not a universal win.
- `8192 x 8192 x 128`, tuned winner (`blockR=16`, `blockC=64`, `blockD=32`, `simdgroups=2`):
  - `cBarrierEvery=0`:
    - `full avg_ms = 17.120`
    - `quantize+int8 avg_ms = 18.326`
  - `cBarrierEvery=4`:
    - `full avg_ms = 18.056`
    - `quantize+int8 avg_ms = 18.658`
  - Conclusion:
    - barrier hurts on the current `8192²` winner.
- `16384 x 16384 x 128`:
  - without barrier, the best no-barrier region was around `simdgroups = 8..16`
    - `simdgroups=8, cBarrierEvery=0`: `full avg_ms = 80.498`
    - `simdgroups=16, cBarrierEvery=0`: `full avg_ms = 80.925`
  - with barrier, the best point moved to lower simdgroups:
    - `simdgroups=4, cBarrierEvery=1`: `full avg_ms = 77.902`
    - `simdgroups=4, cBarrierEvery=4`: `full avg_ms = 79.361`
    - `simdgroups=16, cBarrierEvery=4`: `full avg_ms = 77.987`
  - Conclusion:
    - at `16384²`, a periodic `mem_none` barrier gives a real kernel win, about `3%` to `4%`,
    - and it can change the best `executionSIMDGroups` choice from `8/16` down to `4`.
- Rectangular sanity check, `8192 x 16384 x 128`:
  - `simdgroups=16, cBarrierEvery=0`: `full avg_ms = 39.176`
  - `simdgroups=4, cBarrierEvery=1`: `full avg_ms = 39.847`
  - Conclusion:
    - the barrier result is not a universal “larger C always likes barriers” rule.
    - it helped the large square case, but not enough to beat the best no-barrier rectangular point.
- Practical conclusion:
  - keep `cBarrierEvery` as a bench-only knob for now,
  - do not integrate it blindly into the production path,
  - but it is a credible lever for very large square / long-loop cases and it clearly interacts with `executionSIMDGroups`.
- Updated longer sweep:
  - the useful cadence is not `1`; it is `2`.
  - `cBarrierEvery=2` substantially improved the current single-pass kernel on all the representative shapes rechecked with longer timed runs.
- Representative longer results:
  - `8192 x 8192 x 128`
    - no barrier, `simdgroups=2`:
      - `full avg_ms = 17.120`
      - `quantize+int8 avg_ms = 18.326`
    - `cBarrierEvery=2`, `simdgroups=4`:
      - `full avg_ms = 16.420`
      - `quantize+int8 avg_ms = 17.006`
    - conclusion:
      - barrier cadence `2` improves the kernel and shifts the best simdgroups from `2` to `4`
  - `16384 x 16384 x 128`
    - no barrier, best earlier region:
      - `simdgroups=8`, `full avg_ms = 80.498`
    - `cBarrierEvery=2`, `simdgroups=8`:
      - `full avg_ms = 68.208`
      - `quantize+int8 avg_ms = 70.965`
      - end-to-end speedup `~1.48x`
    - conclusion:
      - this is a major win, about `15%` faster kernel-only than the no-barrier best
  - `8192 x 16384 x 128`
    - `cBarrierEvery=2`, `simdgroups=4`: `full avg_ms = 35.007`
    - `cBarrierEvery=2`, `simdgroups=16`: `full avg_ms = 35.121`
    - conclusion:
      - barrier cadence `2` helps strongly, and the best simdgroup count is flat between `4` and `16`
  - `16384 x 8192 x 128`
    - no barrier, `simdgroups=4`: `full avg_ms = 36.434`
    - `cBarrierEvery=2`, `simdgroups=4`: `full avg_ms = 33.317`
    - `cBarrierEvery=2`, `simdgroups=16`: `full avg_ms = 36.022`
    - conclusion:
      - barrier cadence `2` helps strongly, and this shape clearly prefers lower simdgroups
- Updated practical conclusion:
  - `cBarrierEvery=2` is the first threadgroup-barrier setting that looks genuinely promising.
  - It is no longer just a “large square maybe” lever.
  - It materially changes the best `executionSIMDGroups`, and the preferred simdgroup count does depend on shape.
  - Because the preferred simdgroup count changes with `R/C`, there is now a credible case for checking whether launch ordering also matters once this barrier cadence is fixed.
- Barrier placement matters too, but not in the way the MLX `steel_attention_nax.h` kernel suggests.
  - I briefly added a second bench-only knob to test barrier placement:
    - `tail`: barrier between `c` iterations after `PV`
    - `pre-pv`: barrier after online-softmax stats/correction, before `PV`
    - `mid-pv`: barrier halfway through the `PV` head-block loop
  - On the current winning single-pass int8 path, MLX-style `pre-pv` / `mid-pv` placement lost to the simpler tail barrier.
  - `16384 x 16384 x 128`, `simdgroups=8`, `cBarrierEvery=2`:
    - `tail`: `full avg_ms = 69.322`, `quantize+int8 avg_ms = 70.667`
    - `pre-pv`: `full avg_ms = 75.013`, `quantize+int8 avg_ms = 76.919`
    - `mid-pv`: `full avg_ms = 80.498`, `quantize+int8 avg_ms = 81.082`
  - `8192 x 8192 x 128`, `simdgroups=4`, `cBarrierEvery=2`:
    - `tail`: `full avg_ms = 16.468`, `quantize+int8 avg_ms = 17.658`
    - `pre-pv`: `full avg_ms = 17.053`, `quantize+int8 avg_ms = 17.713`
    - `mid-pv`: `full avg_ms = 18.620`, `quantize+int8 avg_ms = 19.819`
  - Conclusion:
    - the useful barrier in this kernel is the phase-alignment barrier before the next `QK` / softmax / `PV` chunk, not an MLX-style barrier inside the current chunk.
    - the placement-testing knob was removed after recording the result, and the bench path keeps only the winning tail placement plus tunable cadence.

### Bench-Only Morton Launch Order For `NAInt8Attention`
- After fixing the barrier to the winning tail placement with `cBarrierEvery=2`, the remaining simdgroup preference still varied with `R/C`.
- I then added a separate bench-only `mortonOrder` launch mapping over the flattened `(row_groups, Hq)` grid, using the same rectangular Morton decode approach already used in `NAMatMul` / `NAInt8MatMul`.
- This was a real win on every representative shape I checked, and it changed the best `executionSIMDGroups` again.
- Representative results, all with `blockR=16`, `blockC=64`, `blockD=32`, `Q/K/V=int8`, tiled `Q/K/V` scales, and `cBarrierEvery=2`:
  - `16384 x 16384 x 128`
    - no Morton, `simdgroups=8`: `full avg_ms = 69.452`, `quantize+int8 avg_ms = 69.883`
    - Morton, `simdgroups=8`: `full avg_ms = 67.149`, `quantize+int8 avg_ms = 68.201`
    - Morton, retuned `simdgroups=4`: `full avg_ms = 64.223`, `quantize+int8 avg_ms = 64.881`
    - conclusion:
      - Morton is worth about `8%` end-to-end at the retuned point relative to the no-Morton winner, and pushes the realistic path to about `1.63x`.
  - `8192 x 8192 x 128`
    - no Morton, `simdgroups=4`: `full avg_ms = 17.180`, `quantize+int8 avg_ms = 17.294`
    - Morton, `simdgroups=4`: `full avg_ms = 15.453`, `quantize+int8 avg_ms = 15.368`
    - Morton, retuned `simdgroups=2`: `full avg_ms = 14.731`, `quantize+int8 avg_ms = 15.541`
    - conclusion:
      - Morton is another strong win and keeps the realistic path around `1.59x`.
  - `16384 x 8192 x 128`
    - no Morton, `simdgroups=4`: `full avg_ms = 33.809`, `quantize+int8 avg_ms = 34.910`
    - Morton, `simdgroups=4`: `full avg_ms = 30.859`, `quantize+int8 avg_ms = 31.622`
    - conclusion:
      - Morton is clearly positive on the `R > C` rectangular case too, reaching about `1.58x`.
  - `8192 x 16384 x 128`
    - no Morton, `simdgroups=4`: `full avg_ms = 35.039`, `quantize+int8 avg_ms = 36.237`
    - Morton, `simdgroups=4`: `full avg_ms = 31.879`, `quantize+int8 avg_ms = 33.231`
    - conclusion:
      - Morton is also clearly positive on the `C > R` rectangular case, reaching about `1.56x`.
- Practical conclusion:
  - Morton launch order is the first change after the tail barrier that clearly pushes the realistic scaled `Q/K/V` int8 path beyond the `1.5x` target on the main `D=128, H=32` shapes.
  - It should be kept as a separate knob while retuning `executionSIMDGroups`, because the best simdgroup count changes again once Morton is enabled.

### `NAInt8Attention` Quantizer: Specialized `Q/K/V`, Asymmetric Thread Counts, And `vec4` Loads/Stores
- The quantizer is group-wise absmax:
  - one scale per `(batch, head, tile)`
  - `Q` grouped by `blockR`
  - `K/V` grouped by `blockC`
  - scale stored as `max_abs / 127`
  - values quantized symmetrically into signed int8 `[-127, 127]`
- The first useful cleanup was splitting the production quantizer into separate `q`, `k`, and `v` kernels and specializing shape/stride through function constants.
  - This removed the old runtime `sourceKind` branching and made it possible to tune `Q` and `K/V` separately.
- The thread-count result is asymmetric:
  - `Q=128`, `K/V=256` is better than using the same threadgroup size for all three.
  - Reason:
    - `Q` tiles are only `blockR * D`
    - `K/V` tiles are `blockC * D`
    - with the winning attention shape, `K/V` quantization has `4x` the tile work of `Q`
- After that, adding true `vec4` loads/stores on the quantizer path for `D % 4 == 0` was a real improvement and worth keeping.
  - The implementation uses:
    - `vec<IO_TYPE, 4>` device loads
    - `char4` device stores
    - scalar fallback for non-multiple-of-4 head dimensions such as `D=130`
  - Quantization still matches CPU exactly:
    - `max_abs_q_scale = 0`
    - `max_abs_k_scale = 0`
    - `max_abs_v_scale = 0`
    - `mismatched_q/k/v = 0`
- Representative results with the current defaults (`Q=128`, `K/V=256`, `fp16`, `H=32`):
  - `8192 x 8192 x 128`
    - before `vec4`: `quantize-all avg_ms = 0.639`
    - after `vec4`: `quantize-all avg_ms = 0.609`
    - end-to-end after `vec4`: `baseline avg_ms = 25.401`, `quantize+int8 avg_ms = 15.797`, `1.608x`
  - `16384 x 16384 x 128`
    - before `vec4`: `quantize-all avg_ms = 1.232`
    - after `vec4`: `quantize-all avg_ms = 1.117`
    - end-to-end after `vec4`: `baseline avg_ms = 100.400`, `quantize+int8 avg_ms = 64.485`, `1.557x`
  - scalar fallback sanity check:
    - `4096 x 4096 x 130`
    - `quantize+int8 avg_ms = 8.892`
    - validation remained tight (`max_abs_o = 4.52e-4`)
- One direction was tried and rejected:
  - staging the source tile in threadgroup memory to reuse it for both max-reduction and quantization
  - it made `quantize-only` slower and also ran into a Metal limitation with function-constant threadgroup array sizes in the generated production path
  - conclusion:
    - keep the simpler two-pass device-memory quantizer
    - keep the wins from specialization, asymmetric `Q/KV` threads, and `vec4` loads/stores
