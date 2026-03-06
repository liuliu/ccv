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
