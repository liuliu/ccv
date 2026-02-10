# Learnings (Swish Beta Work)

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
  - `lib/nnc/mfa/v2/SwishDescriptor.hpp`
  - `lib/nnc/mfa/v2/SwishDescriptor.cpp`
- Added beta into Swish kernel generation:
  - `lib/nnc/mfa/v2/SwishKernel.hpp`
  - `lib/nnc/mfa/v2/SwishKernel.cpp`
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
