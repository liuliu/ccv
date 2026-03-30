# macOS Agent Notes

## NNC Debug Build + MPS Test Flow

From repo root:

```sh
cd test/int/nnc
make debug -j4
./mpsdnn.tests
./mpsblas.tests
```

Expected result for both test binaries:

```text
all test case(s) passed, congratulations!
```

## Verified on this workspace

- `make debug -j4`: success
- `./mpsdnn.tests`: success (`108/108`, `EXIT:0`)
- `./mpsblas.tests`: success (`130/130`, `EXIT:0`)

## Note for sandboxed agent runs

In this Codex environment, MPS test binaries may crash in the restricted sandbox (`EXIT:139`). Running these binaries with unrestricted execution resolves that environment-specific issue.

## Generic Compilation Troubleshooting

If a generic compilation flow fails, remove stale `.dep.mk` files in directories and retry the build.

Example from repo root:

```sh
find . -name .dep.mk -delete
```

## UBSAN Build Flow (`test/unit/nnc`)

From this workspace, run UBSAN from `test/unit/nnc` (not repo root):

```sh
cd test/unit/nnc
make clean
make ubsan -j64
```

Notes:

- `make clean` is not available at repo root (`make: *** No rule to make target 'clean'.`).
- In this Codex sandbox, ASAN leak detection may fail early with:
  - `LeakSanitizer has encountered a fatal error`
  - `LeakSanitizer does not work under ptrace`
- To run UBSAN/ASAN-built unit tests in this environment, disable leak detection:

```sh
ASAN_OPTIONS=detect_leaks=0 ./dynamic.graph.tests "<filter>"
```

## Command Registry Generation (`lib/nnc/cmd`)

Do not hand-edit these generated files:

- `lib/nnc/cmd/ccv_nnc_cmd.inc`
- `lib/nnc/cmd/ccv_nnc_cmd.h`
- `lib/nnc/cmd/ccv_nnc_backend.h`
- `lib/nnc/cmd/ccv_nnc_cmd_easy.h`

Generate them with the script:

```sh
cd lib/nnc/cmd
./build-cmd.rb .
```

Then verify build artifacts are still compilable (example):

```sh
cd test/int/nnc
make debug -j4
```

Note: generated content may differ slightly across machines / environments, but should still produce compilable artifacts.

To avoid polluting commit history after local validation, restore generated files back to tip:

```sh
git checkout -- lib/nnc/cmd/ccv_nnc_cmd.inc lib/nnc/cmd/ccv_nnc_cmd.h lib/nnc/cmd/ccv_nnc_backend.h lib/nnc/cmd/ccv_nnc_cmd_easy.h lib/nnc/cmd/config.mk
```

## Session Learnings

- Branch sync policy: when asked to keep a branch up to date with another branch, use `git rebase` instead of `git merge` (unless explicitly requested otherwise).
- Operator file naming convention (generic):
  - `ccv_nnc_OPS.c`: operator metadata / registry logic (in-place support, tensor shape inference, etc.).
  - `ccv_nnc_OPS_cpu_ref.c`: CPU reference implementation.
  - `gpu/ccv_nnc_OPS_gpu_cudnn.cu`: GPU implementation via cuDNN.
  - `gpu/ccv_nnc_OPS_gpu_ref.cu`: GPU implementation via direct CUDA kernels.
  - `mps/ccv_nnc_OPS_mps.m`: Apple MPS backend implementation (MPSGraph / MFA).
- Integration test philosophy (`test/int/nnc`):
  - Prefer larger randomized tensors over tiny hand-crafted examples.
  - Validate GPU outputs by comparing against CPU reference implementation outputs, using the same command and compatible tensor formats.
  - Cover both `NCHW` and `NHWC` layouts when backend support exists.
  - Use tolerance-based comparisons for floating-point parity (`REQUIRE_ARRAY_EQ_WITH_TOLERANCE`).
- MPS SDPA test notes:
  - For the `scaled dot product attention + unify head` integration test in `test/int/nnc/mpsblas.tests.c`, prefer a relative-difference check over a pure max-absolute-difference check.
  - In the current workspace, the intermediate attention output drift was small (`~4.5e-4` max abs diff), but the downstream `512`-wide unify-head projection amplified that to about `0.2` max abs diff in the final output.
  - A robust comparison there is `fabs(a - b) / max(max(fabs(a), fabs(b)), 1)` with a threshold around `2e-3`.
- MPS SDPA NA-attention gating note:
  - The neural-accelerator attention path lowers the head tile to the MPP `matmul2d` `N` dimension, which must be a multiple of `8` or `16`.
  - Small head dimensions such as `D = 4` can compile-fail on the NA path even though the generic MFA / non-NA path is valid.
  - In `lib/nnc/cmd/scaled_dot_product_attention/mps/ccv_nnc_scaled_dot_product_attention_mps.m`, gate `use_neural_accelerators` conservatively for SDPA so `D <= 128` only uses NA attention when `(D % 8) == 0`.
- `grid_sample` integration test specifics:
  - NCHW path can be guarded by `CCV_NNC_BACKEND_GPU_CUDNN || CCV_NNC_BACKEND_MPS`.
  - NHWC path is currently guarded by `CCV_NNC_BACKEND_MPS` (cuDNN implementation path is NCHW-only internally).
- Variable naming convention preference:
  - Prefer `<tensor>_nd` style names (for example, `a_nd`, `b_nd`, `w_nd`) as the default rule.
  - `adim` / `bdim` / `astride`-style names are acceptable as specific shape/stride-array exceptions, not the general naming pattern.
- Assertion-only conditional style:
  - Do not write `if (...) assert(...);`.
  - Always brace assertion-only conditionals as `if (...) { assert(...); }` so release builds do not change control flow when assertions compile out.
- `ctags` usage expectation and workflow:
  - `ctags` should be available and used to discover reusable helpers before introducing local utility functions.
  - Practical flow:
    - List relevant helper functions from common headers: `ctags -x --c-kinds=f lib/nnc/ccv_nnc_easy.h lib/nnc/ccv_nnc_internal.h`.
    - Filter by intent (example): `ctags -x --c-kinds=f lib/nnc/ccv_nnc_easy.h lib/nnc/ccv_nnc_internal.h | rg 'tensor_get_|tensor_hw|tensor_view_get_'`.
    - Reuse discovered existing helpers when possible, instead of adding local utility functions.
- MFA cache / dispatch rule:
  - Distinguish the two cache layers clearly:
    - the kernel-object cache should only key source-generation properties;
    - the pipeline / function-constant layer should carry shape-specific values such as `length`, `rowLength`, `scaleOffset`, etc.
  - If a kernel's dispatch geometry depends on runtime shape, do not store that shape inside the cached kernel object.
  - Instead, derive `gridSize(...)` from the current descriptor / params at encode time.
  - Concrete example: `Dequantize8iRowwiseKernel` should stay shape-agnostic, and `gridSize(length)` should use the current `length`; otherwise a later larger dequant can silently reuse a stale smaller dispatch and leave the tail zeroed.
- MFA Conv3D / `NAConv3D` implementation notes:
  - Frontend selection in `lib/nnc/cmd/convolution/mps/ccv_nnc_conv_mps.m` should keep `use_mfa_gemm` and `use_mfa_conv3d` separate.
  - Current `use_mfa_conv3d` support surface is intentionally narrow:
    - 3D convolution only.
    - kernel depth must be `3`.
    - spatial kernel may be any odd square (`3x3`, `5x5`, `7x7`, ...).
    - stride and dilation must be `1`.
    - depth padding is unsupported.
    - input / output channels must both be divisible by `16`.
    - NA hardware must be available in production code.
  - `ccv_nnc_mfa_prepare_conv3d(...)` should stay a no-op, like other MFA prepare entry points that do not need eager work.
  - `NAConv3D` uses the same batching pattern as `NAMatMul`:
    - do not loop batch on the host;
    - use `threadgroup_position_in_grid.z` to encode `batch * output_depth`.
    - host dispatch should iterate only across kernel-depth slices.
  - Conv3D weights are currently accepted in OIDHW / NCHW layout and must be permuted to DHWIO scratch before the MFA kernel runs.
  - Conv3D scratch reservation should mirror GEMM:
    - reserve the front of MFA scratch for permuted weights with `ccv_nnc_mfa_conv3d_reserved_scratch_size(...)`;
    - if weights are palettized, depalettize after that reserved region so the two scratch uses do not overlap.
  - Bias support in `NAConv3D` is fused only into the first multiply kernel:
    - `conv3d_multiply` initializes the destination from bias (or zero);
    - later `conv3d_multiply_accumulate` slices only accumulate.
  - Spatial padding support rules:
    - use named fields `padding_left`, `padding_right`, `padding_top`, `padding_bottom` in the C MFA params.
    - normalize asymmetric padding conservatively by preserving `right` / `bottom` and deriving `left` / `top` from output shape.
    - this matches the repo's existing "prefer more padding in the beginning" rule from `ccv_nnc_hint_auto(...)`.
  - Descriptor / kernel descriptor rule:
    - any non-derived value needed by `NAConv3DKernelDescriptor` must also be present in `NAConv3DDescriptor`;
    - otherwise shader cache keys and generated kernel source can diverge.
    - padding is a `KernelDescriptor` property and should be inlined into kernel source, not passed with `setBytes`.
  - Padded Conv3D test specifics:
    - `hint.border.begin/end` for 3D convolution are `D/H/W` only; channels are not part of hint borders.
    - for padded 3D test cases, prefer explicit `stride = 1` hints instead of `ccv_nnc_hint_auto(...)`;
    - `ccv_nnc_hint_auto(...)` can infer the wrong depth stride for padded 3D shapes, causing `ccv_nnc_hint_verify(...)` to fail before the MFA path is exercised.
  - Test style for `test/int/nnc/mpsdnn.tests.c`:
    - keep focused test cases self-contained instead of factoring a small local helper shared across several similar cases.
  - Local validation workflow for `NAConv3D` on a machine without neural accelerators:
    - temporarily force `use_neural_accelerators = 1` in `ccv_nnc_conv_mps.m`;
    - run `./mpsdnn.tests "mfa conv3d"` from `test/int/nnc`;
    - revert the force after validation so production code uses `ccv_nnc_mfa_has_neural_accelerators(context)`.
- `NAInt8Attention` backward `dS` fallback note:
  - Earlier exploration suggested `dS -> half` might be a fallback worth keeping in mind, but on the current shipped `D=128` fixed-quant setup it is not a win.
  - Rechecked on `4096 x 4096 x 128` with the current selector:
    - fixed-quant `dS`: forward median `4.0495 ms`, backward median `21.8308 ms`, ratio `5.3910x`
    - `dS -> half`: forward median `4.0552 ms`, backward median `23.0083 ms`, ratio `5.6737x`
  - Takeaway:
    - on the current `NAInt8Attention` backward path, `dS -> half` regresses relative to fixed-quant `dS`
    - do not treat it as the preferred fallback without reworking the kernel again
- `NAInt8Attention` backward fixed-quant selector note:
  - For the shipping `D=128` low-precision backward path, the safe production rule is:
    - query: `blockR=16`, `blockC=32`, `blockD=32`, `executionSIMDGroups=4`
    - key/value: `blockR=16`, `blockC=64`, `blockD=64`, `executionSIMDGroups=16`
  - Trust the backward absolute times more than any single reported ratio; forward medians on the probe can move enough to make one-off ratios look too optimistic.
  - Reliable current probe numbers are in this range:
    - `4096 x 4096 x 128`: backward median about `21-23 ms`, typically around `5.2x-5.6x`
    - `8192 x 8192 x 128`: backward median about `82-87 ms`, typically around `5.2x-5.4x`
  - Wider key/value traversal (`blockC=96`) can benchmark slightly faster in the probe but is not accuracy-safe on the real gradient test surface; keep `blockC=64` in production.
