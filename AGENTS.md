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
- `./mpsdnn.tests`: success (`82/82`, `EXIT:0`)
- `./mpsblas.tests`: success (`61/61`, `EXIT:0`)

## Note for sandboxed agent runs

In this Codex environment, MPS test binaries may crash in the restricted sandbox (`EXIT:139`). Running these binaries with unrestricted execution resolves that environment-specific issue.

## Generic Compilation Troubleshooting

If a generic compilation flow fails, remove stale `.dep.mk` files in directories and retry the build.

Example from repo root:

```sh
find . -name .dep.mk -delete
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
- `grid_sample` integration test specifics:
  - NCHW path can be guarded by `CCV_NNC_BACKEND_GPU_CUDNN || CCV_NNC_BACKEND_MPS`.
  - NHWC path is currently guarded by `CCV_NNC_BACKEND_MPS` (cuDNN implementation path is NCHW-only internally).
