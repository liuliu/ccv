Permanent NA application benchmarks

Build from the CCV root after configuring lib/config.mk for MPS:
  make -C lib lib DEBUG=1 -j4
  make -C bin/mfa -o libccv.a na_application_bench

Build actual MLX independently (a static Metal-JIT build is required):
  cmake -S ~/workspace/mlx -B ~/workspace/mlx/build-na \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DMLX_METAL_JIT=ON \
    -DMLX_BUILD_CPU=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF
  cmake --build ~/workspace/mlx/build-na -j4
  make -C bin/mfa mlx_na_bench MLX_ROOT=~/workspace/mlx \
    MLX_BUILD=~/workspace/mlx/build-na

MLX revision 25532871329fe4cafdada9fa0941a1cdbfb3a9f2 needs the accompanying
mlx-dsplit-jit.patch for D256 JIT attention. Check whether your revision already
contains the fix before applying it. It removes an extra template argument;
no math or matmul tuning changes. Historical comparisons used an isolated
static library with exactly this patch. MLX_LIBRARY, MLX_METAL_CPP and MLX_FMT
make variables can point at an existing build. The probe uses internal MLX C++
APIs, so newer incompatible MLX revisions may require adapting the probe.

Run from the CCV root; keep compilation and all other GPU work out of timing:
  python3 bin/mfa/na_benchmark_tests.py
  python3 bin/mfa/na_application_sweep.py --suite smoke --output /tmp/na-smoke
  python3 bin/mfa/na_application_sweep.py --suite inference --output /tmp/na-inference
  python3 bin/mfa/na_application_sweep.py --suite coverage --output /tmp/na-coverage
  python3 bin/mfa/na_application_sweep.py --suite backward --output /tmp/na-backward

The generated manifest has 527 distinct requests: 257 application cases,
266 coverage-tagged cases (some overlap), eight backward cases, and two known
low-precision stress cases. The coverage suite includes inference. Use --list
before running. --operation, --id (repeatable), and --variant select subsets.
--suite all includes known stress failures; --suite stress isolates them.
--no-mlx is supported for forward-only validation when MLX is unavailable.
Backward requires MLX, checks that the timed primitive is its three-output
SDPA VJP, and validates dQ/dK/dV. MLX_SDPA_* tuning overrides are cleared.

Defaults: three warm iterations AND at least 0.5 GPU seconds warmup, seven
samples, two reversed process orders. Each operation has its own process;
compilation, allocation, weight preparation, CPU checks and output hashing are
outside GPU timestamps. Processes run serially. Long H3 cases need several GiB
of memory; use --id/--operation on a memory-constrained machine. The runner has
a per-process timeout, records errors, continues the sweep and exits nonzero
on any failed check. Run Metal tools outside a sandbox that denies GPU access.

Output contains metadata.json, records.jsonl, results.csv, summary.json, raw
logs and exact commands. MLX and CCV revisions, binary/source hashes, device,
OS, memory and complete selected workload definitions are captured. --resume
requires identical metadata and preserves completed results. A failed result
is retained; use a new output directory for a deliberate rerun. Backward dumps
are sampled at 16,384 positions per gradient and removed after checking;
--keep-gradients preserves them. Both probes scan full gradients for finiteness.

Before/after checks on one machine:
  # Save a probe linked against your baseline library before rebuilding CCV.
  python3 bin/mfa/na_application_sweep.py --suite inference \
    --baseline-frontend /path/to/baseline-probe --output /tmp/na-paired
  python3 bin/mfa/na_application_sweep.py --compare /tmp/na-before /tmp/na-after

Live baselines are interleaved with current CCV and MLX, then reversed. Saved
comparisons require a completed run with matching device/OS/memory, workloads, variants and timing
protocol. A change is flagged when its median regresses by more than both 5%
and 0.02 ms; configure --relative/--absolute-ms when appropriate. Failures,
missing cases and missing repeats never count as successful comparisons.
Inspect raw samples and repeat flagged cases: a threshold is not a statistical
proof. Across different M5 machines, compare CCV to that machine's actual MLX;
do not apply Ultra absolute times as a performance requirement.

What is measured:
- FP16 inputs/output, with separate half-allowed and explicit FP32-accumulator
  GEMM requests. The matmul matrix uses B=1; batched/broadcast GEMM correctness
  is covered by integration tests. MLX chooses its own accumulation/tile policy.
- W8A8 matmul includes activation quantization; weights are rowwise INT8 and
  already expanded. Packed i8x/palettized weight decoding is excluded here and
  covered by integration correctness tests, not claimed as end-to-end speed.
- Attention starts in B/S/H/D storage. MLX receives strided metadata views;
  there is no explicit S/H storage transpose in either benchmark. All MLX
  primitive stages, including any internally required copies, are timed.
- D256 short-range and supported causal-inference INT8 requests mirror the
  application's FP16 fallback. The log exposes attention_fp16_fallback; do not
  call that a W8A8 kernel speedup. Causal training retains its existing path.
- Cast workloads include the Float32 conversion. INT8 asks for the production
  half-rounded Float32 epilogue and falls back when unsupported. Full output
  hashes and exact half-to-float checks are recorded. This is operator timing;
  CNNP graph/alias/reinitialization correctness is tested in mpsblas.tests.
- VJP excludes saved forward work in both implementations. It is distinct from
  inference attention; D256 can use the existing generic backward fallback.

Source geometry and limitations:
na_benchmark_manifest.py stores Draw Things source paths/revision, formulas and
assumptions. H3 UI cases are exactly R=C=31018 (832x512, 10s) and 70186
(1280x768, 10s), 56 heads, D128, with 256 text + 810 audio tokens and no references.
The 240 requested frames round to 72 latent / 243 decoded frames; video tokens
are 72*(height/32)*(width/32). They
measure dense SDPA, not Sol approximate attention. Image cases use 512/768/
1024/1536/2048 resolutions; text lengths are stated scenarios. Local Code
covers Qwen3.5 4B/9B/27B projections and growing-cache causal attention.
The deterministic periodic probe input is for reproducible timing and a sampled
independent CPU check (1% FP16 / 5% INT8 normalized L2). Randomized, shifted,
large-magnitude, cache-reuse, alias, bias and decoded-weight tests live in
mpsblas.tests.c. These operator checks do not establish generated-media quality.
BF16, all layouts, masks and training configurations are not benchmarked here.

na_m5_sweep.py and the older standalone NA probes remain diagnostic tile-sweep
tools. They intentionally bypass production selection and are not application
performance baselines. See doc/benchmarks/m5-ultra-na-2026-10-02/ for validated
results, rejected experiments and the gating review.
