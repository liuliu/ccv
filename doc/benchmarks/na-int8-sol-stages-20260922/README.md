# SOL single-pass mean and stage-fusion experiments

Production now uses the five-launch organization. The build script reconstructs
archived seven-launch sources with `seven-launch.patch`, then applies the original
fusion/two-pass patches to temporary copies. Separate kernel/cache types preserve
the exact historical baselines without modifying production files. Zero-context
patches are protected by SHA-256 checks of every current base file.

For pooling/query blocks of 64, compare:

- **8 launches:** original two-pass V-mean and separate subsequent stages.
- **7 launches:** single-pass V-mean and separate subsequent stages (archived).
- **5 launches:** single-pass mean; quantize/pool; combined summary preparation
  and key statistics; routing; combined summary and exact attention.
- **4 launches:** same preparation, with routing also in the attention kernel.

B16/B32 retain a separate pooling launch, adding one to these counts. Combining
summary and exact attention removes the FP32 intermediate numerator and softmax
state from device scratch. The four-launch experiment keeps the byte/packed
routing buffers in device scratch and uses threadgroup device-memory barriers;
it does not add sequence-dependent threadgroup memory. Summary and exact loops
are generated separately with CodeWriter, preserving their traversal order.

From the repository root:

```sh
make -C test/int/nnc debug -j4
python3 doc/benchmarks/na-int8-sol-stages-20260922/build.py /private/tmp/sol-stages-repro
/private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=3 /private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=3 SOL_UNALIGNED_INPUT=1 /private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=3 SOL_SCALE=-0.125 /private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=3 SOL_ZERO_Q=1 SOL_LARGE_V=1 /private/tmp/sol-stages-repro/compare --verify

# Isolated mean validation: bitwise comparison to the original reduction order,
# FP64 accuracy check, scalar-aligned FP16 input offset, output sentinels, and fixed memory/thread caps.
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
  /private/tmp/sol-stages-repro/mean_verify

# Compile production and all experimental stages over representative shape specializations.
SOL_REPORT_RESOURCES=1 MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  /private/tmp/sol-stages-repro/resources

# Compare instrumented fusion outputs to saved uninstrumented reference outputs.
mkdir -p /private/tmp/sol-fixtures
SOL_FUSION=3 SOL_RECORD=1 SOL_FIXTURES=/private/tmp/sol-fixtures \
  /private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=3 SOL_REPLAY=1 SOL_FIXTURES=/private/tmp/sol-fixtures \
  MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
  /private/tmp/sol-stages-repro/compare --verify

# Set SOL_FUSION=0 for archived two-pass versus archived single-pass.
# Set 1 or 2 for paired comparisons to five or four launches; 3 compares all three.
SOL_FUSION=3 /private/tmp/sol-stages-repro/compare 32768 56 6
SOL_FUSION=3 /private/tmp/sol-stages-repro/compare 65536 56 6
SOL_FUSION=3 /private/tmp/sol-stages-repro/compare 103982 56 6
```

Timings include all preprocessing, use three warmups, then six measured rounds,
and rotate execution order. Do not benchmark with Metal validation enabled.
Inputs are uniform seed-42 synthetic Q/K/V, N=1/H=56/D=128, scale=1, tau=0.5,
protected prefix=470. Sparse radius=1; all-exact radius covers all KV blocks.
These measurements compare kernel organizations, not model quality.

Output comparisons reject non-finite values and guard corruption. The mean-only
change requires bit-identical complete attention outputs. Fusion allows relative
L2 up to 1e-5 and per-element relative error up to 0.002 (denominator floored at
1), to accommodate FP16 rounding after compiler changes. Raw logs report actual
errors, not just pass/fail.

The full separate-stage attention pipeline has an intermittent long-tail
numerical failure with shader instrumentation enabled. This also occurs in the
saved two-pass baseline, while the isolated V-mean passes. See the parent
[implementation notes](../../na-int8-sol-attention.md) for results and validation
limits; do not treat the complete instrumented pipeline as passing on the basis
of the isolated mean/resource checks.

## Production five-launch validation

`SOL_FUSION=4` compares the saved five-launch prototype (head-packed scratch)
to current production (token/head-order scratch).
Every execution, including warmups, is checked for finite outputs. Production
comparisons also check relative error every round. Verification supports
`SOL_REPEATS`, `SOL_SEED`, and `SOL_LONG_TAIL` (T=32769/H=2 only).

```sh
SOL_FUSION=4 SOL_REPEATS=3 /private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=4 SOL_LONG_TAIL=1 SOL_REPEATS=20 SOL_SEED=17 \
  MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 \
  /private/tmp/sol-stages-repro/compare --verify
SOL_FUSION=4 /private/tmp/sol-stages-repro/compare 32768 56 6
SOL_FUSION=4 /private/tmp/sol-stages-repro/compare 65536 56 6
SOL_FUSION=4 /private/tmp/sol-stages-repro/compare 103982 56 6
```

Repeat the instrumented stress run with `SOL_SEED=99`. The `production-*.txt`
logs record commands, environment, results, and exit status. Historical
`validation-*.txt` logs describe the earlier implementations; their separate-stage
shader failure is superseded by the production validation results.

Final production results: CPU 2/2, instrumented MPS 4/4, 43 production resource
specializations, 16 isolated mean configurations, and repeated production/prototype
comparisons all passed. The 80 long-tail stress executions across two seeds
reported no non-finite outputs. Paired speedups against the prototype ranged
from 0.99612x to 1.01879x over 32k/65k/104k, with bit-identical final outputs.
At 32k, native-relative paired speedups were 1.0506x all-exact and 3.0721x sparse.
See the parent implementation notes for the full table and timing limitations.


## Token-order scratch correction

Current production quantizes into `[N,T_padded,H,128]` and reads Q/K/V tiles
with token stride `H*128`. It preserves token/head order; `T_padded` only adds
zero-filled sequence-tail rows. No transpose is performed during quantization.
For `SOL_FUSION=4 --verify`, the comparison also copies quantized scratch back
and checks every Q/K/V byte at its expected token/head offset against the
archived prototype, including batches and padding. `TOKEN_LAYOUT_PASS` records
this check independently of final output parity.

The `production-*.txt` logs above describe the former head-packed version;
`token-order-*.txt` logs validate and benchmark the corrected layout. Commands
and environments are recorded in each log. Native comparisons use the actual
all-exact SOL path and include every preprocessing launch.


Final token-order production also uses native-style rectangular Morton attention
dispatch across query tiles and heads, with batches in grid.z. This changes work
scheduling only. It retains the original four SIMD groups for Q64, cached Q,
masking, PV arithmetic, and synchronization. The `token-order-final-*.txt` logs
contain the repeated final validation and six-round native comparisons. Earlier
`token-order-*.txt` logs refer to linear attention dispatch.

The final six-round native-relative paired speedups were 1.0805x / 0.9452x /
0.9997x dense and 3.1615x / 2.7447x / 2.8181x sparse at 32k / 65k / 104k.
A twelve-round 65k confirmation balanced all six execution permutations and
measured **1.0163x dense / 2.9464x sparse**; its log is
`token-order-final-native-65536-balanced.txt`. This checks the initial 65k gap
without changing the production kernel. Both measurements are retained to show
run/order variation. Reproduce the balanced confirmation with:

```sh
make -C bin/mfa na_int8_sol_attention_bench
bin/mfa/na_int8_sol_attention_bench 65536 56 12
```

Run timings without Metal validation; use the instrumented commands above for
correctness. All final validation logs have `EXIT: 0`, including the direct
scratch-layout checks and 80 repeated long-tail executions.
