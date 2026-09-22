# NAInt8MatMul fused regular-Hadamard activation quantization

`NAInt8MatMulDescriptor::activationHadamard256 = true` opts into a fused
rotation + row maximum + INT8 packing kernel. The default remains false.
This probe exercises the kernel directly. The public ROWWISE_X H256 modifier
now prepares matching weights and enables activation rotation through MPS GEMM.

## Numerical contract

For input `A[M,K]` and output-channel-major weights `W[N,K]`, split each row
into consecutive groups of 256 features. In each group use

```
H4 = [ 1  1  1 -1
       1  1 -1  1
       1 -1  1  1
      -1  1  1  1 ]
R256 = (H4 tensor H4 tensor H4 tensor H4) / 16
```

This is the **regular** Hadamard transform used by ConvRot. It is not the
Sylvester transform computed by the existing WHT operator. The offline weight
transform must match its ordering, signs, and normalization exactly.

Prepare `W_rot = W R` offline, then quantize each output row and store its scale.
At inference the new variant computes `A_rot = A R` in FP32 registers and
quantizes each **full row**, not each 256-element group:

```
s[row] = max(abs(A_rot[row,:])) / 127
q[row,k] = clamp(rint(A_rot[row,k] / s[row]), -127, 127)
```

For an all-zero row, the existing convention `s = 1/127, q = 0` is retained.
Scales are stored in `ioPrecision` (FP16, BF16, or FP32); the INT8 GEMM and
its scale/bias epilogue are unchanged. Orthogonality ensures
`(A R)(W R)^T = A W^T` before quantization. This activation-only mode has no output inverse rotation.
Using the flag with ordinary, unrotated weights computes the wrong operation.

The staged byte layout is still contiguous signed INT8 rows plus one scale per
row. The persistent ROWWISE_X H256 modifier records the transform while retaining
the existing packed payload layout. See the [format contract](../../doc/rowwise-x-hadamard256-format.md)
for weight conversion, metadata, runtime routing and deferred dense fallback.

## Implementation boundaries

- Positive `K`, divisible by 256, with `K <= 65536` in this first variant.
- One 256-thread threadgroup per row; one SIMD group handles a 256-feature
  block at a time. Rotated values stay in registers until the row reduction.
- One activation read and one INT8 write, without an intermediate rotated
  tensor or an extra GPU dispatch.
- FP32 arithmetic for rotation and maximum reduction; input/output scale
  precision follows the existing descriptor.
- Static/dynamic M, leading dimensions, and independent source/packed/scale
  batch strides retain their existing meanings.
- Rotation mode is in both descriptor equality/hash keys. K remains a function
  constant, not a stored kernel shape. Metal does not permit function constants
  as thread-array lengths, so a fixed-capacity array is used and unused slots
  are eliminated during specialization. This is the reason for the initial K
  bound.

## Reproduce

From `bin/mfa`:

```sh
make na_int8_convrot_bench -j4
./na_int8_convrot_bench --validate-only
./na_int8_convrot_bench --skip-validation --iterations 31 --repeats 5
./na_int8_convrot_bench --skip-validation --shape 4096 6144 16384 --iterations 101 --repeats 5
```

Metal execution requires access to the GPU outside the restricted Codex sandbox.

Validation forms the 256x256 reference matrix explicitly in CPU FP64, checks
all activation values and row scales, and compares GEMM with an INT32 CPU dot
product using the actual staged bytes. Cases include zero/constant rows,
random values with channel outliers, K=256/768/2304 in FP16/BF16/FP32, plus
FP16 K=5376/16384/65536 boundary cases, M/N tails,
bias, batched strides, dynamic M with untouched inactive rows, unaligned source
strides, and switching between cache entries. It also compares output NRMSE
against the original floating-point operation, with weights rotated offline.
These synthetic outlier checks demonstrate the mechanism, not model accuracy.

Timings use the production descriptor/pipelines. Both arms use the exact same
GEMM pipeline and packed weight bytes to isolate staging overhead. Every timed
command repeats the selected operation (default 3, recorded runs 5). There are
five warmups, alternating arm order, and separate measurements for quantization,
GEMM, and quantization+GEMM. `paired_overhead_pct` is the median of adjacent
rotated/plain timing ratios; `overhead_pct` is the ratio of separate medians.
`paired_p10_pct`/`paired_p90_pct` show sample spread, not confidence limits.
`staging_delta_pct` divides the isolated quantizer median difference by the
plain total median; it is an estimate, not an end-to-end measurement.
All measurements use synthetic tensors and report GPU command-buffer time;
“total” means activation staging + one NAInt8MatMul, not a full diffusion model.

## Shape sources

The default suite uses M=4096 and M=16384 as representative token counts, not
as claims about a particular prompt, resolution, or video duration. N and K
come from the model definitions:

- [Krea 2 transformer implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_krea2.py):
  48 query heads x 128 = 6144 hidden width; 12 KV heads x 128 = 1536;
  SwiGLU intermediate width 16384. Shapes `(N,K)` are `(6144,6144)`,
  `(1536,6144)`, `(16384,6144)`, `(6144,16384)`.
- [MiniMax H3 released configuration](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/transformer/config.json)
  and [transformer implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py):
  hidden width 5376, 56 heads x 128 = 7168 attention width, intermediate width
  14336. Shapes are `(7168,5376)`, `(5376,7168)`, `(14336,5376)`, `(5376,14336)`.

The rotation follows [ConvRot](https://arxiv.org/abs/2512.03673) and the
[Comfy regular Hadamard implementation](https://github.com/Comfy-Org/comfy-quants/blob/main/src/comfy_quants/formats/convrot.py).

## Recorded results

See the [activation-only measurements](../../doc/benchmarks/convrot-h256-20260921/README.md).
Historical output-rotation and tile experiments are archived under
`doc/benchmarks/`; those modes are no longer in this activation probe.

The persistent weight modifier is implemented separately; see
[the H256 format contract](../../doc/rowwise-x-hadamard256-format.md).
