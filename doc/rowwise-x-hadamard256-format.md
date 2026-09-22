# H256 modifier for rowwise-x weights

Implemented: `CCV_NNC_QX_8I_ROWWISE_HADAMARD_256 = 0x100`.
The short name is **H256**. This modifier composes with all ten existing
`CCV_NNC_QX_8I_ROWWISE_X` codecs, including IQ2_XXS and Q6_K.

```c
const int format = CCV_NNC_QX_8I_ROWWISE_Q6_K |
  CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
ccv_nnc_tensor_param_t quantized = ccv_nnc_tensor_8i_rowwise_x(params, format);
size_t bytes = ccv_nnc_quantize_8i_rowwise_x(
  weights, datatype, CCV_TENSOR_CPU_MEMORY, count, K,
  format, imatrix, imatrix_length, output, output_bytes);
// bytes == 0 means quantization failed; output remains untouched.
```

## Representation

`tensor.info.reserved` stores the complete codec + modifier. The low byte is
`CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK = 0xff`; bit 8 is H256. IQ2_XXS+H256 is
`0x108`; Q6_K+H256 is `0x10a`.
Ten of the 255 available nonzero codec IDs are used; 245 remain.

The payload layout is unchanged: packed groups, padding to a 128-byte boundary,
and a source-precision scale per row. Codec packing groups (8, 16 or 32 values)
are independent of the 256-feature rotation groups. Sizing helpers extract the
base codec; tensor construction and persistence retain the full metadata.
SQLite round trips require no schema change. Older readers do not support the
flagged representation. Merely tagging an existing payload does not rotate it.
The modifier is supported on ROWWISE_X, not ordinary unpacked ROWWISE.

## Transform and quantization

H256 is the normalized **regular** Hadamard matching NAInt8MatMul activation
staging, not the repository's Sylvester WHT:

```text
H4 = [ 1  1  1 -1
       1  1 -1  1
       1 -1  1  1
      -1  1  1  1 ]
R256 = (H4 tensor H4 tensor H4 tensor H4) / 16
```

For each weight row of W[N,K], R_K applies R256 independently to contiguous
256-feature blocks. The offline quantizer transforms its existing floating row
buffer before max-absolute-value calculation, row-scale fitting and codec packing.
It uses the quantizer's existing double-precision working values; source and
stored scale types remain FP16, BF16, FP32 or FP64. No rotated tensor allocation
or second packing pass is needed. Rows and experts remain independent.

```text
Offline: W_rot = W R_K
Runtime: X_rot = X R_K
         Y = X_rot W_rot^T + bias = X W^T + bias
```

The equality is before quantization error. There is **no output Hadamard** and
no rotation constraint on N. Output scaling and original-coordinate bias use
the ordinary epilogue. The discarded output-H256 experiment has been removed.

When supplied, `imatrix` must describe the **rotated activation basis**; the
quantizer does not reinterpret or rotate original-basis diagonal importance.
Existing per-slice calibration semantics remain unchanged.

## Validation and runtime support

- H256 requires positive K divisible by 256. No padding or tail transform.
- Quantization and explicit byte-size queries return 0 for invalid shape;
  quantization also returns 0 for insufficient output space, before
  modifying output. These return-value checks remain active with `NDEBUG`.
  Tensor construction/sizing assert that flagged K is valid.
- MPS GEMM extracts the modifier and routes supported flagged weights through
  ANE or NAInt8MatMul activation staging, including the NA small-M path. Raw
  packed-to-INT8 decode receives only the base codec and preserves rotated values.
- ANE and NA activation quantization require K <= 65536 and their supported
  dtype/layout/batching. ANE supports FP16/BF16/FP32, shared weights across
  batches, and optional shared bias on Apple9 or newer. The NA path requires
  NA hardware. If neither path applies (or ANE declines), MPS GEMM reconstructs
  logical floating-point weights and uses ordinary MFA GEMM/GEMV or MPS.
  Packed scaled-GEMV remains disabled for H256.
- ANE uses one activation-preparation dispatch for K <= 8192. Rotated values
  stay in registers through row reduction and quantization, then write directly
  into the transposed ANE surface. Eight rows share two alternating 8 KiB
  threadgroup tiles, packing eight adjacent rows per store. Three SIMD groups
  per row avoid unused transform groups when K <= 6144 is divisible by 768
  but not 1024; otherwise four SIMD groups handle each row.
- Wider ANE rows use two dispatches: compute each row's transformed maximum,
  then recompute H256 in 16-row by 256-feature tiles and quantize/transpose.
  Only float maxima cross dispatches, in the existing scale buffer's reserved
  capacity. There is no intermediate device activation buffer in either path.
  Normalization by 1/16 is folded into the stored/reconstructed activation
  scale; output dequantization still rounds that scale to the source precision.
  Padding repeats the last valid row. Both shader-library and pipeline caches
  distinguish H256; shape and strides remain function constants, and dispatch
  geometry is derived at encode time.
  See the [Hadamard optimization follow-up](ane-hadamard256-optimization.md),
  [ordinary quantization benchmarks](ane-quantization-fusion-benchmark.md), and
  [historical ANE measurements](ane-hadamard256-benchmark.md).
- Tile selection and split-K selection are unchanged.
- CPU and MPS floating-point dequantization accept the full flagged format and
  recover logical weights: `W_decoded = dequant(payload) R_K^T`. H256 is
  symmetric and self-inverse. The CPU reuses a 256-value local block; the Metal
  decoder transforms decoded integers in SIMD registers and folds `1/16` into
  the row scale before its only output rounding. Metal uses one dispatch and
  writes directly into the dense buffer already required by fallback, with no
  additional device scratch or activation transform.
- Dense fallback has no 65536 row-length limit. Rotation follows the stored
  last dimension, including when GEMM uses untransposed weights. Transposed
  activations, batched weights and bias follow ordinary dense GEMM semantics.
  MPSGraph handles BF16 and batched bias when falling back beyond MFA.
- Passing just the base codec to dequantization still decodes stored **rotated**
  coefficients. Packed-to-INT8 decoders retain this behavior for ANE/NA, whose
  activation paths perform the matching rotation. Other operators and CUDA
  have not gained H256 support from this fallback change.

Tests compare all ten codecs and all four source precisions byte-for-byte against
an independent dense rotation followed by ordinary quantization; cover invalid
K, output preservation on failure, and SQLite metadata/payload round trips.
MPS tests compare against independent dense activation rotation, runtime INT8
quantization and CPU GEMM, using IQ2_XXS/Q6_K, small M, dynamic M, batching, bias,
and N=73. The activation probe also covers FP16/BF16/FP32, cache separation,
strides and K up to 65536.

Fallback tests cover independent dense inverse rotation for all ten codecs and
FP16/BF16/FP32/FP64 CPU decode, CPU/Metal parity across codecs and changing cached
shapes, and ordinary MFA plus MPS GEMM with GEMV-sized inputs, both transpose
orientations, batched weights, bias, and K=65792. These paths retain the offline
weight-quantization benefit; they do not dynamically quantize activations.

Current ANE validation is recorded in the
[optimization follow-up](ane-hadamard256-optimization.md#validation-and-reproduction).
Earlier format validation also passed `palettize.tests` (12 passed, 29
backend-dependent skips). A separately compiled `NDEBUG` quantizer rejected
K=384 without writing output.
