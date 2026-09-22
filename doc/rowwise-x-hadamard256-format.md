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
  NAInt8MatMul activation staging, including its existing small-M path. Raw
  packed-to-INT8 decode receives only the base codec and preserves rotated values.
- Existing NA runtime limits still apply: K <= 65536, supported dtype/layout,
  NA hardware, and supported batching. Unsupported flagged GEMM returns
  `CCV_NNC_EXEC_INVALID`. ANE/scaled-GEMV/dense fallback are not silently used.
- Tile selection and split-K selection are unchanged.
- Other consumers have not gained rotation support. Existing MFA codec
  preconditions reject full flagged IDs rather than masking them globally.
- Logical dense-weight dequantization/fallback remains deferred. Explicitly
  passing just the base codec to dense dequantization
  decodes stored **rotated** coefficients; it does not recover logical weights.
  Logical recovery would require `dequant(payload) R_K^T`.

Tests compare all ten codecs and all four source precisions byte-for-byte against
an independent dense rotation followed by ordinary quantization; cover invalid
K, output preservation on failure, and SQLite metadata/payload round trips.
MPS tests compare against independent dense activation rotation, runtime INT8
quantization and CPU GEMM, using IQ2_XXS/Q6_K, small M, dynamic M, batching, bias,
and N=73. The activation probe also covers FP16/BF16/FP32, cache separation,
strides and K up to 65536.

Validation on this workspace: debug build passed; `palettize.tests` passed
(12 passed, 29 backend-dependent skips); `mpsblas.tests` passed 283/283;
`mpsdnn.tests` passed 126/126; the activation probe passed 21/21 cases.
A separately compiled `NDEBUG` quantizer rejected K=384 without writing output.
