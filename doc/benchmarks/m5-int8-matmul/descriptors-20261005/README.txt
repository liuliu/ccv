NAInt8MatMul descriptor / kernel-descriptor separation, 2026-10-05

Baseline: f1628fe5628b884c81605b9fcf3729bbf31be225.
Branch: m5-na-int8-matmul.

report.txt: implementation, final validation and paired Max timings.
results.json: individual run medians, output hashes and thermal states.
core-count-estimates.txt: models, limitations and native/register distinction.
crossover-inputs.txt and crossover-estimates.json: inputs and calculations.

Reproduce the Max correctness checks:
  cd test/int/nnc
  make debug -j4
  ./mpsblas.tests 'rowwise int8 gemm'
  ./swiglu.tests 'MPS SwiGLU quantizes activation once for rowwise int8 prefill'
  ./segmented_swiglu.tests 'MPS segmented SwiGLU executes grouped rowwise int8 prefill'
  ./mpsblas.tests 'mps segmented gemm with row-wise 8i weight NA'
  ./mpsblas.tests 'mps segmented gemm with row-wise 8i weight loadM'
  ./mpsdnn.tests 'mps forward convolution in nchw format with row-wise 8i weight'
  cd ../../../bin/mfa
  make na_int8_convrot_bench dynamic_m_bench -j4
  ./na_int8_convrot_bench --validate-register
  ./dynamic_m_bench 1

--validate-register explicitly configures the low-level register kernel for
CPU-reference validation, including dynamic M even though production selection
now keeps dynamic M on native operands. Its descriptor-cache checks use separate
caches for device profiles 10, 35, 36, 40 and 80; these are selection tests, not
measurements of intermediate GPU SKUs.

Keep GPU tests sequential and cool before timings. The paired Max probe uses
ABBA order, 10-second idle gaps, at least 0.3 GPU-second warmup and 15 samples.
Its GPU times include activation quantization and GEMM. The 80-core policy
case still runs physically on the 40-core Max and does not simulate Ultra speed.

Temporary reproduction directory on this Max:
  /private/tmp/ccv-matmul-descriptors-20261005
It contains the baseline/current production probes, runners, full logs,
configuration comparison and iPad build scripts. iPad uses a standalone signed
MFA harness, not the full iOS integration-test suite.

The non-NA float Int8MatMul SwiGLU fallback hits an existing Metal
simdgroup_matrix_storage compilation failure. The same error is reproduced by
linking the test against the retained ece05fbc baseline library. This refactor
does not change that shader or attempt a precision/backward fix.
