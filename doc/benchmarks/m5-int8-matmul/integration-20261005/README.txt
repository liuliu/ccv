NAInt8MatMul native integration validation

report.txt explains the source changes, fragment traversal, validation and
performance limitations. results.json contains paired-run records and summaries.
geometry-diagnostic.patch flattens the threadgroup of the final kernel for a
controlled comparison. Apply it only for diagnosis and reverse it afterward.

Baseline: d8effcb2 on m5-na-int8-matmul. Measurements include activation quantization.

Reproduction from the repository root:
  cd test/int/nnc
  make debug -j4
  ./mpsblas.tests 'rowwise int8 gemm'
  ./mpsdnn.tests

Direct scalar/rotation/bias/runtime-M validation:
  cd bin/mfa
  make na_int8_convrot_bench -j4
  ./na_int8_convrot_bench --validate-register

Production FP16 timing:
  cd bin/mfa
  make na_int8_matmul_forward_bench -j4
  CCV_NA_WARMUP_SECONDS=0.3 ./na_int8_matmul_forward_bench 512 2559 9248 3 15 1

The final argument selects dynamic M. Run old/new binaries sequentially in
ABBA order, with 10 seconds idle between Max runs; require nominal thermal state.
The 80-core override in the retained probe only exercises the Ultra selector
on Max hardware. It does not reproduce Ultra throughput or cache behavior.

Full local artifacts:
  /private/tmp/ccv-matmul-integration-20261005/
    selector-check.{py,cpp,log}  actual predicate differential comparison
    direct-final-geometry.log           CPU reference and dynamic pipeline reuse checks
    blas-final-geometry.log, dnn.log  integration test logs
    final-geometry/     final paired Max logs and metadata
    io-probe.cpp, paired-final-geometry.py    typed production probe and cooled run driver
    ipad/                     Xcode harness, incremental build and device logs

The temporary iPad app links the changed translation units with unchanged
objects from the d8effcb2 validation build. Both use the same probe wrapper.
The rewritten register path is forced only in the direct correctness validator;
production timing uses the real iPad 10-core policy.
