NAInt8MatMul production cleanup and Ultra occupancy follow-up

Parent: ece05fbc (m5-na-int8-matmul). Read report.txt and results.json.
Scope: NAInt8MatMul forward only. Attention, backward, accumulation formats,
fused output casts, and large-row partitioning policy are unchanged.

Build and validate on M5:
  cd test/int/nnc
  make debug -j4
  ./mpsblas.tests 'rowwise int8 gemm'
  ./mpsdnn.tests
  cd ../../../bin/mfa
  make na_int8_convrot_bench na_int8_matmul_forward_bench
  ./na_int8_convrot_bench --validate-register

The register validation checks FP16/BF16/FP32 scalar IO, plain and H256
activation quantization, static/dynamic M, bias, partial tiles, inactive
output rows, and source-cache reuse against CPU references. This is direct
kernel validation; device policy remains in the production dispatcher.

Ultra measurements to repeat, including activation quantization:
  CCV_NA_WARMUP_SECONDS=0.5 ./na_int8_matmul_forward_bench 512 2432 9216 3 21 1
  CCV_NA_WARMUP_SECONDS=0.5 ./na_int8_matmul_forward_bench 512 2304 9216 3 21 1
  CCV_NA_WARMUP_SECONDS=0.5 ./na_int8_matmul_forward_bench 512 2559 9248 3 21 1
  CCV_NA_WARMUP_SECONDS=0.5 ./na_int8_matmul_forward_bench 512 2560 9248 3 21 1
Repeat adjacent widths 2431/2432/2433 at both K=9216 and K=9248, especially
newly selected 512x2431x9248, and the retained H3 FFN-down. The gate expansion
is not yet evidence of universal profitability for combined tails on Ultra.
Use alternating parent/candidate order, cooling, and multiple run medians.
A nominal thermal-state reading alone does not establish stable clocks.

The three *-diagnostic.patch files are UNRETAINED experiments for further
Ultra investigation, not production fixes. Apply one at a time to this
revision's NAInt8MatMulKernel.cpp; do not combine them. All passed four
Max smoke shapes with matching hashes, but none removed the K-tail cliff.
  fragment: specializes only K32 alignment, not the full reduction length.
  unroll: changes the remainder loop to two-way unrolling.
  separate: moves the final short fragment outside the full-fragment loop.

The local paired-IO probe and scripts are under
/private/tmp/ccv-matmul-production-20261004 on the development Mac.
Their 80-core override changes selection on the 40-core Max. It does not
simulate Ultra hardware; numerical/performance results are labelled as Max.
