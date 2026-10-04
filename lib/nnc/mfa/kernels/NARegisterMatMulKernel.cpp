#include "NARegisterMatMulKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

NARegisterMatMulKernel::NARegisterMatMulKernel(
    NARegisterMatMulKernelDescriptor descriptor, MTL::Device* device) {
  CCV_NNC_MFA_PRECONDITION(!descriptor.castOutputToFloat || descriptor.quantized);
  std::string source =
#include "../3rdparty/mlx/gemm.metal.inc"
  ;
  if (descriptor.quantized) {
    // Expose two successive integer fragment loads / multiplies to the
    // compiler. This improves long reductions without changing the outer
    // synchronization or the exact INT32 sum. Keep the imported FP16 source
    // and its loop policy unchanged.
    const std::string reductionLoop =
        "STEEL_PRAGMA_NO_UNROLL\n    for (int kk1 = 0; kk1 < BK; kk1 += SK)";
    const auto reductionLoopOffset = source.find(reductionLoop);
    CCV_NNC_MFA_PRECONDITION(reductionLoopOffset != std::string::npos);
    source.replace(reductionLoopOffset, reductionLoop.size(),
        "#pragma clang loop unroll_count(2)\n    for (int kk1 = 0; kk1 < BK; kk1 += SK)");
    // A partial outer reduction can still contain complete K32 fragments.
    // Load those normally, retaining bounds checks for partial M/N tiles and
    // the final short K fragment. This avoids masked scalar loads throughout
    // the remainder without adding a reduction-size selection rule.
    const std::string remainderLoads =
        "      Atile.load_safe(A + A_offset, lda, Aklims);\n      Btile.load_safe(B + B_offset, ldb, Bklims);";
    const auto remainderLoadsOffset = source.find(remainderLoads);
    CCV_NNC_MFA_PRECONDITION(remainderLoadsOffset != std::string::npos);
    source.replace(remainderLoadsOffset, remainderLoads.size(), R"(
      if (psk >= SK) {
        if constexpr (kAlignedM)
          Atile.load(A + A_offset, lda);
        else
          Atile.load_safe(A + A_offset, lda, Aklims);
        if constexpr (kAlignedN)
          Btile.load(B + B_offset, ldb);
        else
          Btile.load_safe(B + B_offset, ldb, Bklims);
      } else {
        Atile.load_safe(A + A_offset, lda, Aklims);
        Btile.load_safe(B + B_offset, ldb, Bklims);
      }
)");
    // Reuse MLX's fragment loads and reduction traversal with integer operands.
    // Apply the existing rowwise scales only after the exact INT32 reduction.
    source += R"(
[[kernel, max_total_threads_per_threadgroup(256)]] void matmul_register(
    const device int8_t* A [[buffer(0)]],
    const device int8_t* B [[buffer(1)]],
    const device half* bias [[buffer(2), function_constant(use_out_source)]],
    device )";
    source += descriptor.castOutputToFloat ? "float" : "half";
    source += R"(* D [[buffer(3)]],
    const constant GEMMParams& params [[buffer(4)]],
    const device half* a_scale [[buffer(8)]],
    const device half* b_scale [[buffer(9)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = 32, SN = 32, TM = 2, TN = 2;
  const int row = int(tid.y) * 64 + (sg / 4) * SM;
  const int col = int(tid.x) * 128 + (sg % 4) * SN;
  const short rows = min(int(SM), params.M - row);
  const short cols = min(int(SN), params.N - col);
  A += size_t(row) * params.K;
  B += size_t(col) * params.K;
  D += size_t(row) * params.N + col;
  dispatch_bool(align_K, [&](auto ak) {
    dispatch_bool(align_M || rows == SM, [&](auto am) {
      dispatch_bool(align_N || cols == SN, [&](auto an) {
        auto accum = gemm_loop<int8_t, SM, SN, 32, 512, false, true,
            am.value, an.value, ak.value, int32_t>(
                A, B, params.K, params.K, params.K,
                params.gemm_k_iterations_aligned, rows, cols);
        if (rows <= 0 || cols <= 0)
          return;
        NAXTile<half, TM, TN> output;
        const_for_loop<0, TM, 1>([&](auto mm) {
          const_for_loop<0, TN, 1>([&](auto nn) {
            thread auto& dst = output.template frag_at<mm, nn>();
            thread auto& src = accum.template frag_at<mm, nn>();
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 8; ++i) {
              const short2 xy = BaseNAXFrag::get_coord(i);
              const int r = mm * 16 + xy.y, c = nn * 16 + xy.x;
              const float a = r < rows ? float(a_scale[row + r]) : 0.0f;
              const float b = c < cols ? float(b_scale[col + c]) : 0.0f;
              float value = float(src[i]) * a * b;
              if (use_out_source && c < cols)
                value += float(bias[col + c]);
              dst[i] = half(value);
            }
          });
        });
        if constexpr (am.value && an.value)
          output.store(D, int(params.N));
        else
          output.store_safe(D, int(params.N), short2(cols, rows));
      });
    });
  });
}
)";
  } else if (descriptor.splitK) {
    source +=
#include "../3rdparty/mlx/gemm-splitk.metal.inc"
    ;
    source += "\ntemplate [[host_name(\"matmul_register\")]] kernel "
        "decltype(gemm_splitk_nax<half,128,64,512,4,2,false,true,float>) "
        "gemm_splitk_nax<half,128,64,512,4,2,false,true,float>;\n";
    source += R"(
kernel void matmul_register_reduce(
    const device float* partials [[buffer(0)]],
    device half* output [[buffer(1)]],
    const constant GEMMSpiltKParams& params [[buffer(2)]],
    const device half* bias [[buffer(3), function_constant(use_out_source)]],
    uint2 gid [[thread_position_in_grid]]) {
  const size_t offset = size_t(gid.y) * params.N + gid.x;
  float value = 0;
  for (int i = 0; i < params.split_k_partitions; ++i)
    value += partials[offset + size_t(i) * params.split_k_partition_stride];
  if (use_out_source)
    value += float(bias[gid.x]);
  output[offset] = half(value);
}
)";
  } else if (descriptor.wideM) {
    source += "\ntemplate [[host_name(\"matmul_register\")]] kernel "
        "decltype(gemm<half,128,64,512,4,2,false,true,float>) "
        "gemm<half,128,64,512,4,2,false,true,float>;\n";
  } else {
    source += "\ntemplate [[host_name(\"matmul_register\")]] kernel "
        "decltype(gemm<half,64,128,512,2,4,false,true,float>) "
        "gemm<half,64,128,512,2,4,false,true,float>;\n";
  }
  auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  options->setFastMathEnabled(false);
  NS::Error* error = nullptr;
  library = NS::TransferPtr(device->newLibrary(
      NS::String::string(source.c_str(), NS::UTF8StringEncoding), options.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
