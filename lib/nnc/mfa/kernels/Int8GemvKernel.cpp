#include "Int8GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

Int8GemvKernel::Int8GemvKernel(Int8GemvKernelDescriptor descriptor, MTL::Device* const device) {
  fusedBias = descriptor.fusedBias;
  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

std::string Int8GemvKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float int8_gemv_dot(char4 q, float4 y) {
  return dot(float4(q), y);
}

kernel void int8_gemv(
  device const char *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],
)";
  if (fusedBias) {
    shader += R"(
  device const real *bias [[buffer(3)]],
)";
  }
  shader += R"(
  uint tgpig [[threadgroup_position_in_grid]],
  uint sgitg [[simdgroup_index_in_threadgroup]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  constexpr uint ROWS = )";
  shader += std::to_string(kInt8GemvRowsPerThreadgroup);
  shader += R"(;
  constexpr uint S = )";
  shader += std::to_string(kInt8GemvSIMDGroupsPerThreadgroup);
  shader += R"(;
  const uint rb = tgpig * ROWS;
  const bool active1 = rb + 1 < nrows;
  device const real4* y4 = (device const real4*)src1;
  threadgroup float partials[ROWS][32];
  device const real* scales = (device const real*)((device const uchar*)src0 + scale_offset);
  device const char4* x0 = (device const char4*)((device const char*)src0 + (rb + 0) * ncols);
  device const char4* x1 = (device const char4*)((device const char*)src0 + (rb + 1) * ncols);

  float sum0 = 0;
  float sum1 = 0;
  const uint nvecs = ncols / 4;
  const uint stride = S * 64;
  uint i = sgitg * 64 + tiisg * 2;
  for (; i + 1 < nvecs; i += stride) {
    const float4 yv0 = float4(y4[i]);
    const float4 yv1 = float4(y4[i + 1]);
    const char4 q0 = x0[i];
    const char4 q0b = x0[i + 1];
    sum0 += int8_gemv_dot(q0, yv0);
    sum0 += int8_gemv_dot(q0b, yv1);
    if (active1) {
      const char4 q1 = x1[i];
      const char4 q1b = x1[i + 1];
      sum1 += int8_gemv_dot(q1, yv0);
      sum1 += int8_gemv_dot(q1b, yv1);
    }
  }
  if (i < nvecs) {
    const float4 yv = float4(y4[i]);
    const char4 q0 = x0[i];
    sum0 += int8_gemv_dot(q0, yv);
    if (active1) {
      const char4 q1 = x1[i];
      sum1 += int8_gemv_dot(q1, yv);
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
  }
  const float lane_sum0 = simd_sum(sum0);
  const float lane_sum1 = simd_sum(sum1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum0;
    partials[1][sgitg] = lane_sum1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum0 = simd_sum(partials[0][tiisg]);
    const float all_sum1 = simd_sum(partials[1][tiisg]);
    if (tiisg == 0) {
      float value = all_sum0 * (float)scales[rb + 0];
)";
  if (fusedBias) {
    shader += R"(
      value += (float)bias[rb + 0];
)";
  }
  shader += R"(
      dst[rb + 0] = (real)value;
    }
    if (tiisg == 1 && active1) {
      float value = all_sum1 * (float)scales[rb + 1];
)";
  if (fusedBias) {
    shader += R"(
      value += (float)bias[rb + 1];
)";
  }
  shader += R"(
      dst[rb + 1] = (real)value;
    }
  }
}
  )";
  return shader;
}

std::string Int8GemvKernel::createConstants() const noexcept {
  std::string defines = "";
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += std::string("typedef float real;");
    defines += "\n";
    defines += std::string("typedef float4 real4;");
    defines += "\n";
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    defines += std::string("typedef bfloat real;");
    defines += "\n";
    defines += std::string("typedef bfloat4 real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
    defines += std::string("typedef half4 real4;");
    defines += "\n";
  }
  defines += "constant uint ncols [[function_constant(0)]];";
  defines += "\n";
  defines += "constant uint nrows [[function_constant(1)]];";
  defines += "\n";
  defines += "constant uint scale_offset [[function_constant(2)]];";
  defines += "\n";
  return defines;
}
