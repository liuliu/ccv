#include "Int8SwiGLUKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

Int8SwiGLUKernel::Int8SwiGLUKernel(
  Int8SwiGLUKernelDescriptor descriptor,
  MTL::Device* const device)
{
  if (descriptor.memoryPrecision == GEMMOperandPrecision::FP32)
    source = "typedef float real; typedef float4 real4;\n";
  else if (descriptor.memoryPrecision == GEMMOperandPrecision::BF16)
    source = "typedef bfloat real; typedef bfloat4 real4;\n";
  else
    source = "typedef half real; typedef half4 real4;\n";
  source += R"(
constant uint ncols [[function_constant(0)]];
constant uint nrows [[function_constant(1)]];
)";
  if (descriptor.clamp)
    source += "constant float clamp_limit [[function_constant(2)]];\n";
  source += R"(
#include <metal_stdlib>
using namespace metal;

inline float stable_sigmoid(const float value)
{
  const float tail = 1.0f / (1.0f + exp(abs(value)));
  return tail + step(0.0f, value) * (1.0f - 2.0f * tail);
}

kernel void int8_swiglu(
  device const char* gate_weights [[buffer(0)]],
  device const char* up_weights [[buffer(1)]],
  device const real* activation [[buffer(2)]],
  device real* destination [[buffer(3)]],
  device const real* gate_scales [[buffer(4)]],
  device const real* up_scales [[buffer(5)]],
  uint threadgroup_position [[threadgroup_position_in_grid]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]])
{
  constexpr uint rows_per_threadgroup = )";
  source += std::to_string(kInt8SwiGLURowsPerThreadgroup);
  source += R"(;
  constexpr uint simdgroups_per_threadgroup = )";
  source += std::to_string(kInt8SwiGLUSIMDGroupsPerThreadgroup);
  source += R"(;
  const uint row_base = threadgroup_position * rows_per_threadgroup;
  const bool active1 = row_base + 1 < nrows;
  device const real4* activation4 = (device const real4*)activation;
  device const char4* gate0 =
    (device const char4*)(gate_weights + (ulong)(row_base + 0) * ncols);
  device const char4* gate1 =
    (device const char4*)(gate_weights + (ulong)(row_base + 1) * ncols);
  device const char4* up0 =
    (device const char4*)(up_weights + (ulong)(row_base + 0) * ncols);
  device const char4* up1 =
    (device const char4*)(up_weights + (ulong)(row_base + 1) * ncols);
  threadgroup float partials[4][32];

  float gate_sum0 = 0;
  float gate_sum1 = 0;
  float up_sum0 = 0;
  float up_sum1 = 0;
  const uint vectors = ncols / 4;
  const uint stride = simdgroups_per_threadgroup * 64;
  uint i = simdgroup_index * 64 + lane * 2;
  for (; i + 1 < vectors; i += stride) {
    const float4 a0 = float4(activation4[i]);
    const float4 a1 = float4(activation4[i + 1]);
    gate_sum0 += dot(float4(gate0[i]), a0);
    gate_sum0 += dot(float4(gate0[i + 1]), a1);
    up_sum0 += dot(float4(up0[i]), a0);
    up_sum0 += dot(float4(up0[i + 1]), a1);
    if (active1) {
      gate_sum1 += dot(float4(gate1[i]), a0);
      gate_sum1 += dot(float4(gate1[i + 1]), a1);
      up_sum1 += dot(float4(up1[i]), a0);
      up_sum1 += dot(float4(up1[i + 1]), a1);
    }
  }
  if (i < vectors) {
    const float4 a0 = float4(activation4[i]);
    gate_sum0 += dot(float4(gate0[i]), a0);
    up_sum0 += dot(float4(up0[i]), a0);
    if (active1) {
      gate_sum1 += dot(float4(gate1[i]), a0);
      up_sum1 += dot(float4(up1[i]), a0);
    }
  }

  if (simdgroup_index == 0) {
    partials[0][lane] = 0;
    partials[1][lane] = 0;
    partials[2][lane] = 0;
    partials[3][lane] = 0;
  }
  const float lane_gate0 = simd_sum(gate_sum0);
  const float lane_gate1 = simd_sum(gate_sum1);
  const float lane_up0 = simd_sum(up_sum0);
  const float lane_up1 = simd_sum(up_sum1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    partials[0][simdgroup_index] = lane_gate0;
    partials[1][simdgroup_index] = lane_gate1;
    partials[2][simdgroup_index] = lane_up0;
    partials[3][simdgroup_index] = lane_up1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index == 0) {
    const float all_gate0 = simd_sum(partials[0][lane]);
    const float all_gate1 = simd_sum(partials[1][lane]);
    const float all_up0 = simd_sum(partials[2][lane]);
    const float all_up1 = simd_sum(partials[3][lane]);
    if (lane == 0) {
      float gate = all_gate0 * (float)gate_scales[row_base + 0];
      float up = all_up0 * (float)up_scales[row_base + 0];
)";
  if (descriptor.clamp)
    source += R"(
      gate = min(gate, clamp_limit);
      up = clamp(up, -clamp_limit, clamp_limit);
)";
  source += R"(
      destination[row_base + 0] = (real)(up * gate * stable_sigmoid(gate));
    }
    if (lane == 1 && active1) {
      float gate = all_gate1 * (float)gate_scales[row_base + 1];
      float up = all_up1 * (float)up_scales[row_base + 1];
)";
  if (descriptor.clamp)
    source += R"(
      gate = min(gate, clamp_limit);
      up = clamp(up, -clamp_limit, clamp_limit);
)";
  source += R"(
      destination[row_base + 1] = (real)(up * gate * stable_sigmoid(gate));
    }
  }
}
)";

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
