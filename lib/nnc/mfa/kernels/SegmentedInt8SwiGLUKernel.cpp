#include "SegmentedInt8SwiGLUKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../../ccv_nnc_8i_rowwise_packed_grids.inc"

namespace {

static void appendIQ2XXSScaledGrid(std::string& shader)
{
  static constexpr int scales[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32,
  };
  shader += "constant uint iq2xxs_scaled_grid[8192] = {";
  for (uint32_t scaleCode = 0; scaleCode < 16; ++scaleCode) {
    for (uint32_t index = 0; index < 256; ++index) {
      const uint16_t entry = ccv_nnc_8i_rowwise_packed_iq2xxs_grid[index];
      for (uint32_t block = 0; block < 2; ++block) {
        if (scaleCode != 0 || index != 0 || block != 0)
          shader += ",";
        if (((scaleCode * 512 + index * 2 + block) % 8) == 0)
          shader += "\n  ";
        uint32_t packed = 0;
        for (uint32_t lane = 0; lane < 4; ++lane) {
          const uint32_t code = (entry >> ((block * 4 + lane) * 2)) & 3;
          const int raw = (1 + (int)code * 2) * scales[scaleCode];
          packed |= (uint32_t)(raw < 127 ? raw : 127) << (lane * 8);
        }
        shader += std::to_string(packed) + "u";
      }
    }
  }
  shader += "\n};\n";
}

static void appendIQ2XXSSigns(std::string& shader)
{
  shader += "constant uint iq2xxs_ksigns[128] = {";
  for (uint32_t i = 0; i < 128; ++i) {
    if (i != 0)
      shader += ",";
    if ((i % 16) == 0)
      shader += "\n  ";
    shader += std::to_string((uint32_t)ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[i]) + "u";
  }
  shader += "\n};\n";
}

}

SegmentedInt8SwiGLUKernel::SegmentedInt8SwiGLUKernel(
  SegmentedInt8SwiGLUKernelDescriptor descriptor,
  MTL::Device* const device)
{
  CCV_NNC_MFA_PRECONDITION(
    descriptor.format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
  std::string shader;
  if (descriptor.memoryPrecision == GEMMOperandPrecision::FP32)
    shader = "typedef float real; typedef float4 real4;\n";
  else if (descriptor.memoryPrecision == GEMMOperandPrecision::BF16)
    shader = "typedef bfloat real; typedef bfloat4 real4;\n";
  else
    shader = "typedef half real; typedef half4 real4;\n";
  shader += R"(
constant uint ncols [[function_constant(0)]];
constant uint nrows [[function_constant(1)]];
constant uint segment_count [[function_constant(2)]];
constant uint group_size [[function_constant(3)]];
constant uint groups_per_row [[function_constant(4)]];
constant uint expert_count [[function_constant(5)]];
constant uint broadcast_input [[function_constant(6)]];
constant ulong weight_expert_stride [[function_constant(7)]];
)";
  appendIQ2XXSScaledGrid(shader);
  appendIQ2XXSSigns(shader);
  shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float4 signed_iq2xxs_values(
  const uint packed, const uint signs, const uint sign_lane)
{
  const float4 mag = float4(
    (float)(packed & 255u),
    (float)((packed >> 8) & 255u),
    (float)((packed >> 16) & 255u),
    (float)((packed >> 24) & 255u));
  const uint sign_bits = signs >> sign_lane;
  return select(mag, -mag, bool4(
    (sign_bits & 1u) != 0u,
    (sign_bits & 2u) != 0u,
    (sign_bits & 4u) != 0u,
    (sign_bits & 8u) != 0u));
}

inline float iq2xxs_dot(
  device const uchar* source, const ulong group_index,
  const float4 y0, const float4 y1, const float4 y2, const float4 y3,
  const float4 y4, const float4 y5, const float4 y6, const float4 y7)
{
  device const uchar* p = source + group_index * 8ul;
  const uint sign_codes =
    (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) |
    (((uint)p[7] & 15u) << 24);
  const uint scale_base = ((uint)p[7] >> 4) << 9;
  const uint signs0 = iq2xxs_ksigns[sign_codes & 127u];
  const uint signs1 = iq2xxs_ksigns[(sign_codes >> 7) & 127u];
  const uint signs2 = iq2xxs_ksigns[(sign_codes >> 14) & 127u];
  const uint signs3 = iq2xxs_ksigns[(sign_codes >> 21) & 127u];
  const uint grid0 = scale_base + ((uint)p[0] << 1);
  const uint grid1 = scale_base + ((uint)p[1] << 1);
  const uint grid2 = scale_base + ((uint)p[2] << 1);
  const uint grid3 = scale_base + ((uint)p[3] << 1);
  const float4 v0 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid0], signs0, 0);
  const float4 v1 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid0 + 1], signs0, 4);
  const float4 v2 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid1], signs1, 0);
  const float4 v3 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid1 + 1], signs1, 4);
  const float4 v4 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid2], signs2, 0);
  const float4 v5 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid2 + 1], signs2, 4);
  const float4 v6 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid3], signs3, 0);
  const float4 v7 = signed_iq2xxs_values(iq2xxs_scaled_grid[grid3 + 1], signs3, 4);
  return dot(v0, y0) + dot(v1, y1) + dot(v2, y2) + dot(v3, y3) +
    dot(v4, y4) + dot(v5, y5) + dot(v6, y6) + dot(v7, y7);
}

inline float2 dot_pair(
  device const uchar* gate_weights, device const uchar* up_weights,
  const ulong group_index,
  device const real4* activation, const uint activation_base)
{
  const float4 y0 = float4(activation[activation_base + 0]);
  const float4 y1 = float4(activation[activation_base + 1]);
  const float4 y2 = float4(activation[activation_base + 2]);
  const float4 y3 = float4(activation[activation_base + 3]);
  const float4 y4 = float4(activation[activation_base + 4]);
  const float4 y5 = float4(activation[activation_base + 5]);
  const float4 y6 = float4(activation[activation_base + 6]);
  const float4 y7 = float4(activation[activation_base + 7]);
  return float2(
    iq2xxs_dot(
      gate_weights, group_index, y0, y1, y2, y3, y4, y5, y6, y7),
    iq2xxs_dot(
      up_weights, group_index, y0, y1, y2, y3, y4, y5, y6, y7));
}

inline float stable_sigmoid(const float value)
{
  const float tail = 1.0f / (1.0f + exp(abs(value)));
  return tail + step(0.0f, value) * (1.0f - 2.0f * tail);
}

kernel void segmented_int8_swiglu(
  device const uchar* gate_weights [[buffer(0)]],
  device const uchar* up_weights [[buffer(1)]],
  device const real* activation [[buffer(2)]],
  device real* destination [[buffer(3)]],
  device const real* gate_scales [[buffer(4)]],
  device const real* up_scales [[buffer(5)]],
  device const int* indices [[buffer(6)]],
  device const int* counts [[buffer(7)]],
  device const real* route_weights [[buffer(8)]],
  constant float& limit [[buffer(9)]],
  uint threadgroup_position [[threadgroup_position_in_grid]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]])
{
  constexpr uint rows_per_simdgroup = 1;
  constexpr uint row_simdgroups = 4;
  constexpr uint rows_per_threadgroup = rows_per_simdgroup * row_simdgroups;
  const uint threadgroups_per_route =
    (nrows + rows_per_threadgroup - 1u) / rows_per_threadgroup;
  const uint route = threadgroup_position / threadgroups_per_route;
  threadgroup int route_expert;
  if (simdgroup_index == 0 && lane == 0) {
    route_expert = -1;
    uint row_offset = 0;
    for (uint segment = 0; segment < segment_count; ++segment) {
      const int count = counts[segment];
      if (count <= 0)
        continue;
      const uint next_row_offset = row_offset + (uint)count;
      if (route < next_row_offset) {
        route_expert = indices[segment];
        break;
      }
      row_offset = next_row_offset;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int expert = route_expert;
  if (expert < 0 || expert >= (int)expert_count)
    return;

  gate_weights += (ulong)expert * weight_expert_stride;
  up_weights += (ulong)expert * weight_expert_stride;
  gate_scales += (ulong)expert * nrows;
  up_scales += (ulong)expert * nrows;
  activation += broadcast_input ? 0 : (ulong)route * ncols;
  destination += (ulong)route * nrows;
  const uint local_threadgroup =
    threadgroup_position - route * threadgroups_per_route;
  const uint row_simdgroup = simdgroup_index;
  const uint row_base =
    (local_threadgroup * row_simdgroups + row_simdgroup) *
    rows_per_simdgroup;
  device const real4* activation4 = (device const real4*)activation;

  float gate_sum0 = 0;
  float up_sum0 = 0;
  for (uint group = lane; group < groups_per_row; group += 32) {
    const uint activation_base = (group * group_size) >> 2;
    const float2 projected0 = dot_pair(
      gate_weights, up_weights,
      (ulong)(row_base + 0) * groups_per_row + group,
      activation4, activation_base);
    gate_sum0 += projected0.x;
    up_sum0 += projected0.y;
  }

  const float gate_row_sum0 = simd_sum(gate_sum0);
  const float up_row_sum0 = simd_sum(up_sum0);
  const float route_weight = (float)route_weights[route];
  if (lane == 0) {
    const float gate0 = gate_row_sum0 * (float)gate_scales[row_base + 0];
    const float up0 = up_row_sum0 * (float)up_scales[row_base + 0];
    const float clamped_gate = min(gate0, limit);
    const float clamped_up = clamp(up0, -limit, limit);
    destination[row_base + 0] = (real)(
      route_weight * clamped_up * clamped_gate * stable_sigmoid(clamped_gate));
  }
}
)";

  source = std::move(shader);
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
