#include "SegmentedInt8SwiGLUKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../../ccv_nnc_8i_rowwise_packed_grids.inc"

namespace {

static uint32_t compactIQ2GridEntry(const uint64_t value)
{
  uint32_t packed = 0;
  for (uint32_t lane = 0; lane < 8; ++lane) {
    const uint32_t v = (uint32_t)((value >> (lane * 8)) & 0xff);
    const uint32_t code = v == 8 ? 0 : (v == 25 ? 1 : 2);
    packed |= code << (lane * 2);
  }
  return packed;
}

static uint32_t byteIQ3XXSGridEntry(const uint32_t value)
{
  uint32_t packed = 0;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    const uint32_t v = (value >> (lane * 8)) & 0xff;
    packed |= (v >> 2) << (lane * 8);
  }
  return packed;
}

static void appendIQ2XSGrid(std::string& shader)
{
  shader += "constant uint iq2xs_grid[1024] = {";
  for (uint32_t index = 0; index < 512; ++index) {
    const uint32_t entry = compactIQ2GridEntry(
      ccv_nnc_8i_rowwise_packed_iq2xs_grid[index]);
    for (uint32_t block = 0; block < 2; ++block) {
      if (index != 0 || block != 0)
        shader += ",";
      if (((index * 2 + block) % 8) == 0)
        shader += "\n  ";
      uint32_t packed = 0;
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t code = (entry >> ((block * 4 + lane) * 2)) & 3;
        packed |= (1 + code * 2) << (lane * 8);
      }
      shader += std::to_string(packed) + "u";
    }
  }
  shader += "\n};\n";
}

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

static void appendIQ3XXSScaledGrid(std::string& shader)
{
  shader += "constant uint iq3xxs_scaled_grid[4096] = {";
  for (uint32_t scaleCode = 0; scaleCode < 16; ++scaleCode) {
    for (uint32_t index = 0; index < 256; ++index) {
      if (scaleCode != 0 || index != 0)
        shader += ",";
      if (((scaleCode * 256 + index) % 8) == 0)
        shader += "\n  ";
      const uint32_t entry = byteIQ3XXSGridEntry(
        ccv_nnc_8i_rowwise_packed_iq3xxs_grid[index]);
      uint32_t packed = 0;
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t value = (entry >> (lane * 8)) & 255;
        const uint32_t raw = value * (scaleCode + 1);
        packed |= (raw < 127 ? raw : 127) << (lane * 8);
      }
      shader += std::to_string(packed) + "u";
    }
  }
  shader += "\n};\n";
}

static void appendIQ2XSDecoder(std::string& shader)
{
  appendIQ2XSGrid(shader);
  shader += R"(
constant int iq2xs_scales[16] = {
  1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32
};

inline ulong iq2xs_payload_pair(
  device const uchar* data, const ulong byte_offset, const uint shift)
{
  const ulong value =
    (ulong)data[byte_offset] |
    ((ulong)data[byte_offset + 1] << 8) |
    ((ulong)data[byte_offset + 2] << 16) |
    ((ulong)data[byte_offset + 3] << 24) |
    ((ulong)data[byte_offset + 4] << 32) |
    ((ulong)data[byte_offset + 5] << 40);
  return value >> shift;
}

inline float4 signed_iq2xs_values(
  const uint packed, const uint signs,
  const uint sign_lane, const float scale)
{
  const float4 mag = min(float4(
    (float)(packed & 255u),
    (float)((packed >> 8) & 255u),
    (float)((packed >> 16) & 255u),
    (float)((packed >> 24) & 255u)) * scale, float4(127.0f));
  const uint sign_bits = signs >> sign_lane;
  return select(mag, -mag, bool4(
    (sign_bits & 1u) != 0u,
    (sign_bits & 2u) != 0u,
    (sign_bits & 4u) != 0u,
    (sign_bits & 8u) != 0u));
}

inline float iq2xs_dot(
  const uint payload, const float4 y0, const float4 y1)
{
  const uint grid = (payload & 511u) << 1;
  const uint signs = (payload >> 9) & 255u;
  const float scale = (float)iq2xs_scales[(payload >> 17) & 15u];
  const float4 v0 = signed_iq2xs_values(iq2xs_grid[grid], signs, 0, scale);
  const float4 v1 = signed_iq2xs_values(iq2xs_grid[grid + 1], signs, 4, scale);
  return dot(v0, y0) + dot(v1, y1);
}

inline float2 dot_pair(
  device const uchar* gate_weights, device const uchar* up_weights,
  const ulong bit_offset,
  device const real4* activation, const uint activation_base)
{
  const ulong byte_offset = bit_offset >> 3;
  const uint shift = (uint)(bit_offset & 7ul);
  const float4 y0 = float4(activation[activation_base + 0]);
  const float4 y1 = float4(activation[activation_base + 1]);
  const float4 y2 = float4(activation[activation_base + 2]);
  const float4 y3 = float4(activation[activation_base + 3]);
  const ulong gate_payloads = iq2xs_payload_pair(
    gate_weights, byte_offset, shift);
  const ulong up_payloads = iq2xs_payload_pair(
    up_weights, byte_offset, shift);
  const uint gate_payload0 = (uint)gate_payloads & 0x1fffffu;
  const uint gate_payload1 = (uint)(gate_payloads >> 21);
  const uint up_payload0 = (uint)up_payloads & 0x1fffffu;
  const uint up_payload1 = (uint)(up_payloads >> 21);
  return float2(
    iq2xs_dot(gate_payload0, y0, y1) +
      iq2xs_dot(gate_payload1, y2, y3),
    iq2xs_dot(up_payload0, y0, y1) +
      iq2xs_dot(up_payload1, y2, y3));
}
)";
}

static void appendIQ3XXSDecoder(std::string& shader)
{
  appendIQ3XXSScaledGrid(shader);
  shader += R"(
inline ulong iq3xxs_payload_pair(
  device const uchar* data, const ulong byte_offset)
{
  return
    (ulong)data[byte_offset] |
    ((ulong)data[byte_offset + 1] << 8) |
    ((ulong)data[byte_offset + 2] << 16) |
    ((ulong)data[byte_offset + 3] << 24) |
    ((ulong)data[byte_offset + 4] << 32) |
    ((ulong)data[byte_offset + 5] << 40) |
    ((ulong)data[byte_offset + 6] << 48);
}

inline float4 signed_iq3xxs_values(
  const uint index, const uint signs, const uint sign_lane)
{
  constant uchar* values = (constant uchar*)(iq3xxs_scaled_grid + index);
  const float4 mag = float4(
    (float)values[0],
    (float)values[1],
    (float)values[2],
    (float)values[3]);
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}

inline float iq3xxs_dot(
  const uint payload, const float4 y0, const float4 y1)
{
  const uint grid0 = payload & 255u;
  const uint grid1 = (payload >> 8) & 255u;
  const uint signs = (payload >> 16) & 255u;
  const uint scale_base = ((payload >> 24) & 15u) << 8;
  const float4 v0 = signed_iq3xxs_values(scale_base + grid0, signs, 0);
  const float4 v1 = signed_iq3xxs_values(scale_base + grid1, signs, 4);
  return dot(v0, y0) + dot(v1, y1);
}

inline float2 dot_pair(
  device const uchar* gate_weights, device const uchar* up_weights,
  const ulong bit_offset,
  device const real4* activation, const uint activation_base)
{
  const ulong byte_offset = bit_offset >> 3;
  const float4 y0 = float4(activation[activation_base + 0]);
  const float4 y1 = float4(activation[activation_base + 1]);
  const float4 y2 = float4(activation[activation_base + 2]);
  const float4 y3 = float4(activation[activation_base + 3]);
  const ulong gate_payloads = iq3xxs_payload_pair(gate_weights, byte_offset);
  const ulong up_payloads = iq3xxs_payload_pair(up_weights, byte_offset);
  const uint gate_payload0 = (uint)gate_payloads & 0xfffffffu;
  const uint gate_payload1 = (uint)(gate_payloads >> 28);
  const uint up_payload0 = (uint)up_payloads & 0xfffffffu;
  const uint up_payload1 = (uint)(up_payloads >> 28);
  return float2(
    iq3xxs_dot(gate_payload0, y0, y1) +
      iq3xxs_dot(gate_payload1, y2, y3),
    iq3xxs_dot(up_payload0, y0, y1) +
      iq3xxs_dot(up_payload1, y2, y3));
}
)";
}

static void appendQ2KDecoder(std::string& shader)
{
  shader += R"(
inline float vec_sum(const float4 value)
{
  return value.x + value.y + value.z + value.w;
}

inline ulong q2_payload(
  device const uchar* data, const ulong byte_offset, const uint shift)
{
  const ulong value =
    (ulong)data[byte_offset] |
    ((ulong)data[byte_offset + 1] << 8) |
    ((ulong)data[byte_offset + 2] << 16) |
    ((ulong)data[byte_offset + 3] << 24) |
    ((ulong)data[byte_offset + 4] << 32) |
    ((ulong)data[byte_offset + 5] << 40);
  return value >> shift;
}

inline float4 q2_values(const uint qbits, const uint offset)
{
  const uint q = (qbits >> offset) & 255u;
  return float4(
    (float)(q & 3u),
    (float)((q >> 2) & 3u),
    (float)((q >> 4) & 3u),
    (float)((q >> 6) & 3u));
}

inline float q2_dot(
  const ulong payload,
  const float4 y0, const float4 y1,
  const float4 y2, const float4 y3, const float ysum)
{
  const uint qbits = (uint)payload;
  const uint multiplier = (uint)((payload >> 32) & 63u) + 1u;
  const uint zero = (uint)((payload >> 38) & 15u) << 3;
  return (float)multiplier * (
    dot(q2_values(qbits, 0), y0) + dot(q2_values(qbits, 8), y1) +
    dot(q2_values(qbits, 16), y2) + dot(q2_values(qbits, 24), y3)) -
    (float)zero * ysum;
}

inline float2 dot_pair(
  device const uchar* gate_weights, device const uchar* up_weights,
  const ulong bit_offset,
  device const real4* activation, const uint activation_base)
{
  const ulong byte_offset = bit_offset >> 3;
  const uint shift = (uint)(bit_offset & 7ul);
  const float4 y0 = float4(activation[activation_base + 0]);
  const float4 y1 = float4(activation[activation_base + 1]);
  const float4 y2 = float4(activation[activation_base + 2]);
  const float4 y3 = float4(activation[activation_base + 3]);
  const float ysum = vec_sum(y0) + vec_sum(y1) + vec_sum(y2) + vec_sum(y3);
  const ulong gate_payload = q2_payload(gate_weights, byte_offset, shift);
  const ulong up_payload = q2_payload(up_weights, byte_offset, shift);
  return float2(
    q2_dot(gate_payload, y0, y1, y2, y3, ysum),
    q2_dot(up_payload, y0, y1, y2, y3, ysum));
}
)";
}

static std::string buildIQ2XSShader(
  const SegmentedInt8SwiGLUKernelDescriptor& descriptor)
{
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
  if (descriptor.clamp)
    shader += "constant float clamp_limit [[function_constant(8)]];\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;
)";
  appendIQ2XSDecoder(shader);
  shader += R"(

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
  uint activation_base = (lane * 2u * group_size) >> 2;
  ulong bit_offset =
    ((ulong)row_base * groups_per_row + lane * 2u) * 21ul;
  const uint activation_stride = (64u * group_size) >> 2;
  constexpr ulong bit_stride = 64ul * 21ul;
  for (uint group = lane * 2u; group < groups_per_row; group += 64u) {
    const float2 projected0 = dot_pair(
      gate_weights, up_weights, bit_offset, activation4, activation_base);
    gate_sum0 += projected0.x;
    up_sum0 += projected0.y;
    activation_base += activation_stride;
    bit_offset += bit_stride;
  }

  const float gate_row_sum0 = simd_sum(gate_sum0);
  const float up_row_sum0 = simd_sum(up_sum0);
  const float route_weight = (float)route_weights[route];
  if (lane == 0) {
    float gate0 = gate_row_sum0 * (float)gate_scales[row_base + 0];
    float up0 = up_row_sum0 * (float)up_scales[row_base + 0];
)";
  if (descriptor.clamp)
    shader += R"(
    gate0 = min(gate0, clamp_limit);
    up0 = clamp(up0, -clamp_limit, clamp_limit);
)";
  shader += R"(
    destination[row_base + 0] = (real)(
      route_weight * up0 * gate0 * stable_sigmoid(gate0));
  }
}
)";
  return shader;
}

static std::string buildIQ3XXSShader(
  const SegmentedInt8SwiGLUKernelDescriptor& descriptor)
{
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
  if (descriptor.clamp)
    shader += "constant float clamp_limit [[function_constant(8)]];\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;
)";
  appendIQ3XXSDecoder(shader);
  shader += R"(

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
  uint activation_base = (lane * 2u * group_size) >> 2;
  ulong bit_offset =
    ((ulong)row_base * groups_per_row + lane * 2u) * 28ul;
  const uint activation_stride = (64u * group_size) >> 2;
  constexpr ulong bit_stride = 64ul * 28ul;
  for (uint group = lane * 2u; group < groups_per_row; group += 64u) {
    const float2 projected0 = dot_pair(
      gate_weights, up_weights, bit_offset, activation4, activation_base);
    gate_sum0 += projected0.x;
    up_sum0 += projected0.y;
    activation_base += activation_stride;
    bit_offset += bit_stride;
  }

  const float gate_row_sum0 = simd_sum(gate_sum0);
  const float up_row_sum0 = simd_sum(up_sum0);
  const float route_weight = (float)route_weights[route];
  if (lane == 0) {
    float gate0 = gate_row_sum0 * (float)gate_scales[row_base + 0];
    float up0 = up_row_sum0 * (float)up_scales[row_base + 0];
)";
  if (descriptor.clamp)
    shader += R"(
    gate0 = min(gate0, clamp_limit);
    up0 = clamp(up0, -clamp_limit, clamp_limit);
)";
  shader += R"(
    destination[row_base + 0] = (real)(
      route_weight * up0 * gate0 * stable_sigmoid(gate0));
  }
}
)";
  return shader;
}

static std::string buildQ2KShader(
  const SegmentedInt8SwiGLUKernelDescriptor& descriptor)
{
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
  if (descriptor.clamp)
    shader += "constant float clamp_limit [[function_constant(8)]];\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;
)";
  appendQ2KDecoder(shader);
  shader += R"(

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
  uint activation_base = (lane * group_size) >> 2;
  ulong bit_offset = ((ulong)row_base * groups_per_row + lane) * 42ul;
  const uint activation_stride = (32u * group_size) >> 2;
  constexpr ulong bit_stride = 32ul * 42ul;
  for (uint group = lane; group < groups_per_row; group += 32u) {
    const float2 projected0 = dot_pair(
      gate_weights, up_weights, bit_offset, activation4, activation_base);
    gate_sum0 += projected0.x;
    up_sum0 += projected0.y;
    activation_base += activation_stride;
    bit_offset += bit_stride;
  }

  const float gate_row_sum0 = simd_sum(gate_sum0);
  const float up_row_sum0 = simd_sum(up_sum0);
  const float route_weight = (float)route_weights[route];
  if (lane == 0) {
    float gate0 = gate_row_sum0 * (float)gate_scales[row_base + 0];
    float up0 = up_row_sum0 * (float)up_scales[row_base + 0];
)";
  if (descriptor.clamp)
    shader += R"(
    gate0 = min(gate0, clamp_limit);
    up0 = clamp(up0, -clamp_limit, clamp_limit);
)";
  shader += R"(
    destination[row_base + 0] = (real)(
      route_weight * up0 * gate0 * stable_sigmoid(gate0));
  }
}
)";
  return shader;
}

}

SegmentedInt8SwiGLUKernel::SegmentedInt8SwiGLUKernel(
  SegmentedInt8SwiGLUKernelDescriptor descriptor,
  MTL::Device* const device)
{
  if (descriptor.format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
    source = buildIQ2XSShader(descriptor);
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return;
  }
  if (descriptor.format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
    source = buildIQ3XXSShader(descriptor);
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return;
  }
  if (descriptor.format == CCV_NNC_QX_8I_ROWWISE_Q2_K) {
    source = buildQ2KShader(descriptor);
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return;
  }
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
  if (descriptor.clamp)
    shader += "constant float clamp_limit [[function_constant(8)]];\n";
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
    float gate0 = gate_row_sum0 * (float)gate_scales[row_base + 0];
    float up0 = up_row_sum0 * (float)up_scales[row_base + 0];
)";
  if (descriptor.clamp)
    shader += R"(
    gate0 = min(gate0, clamp_limit);
    up0 = clamp(up0, -clamp_limit, clamp_limit);
)";
  shader += R"(
    destination[row_base + 0] = (real)(
      route_weight * up0 * gate0 * stable_sigmoid(gate0));
  }
}
)";

  source = std::move(shader);
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
