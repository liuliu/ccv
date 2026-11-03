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

)";
  if (descriptor.format != 0) {
    // Q5_K and Q6_K weights: half a SIMD group owns each weight row, and its lanes decode 16
    // gate and 16 up weights at a time, then take them against every activation row. As in the
    // 4 to 8 row Int8GemvKernel, this keeps the lanes busy on short rows.
    if (descriptor.format == CCV_NNC_QX_8I_ROWWISE_Q5_K) {
      source += R"(
// One 11-byte group: sixteen 5-bit values, a 3-bit multiplier and a 5-bit offset.
constant uint unit_bytes = 11;

inline void decode16(device const uchar* p, thread float4& w0, thread float4& w1, thread float4& w2, thread float4& w3)
{
  const uint x = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
  const uint y = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24);
  const uint z = (uint)p[8] | ((uint)p[9] << 8) | ((uint)p[10] << 16);
  const float m = (float)(((z >> 16) & 7u) + 1);
  const float4 o = float4((float)((int)((z >> 19) & 31u) - 16) - 16.0f * m);
  w0 = fma(float4(m), float4(uint4(x, x >> 5, x >> 10, x >> 15) & 31u), o);
  w1 = fma(float4(m), float4(uint4(x >> 20, x >> 25, (x >> 30) | (y << 2), y >> 3) & 31u), o);
  w2 = fma(float4(m), float4(uint4(y >> 8, y >> 13, y >> 18, y >> 23) & 31u), o);
  w3 = fma(float4(m), float4(uint4((y >> 28) | (z << 4), z >> 1, z >> 6, z >> 11) & 31u), o);
}
)";
    } else {
      CCV_NNC_MFA_PRECONDITION(descriptor.format == CCV_NNC_QX_8I_ROWWISE_Q6_K);
      source += R"(
// Two 52-bit groups in 13 bytes: eight signed 6-bit values, a 2-bit multiplier and a 2-bit
// offset each.
constant uint unit_bytes = 13;

inline float q6_offset(const uint meta)
{
  const uint q = (meta >> 2) & 3u;
  return (q & 2u) ? (float)((int)q - 4) : (float)q;
}

inline void decode16(device const uchar* p, thread float4& w0, thread float4& w1, thread float4& w2, thread float4& w3)
{
  const uint a0 = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
  const uint b0 = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16);
  const uint a1 = ((uint)p[6] >> 4) | ((uint)p[7] << 4) | ((uint)p[8] << 12) | ((uint)p[9] << 20) | ((uint)p[10] << 28);
  const uint b1 = ((uint)p[10] >> 4) | ((uint)p[11] << 4) | ((uint)p[12] << 12);
  const uint meta0 = (b0 >> 16) & 15u;
  const uint meta1 = (b1 >> 16) & 15u;
  const float4 m0 = float4((float)((meta0 & 3u) + 1));
  const float4 m1 = float4((float)((meta1 & 3u) + 1));
  const float4 o0 = float4(q6_offset(meta0));
  const float4 o1 = float4(q6_offset(meta1));
  w0 = fma(float4(int4(uint4(a0 << 26, a0 << 20, a0 << 14, a0 << 8)) >> 26), m0, o0);
  w1 = fma(float4(int4(uint4(a0 << 2, ((a0 >> 30) | (b0 << 2)) << 26, b0 << 22, b0 << 16)) >> 26), m0, o0);
  w2 = fma(float4(int4(uint4(a1 << 26, a1 << 20, a1 << 14, a1 << 8)) >> 26), m1, o1);
  w3 = fma(float4(int4(uint4(a1 << 2, ((a1 >> 30) | (b1 << 2)) << 26, b1 << 22, b1 << 16)) >> 26), m1, o1);
}
)";
    }
    source += R"(
kernel void int8_swiglu(
  device const uchar* gate_weights [[buffer(0)]],
  device const uchar* up_weights [[buffer(1)]],
  device const real* activation [[buffer(2)]],
  device real* destination [[buffer(3)]],
  device const real* gate_scales [[buffer(4)]],
  device const real* up_scales [[buffer(5)]],
  uint threadgroup_position [[threadgroup_position_in_grid]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]])
{
  constexpr uint rows_per_threadgroup = )";
    source += std::to_string(kInt8SwiGLUPackedRowsPerThreadgroup);
    source += R"(;
  constexpr uint activation_rows = )";
    source += std::to_string(descriptor.activationRows);
    source += R"(;
  // K is a multiple of 256, so rows start on byte boundaries. A row past the end reads the last
  // row and writes nothing.
  const uint units_per_row = ncols / 16;
  const uint row = threadgroup_position * rows_per_threadgroup + simdgroup_index * 2 + (lane >> 4);
  const uint unit_lane = lane & 15u;
  const ulong row_offset = (ulong)min(row, nrows - 1) * (units_per_row * unit_bytes);
  device const uchar* gate = gate_weights + row_offset;
  device const uchar* up = up_weights + row_offset;

  float gate_sum[activation_rows];
  float up_sum[activation_rows];
  for (uint r = 0; r < activation_rows; r++) {
    gate_sum[r] = 0;
    up_sum[r] = 0;
  }
  for (uint u = unit_lane; u < units_per_row; u += 16) {
    float4 g0, g1, g2, g3;
    float4 u0, u1, u2, u3;
    decode16(gate + u * unit_bytes, g0, g1, g2, g3);
    decode16(up + u * unit_bytes, u0, u1, u2, u3);
    for (uint r = 0; r < activation_rows; r++) {
      device const real4* y = (device const real4*)(activation + (ulong)r * ncols) + u * 4;
      const float4 y0 = float4(y[0]);
      const float4 y1 = float4(y[1]);
      const float4 y2 = float4(y[2]);
      const float4 y3 = float4(y[3]);
      gate_sum[r] += dot(g0, y0) + dot(g1, y1) + dot(g2, y2) + dot(g3, y3);
      up_sum[r] += dot(u0, y0) + dot(u1, y1) + dot(u2, y2) + dot(u3, y3);
    }
  }

  for (uint r = 0; r < activation_rows; r++) {
    for (uint offset = 8; offset > 0; offset >>= 1) {
      gate_sum[r] += simd_shuffle_xor(gate_sum[r], offset);
      up_sum[r] += simd_shuffle_xor(up_sum[r], offset);
    }
  }
  if (unit_lane == 0 && row < nrows) {
    const float gate_scale = (float)gate_scales[row];
    const float up_scale = (float)up_scales[row];
    for (uint r = 0; r < activation_rows; r++) {
      float gate = gate_sum[r] * gate_scale;
      float up = up_sum[r] * up_scale;
)";
    if (descriptor.clamp)
      source += R"(
      gate = min(gate, clamp_limit);
      up = clamp(up, -clamp_limit, clamp_limit);
)";
    source += R"(
      destination[r * nrows + row] = (real)(up * gate * stable_sigmoid(gate));
    }
  }
}
)";
  } else {
    source += R"(
// Each threadgroup computes two weight rows for every activation row, reading the weights once.
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
  constexpr uint activation_rows = )";
    source += std::to_string(descriptor.activationRows);
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
  threadgroup float partials[4 * activation_rows][32];

  float gate_sum0[activation_rows];
  float gate_sum1[activation_rows];
  float up_sum0[activation_rows];
  float up_sum1[activation_rows];
  for (uint r = 0; r < activation_rows; r++) {
    gate_sum0[r] = 0;
    gate_sum1[r] = 0;
    up_sum0[r] = 0;
    up_sum1[r] = 0;
  }
  const uint vectors = ncols / 4;
  const uint stride = simdgroups_per_threadgroup * 64;
  uint i = simdgroup_index * 64 + lane * 2;
  for (; i + 1 < vectors; i += stride) {
    const float4 g00 = float4(gate0[i]);
    const float4 g01 = float4(gate0[i + 1]);
    const float4 u00 = float4(up0[i]);
    const float4 u01 = float4(up0[i + 1]);
    float4 g10 = 0, g11 = 0, u10 = 0, u11 = 0;
    if (active1) {
      g10 = float4(gate1[i]);
      g11 = float4(gate1[i + 1]);
      u10 = float4(up1[i]);
      u11 = float4(up1[i + 1]);
    }
    for (uint r = 0; r < activation_rows; r++) {
      const float4 a0 = float4(activation4[r * vectors + i]);
      const float4 a1 = float4(activation4[r * vectors + i + 1]);
      gate_sum0[r] += dot(g00, a0);
      gate_sum0[r] += dot(g01, a1);
      up_sum0[r] += dot(u00, a0);
      up_sum0[r] += dot(u01, a1);
      if (active1) {
        gate_sum1[r] += dot(g10, a0);
        gate_sum1[r] += dot(g11, a1);
        up_sum1[r] += dot(u10, a0);
        up_sum1[r] += dot(u11, a1);
      }
    }
  }
  if (i < vectors) {
    const float4 g00 = float4(gate0[i]);
    const float4 u00 = float4(up0[i]);
    float4 g10 = 0, u10 = 0;
    if (active1) {
      g10 = float4(gate1[i]);
      u10 = float4(up1[i]);
    }
    for (uint r = 0; r < activation_rows; r++) {
      const float4 a0 = float4(activation4[r * vectors + i]);
      gate_sum0[r] += dot(g00, a0);
      up_sum0[r] += dot(u00, a0);
      if (active1) {
        gate_sum1[r] += dot(g10, a0);
        up_sum1[r] += dot(u10, a0);
      }
    }
  }

  if (simdgroup_index == 0)
    for (uint k = 0; k < 4 * activation_rows; k++)
      partials[k][lane] = 0;
  float lane_sums[4 * activation_rows];
  for (uint r = 0; r < activation_rows; r++) {
    lane_sums[4 * r + 0] = simd_sum(gate_sum0[r]);
    lane_sums[4 * r + 1] = simd_sum(gate_sum1[r]);
    lane_sums[4 * r + 2] = simd_sum(up_sum0[r]);
    lane_sums[4 * r + 3] = simd_sum(up_sum1[r]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0)
    for (uint k = 0; k < 4 * activation_rows; k++)
      partials[k][simdgroup_index] = lane_sums[k];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index == 0) {
    for (uint r = 0; r < activation_rows; r++) {
      const float all_gate0 = simd_sum(partials[4 * r + 0][lane]);
      const float all_gate1 = simd_sum(partials[4 * r + 1][lane]);
      const float all_up0 = simd_sum(partials[4 * r + 2][lane]);
      const float all_up1 = simd_sum(partials[4 * r + 3][lane]);
      const bool second = lane == 2 * r + 1;
      if (lane == 2 * r || (second && active1)) {
        float gate = (second ? all_gate1 : all_gate0) * (float)gate_scales[row_base + (second ? 1 : 0)];
        float up = (second ? all_up1 : all_up0) * (float)up_scales[row_base + (second ? 1 : 0)];
)";
    if (descriptor.clamp)
      source += R"(
        gate = min(gate, clamp_limit);
        up = clamp(up, -clamp_limit, clamp_limit);
)";
    source += R"(
        destination[r * nrows + row_base + (second ? 1 : 0)] = (real)(up * gate * stable_sigmoid(gate));
      }
    }
  }
}
)";
  }

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
