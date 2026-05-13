#include "Int8GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include "../../ccv_nnc_8i_rowwise_packed_grids.inc"

namespace {

static uint32_t compact_iq2_grid_entry(const uint64_t value)
{
  uint32_t packed = 0;
  for (uint32_t lane = 0; lane < 8; ++lane) {
    const uint32_t v = (uint32_t)((value >> (lane * 8)) & 0xff);
    const uint32_t code = v == 8 ? 0 : (v == 25 ? 1 : 2);
    packed |= code << (lane * 2);
  }
  return packed;
}

static uint32_t byte_iq3xxs_grid_entry(const uint32_t value)
{
  uint32_t packed = 0;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    const uint32_t v = (uint32_t)((value >> (lane * 8)) & 0xff);
    packed |= (v >> 2) << (lane * 8);
  }
  return packed;
}

template<typename T, typename Transform>
static void append_compact_grid(std::string& shader, const char* const name, const T* const values, const size_t count, Transform transform)
{
  shader += "constant uint ";
  shader += name;
  shader += "[";
  shader += std::to_string(count);
  shader += "] = {";
  for (size_t i = 0; i < count; ++i) {
    if (i != 0)
      shader += ",";
    if ((i % 8) == 0)
      shader += "\n  ";
    shader += std::to_string(transform(values[i]));
    shader += "u";
  }
  shader += "\n};\n";
}

static void append_iq2xxs_scaled_grid(std::string& shader)
{
  static constexpr int q2_xxs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
  shader += "constant uint iq2xxs_scaled_grid[8192] = {";
  for (uint32_t scale_code = 0; scale_code < 16; ++scale_code) {
    const int scale = q2_xxs_scales[scale_code];
    for (uint32_t index = 0; index < 256; ++index) {
      const uint16_t entry = ccv_nnc_8i_rowwise_packed_iq2xxs_grid[index];
      for (uint32_t block = 0; block < 2; ++block) {
        if (scale_code != 0 || index != 0 || block != 0)
          shader += ",";
        if (((scale_code * 512 + index * 2 + block) % 8) == 0)
          shader += "\n  ";
        uint32_t packed = 0;
        for (uint32_t lane = 0; lane < 4; ++lane) {
          const uint32_t code = (entry >> ((block * 4 + lane) * 2)) & 3;
          const int raw = (1 + (int)code * 2) * scale;
          const uint32_t mag = (uint32_t)(raw < 127 ? raw : 127);
          packed |= mag << (lane * 8);
        }
        shader += std::to_string(packed);
        shader += "u";
      }
    }
  }
  shader += "\n};\n";
}

static void append_packed_m3_kernel(std::string& shader, const bool fusedBias)
{
  shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real4* y2 = (device const real4*)(src1 + ncols * 2);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[6][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  float sum20 = 0;
  float sum21 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = (g * group_size) >> 2;
    const float3 dot0 = dot_group(src0, (rb + 0) * groups_per_row + g, y0, y1, y2, yv_base);
    sum00 += dot0.x;
    sum10 += dot0.y;
    sum20 += dot0.z;
    if (active1) {
      const float3 dot1 = dot_group(src0, (rb + 1) * groups_per_row + g, y0, y1, y2, yv_base);
      sum01 += dot1.x;
      sum11 += dot1.y;
      sum21 += dot1.z;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
    partials[4][tiisg] = 0;
    partials[5][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  const float lane_sum20 = simd_sum(sum20);
  const float lane_sum21 = simd_sum(sum21);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
    partials[4][sgitg] = lane_sum20;
    partials[5][sgitg] = lane_sum21;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    const float all_sum20 = simd_sum(partials[4][tiisg]);
    const float all_sum21 = simd_sum(partials[5][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
      float value2 = all_sum20 * scale;
)";
  if (fusedBias) {
    shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
      value2 += biasv;
)";
  }
  shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
      dst[nrows * 2 + rb + 0] = (real)value2;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
      float value2 = all_sum21 * scale;
)";
  if (fusedBias) {
    shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
      value2 += biasv;
)";
  }
  shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
      dst[nrows * 2 + rb + 1] = (real)value2;
    }
  }
}
)";
}

}

Int8GemvKernel::Int8GemvKernel(Int8GemvKernelDescriptor descriptor, MTL::Device* const device) {
  fusedBias = descriptor.fusedBias;
  mrows = descriptor.mrows;
  format = descriptor.format;
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
  if (format != 0) {
    if (mrows == 3) {
      if (format == CCV_NNC_QX_8I_ROWWISE_Q4_K) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float4 q4_nibbles(device const uchar* p, const uint offset)
{
  const uint q0 = p[offset];
  const uint q1 = p[offset + 1];
  return float4(
    (float)(q0 & 15u),
    (float)(q0 >> 4),
    (float)(q1 & 15u),
    (float)(q1 >> 4));
}

inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  device const uchar* p = source + group_index * 9;
  const int m = (int)(p[8] & 15u) + 1;
  const int b = (int)(p[8] >> 4) - 8;
  const float mf = (float)m;
  const float offset = (float)b - 8.0f * mf;
  const float4 v0 = mf * q4_nibbles(p, 0) + float4(offset);
  const float4 v1 = mf * q4_nibbles(p, 2) + float4(offset);
  const float4 v2 = mf * q4_nibbles(p, 4) + float4(offset);
  const float4 v3 = mf * q4_nibbles(p, 6) + float4(offset);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])));
}
)";
        append_packed_m3_kernel(shader, fusedBias);
        return shader;
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_Q5_K) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

inline uint3 q5_words(device const uchar* p)
{
  return uint3(
    (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24),
    (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24),
    (uint)p[8] | ((uint)p[9] << 8) | ((uint)p[10] << 16));
}

inline float4 q5_values0(const uint q)
{
  return float4(
    (float)(q & 31u),
    (float)((q >> 5) & 31u),
    (float)((q >> 10) & 31u),
    (float)((q >> 15) & 31u));
}

inline float4 q5_values1(const uint q0, const uint q1)
{
  return float4(
    (float)((q0 >> 20) & 31u),
    (float)((q0 >> 25) & 31u),
    (float)(((q0 >> 30) | (q1 << 2)) & 31u),
    (float)((q1 >> 3) & 31u));
}

inline float4 q5_values2(const uint q)
{
  return float4(
    (float)((q >> 8) & 31u),
    (float)((q >> 13) & 31u),
    (float)((q >> 18) & 31u),
    (float)((q >> 23) & 31u));
}

inline float4 q5_values3(const uint q1, const uint q2)
{
  return float4(
    (float)(((q1 >> 28) | (q2 << 4)) & 31u),
    (float)((q2 >> 1) & 31u),
    (float)((q2 >> 6) & 31u),
    (float)((q2 >> 11) & 31u));
}

inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  device const uchar* p = source + group_index * 11u;
  const uint3 q = q5_words(p);
  const int m = (int)((q.z >> 16) & 7u) + 1;
  const int b = (int)((q.z >> 19) & 31u) - 16;
  const float mf = (float)m;
  const float offset = (float)b - 16.0f * mf;
  const float4 v0 = mf * q5_values0(q.x) + float4(offset);
  const float4 v1 = mf * q5_values1(q.x, q.y) + float4(offset);
  const float4 v2 = mf * q5_values2(q.y) + float4(offset);
  const float4 v3 = mf * q5_values3(q.y, q.z) + float4(offset);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])));
}
)";
        append_packed_m3_kernel(shader, fusedBias);
        return shader;
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_Q3_K) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

inline uint2 q3_payload(device const uchar* p)
{
  return uint2(
    (uint)p[0] |
    ((uint)p[1] << 8) |
    ((uint)p[2] << 16) |
    ((uint)p[3] << 24),
    (uint)p[4] |
    ((uint)p[5] << 8) |
    ((uint)p[6] << 16));
}

inline float4 q3_values(const uint q)
{
  return float4(
    (float)(q & 7u),
    (float)((q >> 3) & 7u),
    (float)((q >> 6) & 7u),
    (float)((q >> 9) & 7u));
}

inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  device const uchar* p = source + group_index * 7;
  const uint2 payload = q3_payload(p);
  const int m = (int)((payload.y >> 16) & 31u) + 1;
  const int b = (((int)((payload.y >> 21) & 7u) - 4) << 1);
  const float mf = (float)m;
  const float offset = (float)b - 4.0f * mf;
  const float4 v0 = mf * q3_values(payload.x & 0xfffu) + float4(offset);
  const float4 v1 = mf * q3_values((payload.x >> 12) & 0xfffu) + float4(offset);
  const float4 v2 = mf * q3_values(((payload.x >> 24) | ((payload.y & 15u) << 8)) & 0xfffu) + float4(offset);
  const float4 v3 = mf * q3_values((payload.y >> 4) & 0xfffu) + float4(offset);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])));
}
)";
        append_packed_m3_kernel(shader, fusedBias);
        return shader;
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

inline ulong q2_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 42u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
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

inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  const ulong payload = q2_payload(source, group_index);
  const uint qbits = (uint)payload;
  const uint m = (uint)((payload >> 32) & 63u) + 1;
  const uint z = (uint)((payload >> 38) & 15u) << 3;
  const float mf = (float)m;
  const float zf = (float)z;
  const float4 v0 = mf * q2_values(qbits, 0) - float4(zf);
  const float4 v1 = mf * q2_values(qbits, 8) - float4(zf);
  const float4 v2 = mf * q2_values(qbits, 16) - float4(zf);
  const float4 v3 = mf * q2_values(qbits, 24) - float4(zf);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])));
}
)";
        append_packed_m3_kernel(shader, fusedBias);
        return shader;
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ||
          format == CCV_NNC_QX_8I_ROWWISE_IQ2_S ||
          format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ||
          format == CCV_NNC_QX_8I_ROWWISE_IQ3_S ||
          format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
        if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
          append_iq2xxs_scaled_grid(shader);
          append_compact_grid(shader, "iq2xxs_ksigns", ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns, 128, [](const uint8_t value) { return (uint32_t)value; });
        } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
          append_compact_grid(shader, "iq2s_grid", ccv_nnc_8i_rowwise_packed_iq2s_grid, 1024, compact_iq2_grid_entry);
        else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS)
          append_compact_grid(shader, "iq2xs_grid", ccv_nnc_8i_rowwise_packed_iq2xs_grid, 512, compact_iq2_grid_entry);
        else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S)
          append_compact_grid(shader, "iq3s_grid", ccv_nnc_8i_rowwise_packed_iq3s_grid, 512, [](const uint32_t value) { return value; });
        else
          append_compact_grid(shader, "iq3xxs_grid", ccv_nnc_8i_rowwise_packed_iq3xxs_grid, 256, byte_iq3xxs_grid_entry);
        shader += R"(
#include <metal_stdlib>
using namespace metal;
)";
        if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S) {
          shader += R"(
inline ulong iq2s_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 42u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  const ulong value =
    (ulong)data[byte_offset] |
    ((ulong)data[byte_offset + 1] << 8) |
    ((ulong)data[byte_offset + 2] << 16) |
    ((ulong)data[byte_offset + 3] << 24) |
    ((ulong)data[byte_offset + 4] << 32) |
    ((ulong)data[byte_offset + 5] << 40);
  return value >> shift;
}
)";
        } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
          shader += R"(
inline uint iq2xs_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 21u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  const uint value =
    (uint)data[byte_offset] |
    ((uint)data[byte_offset + 1] << 8) |
    ((uint)data[byte_offset + 2] << 16) |
    ((uint)data[byte_offset + 3] << 24);
  return value >> shift;
}
)";
        } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
          shader += R"(
inline uint iq3xxs_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 28u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  const uint value =
    (uint)data[byte_offset] |
    ((uint)data[byte_offset + 1] << 8) |
    ((uint)data[byte_offset + 2] << 16) |
    ((uint)data[byte_offset + 3] << 24);
  return value >> shift;
}
)";
        }
        if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
          shader += R"(
inline float4 signed_iq2xxs_values(const uint packed, const uint signs, const uint sign_lane)
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
)";
        }
        if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S || format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
          shader += R"(
inline float4 signed_iq2_values(constant uint* grid, const uint index, const uint code_lane, const uint signs, const uint sign_lane, const float scale)
{
  const uint packed = grid[index];
  const float4 mag = min(float4(
    (float)((((packed >> ((code_lane + 0u) * 2u)) & 3u) << 1u) + 1u),
    (float)((((packed >> ((code_lane + 1u) * 2u)) & 3u) << 1u) + 1u),
    (float)((((packed >> ((code_lane + 2u) * 2u)) & 3u) << 1u) + 1u),
    (float)((((packed >> ((code_lane + 3u) * 2u)) & 3u) << 1u) + 1u)) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
        }
        if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S) {
          shader += R"(
inline float4 signed_iq3s_values(const uint index, const uint signs, const uint sign_lane, const float scale)
{
  constant uchar* values = (constant uchar*)(iq3s_grid + index);
  const float4 mag = min(float4(
    (float)values[0],
    (float)values[1],
    (float)values[2],
    (float)values[3]) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
        }
        if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
          shader += R"(
inline float4 signed_iq3xxs_values(const uint index, const uint signs, const uint sign_lane, const float scale)
{
  constant uchar* values = (constant uchar*)(iq3xxs_grid + index);
  const float4 mag = min(float4(
    (float)values[0],
    (float)values[1],
    (float)values[2],
    (float)values[3]) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
        }
        switch (format) {
          case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
            shader += R"(
inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  device const uchar* p = source + group_index * 8u;
  const uint sign_codes = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | (((uint)p[7] & 15u) << 24);
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
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])) + dot(v4, float4(y0[yv_base + 4])) + dot(v5, float4(y0[yv_base + 5])) + dot(v6, float4(y0[yv_base + 6])) + dot(v7, float4(y0[yv_base + 7])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])) + dot(v4, float4(y1[yv_base + 4])) + dot(v5, float4(y1[yv_base + 5])) + dot(v6, float4(y1[yv_base + 6])) + dot(v7, float4(y1[yv_base + 7])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])) + dot(v4, float4(y2[yv_base + 4])) + dot(v5, float4(y2[yv_base + 5])) + dot(v6, float4(y2[yv_base + 6])) + dot(v7, float4(y2[yv_base + 7])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
            shader += R"(
inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  const ulong payload = iq2s_payload(source, group_index);
  const uint grid0 = (uint)(payload & 1023u);
  const uint grid1 = (uint)((payload >> 10) & 1023u);
  const uint signs = (uint)((payload >> 20) & 65535u);
  const float scale = (float)(((payload >> 36) & 63u) + 1u);
  const float4 v0 = signed_iq2_values(iq2s_grid, grid0, 0, signs, 0, scale);
  const float4 v1 = signed_iq2_values(iq2s_grid, grid0, 4, signs, 4, scale);
  const float4 v2 = signed_iq2_values(iq2s_grid, grid1, 0, signs, 8, scale);
  const float4 v3 = signed_iq2_values(iq2s_grid, grid1, 4, signs, 12, scale);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
            shader += R"(
constant int q2_xs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};

inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  const uint payload = iq2xs_payload(source, group_index);
  const uint grid0 = payload & 511u;
  const uint signs = (payload >> 9) & 255u;
  const float scale = (float)q2_xs_scales[(payload >> 17) & 15u];
  const float4 v0 = signed_iq2_values(iq2xs_grid, grid0, 0, signs, 0, scale);
  const float4 v1 = signed_iq2_values(iq2xs_grid, grid0, 4, signs, 4, scale);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
            shader += R"(
inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  device const uchar* p = source + group_index * 7;
  const uint grid0 = (uint)p[0] | (((uint)p[1] & 1u) << 8);
  const uint grid1 = ((uint)p[1] >> 1) | (((uint)p[2] & 3u) << 7);
  const uint grid2 = ((uint)p[2] >> 2) | (((uint)p[3] & 7u) << 6);
  const uint grid3 = ((uint)p[3] >> 3) | (((uint)p[4] & 15u) << 5);
  const uint signs = ((uint)p[4] >> 4) | ((uint)p[5] << 4) | (((uint)p[6] & 15u) << 12);
  const float scale = (float)((p[6] >> 4) + 1u);
  const float4 v0 = signed_iq3s_values(grid0, signs, 0, scale);
  const float4 v1 = signed_iq3s_values(grid1, signs, 4, scale);
  const float4 v2 = signed_iq3s_values(grid2, signs, 8, scale);
  const float4 v3 = signed_iq3s_values(grid3, signs, 12, scale);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])) + dot(v2, float4(y2[yv_base + 2])) + dot(v3, float4(y2[yv_base + 3])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
            shader += R"(
inline float3 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, device const real4* y2, const uint yv_base)
{
  const uint payload = iq3xxs_payload(source, group_index);
  const uint grid0 = payload & 255u;
  const uint grid1 = (payload >> 8) & 255u;
  const uint signs = (payload >> 16) & 255u;
  const float scale = (float)(((payload >> 24) & 15u) + 1u);
  const float4 v0 = signed_iq3xxs_values(grid0, signs, 0, scale);
  const float4 v1 = signed_iq3xxs_values(grid1, signs, 4, scale);
  return float3(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])),
    dot(v0, float4(y2[yv_base + 0])) + dot(v1, float4(y2[yv_base + 1])));
}
)";
            break;
          default:
            break;
        }
        append_packed_m3_kernel(shader, fusedBias);
        return shader;
      }
      CCV_NNC_MFA_PRECONDITION(false);
      return shader;
    }
    if (format == CCV_NNC_QX_8I_ROWWISE_Q5_K) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float vec_sum(const float4 v)
{
  return v.x + v.y + v.z + v.w;
}

inline uint3 q5_words(device const uchar* p)
{
  return uint3(
    (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24),
    (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24),
    (uint)p[8] | ((uint)p[9] << 8) | ((uint)p[10] << 16));
}

inline float4 q5_values0(const uint q)
{
  return float4(
    (float)(q & 31u),
    (float)((q >> 5) & 31u),
    (float)((q >> 10) & 31u),
    (float)((q >> 15) & 31u));
}

inline float4 q5_values1(const uint q0, const uint q1)
{
  return float4(
    (float)((q0 >> 20) & 31u),
    (float)((q0 >> 25) & 31u),
    (float)(((q0 >> 30) | (q1 << 2)) & 31u),
    (float)((q1 >> 3) & 31u));
}

inline float4 q5_values2(const uint q)
{
  return float4(
    (float)((q >> 8) & 31u),
    (float)((q >> 13) & 31u),
    (float)((q >> 18) & 31u),
    (float)((q >> 23) & 31u));
}

inline float4 q5_values3(const uint q1, const uint q2)
{
  return float4(
    (float)(((q1 >> 28) | (q2 << 4)) & 31u),
    (float)((q2 >> 1) & 31u),
    (float)((q2 >> 6) & 31u),
    (float)((q2 >> 11) & 31u));
}
)";
      if (mrows == 2) {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[4][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    device const uchar* p0 = src0 + ((rb + 0) * groups_per_row + g) * 11u;
    const uint3 q0 = q5_words(p0);
    const int m0 = (int)((q0.z >> 16) & 7u) + 1;
    const int b0 = (int)((q0.z >> 19) & 31u) - 16;
    const float m0f = (float)m0;
    const float o0 = (float)b0 - 16.0f * m0f;
    const float4 q00 = q5_values0(q0.x);
    const float4 q01 = q5_values1(q0.x, q0.y);
    const float4 q02 = q5_values2(q0.y);
    const float4 q03 = q5_values3(q0.y, q0.z);
    const float4 y00 = float4(y0[yv_base + 0]);
    const float4 y01 = float4(y0[yv_base + 1]);
    const float4 y02 = float4(y0[yv_base + 2]);
    const float4 y03 = float4(y0[yv_base + 3]);
    const float4 y10 = float4(y1[yv_base + 0]);
    const float4 y11 = float4(y1[yv_base + 1]);
    const float4 y12 = float4(y1[yv_base + 2]);
    const float4 y13 = float4(y1[yv_base + 3]);
    const float ysum0 = vec_sum(y00) + vec_sum(y01) + vec_sum(y02) + vec_sum(y03);
    const float ysum1 = vec_sum(y10) + vec_sum(y11) + vec_sum(y12) + vec_sum(y13);
    sum00 += m0f * (dot(q00, y00) + dot(q01, y01) + dot(q02, y02) + dot(q03, y03)) + o0 * ysum0;
    sum10 += m0f * (dot(q00, y10) + dot(q01, y11) + dot(q02, y12) + dot(q03, y13)) + o0 * ysum1;
    if (active1) {
      device const uchar* p1 = src0 + ((rb + 1) * groups_per_row + g) * 11u;
      const uint3 q1 = q5_words(p1);
      const int m1 = (int)((q1.z >> 16) & 7u) + 1;
      const int b1 = (int)((q1.z >> 19) & 31u) - 16;
      const float m1f = (float)m1;
      const float o1 = (float)b1 - 16.0f * m1f;
      const float4 q10 = q5_values0(q1.x);
      const float4 q11 = q5_values1(q1.x, q1.y);
      const float4 q12 = q5_values2(q1.y);
      const float4 q13 = q5_values3(q1.y, q1.z);
      sum01 += m1f * (dot(q10, y00) + dot(q11, y01) + dot(q12, y02) + dot(q13, y03)) + o1 * ysum0;
      sum11 += m1f * (dot(q10, y10) + dot(q11, y11) + dot(q12, y12) + dot(q13, y13)) + o1 * ysum1;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
)";
      } else {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[ROWS][32];

  float sum0 = 0;
  float sum1 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float4 y0 = float4(y4[yv_base + 0]);
    const float4 y1 = float4(y4[yv_base + 1]);
    const float4 y2 = float4(y4[yv_base + 2]);
    const float4 y3 = float4(y4[yv_base + 3]);
    const float ysum = vec_sum(y0) + vec_sum(y1) + vec_sum(y2) + vec_sum(y3);
    device const uchar* p0 = src0 + ((rb + 0) * groups_per_row + g) * 11u;
    const uint3 q0 = q5_words(p0);
    const int m0 = (int)((q0.z >> 16) & 7u) + 1;
    const int b0 = (int)((q0.z >> 19) & 31u) - 16;
    const float m0f = (float)m0;
    const float offset0 = (float)b0 - 16.0f * m0f;
    sum0 += m0f * dot(q5_values0(q0.x), y0);
    sum0 += m0f * dot(q5_values1(q0.x, q0.y), y1);
    sum0 += m0f * dot(q5_values2(q0.y), y2);
    sum0 += m0f * dot(q5_values3(q0.y, q0.z), y3);
    sum0 += offset0 * ysum;
    if (active1) {
      device const uchar* p1 = src0 + ((rb + 1) * groups_per_row + g) * 11u;
      const uint3 q1 = q5_words(p1);
      const int m1 = (int)((q1.z >> 16) & 7u) + 1;
      const int b1 = (int)((q1.z >> 19) & 31u) - 16;
      const float m1f = (float)m1;
      const float offset1 = (float)b1 - 16.0f * m1f;
      sum1 += m1f * dot(q5_values0(q1.x), y0);
      sum1 += m1f * dot(q5_values1(q1.x, q1.y), y1);
      sum1 += m1f * dot(q5_values2(q1.y), y2);
      sum1 += m1f * dot(q5_values3(q1.y, q1.z), y3);
      sum1 += offset1 * ysum;
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
      }
      return shader;
    }
    if (format == CCV_NNC_QX_8I_ROWWISE_Q4_K) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float vec_sum(const float4 v)
{
  return v.x + v.y + v.z + v.w;
}

inline float4 q4_nibbles(device const uchar* p, const uint offset)
{
  const uint q0 = p[offset];
  const uint q1 = p[offset + 1];
  return float4(
    (float)(q0 & 15u),
    (float)(q0 >> 4),
    (float)(q1 & 15u),
    (float)(q1 >> 4));
}
)";
      if (mrows == 2) {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[4][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    device const uchar* p0 = src0 + ((rb + 0) * groups_per_row + g) * 9;
    const int m0 = (int)(p0[8] & 15u) + 1;
    const int b0 = (int)(p0[8] >> 4) - 8;
    const float m0f = (float)m0;
    const float o0 = (float)b0 - 8.0f * m0f;
    const float4 q00 = q4_nibbles(p0, 0);
    const float4 q01 = q4_nibbles(p0, 2);
    const float4 q02 = q4_nibbles(p0, 4);
    const float4 q03 = q4_nibbles(p0, 6);
    const float4 y00 = float4(y0[yv_base + 0]);
    const float4 y01 = float4(y0[yv_base + 1]);
    const float4 y02 = float4(y0[yv_base + 2]);
    const float4 y03 = float4(y0[yv_base + 3]);
    const float4 y10 = float4(y1[yv_base + 0]);
    const float4 y11 = float4(y1[yv_base + 1]);
    const float4 y12 = float4(y1[yv_base + 2]);
    const float4 y13 = float4(y1[yv_base + 3]);
    const float ysum0 = vec_sum(y00) + vec_sum(y01) + vec_sum(y02) + vec_sum(y03);
    const float ysum1 = vec_sum(y10) + vec_sum(y11) + vec_sum(y12) + vec_sum(y13);
    sum00 += m0f * (dot(q00, y00) + dot(q01, y01) + dot(q02, y02) + dot(q03, y03)) + o0 * ysum0;
    sum10 += m0f * (dot(q00, y10) + dot(q01, y11) + dot(q02, y12) + dot(q03, y13)) + o0 * ysum1;
    if (active1) {
      device const uchar* p1 = src0 + ((rb + 1) * groups_per_row + g) * 9;
      const int m1 = (int)(p1[8] & 15u) + 1;
      const int b1 = (int)(p1[8] >> 4) - 8;
      const float m1f = (float)m1;
      const float o1 = (float)b1 - 8.0f * m1f;
      const float4 q10 = q4_nibbles(p1, 0);
      const float4 q11 = q4_nibbles(p1, 2);
      const float4 q12 = q4_nibbles(p1, 4);
      const float4 q13 = q4_nibbles(p1, 6);
      sum01 += m1f * (dot(q10, y00) + dot(q11, y01) + dot(q12, y02) + dot(q13, y03)) + o1 * ysum0;
      sum11 += m1f * (dot(q10, y10) + dot(q11, y11) + dot(q12, y12) + dot(q13, y13)) + o1 * ysum1;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
)";
      } else {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[ROWS][32];

  float sum0 = 0;
  float sum1 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float4 y0 = float4(y4[yv_base + 0]);
    const float4 y1 = float4(y4[yv_base + 1]);
    const float4 y2 = float4(y4[yv_base + 2]);
    const float4 y3 = float4(y4[yv_base + 3]);
    device const uchar* p0 = src0 + ((rb + 0) * groups_per_row + g) * 9;
    const int m0 = (int)(p0[8] & 15u) + 1;
    const int b0 = (int)(p0[8] >> 4) - 8;
    const float m0f = (float)m0;
    const float offset0 = (float)b0 - 8.0f * m0f;
    const float ysum = vec_sum(y0) + vec_sum(y1) + vec_sum(y2) + vec_sum(y3);
    sum0 += m0f * dot(q4_nibbles(p0, 0), y0);
    sum0 += m0f * dot(q4_nibbles(p0, 2), y1);
    sum0 += m0f * dot(q4_nibbles(p0, 4), y2);
    sum0 += m0f * dot(q4_nibbles(p0, 6), y3);
    sum0 += offset0 * ysum;
    if (active1) {
      device const uchar* p1 = src0 + ((rb + 1) * groups_per_row + g) * 9;
      const int m1 = (int)(p1[8] & 15u) + 1;
      const int b1 = (int)(p1[8] >> 4) - 8;
      const float m1f = (float)m1;
      const float offset1 = (float)b1 - 8.0f * m1f;
      sum1 += m1f * dot(q4_nibbles(p1, 0), y0);
      sum1 += m1f * dot(q4_nibbles(p1, 2), y1);
      sum1 += m1f * dot(q4_nibbles(p1, 4), y2);
      sum1 += m1f * dot(q4_nibbles(p1, 6), y3);
      sum1 += offset1 * ysum;
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
      }
      return shader;
    }
    if (format == CCV_NNC_QX_8I_ROWWISE_Q3_K) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float vec_sum(const float4 v)
{
  return v.x + v.y + v.z + v.w;
}

inline uint2 q3_payload(device const uchar* p)
{
  return uint2(
    (uint)p[0] |
    ((uint)p[1] << 8) |
    ((uint)p[2] << 16) |
    ((uint)p[3] << 24),
    (uint)p[4] |
    ((uint)p[5] << 8) |
    ((uint)p[6] << 16));
}

inline float4 q3_values(const uint q)
{
  return float4(
    (float)(q & 7u),
    (float)((q >> 3) & 7u),
    (float)((q >> 6) & 7u),
    (float)((q >> 9) & 7u));
}
)";
      if (mrows == 2) {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[4][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float4 y00 = float4(y0[yv_base + 0]);
    const float4 y01 = float4(y0[yv_base + 1]);
    const float4 y02 = float4(y0[yv_base + 2]);
    const float4 y03 = float4(y0[yv_base + 3]);
    const float4 y10 = float4(y1[yv_base + 0]);
    const float4 y11 = float4(y1[yv_base + 1]);
    const float4 y12 = float4(y1[yv_base + 2]);
    const float4 y13 = float4(y1[yv_base + 3]);
    const float ysum0 = vec_sum(y00) + vec_sum(y01) + vec_sum(y02) + vec_sum(y03);
    const float ysum1 = vec_sum(y10) + vec_sum(y11) + vec_sum(y12) + vec_sum(y13);
    device const uchar* p0 = src0 + ((rb + 0) * groups_per_row + g) * 7;
    const uint2 payload0 = q3_payload(p0);
    const int m0 = (int)((payload0.y >> 16) & 31u) + 1;
    const int b0 = (((int)((payload0.y >> 21) & 7u) - 4) << 1);
    const float m0f = (float)m0;
    const float o0 = (float)b0 - 4.0f * m0f;
    const float4 q00 = q3_values(payload0.x & 0xfffu);
    const float4 q01 = q3_values((payload0.x >> 12) & 0xfffu);
    const float4 q02 = q3_values(((payload0.x >> 24) | ((payload0.y & 15u) << 8)) & 0xfffu);
    const float4 q03 = q3_values((payload0.y >> 4) & 0xfffu);
    sum00 += m0f * (dot(q00, y00) + dot(q01, y01) + dot(q02, y02) + dot(q03, y03)) + o0 * ysum0;
    sum10 += m0f * (dot(q00, y10) + dot(q01, y11) + dot(q02, y12) + dot(q03, y13)) + o0 * ysum1;
    if (active1) {
      device const uchar* p1 = src0 + ((rb + 1) * groups_per_row + g) * 7;
      const uint2 payload1 = q3_payload(p1);
      const int m1 = (int)((payload1.y >> 16) & 31u) + 1;
      const int b1 = (((int)((payload1.y >> 21) & 7u) - 4) << 1);
      const float m1f = (float)m1;
      const float o1 = (float)b1 - 4.0f * m1f;
      const float4 q10 = q3_values(payload1.x & 0xfffu);
      const float4 q11 = q3_values((payload1.x >> 12) & 0xfffu);
      const float4 q12 = q3_values(((payload1.x >> 24) | ((payload1.y & 15u) << 8)) & 0xfffu);
      const float4 q13 = q3_values((payload1.y >> 4) & 0xfffu);
      sum01 += m1f * (dot(q10, y00) + dot(q11, y01) + dot(q12, y02) + dot(q13, y03)) + o1 * ysum0;
      sum11 += m1f * (dot(q10, y10) + dot(q11, y11) + dot(q12, y12) + dot(q13, y13)) + o1 * ysum1;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
)";
      } else {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[ROWS][32];

  float sum0 = 0;
  float sum1 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float4 y0 = float4(y4[yv_base + 0]);
    const float4 y1 = float4(y4[yv_base + 1]);
    const float4 y2 = float4(y4[yv_base + 2]);
    const float4 y3 = float4(y4[yv_base + 3]);
    const float ysum = vec_sum(y0) + vec_sum(y1) + vec_sum(y2) + vec_sum(y3);
    device const uchar* p0 = src0 + ((rb + 0) * groups_per_row + g) * 7;
    const uint2 payload0 = q3_payload(p0);
    const int m0 = (int)((payload0.y >> 16) & 31u) + 1;
    const int b0 = (((int)((payload0.y >> 21) & 7u) - 4) << 1);
    const float m0f = (float)m0;
    const float offset0 = (float)b0 - 4.0f * m0f;
    sum0 += m0f * dot(q3_values(payload0.x & 0xfffu), y0);
    sum0 += m0f * dot(q3_values((payload0.x >> 12) & 0xfffu), y1);
    sum0 += m0f * dot(q3_values(((payload0.x >> 24) | ((payload0.y & 15u) << 8)) & 0xfffu), y2);
    sum0 += m0f * dot(q3_values((payload0.y >> 4) & 0xfffu), y3);
    sum0 += offset0 * ysum;
    if (active1) {
      device const uchar* p1 = src0 + ((rb + 1) * groups_per_row + g) * 7;
      const uint2 payload1 = q3_payload(p1);
      const int m1 = (int)((payload1.y >> 16) & 31u) + 1;
      const int b1 = (((int)((payload1.y >> 21) & 7u) - 4) << 1);
      const float m1f = (float)m1;
      const float offset1 = (float)b1 - 4.0f * m1f;
      sum1 += m1f * dot(q3_values(payload1.x & 0xfffu), y0);
      sum1 += m1f * dot(q3_values((payload1.x >> 12) & 0xfffu), y1);
      sum1 += m1f * dot(q3_values(((payload1.x >> 24) | ((payload1.y & 15u) << 8)) & 0xfffu), y2);
      sum1 += m1f * dot(q3_values((payload1.y >> 4) & 0xfffu), y3);
      sum1 += offset1 * ysum;
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
      }
      return shader;
    }
    if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float vec_sum(const float4 v)
{
  return v.x + v.y + v.z + v.w;
}

inline ulong q2_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 42u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
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
)";
      if (mrows == 2) {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[4][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float4 y00 = float4(y0[yv_base + 0]);
    const float4 y01 = float4(y0[yv_base + 1]);
    const float4 y02 = float4(y0[yv_base + 2]);
    const float4 y03 = float4(y0[yv_base + 3]);
    const float4 y10 = float4(y1[yv_base + 0]);
    const float4 y11 = float4(y1[yv_base + 1]);
    const float4 y12 = float4(y1[yv_base + 2]);
    const float4 y13 = float4(y1[yv_base + 3]);
    const float ysum0 = vec_sum(y00) + vec_sum(y01) + vec_sum(y02) + vec_sum(y03);
    const float ysum1 = vec_sum(y10) + vec_sum(y11) + vec_sum(y12) + vec_sum(y13);
    const ulong payload0 = q2_payload(src0, (rb + 0) * groups_per_row + g);
    const uint qbits0 = (uint)payload0;
    const uint m0 = (uint)((payload0 >> 32) & 63u) + 1;
    const uint z0 = (uint)((payload0 >> 38) & 15u) << 3;
    const float m0f = (float)m0;
    const float z0f = (float)z0;
    const float4 q00 = q2_values(qbits0, 0);
    const float4 q01 = q2_values(qbits0, 8);
    const float4 q02 = q2_values(qbits0, 16);
    const float4 q03 = q2_values(qbits0, 24);
    sum00 += m0f * (dot(q00, y00) + dot(q01, y01) + dot(q02, y02) + dot(q03, y03)) - z0f * ysum0;
    sum10 += m0f * (dot(q00, y10) + dot(q01, y11) + dot(q02, y12) + dot(q03, y13)) - z0f * ysum1;
    if (active1) {
      const ulong payload1 = q2_payload(src0, (rb + 1) * groups_per_row + g);
      const uint qbits1 = (uint)payload1;
      const uint m1 = (uint)((payload1 >> 32) & 63u) + 1;
      const uint z1 = (uint)((payload1 >> 38) & 15u) << 3;
      const float m1f = (float)m1;
      const float z1f = (float)z1;
      const float4 q10 = q2_values(qbits1, 0);
      const float4 q11 = q2_values(qbits1, 8);
      const float4 q12 = q2_values(qbits1, 16);
      const float4 q13 = q2_values(qbits1, 24);
      sum01 += m1f * (dot(q10, y00) + dot(q11, y01) + dot(q12, y02) + dot(q13, y03)) - z1f * ysum0;
      sum11 += m1f * (dot(q10, y10) + dot(q11, y11) + dot(q12, y12) + dot(q13, y13)) - z1f * ysum1;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
)";
      } else {
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[ROWS][32];

  float sum0 = 0;
  float sum1 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float4 y0 = float4(y4[yv_base + 0]);
    const float4 y1 = float4(y4[yv_base + 1]);
    const float4 y2 = float4(y4[yv_base + 2]);
    const float4 y3 = float4(y4[yv_base + 3]);
    const float ysum = vec_sum(y0) + vec_sum(y1) + vec_sum(y2) + vec_sum(y3);
    const ulong payload0 = q2_payload(src0, (rb + 0) * groups_per_row + g);
    const uint qbits0 = (uint)payload0;
    const uint m0 = (uint)((payload0 >> 32) & 63u) + 1;
    const uint z0 = (uint)((payload0 >> 38) & 15u) << 3;
    const float m0f = (float)m0;
    sum0 += m0f * dot(q2_values(qbits0, 0), y0);
    sum0 += m0f * dot(q2_values(qbits0, 8), y1);
    sum0 += m0f * dot(q2_values(qbits0, 16), y2);
    sum0 += m0f * dot(q2_values(qbits0, 24), y3);
    sum0 -= (float)z0 * ysum;
    if (active1) {
      const ulong payload1 = q2_payload(src0, (rb + 1) * groups_per_row + g);
      const uint qbits1 = (uint)payload1;
      const uint m1 = (uint)((payload1 >> 32) & 63u) + 1;
      const uint z1 = (uint)((payload1 >> 38) & 15u) << 3;
      const float m1f = (float)m1;
      sum1 += m1f * dot(q2_values(qbits1, 0), y0);
      sum1 += m1f * dot(q2_values(qbits1, 8), y1);
      sum1 += m1f * dot(q2_values(qbits1, 16), y2);
      sum1 += m1f * dot(q2_values(qbits1, 24), y3);
      sum1 -= (float)z1 * ysum;
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
      }
      return shader;
    }
    if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
      append_compact_grid(shader, "iq3xxs_grid", ccv_nnc_8i_rowwise_packed_iq3xxs_grid, 256, byte_iq3xxs_grid_entry);
      shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float4 signed_iq3xxs_values(const uint index, const uint signs, const uint sign_lane, const float scale)
{
  constant uchar* values = (constant uchar*)(iq3xxs_grid + index);
  const float4 mag = min(float4(
    (float)values[0],
    (float)values[1],
    (float)values[2],
    (float)values[3]) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
      if (mrows == 2) {
        shader += R"(
inline float2 dot_pair(device const uchar* source, const uint pair_index, device const real4* y0, device const real4* y1, const uint yv_base)
{
  device const uchar* p = source + pair_index * 7;
  const float scale0 = (float)((p[3] & 15u) + 1u);
  const float4 v0 = signed_iq3xxs_values((uint)p[0], (uint)p[2], 0, scale0);
  const float4 v1 = signed_iq3xxs_values((uint)p[1], (uint)p[2], 4, scale0);
  const uint grid2 = ((uint)p[3] >> 4) | (((uint)p[4] & 15u) << 4);
  const uint grid3 = ((uint)p[4] >> 4) | (((uint)p[5] & 15u) << 4);
  const uint signs1 = ((uint)p[5] >> 4) | (((uint)p[6] & 15u) << 4);
  const float scale1 = (float)((p[6] >> 4) + 1u);
  const float4 v2 = signed_iq3xxs_values(grid2, signs1, 0, scale1);
  const float4 v3 = signed_iq3xxs_values(grid3, signs1, 4, scale1);
  return float2(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])));
}

kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  const uint pairs_per_row = groups_per_row >> 1;
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[4][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < pairs_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    const float2 dot0 = dot_pair(src0, (rb + 0) * pairs_per_row + g, y0, y1, yv_base);
    sum00 += dot0.x;
    sum10 += dot0.y;
    if (active1) {
      const float2 dot1 = dot_pair(src0, (rb + 1) * pairs_per_row + g, y0, y1, yv_base);
      sum01 += dot1.x;
      sum11 += dot1.y;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
)";
      } else {
        shader += R"(
inline float dot_pair(device const uchar* source, const uint pair_index, device const real4* y4, const uint yv_base)
{
  device const uchar* p = source + pair_index * 7;
  const float scale0 = (float)((p[3] & 15u) + 1u);
  const float4 v0 = signed_iq3xxs_values((uint)p[0], (uint)p[2], 0, scale0);
  const float4 v1 = signed_iq3xxs_values((uint)p[1], (uint)p[2], 4, scale0);
  const uint grid2 = ((uint)p[3] >> 4) | (((uint)p[4] & 15u) << 4);
  const uint grid3 = ((uint)p[4] >> 4) | (((uint)p[5] & 15u) << 4);
  const uint signs1 = ((uint)p[5] >> 4) | (((uint)p[6] & 15u) << 4);
  const float scale1 = (float)((p[6] >> 4) + 1u);
  const float4 v2 = signed_iq3xxs_values(grid2, signs1, 0, scale1);
  const float4 v3 = signed_iq3xxs_values(grid3, signs1, 4, scale1);
  return dot(v0, float4(y4[yv_base + 0])) + dot(v1, float4(y4[yv_base + 1])) + dot(v2, float4(y4[yv_base + 2])) + dot(v3, float4(y4[yv_base + 3]));
}

kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  const uint pairs_per_row = groups_per_row >> 1;
  device const real4* y4 = (device const real4*)src1;
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[ROWS][32];

  float sum0 = 0;
  float sum1 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < pairs_per_row; g += group_stride) {
    const uint yv_base = g * 4;
    sum0 += dot_pair(src0, (rb + 0) * pairs_per_row + g, y4, yv_base);
    if (active1)
      sum1 += dot_pair(src0, (rb + 1) * pairs_per_row + g, y4, yv_base);
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
      }
      return shader;
    }
    if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ||
        format == CCV_NNC_QX_8I_ROWWISE_IQ2_S ||
        format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ||
        format == CCV_NNC_QX_8I_ROWWISE_IQ3_S ||
        format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
        append_iq2xxs_scaled_grid(shader);
        append_compact_grid(shader, "iq2xxs_ksigns", ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns, 128, [](const uint8_t value) { return (uint32_t)value; });
      } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
        append_compact_grid(shader, "iq2s_grid", ccv_nnc_8i_rowwise_packed_iq2s_grid, 1024, compact_iq2_grid_entry);
      else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS)
        append_compact_grid(shader, "iq2xs_grid", ccv_nnc_8i_rowwise_packed_iq2xs_grid, 512, compact_iq2_grid_entry);
      else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S)
        append_compact_grid(shader, "iq3s_grid", ccv_nnc_8i_rowwise_packed_iq3s_grid, 512, [](const uint32_t value) { return value; });
      else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)
        append_compact_grid(shader, "iq3xxs_grid", ccv_nnc_8i_rowwise_packed_iq3xxs_grid, 256, byte_iq3xxs_grid_entry);
      shader += R"(
#include <metal_stdlib>
using namespace metal;
)";
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S) {
        shader += R"(
inline ulong iq2s_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 42u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  const ulong value =
    (ulong)data[byte_offset] |
    ((ulong)data[byte_offset + 1] << 8) |
    ((ulong)data[byte_offset + 2] << 16) |
    ((ulong)data[byte_offset + 3] << 24) |
    ((ulong)data[byte_offset + 4] << 32) |
    ((ulong)data[byte_offset + 5] << 40);
  return value >> shift;
}
)";
      } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
        shader += R"(
inline uint iq2xs_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 21u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  const uint value =
    (uint)data[byte_offset] |
    ((uint)data[byte_offset + 1] << 8) |
    ((uint)data[byte_offset + 2] << 16) |
    ((uint)data[byte_offset + 3] << 24);
  return value >> shift;
}
)";
      } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
        shader += R"(
inline uint iq3xxs_payload(device const uchar* data, const uint group_index)
{
  const uint bit_offset = group_index * 28u;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  const uint value =
    (uint)data[byte_offset] |
    ((uint)data[byte_offset + 1] << 8) |
    ((uint)data[byte_offset + 2] << 16) |
    ((uint)data[byte_offset + 3] << 24);
  return value >> shift;
}
)";
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
        shader += R"(
inline float4 signed_iq2xxs_values(const uint packed, const uint signs, const uint sign_lane)
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
)";
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S || format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
        shader += R"(
inline float4 signed_iq2_values(constant uint* grid, const uint index, const uint code_lane, const uint signs, const uint sign_lane, const float scale)
{
  const uint packed = grid[index];
  const float4 mag = min(float4(
    (float)((((packed >> ((code_lane + 0u) * 2u)) & 3u) << 1u) + 1u),
    (float)((((packed >> ((code_lane + 1u) * 2u)) & 3u) << 1u) + 1u),
    (float)((((packed >> ((code_lane + 2u) * 2u)) & 3u) << 1u) + 1u),
    (float)((((packed >> ((code_lane + 3u) * 2u)) & 3u) << 1u) + 1u)) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S) {
        shader += R"(
inline float4 signed_iq3s_values(const uint index, const uint signs, const uint sign_lane, const float scale)
{
  constant uchar* values = (constant uchar*)(iq3s_grid + index);
  const float4 mag = min(float4(
    (float)values[0],
    (float)values[1],
    (float)values[2],
    (float)values[3]) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
      }
      if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
        shader += R"(
inline float4 signed_iq3xxs_values(const uint index, const uint signs, const uint sign_lane, const float scale)
{
  constant uchar* values = (constant uchar*)(iq3xxs_grid + index);
  const float4 mag = min(float4(
    (float)values[0],
    (float)values[1],
    (float)values[2],
    (float)values[3]) * scale, float4(127.0f));
  return float4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}
)";
      }
      if (mrows == 2) {
        switch (format) {
          case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
            shader += R"(
inline float2 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, const uint yv_base)
{
  device const uchar* p = source + group_index * 8u;
  const uint sign_codes = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | (((uint)p[7] & 15u) << 24);
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
  return float2(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])) + dot(v4, float4(y0[yv_base + 4])) + dot(v5, float4(y0[yv_base + 5])) + dot(v6, float4(y0[yv_base + 6])) + dot(v7, float4(y0[yv_base + 7])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])) + dot(v4, float4(y1[yv_base + 4])) + dot(v5, float4(y1[yv_base + 5])) + dot(v6, float4(y1[yv_base + 6])) + dot(v7, float4(y1[yv_base + 7])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
            shader += R"(
inline float2 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, const uint yv_base)
{
  const ulong payload = iq2s_payload(source, group_index);
  const uint grid0 = (uint)(payload & 1023u);
  const uint grid1 = (uint)((payload >> 10) & 1023u);
  const uint signs = (uint)((payload >> 20) & 65535u);
  const float scale = (float)(((payload >> 36) & 63u) + 1u);
  const float4 v0 = signed_iq2_values(iq2s_grid, grid0, 0, signs, 0, scale);
  const float4 v1 = signed_iq2_values(iq2s_grid, grid0, 4, signs, 4, scale);
  const float4 v2 = signed_iq2_values(iq2s_grid, grid1, 0, signs, 8, scale);
  const float4 v3 = signed_iq2_values(iq2s_grid, grid1, 4, signs, 12, scale);
  return float2(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
            shader += R"(
constant int q2_xs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};

inline float2 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, const uint yv_base)
{
  const uint payload = iq2xs_payload(source, group_index);
  const uint grid0 = payload & 511u;
  const uint signs = (payload >> 9) & 255u;
  const float scale = (float)q2_xs_scales[(payload >> 17) & 15u];
  const float4 v0 = signed_iq2_values(iq2xs_grid, grid0, 0, signs, 0, scale);
  const float4 v1 = signed_iq2_values(iq2xs_grid, grid0, 4, signs, 4, scale);
  return float2(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
            shader += R"(
inline float2 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, const uint yv_base)
{
  device const uchar* p = source + group_index * 7;
  const uint grid0 = (uint)p[0] | (((uint)p[1] & 1u) << 8);
  const uint grid1 = ((uint)p[1] >> 1) | (((uint)p[2] & 3u) << 7);
  const uint grid2 = ((uint)p[2] >> 2) | (((uint)p[3] & 7u) << 6);
  const uint grid3 = ((uint)p[3] >> 3) | (((uint)p[4] & 15u) << 5);
  const uint signs = ((uint)p[4] >> 4) | ((uint)p[5] << 4) | (((uint)p[6] & 15u) << 12);
  const float scale = (float)((p[6] >> 4) + 1u);
  const float4 v0 = signed_iq3s_values(grid0, signs, 0, scale);
  const float4 v1 = signed_iq3s_values(grid1, signs, 4, scale);
  const float4 v2 = signed_iq3s_values(grid2, signs, 8, scale);
  const float4 v3 = signed_iq3s_values(grid3, signs, 12, scale);
  return float2(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])) + dot(v2, float4(y0[yv_base + 2])) + dot(v3, float4(y0[yv_base + 3])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])) + dot(v2, float4(y1[yv_base + 2])) + dot(v3, float4(y1[yv_base + 3])));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
            shader += R"(
inline float2 dot_group(device const uchar* source, const uint group_index, device const real4* y0, device const real4* y1, const uint yv_base)
{
  const uint payload = iq3xxs_payload(source, group_index);
  const uint grid0 = payload & 255u;
  const uint grid1 = (payload >> 8) & 255u;
  const uint signs = (payload >> 16) & 255u;
  const float scale = (float)(((payload >> 24) & 15u) + 1u);
  const float4 v0 = signed_iq3xxs_values(grid0, signs, 0, scale);
  const float4 v1 = signed_iq3xxs_values(grid1, signs, 4, scale);
  return float2(
    dot(v0, float4(y0[yv_base + 0])) + dot(v1, float4(y0[yv_base + 1])),
    dot(v0, float4(y1[yv_base + 0])) + dot(v1, float4(y1[yv_base + 1])));
}
)";
            break;
          default:
            break;
        }
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[4][32];

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = (g * group_size) >> 2;
    const float2 dot0 = dot_group(src0, (rb + 0) * groups_per_row + g, y0, y1, yv_base);
    sum00 += dot0.x;
    sum10 += dot0.y;
    if (active1) {
      const float2 dot1 = dot_group(src0, (rb + 1) * groups_per_row + g, y0, y1, yv_base);
      sum01 += dot1.x;
      sum11 += dot1.y;
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
        if (fusedBias) {
          shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
        }
        shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
)";
      } else {
        switch (format) {
          case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
            shader += R"(
inline float dot_group(device const uchar* source, const uint group_index, device const real4* y4, const uint yv_base)
{
  device const uchar* p = source + group_index * 8u;
  const uint sign_codes = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | (((uint)p[7] & 15u) << 24);
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
  return dot(v0, float4(y4[yv_base + 0])) + dot(v1, float4(y4[yv_base + 1])) + dot(v2, float4(y4[yv_base + 2])) + dot(v3, float4(y4[yv_base + 3])) + dot(v4, float4(y4[yv_base + 4])) + dot(v5, float4(y4[yv_base + 5])) + dot(v6, float4(y4[yv_base + 6])) + dot(v7, float4(y4[yv_base + 7]));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
            shader += R"(
inline float dot_group(device const uchar* source, const uint group_index, device const real4* y4, const uint yv_base)
{
  const ulong payload = iq2s_payload(source, group_index);
  const uint grid0 = (uint)(payload & 1023u);
  const uint grid1 = (uint)((payload >> 10) & 1023u);
  const uint signs = (uint)((payload >> 20) & 65535u);
  const float scale = (float)(((payload >> 36) & 63u) + 1u);
  const float4 v0 = signed_iq2_values(iq2s_grid, grid0, 0, signs, 0, scale);
  const float4 v1 = signed_iq2_values(iq2s_grid, grid0, 4, signs, 4, scale);
  const float4 v2 = signed_iq2_values(iq2s_grid, grid1, 0, signs, 8, scale);
  const float4 v3 = signed_iq2_values(iq2s_grid, grid1, 4, signs, 12, scale);
  return dot(v0, float4(y4[yv_base + 0])) + dot(v1, float4(y4[yv_base + 1])) + dot(v2, float4(y4[yv_base + 2])) + dot(v3, float4(y4[yv_base + 3]));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
            shader += R"(
constant int q2_xs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};

inline float dot_group(device const uchar* source, const uint group_index, device const real4* y4, const uint yv_base)
{
  const uint payload = iq2xs_payload(source, group_index);
  const uint grid0 = payload & 511u;
  const uint signs = (payload >> 9) & 255u;
  const float scale = (float)q2_xs_scales[(payload >> 17) & 15u];
  const float4 v0 = signed_iq2_values(iq2xs_grid, grid0, 0, signs, 0, scale);
  const float4 v1 = signed_iq2_values(iq2xs_grid, grid0, 4, signs, 4, scale);
  return dot(v0, float4(y4[yv_base + 0])) + dot(v1, float4(y4[yv_base + 1]));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
            shader += R"(
inline float dot_group(device const uchar* source, const uint group_index, device const real4* y4, const uint yv_base)
{
  device const uchar* p = source + group_index * 7;
  const uint grid0 = (uint)p[0] | (((uint)p[1] & 1u) << 8);
  const uint grid1 = ((uint)p[1] >> 1) | (((uint)p[2] & 3u) << 7);
  const uint grid2 = ((uint)p[2] >> 2) | (((uint)p[3] & 7u) << 6);
  const uint grid3 = ((uint)p[3] >> 3) | (((uint)p[4] & 15u) << 5);
  const uint signs = ((uint)p[4] >> 4) | ((uint)p[5] << 4) | (((uint)p[6] & 15u) << 12);
  const float scale = (float)((p[6] >> 4) + 1u);
  const float4 v0 = signed_iq3s_values(grid0, signs, 0, scale);
  const float4 v1 = signed_iq3s_values(grid1, signs, 4, scale);
  const float4 v2 = signed_iq3s_values(grid2, signs, 8, scale);
  const float4 v3 = signed_iq3s_values(grid3, signs, 12, scale);
  return dot(v0, float4(y4[yv_base + 0])) + dot(v1, float4(y4[yv_base + 1])) + dot(v2, float4(y4[yv_base + 2])) + dot(v3, float4(y4[yv_base + 3]));
}
)";
            break;
          case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
            shader += R"(
inline float dot_group(device const uchar* source, const uint group_index, device const real4* y4, const uint yv_base)
{
  const uint payload = iq3xxs_payload(source, group_index);
  const uint grid0 = payload & 255u;
  const uint grid1 = (payload >> 8) & 255u;
  const uint signs = (payload >> 16) & 255u;
  const float scale = (float)(((payload >> 24) & 15u) + 1u);
  const float4 v0 = signed_iq3xxs_values(grid0, signs, 0, scale);
  const float4 v1 = signed_iq3xxs_values(grid1, signs, 4, scale);
  return dot(v0, float4(y4[yv_base + 0])) + dot(v1, float4(y4[yv_base + 1]));
}
)";
            break;
          default:
            break;
        }
        shader += R"(
kernel void int8_gemv(
  device const uchar *src0 [[buffer(0)]],
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
  device const real* scales = (device const real*)(src0 + scale_offset);
  threadgroup float partials[ROWS][32];

  float sum0 = 0;
  float sum1 = 0;
  const uint group_stride = S * 32;
  for (uint g = sgitg * 32 + tiisg; g < groups_per_row; g += group_stride) {
    const uint yv_base = (g * group_size) >> 2;
    sum0 += dot_group(src0, (rb + 0) * groups_per_row + g, y4, yv_base);
    if (active1)
      sum1 += dot_group(src0, (rb + 1) * groups_per_row + g, y4, yv_base);
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
      }
      return shader;
    }
    CCV_NNC_MFA_PRECONDITION(false);
    return shader;
  }
  if (mrows == 3) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  device const real4* y2 = (device const real4*)(src1 + ncols * 2);
  threadgroup float partials[6][32];
  device const real* scales = (device const real*)((device const uchar*)src0 + scale_offset);
  device const char4* x0 = (device const char4*)((device const char*)src0 + (rb + 0) * ncols);
  device const char4* x1 = (device const char4*)((device const char*)src0 + (rb + 1) * ncols);

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  float sum20 = 0;
  float sum21 = 0;
  const uint nvecs = ncols / 4;
  const uint stride = S * 64;
  uint i = sgitg * 64 + tiisg * 2;
  for (; i + 1 < nvecs; i += stride) {
    const float4 y0v0 = float4(y0[i]);
    const float4 y0v1 = float4(y0[i + 1]);
    const float4 y1v0 = float4(y1[i]);
    const float4 y1v1 = float4(y1[i + 1]);
    const float4 y2v0 = float4(y2[i]);
    const float4 y2v1 = float4(y2[i + 1]);
    const char4 q0 = x0[i];
    const char4 q0b = x0[i + 1];
    sum00 += dot(float4(q0), y0v0);
    sum00 += dot(float4(q0b), y0v1);
    sum10 += dot(float4(q0), y1v0);
    sum10 += dot(float4(q0b), y1v1);
    sum20 += dot(float4(q0), y2v0);
    sum20 += dot(float4(q0b), y2v1);
    if (active1) {
      const char4 q1 = x1[i];
      const char4 q1b = x1[i + 1];
      sum01 += dot(float4(q1), y0v0);
      sum01 += dot(float4(q1b), y0v1);
      sum11 += dot(float4(q1), y1v0);
      sum11 += dot(float4(q1b), y1v1);
      sum21 += dot(float4(q1), y2v0);
      sum21 += dot(float4(q1b), y2v1);
    }
  }
  if (i < nvecs) {
    const float4 y0v = float4(y0[i]);
    const float4 y1v = float4(y1[i]);
    const float4 y2v = float4(y2[i]);
    const char4 q0 = x0[i];
    sum00 += dot(float4(q0), y0v);
    sum10 += dot(float4(q0), y1v);
    sum20 += dot(float4(q0), y2v);
    if (active1) {
      const char4 q1 = x1[i];
      sum01 += dot(float4(q1), y0v);
      sum11 += dot(float4(q1), y1v);
      sum21 += dot(float4(q1), y2v);
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
    partials[4][tiisg] = 0;
    partials[5][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  const float lane_sum20 = simd_sum(sum20);
  const float lane_sum21 = simd_sum(sum21);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
    partials[4][sgitg] = lane_sum20;
    partials[5][sgitg] = lane_sum21;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    const float all_sum20 = simd_sum(partials[4][tiisg]);
    const float all_sum21 = simd_sum(partials[5][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
      float value2 = all_sum20 * scale;
)";
    if (fusedBias) {
      shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
      value2 += biasv;
)";
    }
    shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
      dst[nrows * 2 + rb + 0] = (real)value2;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
      float value2 = all_sum21 * scale;
)";
    if (fusedBias) {
      shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
      value2 += biasv;
)";
    }
    shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
      dst[nrows * 2 + rb + 1] = (real)value2;
    }
  }
}
  )";
  } else if (mrows == 2) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

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
  device const real4* y0 = (device const real4*)src1;
  device const real4* y1 = (device const real4*)(src1 + ncols);
  threadgroup float partials[4][32];
  device const real* scales = (device const real*)((device const uchar*)src0 + scale_offset);
  device const char4* x0 = (device const char4*)((device const char*)src0 + (rb + 0) * ncols);
  device const char4* x1 = (device const char4*)((device const char*)src0 + (rb + 1) * ncols);

  float sum00 = 0;
  float sum01 = 0;
  float sum10 = 0;
  float sum11 = 0;
  const uint nvecs = ncols / 4;
  const uint stride = S * 64;
  uint i = sgitg * 64 + tiisg * 2;
  for (; i + 1 < nvecs; i += stride) {
    const float4 y0v0 = float4(y0[i]);
    const float4 y0v1 = float4(y0[i + 1]);
    const float4 y1v0 = float4(y1[i]);
    const float4 y1v1 = float4(y1[i + 1]);
    const char4 q0 = x0[i];
    const char4 q0b = x0[i + 1];
    sum00 += dot(float4(q0), y0v0);
    sum00 += dot(float4(q0b), y0v1);
    sum10 += dot(float4(q0), y1v0);
    sum10 += dot(float4(q0b), y1v1);
    if (active1) {
      const char4 q1 = x1[i];
      const char4 q1b = x1[i + 1];
      sum01 += dot(float4(q1), y0v0);
      sum01 += dot(float4(q1b), y0v1);
      sum11 += dot(float4(q1), y1v0);
      sum11 += dot(float4(q1b), y1v1);
    }
  }
  if (i < nvecs) {
    const float4 y0v = float4(y0[i]);
    const float4 y1v = float4(y1[i]);
    const char4 q0 = x0[i];
    sum00 += dot(float4(q0), y0v);
    sum10 += dot(float4(q0), y1v);
    if (active1) {
      const char4 q1 = x1[i];
      sum01 += dot(float4(q1), y0v);
      sum11 += dot(float4(q1), y1v);
    }
  }

  if (sgitg == 0) {
    partials[0][tiisg] = 0;
    partials[1][tiisg] = 0;
    partials[2][tiisg] = 0;
    partials[3][tiisg] = 0;
  }
  const float lane_sum00 = simd_sum(sum00);
  const float lane_sum01 = simd_sum(sum01);
  const float lane_sum10 = simd_sum(sum10);
  const float lane_sum11 = simd_sum(sum11);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiisg == 0) {
    partials[0][sgitg] = lane_sum00;
    partials[1][sgitg] = lane_sum01;
    partials[2][sgitg] = lane_sum10;
    partials[3][sgitg] = lane_sum11;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgitg == 0) {
    const float all_sum00 = simd_sum(partials[0][tiisg]);
    const float all_sum01 = simd_sum(partials[1][tiisg]);
    const float all_sum10 = simd_sum(partials[2][tiisg]);
    const float all_sum11 = simd_sum(partials[3][tiisg]);
    if (tiisg == 0) {
      const float scale = (float)scales[rb + 0];
      float value0 = all_sum00 * scale;
      float value1 = all_sum10 * scale;
)";
    if (fusedBias) {
      shader += R"(
      const float biasv = (float)bias[rb + 0];
      value0 += biasv;
      value1 += biasv;
)";
    }
    shader += R"(
      dst[rb + 0] = (real)value0;
      dst[nrows + rb + 0] = (real)value1;
    }
    if (tiisg == 1 && active1) {
      const float scale = (float)scales[rb + 1];
      float value0 = all_sum01 * scale;
      float value1 = all_sum11 * scale;
)";
    if (fusedBias) {
      shader += R"(
      const float biasv = (float)bias[rb + 1];
      value0 += biasv;
      value1 += biasv;
)";
    }
    shader += R"(
      dst[rb + 1] = (real)value0;
      dst[nrows + rb + 1] = (real)value1;
    }
  }
}
  )";
  } else {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

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
    sum0 += dot(float4(q0), yv0);
    sum0 += dot(float4(q0b), yv1);
    if (active1) {
      const char4 q1 = x1[i];
      const char4 q1b = x1[i + 1];
      sum1 += dot(float4(q1), yv0);
      sum1 += dot(float4(q1b), yv1);
    }
  }
  if (i < nvecs) {
    const float4 yv = float4(y4[i]);
    const char4 q0 = x0[i];
    sum0 += dot(float4(q0), yv);
    if (active1) {
      const char4 q1 = x1[i];
      sum1 += dot(float4(q1), yv);
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
  }
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
  if (format != 0) {
    defines += "constant uint group_size [[function_constant(3)]];";
    defines += "\n";
    defines += "constant uint groups_per_row [[function_constant(4)]];";
    defines += "\n";
  }
  return defines;
}
