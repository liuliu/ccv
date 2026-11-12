#include "SegmentedInt8SwiGLUKernel.hpp"
#include "SegmentedInt8GemvDescriptor.hpp"
#include "TMCQDecodeSource.hpp"
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
  shader += "constant uint2 iq2xxs_ksigns[128] = {";
  for (uint32_t i = 0; i < 128; ++i) {
    if (i != 0)
      shader += ",";
    shader += "\n  uint2(";
    for (uint32_t word = 0; word < 2; ++word) {
      uint32_t mask = 0;
      for (uint32_t lane = 0; lane < 4; ++lane)
        if (ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[i] & (1u << (word * 4 + lane)))
          mask |= 255u << (lane * 8);
      shader += std::to_string(mask) + (word == 0 ? "u, " : "u)");
    }
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

// IQ1_S / IQ1_M helpers. Like the IQ2 helpers, every packed-format kernel keeps
// its own copy of what it uses. The next time these need a change, factor the
// IQ1 and IQ2 logic out into shared code instead of editing each copy.
// iq1_unit_dot(unit, y0, ..., y7): the dot of a unit's 32 int8 values with 32
// activations, for one IQ1_S group or two IQ1_M groups.
static std::string createIQ1DotSource(const uint32_t format)
{
  std::string shader;
  shader += R"(
// The 64 bits starting `shift` (0...7) bits into p.
inline uint2 iq1_payload_bytes(device const uchar* p, const uint shift)
{
  const uint lo = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
  const uint hi = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24);
  // Shifting by 1 and then 31 - shift keeps every shift below 32, also for shift == 0.
  return uint2((lo >> shift) | ((hi << 1) << (31u - shift)), hi >> shift);
}

struct iq1_unit {
  uint4 index;
  uint shifts;
  uint2 m;
};
)";
  shader += (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) == CCV_NNC_QX_8I_ROWWISE_IQ1_S ? R"(
constant uint iq1s_multipliers[8] = {1, 2, 3, 5, 7, 9, 12, 16};

// Four 11-bit indices, three shift bits and a 3-bit multiplier code. The
// fourth shift is the parity of the first three (0x96 is the parity table).
inline iq1_unit iq1s_unit(const uint2 payload)
{
  const uint shifts = (payload.y >> 12) & 7u;
  iq1_unit unit;
  unit.index = uint4(payload.x & 2047u, (payload.x >> 11) & 2047u, (payload.x >> 22) | ((payload.y & 1u) << 10), (payload.y >> 1) & 2047u);
  unit.shifts = shifts | (((0x96u >> shifts) & 1u) << 3);
  unit.m = uint2(iq1s_multipliers[(payload.y >> 15) & 7u]);
  return unit;
}
)" : R"(
// Two 28-bit groups: per group two 12-bit (index, shift) codes and a 4-bit
// multiplier code m - 1.
inline iq1_unit iq1m_unit(const uint2 payload)
{
  iq1_unit unit;
  unit.index = uint4(payload.x & 2047u, (payload.x >> 12) & 2047u, (payload.x >> 28) | ((payload.y & 127u) << 4), (payload.y >> 8) & 2047u);
  unit.shifts = ((payload.x >> 11) & 1u) | ((payload.x >> 22) & 2u) | ((payload.y >> 5) & 4u) | ((payload.y >> 16) & 8u);
  unit.m = uint2(((payload.x >> 24) & 15u) + 1u, ((payload.y >> 20) & 15u) + 1u);
  return unit;
}
)";
  // Entry 2 * index + negative holds the eight half lanes v = 8 g + s, with
  // s = -1 for negative: v is one of -9, -7, -1, 1, 7, 9.
  shader += "constant uint4 iq1_grid_levels[4096] = {";
  for (uint32_t i = 0; i < 4096; ++i) {
    const uint16_t value = ccv_nnc_8i_rowwise_packed_iq1_grid[i >> 1];
    const int shift = (i & 1) ? -1 : 1;
    uint32_t words[4] = {0, 0, 0, 0};
    for (uint32_t j = 0; j < 8; ++j) {
      const int v = 8 * ((int)((value >> (j * 2)) & 3u) - 1) + shift;
      // Half bit patterns of the six levels.
      const uint32_t bits = v == -9 ? 0xc880u : v == -7 ? 0xc700u : v == -1 ? 0xbc00u : v == 1 ? 0x3c00u : v == 7 ? 0x4700u : 0x4880u;
      words[j / 2] |= bits << ((j % 2) * 16);
    }
    if (i != 0)
      shader += ",";
    if ((i % 4) == 0)
      shader += "\n  ";
    shader += "uint4(" + std::to_string(words[0]) + "u, " + std::to_string(words[1]) + "u, " + std::to_string(words[2]) + "u, " + std::to_string(words[3]) + "u)";
  }
  shader += "\n};\n";
  shader += R"(
inline float iq1_subgroup_dot(const uint index, const uint negative, const float4 y0, const float4 y1)
{
  const uint4 lanes = iq1_grid_levels[(index << 1) | negative];
  return dot(float4(as_type<half4>(lanes.xy)), y0) + dot(float4(as_type<half4>(lanes.zw)), y1);
}

// The sum of sign(v) * y over the lanes whose level is +/-9.
inline float iq1_subgroup_saturation_dot(const uint index, const uint negative, const float4 y0, const float4 y1)
{
  const uint4 lanes = iq1_grid_levels[(index << 1) | negative];
  const float4 v0 = float4(as_type<half4>(lanes.xy));
  const float4 v1 = float4(as_type<half4>(lanes.zw));
  return dot(select(float4(0.0f), sign(v0), abs(v0) > 8.0f), y0) + dot(select(float4(0.0f), sign(v1), abs(v1) > 8.0f), y1);
}
)";
  shader += R"(
// A codeword contributes m * sum(v * y) with v = 8 g + s. Where 9 m > 127
// the +/-9 levels saturate to +/-127, removing c = 9 m - 127 times
// sign(v) * y on those lanes. Saturating multipliers are rare.
inline float iq1_unit_dot(const iq1_unit unit,
  const float4 y0, const float4 y1, const float4 y2, const float4 y3,
  const float4 y4, const float4 y5, const float4 y6, const float4 y7)
{
  const uint4 negative = (uint4(unit.shifts) >> uint4(0, 1, 2, 3)) & 1u;
  const float4 d = float4(
    iq1_subgroup_dot(unit.index.x, negative.x, y0, y1),
    iq1_subgroup_dot(unit.index.y, negative.y, y2, y3),
    iq1_subgroup_dot(unit.index.z, negative.z, y4, y5),
    iq1_subgroup_dot(unit.index.w, negative.w, y6, y7));
  const float2 m = float2(unit.m);
  float result = m.x * (d.x + d.y) + m.y * (d.z + d.w);
  if (max(unit.m.x, unit.m.y) > 14u) {
    const float4 e = float4(
      iq1_subgroup_saturation_dot(unit.index.x, negative.x, y0, y1),
      iq1_subgroup_saturation_dot(unit.index.y, negative.y, y2, y3),
      iq1_subgroup_saturation_dot(unit.index.z, negative.z, y4, y5),
      iq1_subgroup_saturation_dot(unit.index.w, negative.w, y6, y7));
    const float2 c = max(9.0f * m - 127.0f, float2(0.0f));
    result -= c.x * (e.x + e.y) + c.y * (e.z + e.w);
  }
  return result;
}
)";
  return shader;
}

static void appendIQ1Decoder(std::string& shader, const uint32_t format)
{
  shader += createIQ1DotSource(format);
  const bool iq1s = format == CCV_NNC_QX_8I_ROWWISE_IQ1_S;
  shader += iq1s ? "constant uint unit_bits = 50;\n" : "constant uint unit_bits = 56;\n";
  shader += iq1s ? "#define IQ1_UNIT iq1s_unit\n" : "#define IQ1_UNIT iq1m_unit\n";
  shader += R"(
inline float2 dot_pair(
  device const uchar* gate_row, device const uchar* up_row,
  const uint bit_offset,
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
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  return float2(
    iq1_unit_dot(IQ1_UNIT(iq1_payload_bytes(gate_row + byte_offset, shift)),
      y0, y1, y2, y3, y4, y5, y6, y7),
    iq1_unit_dot(IQ1_UNIT(iq1_payload_bytes(up_row + byte_offset, shift)),
      y0, y1, y2, y3, y4, y5, y6, y7));
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

// Every packed format's routed rows are segmented_int8_swiglu_routes, so the plain kernel and the
// shared-expert kernel below work for all of them.
static void appendRoutesSignature(std::string& shader)
{
  shader += R"(
inline void segmented_int8_swiglu_routes(
  device const uchar* gate_weights,
  device const uchar* up_weights,
  device const real* activation,
  device real* destination,
  device const real* gate_scales,
  device const real* up_scales,
  device const int* indices,
  device const int* counts,
  device const real* route_weights,
  threadgroup int& route_expert,
  threadgroup float* route_scratch,
  uint threadgroup_position,
  uint simdgroup_index,
  uint lane)
{)";
}

static void appendSegmentedSwiGLUKernels(
  std::string& shader, const SegmentedInt8SwiGLUKernelDescriptor& descriptor)
{
  // TMCQ: gate and up partial sums of each K split, then the half2 pair table (512 floats' worth).
  const std::string scratch = segmentedTMCQRate(descriptor.format) ?
    "2 * TMCQ_K_SPLITS * 16 + 512" : "2 * 4 * 16";
  shader += R"(
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
  threadgroup int route_expert;
  // Cross-SIMD-group partial sums for the formats that split K (TMCQ, followed by its threadgroup
  // copy of the pair table); the others leave it unused.
  threadgroup float route_scratch[)" + scratch + R"(];
  segmented_int8_swiglu_routes(
    gate_weights, up_weights, activation, destination, gate_scales, up_scales,
    indices, counts, route_weights, route_expert, route_scratch, threadgroup_position,
    simdgroup_index, lane);
}
)";
  if (descriptor.sharedFused && descriptor.sharedFormat != 0) {
    // The shared expert's Q5_K or Q6_K SwiGLU, as Int8SwiGLUKernel computes it for one row: half
    // a SIMD group owns each weight row, eight rows per threadgroup. Keep the arithmetic the same,
    // so the fused rows match SWIGLU_FORWARD bit for bit.
    shader += "constant uint shared_nrows [[function_constant(9)]];\n";
    shader += "constant uint routed_threadgroups [[function_constant(10)]];\n";
    shader += "constexpr constant uint shared_rows_per_threadgroup = 8;\n";
    if (descriptor.sharedFormat == CCV_NNC_QX_8I_ROWWISE_Q5_K) {
      shader += R"(
// One 11-byte group: sixteen 5-bit values, a 3-bit multiplier and a 5-bit offset.
constant uint shared_unit_bytes = 11;

inline void shared_decode16(device const uchar* p, thread float4& w0, thread float4& w1, thread float4& w2, thread float4& w3)
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
      CCV_NNC_MFA_PRECONDITION(descriptor.sharedFormat == CCV_NNC_QX_8I_ROWWISE_Q6_K);
      shader += R"(
// Two 52-bit groups in 13 bytes: eight signed 6-bit values, a 2-bit multiplier and a 2-bit
// offset each.
constant uint shared_unit_bytes = 13;

inline float shared_q6_offset(const uint meta)
{
  const uint q = (meta >> 2) & 3u;
  return (q & 2u) ? (float)((int)q - 4) : (float)q;
}

inline void shared_decode16(device const uchar* p, thread float4& w0, thread float4& w1, thread float4& w2, thread float4& w3)
{
  const uint a0 = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
  const uint b0 = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16);
  const uint a1 = ((uint)p[6] >> 4) | ((uint)p[7] << 4) | ((uint)p[8] << 12) | ((uint)p[9] << 20) | ((uint)p[10] << 28);
  const uint b1 = ((uint)p[10] >> 4) | ((uint)p[11] << 4) | ((uint)p[12] << 12);
  const uint meta0 = (b0 >> 16) & 15u;
  const uint meta1 = (b1 >> 16) & 15u;
  const float4 m0 = float4((float)((meta0 & 3u) + 1));
  const float4 m1 = float4((float)((meta1 & 3u) + 1));
  const float4 o0 = float4(shared_q6_offset(meta0));
  const float4 o1 = float4(shared_q6_offset(meta1));
  w0 = fma(float4(int4(uint4(a0 << 26, a0 << 20, a0 << 14, a0 << 8)) >> 26), m0, o0);
  w1 = fma(float4(int4(uint4(a0 << 2, ((a0 >> 30) | (b0 << 2)) << 26, b0 << 22, b0 << 16)) >> 26), m0, o0);
  w2 = fma(float4(int4(uint4(a1 << 26, a1 << 20, a1 << 14, a1 << 8)) >> 26), m1, o1);
  w3 = fma(float4(int4(uint4(a1 << 2, ((a1 >> 30) | (b1 << 2)) << 26, b1 << 22, b1 << 16)) >> 26), m1, o1);
}
)";
    }
    shader += R"(
inline void segmented_int8_swiglu_shared_rows(
  device const uchar* gate_weights,
  device const uchar* up_weights,
  device const real* activation,
  device real* destination,
  device const real* gate_scales,
  device const real* up_scales,
  uint threadgroup_position,
  uint simdgroup_index,
  uint lane)
{
  const uint units_per_row = ncols / 16;
  const uint row = threadgroup_position * shared_rows_per_threadgroup + simdgroup_index * 2 + (lane >> 4);
  const uint unit_lane = lane & 15u;
  const ulong row_offset = (ulong)min(row, shared_nrows - 1) * (units_per_row * shared_unit_bytes);
  device const uchar* gate = gate_weights + row_offset;
  device const uchar* up = up_weights + row_offset;

  float gate_sum = 0;
  float up_sum = 0;
  for (uint u = unit_lane; u < units_per_row; u += 16) {
    float4 g0, g1, g2, g3;
    float4 u0, u1, u2, u3;
    shared_decode16(gate + u * shared_unit_bytes, g0, g1, g2, g3);
    shared_decode16(up + u * shared_unit_bytes, u0, u1, u2, u3);
    device const real4* y = (device const real4*)activation + u * 4;
    const float4 y0 = float4(y[0]);
    const float4 y1 = float4(y[1]);
    const float4 y2 = float4(y[2]);
    const float4 y3 = float4(y[3]);
    gate_sum += dot(g0, y0) + dot(g1, y1) + dot(g2, y2) + dot(g3, y3);
    up_sum += dot(u0, y0) + dot(u1, y1) + dot(u2, y2) + dot(u3, y3);
  }

  for (uint offset = 8; offset > 0; offset >>= 1) {
    gate_sum += simd_shuffle_xor(gate_sum, offset);
    up_sum += simd_shuffle_xor(up_sum, offset);
  }
  if (unit_lane == 0 && row < shared_nrows) {
    float gate = gate_sum * (float)gate_scales[row];
    float up = up_sum * (float)up_scales[row];
)";
    if (descriptor.clamp)
      shader += R"(
    gate = min(gate, clamp_limit);
    up = clamp(up, -clamp_limit, clamp_limit);
)";
    shader += R"(
    destination[row] = (real)(up * gate * stable_sigmoid(gate));
  }
}

kernel void segmented_int8_swiglu_shared(
  device const uchar* gate_weights [[buffer(0)]],
  device const uchar* up_weights [[buffer(1)]],
  device const real* activation [[buffer(2)]],
  device real* destination [[buffer(3)]],
  device const real* gate_scales [[buffer(4)]],
  device const real* up_scales [[buffer(5)]],
  device const int* indices [[buffer(6)]],
  device const int* counts [[buffer(7)]],
  device const real* route_weights [[buffer(8)]],
  device const uchar* shared_gate_weights [[buffer(9)]],
  device const uchar* shared_up_weights [[buffer(10)]],
  device const real* shared_activation [[buffer(11)]],
  device real* shared_destination [[buffer(12)]],
  device const real* shared_gate_scales [[buffer(13)]],
  device const real* shared_up_scales [[buffer(14)]],
  uint threadgroup_position [[threadgroup_position_in_grid]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]])
{
  threadgroup int route_expert;
  // Cross-SIMD-group partial sums for the formats that split K (TMCQ, followed by its threadgroup
  // copy of the pair table); the others leave it unused.
  threadgroup float route_scratch[)" + scratch + R"(];
  // Spread the shared threadgroups evenly through the grid, as the Q8_0 shared rows do.
  const uint shared_threadgroups = (shared_nrows + shared_rows_per_threadgroup - 1u) / shared_rows_per_threadgroup;
  const uint total = routed_threadgroups + shared_threadgroups;
  const uint shared_before = (uint)((ulong)threadgroup_position * shared_threadgroups / total);
  if ((uint)((ulong)(threadgroup_position + 1) * shared_threadgroups / total) != shared_before)
    segmented_int8_swiglu_shared_rows(
      shared_gate_weights, shared_up_weights, shared_activation, shared_destination,
      shared_gate_scales, shared_up_scales, shared_before, simdgroup_index, lane);
  else
    segmented_int8_swiglu_routes(
      gate_weights, up_weights, activation, destination, gate_scales, up_scales,
      indices, counts, route_weights, route_expert, route_scratch, threadgroup_position - shared_before,
      simdgroup_index, lane);
}
)";
  } else if (descriptor.sharedFused) {
    // The shared expert's int8 SwiGLU, two rows per threadgroup as in Int8SwiGLUKernel. Its
    // threadgroups interleave with the routed ones: the routed rows are ALU bound on the packed
    // decode and the shared rows bandwidth bound, so the GPU overlaps the two.
    shader += "constant uint shared_nrows [[function_constant(9)]];\n";
    shader += "constant uint routed_threadgroups [[function_constant(10)]];\n";
    shader += R"(
inline void segmented_int8_swiglu_shared_rows(
  device const char* gate_weights,
  device const char* up_weights,
  device const real* activation,
  device real* destination,
  device const real* gate_scales,
  device const real* up_scales,
  threadgroup float (&partials)[4][32],
  uint threadgroup_position,
  uint simdgroup_index,
  uint lane)
{
  constexpr uint rows_per_threadgroup = 2;
  constexpr uint simdgroups_per_threadgroup = 4;
  const uint row_base = threadgroup_position * rows_per_threadgroup;
  const bool active1 = row_base + 1 < shared_nrows;
  device const real4* activation4 = (device const real4*)activation;
  device const char4* gate0 =
    (device const char4*)(gate_weights + (ulong)(row_base + 0) * ncols);
  device const char4* gate1 =
    (device const char4*)(gate_weights + (ulong)(row_base + 1) * ncols);
  device const char4* up0 =
    (device const char4*)(up_weights + (ulong)(row_base + 0) * ncols);
  device const char4* up1 =
    (device const char4*)(up_weights + (ulong)(row_base + 1) * ncols);

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
      shader += R"(
      gate = min(gate, clamp_limit);
      up = clamp(up, -clamp_limit, clamp_limit);
)";
    shader += R"(
      destination[row_base + 0] = (real)(up * gate * stable_sigmoid(gate));
    }
    if (lane == 1 && active1) {
      float gate = all_gate1 * (float)gate_scales[row_base + 1];
      float up = all_up1 * (float)up_scales[row_base + 1];
)";
    if (descriptor.clamp)
      shader += R"(
      gate = min(gate, clamp_limit);
      up = clamp(up, -clamp_limit, clamp_limit);
)";
    shader += R"(
      destination[row_base + 1] = (real)(up * gate * stable_sigmoid(gate));
    }
  }
}

kernel void segmented_int8_swiglu_shared(
  device const uchar* gate_weights [[buffer(0)]],
  device const uchar* up_weights [[buffer(1)]],
  device const real* activation [[buffer(2)]],
  device real* destination [[buffer(3)]],
  device const real* gate_scales [[buffer(4)]],
  device const real* up_scales [[buffer(5)]],
  device const int* indices [[buffer(6)]],
  device const int* counts [[buffer(7)]],
  device const real* route_weights [[buffer(8)]],
  device const char* shared_gate_weights [[buffer(9)]],
  device const char* shared_up_weights [[buffer(10)]],
  device const real* shared_activation [[buffer(11)]],
  device real* shared_destination [[buffer(12)]],
  device const real* shared_gate_scales [[buffer(13)]],
  device const real* shared_up_scales [[buffer(14)]],
  uint threadgroup_position [[threadgroup_position_in_grid]],
  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]])
{
  threadgroup int route_expert;
  // Cross-SIMD-group partial sums for the formats that split K (TMCQ, followed by its threadgroup
  // copy of the pair table); the others leave it unused.
  threadgroup float route_scratch[)" + scratch + R"(];
  threadgroup float partials[4][32];
  // Spread the shared threadgroups evenly through the grid: grouping them at either end leaves
  // most of the routed work without bandwidth-bound neighbors.
  const uint shared_threadgroups = (shared_nrows + 1u) / 2u;
  const uint total = routed_threadgroups + shared_threadgroups;
  const uint shared_before = (uint)((ulong)threadgroup_position * shared_threadgroups / total);
  if ((uint)((ulong)(threadgroup_position + 1) * shared_threadgroups / total) != shared_before)
    segmented_int8_swiglu_shared_rows(
      shared_gate_weights, shared_up_weights, shared_activation, shared_destination,
      shared_gate_scales, shared_up_scales, partials, shared_before, simdgroup_index, lane);
  else
    segmented_int8_swiglu_routes(
      gate_weights, up_weights, activation, destination, gate_scales, up_scales,
      indices, counts, route_weights, route_expert, route_scratch, threadgroup_position - shared_before,
      simdgroup_index, lane);
}
)";
  }
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

)";
  appendRoutesSignature(shader);
  shader += R"(
  constexpr uint rows_per_simdgroup = 1;
  constexpr uint row_simdgroups = 4;
  constexpr uint rows_per_threadgroup = rows_per_simdgroup * row_simdgroups;
  const uint threadgroups_per_route =
    (nrows + rows_per_threadgroup - 1u) / rows_per_threadgroup;
  const uint route = threadgroup_position / threadgroups_per_route;
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

  // Keep this load before the projections: loading after the SIMD reductions
  // produced intermittent zero outputs with Metal shader validation in the
  // IQ2_XXS 257-expert case. Preserve this workaround across packed formats;
  // see "MPS segmented SwiGLU preserves route weights with 257 experts" in
  // test/int/nnc/segmented_swiglu.tests.c.
  const float route_weight = (float)route_weights[route];

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
  appendSegmentedSwiGLUKernels(shader, descriptor);
  return shader;
}

static std::string buildIQ1Shader(
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
  appendIQ1Decoder(shader, descriptor.format);
  shader += R"(

inline float stable_sigmoid(const float value)
{
  const float tail = 1.0f / (1.0f + exp(abs(value)));
  return tail + step(0.0f, value) * (1.0f - 2.0f * tail);
}

)";
  appendRoutesSignature(shader);
  shader += R"(
  constexpr uint rows_per_simdgroup = 1;
  constexpr uint row_simdgroups = 4;
  constexpr uint rows_per_threadgroup = rows_per_simdgroup * row_simdgroups;
  const uint threadgroups_per_route =
    (nrows + rows_per_threadgroup - 1u) / rows_per_threadgroup;
  const uint route = threadgroup_position / threadgroups_per_route;
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

  // Keep this load before the projections: loading after the SIMD reductions
  // produced intermittent zero outputs with Metal shader validation in the
  // IQ2_XXS 257-expert case. Preserve this workaround across packed formats;
  // see "MPS segmented SwiGLU preserves route weights with 257 experts" in
  // test/int/nnc/segmented_swiglu.tests.c.
  const float route_weight = (float)route_weights[route];

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
  // Each lane decodes 32 weights per step: one IQ1_S group or two IQ1_M groups.
  // K is a multiple of 256, so every packed row starts on a byte boundary.
  const uint units_per_row = ncols >> 5;
  const ulong row_offset = (ulong)row_base * (units_per_row * unit_bits / 8u);
  device const uchar* gate_row = gate_weights + row_offset;
  device const uchar* up_row = up_weights + row_offset;
  uint activation_base = lane * 8u;
  uint bit_offset = lane * unit_bits;
  for (uint unit = lane; unit < units_per_row; unit += 32u) {
    const float2 projected0 = dot_pair(
      gate_row, up_row, bit_offset, activation4, activation_base);
    gate_sum0 += projected0.x;
    up_sum0 += projected0.y;
    activation_base += 256u;
    bit_offset += 32u * unit_bits;
  }

  const float gate_row_sum0 = simd_sum(gate_sum0);
  const float up_row_sum0 = simd_sum(up_sum0);
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
  appendSegmentedSwiGLUKernels(shader, descriptor);
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

)";
  appendRoutesSignature(shader);
  shader += R"(
  constexpr uint rows_per_simdgroup = 1;
  constexpr uint row_simdgroups = 4;
  constexpr uint rows_per_threadgroup = rows_per_simdgroup * row_simdgroups;
  const uint threadgroups_per_route =
    (nrows + rows_per_threadgroup - 1u) / rows_per_threadgroup;
  const uint route = threadgroup_position / threadgroups_per_route;
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

  // Keep this load before the projections: loading after the SIMD reductions
  // produced intermittent zero outputs with Metal shader validation in the
  // IQ2_XXS 257-expert case. Preserve this workaround across packed formats;
  // see "MPS segmented SwiGLU preserves route weights with 257 experts" in
  // test/int/nnc/segmented_swiglu.tests.c.
  const float route_weight = (float)route_weights[route];

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
  appendSegmentedSwiGLUKernels(shader, descriptor);
  return shader;
}

// TMCQ routes: each threadgroup owns one 16-row tile row of the routed expert's gate and up weights,
// and its SIMD groups split K. In a SIMD group each run of eight lanes takes one tile of its K
// streams, every lane two adjacent rows of it, whose codes share one run of tile words; the eight
// lanes share each activation load. The codes come from a threadgroup half2 copy of the pair
// table. Gate and up each finish their whole dot product and take their row scale before the clamp
// and SiLU, as the other formats do.
static std::string buildTMCQShader(
  const SegmentedInt8SwiGLUKernelDescriptor& descriptor, const uint32_t rate)
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
  shader += ccv_nnc_mfa_tmcq_decode_source();
  shader += ccv_nnc_mfa_tmcq_w2_source();
  shader += "constant constexpr uint TMCQ_N = " + std::to_string(rate) + ";\n";
  shader += "constant constexpr uint TMCQ_K_SPLITS = " + std::to_string(kSegmentedInt8SwiGLUTMCQSIMDGroups) + ";\n";
  shader += "constant constexpr uint TMCQ_K_SIMDGROUPS = " +
    std::to_string(segmentedInt8SwiGLUTMCQSIMDGroups(descriptor.sharedFused)) + ";\n";
  shader += "constant constexpr uint TMCQ_TILE_WORDS = TMCQ_N / 2;\n";
  shader += R"(
inline float stable_sigmoid(const float value)
{
  const float tail = 1.0f / (1.0f + exp(abs(value)));
  return tail + step(0.0f, value) * (1.0f - 2.0f * tail);
}

)";
  appendRoutesSignature(shader);
  shader += R"(
  // One threadgroup per 16-row tile row. K is split k_splits ways and the partial sums add up in
  // route_scratch in split order. Each SIMD group takes splits simdgroup_index, + k_simdgroups, ...:
  // one with eight SIMD groups, two with the four of a fused shared expert, which keeps the routed
  // rows bit for bit the same either way.
  constexpr uint k_splits = TMCQ_K_SPLITS;
  constexpr uint k_simdgroups = TMCQ_K_SIMDGROUPS;
  const uint tile_rows = nrows / 16u;
  const uint threadgroups_per_route = tile_rows;
  const uint route = threadgroup_position / threadgroups_per_route;
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
  threadgroup half2* const table = (threadgroup half2*)(route_scratch + 2u * k_splits * 16u);
  tmcq_w2_load_table(table, simdgroup_index * 32u + lane, 32u * k_simdgroups);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int expert = route_expert;
  if (expert < 0 || expert >= (int)expert_count)
    return;

  // Keep this load before the projections: loading after the SIMD reductions
  // produced intermittent zero outputs with Metal shader validation in the
  // IQ2_XXS 257-expert case. Preserve this workaround across packed formats;
  // see "MPS segmented SwiGLU preserves route weights with 257 experts" in
  // test/int/nnc/segmented_swiglu.tests.c.
  const float route_weight = (float)route_weights[route];

  const uint tile_row = threadgroup_position - route * threadgroups_per_route;
  // An expert is nrows / 16 whole tile rows, so weight_expert_stride (bytes) is a multiple of the
  // 2n-byte tile and every tile starts on a word.
  const uint tiles_per_row = ncols / 16u;
  const ulong tile_row_word = (ulong)tile_row * tiles_per_row * TMCQ_TILE_WORDS;
  device const uint* gate_tiles =
    (device const uint*)(gate_weights + (ulong)expert * weight_expert_stride) + tile_row_word;
  device const uint* up_tiles =
    (device const uint*)(up_weights + (ulong)expert * weight_expert_stride) + tile_row_word;
  gate_scales += (ulong)expert * nrows;
  up_scales += (ulong)expert * nrows;
  activation += broadcast_input ? 0 : (ulong)route * ncols;
  destination += (ulong)route * nrows;
  device const real4* activation4 = (device const real4*)activation;

  // Each lane decodes two adjacent rows of a tile (one run of tile words), so eight lanes cover a
  // tile and a SIMD group four tiles of its K stream at once.
  constexpr uint rows_per_lane = 2;
  constexpr uint lanes_per_tile = 16u / rows_per_lane;
  constexpr uint tiles_per_simdgroup = 32u / lanes_per_tile;
  const uint r0 = (lane % lanes_per_tile) * rows_per_lane;
  for (uint split = simdgroup_index; split < k_splits; split += k_simdgroups) {
    float gate_sum[rows_per_lane] = {};
    float up_sum[rows_per_lane] = {};
    // A lane whose stream runs past the tile row (ncols % 512 != 0) adds nothing.
    for (uint tc = split * tiles_per_simdgroup + lane / lanes_per_tile; tc < tiles_per_row;
         tc += tiles_per_simdgroup * k_splits) {
      const ulong tile_word = (ulong)tc * TMCQ_TILE_WORDS;
      const float4 y0 = float4(activation4[tc * 4u + 0u]);
      const float4 y1 = float4(activation4[tc * 4u + 1u]);
      const float4 y2 = float4(activation4[tc * 4u + 2u]);
      const float4 y3 = float4(activation4[tc * 4u + 3u]);
      half4 g[4 * rows_per_lane];
      half4 u[4 * rows_per_lane];
      tmcq_w2_decode_rows<TMCQ_N, rows_per_lane>(gate_tiles + tile_word, r0, g, table);
      tmcq_w2_decode_rows<TMCQ_N, rows_per_lane>(up_tiles + tile_word, r0, u, table);
      #pragma clang loop unroll(full)
      for (uint rr = 0; rr < rows_per_lane; ++rr) {
        gate_sum[rr] += dot(float4(g[4 * rr + 0]), y0) + dot(float4(g[4 * rr + 1]), y1) +
          dot(float4(g[4 * rr + 2]), y2) + dot(float4(g[4 * rr + 3]), y3);
        up_sum[rr] += dot(float4(u[4 * rr + 0]), y0) + dot(float4(u[4 * rr + 1]), y1) +
          dot(float4(u[4 * rr + 2]), y2) + dot(float4(u[4 * rr + 3]), y3);
      }
    }
    #pragma clang loop unroll(full)
    for (uint rr = 0; rr < rows_per_lane; ++rr) {
      #pragma clang loop unroll(full)
      for (uint offset = lanes_per_tile; offset < 32u; offset <<= 1) {
        gate_sum[rr] += simd_shuffle_xor(gate_sum[rr], offset);
        up_sum[rr] += simd_shuffle_xor(up_sum[rr], offset);
      }
    }
    if (lane < lanes_per_tile) {
      #pragma clang loop unroll(full)
      for (uint rr = 0; rr < rows_per_lane; ++rr) {
        route_scratch[split * 16u + r0 + rr] = gate_sum[rr];
        route_scratch[(k_splits + split) * 16u + r0 + rr] = up_sum[rr];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_index == 0 && lane < 16) {
    const uint r = lane;
    float gate_sum = 0;
    float up_sum = 0;
    for (uint s = 0; s < k_splits; ++s) {
      gate_sum += route_scratch[s * 16u + r];
      up_sum += route_scratch[(k_splits + s) * 16u + r];
    }
    const uint row = tile_row * 16u + r;
    float gate0 = gate_sum * (float)gate_scales[row];
    float up0 = up_sum * (float)up_scales[row];
)";
  if (descriptor.clamp)
    shader += R"(
    gate0 = min(gate0, clamp_limit);
    up0 = clamp(up0, -clamp_limit, clamp_limit);
)";
  shader += R"(
    destination[row] = (real)(
      route_weight * up0 * gate0 * stable_sigmoid(gate0));
  }
}
)";
  appendSegmentedSwiGLUKernels(shader, descriptor);
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

)";
  appendRoutesSignature(shader);
  shader += R"(
  constexpr uint rows_per_simdgroup = 1;
  constexpr uint row_simdgroups = 4;
  constexpr uint rows_per_threadgroup = rows_per_simdgroup * row_simdgroups;
  const uint threadgroups_per_route =
    (nrows + rows_per_threadgroup - 1u) / rows_per_threadgroup;
  const uint route = threadgroup_position / threadgroups_per_route;
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

  // Keep this load before the projections: loading after the SIMD reductions
  // produced intermittent zero outputs with Metal shader validation in the
  // IQ2_XXS 257-expert case. Preserve this workaround across packed formats;
  // see "MPS segmented SwiGLU preserves route weights with 257 experts" in
  // test/int/nnc/segmented_swiglu.tests.c.
  const float route_weight = (float)route_weights[route];

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
  appendSegmentedSwiGLUKernels(shader, descriptor);
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
  if (descriptor.format == CCV_NNC_QX_8I_ROWWISE_IQ1_S ||
      descriptor.format == CCV_NNC_QX_8I_ROWWISE_IQ1_M) {
    source = buildIQ1Shader(descriptor);
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return;
  }
  if (const uint32_t rate = segmentedTMCQRate(descriptor.format)) {
    source = buildTMCQShader(descriptor, rate);
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
  const uint packed, const uint2 signs, const uint sign_lane)
{
  const uint mask = sign_lane == 0 ? signs.x : signs.y;
  // Positive magnitudes are 1...127, so negating each byte cannot carry between bytes.
  return float4(as_type<char4>((packed ^ mask) + (mask & 0x01010101u)));
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
  const uint2 signs0 = iq2xxs_ksigns[sign_codes & 127u];
  const uint2 signs1 = iq2xxs_ksigns[(sign_codes >> 7) & 127u];
  const uint2 signs2 = iq2xxs_ksigns[(sign_codes >> 14) & 127u];
  const uint2 signs3 = iq2xxs_ksigns[(sign_codes >> 21) & 127u];
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

)";
  appendRoutesSignature(shader);
  shader += R"(
  constexpr uint rows_per_simdgroup = 1;
  constexpr uint row_simdgroups = 4;
  constexpr uint rows_per_threadgroup = rows_per_simdgroup * row_simdgroups;
  const uint threadgroups_per_route =
    (nrows + rows_per_threadgroup - 1u) / rows_per_threadgroup;
  const uint route = threadgroup_position / threadgroups_per_route;
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

  // Keep this load before the projections: loading after the SIMD reductions
  // produced intermittent zero outputs with Metal shader validation in the
  // IQ2_XXS 257-expert case. Preserve this workaround across packed formats;
  // see "MPS segmented SwiGLU preserves route weights with 257 experts" in
  // test/int/nnc/segmented_swiglu.tests.c.
  const float route_weight = (float)route_weights[route];

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
  appendSegmentedSwiGLUKernels(shader, descriptor);

  source = std::move(shader);
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
