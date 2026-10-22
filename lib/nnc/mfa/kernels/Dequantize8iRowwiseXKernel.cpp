#include "Dequantize8iRowwiseXKernel.hpp"
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

static uint32_t compact_iq3s_grid_entry(const uint32_t value)
{
	uint32_t packed = 0;
	for (uint32_t lane = 0; lane < 4; ++lane) {
		const uint32_t v = (uint32_t)((value >> (lane * 8)) & 0xff);
		packed |= v << (lane * 4);
	}
	return packed;
}

static uint32_t compact_iq3xxs_grid_entry(const uint32_t value)
{
	uint32_t packed = 0;
	for (uint32_t lane = 0; lane < 4; ++lane) {
		const uint32_t v = (uint32_t)((value >> (lane * 8)) & 0xff);
		packed |= (v >> 2) << (lane * 4);
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


// IQ1_S / IQ1_M helpers. Like the IQ2 helpers, every packed-format kernel keeps
// its own copy of what it uses. The next time these need a change, factor the
// IQ1 and IQ2 logic out into shared code instead of editing each copy.
template<typename Transform>
static void append_iq1_grid(std::string& shader, const char* const type, const char* const name, Transform transform)
{
	shader += "constant ";
	shader += type;
	shader += " ";
	shader += name;
	shader += "[2048] = {";
	for (uint32_t i = 0; i < 2048; ++i) {
		if (i != 0)
			shader += ",";
		if ((i % 4) == 0)
			shader += "\n  ";
		shader += transform(ccv_nnc_8i_rowwise_packed_iq1_grid[i]);
	}
	shader += "\n};\n";
}

// decode_group(source, group_index, q8) writing the group's int8 values.
static std::string create_iq1_decode_group_source(const uint32_t format)
{
	std::string shader;
	append_iq1_grid(shader, "ushort", "iq1_grid", [](const uint16_t value) { return std::to_string(value); });
	shader += R"(
// The 64 bits starting `shift` (0...7) bits into p.
inline uint2 iq1_payload_bytes(device const uchar* p, const uint shift)
{
  const uint lo = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
  const uint hi = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24);
  // Shifting by 1 and then 31 - shift keeps every shift below 32, also for shift == 0.
  return uint2((lo >> shift) | ((hi << 1) << (31u - shift)), hi >> shift);
}

inline uint2 iq1_payload(device const uchar* data, const ulong bit)
{
  return iq1_payload_bytes(data + (bit >> 3), (uint)bit & 7u);
}
)";
	if ((format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) == CCV_NNC_QX_8I_ROWWISE_IQ1_S)
		shader += R"(
struct iq1_unit {
  uint4 index;
  uint shifts;
  uint2 m;
};

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
)";
	shader += R"(
inline void iq1_decode_subgroup(const uint index, const bool negative, const int m, thread int* q8)
{
  const uint packed = iq1_grid[index];
  const int shift = negative ? -1 : 1;
  for (uint j = 0; j < 8; ++j)
    q8[j] = clamp(m * (8 * ((int)((packed >> (j * 2u)) & 3u) - 1) + shift), -127, 127);
}
)";
	shader += (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) == CCV_NNC_QX_8I_ROWWISE_IQ1_S ? R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const iq1_unit unit = iq1s_unit(iq1_payload(source, group_index * 50ul));
  for (uint sg = 0; sg < 4; ++sg)
    iq1_decode_subgroup(unit.index[sg], (unit.shifts >> sg) & 1u, (int)unit.m.x, q8 + sg * 8);
}
)" : R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const uint payload = iq1_payload(source, group_index * 28ul).x;
  const int m = (int)((payload >> 24) & 15u) + 1;
  iq1_decode_subgroup(payload & 2047u, (payload >> 11) & 1u, m, q8);
  iq1_decode_subgroup((payload >> 12) & 2047u, (payload >> 23) & 1u, m, q8 + 8);
}
)";
	return shader;
}

// decode_store_group's helpers: iq1_store_unit_bytes for an IQ1_S group,
// iq1_pair_bytes for an IQ1_M group.
static std::string create_iq1_byte_source(const uint32_t format)
{
	std::string shader;
	// The trits of lanes 0-3 sit at bits 0, 8, 4, 12 and those of lanes 4-7 at
	// bits 2, 10, 6, 14, so two shift-ors spread them into the bytes of two words.
	append_iq1_grid(shader, "ushort", "iq1_grid_spread", [](const uint16_t value) {
		static constexpr uint32_t positions[8] = {0, 8, 4, 12, 2, 10, 6, 14};
		uint32_t spread = 0;
		for (uint32_t j = 0; j < 8; ++j)
			spread |= ((value >> (j * 2)) & 3u) << positions[j];
		return std::to_string(spread);
	});
	shader += R"(
inline uint2 iq1_grid_bytes(const uint index)
{
  const uint packed = iq1_grid_spread[index];
  return uint2((packed | (packed << 12)) & 0x03030303u, ((packed >> 2) | (packed << 10)) & 0x03030303u);
}

// Each byte of t holds a trit in 0...2, and the levels offset by 128 lie in
// 1...255. Below saturation (m <= 14) they are (s - 8) m + 8 m t, so one
// multiply-add per 16-bit half (whose two bytes sum below 2^16) expands two
// trits; flipping the top bits removes the offset. `base` is the offset lowest
// level in both bytes.
inline uint2 iq1_unsaturated_bytes(const uint2 t, const ushort base, const ushort step)
{
  return uint2(
    as_type<uint>(ushort2(base) + as_type<ushort2>(t.x) * step),
    as_type<uint>(ushort2(base) + as_type<ushort2>(t.y) * step)) ^ 0x80808080u;
}

// With saturation the levels L0, L1, L2 are no longer evenly spaced, so the
// t == 2 bytes add E = (L2 - L0) - 2 (L1 - L0). Every byte's final value is in
// range, so wrapping intermediate sums still pack exactly.
inline uint2 iq1_saturated_bytes(const uint2 t, const bool negative, const int m)
{
  const int shift = negative ? -1 : 1;
  const int low = clamp(m * (shift - 8), -127, 127);
  const int step = m * shift - low;
  const int extra = clamp(m * (shift + 8), -127, 127) - low - 2 * step;
  return (uint2((uint)(low + 128) * 0x01010101u) + t * (uint)step + ((t >> 1) & 0x01010101u) * (uint)extra) ^ 0x80808080u;
}
)";
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S)
		shader += R"(
// Writes a unit's 32 int8 weights as eight-byte codewords to destination[0...3],
// one at a time, which keeps few values live next to a matmul's accumulators.
template<typename Destination>
inline void iq1_store_unit_bytes(const iq1_unit unit, Destination destination)
{
  const uint2 low = 128u - 7u * unit.m;
  const uint2 low_negative = low - 2u * unit.m;
  const ushort2 step = ushort2(unit.m * 8u);
  uint2 t[4];
  #pragma clang loop unroll(full)
  for (uint sg = 0; sg < 4; ++sg) {
    const uint pair = sg >> 1;
    const uint base = ((unit.shifts >> sg) & 1u) ? low_negative[pair] : low[pair];
    t[sg] = iq1_grid_bytes(unit.index[sg]);
    destination[sg] = iq1_unsaturated_bytes(t[sg], (ushort)(base * 0x0101u), step[pair]);
  }
  // Rewrite only the codeword pairs whose multiplier saturates; they are rare,
  // so diverging SIMD lanes seldom wait on this.
  #pragma clang loop unroll(full)
  for (uint pair = 0; pair < 2; ++pair)
    if (unit.m[pair] > 14u) {
      destination[pair * 2u] = iq1_saturated_bytes(t[pair * 2u], (unit.shifts >> (pair * 2u)) & 1u, (int)unit.m[pair]);
      destination[pair * 2u + 1u] = iq1_saturated_bytes(t[pair * 2u + 1u], (unit.shifts >> (pair * 2u + 1u)) & 1u, (int)unit.m[pair]);
    }
}
)";
	else
		shader += R"(
// Sixteen int8 weights of two codewords sharing multiplier m.
inline uint4 iq1_pair_bytes(const uint index0, const uint index1, const uint shifts, const uint m)
{
  const uint2 t0 = iq1_grid_bytes(index0);
  const uint2 t1 = iq1_grid_bytes(index1);
  if (m <= 14u) {
    // s = -1 lowers every level by 2 m.
    const uint low = 128u - 7u * m;
    const ushort step = (ushort)(m * 8u);
    return uint4(
      iq1_unsaturated_bytes(t0, (ushort)(((shifts & 1u) ? low - 2u * m : low) * 0x0101u), step),
      iq1_unsaturated_bytes(t1, (ushort)(((shifts & 2u) ? low - 2u * m : low) * 0x0101u), step));
  }
  return uint4(iq1_saturated_bytes(t0, shifts & 1u, (int)m), iq1_saturated_bytes(t1, shifts & 2u, (int)m));
}
)";
	return shader;
}
}

Dequantize8iRowwiseXKernel::Dequantize8iRowwiseXKernel(Dequantize8iRowwiseXKernelDescriptor descriptor, MTL::Device* const device)
{
	format = descriptor.format;
	source = createSource();
	threadgroupSize = MTL::Size(256, 1, 1);

	auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(string, nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}

MTL::Size Dequantize8iRowwiseXKernel::gridSize(uint32_t dispatchItems) const noexcept
{
	return MTL::Size((dispatchItems + 255) / 256, 1, 1);
}

std::string Dequantize8iRowwiseXKernel::createDecodeGroupSource(const uint32_t format) noexcept
{
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S || format == CCV_NNC_QX_8I_ROWWISE_IQ1_M)
		return create_iq1_decode_group_source(format);
	std::string shader;
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
		append_compact_grid(shader, "iq2xxs_grid", ccv_nnc_8i_rowwise_packed_iq2xxs_grid, 256, [](const uint16_t value) { return (uint32_t)value; });
		append_compact_grid(shader, "iq2xxs_ksigns", ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns, 128, [](const uint8_t value) { return (uint32_t)value; });
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
		append_compact_grid(shader, "iq2s_grid", ccv_nnc_8i_rowwise_packed_iq2s_grid, 1024, compact_iq2_grid_entry);
	else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS)
		append_compact_grid(shader, "iq2xs_grid", ccv_nnc_8i_rowwise_packed_iq2xs_grid, 512, compact_iq2_grid_entry);
	else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S)
		append_compact_grid(shader, "iq3s_grid", ccv_nnc_8i_rowwise_packed_iq3s_grid, 512, compact_iq3s_grid_entry);
	else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)
		append_compact_grid(shader, "iq3xxs_grid", ccv_nnc_8i_rowwise_packed_iq3xxs_grid, 256, compact_iq3xxs_grid_entry);

	if (format == CCV_NNC_QX_8I_ROWWISE_Q6_K) {
		shader += R"(
inline ulong packed52_payload(device const uchar* data, const ulong group_index)
{
  const ulong bit_offset = (ulong)group_index * 52ul;
  const ulong byte_offset = bit_offset >> 3;
  const uint shift = (uint)(bit_offset & 7ul);
  const ulong value =
    (ulong)data[byte_offset] |
    ((ulong)data[byte_offset + 1] << 8) |
    ((ulong)data[byte_offset + 2] << 16) |
    ((ulong)data[byte_offset + 3] << 24) |
    ((ulong)data[byte_offset + 4] << 32) |
    ((ulong)data[byte_offset + 5] << 40) |
    ((ulong)data[byte_offset + 6] << 48);
  return value >> shift;
}
)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K || format == CCV_NNC_QX_8I_ROWWISE_IQ2_S) {
			shader += R"(
inline ulong packed42_payload(device const uchar* data, const ulong group_index)
{
  const ulong bit_offset = (ulong)group_index * 42ul;
  const ulong byte_offset = bit_offset >> 3;
  const uint shift = (uint)(bit_offset & 7ul);
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
inline uint packed21_payload(device const uchar* data, const ulong group_index)
{
  const ulong bit_offset = (ulong)group_index * 21ul;
  const ulong byte_offset = bit_offset >> 3;
  const uint shift = (uint)(bit_offset & 7ul);
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
inline uint packed28_payload(device const uchar* data, const ulong group_index)
{
  const ulong bit_offset = (ulong)group_index * 28ul;
  const ulong byte_offset = bit_offset >> 3;
  const uint shift = (uint)(bit_offset & 7ul);
  const uint value =
    (uint)data[byte_offset] |
    ((uint)data[byte_offset + 1] << 8) |
    ((uint)data[byte_offset + 2] << 16) |
    ((uint)data[byte_offset + 3] << 24);
  return value >> shift;
}
)";
	}

	if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS || format == CCV_NNC_QX_8I_ROWWISE_IQ2_S || format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
		shader += R"(
inline int iq2_value(constant uint* grid, const uint index, const uint lane)
{
  return (int)((((grid[index] >> (lane * 2)) & 3u) << 1) + 1u);
}
)";
	}
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S) {
		shader += R"(
inline int iq3s_value(const uint index, const uint lane)
{
  return (int)((iq3s_grid[index] >> (lane * 4)) & 15u);
}
)";
	}
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
		shader += R"(
inline int iq3xxs_value(const uint index, const uint lane)
{
  return (int)((iq3xxs_grid[index] >> (lane * 4)) & 15u);
}
)";
	}

	switch (format) {
			case CCV_NNC_QX_8I_ROWWISE_Q5_K:
				shader += R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 11ul;
  const ulong lo =
    (ulong)p[0] |
    ((ulong)p[1] << 8) |
    ((ulong)p[2] << 16) |
    ((ulong)p[3] << 24) |
    ((ulong)p[4] << 32) |
    ((ulong)p[5] << 40) |
    ((ulong)p[6] << 48) |
    ((ulong)p[7] << 56);
  const uint hi =
    (uint)p[8] |
    ((uint)p[9] << 8) |
    ((uint)p[10] << 16);
  const int m = (int)((hi >> 16) & 7u) + 1;
  const int b = (int)((hi >> 19) & 31u) - 16;
  for (uint j = 0; j < 12; ++j)
    q8[j] = ((int)((lo >> (j * 5)) & 31ul) - 16) * m + b;
  q8[12] = ((int)(((lo >> 60) | ((ulong)hi << 4)) & 31ul) - 16) * m + b;
  q8[13] = ((int)((hi >> 1) & 31u) - 16) * m + b;
  q8[14] = ((int)((hi >> 6) & 31u) - 16) * m + b;
  q8[15] = ((int)((hi >> 11) & 31u) - 16) * m + b;
}
)";
				break;
			case CCV_NNC_QX_8I_ROWWISE_Q6_K:
				shader += R"(
inline int q6_signed_value(const uint q)
{
  return (q & 32u) ? (int)q - 64 : (int)q;
}

inline int q2_signed_value(const uint q)
{
  return (q & 2u) ? (int)q - 4 : (int)q;
}

inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const ulong payload = packed52_payload(source, group_index);
  const int m = (int)((payload >> 48u) & 3ul) + 1;
  const int b = q2_signed_value((uint)((payload >> 50u) & 3ul));
  for (uint j = 0; j < 8; ++j)
    q8[j] = q6_signed_value((uint)((payload >> (j * 6u)) & 63ul)) * m + b;
}
)";
				break;
			case CCV_NNC_QX_8I_ROWWISE_Q4_K:
				shader += R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 9ul;
  const int m = (int)(p[8] & 15u) + 1;
  const int b = (int)(p[8] >> 4) - 8;
  for (uint j = 0; j < 8; ++j) {
    const uint q = p[j];
    q8[j * 2] = ((int)(q & 15u) - 8) * m + b;
    q8[j * 2 + 1] = ((int)(q >> 4) - 8) * m + b;
  }
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
			shader += R"(
inline void q3_values(const uint q, const uint base, const int m, const int b, thread int* q8)
{
  q8[base + 0] = ((int)(q & 7u) - 4) * m + b;
  q8[base + 1] = ((int)((q >> 3) & 7u) - 4) * m + b;
  q8[base + 2] = ((int)((q >> 6) & 7u) - 4) * m + b;
  q8[base + 3] = ((int)((q >> 9) & 7u) - 4) * m + b;
}

inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 7ul;
  const uint lo =
    (uint)p[0] |
    ((uint)p[1] << 8) |
    ((uint)p[2] << 16) |
    ((uint)p[3] << 24);
  const uint hi =
    (uint)p[4] |
    ((uint)p[5] << 8) |
    ((uint)p[6] << 16);
  const int m = (int)((hi >> 16) & 31u) + 1;
  const int b = (((int)((hi >> 21) & 7u) - 4) << 1);
  q3_values(lo & 0xfffu, 0, m, b, q8);
  q3_values((lo >> 12) & 0xfffu, 4, m, b, q8);
  q3_values(((lo >> 24) | ((hi & 15u) << 8)) & 0xfffu, 8, m, b, q8);
  q3_values((hi >> 4) & 0xfffu, 12, m, b, q8);
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
			shader += R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const ulong payload = packed42_payload(source, group_index);
  const uint q = (uint)payload;
  const int m = (int)((payload >> 32) & 63u) + 1;
  const int z = (int)((payload >> 38) & 15u) << 3;
  for (uint j = 0; j < 16; ++j)
    q8[j] = (int)((q >> (j * 2)) & 3u) * m - z;
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			shader += R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const ulong payload = packed42_payload(source, group_index);
  const uint grid0 = (uint)(payload & 1023u);
  const uint grid1 = (uint)((payload >> 10) & 1023u);
  const uint signs = (uint)((payload >> 20) & 65535u);
  const int scale = (int)((payload >> 36) & 63u) + 1;
  for (uint j = 0; j < 8; ++j) {
    const int mag0 = min(iq2_value(iq2s_grid, grid0, j) * scale, 127);
    const int mag1 = min(iq2_value(iq2s_grid, grid1, j) * scale, 127);
    q8[j] = (signs & (1u << j)) ? -mag0 : mag0;
    q8[8 + j] = (signs & (1u << (8 + j))) ? -mag1 : mag1;
  }
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			shader += R"(
constant int q2_xs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};

inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const uint payload = packed21_payload(source, group_index);
  const uint grid0 = payload & 511u;
  const uint signs = (payload >> 9) & 255u;
  const int scale = q2_xs_scales[(payload >> 17) & 15u];
  for (uint j = 0; j < 8; ++j) {
    const int mag = min(iq2_value(iq2xs_grid, grid0, j) * scale, 127);
    q8[j] = (signs & (1u << j)) ? -mag : mag;
  }
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			shader += R"(
constant int q2_xxs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};

inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 8ul;
  const uint sign_codes = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | (((uint)p[7] & 15u) << 24);
  const int scale = q2_xxs_scales[p[7] >> 4];
  for (uint sg = 0; sg < 4; ++sg) {
    const uint grid = p[sg];
    const uint signs = iq2xxs_ksigns[(sign_codes >> (sg * 7u)) & 127u];
    for (uint j = 0; j < 8; ++j) {
      const uint lane = sg * 8u + j;
      const int mag = min(iq2_value(iq2xxs_grid, grid, j) * scale, 127);
      q8[lane] = (signs & (1u << j)) ? -mag : mag;
    }
  }
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			shader += R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 7ul;
  const uint grid0 = (uint)p[0] | (((uint)p[1] & 1u) << 8);
  const uint grid1 = ((uint)p[1] >> 1) | (((uint)p[2] & 3u) << 7);
  const uint grid2 = ((uint)p[2] >> 2) | (((uint)p[3] & 7u) << 6);
  const uint grid3 = ((uint)p[3] >> 3) | (((uint)p[4] & 15u) << 5);
  const uint signs = ((uint)p[4] >> 4) | ((uint)p[5] << 4) | (((uint)p[6] & 15u) << 12);
  const int scale = (int)(p[6] >> 4) + 1;
  for (uint sg = 0; sg < 4; ++sg) {
    const uint grid = sg == 0 ? grid0 : (sg == 1 ? grid1 : (sg == 2 ? grid2 : grid3));
    for (uint j = 0; j < 4; ++j) {
      const uint lane = sg * 4 + j;
      const int mag = min(iq3s_value(grid, j) * scale, 127);
      q8[lane] = (signs & (1u << lane)) ? -mag : mag;
    }
  }
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			shader += R"(
inline void decode_group(device const uchar* source, const ulong group_index, thread int* q8)
{
  const uint payload = packed28_payload(source, group_index);
  const uint grid0 = payload & 255u;
  const uint grid1 = (payload >> 8) & 255u;
  const uint signs = (payload >> 16) & 255u;
  const int scale = (int)((payload >> 24) & 15u) + 1;
  for (uint j = 0; j < 4; ++j) {
    const int mag0 = min(iq3xxs_value(grid0, j) * scale, 127);
    const int mag1 = min(iq3xxs_value(grid1, j) * scale, 127);
    q8[j] = (signs & (1u << j)) ? -mag0 : mag0;
    q8[4 + j] = (signs & (1u << (4 + j))) ? -mag1 : mag1;
  }
}
)";
			break;
		default:
			break;
	}
	return shader;
}

std::string Dequantize8iRowwiseXKernel::createSource() const noexcept
{
	std::string shader = createConstants() + "\n";
	shader += R"(
#include <metal_stdlib>
using namespace metal;
)";
	shader += createDecodeGroupSource(format);

	shader += R"(
inline char4 q8_char4(const int4 v)
{
  return char4((char)v.x, (char)v.y, (char)v.z, (char)v.w);
}

inline void store_q8_scalar(device uchar* destination, const uint destination_offset, const uint col_base, thread int* q8)
{
  device int8_t* destination_q = reinterpret_cast<device int8_t*>(destination);
  for (uint j = 0; j < group_size; ++j) {
    const uint col = col_base + j;
    if (col < row_length)
      destination_q[destination_offset + j] = (int8_t)q8[j];
  }
}

)";
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS || format == CCV_NNC_QX_8I_ROWWISE_IQ2_S || format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
		shader += R"(
inline int4 signed_iq2_values_i4(constant uint* grid, const uint index, const uint code_lane, const uint signs, const uint sign_lane, const int scale)
{
  const uint packed = grid[index];
  const int4 mag = min(int4(
    (int)((((packed >> ((code_lane + 0u) * 2u)) & 3u) << 1u) + 1u),
    (int)((((packed >> ((code_lane + 1u) * 2u)) & 3u) << 1u) + 1u),
    (int)((((packed >> ((code_lane + 2u) * 2u)) & 3u) << 1u) + 1u),
    (int)((((packed >> ((code_lane + 3u) * 2u)) & 3u) << 1u) + 1u)) * int4(scale), int4(127));
  return int4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}

)";
	}
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S) {
		shader += R"(
inline int4 signed_iq3s_values_i4(const uint index, const uint signs, const uint sign_lane, const int scale)
{
  const uint packed = iq3s_grid[index];
  const int4 mag = min(int4(
    (int)((packed >> 0u) & 15u),
    (int)((packed >> 4u) & 15u),
    (int)((packed >> 8u) & 15u),
    (int)((packed >> 12u) & 15u)) * int4(scale), int4(127));
  return int4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}

)";
	}
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
		shader += R"(
inline int4 signed_iq3xxs_values_i4(const uint index, const uint signs, const uint sign_lane, const int scale)
{
  const uint packed = iq3xxs_grid[index];
  const int4 mag = min(int4(
    (int)((packed >> 0u) & 15u),
    (int)((packed >> 4u) & 15u),
    (int)((packed >> 8u) & 15u),
    (int)((packed >> 12u) & 15u)) * int4(scale), int4(127));
  return int4(
    (signs & (1u << (sign_lane + 0u))) ? -mag.x : mag.x,
    (signs & (1u << (sign_lane + 1u))) ? -mag.y : mag.y,
    (signs & (1u << (sign_lane + 2u))) ? -mag.z : mag.z,
    (signs & (1u << (sign_lane + 3u))) ? -mag.w : mag.w);
}

	)";
		}
		if (format == CCV_NNC_QX_8I_ROWWISE_Q5_K) {
			shader += R"(
inline int4 q5_values(const ulong lo, const uint lane, const int m, const int b)
{
  return (int4(
    (int)((lo >> ((lane + 0u) * 5u)) & 31ul),
    (int)((lo >> ((lane + 1u) * 5u)) & 31ul),
    (int)((lo >> ((lane + 2u) * 5u)) & 31ul),
    (int)((lo >> ((lane + 3u) * 5u)) & 31ul)) - int4(16)) * int4(m) + int4(b);
}

inline int4 q5_tail_values(const ulong lo, const uint hi, const int m, const int b)
{
  return (int4(
    (int)(((lo >> 60u) | ((ulong)hi << 4u)) & 31ul),
    (int)((hi >> 1u) & 31u),
    (int)((hi >> 6u) & 31u),
    (int)((hi >> 11u) & 31u)) - int4(16)) * int4(m) + int4(b);
}

inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 11ul;
  const ulong lo =
    (ulong)p[0] |
    ((ulong)p[1] << 8) |
    ((ulong)p[2] << 16) |
    ((ulong)p[3] << 24) |
    ((ulong)p[4] << 32) |
    ((ulong)p[5] << 40) |
    ((ulong)p[6] << 48) |
    ((ulong)p[7] << 56);
  const uint hi =
    (uint)p[8] |
    ((uint)p[9] << 8) |
    ((uint)p[10] << 16);
  const int m = (int)((hi >> 16) & 7u) + 1;
  const int b = (int)((hi >> 19) & 31u) - 16;
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(q5_values(lo, 0, m, b));
    destination4[1] = q8_char4(q5_values(lo, 4, m, b));
    destination4[2] = q8_char4(q5_values(lo, 8, m, b));
    destination4[3] = q8_char4(q5_tail_values(lo, hi, m, b));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

	)";
		} else if (format == CCV_NNC_QX_8I_ROWWISE_Q6_K) {
			shader += R"(
inline int4 q6_values_i4(const ulong payload, const uint lane, const int m, const int b)
{
  return int4(
    q6_signed_value((uint)((payload >> ((lane + 0u) * 6u)) & 63ul)),
    q6_signed_value((uint)((payload >> ((lane + 1u) * 6u)) & 63ul)),
    q6_signed_value((uint)((payload >> ((lane + 2u) * 6u)) & 63ul)),
    q6_signed_value((uint)((payload >> ((lane + 3u) * 6u)) & 63ul))) * int4(m) + int4(b);
}

inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  const ulong payload = packed52_payload(source, group_index);
  const int m = (int)((payload >> 48u) & 3ul) + 1;
  const int b = q2_signed_value((uint)((payload >> 50u) & 3ul));
  if (((row_length & 3u) == 0u) && col_base + 8u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(q6_values_i4(payload, 0, m, b));
    destination4[1] = q8_char4(q6_values_i4(payload, 4, m, b));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

	)";
		} else if (format == CCV_NNC_QX_8I_ROWWISE_Q4_K) {
			shader += R"(
inline int4 q4_values(device const uchar* p, const uint offset, const int m, const int b)
{
  const uint q0 = p[offset];
  const uint q1 = p[offset + 1];
  return (int4(
    (int)(q0 & 15u),
    (int)(q0 >> 4),
    (int)(q1 & 15u),
    (int)(q1 >> 4)) - int4(8)) * int4(m) + int4(b);
}

inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 9ul;
  const int m = (int)(p[8] & 15u) + 1;
  const int b = (int)(p[8] >> 4) - 8;
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(q4_values(p, 0, m, b));
    destination4[1] = q8_char4(q4_values(p, 2, m, b));
    destination4[2] = q8_char4(q4_values(p, 4, m, b));
    destination4[3] = q8_char4(q4_values(p, 6, m, b));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_Q3_K) {
		shader += R"(
inline int4 q3_values_i4(const uint q, const int m, const int b)
{
  return (int4(
    (int)((q >> 0u) & 7u),
    (int)((q >> 3u) & 7u),
    (int)((q >> 6u) & 7u),
    (int)((q >> 9u) & 7u)) - int4(4)) * int4(m) + int4(b);
}

inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 7ul;
  const uint lo =
    (uint)p[0] |
    ((uint)p[1] << 8) |
    ((uint)p[2] << 16) |
    ((uint)p[3] << 24);
  const uint hi =
    (uint)p[4] |
    ((uint)p[5] << 8) |
    ((uint)p[6] << 16);
  const int m = (int)((hi >> 16) & 31u) + 1;
  const int b = (((int)((hi >> 21) & 7u) - 4) << 1);
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(q3_values_i4(lo & 0xfffu, m, b));
    destination4[1] = q8_char4(q3_values_i4((lo >> 12u) & 0xfffu, m, b));
    destination4[2] = q8_char4(q3_values_i4(((lo >> 24u) | ((hi & 15u) << 8u)) & 0xfffu, m, b));
    destination4[3] = q8_char4(q3_values_i4((hi >> 4u) & 0xfffu, m, b));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K) {
		shader += R"(
inline int4 q2_values_i4(const uint q, const uint lane, const int m, const int z)
{
  return int4(
    (int)((q >> ((lane + 0u) * 2u)) & 3u),
    (int)((q >> ((lane + 1u) * 2u)) & 3u),
    (int)((q >> ((lane + 2u) * 2u)) & 3u),
    (int)((q >> ((lane + 3u) * 2u)) & 3u)) * int4(m) - int4(z);
}

inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  const ulong payload = packed42_payload(source, group_index);
  const uint q = (uint)payload;
  const int m = (int)((payload >> 32) & 63u) + 1;
  const int z = (int)((payload >> 38) & 15u) << 3;
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(q2_values_i4(q, 0, m, z));
    destination4[1] = q8_char4(q2_values_i4(q, 4, m, z));
    destination4[2] = q8_char4(q2_values_i4(q, 8, m, z));
    destination4[3] = q8_char4(q2_values_i4(q, 12, m, z));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 8ul;
  const uint sign_codes = (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | (((uint)p[7] & 15u) << 24);
  const int scale = q2_xxs_scales[p[7] >> 4];
  if (((row_length & 3u) == 0u) && col_base + 32u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    for (uint sg = 0; sg < 4; ++sg) {
      const uint signs = iq2xxs_ksigns[(sign_codes >> (sg * 7u)) & 127u];
      destination4[sg * 2u] = q8_char4(signed_iq2_values_i4(iq2xxs_grid, p[sg], 0, signs, 0, scale));
      destination4[sg * 2u + 1u] = q8_char4(signed_iq2_values_i4(iq2xxs_grid, p[sg], 4, signs, 4, scale));
    }
  } else {
    int q8[32] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S) {
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  const ulong payload = packed42_payload(source, group_index);
  const uint grid0 = (uint)(payload & 1023u);
  const uint grid1 = (uint)((payload >> 10) & 1023u);
  const uint signs = (uint)((payload >> 20) & 65535u);
  const int scale = (int)((payload >> 36) & 63u) + 1;
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(signed_iq2_values_i4(iq2s_grid, grid0, 0, signs, 0, scale));
    destination4[1] = q8_char4(signed_iq2_values_i4(iq2s_grid, grid0, 4, signs, 4, scale));
    destination4[2] = q8_char4(signed_iq2_values_i4(iq2s_grid, grid1, 0, signs, 8, scale));
    destination4[3] = q8_char4(signed_iq2_values_i4(iq2s_grid, grid1, 4, signs, 12, scale));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  const uint payload = packed21_payload(source, group_index);
  const uint grid0 = payload & 511u;
  const uint signs = (payload >> 9) & 255u;
  const int scale = q2_xs_scales[(payload >> 17) & 15u];
  if (((row_length & 3u) == 0u) && col_base + 8u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(signed_iq2_values_i4(iq2xs_grid, grid0, 0, signs, 0, scale));
    destination4[1] = q8_char4(signed_iq2_values_i4(iq2xs_grid, grid0, 4, signs, 4, scale));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_S) {
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 7ul;
  const uint grid0 = (uint)p[0] | (((uint)p[1] & 1u) << 8);
  const uint grid1 = ((uint)p[1] >> 1) | (((uint)p[2] & 3u) << 7);
  const uint grid2 = ((uint)p[2] >> 2) | (((uint)p[3] & 7u) << 6);
  const uint grid3 = ((uint)p[3] >> 3) | (((uint)p[4] & 15u) << 5);
  const uint signs = ((uint)p[4] >> 4) | ((uint)p[5] << 4) | (((uint)p[6] & 15u) << 12);
  const int scale = (int)(p[6] >> 4) + 1;
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(signed_iq3s_values_i4(grid0, signs, 0, scale));
    destination4[1] = q8_char4(signed_iq3s_values_i4(grid1, signs, 4, scale));
    destination4[2] = q8_char4(signed_iq3s_values_i4(grid2, signs, 8, scale));
    destination4[3] = q8_char4(signed_iq3s_values_i4(grid3, signs, 12, scale));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  const uint payload = packed28_payload(source, group_index);
  const uint grid0 = payload & 255u;
  const uint grid1 = (payload >> 8) & 255u;
  const uint signs = (payload >> 16) & 255u;
  const int scale = (int)((payload >> 24) & 15u) + 1;
  if (((row_length & 3u) == 0u) && col_base + 8u <= row_length) {
    device char4* destination4 = reinterpret_cast<device char4*>(destination + destination_offset);
    destination4[0] = q8_char4(signed_iq3xxs_values_i4(grid0, signs, 0, scale));
    destination4[1] = q8_char4(signed_iq3xxs_values_i4(grid1, signs, 4, scale));
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S) {
		shader += create_iq1_byte_source(format);
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  if (((row_length & 3u) == 0u) && col_base + 32u <= row_length) {
    uint2 words[4];
    iq1_store_unit_bytes(iq1s_unit(iq1_payload(source, group_index * 50ul)), words);
    device uint* destination4 = reinterpret_cast<device uint*>(destination + destination_offset);
    for (uint j = 0; j < 4; ++j) {
      destination4[j * 2u] = words[j].x;
      destination4[j * 2u + 1u] = words[j].y;
    }
  } else {
    int q8[32] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_M) {
		shader += create_iq1_byte_source(format);
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  if (((row_length & 3u) == 0u) && col_base + 16u <= row_length) {
    const uint payload = iq1_payload(source, group_index * 28ul).x;
    const uint4 words = iq1_pair_bytes(payload & 2047u, (payload >> 12) & 2047u,
      ((payload >> 11) & 1u) | ((payload >> 22) & 2u), ((payload >> 24) & 15u) + 1u);
    device uint* destination4 = reinterpret_cast<device uint*>(destination + destination_offset);
    for (uint j = 0; j < 4; ++j)
      destination4[j] = words[j];
  } else {
    int q8[16] = {0};
    decode_group(source, group_index, q8);
    store_q8_scalar(destination, destination_offset, col_base, q8);
  }
}

)";
	} else {
		shader += R"(
inline void decode_store_group(device const uchar* source, const ulong group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
)";
		if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS)
			shader += "  int q8[32] = {0};\n";
		else
			shader += "  int q8[16] = {0};\n";
		shader += R"(
  decode_group(source, group_index, q8);
  store_q8_scalar(destination, destination_offset, col_base, q8);
}

)";
	}

	shader += R"(
kernel void dequantize_8i_rowwise_x(
  device const uchar* source [[buffer(0)]],
  device uchar* destination [[buffer(1)]],
  device const uchar* input_scales [[buffer(2)]],
  device uchar* output_scales [[buffer(3)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  if (x >= dispatch_items)
    return;
  if (x < total_groups) {
    const uint row = x / groups_per_row;
    const uint group = x - row * groups_per_row;
    const uint col_base = group * group_size;
    decode_store_group(source, x, destination, row * row_length + col_base, col_base);
  }
  if (x < scale_bytes)
    output_scales[x] = input_scales[x];
}

kernel void dequantize_8i_rowwise_x_selected(
  device const uchar* source [[buffer(0)]],
  device const uint* active_experts [[buffer(1)]],
  device uchar* destination [[buffer(2)]],
  device const uchar* input_scales [[buffer(3)]],
  device uchar* output_scales [[buffer(4)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  const uint active_expert = tgid.y;
  if (x >= selected_dispatch_items_per_expert)
    return;
  const uint expert = active_experts[active_expert];
  if (expert >= expert_count)
    return;
  if (x < groups_per_expert) {
    const uint row = x / groups_per_row;
    const uint group = x - row * groups_per_row;
    const uint col_base = group * group_size;
    device uchar* expert_destination = destination + (ulong)expert * rows_per_expert * row_length;
    decode_store_group(source, (ulong)expert * groups_per_expert + x, expert_destination, row * row_length + col_base, col_base);
  }
  if (x < scale_bytes_per_expert) {
    output_scales[expert * scale_bytes_per_expert + x] =
      input_scales[expert * scale_bytes_per_expert + x];
  }
}

kernel void dequantize_8i_rowwise_x_selected_plan(
  device const int* indices [[buffer(0)]],
  device const int* counts [[buffer(1)]],
  device uint* active_experts [[buffer(2)]],
  device uint* dispatch_args [[buffer(3)]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  if (lid != 0)
    return;
  uint active_count = 0;
  for (uint bin = 0; bin < bincount; ++bin) {
    const int count = counts[bin];
    const int expert_i = indices[bin];
    if (count > 0 && expert_i >= 0 && expert_i < (int)expert_count)
      active_experts[active_count++] = (uint)expert_i;
  }
  if (active_count == 0)
    active_experts[0] = expert_count;
  dispatch_args[0] = (selected_dispatch_items_per_expert + threadgroup_size - 1) / threadgroup_size;
  dispatch_args[1] = max(active_count, 1u);
  dispatch_args[2] = 1;
}
)";
	return shader;
}

std::string Dequantize8iRowwiseXKernel::createConstants() const noexcept
{
	std::string defines = "";
	defines += "constant ushort threadgroup_size = 256;\n";
	defines += "constant uint row_length [[function_constant(0)]];\n";
	defines += "constant uint group_size [[function_constant(1)]];\n";
	defines += "constant uint groups_per_row [[function_constant(2)]];\n";
	defines += "constant uint total_groups [[function_constant(6)]];\n";
	defines += "constant uint scale_bytes [[function_constant(7)]];\n";
	defines += "constant uint dispatch_items [[function_constant(8)]];\n";
	defines += "constant uint groups_per_expert [[function_constant(9)]];\n";
	defines += "constant uint scale_bytes_per_expert [[function_constant(10)]];\n";
	defines += "constant uint selected_dispatch_items_per_expert [[function_constant(11)]];\n";
	defines += "constant uint rows_per_expert [[function_constant(12)]];\n";
	defines += "constant uint expert_count [[function_constant(13)]];\n";
	defines += "constant uint bincount [[function_constant(14)]];\n";
	return defines;
}
