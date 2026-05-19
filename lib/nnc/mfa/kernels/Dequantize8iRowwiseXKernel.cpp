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

std::string Dequantize8iRowwiseXKernel::createSource() const noexcept
{
	std::string shader = createConstants() + "\n";
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

	shader += R"(
#include <metal_stdlib>
using namespace metal;
)";

	if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K || format == CCV_NNC_QX_8I_ROWWISE_IQ2_S) {
		shader += R"(
inline ulong packed42_payload(device const uchar* data, const uint group_index)
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
inline uint packed21_payload(device const uchar* data, const uint group_index)
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
inline uint packed28_payload(device const uchar* data, const uint group_index)
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
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 11u;
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
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			shader += R"(
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 9u;
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

inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 7u;
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
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
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
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
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

inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
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

inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 8u;
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
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  device const uchar* p = source + group_index * 7u;
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
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
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

inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 11u;
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

inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 9u;
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

inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 7u;
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

inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
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
inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 8u;
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
inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
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
inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
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
inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
{
  device const uchar* p = source + group_index * 7u;
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
inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
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
	} else {
		shader += R"(
inline void decode_store_group(device const uchar* source, const uint group_index, device uchar* destination, const uint destination_offset, const uint col_base)
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
    destination[output_scale_offset + x] = source[input_scale_offset + x];
}

kernel void dequantize_8i_rowwise_x_selected(
  device const uchar* source [[buffer(0)]],
  device const uint* active_experts [[buffer(1)]],
  device uchar* destination [[buffer(2)]],

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
    const uint destination_row = expert * rows_per_expert + row;
    decode_store_group(source, expert * groups_per_expert + x, destination, destination_row * row_length + col_base, col_base);
  }
  if (x < scale_bytes_per_expert) {
    destination[output_scale_offset + expert * scale_bytes_per_expert + x] =
      source[input_scale_offset + expert * scale_bytes_per_expert + x];
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
  for (uint segment = 0; segment < segment_count; ++segment) {
    const int count = counts[segment];
    const int expert_i = indices[segment];
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
	defines += "constant uint input_scale_offset [[function_constant(4)]];\n";
	defines += "constant uint output_scale_offset [[function_constant(5)]];\n";
	defines += "constant uint total_groups [[function_constant(6)]];\n";
	defines += "constant uint scale_bytes [[function_constant(7)]];\n";
	defines += "constant uint dispatch_items [[function_constant(8)]];\n";
	defines += "constant uint groups_per_expert [[function_constant(9)]];\n";
	defines += "constant uint scale_bytes_per_expert [[function_constant(10)]];\n";
	defines += "constant uint selected_dispatch_items_per_expert [[function_constant(11)]];\n";
	defines += "constant uint rows_per_expert [[function_constant(12)]];\n";
	defines += "constant uint expert_count [[function_constant(13)]];\n";
	defines += "constant uint segment_count [[function_constant(14)]];\n";
	return defines;
}
