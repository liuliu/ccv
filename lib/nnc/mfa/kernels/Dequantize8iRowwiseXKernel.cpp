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
	if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
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

inline uint read_bits(device const uchar* data, const uint bit_offset, const uint bits)
{
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7;
  const uint value =
    (uint)data[byte_offset] |
    ((uint)data[byte_offset + 1] << 8) |
    ((uint)data[byte_offset + 2] << 16);
  return (value >> shift) & ((1u << bits) - 1u);
}
)";

	if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_S || format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS) {
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
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			shader += R"(
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  uint bit = group_index * group_bits;
  int q[16];
  for (uint j = 0; j < 16; ++j, bit += 4)
    q[j] = (int)read_bits(source, bit, 4) - 8;
  const int m = (int)read_bits(source, bit, 4) + 1;
  const int b = (int)read_bits(source, bit + 4, 4) - 8;
  for (uint j = 0; j < 16; ++j)
    q8[j] = q[j] * m + b;
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
			shader += R"(
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  uint bit = group_index * group_bits;
  int q[16];
  for (uint j = 0; j < 16; ++j, bit += 3)
    q[j] = (int)read_bits(source, bit, 3) - 4;
  const int m = (int)read_bits(source, bit, 5) + 1;
  const int b = ((int)read_bits(source, bit + 5, 3) - 4) << 1;
  for (uint j = 0; j < 16; ++j)
    q8[j] = q[j] * m + b;
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
			shader += R"(
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  uint bit = group_index * group_bits;
  int q[16];
  for (uint j = 0; j < 16; ++j, bit += 2)
    q[j] = (int)read_bits(source, bit, 2);
  const int m = (int)read_bits(source, bit, 6) + 1;
  const int z = (int)read_bits(source, bit + 6, 4) << 3;
  for (uint j = 0; j < 16; ++j)
    q8[j] = q[j] * m - z;
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			shader += R"(
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  const uint bit = group_index * group_bits;
  const uint grid0 = read_bits(source, bit, 10);
  const uint grid1 = read_bits(source, bit + 10, 10);
  const uint signs = read_bits(source, bit + 20, 16);
  const int scale = (int)read_bits(source, bit + 36, 6) + 1;
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
  const uint bit = group_index * group_bits;
  const uint grid0 = read_bits(source, bit, 9);
  const uint signs = read_bits(source, bit + 9, 8);
  const int scale = q2_xs_scales[read_bits(source, bit + 17, 4)];
  for (uint j = 0; j < 8; ++j) {
    const int mag = min(iq2_value(iq2xs_grid, grid0, j) * scale, 127);
    q8[j] = (signs & (1u << j)) ? -mag : mag;
  }
}
)";
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			shader += R"(
inline void decode_group(device const uchar* source, const uint group_index, thread int* q8)
{
  const uint bit = group_index * group_bits;
  uint grid[4];
  for (uint j = 0; j < 4; ++j)
    grid[j] = read_bits(source, bit + j * 9, 9);
  const uint signs = read_bits(source, bit + 36, 16);
  const int scale = (int)read_bits(source, bit + 52, 4) + 1;
  for (uint sg = 0; sg < 4; ++sg) {
    for (uint j = 0; j < 4; ++j) {
      const uint lane = sg * 4 + j;
      const int mag = min(iq3s_value(grid[sg], j) * scale, 127);
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
  const uint bit = group_index * group_bits;
  const uint grid0 = read_bits(source, bit, 8);
  const uint grid1 = read_bits(source, bit + 8, 8);
  const uint signs = read_bits(source, bit + 16, 8);
  const int scale = (int)read_bits(source, bit + 24, 4) + 1;
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
    int q8[16] = {0};
    decode_group(source, x, q8);
    device int8_t* destination_q = reinterpret_cast<device int8_t*>(destination);
    const uint col_base = group * group_size;
    for (uint j = 0; j < group_size; ++j) {
      const uint col = col_base + j;
      if (col < row_length)
        destination_q[row * row_length + col] = (int8_t)q8[j];
    }
  }
  if (x < scale_bytes)
    destination[output_scale_offset + x] = source[input_scale_offset + x];
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
	defines += "constant uint group_bits [[function_constant(3)]];\n";
	defines += "constant uint input_scale_offset [[function_constant(4)]];\n";
	defines += "constant uint output_scale_offset [[function_constant(5)]];\n";
	defines += "constant uint total_groups [[function_constant(6)]];\n";
	defines += "constant uint scale_bytes [[function_constant(7)]];\n";
	defines += "constant uint dispatch_items [[function_constant(8)]];\n";
	return defines;
}
