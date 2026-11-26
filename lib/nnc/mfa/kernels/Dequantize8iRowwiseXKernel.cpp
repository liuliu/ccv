#include "Dequantize8iRowwiseXKernel.hpp"
#include "TMCQDecodeSource.hpp"
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

namespace {

static bool is_tmcq(const uint32_t format)
{
	const uint32_t codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
	return codec >= CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 && codec <= CCV_NNC_QX_8I_ROWWISE_TMCQ_N32;
}

// The 2-tile block map of tile_pair_row_column covers 16 tiles_per_row threads of a tile row only for an even
// tiles_per_row. For an odd one the last 16 threads (from 16 (tiles_per_row - 1), a multiple of 32) take the last tile,
// one row each, so every (row, tile column) segment is still decoded by exactly one of the 16 tiles_per_row threads.
// The same rewrite serves TMCQ and static-rate FTQ, whose sources share the function.
static void tile_pair_tail_source(std::string& shader)
{
	const std::string body = "  return uint2(tile_row * 16u + ((in_tile_row & 31u) >> 1), (in_tile_row >> 5) * 2u + (in_tile_row & 1u));\n";
	const size_t at = shader.find(body);
	CCV_NNC_MFA_PRECONDITION(at != std::string::npos && shader.find(body, at + 1) == std::string::npos);
	shader.insert(at,
		"  // An odd tiles_per_row: the last 16 threads of the tile row take its last tile, one row each.\n"
		"  if ((tiles_per_row & 1u) && in_tile_row >= 16u * (tiles_per_row - 1u))\n"
		"    return uint2(tile_row * 16u + (in_tile_row & 15u), tiles_per_row - 1u);\n");
	const std::string note = "Needs tiles_per_row even (row_length % 256 == 0 here).";
	const size_t comment = shader.find(note);
	if (comment != std::string::npos)
		shader.replace(comment, note.size(), "An odd tiles_per_row ends in 16 threads on its last tile.");
}

// TMCQ int8 decode. A tile spans 16 rows, so the per-group kernels above do not apply. Thread
// mapping: one thread per row segment, i.e. the 16 weights one output row takes from one tile.
// Threads go in blocks of 2 neighbouring tiles x 16 rows (tile_pair_row_column): the 32 lanes of a
// SIMD group read the n adjacent words of two tiles (coalesced) and write 32-byte pieces of 16 output
// rows. dispatch_items (see the descriptors) is max(rows * K / 16, scale bytes); total_groups /
// groups_per_expert count tiles. `tail` (an odd tiles_per_row, K % 32 == 16) gives the last tile of every tile row to
// 16 threads of its own (tile_pair_tail_source); without it the source is the original one for even tiles_per_row.
static std::string create_tmcq_source(const uint32_t format, const bool tail)
{
	const uint32_t n = 16 + 2 * ((format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) - CCV_NNC_QX_8I_ROWWISE_TMCQ_N16);
	std::string shader = R"(
#include <metal_stdlib>
using namespace metal;
)";
	shader += "constant uint tmcq_n = " + std::to_string(n) + ";\n";
	shader += "constant uint tmcq_tile_words = " + std::to_string(n / 2) + ";\n";
	shader += ccv_nnc_mfa_tmcq_decode_source();
	shader += R"(
// Thread x of a tile grid -> (output row, tile column) in blocks of 2 neighbouring tiles x 16 rows: the 32 lanes of a
// SIMD group decode the 16 rows of two adjacent tiles, so they read adjacent code words (the row-major order spread a
// tile's 16 readers over 16 SIMD groups, uncoalesced). Needs tiles_per_row even (row_length % 256 == 0 here).
inline uint2 tile_pair_row_column(const uint x, const uint tiles_per_row)
{
  const uint tile_row = x / (16u * tiles_per_row);
  const uint in_tile_row = x - tile_row * (16u * tiles_per_row);
  return uint2(tile_row * 16u + ((in_tile_row & 31u) >> 1), (in_tile_row >> 5) * 2u + (in_tile_row & 1u));
}

// Writes the 16 int8 weights of `row` (local to the tile grid starting at `source`) in tile column
// `tc` to destination[row * row_length + 16 * tc ...].
inline void tmcq_store_segment(device const uchar* source, device uchar* destination, const uint row, const uint tc)
{
  const uint tiles_per_row = row_length >> 4;
  const ulong tile_index = (ulong)(row >> 4) * tiles_per_row + tc;
  device const uint* tile = reinterpret_cast<device const uint*>(source) + tile_index * tmcq_tile_words;
  int4 q[4];
  tmcq_decode_row16(tile, tmcq_n, row & 15u, q, tmcq_table);
  device char4* destination4 = reinterpret_cast<device char4*>(destination + (ulong)row * row_length + (tc << 4));
  destination4[0] = char4(q[0]);
  destination4[1] = char4(q[1]);
  destination4[2] = char4(q[2]);
  destination4[3] = char4(q[3]);
}

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
  // total_groups tiles of 16 segments each.
  if (x < total_groups * 16u) {
    const uint2 rc = tile_pair_row_column(x, row_length >> 4);
    tmcq_store_segment(source, destination, rc.x, rc.y);
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
  // groups_per_expert tiles of 16 segments each; rows_per_expert % 16 == 0, so experts start on
  // tile rows and the 16-byte tile stride keeps every expert on a word boundary.
  if (x < groups_per_expert * 16u) {
    const uint2 rc = tile_pair_row_column(x, row_length >> 4);
    device const uchar* expert_source = source + (ulong)expert * groups_per_expert * (tmcq_tile_words * 4u);
    device uchar* expert_destination = destination + (ulong)expert * rows_per_expert * row_length;
    tmcq_store_segment(expert_source, expert_destination, rc.x, rc.y);
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
	if (tail)
		tile_pair_tail_source(shader);
	return shader;
}

}

Dequantize8iRowwiseXKernel::Dequantize8iRowwiseXKernel(Dequantize8iRowwiseXKernelDescriptor descriptor, MTL::Device* const device)
{
	format = descriptor.format;
	levelMask = descriptor.levelMask;
	tileTail = descriptor.tileTail;
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

// FTQ (CCVFTQSR blobs, see CCV_NNC_QX_8I_ROWWISE_FTQ_N16).
static bool is_ftq(const uint32_t format)
{
	const uint32_t codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
	return codec >= CCV_NNC_QX_8I_ROWWISE_FTQ_N16 && codec <= CCV_NNC_QX_8I_ROWWISE_FTQ_N48;
}

// FTQ int8 decode (see CCV_NNC_QX_8I_ROWWISE_FTQ_N16): one thread per row segment, the 8 pairs of one output row
// inside one tile, in blocks of 2 tiles x 16 rows (tile_pair_row_column). Tile g has 16 n bits from word g n / 2 of
// the code, which follows the 32-byte header. A row segment's windows start at bit r n + floor(n / 8) - 16 of the
// tile: the thread reads three of the tile's n / 2 words taken mod n / 2 (the windows span 7 n / 8 + 16 <= 64 bits
// for n <= 48) and funnels them once, fixed per row r: row 0 starts before its tile and wraps to the
// tile end (a tail-biting loop, indexed like TMCQ's words), and row 15's unused high bits wrap to the tile start, so
// every read stays inside the tile. The window offsets are compile-time. `ftq_n` is n; `index_shift` is 32 - log2 of
// the table's pair count (23 for the built-in 512-pair table). `tail` is TMCQ's (tile_pair_tail_source): an odd
// tiles_per_row.
// The selected kernels decode the experts that indices / counts route to out of a blob of expert_count experts of
// rows_per_expert rows each: one (expert_count rows_per_expert) x row_length blob, so expert e's tiles start
// groups_per_expert tiles of n / 2 words after the code start and its FP16 scales rows_per_expert after the blob's
// scale start. The output scales are FP16 or, for an FP32 base, FP32 (scale_bytes_per_expert = 4 rows_per_expert).
// The plain, selected and plan kernels share one library (Dequantize8iRowwiseXKernelDescriptor has no kernel kind).
static std::string create_ftq_source(const uint32_t n, const uint32_t index_shift, const bool tail)
{
	std::string shader = R"(
#include <metal_stdlib>
using namespace metal;
)";
	// Table id 0 is the built-in table.
	shader += ccv_nnc_mfa_tmcq_decode_source();
	shader += "constant uint ftq_index_shift = " + std::to_string(index_shift) + ";\n";
	shader += "constant uint ftq_n = " + std::to_string(n) + ";\n";
	shader += R"(
inline uint2 tile_pair_row_column(const uint x, const uint tiles_per_row)
{
  const uint tile_row = x / (16u * tiles_per_row);
  const uint in_tile_row = x - tile_row * (16u * tiles_per_row);
  return uint2(tile_row * 16u + ((in_tile_row & 31u) >> 1), (in_tile_row >> 5) * 2u + (in_tile_row & 1u));
}

// Writes the 16 int8 weights of `row` (local to the tile grid whose code starts at `code`) in tile column `tc` to
// destination[row * row_length + 16 * tc ...]. An external table arrives as `tmcq_ext` (rewrite_decode_codebook); the
// built-in table is the constant tmcq_table.
inline void ftq_store_segment(device const uchar* code, device uchar* destination, const uint row, const uint tc, constant char2* tmcq_ext)
{
  const uint tiles_per_row = row_length >> 4;
  char q[16];
  auto ftq_pair = [&](const uint state) -> char2 { return tmcq_table[(state * 0x83DCD12Du) >> ftq_index_shift]; };
  const uint W = ftq_n / 2u;
  const uint r = row & 15u;
  device const uint* t = reinterpret_cast<device const uint*>(code) + (ulong)((row >> 4) * tiles_per_row + tc) * W;
  const uint lane_bit = (r == 0u ? 16u * ftq_n : r * ftq_n) + (ftq_n >> 3) - 16u;
  const uint o0 = lane_bit >> 5;
  const uint o1 = o0 + 1u == W ? 0u : o0 + 1u;
  const uint o2 = o1 + 1u == W ? 0u : o1 + 1u;
  const uint shift = lane_bit & 31u;
  const uint w0 = t[o0], w1 = t[o1], w2 = t[o2];
  const ulong lo = ((ulong)tmcq_funnel(w1, w2, shift) << 32) | tmcq_funnel(w0, w1, shift);
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; i++) {
    // Window i at offset floor((i + 1) n / 8) - floor(n / 8) of the low 64 bits.
    const uint k = (((i + 1u) * ftq_n) >> 3) - (ftq_n >> 3);
    const char2 v = ftq_pair(uint(lo >> k) & 0xffffu);
    q[2 * i] = v.x;
    q[2 * i + 1] = v.y;
  }
  device char4* destination4 = reinterpret_cast<device char4*>(destination + (ulong)row * row_length + (tc << 4));
  destination4[0] = char4(q[0], q[1], q[2], q[3]);
  destination4[1] = char4(q[4], q[5], q[6], q[7]);
  destination4[2] = char4(q[8], q[9], q[10], q[11]);
  destination4[3] = char4(q[12], q[13], q[14], q[15]);
}

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
  if (x < total_groups * 16u) {
    const uint2 rc = tile_pair_row_column(x, row_length >> 4);
    ftq_store_segment(source + 32, destination, rc.x, rc.y, tmcq_table);
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
  // groups_per_expert tiles of 16 segments each; rows_per_expert % 16 == 0, so experts start on tile rows and every
  // tile on a word.
  if (x < groups_per_expert * 16u) {
    const uint2 rc = tile_pair_row_column(x, row_length >> 4);
    device const uchar* expert_code = source + 32 + (ulong)expert * groups_per_expert * (ftq_n * 2u);
    device uchar* expert_destination = destination + (ulong)expert * rows_per_expert * row_length;
    ftq_store_segment(expert_code, expert_destination, rc.x, rc.y, tmcq_table);
  }
  if (x < rows_per_expert) {
    const ulong row = (ulong)expert * rows_per_expert + x;
    const half scale = reinterpret_cast<device const half*>(input_scales)[row];
    if (scale_bytes_per_expert == rows_per_expert * 4u)
      reinterpret_cast<device float*>(output_scales)[row] = float(scale);
    else
      reinterpret_cast<device half*>(output_scales)[row] = scale;
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
	if (tail)
		tile_pair_tail_source(shader);
	return shader;
}

// Variable-rate FTQ (CCVFTQVR blobs, see CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16).
static bool is_ftq_vr(const uint32_t format)
{
	const uint32_t codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
	return codec >= CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 && codec <= CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N48;
}

// The words of one row segment of a tile at the compile-time rate N: the same reads and windows as the static-rate
// FTQ decode at rate N (create_ftq_source), from `t`, the tile's first word, into q[16]. Below n = 16 the first windows
// of rows 0 and 1 start before the tile and wrap to its end.
static std::string ftq_segment_source(const uint32_t n)
{
	std::string body = "      constexpr uint N = " + std::to_string(n) + "u;\n      constexpr uint W = N / 2u;\n";
	if (n >= 16)
		body += "      const uint lane_bit = (r == 0u ? 16u * N : r * N) + (N >> 3) - 16u;\n";
	else
		body += "      const uint lane_raw = r * N + (N >> 3) + 16u * N - 16u;\n"
			"      const uint lane_bit = lane_raw >= 16u * N ? lane_raw - 16u * N : lane_raw;\n";
	body += "      const uint o0 = lane_bit >> 5;\n"
		"      const uint o1 = o0 + 1u == W ? 0u : o0 + 1u;\n"
		"      const uint o2 = o1 + 1u == W ? 0u : o1 + 1u;\n"
		"      const uint shift = lane_bit & 31u;\n";
	return body;
}

static std::string ftq_segment_decode_source(const uint32_t n)
{
	std::string body;
	body += "        const uint w0 = t[o0], w1 = t[o1], w2 = t[o2];\n"
		"        const ulong lo = ((ulong)tmcq_funnel(w1, w2, shift) << 32) | tmcq_funnel(w0, w1, shift);\n";
	body += "        #pragma clang loop unroll(full)\n"
		"        for (uint i = 0; i < 8u; i++) {\n"
		"          const uint k = (((i + 1u) * N) >> 3) - (N >> 3);\n"
		"          const char2 v = ftq_pair(uint(lo >> k) & 0xffffu);\n"
		"          q[2 * i] = v.x;\n"
		"          q[2 * i + 1] = v.y;\n"
		"        }\n";
	return body;
}

// Variable-rate FTQ int8 decode. Tile row tr spans words prefix[tr] .. prefix[tr + 1] of the code, which follows the
// header and the R + 1 word prefix; n_r = 2 (prefix[tr + 1] - prefix[tr]) / tiles_per_row is uniform in a tile row.
// The prefix is bound again as a constant buffer at buffer(6). A thread decodes T blocks of 2 tiles x 16 rows of one
// tile row (ftq_vr_blocks_per_thread: 2, or 1 for a folded table on fewer than 2^20 row segments), strided so a SIMD
// group reads adjacent code words; one switch over n_r per thread picks the per-rate body of the static-rate decode
// for every level of `level_mask` (bit i is n = 8 + 2 i). `index_shift` and `fold` are as for static-rate FTQ.
// `generic` (ftq_vr_generic_blocks: tiles_per_row % (2 T) != 0) strides G = ceil(floor(tiles_per_row / 2) / T) blocks,
// skips the block indices past the row, and gives the last tile of an odd row to 16 threads of its own, one row each
// (ftq_vr_threads_per_tile_row threads per tile row); without it the source is the original one.
static std::string create_ftq_vr_source(const uint32_t level_mask, const uint32_t index_shift, const bool fold, const bool generic)
{
	std::string shader = R"(
#include <metal_stdlib>
using namespace metal;
)";
	shader += ccv_nnc_mfa_tmcq_decode_source();
	shader += "constant uint ftq_index_shift = " + std::to_string(index_shift) + ";\n";
	shader += std::string("#define FTQ_VR_FOLD ") + (fold ? "1" : "0") + "\n";
	shader += R"(
kernel void dequantize_8i_rowwise_x(
  device const uchar* source [[buffer(0)]],
  device uchar* destination [[buffer(1)]],
  device const uchar* input_scales [[buffer(2)]],
  device uchar* output_scales [[buffer(3)]],
  constant uint* ftq_vr_prefix [[buffer(6)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  if (x >= dispatch_items)
    return;
)";
	shader += generic ? R"(  const uint tiles_per_row = row_length >> 4;
  const uint segments = total_groups * 16u;
  const uint T = (FTQ_VR_FOLD && segments < (1u << 20)) ? 1u : 2u;
  // ftq_vr_threads_per_tile_row: G strided blocks of 2 tiles x 16 rows, then 16 threads for the last tile of an odd
  // tile row (one row each).
  const uint pairs = tiles_per_row >> 1;
  const uint block_stride = (pairs + T - 1u) / T;
  const uint per_tile_row = 32u * block_stride + ((tiles_per_row & 1u) << 4);
  if (x < (total_groups / tiles_per_row) * per_tile_row) {
    const uint tile_row = x / per_tile_row;
    const uint in_tile_row = x - tile_row * per_tile_row;
    const bool tail = in_tile_row >= 32u * block_stride;
    const uint lane = in_tile_row & 31u;
    const uint group = in_tile_row >> 5;
    const uint row = tile_row * 16u + (tail ? (in_tile_row & 15u) : (lane >> 1));
    const uint r = row & 15u;
)" : R"(  const uint tiles_per_row = row_length >> 4;
  const uint segments = total_groups * 16u;
  const uint T = (FTQ_VR_FOLD && segments < (1u << 20)) ? 1u : 2u;
  if (x < segments / T) {
    const uint per_tile_row = (16u * tiles_per_row) / T;
    const uint tile_row = x / per_tile_row;
    const uint in_tile_row = x - tile_row * per_tile_row;
    const uint lane = in_tile_row & 31u;
    const uint group = in_tile_row >> 5;
    const uint row = tile_row * 16u + (lane >> 1);
    const uint r = row & 15u;
    const uint block_stride = tiles_per_row / (2u * T);
)";
	shader += R"(    device const uint* code = reinterpret_cast<device const uint*>(source + 32) + (total_groups / tiles_per_row + 1u);
    const uint off0 = ftq_vr_prefix[tile_row];
    const uint VW = (ftq_vr_prefix[tile_row + 1u] - off0) / tiles_per_row;
    device const uint* row_code = code + off0;
    device char4* dst_row = reinterpret_cast<device char4*>(destination + (ulong)row * row_length);
    char q[16];
    auto ftq_pair = [&](const uint state) -> char2 { return tmcq_table[(state * 0x83DCD12Du) >> ftq_index_shift]; };
    switch (VW) {
)";
	// Tile column tc of block j of this thread; the generic map skips the columns past the row.
	const std::string tile_column = generic ?
		"        const uint block = group + j * block_stride;\n"
		"        const uint tc = tail ? (j == 0u ? tiles_per_row - 1u : tiles_per_row) : (block < pairs ? ((block << 1) | (lane & 1u)) : tiles_per_row);\n"
		"        if (tc >= tiles_per_row)\n"
		"          continue;\n" :
		"        const uint tc = ((group + j * block_stride) << 1) | (lane & 1u);\n";
	uint32_t i;
	for (i = 0; i < 21; i++)
		if (level_mask & (1u << i))
		{
			const uint32_t n = 8 + 2 * i;
			shader += "    case " + std::to_string(n / 2) + "u: {\n" + ftq_segment_source(n) +
				"      for (uint j = 0; j < T; j++) {\n" + tile_column +
				"        device const uint* t = row_code + tc * W;\n" + ftq_segment_decode_source(n) +
				"        device char4* d = dst_row + tc * 4u;\n"
				"        d[0] = char4(q[0], q[1], q[2], q[3]);\n"
				"        d[1] = char4(q[4], q[5], q[6], q[7]);\n"
				"        d[2] = char4(q[8], q[9], q[10], q[11]);\n"
				"        d[3] = char4(q[12], q[13], q[14], q[15]);\n"
				"      }\n"
				"    } break;\n";
		}
	if (generic)
		shader += "    default:\n"
			"      // A rate outside the level mask (rejected when the weights are loaded): zeros.\n"
			"      for (uint j = 0; j < T; j++) {\n" + tile_column +
			"        device char4* d = dst_row + tc * 4u;\n"
			"        d[0] = d[1] = d[2] = d[3] = char4(0);\n"
			"      }\n"
			"      break;\n";
	else
		shader += R"(    default:
      // A rate outside the level mask (rejected when the weights are loaded): zeros.
      for (uint j = 0; j < T; j++) {
        device char4* d = dst_row + ((((group + j * block_stride) << 1) | (lane & 1u)) * 4u);
        d[0] = d[1] = d[2] = d[3] = char4(0);
      }
      break;
)";
	shader += R"(    }
  }
  if (x < scale_bytes)
    output_scales[x] = input_scales[x];
}
)";
	return shader;
}

// An external codebook (tmcq_codebook_log2(format) != 0) for the TMCQ / FTQ int8 decode: every decoding kernel of
// the shader takes the table at buffer(5) (the plain kernel binds 0 - 3, the selected one 0 - 4). TMCQ copies its
// half2 part into threadgroup memory before any thread returns and reads it through tmcq_codebook_char2; FTQ reads
// its int8 part as constant char2, like the built-in tmcq_table.
static void rewrite_decode_codebook(std::string& shader, const uint32_t log2, const bool ftq)
{
	CCV_NNC_MFA_PRECONDITION(log2 >= 9 && log2 <= 11);
	if (ftq) {
		// The FTQ source already shifts by ftq_index_shift; only the helpers go in.
		const std::string anchor = "using namespace metal;\n";
		const size_t at = shader.find(anchor);
		CCV_NNC_MFA_PRECONDITION(at != std::string::npos);
		shader.insert(at + anchor.size(), tmcq_codebook_source(log2));
	} else
		tmcq_codebook_rewrite(shader, log2);
	if (ftq) {
		// FTQ reads the int8 copy of the pairs that follows the half2 table in the buffer (see
		// ccv_nnc_mfa_codebook_buffer) straight from the constant address space, like the built-in tmcq_table:
		// no threadgroup copy or barrier, and no half2 -> char2 conversion per lookup (the lookups are most of
		// this decode's work).
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "\n\n  uint3 tgid [[threadgroup_position_in_grid]],", "\n  constant char2* codebook [[buffer(5)]],\n\n  uint3 tgid [[threadgroup_position_in_grid]],") >= 1);
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "  const uint x = tgid.x * threadgroup_size + lid;\n",
			"  constant char2* const tmcq_ext = codebook + 2u * TMCQ_CODEBOOK_ENTRIES;\n"
			"  const uint x = tgid.x * threadgroup_size + lid;\n") >= 1);
	} else {
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "\n\n  uint3 tgid [[threadgroup_position_in_grid]],", "\n  device const uint4* codebook [[buffer(5)]],\n\n  uint3 tgid [[threadgroup_position_in_grid]],") >= 1);
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "  const uint x = tgid.x * threadgroup_size + lid;\n",
			"  threadgroup half2 tmcq_codes[TMCQ_CODEBOOK_ENTRIES];\n"
			"  tmcq_codebook_load(tmcq_codes, codebook, lid, threadgroup_size);\n"
			"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
			"  const tmcq_codebook_char2 tmcq_ext = { tmcq_codes };\n"
			"  const uint x = tgid.x * threadgroup_size + lid;\n") >= 1);
	}
	if (ftq) {
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "tmcq_table[(state", "tmcq_ext[(state") == 1);
	} else {
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "const uint row, const uint tc)\n{", "const uint row, const uint tc, const tmcq_codebook_char2 tmcq_ext)\n{") == 1);
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "q, tmcq_table);", "q, tmcq_ext);") == 1);
		CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "rc.x, rc.y);", "rc.x, rc.y, tmcq_ext);") == 2);
	}
}

std::string Dequantize8iRowwiseXKernel::createSource() const noexcept
{
	if (is_tmcq(format)) {
		std::string shader = createConstants() + "\n" + create_tmcq_source(format, (tileTail & 1) != 0);
		if (tmcq_codebook_log2(format))
			rewrite_decode_codebook(shader, tmcq_codebook_log2(format), false);
		return shader;
	}
	if (is_ftq(format) || is_ftq_vr(format)) {
		// FTQ takes the index shift as a source constant (ftq_index_shift), 32 - log2 of the table size.
		const uint32_t log2 = tmcq_codebook_log2(format);
		std::string shader = createConstants() + "\n" + (is_ftq_vr(format) ?
			create_ftq_vr_source(levelMask, log2 ? 32 - log2 : 23, tmcq_codebook_fold(format), (tileTail & 2) != 0) :
			create_ftq_source(8 + 2 * ((format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) - CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST), log2 ? 32 - log2 : 23, (tileTail & 1) != 0));
		if (log2) {
			rewrite_decode_codebook(shader, log2, true);
			// Static-rate FTQ decodes in ftq_store_segment, which takes the table from both decoding kernels.
			if (is_ftq(format))
				CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "rc.y, tmcq_table);", "rc.y, tmcq_ext);") == 2);
		}
		if (tmcq_codebook_fold(format)) {
			// Quadrant fold (512 base pairs): base pair (h >> 16) & 511 of h = state * 0x83DCD12D, the values
			// negated in the int8 domain by bits 15 / 31 of h, as the fused GEMV does on the half2 bits.
			CCV_NNC_MFA_PRECONDITION(log2 == 9);
			const std::string anchor = "struct tmcq_codebook_ushort";
			const size_t at = shader.find(anchor);
			CCV_NNC_MFA_PRECONDITION(at != std::string::npos);
			shader.insert(at, "inline char2 ftq_fold_lookup(constant char2* t, uint state)\n"
				"{\n"
				"  const uint h = state * 0x83DCD12Du;\n"
				"  const char2 v = t[(h >> 16) & 511u];\n"
				"  return char2((h & 0x8000u) ? -v.x : v.x, (h & 0x80000000u) ? -v.y : v.y);\n"
				"}\n\n");
			CCV_NNC_MFA_PRECONDITION(tmcq_codebook_replace(shader, "tmcq_ext[(state * 0x83DCD12Du) >> ftq_index_shift]", "ftq_fold_lookup(tmcq_ext, state)") == 1);
		}
		return shader;
	}
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
