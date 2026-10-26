#include "SegmentedScaledGEMMKernel.hpp"
#include "SegmentedScaledGEMMKernelDescriptor.hpp"
#include "CodeWriter.hpp"
#include "Dequantize8iRowwiseXDescriptor.hpp"
#include "Dequantize8iRowwiseXKernel.hpp"
#include "../../ccv_nnc_8i_rowwise_packed_grids.inc"
#include "../ccv_nnc_mfa.hpp"

namespace {

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

// The fused tile decoder's helpers: payload words, the unit unpacker, and
// iq1_store_unit_bytes expanding a unit to int8.
static std::string create_iq1_tile_source(const uint32_t format)
{
  std::string shader;
  shader += R"(
// The 64 bits starting `shift` (0...31) bits into three consecutive words.
inline uint2 iq1_payload_words(const uint3 words, const uint shift)
{
  return uint2((words.x >> shift) | ((words.y << 1) << (31u - shift)), (words.y >> shift) | ((words.z << 1) << (31u - shift)));
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
  return shader;
}

}

SegmentedScaledGEMMKernel::SegmentedScaledGEMMKernel(
    SegmentedScaledGEMMKernelDescriptor descriptor,
    MTL::Device* const device)
{
  blockDimensions = descriptor.blockDimensions;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  ioPrecision = descriptor.ioPrecision;
  useBias = descriptor.useBias;
  loadM = descriptor.loadM;
  format = descriptor.format;
  CCV_NNC_MFA_PRECONDITION(format == 0 || ((format & ~CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) == 0 && !useBias));

  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  // MPP tensor operations require Metal 4, including when built with an older SDK.
  if (options->languageVersion() < MTL::LanguageVersion(0x40000))
    options->setLanguageVersion(MTL::LanguageVersion(0x40000));
  library = NS::TransferPtr(device->newLibrary(string, options.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t SegmentedScaledGEMMKernel::threadgroupSize(MTL::ComputePipelineState* const pipelineState) const noexcept
{
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

uint32_t SegmentedScaledGEMMKernel::maxTileRecords(uint32_t originalM, uint32_t bincount) const noexcept
{
  const uint32_t tilesPerBin = (originalM + blockDimensions[0] - 1) / blockDimensions[0];
  return bincount * (tilesPerBin > 0 ? tilesPerBin : 1u);
}

bool SegmentedScaledGEMMKernel::fusesPackedFormat(uint32_t format, uint64_t M, uint32_t binCount) noexcept
{
  if (format == 0 || (format & ~CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK) != 0)
    return false;
  // Dedicated tile decoders beat expanding the expert bank at every measured batch size.
  if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS || format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS || format == CCV_NNC_QX_8I_ROWWISE_Q2_K ||
      format == CCV_NNC_QX_8I_ROWWISE_IQ1_S || format == CCV_NNC_QX_8I_ROWWISE_IQ1_M)
    return true;
  // The shared group decoder decodes each weight again for every row tile. It wins up to
  // about 64 rows per bin, but expanding once is faster for longer prefill.
  return M <= (uint64_t)binCount * 64;
}

std::string SegmentedScaledGEMMKernel::createSource() const noexcept
{
  CodeWriter source;
  source.SetValue("BLOCK_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_K", std::to_string(blockDimensions[2]));
  source.SetValue("SIMDGROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("IO_TYPE", ioPrecision.name());
  if (format) {
    Dequantize8iRowwiseXDescriptor packed = {};
    packed.format = format;
    // K is a multiple of 256, so every packed row starts on a byte boundary.
    source.SetValue("GROUP_SIZE", std::to_string(packed.groupSize()));
    source.SetValue("PACKED_ROW_BYTES", "(K / " + std::to_string(packed.groupSize()) + "u * " + std::to_string(packed.groupBits()) + "u / 8u)");
    source.SetValue("IQ1_UNIT", format == CCV_NNC_QX_8I_ROWWISE_IQ1_S ? "iq1s_unit" : "iq1m_unit");
    source.SetValue("IQ1_UNIT_BITS", format == CCV_NNC_QX_8I_ROWWISE_IQ1_S ? "50" : "56");
  }
  source.SetValue("MAX_TILE_RECORDS_CONSTANT", loadM ? "" : "constant uint max_tile_records [[function_constant(6)]];\n");
  source.SetValue("LOAD_M_ARGUMENT", loadM ? "    const device uint* loadM [[buffer(4)]],\n" : "");
  source.SetValue("LOAD_M_VALUE", loadM ? "  const uniform<uint> max_tile_records = make_uniform(loadM[0]);\n" : "");
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct TileRecord {
  uint row_offset;
  uint local_row;
  uint count;
  uint expert;
};

constant uint N [[function_constant(0)]];
constant uint K [[function_constant(1)]];
constant uint expert_count [[function_constant(2)]];
constant uint M_block [[function_constant(3)]];
constant uint N_block [[function_constant(4)]];
constant uint K_block [[function_constant(5)]];
constant uint bincount [[function_constant(7)]];
{{MAX_TILE_RECORDS_CONSTANT}}
kernel void segmented_scaled_gemm_plan(
    device const int* indices [[buffer(0)]],
    device const int* counts [[buffer(1)]],
    device TileRecord* records [[buffer(2)]],
    device uint* dispatch_args [[buffer(3)]],
{{LOAD_M_ARGUMENT}}    uint tid [[thread_index_in_threadgroup]])
{
{{LOAD_M_VALUE}}  if (tid != 0)
    return;
  uint row_offset = 0;
  uint record_count = 0;
  for (uint bin = 0; bin < bincount; ++bin) {
    const int count_i = counts[bin];
    const int expert_i = indices[bin];
    const uint count = count_i > 0 ? (uint)count_i : 0u;
    if (count > 0 && expert_i >= 0 && expert_i < (int)expert_count) {
      for (uint local = 0; local < count; local += M_block) {
        if (record_count < max_tile_records) {
          records[record_count] = TileRecord {
            row_offset,
            local,
            count,
            (uint)expert_i,
          };
        }
        ++record_count;
      }
    }
    row_offset += count;
  }
  dispatch_args[0] = min(record_count, max_tile_records);
  dispatch_args[1] = (N + N_block - 1) / N_block;
  dispatch_args[2] = 1;
}
)";
  // IQ2_XXS, Q2_K and IQ3_XXS expand their weights in registers. Other packed projections
  // expand only the current weight tile: IQ1_S and IQ1_M have dedicated tile decoders, and
  // other formats fall back to the dequantizer's group decoder.
  if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
    source += "constant uint iq2_grid[512] = {";
    for (uint32_t code = 0; code < 256; ++code)
      for (uint32_t word = 0; word < 2; ++word) {
        uint32_t packed = 0;
        for (uint32_t j = 0; j < 4; ++j)
          packed |= (((ccv_nnc_8i_rowwise_packed_iq2xxs_grid[code] >> ((word * 4 + j) * 2)) & 3) * 2 + 1) << (j * 8);
        source += std::to_string(packed) + (code == 255 && word == 1 ? "u};" : "u,");
      }
    source += "constant uint iq2_signs[256] = {";
    for (uint32_t code = 0; code < 128; ++code)
      for (uint32_t word = 0; word < 2; ++word) {
        uint32_t packed = 0;
        for (uint32_t j = 0; j < 4; ++j)
          if (ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[code] & (1u << (word * 4 + j)))
            packed |= 255u << (j * 8);
        source += std::to_string(packed) + (code == 127 && word == 1 ? "u};" : "u,");
      }
  } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
    source += "constant uint iq3_grid[256] = {";
    for (uint32_t code = 0; code < 256; ++code) {
      uint32_t packed = 0;
      for (uint32_t j = 0; j < 4; ++j)
        packed |= ((ccv_nnc_8i_rowwise_packed_iq3xxs_grid[code] >> (j * 8)) & 255) >> 2 << (j * 8);
      source += std::to_string(packed) + (code == 255 ? "u};" : "u,");
    }
  } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S || format == CCV_NNC_QX_8I_ROWWISE_IQ1_M) {
    source += create_iq1_tile_source(format);
  } else if (format && format != CCV_NNC_QX_8I_ROWWISE_Q2_K)
    source += Dequantize8iRowwiseXKernel::createDecodeGroupSource(format);
  if (blockDimensions[2] == 64) {
    // Register operands: each SIMD group owns 32 rows x 32 columns of the output and the
    // SIMD groups sit side by side along N. A SIMD group loads its activation fragments and
    // expands its own weight fragments straight into registers, so no tile goes through
    // threadgroup memory or waits on a threadgroup barrier per K step. Fragment layout and
    // MPP register assembly follow NAInt8MatMulKernel's register path.
    CCV_NNC_MFA_PRECONDITION(format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS || format == CCV_NNC_QX_8I_ROWWISE_Q2_K || format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS);
    CCV_NNC_MFA_PRECONDITION(blockDimensions[0] == 32 && blockDimensions[1] == 32 * executionSIMDGroups);
    source += R"(
// A 16x16 fragment has eight values per lane: four adjacent columns in each of
// two rows eight apart. A SIMD group owns a 2x2 array of fragments, where
// fragment m * 2 + k covers rows m * 16 and columns k * 16.
__attribute__((always_inline)) inline short2 register_fragment_coordinate(ushort lane) {
  const short quad = lane >> 2;
  return short2(((quad & 2) | (lane & 1)) * 4, (quad & 4) | ((lane >> 1) & 3));
}

struct register_tile {
  vec<int8_t, 8> fragments[4];
};

struct accumulator_tile {
  vec<int32_t, 8> fragments[4];
};

// Both operands step through K64 at a time, permuted identically, which leaves
// the product unchanged: in K32 step kk, a lane's fragment columns origin.x...+3
// and origin.x + 16...+19 read K offsets 4 * origin.x + 8 * kk...+7. A lane
// therefore holds 16 consecutive K values per row: one 16-byte activation load,
// half an IQ2_XXS group, a whole Q2_K group, or a pair of IQ3_XXS groups.
__attribute__((always_inline)) inline void set_fragments(thread register_tile (&tiles)[2], short m, short r, uint4 words) {
  #pragma clang loop unroll(full)
  for (short kk = 0; kk < 2; ++kk) {
    #pragma clang loop unroll(full)
    for (short h = 0; h < 2; ++h) {
      const char4 values = as_type<char4>(words[kk * 2 + h]);
      #pragma clang loop unroll(full)
      for (short c = 0; c < 4; ++c)
        tiles[kk].fragments[m * 2 + h][r * 4 + c] = values[c];
    }
  }
}

template<bool full_rows>
__attribute__((always_inline)) inline void load_activation_tiles(thread register_tile (&tiles)[2],
    const device int8_t* src, short rows, short2 origin) {
  #pragma clang loop unroll(full)
  for (short m = 0; m < 2; ++m) {
    #pragma clang loop unroll(full)
    for (short r = 0; r < 2; ++r) {
      const short row = m * 16 + r * 8;
      set_fragments(tiles, m, r, full_rows || row < rows - origin.y ? *(const device uint4*)(src + row * K) : uint4(0));
    }
  }
}
)";
    if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
      source += R"(
// Lanes index the grid and the sign masks divergently, so both are read from
// threadgroup copies.
struct decode_tables {
  threadgroup const uint2* grid;
  threadgroup const uint2* signs;
};

__attribute__((always_inline)) inline void decode_weight_tiles(thread register_tile (&tiles)[2],
    const device uchar* B, decode_tables tables, uint k, short2 origin) {
  // Half of one 32-value group: sub-blocks first_sg and first_sg + 1.
  const uint offset = (k + uint(origin.x) * 4u) / 32u * 8u;
  const uint first_sg = (uint(origin.x) >> 1) & 2u;
  #pragma clang loop unroll(full)
  for (short n = 0; n < 2; ++n) {
    #pragma clang loop unroll(full)
    for (short r = 0; r < 2; ++r) {
      const short row = n * 16 + r * 8;
      const uint2 payload = *(const device uint2*)(B + uint(row) * (K / 4u) + offset);
      // iq2_scales: 1...8, then steps of 2 to 16, then steps of 4 to 32.
      const uint code = payload.y >> 28;
      const uint scale = code < 8u ? code + 1u : code < 12u ? (code - 3u) * 2u : (code - 7u) * 4u;
      uint4 words;
      #pragma clang loop unroll(full)
      for (uint kk = 0; kk < 2; ++kk) {
        const uint sg = first_sg + kk;
        const uint2 levels = tables.grid[(payload.x >> (sg * 8u)) & 255u];
        const uint2 masks = tables.signs[(payload.y >> (sg * 7u)) & 127u];
        #pragma clang loop unroll(full)
        for (uint h = 0; h < 2; ++h) {
          // Each byte is at most 7 * 32, so multiplying the packed word has no byte carries.
          const uint raw = levels[h] * scale;
          const uint high = raw & 0x80808080u;
          const uint magnitude = (raw & 0x7f7f7f7fu) | (high - (high >> 7));
          // Magnitudes are in 1...127, so the byte-wise negation cannot carry into another byte.
          words[kk * 2 + h] = (magnitude ^ masks[h]) + (masks[h] & 0x01010101u);
        }
      }
      set_fragments(tiles, n, r, words);
    }
  }
}
)";
    } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
      source += R"(
// Lanes index the grid divergently, so it is read from a threadgroup copy.
struct decode_tables {
  threadgroup const uint* grid;
};

// Two 28-bit groups of eight values: two grid indices, eight sign bits and a 4-bit scale each.
__attribute__((always_inline)) inline void decode_weight_tiles(thread register_tile (&tiles)[2],
    const device uchar* B, decode_tables tables, uint k, short2 origin) {
  const uint bit = (k / 16u + (uint(origin.x) >> 2)) * 56u;
  const uint word = bit >> 5, shift = bit & 31u;
  #pragma clang loop unroll(full)
  for (short n = 0; n < 2; ++n) {
    #pragma clang loop unroll(full)
    for (short r = 0; r < 2; ++r) {
      const short row = n * 16 + r * 8;
      const device uint* p = (const device uint*)(B + uint(row) * {{PACKED_ROW_BYTES}}) + word;
      const uint p0 = p[0], p1 = p[1];
      // The pair reaches a third word only when it starts past the first byte.
      const uint p2 = shift > 8u ? p[2] : 0u;
      const uint first = (p0 >> shift) | ((p1 << 1) << (31u - shift));
      const uint second = (p1 >> shift) | ((p2 << 1) << (31u - shift));
      const uint payloads[2] = { first, (first >> 28) | (second << 4) };
      uint4 words;
      #pragma clang loop unroll(full)
      for (uint group = 0; group < 2; ++group) {
        const uint payload = payloads[group];
        const uint scale = ((payload >> 24) & 15u) + 1u;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 2; ++w) {
          // Grid bytes are at most 15 and scales at most 16, so multiplying the packed word has no byte carries.
          const uint raw = tables.grid[(payload >> (w * 8u)) & 255u] * scale;
          const uint high = raw & 0x80808080u;
          const uint magnitude = (raw & 0x7f7f7f7fu) | (high - (high >> 7));
          // Spread four sign bits to the low bit of each byte, then widen them to byte masks.
          const uint mask = ((((payload >> (16u + w * 4u)) & 15u) * 0x00204081u) & 0x01010101u) * 255u;
          // Magnitudes are in 1...127, so the byte-wise negation cannot carry into another byte.
          words[group * 2u + w] = (magnitude ^ mask) + (mask & 0x01010101u);
        }
      }
      set_fragments(tiles, n, r, words);
    }
  }
}
)";
    } else {
      CCV_NNC_MFA_PRECONDITION(format == CCV_NNC_QX_8I_ROWWISE_Q2_K);
      source += R"(
struct decode_tables {
};

// A Q2_K group packs sixteen 2-bit codes, a 6-bit scale and a 4-bit zero point into 42 bits.
__attribute__((always_inline)) inline void decode_weight_tiles(thread register_tile (&tiles)[2],
    const device uchar* B, decode_tables tables, uint k, short2 origin) {
  const uint bit = (k / 16u + (uint(origin.x) >> 2)) * 42u;
  const uint word = bit >> 5, shift = bit & 31u;
  #pragma clang loop unroll(full)
  for (short n = 0; n < 2; ++n) {
    #pragma clang loop unroll(full)
    for (short r = 0; r < 2; ++r) {
      const short row = n * 16 + r * 8;
      const device uint* p = (const device uint*)(B + uint(row) * {{PACKED_ROW_BYTES}}) + word;
      const uint p0 = p[0], p1 = p[1];
      // The group reaches a third word only when it starts late in the first.
      const uint p2 = shift > 22u ? p[2] : 0u;
      const uint codes = (p0 >> shift) | ((p1 << 1) << (31u - shift));
      const uint header = (p1 >> shift) | ((p2 << 1) << (31u - shift));
      const uint scale = (header & 63u) + 1u;
      const uint zero = ((header >> 6) & 15u) * 8u * 0x01010101u;
      uint4 words;
      #pragma clang loop unroll(full)
      for (uint i = 0; i < 4; ++i) {
        // Spread four 2-bit codes into bytes; a code times the scale is at most 192, so no byte carries.
        const uint q = (codes >> (i * 8u)) & 255u;
        const uint x = ((q | (q << 6) | (q << 12) | (q << 18)) & 0x03030303u) * scale;
        // Byte-wise x - zero modulo 256: the zero point is below 128, so setting each top bit prevents borrows.
        words[i] = ((x | 0x80808080u) - zero) ^ (~x & 0x80808080u);
      }
      set_fragments(tiles, n, r, words);
    }
  }
}
)";
    }
    source += R"(
// Four M16/N32/K16 operations form a SIMD group's M32/N32/K32 update.
__attribute__((always_inline)) inline void multiply_register_tiles(thread accumulator_tile& accum,
    thread const register_tile& a, thread const register_tile& b) {
  constexpr auto descriptor = matmul2d_descriptor(
      16, 32, 16, false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
  #pragma clang loop unroll(full)
  for (short m = 0; m < 2; ++m) {
    #pragma clang loop unroll(full)
    for (short k = 0; k < 2; ++k) {
      auto aT = op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
      auto bT = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
      auto cT = op.get_destination_cooperative_tensor<
          metal::remove_addrspace_t<decltype(aT)>,
          metal::remove_addrspace_t<decltype(bT)>, int32_t>();
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; ++i) {
        aT[i] = a.fragments[m * 2 + k][i];
        bT[i] = b.fragments[k][i];
        bT[8 + i] = b.fragments[2 + k][i];
        cT[i] = accum.fragments[m * 2][i];
        cT[8 + i] = accum.fragments[m * 2 + 1][i];
      }
      op.run(aT, bT, cT);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; ++i) {
        accum.fragments[m * 2][i] = cT[i];
        accum.fragments[m * 2 + 1][i] = cT[8 + i];
      }
    }
  }
}

template<bool full_rows>
inline void multiply_register(const device int8_t* A, const device uchar* B, decode_tables tables,
    device {{IO_TYPE}}* C, const device {{IO_TYPE}}* A_scale, const device {{IO_TYPE}}* B_scale,
    short rows, ushort lane) {
  const short2 origin = register_fragment_coordinate(lane);
  A += origin.y * K + origin.x * 4;
  B += uint(origin.y) * {{PACKED_ROW_BYTES}};
  accumulator_tile accum;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 4; ++i)
    accum.fragments[i] = vec<int32_t, 8>(0);
  // Keep the SIMD groups in step every K512 so they share activation lines in cache.
  #pragma clang loop unroll(disable)
  for (uint block = 0; block < K / 512u; ++block) {
    threadgroup_barrier(mem_flags::mem_none);
    for (uint k = block * 512u; k < block * 512u + 512u; k += 64u) {
      register_tile a[2], b[2];
      load_activation_tiles<full_rows>(a, A + k, rows, origin);
      decode_weight_tiles(b, B, tables, k, origin);
      multiply_register_tiles(accum, a[0], b[0]);
      multiply_register_tiles(accum, a[1], b[1]);
    }
  }
  // K is a multiple of 256.
  #pragma clang loop unroll(disable)
  for (uint k = K / 512u * 512u; k < K; k += 64u) {
    register_tile a[2], b[2];
    load_activation_tiles<full_rows>(a, A + k, rows, origin);
    decode_weight_tiles(b, B, tables, k, origin);
    multiply_register_tiles(accum, a[0], b[0]);
    multiply_register_tiles(accum, a[1], b[1]);
  }
  C += size_t(origin.y) * N + origin.x;
  #pragma clang loop unroll(full)
  for (short m = 0; m < 2; ++m) {
    #pragma clang loop unroll(full)
    for (short n = 0; n < 2; ++n) {
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; ++i) {
        const short row = m * 16 + (i / 4) * 8;
        const short col = n * 16 + i % 4;
        if (full_rows || row < rows - origin.y)
          C[size_t(row) * N + col] = ({{IO_TYPE}})(float(accum.fragments[m * 2 + n][i]) *
            float(A_scale[origin.y + row]) * float(B_scale[origin.x + col]));
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup({{SIMDGROUPS}} * 32)]] void segmented_scaled_gemm(
    device int8_t* A_buf [[buffer(0)]],
    device const uchar* B_buf [[buffer(1)]],
    device {{IO_TYPE}}* C_buf [[buffer(2)]],
    device const {{IO_TYPE}}* A_scale_buf [[buffer(3)]],
    device const {{IO_TYPE}}* B_scale_buf [[buffer(4)]],
    device const TileRecord* records [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
  const TileRecord record = records[tgid.x];
  const uint n_start = tgid.y * N_block;
  const uint local_row = record.local_row;
  if (n_start >= N || local_row >= record.count)
    return;
  // Packed formats need N % 256 == 0, so every column tile is full.
  const uint m_count = min(M_block, record.count - local_row);
  const uint col = sgid * 32u;
  const short rows = m_count;
  const uint global_row = record.row_offset + local_row;
  A_buf += (ulong)global_row * K;
  B_buf += ((ulong)record.expert * N + n_start + col) * {{PACKED_ROW_BYTES}};
  C_buf += (ulong)global_row * N + n_start + col;
  A_scale_buf += global_row;
  B_scale_buf += (ulong)record.expert * N + n_start + col;
)";
    if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
      source += R"(
  threadgroup uint2 grid[256];
  threadgroup uint2 signs[128];
  for (uint i = sgid * 32u + lane; i < 256u; i += {{SIMDGROUPS}}u * 32u)
    grid[i] = uint2(iq2_grid[i * 2u], iq2_grid[i * 2u + 1u]);
  for (uint i = sgid * 32u + lane; i < 128u; i += {{SIMDGROUPS}}u * 32u)
    signs[i] = uint2(iq2_signs[i * 2u], iq2_signs[i * 2u + 1u]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const decode_tables tables = { grid, signs };
)";
    } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
      source += R"(
  threadgroup uint grid[256];
  for (uint i = sgid * 32u + lane; i < 256u; i += {{SIMDGROUPS}}u * 32u)
    grid[i] = iq3_grid[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const decode_tables tables = { grid };
)";
    } else {
      source += R"(
  const decode_tables tables = {};
)";
    }
    source += R"(
  // The row check uses the whole threadgroup tile, because multiply_register
  // contains threadgroup barriers.
  if (m_count == M_block)
    multiply_register<true>(A_buf, B_buf, tables, C_buf, A_scale_buf, B_scale_buf, rows, lane);
  else
    multiply_register<false>(A_buf, B_buf, tables, C_buf, A_scale_buf, B_scale_buf, rows, lane);
}
)";
    return source.ToString();
  }
  if (format) {
    source += R"(
kernel void segmented_scaled_gemm(
    device int8_t* A_buf [[buffer(0)]],
    device const uchar* B_buf [[buffer(1)]],
    device {{IO_TYPE}}* C_buf [[buffer(2)]],
    device const {{IO_TYPE}}* A_scale_buf [[buffer(3)]],
    device const {{IO_TYPE}}* B_scale_buf [[buffer(4)]],
    device const TileRecord* records [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
  const TileRecord record = records[tgid.x];
  const uint n_start = tgid.y * N_block;
  const uint local_row = record.local_row;
  if (n_start >= N || local_row >= record.count)
    return;
  const uint m_count = min(M_block, record.count - local_row);
  const uint n_count = min(N_block, N - n_start);
  const uint global_row = record.row_offset + local_row;
  A_buf += (ulong)global_row * K;
  B_buf += ((ulong)record.expert * N + n_start) * {{PACKED_ROW_BYTES}};
  C_buf += (ulong)global_row * N + n_start;
  A_scale_buf += global_row;
  B_scale_buf += (ulong)record.expert * N + n_start;
  threadgroup uint weight_words[{{BLOCK_N}} * {{BLOCK_K}} / 4];
  threadgroup int8_t* weights = (threadgroup int8_t*)weight_words;
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, m_count));
  auto B = tensor<threadgroup int8_t, extents<int32_t, {{BLOCK_K}}, {{BLOCK_N}}>, tensor_inline>(weights, extents<int32_t, {{BLOCK_K}}, {{BLOCK_N}}>());
  constexpr auto desc = matmul2d_descriptor({{BLOCK_M}}, {{BLOCK_N}}, {{BLOCK_K}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<{{SIMDGROUPS}}>> op;
  auto mA = A.slice<{{BLOCK_K}}, dynamic_extent>(0, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(B), int32_t>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT[i] = 0;
)";
    if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S || format == CCV_NNC_QX_8I_ROWWISE_IQ1_M) {
      source += R"(
  // Each thread expands whole 32-weight units (one IQ1_S group or two IQ1_M
  // groups). Their payload words for the next slice load while the current
  // slice multiplies. The tile starts at a multiple of 128 rows, a 4-byte
  // boundary, so the bit offsets below are relative to aligned words.
  constexpr uint units_per_thread = {{BLOCK_N}}u * {{BLOCK_K}}u / 32u / ({{SIMDGROUPS}}u * 32u);
  static_assert(units_per_thread * {{SIMDGROUPS}}u * 32u * 32u == {{BLOCK_N}}u * {{BLOCK_K}}u, "units must divide evenly");
  device const uint* B_words = (device const uint*)B_buf;
  uint3 payload_words[units_per_thread];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < units_per_thread; ++i) {
    const uint g = tid + i * {{SIMDGROUPS}}u * 32u;
    const uint bit = g / ({{BLOCK_K}}u / 32u) * ({{PACKED_ROW_BYTES}} * 8u) + g % ({{BLOCK_K}}u / 32u) * {{IQ1_UNIT_BITS}}u;
    device const uint* p = B_words + (bit >> 5);
    payload_words[i] = uint3(p[0], p[1], p[2]);
  }
)";
    }
    source += R"(
  for (uint k = 0; k < K; k += {{BLOCK_K}}u) {
)";
    if (format == CCV_NNC_QX_8I_ROWWISE_IQ1_S || format == CCV_NNC_QX_8I_ROWWISE_IQ1_M) {
      source += R"(
    #pragma clang loop unroll(full)
    for (uint i = 0; i < units_per_thread; ++i) {
      const uint g = tid + i * {{SIMDGROUPS}}u * 32u;
      const uint row = g / ({{BLOCK_K}}u / 32u);
      const uint unit = g % ({{BLOCK_K}}u / 32u);
      const uint bit = row * ({{PACKED_ROW_BYTES}} * 8u) + (k / 32u + unit) * {{IQ1_UNIT_BITS}}u;
      const uint3 words = payload_words[i];
      if (k + {{BLOCK_K}}u < K) {
        device const uint* p = B_words + ((bit + {{BLOCK_K}}u / 32u * {{IQ1_UNIT_BITS}}u) >> 5);
        payload_words[i] = uint3(p[0], p[1], p[2]);
      }
      iq1_store_unit_bytes({{IQ1_UNIT}}(iq1_payload_words(words, bit & 31u)), (threadgroup uint2*)(weight_words + row * ({{BLOCK_K}}u / 4u) + unit * 8u));
    }
)";
    } else {
      source += R"(
    for (uint g = tid; g < {{BLOCK_N}}u * ({{BLOCK_K}}u / {{GROUP_SIZE}}u); g += {{SIMDGROUPS}}u * 32u) {
      const uint row = g / ({{BLOCK_K}}u / {{GROUP_SIZE}}u);
      const uint group = g % ({{BLOCK_K}}u / {{GROUP_SIZE}}u);
      int q8[{{GROUP_SIZE}}];
      decode_group(B_buf + (ulong)row * {{PACKED_ROW_BYTES}}, k / {{GROUP_SIZE}}u + group, q8);
      #pragma clang loop unroll(full)
      for (uint j = 0; j < {{GROUP_SIZE}}u / 4u; ++j)
        ((threadgroup char4*)weights)[row * ({{BLOCK_K}}u / 4u) + group * ({{GROUP_SIZE}}u / 4u) + j] =
          char4((char)q8[j * 4u], (char)q8[j * 4u + 1u], (char)q8[j * 4u + 2u], (char)q8[j * 4u + 3u]);
    }
)";
    }
    source += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto aTile = A.slice<{{BLOCK_K}}, dynamic_extent>(k, 0);
    op.run(aTile, B, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      const auto idx = cT.get_multidimensional_index(i);
      const uint row = (uint)idx[1];
      const uint col = (uint)idx[0];
      if (row < m_count && col < n_count)
        C_buf[row * N + col] = ({{IO_TYPE}})((float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col]);
    }
  }
}
)";
    return source.ToString();
  }
  source += R"(
kernel void segmented_scaled_gemm(
    device int8_t* A_buf [[buffer(0)]],
    device int8_t* B_buf [[buffer(1)]],
    device {{IO_TYPE}}* C_buf [[buffer(2)]],
    device const {{IO_TYPE}}* A_scale_buf [[buffer(3)]],
    device const {{IO_TYPE}}* B_scale_buf [[buffer(4)]],
    device const TileRecord* records [[buffer(5)]],
)";
  if (useBias) {
    source += R"(
    device const {{IO_TYPE}}* bias_buf [[buffer(6)]],
)";
  }
  source += R"(
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const TileRecord record = records[tgid.x];
  const uint N_block_start = tgid.y * N_block;
  if (N_block_start >= N)
    return;
  const uint local_row = record.local_row;
  if (local_row >= record.count)
    return;
  const uint M_dynamic = min(M_block, record.count - local_row);
  const uint N_dynamic = min(N_block, N - N_block_start);
  const uint global_row = record.row_offset + local_row;

  A_buf += global_row * K;
  B_buf += record.expert * (ulong)(N * K) + N_block_start * K;
  C_buf += global_row * N + N_block_start;
  A_scale_buf += global_row;
  B_scale_buf += record.expert * N + N_block_start;
)";
  if (useBias) {
    source += R"(
  bias_buf += record.expert * N + N_block_start;
)";
  }
  source += R"(

  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_dynamic));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_dynamic));

  if (M_dynamic == M_block && N_dynamic == N_block) {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_M}},
        {{BLOCK_N}},
        {{BLOCK_K}},
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> matmul_op;
    auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(0, 0);
    auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(0, 0);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k + K_block <= K; k += K_block) {
      auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(k, 0);
      auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(k, 0);
      matmul_op.run(mA, mB, cT);
    }
    if (K % K_block != 0) {
      constexpr auto residual_descriptor = matmul2d_descriptor(
          {{BLOCK_M}},
          {{BLOCK_N}},
          dynamic_length_v<int>,
          false,
          true,
          true,
          matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<residual_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> residual_op;
      auto mA = A.slice<dynamic_extent, {{BLOCK_M}}>(K / K_block * K_block, 0);
      auto mB = B.slice<dynamic_extent, {{BLOCK_N}}>(K / K_block * K_block, 0);
      residual_op.run(mA, mB, cT);
    }
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = (uint)idx[1];
        const uint col = (uint)idx[0];
        float value = (float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col];
)";
  if (useBias) {
    source += R"(
        value += (float)bias_buf[col];
)";
  }
  source += R"(
        C_buf[row * N + col] = ({{IO_TYPE}})value;
      }
    }
  } else {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_M}},
        {{BLOCK_N}},
        dynamic_length_v<int>,
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> matmul_op;
    auto mA = A.slice(0, 0);
    auto mB = B.slice(0, 0);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    matmul_op.run(mA, mB, cT);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = (uint)idx[1];
        const uint col = (uint)idx[0];
        if (row < M_dynamic && col < N_dynamic) {
          float value = (float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col];
)";
  if (useBias) {
    source += R"(
          value += (float)bias_buf[col];
)";
  }
  source += R"(
          C_buf[row * N + col] = ({{IO_TYPE}})value;
        }
      }
    }
  }
}
)";
  return source.ToString();
}
