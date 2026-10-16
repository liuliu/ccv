#include "SegmentedScaledGEMMKernel.hpp"
#include "SegmentedScaledGEMMKernelDescriptor.hpp"
#include "CodeWriter.hpp"
#include "Dequantize8iRowwiseXDescriptor.hpp"
#include "Dequantize8iRowwiseXKernel.hpp"
#include "../../ccv_nnc_8i_rowwise_packed_grids.inc"
#include "../ccv_nnc_mfa.hpp"

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
  if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS || format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS || format == CCV_NNC_QX_8I_ROWWISE_Q2_K)
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
  // Packed projections expand only the current weight tile. IQ2_XXS, IQ3_XXS and Q2_K
  // have dedicated tile decoders; other formats fall back to the dequantizer's group decoder.
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
    source += "constant uint iq2_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};";
  } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
    source += "constant uint iq3_grid[256] = {";
    for (uint32_t code = 0; code < 256; ++code) {
      uint32_t packed = 0;
      for (uint32_t j = 0; j < 4; ++j)
        packed |= ((ccv_nnc_8i_rowwise_packed_iq3xxs_grid[code] >> (j * 8)) & 255) >> 2 << (j * 8);
      source += std::to_string(packed) + (code == 255 ? "u};" : "u,");
    }
  } else if (format && format != CCV_NNC_QX_8I_ROWWISE_Q2_K)
    source += Dequantize8iRowwiseXKernel::createDecodeGroupSource(format);
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
  for (uint k = 0; k < K; k += {{BLOCK_K}}u) {
)";
    if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K) {
      source += R"(
    for (uint g = tid; g < {{BLOCK_N}}u * {{BLOCK_K}}u / 16u; g += {{SIMDGROUPS}}u * 32u) {
      const uint row = g / ({{BLOCK_K}}u / 16u);
      const uint group = g % ({{BLOCK_K}}u / 16u);
      // Q2_K stores sixteen 2-bit codes, a 6-bit scale and a 4-bit zero point.
      const uint bit_offset = (k / 16u + group) * 42u;
      const uint byte_offset = bit_offset / 8u;
      const uint shift = bit_offset % 8u;
      device const uchar* p = B_buf + (ulong)row * {{PACKED_ROW_BYTES}} + byte_offset;
      const ulong payload = (ulong(p[0]) | (ulong(p[1]) << 8) | (ulong(p[2]) << 16) |
        (ulong(p[3]) << 24) | (ulong(p[4]) << 32) | (ulong(p[5]) << 40)) >> shift;
      const uint q = uint(payload);
      const int scale = int((payload >> 32) & 63u) + 1;
      const int zero = int((payload >> 38) & 15u) * 8;
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 4; ++j) {
        const uint bits = q >> (j * 8u);
        const int4 values = int4(int(bits & 3u), int((bits >> 2) & 3u),
          int((bits >> 4) & 3u), int((bits >> 6) & 3u)) * scale - zero;
        ((threadgroup char4*)weights)[row * ({{BLOCK_K}}u / 4u) + group * 4u + j] = char4(values);
      }
    }
)";
    } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS) {
      source += R"(
    for (uint g = tid; g < {{BLOCK_N}}u * {{BLOCK_K}}u / 32u; g += {{SIMDGROUPS}}u * 32u) {
      const uint row = g / ({{BLOCK_K}}u / 32u);
      const uint group = g % ({{BLOCK_K}}u / 32u);
      device const uchar* p = B_buf + (ulong)row * (K / 4u) + k / 4u + group * 8u;
      const uint codes = ((device const uint*)p)[1];
      const uint scale = iq2_scales[codes >> 28];
      #pragma clang loop unroll(full)
      for (uint sg = 0; sg < 4; ++sg) {
        const uint grid = uint(p[sg]) * 2u;
        const uint signs = ((codes >> (sg * 7u)) & 127u) * 2u;
        #pragma clang loop unroll(full)
        for (uint word = 0; word < 2; ++word) {
          // Each byte is at most 7 * 32, so multiplying the packed word has no byte carries.
          const uint raw = iq2_grid[grid + word] * scale;
          const uint high = raw & 0x80808080u;
          const uint magnitude = (raw & 0x7f7f7f7fu) | (high - (high >> 7));
          const uint mask = iq2_signs[signs + word];
          // Magnitudes are in 1...127, so the byte-wise negation cannot carry into another byte.
          weight_words[row * ({{BLOCK_K}}u / 4u) + group * 8u + sg * 2u + word] = (magnitude ^ mask) + (mask & 0x01010101u);
        }
      }
    }
)";
    } else if (format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) {
      source += R"(
    for (uint g = tid; g < {{BLOCK_N}}u * {{BLOCK_K}}u / 16u; g += {{SIMDGROUPS}}u * 32u) {
      const uint row = g / ({{BLOCK_K}}u / 16u);
      const uint pair = g % ({{BLOCK_K}}u / 16u);
      // Two 28-bit IQ3_XXS groups of eight values occupy seven bytes.
      device const uchar* p = B_buf + (ulong)row * {{PACKED_ROW_BYTES}} + (k / 16u + pair) * 7u;
      const ulong payloads = ulong(p[0]) | (ulong(p[1]) << 8) | (ulong(p[2]) << 16) |
        (ulong(p[3]) << 24) | (ulong(p[4]) << 32) | (ulong(p[5]) << 40) | (ulong(p[6]) << 48);
      #pragma clang loop unroll(full)
      for (uint group = 0; group < 2; ++group) {
        const uint payload = uint(payloads >> (group * 28u));
        const uint scale = ((payload >> 24) & 15u) + 1u;
        #pragma clang loop unroll(full)
        for (uint word = 0; word < 2; ++word) {
          // Grid bytes are at most 15 and scales at most 16, so multiplying the packed word has no byte carries.
          const uint raw = iq3_grid[(payload >> (word * 8u)) & 255u] * scale;
          const uint high = raw & 0x80808080u;
          const uint magnitude = (raw & 0x7f7f7f7fu) | (high - (high >> 7));
          // Spread four sign bits to the low bit of each byte, then widen them to byte masks.
          const uint mask = ((((payload >> (16u + word * 4u)) & 15u) * 0x00204081u) & 0x01010101u) * 255u;
          // Magnitudes are in 1...127, so the byte-wise negation cannot carry into another byte.
          weight_words[row * ({{BLOCK_K}}u / 4u) + pair * 4u + group * 2u + word] = (magnitude ^ mask) + (mask & 0x01010101u);
        }
      }
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
