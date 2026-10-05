#include "NAInt8MatMulKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

namespace {

static uint32_t ceilLog2(uint64_t x) noexcept {
  if (x <= 1)
    return 0;
  --x;
  uint32_t bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

}

NAInt8MatMulKernel::NAInt8MatMulKernel(
    NAInt8MatMulKernelDescriptor descriptor,
    MTL::Device *const device)
{
  blockDimensions = descriptor.blockDimensions;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  ioPrecision = descriptor.ioPrecision;
  useRegisterOperands = descriptor.useRegisterOperands;
  useBias = descriptor.useBias;
  loadM = descriptor.loadM;
  useLeadingDimensions = descriptor.useLeadingDimensions;
  activationQuantizeThreads = descriptor.activationQuantizeThreads;
  activationHadamard256 = descriptor.activationHadamard256;
  CCV_NNC_MFA_PRECONDITION(activationQuantizeThreads > 0 && activationQuantizeThreads % 32 == 0 && activationQuantizeThreads <= 1024);
  groupM = descriptor.groupM;
  groupN = descriptor.groupN;

  CCV_NNC_MFA_PRECONDITION(!useRegisterOperands || !useLeadingDimensions);
  if (useRegisterOperands) {
    CCV_NNC_MFA_PRECONDITION(blockDimensions[0] == 64 && blockDimensions[1] == 128 && blockDimensions[2] == 32);
    CCV_NNC_MFA_PRECONDITION(executionSIMDGroups == 8);
  }
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t NAInt8MatMulKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept {
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAInt8MatMulKernel::threadsPerThreadgroup(MTL::ComputePipelineState *const pipelineState) const noexcept {
  // One SIMD group along x, four output-column groups along y, two
  // output-row groups along z. Flattening this layout regresses large GEMMs.
  if (useRegisterOperands)
    return MTL::Size(pipelineState->threadExecutionWidth(), 4, 2);
  return MTL::Size(threadgroupSize(pipelineState), 1, 1);
}

MTL::Size NAInt8MatMulKernel::threadgroupsPerGrid(uint32_t M, uint32_t N, uint32_t batchDimension) const noexcept {
  auto ceilDivide =
    [=](int64_t target, uint16_t granularity) -> int64_t {
      return (target + int64_t(granularity) - 1) / int64_t(granularity);
    };
  if (useRegisterOperands) {
    CCV_NNC_MFA_PRECONDITION(batchDimension == 1);
    return MTL::Size(ceilDivide(N, blockDimensions[1]), ceilDivide(M, blockDimensions[0]), 1);
  }
  const int64_t M_tiles = ceilDivide(int64_t(M), blockDimensions[0]);
  const int64_t N_tiles = ceilDivide(int64_t(N), blockDimensions[1]);
  const uint32_t M_bits = ceilLog2(M_tiles);
  const uint32_t N_bits = ceilLog2(N_tiles);
  return MTL::Size(int64_t(1) << (M_bits + N_bits), 1, batchDimension);
}

std::string NAInt8MatMulKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("BLOCK_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_K", std::to_string(blockDimensions[2]));
  source.SetValue("SIMDGROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("QUANT_THREADS", std::to_string(activationQuantizeThreads));
  source.SetValue("QUANT_SIMDGROUPS", std::to_string(activationQuantizeThreads / 32));
  source.SetValue("GROUP_M", std::to_string(groupM));
  source.SetValue("GROUP_N", std::to_string(groupN));
  source.SetValue("IO_TYPE", ioPrecision.name());
  source.SetValue("LEADING_DIMENSION_CONSTANTS", useLeadingDimensions ? "constant uint A_leading_dimension [[function_constant(22)]];\nconstant uint C_leading_dimension [[function_constant(23)]];\n" : "");
  source.SetValue("QUANTIZATION_BASES", useLeadingDimensions ? "  const uint src_base = row * A_leading_dimension;\n  const uint dst_base = row * K;\n" : "  const uint base = row * K;\n");
  source.SetValue("QUANTIZATION_VECTOR_BASES", useLeadingDimensions ? "    const uint src_vector_base = src_base / 4;\n    const uint dst_vector_base = dst_base / 4;\n" : "    const uint vector_base = row * vectors_per_row;\n");
  source.SetValue("QUANTIZATION_SOURCE_VECTOR_BASE", useLeadingDimensions ? "src_vector_base" : "vector_base");
  source.SetValue("QUANTIZATION_DESTINATION_VECTOR_BASE", useLeadingDimensions ? "dst_vector_base" : "vector_base");
  source.SetValue("QUANTIZATION_SOURCE_BASE", useLeadingDimensions ? "src_base" : "base");
  source.SetValue("QUANTIZATION_DESTINATION_BASE", useLeadingDimensions ? "dst_base" : "base");
  source.SetValue("C_LEADING_DIMENSION", useLeadingDimensions ? "C_leading_dimension" : "N");
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline uint compact_morton_even_bits(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

inline uint2 morton_decode_2d(uint code) {
  return uint2(compact_morton_even_bits(code),
               compact_morton_even_bits(code >> 1));
}

inline uint lower_bits_mask(uint bit_count) {
  if (bit_count == 0)
    return 0;
  return (1u << bit_count) - 1;
}

inline uint2 morton_decode_rectangular_2d(uint code,
                                          uint x_bits,
                                          uint y_bits) {
  const uint paired_bits = min(x_bits, y_bits);
  const uint paired_code = code & lower_bits_mask(paired_bits * 2);
  uint2 tile = morton_decode_2d(paired_code);
  uint tail = code >> (paired_bits * 2);
  if (x_bits > paired_bits) {
    const uint x_extra_bits = x_bits - paired_bits;
    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;
    tail >>= x_extra_bits;
  }
  if (y_bits > paired_bits) {
    tile.y |= tail << paired_bits;
  }
  return tile;
}

constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];
constant bool batched [[function_constant(11)]];
constant uint B_batch_stride [[function_constant(16)]];
constant uint bias_batch_stride [[function_constant(18)]];
constant uint B_scale_batch_stride [[function_constant(20)]];
{{LEADING_DIMENSION_CONSTANTS}}
inline float quantize_reduce_max(float value,
                                 threadgroup float* scratch,
                                 ushort sgid,
                                 ushort lane_id)
{
  value = max(value, simd_shuffle_xor(value, 16));
  value = max(value, simd_shuffle_xor(value, 8));
  value = max(value, simd_shuffle_xor(value, 4));
  value = max(value, simd_shuffle_xor(value, 2));
  value = max(value, simd_shuffle_xor(value, 1));
  if (lane_id == 0)
    scratch[sgid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    value = lane_id < {{QUANT_SIMDGROUPS}} ? scratch[lane_id] : 0.0f;
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane_id == 0)
      scratch[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

)";
  if (!loadM) {
    source += R"(
constant uint M [[function_constant(0)]];
constant uint A_batch_stride [[function_constant(15)]];
constant uint C_batch_stride [[function_constant(17)]];
constant uint A_scale_batch_stride [[function_constant(19)]];
constant uint A_packed_batch_stride [[function_constant(21)]];
)";
  }
  source += R"(
kernel void quantize_activation(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device {{IO_TYPE}}* scales [[buffer(2)]],
)";
  if (loadM) {
    source += R"(
    const device uint *loadM_buf [[buffer(3)]],
)";
  }
  source += R"(
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
)";
  if (loadM) {
    source += R"(
  const uniform<uint> M = make_uniform(loadM_buf[0]);
  const uniform<uint> A_batch_stride = make_uniform(batched ? loadM_buf[1] : 0);
  const uniform<uint> A_packed_batch_stride = make_uniform(batched ? loadM_buf[3] : 0);
  const uniform<uint> A_scale_batch_stride = make_uniform(batched ? loadM_buf[4] : 0);
)";
  }
  source += R"(
  threadgroup float scratch[{{QUANT_SIMDGROUPS}}];
  const uint row = tgid.x;
  if (row >= M)
    return;
  if (batched) {
    src += A_batch_stride * tgid.z;
    dst += A_packed_batch_stride * tgid.z;
    scales += A_scale_batch_stride * tgid.z;
  }
  float local_max = 0.0f;
{{QUANTIZATION_BASES}}
)";
  if (activationHadamard256) {
    source += R"(
  // One SIMD group rotates 256 consecutive features, with eight floats per
  // lane. Keep the rotated row in registers through the row-scale reduction.
  // H4 has +1 everywhere except its anti-diagonal (-1). Its fourth Kronecker
  // power, normalized by 1/16, is the regular Hadamard used by ConvRot.
  const uint groups_per_simd = (K / 256 + {{QUANT_SIMDGROUPS}} - 1) / {{QUANT_SIMDGROUPS}};
  // Metal cannot size a thread array with a function constant. Specialization
  // eliminates unused slots; the descriptor bounds K to 65536 at 256 threads.
  float4 rotated[64];
  #pragma clang loop unroll(full)
  for (uint g = 0; g < groups_per_simd; ++g) {
    const uint group = g * {{QUANT_SIMDGROUPS}} + sgid;
    float4 x[2];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 2; ++j) {
      const uint k = group * 256 + j * 128 + lane_id * 4;
      float4 v = float4(0);
      if (group < K / 256) {
        const uint offset = {{QUANTIZATION_SOURCE_BASE}} + k;
        v = float4(src[offset], src[offset + 1], src[offset + 2], src[offset + 3]);
      }
      v = (v.x + v.y + v.z + v.w) - 2.0f * v.wzyx;
      #pragma clang loop unroll(full)
      for (ushort stride = 1; stride <= 4; stride *= 4) {
        const float4 a = simd_shuffle_xor(v, stride);
        const float4 b = simd_shuffle_xor(v, ushort(stride * 2));
        const float4 c = simd_shuffle_xor(v, ushort(stride * 3));
        v = ((v + a) + b) - c;
      }
      x[j] = v;
    }
    const float4 a = simd_shuffle_xor(x[0], 16);
    const float4 b = simd_shuffle_xor(x[1], 16);
    rotated[g * 2] = (((x[0] + a) + x[1]) - b) * (1.0f / 16.0f);
    rotated[g * 2 + 1] = (((x[1] + b) + x[0]) - a) * (1.0f / 16.0f);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 2; ++j) {
      const float4 v = abs(rotated[g * 2 + j]);
      local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
    }
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[row] = ({{IO_TYPE}})scale;
  device char4* dst4 = reinterpret_cast<device char4*>(dst);
  #pragma clang loop unroll(full)
  for (uint g = 0; g < groups_per_simd; ++g) {
    const uint group = g * {{QUANT_SIMDGROUPS}} + sgid;
    if (group < K / 256) {
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 2; ++j) {
        const int4 rounded = int4(rint(rotated[g * 2 + j] * inv_scale));
        dst4[{{QUANTIZATION_DESTINATION_BASE}} / 4 + group * 64 + j * 32 + lane_id] = char4(clamp(rounded, int4(-127), int4(127)));
      }
    }
  }
)";
  } else {
    source += R"(
  if ((K % 4) == 0) {
    const uint vectors_per_row = K / 4;
    device const {{IO_TYPE}}4* src4 = reinterpret_cast<device const {{IO_TYPE}}4*>(src);
    device char4* dst4 = reinterpret_cast<device char4*>(dst);
{{QUANTIZATION_VECTOR_BASES}}    for (uint i = tid; i < vectors_per_row; i += {{QUANT_THREADS}}) {
      const float4 value = float4(src4[{{QUANTIZATION_SOURCE_VECTOR_BASE}} + i]);
      local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])), max(fabs(value[2]), fabs(value[3]))));
    }
    const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
    const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0)
      scales[row] = ({{IO_TYPE}})scale;
    for (uint i = tid; i < vectors_per_row; i += {{QUANT_THREADS}}) {
      const int4 rounded = int4(rint(float4(src4[{{QUANTIZATION_SOURCE_VECTOR_BASE}} + i]) * inv_scale));
      dst4[{{QUANTIZATION_DESTINATION_VECTOR_BASE}} + i] = char4(clamp(rounded, int4(-127), int4(127)));
    }
  } else {
    for (uint i = tid; i < K; i += {{QUANT_THREADS}})
      local_max = max(local_max, fabs((float)src[{{QUANTIZATION_SOURCE_BASE}} + i]));
    const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
    const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0)
      scales[row] = ({{IO_TYPE}})scale;
    for (uint i = tid; i < K; i += {{QUANT_THREADS}}) {
      const int rounded = (int)rint((float)src[{{QUANTIZATION_SOURCE_BASE}} + i] * inv_scale);
      dst[{{QUANTIZATION_DESTINATION_BASE}} + i] = (int8_t)clamp(rounded, -127, 127);
    }
  }
)";
  }
  source += "}";
  if (useRegisterOperands) {
    // Fragment layout and MPP register assembly adapted from MLX Steel's
    // gemm_nax.h (Apple Inc., MIT); see ../3rdparty/mlx/README and LICENSE.
    source += R"(
#pragma METAL fp math_mode(safe)

// A 16x16 fragment has eight values per lane: four adjacent columns in
// each of two rows eight apart. A SIMD group owns a 2x2 array of fragments.
__attribute__((always_inline)) inline short2 register_fragment_coordinate(ushort lane) {
  const short quad = lane >> 2;
  return short2(((quad & 2) | (lane & 1)) * 4,
                (quad & 4) | ((lane >> 1) & 3));
}

template<typename T>
struct register_tile {
  vec<T, 8> fragments[4];
};

// Both packed A and transposed B have contiguous K. Full K32 steps need
// only a row bound on edge tiles; only the last short K step masks columns.
template<bool full_rows, bool full_k>
__attribute__((always_inline)) inline void load_register_tile(thread register_tile<int8_t>& tile,
    const device int8_t* src, short rows, short columns, short2 origin) {
  src += origin.y * K + origin.x;
  #pragma clang loop unroll(full)
  for (short m = 0; m < 2; ++m) {
    #pragma clang loop unroll(full)
    for (short k = 0; k < 2; ++k) {
      #pragma clang loop unroll(full)
      for (short r = 0; r < 2; ++r) {
        const short row = m * 16 + r * 8;
        if (full_rows || row < rows - origin.y) {
          #pragma clang loop unroll(full)
          for (short c = 0; c < 4; ++c) {
            const short col = k * 16 + c;
            tile.fragments[m * 2 + k][r * 4 + c] =
                (full_k || col < columns - origin.x) ? src[row * K + col] : 0;
          }
        } else {
          #pragma clang loop unroll(full)
          for (short c = 0; c < 4; ++c)
            tile.fragments[m * 2 + k][r * 4 + c] = 0;
        }
      }
    }
  }
}

// Four M16/N32/K16 operations form a SIMD group's M32/N32/K32 update.
// B is stored as [N,K], so its two N fragments become the right operand
// of a transposed MPP multiply. The accumulator stays INT32 throughout K.
__attribute__((always_inline)) inline void multiply_register_tiles(thread register_tile<int32_t>& accum,
    thread const register_tile<int8_t>& a, thread const register_tile<int8_t>& b) {
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

template<bool full_m, bool full_n>
inline void multiply_register(const device int8_t* A, const device int8_t* B,
    device {{IO_TYPE}}* C, const device {{IO_TYPE}}* A_scale,
    const device {{IO_TYPE}}* B_scale,
)";
    if (useBias)
      source += "    const device {{IO_TYPE}}* bias,";
    source += R"(
    short rows, short columns, ushort lane) {
  const short2 origin = register_fragment_coordinate(lane);
  register_tile<int32_t> accum;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 4; ++i)
    accum.fragments[i] = vec<int32_t, 8>(0);
  const bool has_output = rows > 0 && columns > 0;

  // Keep the eight SIMD groups in step every K512. Empty edge groups must
  // participate in every threadgroup barrier, even when they skip arithmetic.
  #pragma clang loop unroll(disable)
  for (uint block = 0; block < K / 512; ++block) {
    threadgroup_barrier(mem_flags::mem_none);
    if ((!full_m || !full_n) && !has_output)
      continue;
    // Expose pairs of K32 loads / multiplies without unrolling the entire K.
    #pragma clang loop unroll_count(2)
    for (uint k = 0; k < 512; k += 32) {
      register_tile<int8_t> a, b;
      load_register_tile<full_m, true>(a, A + k, rows, 32, origin);
      load_register_tile<full_n, true>(b, B + k, columns, 32, origin);
      multiply_register_tiles(accum, a, b);
    }
    A += 512;
    B += 512;
  }
  if (K % 512 != 0) {
    simdgroup_barrier(mem_flags::mem_none);
    if ((!full_m || !full_n) && !has_output)
      return;
    #pragma clang loop unroll(disable)
    for (uint k = 0; k < K % 512; k += 32) {
      register_tile<int8_t> a, b;
      const short remaining = K % 512 - k;
      if (remaining >= 32) {
        load_register_tile<full_m, true>(a, A + k, rows, 32, origin);
        load_register_tile<full_n, true>(b, B + k, columns, 32, origin);
      } else {
        load_register_tile<false, false>(a, A + k, rows, remaining, origin);
        load_register_tile<false, false>(b, B + k, columns, remaining, origin);
      }
      multiply_register_tiles(accum, a, b);
    }
  }
  if (!has_output)
    return;
  register_tile<{{IO_TYPE}}> output;
  #pragma clang loop unroll(full)
  for (short m = 0; m < 2; ++m) {
    #pragma clang loop unroll(full)
    for (short n = 0; n < 2; ++n) {
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; ++i) {
        const short row = m * 16 + origin.y + (i / 4) * 8;
        const short col = n * 16 + origin.x + i % 4;
        const float a = row < rows ? float(A_scale[row]) : 0.0f;
        const float b = col < columns ? float(B_scale[col]) : 0.0f;
        float value = float(accum.fragments[m * 2 + n][i]) * a * b;
)";
    if (useBias)
      source += "        if (col < columns) value += float(bias[col]);";
    source += R"(
        output.fragments[m * 2 + n][i] = {{IO_TYPE}}(value);
      }
    }
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
        if ((full_m && full_n) || (row < rows - origin.y && col < columns - origin.x))
          C[size_t(row) * N + col] = output.fragments[m * 2 + n][i];
      }
    }
  }
}
)";
  }
  source += useRegisterOperands ?
      "[[kernel, max_total_threads_per_threadgroup(256)]] void int8_matmul(" :
      "kernel void int8_matmul(";
  source += R"(
    device int8_t *A_buf [[buffer(0)]],
    device int8_t *B_buf [[buffer(1)]],
    device {{IO_TYPE}} *C_buf [[buffer(2)]],
    device const {{IO_TYPE}} *A_scale_buf [[buffer(3)]],
    device const {{IO_TYPE}} *B_scale_buf [[buffer(4)]],
)";
  if (useBias) {
    source += R"(
    device const {{IO_TYPE}} *bias_buf [[buffer(5)]],
)";
  }
  if (loadM) {
    source += useBias ? R"(
    const device uint *loadM_buf [[buffer(6)]],
)" : R"(
    const device uint *loadM_buf [[buffer(5)]],
)";
  }
  if (useRegisterOperands) {
    source += R"(
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
)";
  }
  source += R"(
    uint3 tgid [[threadgroup_position_in_grid]])
{
)";
  if (loadM) {
    source += R"(
  const uniform<uint> M = make_uniform(loadM_buf[0]);
  const uniform<uint> A_batch_stride = make_uniform(batched ? loadM_buf[1] : 0);
  const uniform<uint> C_batch_stride = make_uniform(batched ? loadM_buf[2] : 0);
  const uniform<uint> A_packed_batch_stride = make_uniform(batched ? loadM_buf[3] : 0);
  const uniform<uint> A_scale_batch_stride = make_uniform(batched ? loadM_buf[4] : 0);
)";
  }
  source += R"(
  if (batched) {
    A_buf += A_packed_batch_stride * tgid.z;
    B_buf += B_batch_stride * tgid.z;
    C_buf += C_batch_stride * tgid.z;
    A_scale_buf += A_scale_batch_stride * tgid.z;
    B_scale_buf += B_scale_batch_stride * tgid.z;
)";
  if (useBias) {
    source += R"(
    bias_buf += bias_batch_stride * tgid.z;
)";
  }
  source += R"(
  }

)";
  if (useRegisterOperands) {
    source += R"(
  // Eight SIMD groups cover 64x128. Alignment decisions use the complete
  // threadgroup tile, because multiply_register contains threadgroup barriers.
  const int row = int(tgid.y) * 64 + (sgid / 4) * 32;
  const int col = int(tgid.x) * 128 + (sgid % 4) * 32;
  const short rows = min(32, int(M) - row);
  const short columns = min(32, int(N) - col);
  A_buf += size_t(row) * K;
  B_buf += size_t(col) * K;
  C_buf += size_t(row) * N + col;
  A_scale_buf += row;
  B_scale_buf += col;
)";
    if (useBias)
      source += "  bias_buf += col;";
    const std::string arguments = "(A_buf, B_buf, C_buf, A_scale_buf, B_scale_buf, " +
        std::string(useBias ? "bias_buf, " : "") + "rows, columns, lane);";
    source += "  if (M % 64 == 0 || tgid.y * 64 + 64 <= M) {";
    source += "    if (N % 128 == 0 || tgid.x * 128 + 128 <= N)";
    source += "      multiply_register<true, true>" + arguments;
    source += "    else";
    source += "      multiply_register<true, false>" + arguments;
    source += "  } else {";
    source += "    if (N % 128 == 0 || tgid.x * 128 + 128 <= N)";
    source += "      multiply_register<false, true>" + arguments;
    source += "    else";
    source += "      multiply_register<false, false>" + arguments;
    source += "  }";
    source += "}";
    return source.ToString();
  }
  source += R"(
  const uint M_tiles = (M + {{BLOCK_M}} - 1) / {{BLOCK_M}};
  const uint N_tiles = (N + {{BLOCK_N}} - 1) / {{BLOCK_N}};
  const uint M_tile_bits = M_tiles <= 1 ? 0 : 32 - clz(M_tiles - 1);
  const uint N_tile_bits = N_tiles <= 1 ? 0 : 32 - clz(N_tiles - 1);
  uint2 morton_tile = morton_decode_rectangular_2d(tgid.x, N_tile_bits, M_tile_bits);
  tgid.x = morton_tile.x;
  tgid.y = morton_tile.y;
  if (tgid.x >= N_tiles || tgid.y >= M_tiles) {
    return;
  }

  const uint M_block_start = tgid.y * {{BLOCK_M}};
  const uint M_block_size = min((uint){{BLOCK_M}}, M - M_block_start);
  const uint N_block_start = tgid.x * {{BLOCK_N}};
  const uint N_block_size = min((uint){{BLOCK_N}}, N - N_block_start);
  const uint M_group_start = {{GROUP_M}} ? (M_block_start / {{GROUP_M}}) * {{GROUP_M}} : M_block_start;
  const uint M_group_offset = M_block_start - M_group_start;
  const uint M_group_size = M - M_group_start;
  const uint N_group_start = {{GROUP_N}} ? (N_block_start / {{GROUP_N}}) * {{GROUP_N}} : N_block_start;
  const uint N_group_offset = N_block_start - N_group_start;
  const uint N_group_size = N - N_group_start;

  // Widen before multiplying: packed weights can span more than 4 GiB.
  A_buf += size_t(M_group_start) * K;
  B_buf += size_t(N_group_start) * K;
  C_buf += size_t(M_group_start) * {{C_LEADING_DIMENSION}};
  A_scale_buf += M_group_start;
  B_scale_buf += N_group_start;
)";
  if (useBias) {
    source += R"(
  bias_buf += N_group_start;
)";
  }
  source += R"(
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_group_size));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_group_size));
  if (N_block_start + {{BLOCK_N}} - 1 < N && M_block_start + {{BLOCK_M}} - 1 < M) {
    constexpr auto matmul_descriptor = matmul2d_descriptor(
        {{BLOCK_M}},
        {{BLOCK_N}},
        {{BLOCK_K}},
        false,
        true,
        true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<matmul_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> matmul_op;

    auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(0, M_group_offset);
    auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(0, N_group_offset);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k + {{BLOCK_K}} <= K; k += {{BLOCK_K}}) {
      auto mA = A.slice<{{BLOCK_K}}, {{BLOCK_M}}>(k, M_group_offset);
      auto mB = B.slice<{{BLOCK_K}}, {{BLOCK_N}}>(k, N_group_offset);
      matmul_op.run(mA, mB, cT);
    }
    if (K % {{BLOCK_K}} != 0) {
      constexpr auto residual_descriptor = matmul2d_descriptor(
          {{BLOCK_M}},
          {{BLOCK_N}},
          dynamic_length_v<int>,
          false,
          true,
          true,
          matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<residual_descriptor, execution_simdgroups<{{SIMDGROUPS}}>> residual_op;
      auto mA = A.slice<dynamic_extent, {{BLOCK_M}}>(K / {{BLOCK_K}} * {{BLOCK_K}}, M_group_offset);
      auto mB = B.slice<dynamic_extent, {{BLOCK_N}}>(K / {{BLOCK_K}} * {{BLOCK_K}}, N_group_offset);
      residual_op.run(mA, mB, cT);
    }
    auto mC = C_buf + size_t(M_group_offset) * {{C_LEADING_DIMENSION}} + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = M_group_offset + (uint)idx[1];
        const uint col = N_group_offset + (uint)idx[0];
        float value = (float)cT[i] * (float)A_scale_buf[row] * (float)B_scale_buf[col];
)";
  if (useBias) {
    source += R"(
        value += (float)bias_buf[col];
)";
  }
  source += R"(
        mC[size_t(idx[1]) * {{C_LEADING_DIMENSION}} + idx[0]] = ({{IO_TYPE}})value;
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
    auto mA = A.slice(0, M_group_offset);
    auto mB = B.slice(0, N_group_offset);
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i))
        cT[i] = 0;
    }
    matmul_op.run(mA, mB, cT);
    auto mC = C_buf + size_t(M_group_offset) * {{C_LEADING_DIMENSION}} + N_block_start;
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const uint row = (uint)idx[1];
        const uint col = (uint)idx[0];
        if (col < N_block_size && row < M_block_size) {
          float value = (float)cT[i] *
              (float)A_scale_buf[M_group_offset + row] *
              (float)B_scale_buf[N_group_offset + col];
)";
  if (useBias) {
    source += R"(
          value += (float)bias_buf[N_group_offset + col];
)";
  }
  source += R"(
          mC[size_t(row) * {{C_LEADING_DIMENSION}} + col] = ({{IO_TYPE}})value;
        }
      }
    }
  }
}
)";
  return source.ToString();
}
