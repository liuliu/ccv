#include "ANERowwiseTransformKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

ANERowwiseTransformKernel::ANERowwiseTransformKernel(
    ANERowwiseTransformKernelDescriptor descriptor,
    MTL::Device* const device)
{
  memoryPrecision = descriptor.memoryPrecision;
  activationScaleThreads = 256;
  quantTileDimension = descriptor.supportsApple10 ? 64 : 32;
  quantBlockRows = descriptor.supportsApple10 ? 4 : 8;
  quantTilePad = descriptor.supportsApple10 ? 65 : 33;
  outputTileDimensions = simd::ushort2 { 16, 16 };

  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

MTL::Size ANERowwiseTransformKernel::activationScaleThreadgroupSize() const noexcept
{
  return MTL::Size(activationScaleThreads, 1, 1);
}

MTL::Size ANERowwiseTransformKernel::activationQuantizeThreadgroupSize() const noexcept
{
  return MTL::Size(quantTileDimension, quantBlockRows, 1);
}

MTL::Size ANERowwiseTransformKernel::outputDequantizeThreadgroupSize() const noexcept
{
  return MTL::Size(outputTileDimensions[0], outputTileDimensions[1], 1);
}

MTL::Size ANERowwiseTransformKernel::activationScaleGridSize(uint32_t paddedM) const noexcept
{
  return MTL::Size(paddedM, 1, 1);
}

MTL::Size ANERowwiseTransformKernel::activationQuantizeGridSize(uint32_t paddedM, uint32_t K) const noexcept
{
  return MTL::Size(
      (K + quantTileDimension - 1) / quantTileDimension,
      (paddedM + quantTileDimension - 1) / quantTileDimension,
      1);
}

MTL::Size ANERowwiseTransformKernel::outputDequantizeGridSize(uint32_t paddedM, uint32_t N) const noexcept
{
  return MTL::Size(
      (N + outputTileDimensions[0] - 1) / outputTileDimensions[0],
      (paddedM + outputTileDimensions[1] - 1) / outputTileDimensions[1],
      1);
}

std::string ANERowwiseTransformKernel::createSource() const noexcept
{
  CodeWriter source;
  source.SetValue("IO_TYPE", memoryPrecision.name());
  source.SetValue("THREADGROUP_SIZE", std::to_string(activationScaleThreads));
  source.SetValue("QUANT_SIMDGROUPS", std::to_string(activationScaleThreads / 32));
  source.SetValue("QUANT_TILE_DIM", std::to_string(quantTileDimension));
  source.SetValue("QUANT_BLOCK_ROWS", std::to_string(quantBlockRows));
  source.SetValue("QUANT_TILE_PAD", std::to_string(quantTilePad));
  source.SetValue("OUTPUT_TILE_DIM_X", std::to_string(outputTileDimensions[0]));
  source.SetValue("OUTPUT_TILE_DIM_Y", std::to_string(outputTileDimensions[1]));
  source += R"(
#include <metal_stdlib>

using namespace metal;

constant uint SRC_ROWS [[function_constant(0)]];
constant uint PADDED_ROWS [[function_constant(1)]];
constant uint BATCH_DIMENSION [[function_constant(2)]];
constant uint N [[function_constant(3)]];
constant uint K [[function_constant(4)]];
constant uint BATCH_STRIDE_A [[function_constant(5)]];
constant uint BATCH_STRIDE_C [[function_constant(6)]];
constant uint SOURCE_ROW_OFFSET [[function_constant(7)]];
constant uint OUTPUT_ROW_OFFSET [[function_constant(8)]];
constant uint ACTIVATION_SCALE_BATCH_STRIDE [[function_constant(9)]];
constant uint ACTIVATION_SCALE_ROW_OFFSET [[function_constant(10)]];
constant float SCALE_CORRECTION  = (float)K;

constant uint TOTAL_ROWS = SRC_ROWS * BATCH_DIMENSION;
constant uint SOURCE_BATCH_STRIDE = BATCH_STRIDE_A ? BATCH_STRIDE_A : SRC_ROWS * K;
constant uint OUTPUT_BATCH_STRIDE = BATCH_STRIDE_C ? BATCH_STRIDE_C : SRC_ROWS * N;
constant uint SCALE_BATCH_STRIDE = ACTIVATION_SCALE_BATCH_STRIDE ? ACTIVATION_SCALE_BATCH_STRIDE : SRC_ROWS;

inline uint source_offset(const uint flat_row)
{
  const uint batch_index = flat_row / SRC_ROWS;
  const uint row = flat_row - batch_index * SRC_ROWS;
  return batch_index * SOURCE_BATCH_STRIDE + (row + SOURCE_ROW_OFFSET) * K;
}

inline uint output_offset(const uint flat_row, const uint col)
{
  const uint batch_index = flat_row / SRC_ROWS;
  const uint row = flat_row - batch_index * SRC_ROWS;
  return batch_index * OUTPUT_BATCH_STRIDE + (row + OUTPUT_ROW_OFFSET) * N + col;
}

inline uint activation_scale_offset(const uint flat_row)
{
  const uint batch_index = flat_row / SRC_ROWS;
  const uint row = flat_row - batch_index * SRC_ROWS;
  return batch_index * SCALE_BATCH_STRIDE + row + ACTIVATION_SCALE_ROW_OFFSET;
}

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

kernel void compute_activation_scales(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device {{IO_TYPE}}* scales [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[{{QUANT_SIMDGROUPS}}];
  const uint row = tgid.x;
  if (row >= PADDED_ROWS)
    return;
  const uint effective_row = min(row, TOTAL_ROWS > 0 ? TOTAL_ROWS - 1 : 0u);
  float local_max = 0.0f;
  const uint src_base = source_offset(effective_row);
  for (uint i = tid; i < K; i += {{THREADGROUP_SIZE}})
    local_max = max(local_max, fabs((float)src[src_base + i]));
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  if (tid == 0)
    scales[row] = ({{IO_TYPE}})scale;
}

kernel void quantize_transpose_activation(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device const {{IO_TYPE}}* scales [[buffer(1)]],
    device int8_t* dst [[buffer(2)]],
    ushort2 tid2 [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  threadgroup {{IO_TYPE}} tile[{{QUANT_TILE_DIM}}][{{QUANT_TILE_PAD}}];
  const uint src_row_base = tgid.y * {{QUANT_TILE_DIM}};
  const uint src_col = tgid.x * {{QUANT_TILE_DIM}} + tid2.x;
  for (uint j = 0; j < {{QUANT_TILE_DIM}}; j += {{QUANT_BLOCK_ROWS}}) {
    const uint src_row = src_row_base + tid2.y + j;
    if (src_row < PADDED_ROWS && src_col < K) {
      const uint effective_row = min(src_row, TOTAL_ROWS > 0 ? TOTAL_ROWS - 1 : 0u);
      tile[tid2.y + j][tid2.x] = src[source_offset(effective_row) + src_col];
    } else {
      tile[tid2.y + j][tid2.x] = ({{IO_TYPE}})0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint dst_col = src_row_base + tid2.x;
  const float scale = dst_col < PADDED_ROWS ? (float)scales[activation_scale_offset(dst_col)] : 0.0f;
  const float inv_scale = scale > 0.0f ? 1.0f / scale : 127.0f;
  const uint dst_row_base = tgid.x * {{QUANT_TILE_DIM}};
  for (uint j = 0; j < {{QUANT_TILE_DIM}}; j += {{QUANT_BLOCK_ROWS}}) {
    const uint dst_row = dst_row_base + tid2.y + j;
    if (dst_col < PADDED_ROWS && dst_row < K) {
      const int rounded = (int)rint((float)tile[tid2.x][tid2.y + j] * inv_scale);
      dst[dst_row * PADDED_ROWS + dst_col] = (int8_t)clamp(rounded, -127, 127);
    }
  }
}

kernel void transpose_quantized_activation(
    device const int8_t* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    ushort2 tid2 [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  threadgroup int8_t tile[{{QUANT_TILE_DIM}}][{{QUANT_TILE_PAD}}];
  const uint src_row_base = tgid.y * {{QUANT_TILE_DIM}};
  const uint src_col = tgid.x * {{QUANT_TILE_DIM}} + tid2.x;
  for (uint j = 0; j < {{QUANT_TILE_DIM}}; j += {{QUANT_BLOCK_ROWS}}) {
    const uint src_row = src_row_base + tid2.y + j;
    if (src_row < PADDED_ROWS && src_col < K) {
      const uint effective_row = min(src_row, TOTAL_ROWS > 0 ? TOTAL_ROWS - 1 : 0u);
      tile[tid2.y + j][tid2.x] = src[source_offset(effective_row) + src_col];
    } else {
      tile[tid2.y + j][tid2.x] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint dst_col = src_row_base + tid2.x;
  const uint dst_row_base = tgid.x * {{QUANT_TILE_DIM}};
  for (uint j = 0; j < {{QUANT_TILE_DIM}}; j += {{QUANT_BLOCK_ROWS}}) {
    const uint dst_row = dst_row_base + tid2.y + j;
    if (dst_col < PADDED_ROWS && dst_row < K)
      dst[dst_row * PADDED_ROWS + dst_col] = tile[tid2.x][tid2.y + j];
  }
}

kernel void dequantize_output_transposed(
    device const half* src [[buffer(0)]],
    device {{IO_TYPE}}* dst [[buffer(1)]],
    device const {{IO_TYPE}}* activation_scales [[buffer(2)]],
    device const {{IO_TYPE}}* weight_scales [[buffer(3)]],
    ushort2 tid2 [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  const uint dst_row = tgid.y * {{OUTPUT_TILE_DIM_Y}} + tid2.y;
  const uint dst_col = tgid.x * {{OUTPUT_TILE_DIM_X}} + tid2.x;
  if (dst_row < TOTAL_ROWS && dst_col < N) {
    const float value =
        (float)src[dst_col * PADDED_ROWS + dst_row] *
        (float)activation_scales[activation_scale_offset(dst_row)] *
        (float)weight_scales[dst_col] *
        SCALE_CORRECTION;
    dst[output_offset(dst_row, dst_col)] = ({{IO_TYPE}})value;
  }
}

kernel void dequantize_output_transposed_bias(
    device const half* src [[buffer(0)]],
    device {{IO_TYPE}}* dst [[buffer(1)]],
    device const {{IO_TYPE}}* activation_scales [[buffer(2)]],
    device const {{IO_TYPE}}* weight_scales [[buffer(3)]],
    device const {{IO_TYPE}}* bias [[buffer(4)]],
    ushort2 tid2 [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  const uint dst_row = tgid.y * {{OUTPUT_TILE_DIM_Y}} + tid2.y;
  const uint dst_col = tgid.x * {{OUTPUT_TILE_DIM_X}} + tid2.x;
  if (dst_row < TOTAL_ROWS && dst_col < N) {
    const float value =
        (float)src[dst_col * PADDED_ROWS + dst_row] *
        (float)activation_scales[activation_scale_offset(dst_row)] *
        (float)weight_scales[dst_col] *
        SCALE_CORRECTION;
    dst[output_offset(dst_row, dst_col)] = ({{IO_TYPE}})(value + (float)bias[dst_col]);
  }
}
)";
  return source.ToString();
}
