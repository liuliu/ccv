#include "ANERowwiseTransformKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

ANERowwiseTransformKernel::ANERowwiseTransformKernel(
    ANERowwiseTransformKernelDescriptor descriptor,
    MTL::Device* const device)
{
  memoryPrecision = descriptor.memoryPrecision;
  activationHadamard256 = descriptor.activationHadamard256;
  activationPrepareThreads = 1024;
  quantTileDimension = descriptor.supportsApple10 ? 64 : 32;
  quantBlockRows = descriptor.supportsApple10 ? 4 : 8;
  quantTilePad = quantTileDimension + 1;
  outputTileDimensions = simd::ushort2 { 16, 16 };

  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

MTL::Size ANERowwiseTransformKernel::activationPrepareThreadgroupSize(uint32_t K) const noexcept
{
  // Three SIMD groups avoid unused H256 groups on these narrower rows.
  const bool three_simdgroups = activationHadamard256 && K <= 6144 && K % 768 == 0 && K % 1024 != 0;
  return MTL::Size(three_simdgroups ? 768 : activationPrepareThreads, 1, 1);
}

MTL::Size ANERowwiseTransformKernel::activationQuantizeThreadgroupSize() const noexcept
{
  return activationHadamard256 ? MTL::Size(512, 1, 1) : MTL::Size(quantTileDimension, quantBlockRows, 1);
}

MTL::Size ANERowwiseTransformKernel::outputDequantizeThreadgroupSize() const noexcept
{
  return MTL::Size(outputTileDimensions[0], outputTileDimensions[1], 1);
}

MTL::Size ANERowwiseTransformKernel::activationPrepareGridSize(uint32_t paddedM, uint32_t K) const noexcept
{
  return MTL::Size(paddedM / (K > 8192 ? 4 : 8), 1, 1);
}

MTL::Size ANERowwiseTransformKernel::activationQuantizeGridSize(uint32_t paddedM, uint32_t K) const noexcept
{
  if (activationHadamard256)
    return MTL::Size(paddedM / 16, K / 256, 1);
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
  source.SetValue("ACTIVATION_HADAMARD", activationHadamard256 ? "true" : "false");
  source.SetValue("THREADGROUP_SIZE", std::to_string(activationHadamard256 ? activationPrepareThreads : 256));
  source.SetValue("QUANT_SIMDGROUPS", "8");
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

// Wide Hadamard preparation retains float maxima in the existing scale buffer.
// Round the reconstructed scale to IO precision, preserving the fused path's
// output dequantization behavior without an intermediate activation buffer.
inline float read_activation_scale(device const {{IO_TYPE}}* scales, uint row)
{
  if ({{ACTIVATION_HADAMARD}} && K > 8192) {
    const float maximum = reinterpret_cast<device const float*>(scales)[row];
    return float({{IO_TYPE}}(maximum > 0.0f ? maximum / 2032.0f : 1.0f / 127.0f));
  }
  return float(scales[row]);
}

inline float quantize_reduce_max(float value,
                                 threadgroup float* scratch,
                                 ushort sgid,
                                 ushort lane_id,
                                 ushort num_simdgroups = {{QUANT_SIMDGROUPS}})
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
    value = lane_id < num_simdgroups ? scratch[lane_id] : 0.0f;
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

// Complete rows per threadgroup make the row reductions local while
// a small INT8 tile packs adjacent rows into four-byte ANE surface stores.
kernel void quantize_activation(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device {{IO_TYPE}}* scales [[buffer(1)]],
    device char* dst [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
  const uint rows = K <= 8192 ? 8 : 4;
  const uint threads = 1024 / rows;
  const uint simdgroups = threads / 32;
  const uint features = threads * 4;
  threadgroup float partial[32];
  threadgroup char tile[4096];
  const uint row_in_tile = tid / threads;
  const uint row = gid * rows + row_in_tile;
  const uint local = tid % threads;
  const uint sg = local / 32;
  const uint base = source_offset(min(row, TOTAL_ROWS - 1));
  float maximum = 0;
  // Each lane retains at most 64 values. Wider rows get more threads per row;
  // Production uses this fused kernel through K=8192. The wider tile remains
  // available to the benchmark; wider production rows use two dispatches.
  float4 values[16];
  for (uint g = 0; g < (K + features - 1) / features; ++g) {
    const uint col = g * features + local * 4;
    float4 v = 0;
    for (uint c = 0; c < 4; ++c)
      if (col + c < K)
        v[c] = float(src[base + col + c]);
    if (K <= 16384)
      values[g] = v;
    const float4 a = abs(v);
    maximum = max(maximum, max(max(a.x, a.y), max(a.z, a.w)));
  }
  maximum = simd_max(maximum);
  if (lane == 0)
    partial[row_in_tile * simdgroups + sg] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum = 0;
  for (uint g = 0; g < simdgroups; ++g)
    maximum = max(maximum, partial[row_in_tile * simdgroups + g]);
  // Match the two-dispatch path: quantize with the scale rounded to IO precision.
  const {{IO_TYPE}} stored_scale = {{IO_TYPE}}(maximum > 0 ? maximum / 127.0f : 1.0f / 127.0f);
  if (local == 0)
    scales[row] = stored_scale;
  const float inv = float(stored_scale) > 0 ? 1.0f / float(stored_scale) : 127.0f;
  for (uint g = 0; g < (K + features - 1) / features; ++g) {
    const uint col = g * features + local * 4;
    float4 v = 0;
    if (K <= 16384)
      v = values[g];
    else
      for (uint c = 0; c < 4; ++c)
        if (col + c < K)
          v[c] = float(src[base + col + c]);
    const char4 q = char4(clamp(int4(rint(v * inv)), int4(-127), int4(127)));
    *reinterpret_cast<threadgroup char4*>(tile + row_in_tile * features + local * 4) = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint feature = tid / (rows / 4);
    const uint r = (tid % (rows / 4)) * 4;
    const uint output_k = g * features + feature;
    if (output_k < K) {
      const char4 packed = char4(tile[r * features + feature], tile[(r + 1) * features + feature], tile[(r + 2) * features + feature], tile[(r + 3) * features + feature]);
      reinterpret_cast<device char4*>(dst)[output_k * (PADDED_ROWS / 4) + gid * (rows / 4) + r / 4] = packed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
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
        read_activation_scale(activation_scales, activation_scale_offset(dst_row)) *
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
        read_activation_scale(activation_scales, activation_scale_offset(dst_row)) *
        (float)weight_scales[dst_col] *
        SCALE_CORRECTION;
    dst[output_offset(dst_row, dst_col)] = ({{IO_TYPE}})(value + (float)bias[dst_col]);
  }
}
)";
  if (activationHadamard256) {
    source += R"(
// Rotate one 256-feature group in SIMD registers. H4 has +1 everywhere
// except its anti-diagonal (-1). Its fourth Kronecker power divided by 16
// is the stored ConvRot transform. Fold that normalization into the scale.
inline void hadamard256_group(device const {{IO_TYPE}}* src,
                             uint row_offset, uint group, ushort lane_id,
                             thread float4& y0, thread float4& y1)
{
  float4 x[2];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < 2; ++j) {
    const uint k = group * 256 + j * 128 + lane_id * 4;
    float4 v = float4(0);
    if (group < K / 256) {
      const uint offset = row_offset + k;
      v = float4(src[offset], src[offset + 1], src[offset + 2], src[offset + 3]);
    }
    v = (v.x + v.y + v.z + v.w) - 2.0f * v.wzyx;
    #pragma clang loop unroll(full)
    for (ushort stride = 1; stride <= 4; stride *= 4) {
      const float4 peer = simd_shuffle_xor(v, stride);
      v = (v + peer) + simd_shuffle_xor(v - peer, ushort(stride * 2));
    }
    x[j] = v;
  }
  const float4 sum = x[0] + x[1];
  const float4 diff = simd_shuffle_xor(x[0] - x[1], 16);
  y0 = (sum + diff);
  y1 = (sum - diff);
}

kernel void quantize_hadamard_activation(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device {{IO_TYPE}}* scales [[buffer(1)]],
    device int8_t* dst [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint rows = 8;
  const uint threads = K <= 6144 && K % 768 == 0 && K % 1024 != 0 ? 96 : 128;
  const uint simdgroups = threads / 32;
  const uint features = simdgroups * 256;
  threadgroup float scratch[8][4];
  // Alternate two 8 KiB tiles so the next write does not need to wait for
  // this tile's reads. No intermediate device activation buffer is needed.
  threadgroup char storage[16384];
  const uint row_in_tile = tid / threads;
  sgid %= simdgroups;
  const uint row = tgid.x * rows + row_in_tile;
  if (row >= PADDED_ROWS)
    return;
  const uint effective_row = min(row, TOTAL_ROWS - 1);
  float local_max = 0.0f;
  // One SIMD group rotates 256 consecutive features, with eight floats per
  // lane. Keep the rotated row in registers through the row-scale reduction.
  const uint groups_per_simd = (K / 256 + simdgroups - 1) / simdgroups;
  // Metal cannot size a thread array with a function constant. This entry
  // point is selected only for K <= 8192, at 96 or 128 threads per row.
  float4 rotated[16];
  #pragma clang loop unroll(full)
  for (uint g = 0; g < groups_per_simd; ++g) {
    const uint group = g * simdgroups + sgid;
    hadamard256_group(src, source_offset(effective_row), group, lane_id,
                      rotated[g * 2], rotated[g * 2 + 1]);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 2; ++j) {
      const float4 v = abs(rotated[g * 2 + j]);
      local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
    }
  }
  local_max = simd_max(local_max);
  if (lane_id == 0)
    scratch[row_in_tile][sgid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float max_abs = 0;
  for (uint g = 0; g < simdgroups; ++g)
    max_abs = max(max_abs, scratch[row_in_tile][g]);
  // 2032 = 16 * 127: store the normalized transform's quantization scale.
  const float scale = max_abs > 0.0f ? max_abs / 2032.0f : (1.0f / 127.0f);
  // The unrounded maximum bounds every quantized value by 127, so unlike
  // ordinary quantization using a rounded IO scale, no clamp is needed.
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid % threads == 0)
    scales[row] = ({{IO_TYPE}})scale;
  #pragma clang loop unroll(full)
  for (uint g = 0; g < groups_per_simd; ++g) {
    threadgroup char* tile = storage + (g % 2) * 8192;
    const uint group = g * simdgroups + sgid;
    if (group < K / 256) {
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 2; ++j) {
        const int4 rounded = int4(rint(rotated[g * 2 + j] * inv_scale));
        const char4 quantized = char4(rounded);
        *reinterpret_cast<threadgroup char4*>(tile + row_in_tile * features + sgid * 256 + j * 128 + lane_id * 4) = quantized;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = tid;
    if (g * features + c < K) {
      const uint2 packed = uint2(
          as_type<uint>(char4(tile[c], tile[features + c], tile[2 * features + c], tile[3 * features + c])),
          as_type<uint>(char4(tile[4 * features + c], tile[5 * features + c], tile[6 * features + c], tile[7 * features + c])));
      reinterpret_cast<device uint2*>(dst)[(g * features + c) * (PADDED_ROWS / 8) + tgid.x] = packed;
    }
  }
}

// Wider rows avoid retaining a complete transformed row. Only maxima cross
// dispatches, using the float capacity already reserved in the scale buffer.
kernel void compute_hadamard_activation_maxima(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device float* maxima [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[4][8];
  const uint row_in_tile = tid / 256;
  const uint row = gid * 4 + row_in_tile;
  const uint row_offset = source_offset(min(row, TOTAL_ROWS - 1));
  sgid %= 8;
  float maximum = 0.0f;
  #pragma clang loop unroll(full)
  for (uint g = 0; g < (K / 256 + 7) / 8; ++g) {
    float4 a, b;
    hadamard256_group(src, row_offset, g * 8 + sgid, lane_id, a, b);
    const float4 v = max(abs(a), abs(b));
    maximum = max(maximum, max(max(v.x, v.y), max(v.z, v.w)));
  }
  maximum = quantize_reduce_max(maximum, scratch[row_in_tile], sgid, lane_id);
  if (tid % 256 == 0)
    maxima[row] = maximum;
}

// Recompute one group per row, quantize it, and transpose a 16 x 256 tile.
// The additional arithmetic avoids full-row register retention and permits
// contiguous writes across more rows of the ANE activation surface.
kernel void quantize_transpose_hadamard_activation(
    device const {{IO_TYPE}}* src [[buffer(0)]],
    device const float* maxima [[buffer(1)]],
    device char* dst [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
  threadgroup char tile[16][256];
  const uint row = gid.x * 16 + sgid;
  const float maximum = maxima[row];
  const float inv_scale = maximum > 0.0f ? 127.0f / maximum : 127.0f;
  float4 a, b;
  hadamard256_group(src, source_offset(min(row, TOTAL_ROWS - 1)), gid.y, lane_id, a, b);
  const char4 qa = char4(rint(a * inv_scale));
  const char4 qb = char4(rint(b * inv_scale));
  *reinterpret_cast<threadgroup char4*>(tile[sgid] + lane_id * 4) = qa;
  *reinterpret_cast<threadgroup char4*>(tile[sgid] + 128 + lane_id * 4) = qb;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = tid; index < 1024; index += 512) {
    const uint c = index / 4;
    const uint r = (index % 4) * 4;
    const char4 packed = char4(tile[r][c], tile[r + 1][c], tile[r + 2][c], tile[r + 3][c]);
    reinterpret_cast<device char4*>(dst)[(gid.y * 256 + c) * (PADDED_ROWS / 4) + gid.x * 4 + r / 4] = packed;
  }
}

)";
  }
  return source.ToString();
}
