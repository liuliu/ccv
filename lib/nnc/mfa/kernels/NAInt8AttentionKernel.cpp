#include "NAInt8AttentionKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

static std::string high_precision_to_string(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

static std::string dot_product_scale(float rsqrt_d) {
  constexpr float log_base_2_e = 1.442695041f;
  return high_precision_to_string(log_base_2_e * rsqrt_d);
}

static std::string dot_product_scale(float rsqrt_d, bool derivative) {
  if (!derivative) {
    return dot_product_scale(rsqrt_d);
  }
  return high_precision_to_string(rsqrt_d);
}

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

NAInt8AttentionKernel::NAInt8AttentionKernel(
    NAInt8AttentionKernelDescriptor descriptor,
    MTL::Device *const device)
{
  blockDimensions = descriptor.blockDimensions;
  type = descriptor.type;
  headDimension = descriptor.headDimension;
  Hq = descriptor.Hq;
  Hk = descriptor.Hk;
  qScaleTileSize = descriptor.qScaleTileSize;
  kvScaleTileSize = descriptor.kvScaleTileSize;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  vMeanThreads = descriptor.vMeanThreads;
  hasCRemainder = descriptor.hasCRemainder;
  threadBarrierEveryC = descriptor.threadBarrierEveryC;
  ioPrecision = descriptor.ioPrecision;
  lowPrecisionIntermediates = descriptor.lowPrecisionIntermediates;
  scale = descriptor.scale;

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint32_t NAInt8AttentionKernel::threadgroupMemoryAllocation() const noexcept {
  if (type == AttentionKernelType::forward) {
    if (!hasCRemainder) {
      return 0;
    }
    return blockDimensions[0] * blockDimensions[1] * executionSIMDGroups * sizeof(int8_t);
  }
  if (type == AttentionKernelType::backwardKeyValue) {
    return headDimension * blockDimensions[0] * executionSIMDGroups * sizeof(int8_t) * 2;
  }
  return 0;
}

uint16_t NAInt8AttentionKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept {
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAInt8AttentionKernel::threadgroupsPerGrid(uint32_t batchDimension, uint32_t rowDimension) const noexcept {
  auto ceilDivide =
    [=](int64_t target, uint16_t granularity) -> int64_t {
      return (target + int64_t(granularity) - 1) / int64_t(granularity);
    };
  const int64_t row_groups = ceilDivide(rowDimension, blockDimensions[0] * executionSIMDGroups);
  const uint32_t row_bits = ceilLog2(row_groups);
  const uint32_t heads =
      type == AttentionKernelType::backwardKeyValue ? Hk : Hq;
  const uint32_t head_bits = ceilLog2(heads);
  return MTL::Size(int64_t(1) << (row_bits + head_bits), 1, batchDimension);
}

std::string NAInt8AttentionKernel::createSource() const noexcept {
  CodeWriter source;
  const bool vectorizeQuantize = (headDimension % 4) == 0;
  const GEMMOperandPrecision lPrecision =
      lowPrecisionIntermediates ?
      (ioPrecision == GEMMOperandPrecision::BF16 ? GEMMOperandPrecision::BF16 :
          (ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16)) :
      GEMMOperandPrecision::FP32;
  const GEMMOperandPrecision dPrecision =
      lowPrecisionIntermediates ?
      (ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::BF16) :
      GEMMOperandPrecision::FP32;
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

  )";
  source.SetValue("IO_MEMORY_NAME", ioPrecision.name());
  source.SetValue("L_MEMORY_NAME", lPrecision.name());
  source.SetValue("D_MEMORY_NAME", dPrecision.name());
  source.SetValue("V_MEAN_MEMORY_NAME", "float");
  source.SetValue("ACCUM_MEMORY_NAME", ioPrecision == GEMMOperandPrecision::FP32 ? "float" : "half");
  source += R"(
constant uint QUANTIZE_Q_THREADS = )" + std::to_string(qQuantizeThreads) + R"(;
constant uint QUANTIZE_KV_THREADS = )" + std::to_string(kvQuantizeThreads) + R"(;
constant uint QUANTIZE_V_MEAN_THREADS = )" + std::to_string(vMeanThreads) + R"(;
constant uint QUANTIZE_SIMD_LANES = 32;
constant uint QUANTIZE_Q_SIMDGROUPS = QUANTIZE_Q_THREADS / QUANTIZE_SIMD_LANES;
constant uint QUANTIZE_KV_SIMDGROUPS = QUANTIZE_KV_THREADS / QUANTIZE_SIMD_LANES;
constant uint QUANTIZE_V_MEAN_SIMDGROUPS = QUANTIZE_V_MEAN_THREADS / QUANTIZE_SIMD_LANES;
constant uint QUANTIZE_Q_SEQUENCE [[function_constant(900)]];
constant uint QUANTIZE_KV_SEQUENCE [[function_constant(901)]];
constant uint QUANTIZE_Q_HEADS [[function_constant(902)]];
constant uint QUANTIZE_KV_HEADS [[function_constant(903)]];
constant uint QUANTIZE_Q_TILE_SIZE [[function_constant(904)]];
constant uint QUANTIZE_KV_TILE_SIZE [[function_constant(905)]];
constant uint QUANTIZE_Q_SCALE_TILES [[function_constant(906)]];
constant uint QUANTIZE_KV_SCALE_TILES [[function_constant(907)]];
constant uint QUANTIZE_Q_BATCH_STRIDE [[function_constant(908)]];
constant uint QUANTIZE_K_BATCH_STRIDE [[function_constant(909)]];
constant uint QUANTIZE_V_BATCH_STRIDE [[function_constant(910)]];
constant uint QUANTIZE_Q_SCALE_BATCH_STRIDE [[function_constant(911)]];
constant uint QUANTIZE_KV_SCALE_BATCH_STRIDE [[function_constant(912)]];

inline float quantize_reduce_max(float value,
                                 threadgroup float *scratch,
                                 ushort sgid,
                                 ushort lane_id,
                                 uint simdgroup_count) {
  value = max(value, simd_shuffle_xor(value, 16));
  value = max(value, simd_shuffle_xor(value, 8));
  value = max(value, simd_shuffle_xor(value, 4));
  value = max(value, simd_shuffle_xor(value, 2));
  value = max(value, simd_shuffle_xor(value, 1));
  if (lane_id == 0)
    scratch[sgid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    value = lane_id < simdgroup_count ? scratch[lane_id] : 0.0f;
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
  if (vectorizeQuantize) {
    source += R"(
using io_vec4 = vec<{{IO_MEMORY_NAME}}, 4>;
using v_mean_vec4 = vec<{{V_MEAN_MEMORY_NAME}}, 4>;

inline void quantize_tile(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    threadgroup float *scratch,
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sequence,
    uint heads,
    uint tile_size,
    uint scale_tiles,
    uint batch_stride,
    uint scale_batch_stride,
    uint thread_count,
    uint simdgroup_count) {
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint start = tile * tile_size;
  const uint extent = min(tile_size, sequence - start);
  const uint vectors_per_row = {{HEAD_DIMENSION}} / 4;
  const uint total_vectors = extent * vectors_per_row;
  device const io_vec4 *src4 = reinterpret_cast<device const io_vec4 *>(src);
  device char4 *dst4 = reinterpret_cast<device char4 *>(dst);
  float local_max = 0;
  for (uint i = tid; i < total_vectors; i += thread_count) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * batch_stride +
        ((row * heads + head) * {{HEAD_DIMENSION}} + vec_dim * 4);
    const float4 value = float4(src4[index / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id, simdgroup_count);
  const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[batch * scale_batch_stride + head * scale_tiles + tile] = scale;
  for (uint i = tid; i < total_vectors; i += thread_count) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * batch_stride +
        ((row * heads + head) * {{HEAD_DIMENSION}} + vec_dim * 4);
    const float4 value = float4(src4[index / 4]) * inv_scale;
    const int4 rounded = int4(rint(value));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}

)";
  } else {
    source += R"(
inline void quantize_tile(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    threadgroup float *scratch,
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sequence,
    uint heads,
    uint tile_size,
    uint scale_tiles,
    uint batch_stride,
    uint scale_batch_stride,
    uint thread_count,
    uint simdgroup_count) {
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint start = tile * tile_size;
  const uint extent = min(tile_size, sequence - start);
  const uint total = extent * {{HEAD_DIMENSION}};
  float local_max = 0;
  for (uint i = tid; i < total; i += thread_count) {
    const uint row = start + i / {{HEAD_DIMENSION}};
    const uint dim = i % {{HEAD_DIMENSION}};
    const uint index = batch * batch_stride + ((row * heads + head) * {{HEAD_DIMENSION}} + dim);
    local_max = max(local_max, fabs((float)src[index]));
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id, simdgroup_count);
  const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[batch * scale_batch_stride + head * scale_tiles + tile] = scale;
  for (uint i = tid; i < total; i += thread_count) {
    const uint row = start + i / {{HEAD_DIMENSION}};
    const uint dim = i % {{HEAD_DIMENSION}};
    const uint index = batch * batch_stride + ((row * heads + head) * {{HEAD_DIMENSION}} + dim);
    const int rounded = (int)rint((float)src[index] * inv_scale);
    dst[index] = (int8_t)clamp(rounded, -127, 127);
  }
}

)";
  }
  source += R"(
kernel void quantize_q(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float scratch[QUANTIZE_Q_SIMDGROUPS];
  quantize_tile(src, dst, scales, scratch, tid, sgid, lane_id, tgid,
      QUANTIZE_Q_SEQUENCE,
      QUANTIZE_Q_HEADS,
      QUANTIZE_Q_TILE_SIZE,
      QUANTIZE_Q_SCALE_TILES,
      QUANTIZE_Q_BATCH_STRIDE,
      QUANTIZE_Q_SCALE_BATCH_STRIDE,
      QUANTIZE_Q_THREADS,
      QUANTIZE_Q_SIMDGROUPS);
}

kernel void quantize_k(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float scratch[QUANTIZE_KV_SIMDGROUPS];
  quantize_tile(src, dst, scales, scratch, tid, sgid, lane_id, tgid,
      QUANTIZE_KV_SEQUENCE,
      QUANTIZE_KV_HEADS,
      QUANTIZE_KV_TILE_SIZE,
      QUANTIZE_KV_SCALE_TILES,
      QUANTIZE_K_BATCH_STRIDE,
      QUANTIZE_KV_SCALE_BATCH_STRIDE,
      QUANTIZE_KV_THREADS,
      QUANTIZE_KV_SIMDGROUPS);
}

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

inline uint ceil_log2_u32(uint x) {
  if (x <= 1)
    return 0;
  x -= 1;
  uint bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

)";
  if (vectorizeQuantize) {
      source += R"(
kernel void compute_v_mean(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device {{V_MEAN_MEMORY_NAME}} *mean [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float4 scratch[QUANTIZE_V_MEAN_SIMDGROUPS];
  device const io_vec4 *src4 = reinterpret_cast<device const io_vec4 *>(src);
  device v_mean_vec4 *mean4 = reinterpret_cast<device v_mean_vec4 *>(mean);
  const uint mean_tiles = {{HEAD_DIMENSION}} / 4;
  const uint vec_bits = ceil_log2_u32(mean_tiles);
  const uint head_bits = ceil_log2_u32(QUANTIZE_KV_HEADS);
  const uint2 morton = morton_decode_rectangular_2d(tgid.x, vec_bits, head_bits);
  const uint vec_dim = morton.x;
  const uint head = morton.y;
  const uint batch = tgid.z;
  if (vec_dim >= mean_tiles || head >= QUANTIZE_KV_HEADS)
    return;
  float4 local_sum = float4(0.0f);
  for (uint column = tid; column < QUANTIZE_KV_SEQUENCE; column += QUANTIZE_V_MEAN_THREADS) {
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE +
        ((column * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + vec_dim * 4);
    local_sum += float4(src4[index / 4]);
  }
  local_sum[0] += simd_shuffle_xor(local_sum[0], 16);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 16);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 16);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 16);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 8);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 8);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 8);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 8);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 4);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 4);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 4);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 4);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 2);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 2);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 2);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 2);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 1);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 1);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 1);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 1);
  if (lane_id == 0)
    scratch[sgid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    float4 reduced = lane_id < QUANTIZE_V_MEAN_SIMDGROUPS ? scratch[lane_id] : float4(0.0f);
    reduced[0] += simd_shuffle_xor(reduced[0], 16);
    reduced[1] += simd_shuffle_xor(reduced[1], 16);
    reduced[2] += simd_shuffle_xor(reduced[2], 16);
    reduced[3] += simd_shuffle_xor(reduced[3], 16);
    reduced[0] += simd_shuffle_xor(reduced[0], 8);
    reduced[1] += simd_shuffle_xor(reduced[1], 8);
    reduced[2] += simd_shuffle_xor(reduced[2], 8);
    reduced[3] += simd_shuffle_xor(reduced[3], 8);
    reduced[0] += simd_shuffle_xor(reduced[0], 4);
    reduced[1] += simd_shuffle_xor(reduced[1], 4);
    reduced[2] += simd_shuffle_xor(reduced[2], 4);
    reduced[3] += simd_shuffle_xor(reduced[3], 4);
    reduced[0] += simd_shuffle_xor(reduced[0], 2);
    reduced[1] += simd_shuffle_xor(reduced[1], 2);
    reduced[2] += simd_shuffle_xor(reduced[2], 2);
    reduced[3] += simd_shuffle_xor(reduced[3], 2);
    reduced[0] += simd_shuffle_xor(reduced[0], 1);
    reduced[1] += simd_shuffle_xor(reduced[1], 1);
    reduced[2] += simd_shuffle_xor(reduced[2], 1);
    reduced[3] += simd_shuffle_xor(reduced[3], 1);
    if (lane_id == 0) {
      mean4[((batch * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + vec_dim * 4) / 4] =
          reduced * (1.0f / float(QUANTIZE_KV_SEQUENCE));
    }
  }
}

kernel void quantize_v(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    device const {{V_MEAN_MEMORY_NAME}} *mean [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float scratch[QUANTIZE_KV_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint start = tile * QUANTIZE_KV_TILE_SIZE;
  const uint extent = min(QUANTIZE_KV_TILE_SIZE, QUANTIZE_KV_SEQUENCE - start);
  const uint vectors_per_row = {{HEAD_DIMENSION}} / 4;
  const uint total_vectors = extent * vectors_per_row;
  device const io_vec4 *src4 = reinterpret_cast<device const io_vec4 *>(src);
  device const v_mean_vec4 *mean4 = reinterpret_cast<device const v_mean_vec4 *>(mean);
  device char4 *dst4 = reinterpret_cast<device char4 *>(dst);
  float local_max = 0;
  for (uint i = tid; i < total_vectors; i += QUANTIZE_KV_THREADS) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE +
        ((row * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + vec_dim * 4);
    const float4 value = float4(src4[index / 4]) -
        float4(mean4[((batch * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + vec_dim * 4) / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id, QUANTIZE_KV_SIMDGROUPS);
  const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[batch * QUANTIZE_KV_SCALE_BATCH_STRIDE + head * QUANTIZE_KV_SCALE_TILES + tile] = scale;
  for (uint i = tid; i < total_vectors; i += QUANTIZE_KV_THREADS) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE +
        ((row * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + vec_dim * 4);
    const float4 value = float4(src4[index / 4]) -
        float4(mean4[((batch * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + vec_dim * 4) / 4]);
    const int4 rounded = int4(rint(value * inv_scale));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}

)";
  } else {
      source += R"(
kernel void compute_v_mean(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device {{V_MEAN_MEMORY_NAME}} *mean [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float scratch[QUANTIZE_V_MEAN_SIMDGROUPS];
  const uint dim_bits = ceil_log2_u32({{HEAD_DIMENSION}});
  const uint head_bits = ceil_log2_u32(QUANTIZE_KV_HEADS);
  const uint2 morton = morton_decode_rectangular_2d(tgid.x, dim_bits, head_bits);
  const uint dim = morton.x;
  const uint head = morton.y;
  const uint batch = tgid.z;
  if (dim >= {{HEAD_DIMENSION}} || head >= QUANTIZE_KV_HEADS)
    return;
  float local_sum = 0.0f;
  for (uint column = tid; column < QUANTIZE_KV_SEQUENCE; column += QUANTIZE_V_MEAN_THREADS) {
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE +
        ((column * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + dim);
    local_sum += (float)src[index];
  }
  local_sum += simd_shuffle_xor(local_sum, 16);
  local_sum += simd_shuffle_xor(local_sum, 8);
  local_sum += simd_shuffle_xor(local_sum, 4);
  local_sum += simd_shuffle_xor(local_sum, 2);
  local_sum += simd_shuffle_xor(local_sum, 1);
  if (lane_id == 0)
    scratch[sgid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    float reduced = lane_id < QUANTIZE_V_MEAN_SIMDGROUPS ? scratch[lane_id] : 0.0f;
    reduced += simd_shuffle_xor(reduced, 16);
    reduced += simd_shuffle_xor(reduced, 8);
    reduced += simd_shuffle_xor(reduced, 4);
    reduced += simd_shuffle_xor(reduced, 2);
    reduced += simd_shuffle_xor(reduced, 1);
    if (lane_id == 0)
      mean[(batch * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + dim] =
          reduced * (1.0f / float(QUANTIZE_KV_SEQUENCE));
  }
}

kernel void quantize_v(
    device const {{IO_MEMORY_NAME}} *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    device const {{V_MEAN_MEMORY_NAME}} *mean [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float scratch[QUANTIZE_KV_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint start = tile * QUANTIZE_KV_TILE_SIZE;
  const uint extent = min(QUANTIZE_KV_TILE_SIZE, QUANTIZE_KV_SEQUENCE - start);
  const uint total = extent * {{HEAD_DIMENSION}};
  float local_max = 0;
  for (uint i = tid; i < total; i += QUANTIZE_KV_THREADS) {
    const uint row = start + i / {{HEAD_DIMENSION}};
    const uint dim = i % {{HEAD_DIMENSION}};
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE + ((row * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + dim);
    const float value = (float)src[index] - (float)mean[(batch * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + dim];
    local_max = max(local_max, fabs(value));
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id, QUANTIZE_KV_SIMDGROUPS);
  const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[batch * QUANTIZE_KV_SCALE_BATCH_STRIDE + head * QUANTIZE_KV_SCALE_TILES + tile] = scale;
  for (uint i = tid; i < total; i += QUANTIZE_KV_THREADS) {
    const uint row = start + i / {{HEAD_DIMENSION}};
    const uint dim = i % {{HEAD_DIMENSION}};
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE + ((row * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + dim);
    const float value = (float)src[index] - (float)mean[(batch * QUANTIZE_KV_HEADS + head) * {{HEAD_DIMENSION}} + dim];
    const int rounded = (int)rint(value * inv_scale);
    dst[index] = (int8_t)clamp(rounded, -127, 127);
  }
}

)";
  }
  createConstants(source);
  if (type == AttentionKernelType::backwardQuery) {
    source += createComputeD();
  }
  switch (type.value) {
  case AttentionKernelType::forward:
    source.SetValue("MAIN_KERNEL_NAME", "int8_attention");
    source.SetValue("ROW_DIMENSION_SYMBOL", "R");
    source.SetValue("GRID_HEADS", std::to_string(Hq));
    source += R"(
kernel void {{MAIN_KERNEL_NAME}}(
)";
    source += createBufferBindings();
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
    source.SetValue("MATMUL_SIMDGROUPS", "1");
    source.SetValue("THREAD_BARRIER_EVERY_C", std::to_string(threadBarrierEveryC));
	    source += R"(
	    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
	    ushort tid [[thread_index_in_threadgroup]],
	    ushort sgid [[simdgroup_index_in_threadgroup]],
	    uint3 tgid [[threadgroup_position_in_grid]]
	  ) {
  const uint row_group_count = ({{ROW_DIMENSION_SYMBOL}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group_bits = ceil_log2_u32(row_group_count);
  const uint head_bits = ceil_log2_u32({{GRID_HEADS}});
  const uint tile_code = tgid.x;
  const uint2 morton_tile = morton_decode_rectangular_2d(tile_code, row_group_bits, head_bits);
  tgid = uint3(morton_tile.x, morton_tile.y, tgid.z);
  if (tgid.y >= {{GRID_HEADS}} || tgid.x >= row_group_count) {
    return;
  }
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{ROW_DIMENSION_SYMBOL}}) {
    return;
  }
)";
    source += createAdjustOffsets();
    loopForward(source);
    source += "}\n";
    break;
  case AttentionKernelType::backwardQuery:
    source.SetValue("MAIN_KERNEL_NAME", "int8_backward_query");
    source.SetValue("ROW_DIMENSION_SYMBOL", "R");
    source.SetValue("GRID_HEADS", std::to_string(Hq));
    source += R"(
kernel void {{MAIN_KERNEL_NAME}}(
)";
    source += createBufferBindings();
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
	    source += R"(
	    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
	    ushort tid [[thread_index_in_threadgroup]],
	    ushort sgid [[simdgroup_index_in_threadgroup]],
	    uint3 tgid [[threadgroup_position_in_grid]]
	  ) {
  const uint row_group_count = ({{ROW_DIMENSION_SYMBOL}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group_bits = ceil_log2_u32(row_group_count);
  const uint head_bits = ceil_log2_u32({{GRID_HEADS}});
  const uint tile_code = tgid.x;
  const uint2 morton_tile = morton_decode_rectangular_2d(tile_code, row_group_bits, head_bits);
  tgid = uint3(morton_tile.x, morton_tile.y, tgid.z);
  if (tgid.y >= {{GRID_HEADS}} || tgid.x >= row_group_count) {
    return;
  }
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{ROW_DIMENSION_SYMBOL}}) {
    return;
  }
)";
    source += createAdjustOffsets();
    loopBackwardQuery(source);
    source += "}\n";
    break;
  case AttentionKernelType::backwardKeyValue:
    source.SetValue("MAIN_KERNEL_NAME", "int8_backward_keyvalue");
    source.SetValue("ROW_DIMENSION_SYMBOL", "C");
    source.SetValue("GRID_HEADS", std::to_string(Hk));
    source += R"(
kernel void {{MAIN_KERNEL_NAME}}(
)";
    source += createBufferBindings();
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
	    source += R"(
	    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
	    ushort tid [[thread_index_in_threadgroup]],
	    ushort sgid [[simdgroup_index_in_threadgroup]],
	    ushort lane_id [[thread_index_in_simdgroup]],
	    uint3 tgid [[threadgroup_position_in_grid]]
	  ) {
  const uint row_group_count = ({{ROW_DIMENSION_SYMBOL}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group_bits = ceil_log2_u32(row_group_count);
  const uint head_bits = ceil_log2_u32({{GRID_HEADS}});
  const uint tile_code = tgid.x;
  const uint2 morton_tile = morton_decode_rectangular_2d(tile_code, row_group_bits, head_bits);
  tgid = uint3(morton_tile.x, morton_tile.y, tgid.z);
  if (tgid.y >= {{GRID_HEADS}} || tgid.x >= row_group_count) {
    return;
  }
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{ROW_DIMENSION_SYMBOL}}) {
    return;
  }
)";
    source += createAdjustOffsets();
    loopBackwardKeyValue(source);
    source += "}\n";
    break;
  }
  return source.ToString();
}

void NAInt8AttentionKernel::createConstants(CodeWriter& source) const noexcept {
  source.SetValue("HQ", std::to_string(Hq));
  source.SetValue("HK", std::to_string(Hk));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("Q_SCALE_TILE_SIZE", std::to_string(qScaleTileSize));
  source.SetValue("KV_SCALE_TILE_SIZE", std::to_string(kvScaleTileSize));
  source.SetValue("Q_SCALE_TILES", "(R + " + std::to_string(qScaleTileSize) + " - 1) / " + std::to_string(qScaleTileSize));
  source.SetValue("K_SCALE_TILES", "(C + " + std::to_string(kvScaleTileSize) + " - 1) / " + std::to_string(kvScaleTileSize));
  source += R"(
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint Q_batch_stride [[function_constant(2)]];
constant uint K_batch_stride [[function_constant(3)]];
constant uint V_batch_stride [[function_constant(4)]];
constant uint O_batch_stride [[function_constant(5)]];
constant uint dO_batch_stride [[function_constant(6)]];
constant uint dV_batch_stride [[function_constant(7)]];
constant uint dK_batch_stride [[function_constant(8)]];
constant uint dQ_batch_stride [[function_constant(9)]];
constant uint Q_scale_batch_stride [[function_constant(10)]];
constant uint K_scale_batch_stride [[function_constant(11)]];
constant uint V_scale_batch_stride [[function_constant(12)]];
constant uint dO_scale_batch_stride [[function_constant(13)]];
constant uint V_mean_batch_stride [[function_constant(14)]];
)";
  source += R"(

constant uint Hq = {{HQ}};
constant uint Hk = {{HK}};
constant uint K_Hq = {{HEAD_DIMENSION}} * Hq;
constant uint K_Hk = {{HEAD_DIMENSION}} * Hk;
constant uint Q_scale_tile_size = {{Q_SCALE_TILE_SIZE}};
constant uint KV_scale_tile_size = {{KV_SCALE_TILE_SIZE}};
constant uint Q_scale_tiles = {{Q_SCALE_TILES}};
constant uint K_scale_tiles = {{K_SCALE_TILES}};
constant uint C_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
)";
  source += R"(
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
constant uint KV_R_edge = R >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? R + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint KV_R_remainder = R % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint KV_C_edge = C >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? C + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint KV_C_remainder = C % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
  source.SetValue("QK_SCALE_FACTOR_0", "Q_scale_buf[0] * K_scale_buf[c / KV_scale_tile_size] * ");
  source.SetValue("QK_SCALE_FACTOR_REM", "Q_scale_buf[0] * K_scale_buf[(C - C_remainder) / KV_scale_tile_size] * ");
  source.SetValue("V_SCALE_FACTOR_0", "V_scale_buf[c / KV_scale_tile_size]");
  source.SetValue("V_SCALE_FACTOR_REM", "V_scale_buf[(C - C_remainder) / KV_scale_tile_size]");
}

std::string NAInt8AttentionKernel::createBufferBindings() const noexcept {
  CodeWriter source;
  const GEMMOperandPrecision lPrecision =
      lowPrecisionIntermediates ?
      (ioPrecision == GEMMOperandPrecision::BF16 ? GEMMOperandPrecision::BF16 :
          (ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16)) :
      GEMMOperandPrecision::FP32;
  const GEMMOperandPrecision dPrecision =
      lowPrecisionIntermediates ?
      (ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::BF16) :
      GEMMOperandPrecision::FP32;
  source.SetValue("IO_MEMORY_NAME", ioPrecision.name());
  source.SetValue("L_MEMORY_NAME", lPrecision.name());
  source.SetValue("D_MEMORY_NAME", dPrecision.name());
  source.SetValue("V_MEAN_MEMORY_NAME", "float");
  source.SetValue("QK_MEMORY_NAME", "int8_t");
  const char* v_memory_name = "int8_t";
  source.SetValue("QK_MEMORY_NAME", "int8_t");
  source.SetValue("V_MEMORY_NAME", v_memory_name);
  switch (type.value) {
  case AttentionKernelType::forward:
    source += R"(
    device {{QK_MEMORY_NAME}} *Q_buf [[buffer(0)]],
    device {{QK_MEMORY_NAME}} *K_buf [[buffer(1)]],
    device {{V_MEMORY_NAME}} *V_buf [[buffer(2)]],
    device {{IO_MEMORY_NAME}} *O_buf [[buffer(3)]],
    device {{L_MEMORY_NAME}} *L_buf [[buffer(4)]],
    device float *Q_scale_buf [[buffer(10)]],
    device float *K_scale_buf [[buffer(11)]],
    device float *V_scale_buf [[buffer(12)]],
    device const {{V_MEAN_MEMORY_NAME}} *V_mean_buf [[buffer(14)]],
)";
    break;
  case AttentionKernelType::backwardQuery:
    source += R"(
    device int8_t *Q_buf [[buffer(0)]],
    device int8_t *K_buf [[buffer(1)]],
    device int8_t *V_buf [[buffer(2)]],
    device const {{L_MEMORY_NAME}} *L_buf [[buffer(4)]],
    device const {{D_MEMORY_NAME}} *D_buf [[buffer(5)]],
    device int8_t *dO_buf [[buffer(6)]],
    device {{IO_MEMORY_NAME}} *dQ_buf [[buffer(9)]],
    device const float *Q_scale_buf [[buffer(10)]],
    device const float *K_scale_buf [[buffer(11)]],
    device const float *V_scale_buf [[buffer(12)]],
    device const float *dO_scale_buf [[buffer(13)]],
)";
    break;
  case AttentionKernelType::backwardKeyValue:
    source += R"(
    device int8_t *Q_buf [[buffer(0)]],
    device int8_t *K_buf [[buffer(1)]],
    device int8_t *V_buf [[buffer(2)]],
    device const {{L_MEMORY_NAME}} *L_buf [[buffer(4)]],
    device const {{D_MEMORY_NAME}} *D_buf [[buffer(5)]],
    device int8_t *dO_buf [[buffer(6)]],
    device {{IO_MEMORY_NAME}} *dV_buf [[buffer(7)]],
    device {{IO_MEMORY_NAME}} *dK_buf [[buffer(8)]],
    device const float *Q_scale_buf [[buffer(10)]],
    device const float *K_scale_buf [[buffer(11)]],
    device const float *V_scale_buf [[buffer(12)]],
    device const float *dO_scale_buf [[buffer(13)]],
)";
    break;
  }
  return source.ToString();
}

std::string NAInt8AttentionKernel::createAdjustOffsets() const noexcept {
  CodeWriter source;
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", " / " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  switch (type.value) {
  case AttentionKernelType::forward:
    source += R"(
  Q_buf += tgid.z * Q_batch_stride;
  K_buf += tgid.z * K_batch_stride;
  V_buf += tgid.z * V_batch_stride;
  O_buf += tgid.z * O_batch_stride;
  L_buf += (tgid.z * Hq + tgid.y) * R;
  Q_scale_buf += tgid.z * Q_scale_batch_stride + tgid.y * Q_scale_tiles + (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}}) / Q_scale_tile_size;
  K_scale_buf += tgid.z * K_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles;
  V_scale_buf += tgid.z * V_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles;
  V_mean_buf += tgid.z * V_mean_batch_stride + (tgid.y {{H_HK_RATIO}}) * {{HEAD_DIMENSION}};
)";
    break;
  case AttentionKernelType::backwardQuery:
    source += R"(
  Q_buf += tgid.z * Q_batch_stride;
  K_buf += tgid.z * K_batch_stride;
  V_buf += tgid.z * V_batch_stride;
  dO_buf += tgid.z * dO_batch_stride;
  dQ_buf += tgid.z * dQ_batch_stride;
  L_buf += (tgid.z * Hq + tgid.y) * R;
  D_buf += (tgid.z * Hq + tgid.y) * R;
  Q_scale_buf += tgid.z * Q_scale_batch_stride + tgid.y * Q_scale_tiles + (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}}) / Q_scale_tile_size;
  K_scale_buf += tgid.z * K_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles;
  V_scale_buf += tgid.z * V_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles;
  dO_scale_buf += tgid.z * dO_scale_batch_stride + tgid.y * Q_scale_tiles + (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}}) / Q_scale_tile_size;
)";
    break;
  case AttentionKernelType::backwardKeyValue:
    source += R"(
  Q_buf += tgid.z * Q_batch_stride;
  K_buf += tgid.z * K_batch_stride;
  V_buf += tgid.z * V_batch_stride;
  dO_buf += tgid.z * dO_batch_stride;
  dV_buf += tgid.z * dV_batch_stride;
  dK_buf += tgid.z * dK_batch_stride;
  L_buf += (tgid.z * Hq + tgid.y) * R;
  D_buf += (tgid.z * Hq + tgid.y) * R;
  Q_scale_buf += tgid.z * Q_scale_batch_stride + tgid.y * Q_scale_tiles;
  K_scale_buf += tgid.z * K_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles + (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}}) / KV_scale_tile_size;
  V_scale_buf += tgid.z * V_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles + (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}}) / KV_scale_tile_size;
  dO_scale_buf += tgid.z * dO_scale_batch_stride + tgid.y * Q_scale_tiles;
)";
    break;
  }
  return source.ToString();
}

std::string NAInt8AttentionKernel::createComputeD() const noexcept {
  CodeWriter source;
  const GEMMOperandPrecision dPrecision =
      lowPrecisionIntermediates ?
      (ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::BF16) :
      GEMMOperandPrecision::FP32;
  source.SetValue("IO_MEMORY_NAME", ioPrecision.name());
  source.SetValue("D_MEMORY_NAME", dPrecision.name());
  source.SetValue("V_MEAN_MEMORY_NAME", "float");
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("COMPUTE_D_THREADS", std::to_string(computeDThreads));
  source.SetValue("DOT_SCALE_DERIVATIVE", dot_product_scale(scale, true));
  source += R"(

kernel void compute_d(
    device const {{IO_MEMORY_NAME}}* O_buf [[buffer(3)]],
    device const {{IO_MEMORY_NAME}}* dO_buf [[buffer(6)]],
    device {{D_MEMORY_NAME}}* D_buf [[buffer(5)]],
    device const {{V_MEAN_MEMORY_NAME}}* V_mean_buf [[buffer(14)]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x % R;
  const uint head = tgid.x / R;
  O_buf += tgid.z * O_batch_stride;
  dO_buf += tgid.z * dO_batch_stride;
  D_buf += (tgid.z * Hq + head) * R;
  V_mean_buf += tgid.z * V_mean_batch_stride + head * {{HEAD_DIMENSION}};
  const uint offset = row * K_Hq + head * {{HEAD_DIMENSION}};
  float D_accumulator = 0.0f;
  for (uint d = lane_id; d < {{HEAD_DIMENSION}}; d += {{COMPUTE_D_THREADS}}) {
    const float centered_o = (float)O_buf[offset + d] - (float)V_mean_buf[d];
    D_accumulator += centered_o * (float)dO_buf[offset + d];
  }
  D_accumulator += simd_shuffle_xor(D_accumulator, 16);
  D_accumulator += simd_shuffle_xor(D_accumulator, 8);
  D_accumulator += simd_shuffle_xor(D_accumulator, 4);
  D_accumulator += simd_shuffle_xor(D_accumulator, 2);
  D_accumulator += simd_shuffle_xor(D_accumulator, 1);
  if (lane_id == 0) {
    D_buf[row] = ({{D_MEMORY_NAME}})(D_accumulator * {{DOT_SCALE_DERIVATIVE}});
  }
}

)";
  return source.ToString();
}

void NAInt8AttentionKernel::loopBackwardQuery(CodeWriter& source) const noexcept {
  const unsigned short kBlocks =
      (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  const bool multiHeadBlocks =
      kBlocks > 1 && (headDimension % blockDimensions[2]) == 0;
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("DOT_SCALE", dot_product_scale(scale));
  source.SetValue("DOT_SCALE_DERIVATIVE", dot_product_scale(scale, true));
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  source += R"(
  auto Q = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  auto dO = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(dO_buf, dextents<int32_t, 2>(K_Hq, R));
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
)";
  if (multiHeadBlocks) {
    source += R"(
  auto mQ_0 = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mdO_0 = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto mV_0 = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ_0), decltype(mK_0), int32_t>();
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO_0), decltype(mV_0), int32_t>();
  constexpr auto dsk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dsk_desc, execution_simdgroups<1>> matmul_dsk_op;
  using dsk_accum_left_tensor_t = decltype(matmul_dsk_op.get_left_input_cooperative_tensor<{{D_MEMORY_NAME}}, int8_t, float>());
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<{{D_MEMORY_NAME}}, int8_t, float>();
	)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      if (i > 0) {
        source += R"(
  auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
  auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
	)";
      }
      source += R"(
  auto cDQ_{{LOOP_INDEX}} = matmul_dsk_op.get_destination_cooperative_tensor<dsk_accum_left_tensor_t, decltype(mK_0), float>();
	)";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
    if (cDQ_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDQ_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
    }
    source += R"(
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const float q_scale = Q_scale_buf[0];
  const float dO_scale = dO_scale_buf[0];
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
    const float k_scale = K_scale_buf[c / KV_scale_tile_size];
    const float v_scale = V_scale_buf[c / KV_scale_tile_size];
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
    matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        const float P = fast::exp2((float)cS[k] * (q_scale * k_scale * {{DOT_SCALE}}) - (float)L[idx[1]]);
        const float dS = P * ((float)cDP[k] * (dO_scale * v_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D[idx[1]]);
        cDS[k] = ({{D_MEMORY_NAME}})(dS * k_scale);
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "    matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});\n";
    }
    source += R"(
  }
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
    const float k_scale = K_scale_buf[(C - C_remainder) / KV_scale_tile_size];
    const float v_scale = V_scale_buf[(C - C_remainder) / KV_scale_tile_size];
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
    matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2((float)cS[k] * (q_scale * k_scale * {{DOT_SCALE}}) - (float)L[idx[1]]);
          const float dS = P * ((float)cDP[k] * (dO_scale * v_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D[idx[1]]);
          cDS[k] = ({{D_MEMORY_NAME}})(dS * k_scale);
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "    matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});\n";
    }
    source += R"(
  }
  auto dQ = dQ_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
      if (cDQ_0.is_valid_element(k)) {
        auto idx = cDQ_0.get_multidimensional_index(k);
        if (idx[1] >= (int)R_remainder) {
          continue;
        }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += "      dQ[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cDQ_{{LOOP_INDEX}}[k];\n";
    }
    source += R"(
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
      if (cDQ_0.is_valid_element(k)) {
        auto idx = cDQ_0.get_multidimensional_index(k);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += "      dQ[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cDQ_{{LOOP_INDEX}}[k];\n";
    }
    source += R"(
      }
    }
  }
)";
    return;
  }
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), int32_t>();
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO), decltype(mV), int32_t>();
  constexpr auto dsk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dsk_desc, execution_simdgroups<1>> matmul_dsk_op;
  using dsk_accum_left_tensor_t = decltype(matmul_dsk_op.get_left_input_cooperative_tensor<{{D_MEMORY_NAME}}, int8_t, float>());
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<{{D_MEMORY_NAME}}, int8_t, float>();
  auto cDQ = matmul_dsk_op.get_destination_cooperative_tensor<dsk_accum_left_tensor_t, decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ.get_capacity(); ++k) {
    if (cDQ.is_valid_element(k)) {
      cDQ[k] = 0;
    }
  }
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const float q_scale = Q_scale_buf[0];
  const float dO_scale = dO_scale_buf[0];
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
    const float k_scale = K_scale_buf[c / KV_scale_tile_size];
    const float v_scale = V_scale_buf[c / KV_scale_tile_size];
    auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, c);
    auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, c);
    matmul_qk_op.run(mQ, mK, cS);
    matmul_qk_op.run(mdO, mV, cDP);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        const float P = fast::exp2((float)cS[k] * (q_scale * k_scale * {{DOT_SCALE}}) - (float)L[idx[1]]);
        const float dS = P * ((float)cDP[k] * (dO_scale * v_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D[idx[1]]);
        cDS[k] = ({{D_MEMORY_NAME}})(dS * k_scale);
      }
    }
    matmul_dsk_op.run(cDS, mK, cDQ);
  }
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
    const float k_scale = K_scale_buf[(C - C_remainder) / KV_scale_tile_size];
    const float v_scale = V_scale_buf[(C - C_remainder) / KV_scale_tile_size];
    auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, c);
    auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, c);
    matmul_qk_op.run(mQ, mK, cS);
    matmul_qk_op.run(mdO, mV, cDP);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2((float)cS[k] * (q_scale * k_scale * {{DOT_SCALE}}) - (float)L[idx[1]]);
          const float dS = P * ((float)cDP[k] * (dO_scale * v_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D[idx[1]]);
          cDS[k] = ({{D_MEMORY_NAME}})(dS * k_scale);
        }
      }
    }
    matmul_dsk_op.run(cDS, mK, cDQ);
  }
  auto dQ = dQ_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ.get_capacity(); ++k) {
      if (cDQ.is_valid_element(k)) {
        auto idx = cDQ.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
          dQ[idx[0] + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cDQ[k];
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ.get_capacity(); ++k) {
      if (cDQ.is_valid_element(k)) {
        auto idx = cDQ.get_multidimensional_index(k);
        dQ[idx[0] + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cDQ[k];
      }
    }
  }
)";
}

void NAInt8AttentionKernel::loopBackwardKeyValue(CodeWriter& source) const noexcept {
  const unsigned short kBlocks =
      (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  const bool multiHeadBlocks =
      kBlocks > 1 && (headDimension % blockDimensions[2]) == 0;
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("DOT_SCALE", dot_product_scale(scale));
  source.SetValue("DOT_SCALE_DERIVATIVE", dot_product_scale(scale, true));
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  source += R"(
  auto Q = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  auto dO = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(dO_buf, dextents<int32_t, 2>(K_Hq, R));
  constexpr auto kqt_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<kqt_desc, execution_simdgroups<1>> matmul_kqt_op;
)";
  if (multiHeadBlocks) {
    source += R"(
  threadgroup int8_t *K_shared_buf = (threadgroup int8_t*)threadgroup_block + sgid * ({{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * 2);
  threadgroup int8_t *V_shared_buf = K_shared_buf + {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto K_shared = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(K_shared_buf, extents<int32_t, {{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto V_shared = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(V_shared_buf, extents<int32_t, {{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto mQ_0 = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto mdO_0 = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto mK_0 = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(0, 0);
  auto mV_0 = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(0, 0);
  auto cST = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mK_0), decltype(mQ_0), int32_t>();
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV_0), decltype(mdO_0), int32_t>();
  constexpr auto pdo_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pdo_desc, execution_simdgroups<1>> matmul_pdo_op;
  using pdo_float_left_tensor_t = decltype(matmul_pdo_op.get_left_input_cooperative_tensor<float, int8_t, {{ACCUM_MEMORY_NAME}}>());
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<{{D_MEMORY_NAME}}, int8_t, float>();
  constexpr auto pdo_int8_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<pdo_int8_desc, execution_simdgroups<1>> matmul_pdo_int8_op;
  using pdo_int8_left_tensor_t = decltype(matmul_pdo_int8_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>());
  thread pdo_int8_left_tensor_t& cP_q = reinterpret_cast<thread pdo_int8_left_tensor_t&>(cST);
	)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      if (i > 0) {
        source += R"(
  auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
  auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
  auto mK_{{LOOP_INDEX}} = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>( {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
  auto mV_{{LOOP_INDEX}} = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>( {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
)";
      }
      source += R"(
  auto cDV_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<pdo_float_left_tensor_t, decltype(mdO_0), {{ACCUM_MEMORY_NAME}}>();
  auto cDK_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mQ_0), float>();
  auto cDV_q_{{LOOP_INDEX}} = matmul_pdo_int8_op.get_destination_cooperative_tensor<pdo_int8_left_tensor_t, decltype(mdO_0), int32_t>();
	)";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
    if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDV_{{LOOP_INDEX}}[k] = 0;\n";
      source += "      cDK_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
    }
    source += R"(
  const float k_scale = K_scale_buf[0];
  const float v_scale = V_scale_buf[0];
  const uint lane = tid % 32;
  for (uint load_index = lane; load_index < {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{HEAD_DIMENSION}};
    const uint row_idx = load_index / {{HEAD_DIMENSION}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    if (row < C) {
      K_shared_buf[load_index] = K_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
      V_shared_buf[load_index] = V_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
    } else {
      K_shared_buf[load_index] = 0;
      V_shared_buf[load_index] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0; r < KV_R_edge; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
    const float q_scale = Q_scale_buf[r / Q_scale_tile_size];
    const float dO_scale = dO_scale_buf[r / Q_scale_tile_size];
    const float dO_scale_recip_127 = dO_scale * (1.0f / 127.0f);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
    auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
    matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
    matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_q.get_capacity(); ++k) {
      if (cP_q.is_valid_element(k)) {
        auto idx = cP_q.get_multidimensional_index(k);
        const float P = fast::exp2((float)cST[k] * (k_scale * q_scale * {{DOT_SCALE}}) - (float)L_buf[r + idx[0]]);
        const int quantized = (int)rint(P * 127.0f);
        cP_q[k] = (int8_t)clamp(quantized, 0, 127);
        const float dS = P * ((float)cDP[k] * (v_scale * dO_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D_buf[r + idx[0]]);
        cDS[k] = ({{D_MEMORY_NAME}})(dS * q_scale);
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "    matmul_pdo_int8_op.run(cP_q, mdO_{{LOOP_INDEX}}, cDV_q_{{LOOP_INDEX}});\n";
      source += "    matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});\n";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "        cDV_{{LOOP_INDEX}}[k] += ({{ACCUM_MEMORY_NAME}})((float)cDV_q_{{LOOP_INDEX}}[k] * dO_scale_recip_127);\n";
    }
    source += R"(
      }
    }
  }
  if (KV_R_remainder > 0) {
    const uint r = R - KV_R_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
    const float q_scale = Q_scale_buf[(R - KV_R_remainder) / Q_scale_tile_size];
    const float dO_scale = dO_scale_buf[(R - KV_R_remainder) / Q_scale_tile_size];
    const float dO_scale_recip_127 = dO_scale * (1.0f / 127.0f);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
    auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
    matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
    matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_q.get_capacity(); ++k) {
      if (cP_q.is_valid_element(k)) {
        auto idx = cP_q.get_multidimensional_index(k);
        if (idx[0] >= (int)KV_R_remainder) {
          cP_q[k] = 0;
          cDS[k] = 0;
        } else {
          const float P = fast::exp2((float)cST[k] * (k_scale * q_scale * {{DOT_SCALE}}) - (float)L_buf[r + idx[0]]);
          const int quantized = (int)rint(P * 127.0f);
          cP_q[k] = (int8_t)clamp(quantized, 0, 127);
          const float dS = P * ((float)cDP[k] * (v_scale * dO_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D_buf[r + idx[0]]);
          cDS[k] = ({{D_MEMORY_NAME}})(dS * q_scale);
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "    matmul_pdo_int8_op.run(cP_q, mdO_{{LOOP_INDEX}}, cDV_q_{{LOOP_INDEX}});\n";
      source += "    matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});\n";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "        cDV_{{LOOP_INDEX}}[k] += ({{ACCUM_MEMORY_NAME}})((float)cDV_q_{{LOOP_INDEX}}[k] * dO_scale_recip_127);\n";
    }
    source += R"(
      }
    }
  }
  auto dK = dK_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  auto dV = dV_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  if (KV_C_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= KV_C_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
        auto idx = cDV_0.get_multidimensional_index(k);
        if (idx[1] >= (int)KV_C_remainder) {
          continue;
        }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += "      dV[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDV_{{LOOP_INDEX}}[k];\n";
      source += "      dK[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDK_{{LOOP_INDEX}}[k];\n";
    }
    source += R"(
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
        auto idx = cDV_0.get_multidimensional_index(k);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += "      dV[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDV_{{LOOP_INDEX}}[k];\n";
      source += "      dK[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDK_{{LOOP_INDEX}}[k];\n";
    }
    source += R"(
      }
    }
  }
)";
    return;
  }
  source += R"(
  threadgroup int8_t *K_shared_buf = (threadgroup int8_t*)threadgroup_block + sgid * ({{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * 2);
  threadgroup int8_t *V_shared_buf = K_shared_buf + {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto K_shared = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(K_shared_buf, extents<int32_t, {{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto V_shared = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(V_shared_buf, extents<int32_t, {{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto mK = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(0, 0);
  auto mV = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(0, 0);
  auto cST = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mK), decltype(mQ), int32_t>();
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV), decltype(mdO), int32_t>();
  constexpr auto pdo_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pdo_desc, execution_simdgroups<1>> matmul_pdo_op;
  using pdo_float_left_tensor_t = decltype(matmul_pdo_op.get_left_input_cooperative_tensor<float, int8_t, float>());
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<{{D_MEMORY_NAME}}, int8_t, float>();
  constexpr auto pdo_int8_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<pdo_int8_desc, execution_simdgroups<1>> matmul_pdo_int8_op;
  using pdo_int8_left_tensor_t = decltype(matmul_pdo_int8_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>());
  thread pdo_int8_left_tensor_t& cP_q = reinterpret_cast<thread pdo_int8_left_tensor_t&>(cST);
  auto cDV = matmul_pdo_op.get_destination_cooperative_tensor<pdo_float_left_tensor_t, decltype(mdO), {{ACCUM_MEMORY_NAME}}>();
  auto cDK = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mQ), float>();
  auto cDV_q = matmul_pdo_int8_op.get_destination_cooperative_tensor<pdo_int8_left_tensor_t, decltype(mdO), int32_t>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV.get_capacity(); ++k) {
    if (cDV.is_valid_element(k)) {
      cDV[k] = 0;
      cDK[k] = 0;
    }
  }
  const float k_scale = K_scale_buf[0];
  const float v_scale = V_scale_buf[0];
  for (uint load_index = lane_id; load_index < {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{HEAD_DIMENSION}};
    const uint row_idx = load_index / {{HEAD_DIMENSION}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    if (row < C) {
      K_shared_buf[load_index] = K_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
      V_shared_buf[load_index] = V_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
    } else {
      K_shared_buf[load_index] = 0;
      V_shared_buf[load_index] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0; r < KV_R_edge; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
    const float q_scale = Q_scale_buf[r / Q_scale_tile_size];
    const float dO_scale = dO_scale_buf[r / Q_scale_tile_size];
    const float dO_scale_recip_127 = dO_scale * (1.0f / 127.0f);
    auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, r);
    auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, r);
    matmul_kqt_op.run(mK, mQ, cST);
    matmul_kqt_op.run(mV, mdO, cDP);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_q.get_capacity(); ++k) {
      if (cP_q.is_valid_element(k)) {
        auto idx = cP_q.get_multidimensional_index(k);
        const float P = fast::exp2((float)cST[k] * (k_scale * q_scale * {{DOT_SCALE}}) - (float)L_buf[r + idx[0]]);
        const int quantized = (int)rint(P * 127.0f);
        cP_q[k] = (int8_t)clamp(quantized, 0, 127);
        const float dS = P * ((float)cDP[k] * (v_scale * dO_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D_buf[r + idx[0]]);
        cDS[k] = ({{D_MEMORY_NAME}})(dS * q_scale);
      }
    }
    matmul_pdo_int8_op.run(cP_q, mdO, cDV_q);
    matmul_pdo_op.run(cDS, mQ, cDK);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV.get_capacity(); ++k) {
      if (cDV.is_valid_element(k)) {
        cDV[k] += ({{ACCUM_MEMORY_NAME}})((float)cDV_q[k] * dO_scale_recip_127);
      }
    }
  }
  if (KV_R_remainder > 0) {
    const uint r = R - KV_R_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
    const float q_scale = Q_scale_buf[(R - KV_R_remainder) / Q_scale_tile_size];
    const float dO_scale = dO_scale_buf[(R - KV_R_remainder) / Q_scale_tile_size];
    const float dO_scale_recip_127 = dO_scale * (1.0f / 127.0f);
    auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, r);
    auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, r);
    matmul_kqt_op.run(mK, mQ, cST);
    matmul_kqt_op.run(mV, mdO, cDP);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_q.get_capacity(); ++k) {
      if (cP_q.is_valid_element(k)) {
        auto idx = cP_q.get_multidimensional_index(k);
        if (idx[0] >= (int)KV_R_remainder) {
          cP_q[k] = 0;
          cDS[k] = 0;
        } else {
          const float P = fast::exp2((float)cST[k] * (k_scale * q_scale * {{DOT_SCALE}}) - (float)L_buf[r + idx[0]]);
          const int quantized = (int)rint(P * 127.0f);
          cP_q[k] = (int8_t)clamp(quantized, 0, 127);
          const float dS = P * ((float)cDP[k] * (v_scale * dO_scale * {{DOT_SCALE_DERIVATIVE}}) - (float)D_buf[r + idx[0]]);
          cDS[k] = ({{D_MEMORY_NAME}})(dS * q_scale);
        }
      }
    }
    matmul_pdo_int8_op.run(cP_q, mdO, cDV_q);
    matmul_pdo_op.run(cDS, mQ, cDK);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV.get_capacity(); ++k) {
      if (cDV.is_valid_element(k)) {
        cDV[k] += ({{ACCUM_MEMORY_NAME}})((float)cDV_q[k] * dO_scale_recip_127);
      }
    }
  }
  auto dK = dK_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  auto dV = dV_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  if (KV_C_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= KV_C_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV.get_capacity(); ++k) {
      if (cDV.is_valid_element(k)) {
        auto idx = cDV.get_multidimensional_index(k);
        if (idx[1] >= (int)KV_C_remainder) {
          continue;
        }
        dV[idx[0] + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDV[k];
        dK[idx[0] + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDK[k];
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV.get_capacity(); ++k) {
      if (cDV.is_valid_element(k)) {
        auto idx = cDV.get_multidimensional_index(k);
        dV[idx[0] + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDV[k];
        dK[idx[0] + idx[1] * K_Hk] = ({{IO_MEMORY_NAME}})cDK[k];
      }
    }
  }
)";
}

void NAInt8AttentionKernel::loopForward(CodeWriter& source) const noexcept {
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
  source.SetValue("DOT_SCALE", dot_product_scale(scale));
  source.SetValue("PV_RIGHT_MEMORY_NAME", "int8_t");
  if ((headDimension % blockDimensions[2]) % 32 == 0) {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", std::to_string(headDimension % blockDimensions[2]));
  } else {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  const unsigned short kBlocks =
      (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }

  source += R"(
  auto Q = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{PV_RIGHT_MEMORY_NAME}}, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
)";
  if (headDimension % blockDimensions[2] > 0) {
    source += R"(
  constexpr auto qk_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_remainder, execution_simdgroups<1>> matmul_qk_op_remainder;
)";
  }
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), int32_t>();
  using qk_float_tensor_t = decltype(matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>());
  thread qk_float_tensor_t& cP_0 = reinterpret_cast<thread qk_float_tensor_t&>(cS_0);
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
  using pv_float_left_tensor_t = decltype(matmul_pv_op.get_left_input_cooperative_tensor<half, {{PV_RIGHT_MEMORY_NAME}}, float>());
  constexpr auto pv_int8_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<pv_int8_desc, execution_simdgroups<1>> matmul_pv_int8_op;
  using pv_int8_left_tensor_t = decltype(matmul_pv_int8_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>());
  auto cOq = matmul_pv_int8_op.get_destination_cooperative_tensor<pv_int8_left_tensor_t, decltype(mV), int32_t>();
  threadgroup int8_t *Pq_buf = (threadgroup int8_t*)threadgroup_block + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sgid;
  auto Pq = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(Pq_buf, extents<int32_t, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  constexpr auto pv_int8_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<pv_int8_desc_remainder, execution_simdgroups<1>> matmul_pv_int8_op_remainder;
  auto cOq_remainder = matmul_pv_int8_op_remainder.get_destination_cooperative_tensor<decltype(Pq), decltype(mV), int32_t>();
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<pv_float_left_tensor_t, decltype(mV), float>();\n";
  }
  source += R"(
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    const float block_scale = {{QK_SCALE_FACTOR_0}}{{DOT_SCALE}};
    const float v_scale_recip_127 = {{V_SCALE_FACTOR_0}} / 127.0f;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        const int score_0 = cS_0[k];
        cP_0[k] = (float)score_0 * block_scale;
      }
    }
    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP_0, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k];
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
        cL[k] *= correction[k];
      }
    }
    if (c == 0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  source += R"(
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
  }
  source += R"(
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        auto it = cP_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cP_0[k] = fast::exp2(cP_0[k] - *dst_it);
      }
    }
    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP_0, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
        cL[k] += cL_new[k];
      }
    }
    thread pv_int8_left_tensor_t& cP_q_0 = reinterpret_cast<thread pv_int8_left_tensor_t&>(cS_0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        const int quantized = (int)rint(cP_0[k] * 127.0f);
        cP_q_0[k] = (int8_t)clamp(quantized, 0, 127);
      }
    }
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_int8_op.run(cP_q_0, mV_0_{{LOOP_INDEX}}, cOq);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cOq.get_capacity(); ++k) {
      if (cOq.is_valid_element(k)) {
        cO_{{LOOP_INDEX}}[k] += (float)cOq[k] * v_scale_recip_127;
      }
    }
)";
  }
  source += R"(
    if ({{THREAD_BARRIER_EVERY_C}} > 0 &&
        c + {{BLOCK_DIMENSIONS_TRAVERSAL}} < C &&
        ((((c / {{BLOCK_DIMENSIONS_TRAVERSAL}}) + 1) % {{THREAD_BARRIER_EVERY_C}}) == 0)) {
      threadgroup_barrier(mem_flags::mem_none);
    }
  }
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
    const float block_scale = {{QK_SCALE_FACTOR_REM}}{{DOT_SCALE}};
    const float v_scale_recip_127 = {{V_SCALE_FACTOR_REM}} / 127.0f;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        auto idx = cP_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cP_0[k] = -numeric_limits<float>::infinity();
        } else {
          cP_0[k] = 0;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        auto idx = cP_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cP_0[k] = -numeric_limits<float>::infinity();
        } else {
          cP_0[k] = (float)cS_0[k] * block_scale;
        }
      }
    }
    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP_0, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k];
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
        cL[k] *= correction[k];
      }
    }
    if (c == 0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  source += R"(
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
  }
  source += R"(
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        auto it = cP_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cP_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cP_0[k] = 0;
        } else {
          cP_0[k] = fast::exp2(cP_0[k] - *dst_it);
        }
      }
    }
    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP_0, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
        cL[k] += cL_new[k];
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        auto idx = cP_0.get_multidimensional_index(k);
        const int quantized = (int)rint(cP_0[k] * 127.0f);
        if (idx[0] >= (int)C_remainder) {
          Pq_buf[idx[0] - C_remainder + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
        } else {
          Pq_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = (int8_t)clamp(quantized, 0, 127);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto mP_q = Pq.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder, 0);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_int8_op_remainder.run(mP_q, mV_0_{{LOOP_INDEX}}, cOq_remainder);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cOq_remainder.get_capacity(); ++k) {
      if (cOq_remainder.is_valid_element(k)) {
        cO_{{LOOP_INDEX}}[k] += (float)cOq_remainder[k] * v_scale_recip_127;
      }
    }
)";
  }
  source += R"(
  }
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = cL.map_iterator(it);
          auto L_reciprocal = fast::divide(1, *dst_it);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal)";
      source += " + (float)V_mean_buf[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}]";
      source += ");\n";
    } else {
      source += R"(
          if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
            O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal)";
      source += " + (float)V_mean_buf[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}]";
      source += ");\n";
      source += R"(
          }
)";
    }
  }
  source += R"(
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        if (idx[0] < (int)R_remainder) {
          L[idx[0]] = cM[k] + fast::log2(cL[k]);
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
        O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal)";
      source += " + (float)V_mean_buf[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}]";
      source += ");\n";
    } else {
      source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal)";
      source += " + (float)V_mean_buf[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}]";
      source += ");\n";
      source += R"(
        }
)";
    }
  }
  source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        L[idx[0]] = cM[k] + fast::log2(cL[k]);
      }
    }
  }
)";
}
