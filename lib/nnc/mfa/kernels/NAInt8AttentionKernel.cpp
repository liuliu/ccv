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

static std::string raw_score_scale(float rsqrt_d) {
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
  headDimension = descriptor.headDimension;
  Hq = descriptor.Hq;
  Hk = descriptor.Hk;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  checkCEdge1 = descriptor.checkCEdge1;
  useInt8QK = descriptor.useInt8QK;
  useQKScales = descriptor.useQKScales;
  threadBarrierOverC = descriptor.threadBarrierOverC;
  mortonOrder = descriptor.mortonOrder;
  ioPrecision = descriptor.ioPrecision;
  mode = descriptor.mode;
  scale = descriptor.scale;

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

unsigned short NAInt8AttentionKernel::threadgroupMemoryAllocation() const noexcept {
  if (mode == NAInt8AttentionKernelMode::qk_only ||
      mode == NAInt8AttentionKernelMode::softmax_stats)
  {
    return 0;
  }
  if (mode == NAInt8AttentionKernelMode::full)
    return 0;
  const uint32_t simdgroup_buffers = executionSIMDGroups;
  const uint32_t total_bytes =
      blockDimensions[0] * blockDimensions[1] * simdgroup_buffers * sizeof(_Float16);
  CCV_NNC_MFA_PRECONDITION(total_bytes <= std::numeric_limits<unsigned short>::max());
  return (unsigned short)total_bytes;
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
  if (!mortonOrder)
    return MTL::Size(row_groups * Hq * batchDimension, 1, 1);
  const uint32_t row_bits = ceilLog2(row_groups);
  const uint32_t head_bits = ceilLog2(Hq);
  return MTL::Size(int64_t(1) << (row_bits + head_bits), 1, batchDimension);
}

std::string NAInt8AttentionKernel::createSource() const noexcept {
  CodeWriter source;
  const bool vectorizeQuantize = (headDimension % 4) == 0;
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";
  source.SetValue("IO_MEMORY_NAME", ioPrecision.name());
  source += R"(
constant uint QUANTIZE_Q_THREADS = )" + std::to_string(qQuantizeThreads) + R"(;
constant uint QUANTIZE_KV_THREADS = )" + std::to_string(kvQuantizeThreads) + R"(;
constant uint QUANTIZE_SIMD_LANES = 32;
constant uint QUANTIZE_Q_SIMDGROUPS = QUANTIZE_Q_THREADS / QUANTIZE_SIMD_LANES;
constant uint QUANTIZE_KV_SIMDGROUPS = QUANTIZE_KV_THREADS / QUANTIZE_SIMD_LANES;
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

kernel void quantize_v(
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
      QUANTIZE_V_BATCH_STRIDE,
      QUANTIZE_KV_SCALE_BATCH_STRIDE,
      QUANTIZE_KV_THREADS,
      QUANTIZE_KV_SIMDGROUPS);
}

)";
  source += R"(
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
  createConstants(source);
  source += R"(
kernel void int8_attention(
)";
  source += createBufferBindings();
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("HQ", std::to_string(Hq));
  source.SetValue("MATMUL_SIMDGROUPS", "1");
  source.SetValue("THREAD_BARRIER_OVER_C", std::to_string(threadBarrierOverC ? 1 : 0));
  source.SetValue("MORTON_ORDER", std::to_string(mortonOrder ? 1 : 0));
  source += R"(
    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
)";
  if (mortonOrder) {
    source += R"(
  const uint row_group_count = (R + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group_bits = ceil_log2_u32(row_group_count);
  const uint head_bits = ceil_log2_u32({{HQ}});
  const uint tile_code = tgid.x;
  const uint2 morton_tile = morton_decode_rectangular_2d(tile_code, row_group_bits, head_bits);
  tgid = uint3(morton_tile.x, morton_tile.y, tgid.z);
  if (tgid.y >= {{HQ}} || tgid.x >= row_group_count) {
    return;
  }
)";
  } else {
    source += R"(
  tgid = { (tgid.x / {{HQ}}) % ((R + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}})), tgid.x % {{HQ}}, tgid.x / {{HQ}} / ((R + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}})) };
)";
  }
  source += R"(
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R) {
    return;
  }
)";
  source += createAdjustOffsets();
  loopForward(source);
  source += "}\n";
  return source.ToString();
}

void NAInt8AttentionKernel::createConstants(CodeWriter& source) const noexcept {
  source.SetValue("HQ", std::to_string(Hq));
  source.SetValue("HK", std::to_string(Hk));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_2", std::to_string(blockDimensions[1] * 2));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("Q_SCALE_TILES", "(R + " + std::to_string(blockDimensions[0]) + " - 1) / " + std::to_string(blockDimensions[0]));
  source.SetValue("K_SCALE_TILES", "(C + " + std::to_string(blockDimensions[1]) + " - 1) / " + std::to_string(blockDimensions[1]));
  source += R"(
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint Q_batch_stride [[function_constant(2)]];
constant uint K_batch_stride [[function_constant(3)]];
constant uint V_batch_stride [[function_constant(4)]];
constant uint O_batch_stride [[function_constant(5)]];
constant uint Q_scale_batch_stride [[function_constant(6)]];
constant uint K_scale_batch_stride [[function_constant(7)]];
constant uint V_scale_batch_stride [[function_constant(8)]];

constant uint Hq = {{HQ}};
constant uint Hk = {{HK}};
constant uint K_Hq = {{HEAD_DIMENSION}} * Hq;
constant uint K_Hk = {{HEAD_DIMENSION}} * Hk;
constant uint Q_scale_tiles = {{Q_SCALE_TILES}};
constant uint K_scale_tiles = {{K_SCALE_TILES}};
constant uint C_remainder = (C % {{BLOCK_DIMENSIONS_TRAVERSAL_2}}) == {{BLOCK_DIMENSIONS_TRAVERSAL}} ? {{BLOCK_DIMENSIONS_TRAVERSAL}} : (C % {{BLOCK_DIMENSIONS_TRAVERSAL}});
)";
  if (checkCEdge1) {
    source += R"(
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint C_edge_1 = C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL_2}} : 0;
)";
  } else {
    source += R"(
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL_2}} : 0;
constant uint C_edge_1 = C_edge;
)";
  }
  source += R"(
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
  source.SetValue("QK_SCALE_FACTOR_0", useQKScales ? ("Q_scale_buf[0] * K_scale_buf[c / " + std::to_string(blockDimensions[1]) + "] * ") : "");
  source.SetValue("QK_SCALE_FACTOR_1", useQKScales ? ("Q_scale_buf[0] * K_scale_buf[c / " + std::to_string(blockDimensions[1]) + " + 1] * ") : "");
  source.SetValue("QK_SCALE_FACTOR_REM", useQKScales ? ("Q_scale_buf[0] * K_scale_buf[(C - C_remainder) / " + std::to_string(blockDimensions[1]) + "] * ") : "");
  source.SetValue("V_SCALE_FACTOR_0", "V_scale_buf[c / " + std::to_string(blockDimensions[1]) + "]");
  source.SetValue("V_SCALE_FACTOR_1", "V_scale_buf[c / " + std::to_string(blockDimensions[1]) + " + 1]");
  source.SetValue("V_SCALE_FACTOR_REM", "V_scale_buf[(C - C_remainder) / " + std::to_string(blockDimensions[1]) + "]");
}

std::string NAInt8AttentionKernel::createBufferBindings() const noexcept {
  const std::string qk_memory_name = useInt8QK ? "int8_t" : ioPrecision.name();
  CodeWriter source;
  source.SetValue("IO_MEMORY_NAME", ioPrecision.name());
  source.SetValue("QK_MEMORY_NAME", qk_memory_name);
  const char* v_memory_name = "int8_t";
  source.SetValue("QK_MEMORY_NAME", qk_memory_name);
  source.SetValue("V_MEMORY_NAME", v_memory_name);
  source += R"(
    device {{QK_MEMORY_NAME}} *Q_buf [[buffer(0)]],
    device {{QK_MEMORY_NAME}} *K_buf [[buffer(1)]],
    device {{V_MEMORY_NAME}} *V_buf [[buffer(2)]],
    device {{IO_MEMORY_NAME}} *O_buf [[buffer(3)]],
    device float *L_buf [[buffer(4)]],
    device float *Q_scale_buf [[buffer(5)]],
    device float *K_scale_buf [[buffer(6)]],
    device float *V_scale_buf [[buffer(7)]],
)";
  return source.ToString();
}

std::string NAInt8AttentionKernel::createAdjustOffsets() const noexcept {
  CodeWriter source;
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", " / " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source += R"(
  Q_buf += tgid.z * Q_batch_stride;
  K_buf += tgid.z * K_batch_stride;
  V_buf += tgid.z * V_batch_stride;
  O_buf += tgid.z * O_batch_stride;
  L_buf += (tgid.z * Hq + tgid.y) * R;
)";
  if (useQKScales) {
    source += R"(
  Q_scale_buf += tgid.z * Q_scale_batch_stride + tgid.y * Q_scale_tiles + tgid.x;
  K_scale_buf += tgid.z * K_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles;
)";
  }
  source += R"(
  V_scale_buf += tgid.z * V_scale_batch_stride + (tgid.y {{H_HK_RATIO}}) * K_scale_tiles;
)";
  return source.ToString();
}

void NAInt8AttentionKernel::loopForward(CodeWriter& source) const noexcept {
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
  source.SetValue("DOT_SCALE", dot_product_scale(scale));
  source.SetValue("RAW_SCALE", raw_score_scale(scale));
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
  const unsigned short kBlocks =
      (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  const bool uses_qk = mode != NAInt8AttentionKernelMode::pv_only;
  const bool stats_only_mode = mode == NAInt8AttentionKernelMode::softmax_stats;
  const bool uses_pv = mode != NAInt8AttentionKernelMode::qk_only &&
      !stats_only_mode;
  const bool full_mode = mode == NAInt8AttentionKernelMode::full;
  const bool qk_only_mode = mode == NAInt8AttentionKernelMode::qk_only;
  const bool raw_mode = mode == NAInt8AttentionKernelMode::qk_pv_raw;
  const bool int8_pv_mode = mode == NAInt8AttentionKernelMode::pv_only;

  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }

  if (full_mode) {
    CCV_NNC_MFA_PRECONDITION(useInt8QK);
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
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<pv_float_left_tensor_t, decltype(mV), float>();\n";
    }
    source += R"(
  for (uint c = 0; c < C; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    const float block_scale = {{QK_SCALE_FACTOR_0}}{{DOT_SCALE}};
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
)";
    source += R"(
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
        cO_{{LOOP_INDEX}}[k] += (float)cOq[k] * ({{V_SCALE_FACTOR_0}} / 127.0f);
      }
    }
)";
    }
    source += R"(
    if ({{THREAD_BARRIER_OVER_C}} &&
        c + {{BLOCK_DIMENSIONS_TRAVERSAL}} < C &&
        (((c / {{BLOCK_DIMENSIONS_TRAVERSAL}}) & 1) == 1)) {
      threadgroup_barrier(mem_flags::mem_none);
    }
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
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
      } else {
        source += R"(
          if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
            O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
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
        O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
      } else {
        source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
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
    return;
  }

  if (uses_qk) {
    if (useInt8QK) {
      source += R"(
  auto Q = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
)";
    } else {
      source += R"(
  auto Q = tensor<device {{QK_MEMORY_NAME}}, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{QK_MEMORY_NAME}}, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
)";
    }
  }
  if (uses_pv) {
    source += R"(
  auto V = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
)";
    source += R"(
  threadgroup half *P_buf = (threadgroup half*)threadgroup_block + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sgid;
)";
    source += R"(
  auto P = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
)";
  }
  if (uses_qk) {
    source += R"(
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<{{MATMUL_SIMDGROUPS}}>> matmul_qk_op;
)";
    if (headDimension % blockDimensions[2] > 0) {
      source += R"(
  constexpr auto qk_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_remainder, execution_simdgroups<{{MATMUL_SIMDGROUPS}}>> matmul_qk_op_remainder;
)";
    }
    source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
)";
    if (useInt8QK) {
      source += R"(
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), int32_t>();
)";
    } else {
      source += R"(
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
)";
    }
    if (stats_only_mode || (useInt8QK && raw_mode)) {
      source += R"(
  auto cP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
)";
    }
    if (stats_only_mode) {
      source += R"(
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
)";
    } else if (qk_only_mode) {
      if (useInt8QK) {
        source += R"(
  auto cChecksum = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), int32_t>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cChecksum.get_capacity(); ++k) {
    if (cChecksum.is_valid_element(k)) {
      cChecksum[k] = 0;
    }
  }
)";
      } else {
        source += R"(
  auto cChecksum = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cChecksum.get_capacity(); ++k) {
    if (cChecksum.is_valid_element(k)) {
      cChecksum[k] = 0;
    }
  }
)";
      }
    }
  }
  if (uses_pv) {
    source += R"(
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<{{MATMUL_SIMDGROUPS}}>> matmul_pv_op;
)";
    if (raw_mode) {
      source += "  auto cP_raw = matmul_pv_op.get_left_input_cooperative_tensor<half, {{PV_RIGHT_MEMORY_NAME}}, float>();\n";
    } else if (int8_pv_mode) {
      source += "  auto cP_raw = matmul_pv_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();\n";
    }
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      if (raw_mode) {
        source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP_raw), decltype(mV), float>();\n";
      } else if (int8_pv_mode) {
        source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP_raw), decltype(mV), int32_t>();\n";
      } else {
        source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();\n";
      }
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
)";
      for (unsigned short i = 0; i < kBlocks; ++i) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source += "      cO_{{LOOP_INDEX}}[k] = 0;\n";
      }
    source += R"(
    }
  }
)";
  }
  source += R"(
  for (uint c = 0; c < C; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
)";
  if (uses_qk) {
    if (useInt8QK) {
      if (stats_only_mode) {
        source += R"(
    const float block_scale = {{QK_SCALE_FACTOR_0}}{{DOT_SCALE}};
)";
      } else {
        source += R"(
    const float block_scale = {{QK_SCALE_FACTOR_0}}{{RAW_SCALE}};
)";
      }
    } else {
      if (stats_only_mode) {
        source += R"(
    const float block_scale = {{DOT_SCALE}};
)";
      } else {
        source += R"(
    const float block_scale = {{RAW_SCALE}};
)";
      }
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK, cS);
    }
)";
    if (headDimension % blockDimensions[2] > 0) {
      source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
      source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK, cS);
    }
)";
    }
  }
  if (stats_only_mode) {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        if (c + idx[0] >= C) {
          cP[k] = -numeric_limits<float>::infinity();
        } else {
          cP[k] = (float)cS[k] * block_scale;
        }
      }
    }
    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k];
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        if (!isfinite(cP[k])) {
          cP[k] = 0;
        } else {
          auto it = cP.get_iterator(k);
          auto dst_it = cM.map_iterator(it);
          cP[k] = fast::exp2(cP[k] - *dst_it);
        }
      }
    }
    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_new[k];
      }
    }
)";
  } else if (qk_only_mode) {
    if (useInt8QK) {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        auto idx = cS.get_multidimensional_index(k);
        if (idx[0] == 0 && c + idx[0] < C) {
          auto it = cS.get_iterator(k);
          auto dst_it = cChecksum.map_iterator(it);
          *dst_it += cS[k];
        }
      }
    }
)";
    } else {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        auto idx = cS.get_multidimensional_index(k);
        if (idx[0] == 0 && c + idx[0] < C) {
          auto it = cS.get_iterator(k);
          auto dst_it = cChecksum.map_iterator(it);
          *dst_it += cS[k] * block_scale;
        }
      }
    }
)";
    }
  } else if (raw_mode) {
    if (useInt8QK) {
      source += R"(
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        const bool valid_column = c + idx[0] < C;
        cP_raw[k] = valid_column ? (half)((float)cS[k] * block_scale) : (half)0;
      }
    }
)";
    } else {
      source += R"(
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        auto idx = cS.get_multidimensional_index(k);
        const bool valid_column = c + idx[0] < C;
        cP_raw[k] = valid_column ? (half)(cS[k] * block_scale) : (half)0;
      }
    }
)";
    }
  } else if (int8_pv_mode) {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_raw.get_capacity(); ++k) {
      if (cP_raw.is_valid_element(k)) {
        auto idx = cP_raw.get_multidimensional_index(k);
        const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + idx[1];
        const uint column = c + idx[0];
        if (column >= C) {
          cP_raw[k] = (int8_t)0;
        } else {
          const int value = (int)((row * 13 + column * 7 + tgid.y * 5) % 29) - 14;
          cP_raw[k] = (int8_t)value;
        }
      }
    }
)";
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (ushort row_idx = 0; row_idx < {{BLOCK_DIMENSIONS_PARALLELIZATION}}; ++row_idx) {
      #pragma clang loop unroll(full)
      for (ushort column_idx = 0; column_idx < {{BLOCK_DIMENSIONS_TRAVERSAL}}; ++column_idx) {
        const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
        const uint column = c + column_idx;
        if (column >= C) {
          P_buf[column_idx + row_idx * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = (half)0;
        } else {
          const int value = (int)((row * 13 + column * 7 + tgid.y * 5) % 29) - 14;
          P_buf[column_idx + row_idx * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = (half)(value * 0.0625f);
        }
      }
    }
)";
  }
  if (uses_pv) {
    if (!raw_mode && !int8_pv_mode) {
      source += R"(
    simdgroup_barrier(mem_flags::mem_threadgroup);
)";
    }
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      if (raw_mode) {
        source += R"(
    auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(cP_raw, mV_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      } else if (int8_pv_mode) {
        source += R"(
    auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(cP_raw, mV_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      } else {
        source += R"(
    auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(P, mV_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    }
  }
  source += R"(
  }
)";
  if (stats_only_mode) {
    source += R"(
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
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
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        L[idx[0]] = cM[k] + fast::log2(cL[k]);
      }
    }
  }
")";
  } else if (qk_only_mode) {
    source += R"(
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cChecksum.get_capacity(); ++k) {
      if (cChecksum.is_valid_element(k)) {
        auto idx = cChecksum.get_multidimensional_index(k);
        if (idx[0] < (int)R_remainder) {
          L[idx[0]] = (float)cChecksum[k];
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cChecksum.get_capacity(); ++k) {
      if (cChecksum.is_valid_element(k)) {
        auto idx = cChecksum.get_multidimensional_index(k);
        L[idx[0]] = (float)cChecksum[k];
      }
    }
  }
)";
  } else {
    source += R"(
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
        source += R"(
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cO_{{LOOP_INDEX}}[k];
)";
      } else {
        source += R"(
          if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
            O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cO_{{LOOP_INDEX}}[k];
          }
)";
      }
    }
    source += R"(
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
        source += R"(
        O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cO_{{LOOP_INDEX}}[k];
)";
      } else {
        source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{IO_MEMORY_NAME}})cO_{{LOOP_INDEX}}[k];
        }
)";
      }
    }
    source += R"(
      }
    }
  }
)";
  }
}
