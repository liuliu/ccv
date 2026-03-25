#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/AttentionOperand.hpp"
#include "nnc/mfa/kernels/AttentionKernelType.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAAttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAAttentionKernel.hpp"
#include "nnc/mfa/kernels/NAAttentionKernelDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernelDescriptor.hpp"

namespace {

using half_float = _Float16;

enum class InputPrecision : uint8_t {
  fp16 = 0,
  bf16 = 1,
  fp32 = 2,
};

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
};

struct AttentionCase {
  uint32_t batch = 1;
  uint32_t R = 4096;
  uint32_t C = 4096;
  uint32_t Hq = 16;
  uint32_t Hk = 16;
  uint32_t D = 128;
};

struct VariantConfig {
  uint16_t baseline_execution_simd_groups_override = 0;
  uint16_t int8_execution_simd_groups_override = 0;
  uint16_t int8_block_r_override = 0;
  uint16_t int8_block_c_override = 0;
  uint16_t int8_block_d_override = 0;
  uint16_t q_quant_threads_override = 0;
  uint16_t kv_quant_threads_override = 0;
  bool int8_thread_barrier_over_c = true;
  bool int8_qk_scales = true;
  bool int8_qk = true;
  bool quantize_only = false;
  InputPrecision input_precision = InputPrecision::fp16;
  const char* capture_path = nullptr;
};

struct Stats {
  double average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct ValidationStats {
  bool passed = false;
  bool full_reference = false;
  size_t checked_batches = 0;
  size_t checked_heads = 0;
  size_t checked_rows = 0;
  double max_abs_o = 0;
  double max_rel_o = 0;
  double max_abs_l = 0;
  double max_rel_l = 0;
};

struct BaselinePipeline {
  NAAttentionDescriptor descriptor;
  AttentionOperands<GEMMOperandPrecision> memory_precisions;
  std::unique_ptr<NAAttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct Int8Pipeline {
  std::unique_ptr<NAInt8AttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  simd::ushort3 block_dimensions = simd::ushort3 { 0, 0, 0 };
  uint16_t execution_simd_groups = 0;
  bool use_int8_qk = true;
  bool use_qk_scales = true;
  bool thread_barrier_over_c = false;
};

struct QuantizePipelines {
  NS::SharedPtr<MTL::ComputePipelineState> q_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> k_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> v_pipeline;
  uint16_t q_threads = 0;
  uint16_t kv_threads = 0;
};

struct QuantizedQK {
  std::vector<int8_t> q_int8;
  std::vector<float> q_scale;
  std::vector<int8_t> k_int8;
  std::vector<float> k_scale;
  std::vector<int8_t> v_int8;
  std::vector<float> v_scale;
  uint32_t q_scale_tiles = 0;
  uint32_t k_scale_tiles = 0;
};

struct QuantizationValidationStats {
  bool passed = false;
  double max_abs_q_scale = 0;
  double max_abs_k_scale = 0;
  double max_abs_v_scale = 0;
  size_t mismatched_q = 0;
  size_t mismatched_k = 0;
  size_t mismatched_v = 0;
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

constexpr uint32_t kDefaultQQuantizeThreads = 128;
constexpr uint32_t kDefaultKVQuantizeThreads = 256;

size_t q_index(
    const AttentionCase& attention,
    uint32_t batch,
    uint32_t row,
    uint32_t head,
    uint32_t dim)
{
  return (((size_t)batch * attention.R + row) * attention.Hq + head) *
      attention.D + dim;
}

size_t kv_index(
    const AttentionCase& attention,
    uint32_t batch,
    uint32_t column,
    uint32_t head,
    uint32_t dim)
{
  return (((size_t)batch * attention.C + column) * attention.Hk + head) *
      attention.D + dim;
}

size_t o_index(
    const AttentionCase& attention,
    uint32_t batch,
    uint32_t row,
    uint32_t head,
    uint32_t dim)
{
  return (((size_t)batch * attention.R + row) * attention.Hq + head) *
      attention.D + dim;
}

size_t l_index(
    const AttentionCase& attention,
    uint32_t batch,
    uint32_t head,
    uint32_t row)
{
  return ((size_t)batch * attention.Hq + head) * attention.R + row;
}

template <typename T>
std::vector<T> make_data(size_t size, float scale, int phase)
{
  std::vector<T> values(size);
  for (size_t i = 0; i < size; ++i) {
    const int centered = (int)((i * 17 + phase * 13) % 29) - 14;
    values[i] = static_cast<T>(centered * scale);
  }
  return values;
}

float create_scale(const AttentionCase& attention)
{
  return 1.0f / std::sqrt((float)attention.D);
}

const char* input_precision_name(InputPrecision precision)
{
  switch (precision) {
  case InputPrecision::fp16:
    return "fp16";
  case InputPrecision::bf16:
    return "bf16";
  case InputPrecision::fp32:
    return "fp32";
  }
  return "unknown";
}

InputPrecision parse_input_precision(const char* text)
{
  if (!std::strcmp(text, "fp16") || !std::strcmp(text, "16f"))
    return InputPrecision::fp16;
  if (!std::strcmp(text, "bf16") || !std::strcmp(text, "16bf"))
    return InputPrecision::bf16;
  if (!std::strcmp(text, "fp32") || !std::strcmp(text, "32f"))
    return InputPrecision::fp32;
  std::cerr << "unknown input precision: " << text << '\n';
  std::exit(1);
}

GEMMOperandPrecision create_io_precision(InputPrecision precision)
{
  switch (precision) {
  case InputPrecision::fp16:
    return GEMMOperandPrecision::FP16;
  case InputPrecision::bf16:
    return GEMMOperandPrecision::BF16;
  case InputPrecision::fp32:
    return GEMMOperandPrecision::FP32;
  }
  return GEMMOperandPrecision::FP16;
}

size_t input_precision_size(InputPrecision precision)
{
  return (size_t)create_io_precision(precision).size();
}

uint16_t float_to_bfloat_bits(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7fffu + lsb;
  return (uint16_t)(bits >> 16);
}

float bfloat_bits_to_float(uint16_t value)
{
  uint32_t bits = (uint32_t)value << 16;
  float output;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

std::vector<uint8_t> encode_values(
    const std::vector<float>& values,
    InputPrecision precision)
{
  std::vector<uint8_t> bytes(values.size() * input_precision_size(precision));
  if (precision == InputPrecision::fp16) {
    auto* dst = reinterpret_cast<half_float*>(bytes.data());
    for (size_t i = 0; i < values.size(); ++i)
      dst[i] = (half_float)values[i];
  } else if (precision == InputPrecision::bf16) {
    auto* dst = reinterpret_cast<uint16_t*>(bytes.data());
    for (size_t i = 0; i < values.size(); ++i)
      dst[i] = float_to_bfloat_bits(values[i]);
  } else {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

std::vector<float> decode_values(
    const void* raw,
    size_t count,
    InputPrecision precision)
{
  std::vector<float> values(count);
  if (precision == InputPrecision::fp16) {
    const auto* src = static_cast<const half_float*>(raw);
    for (size_t i = 0; i < count; ++i)
      values[i] = (float)src[i];
  } else if (precision == InputPrecision::bf16) {
    const auto* src = static_cast<const uint16_t*>(raw);
    for (size_t i = 0; i < count; ++i)
      values[i] = bfloat_bits_to_float(src[i]);
  } else {
    std::memcpy(values.data(), raw, count * sizeof(float));
  }
  return values;
}

const char* qk_precision_name(bool int8_qk)
{
  return int8_qk ? "int8" : "fp16";
}

bool use_qk_scales(const VariantConfig& variant)
{
  return variant.int8_qk_scales;
}

const char* qk_scale_mode_name(bool enabled)
{
  if (enabled)
    return "tile";
  return "none";
}

bool start_metal_capture(MTL::CommandQueue* command_queue, const char* path)
{
  auto descriptor = NS::TransferPtr(MTL::CaptureDescriptor::alloc()->init());
  descriptor->setCaptureObject(command_queue);
  if (path && path[0]) {
    auto string = NS::String::string(path, NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error = nil;
  const bool started = manager->startCapture(descriptor.get(), &error);
  if (!started) {
    std::cerr << "failed to start Metal capture: "
              << (error ? error->localizedDescription()->utf8String() : "unknown error")
              << '\n';
  }
  return started;
}

void stop_metal_capture()
{
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

simd::ushort3 create_baseline_block_dimensions(
    const AttentionCase& attention)
{
  const unsigned short head_dimension = attention.D;
  unsigned short revised_head = (head_dimension + 15) / 16 * 16;
  if (head_dimension <= 128) {
    revised_head = std::min<unsigned short>(head_dimension, revised_head);
  } else {
    revised_head =
        revised_head / std::max<unsigned short>(revised_head / 128, 2);
  }
  if (attention.C % 64 == 0) {
    return simd::ushort3 { 16, 64, revised_head };
  } else if (attention.C % 48 == 0) {
    return simd::ushort3 { 16, 48, revised_head };
  }
  if (attention.C % 128 > 64 && attention.C % 96 < 48) {
    return simd::ushort3 { 16, 64, revised_head };
  } else if (attention.C % 128 < 64 && attention.C % 96 > 48) {
    return simd::ushort3 { 16, 48, revised_head };
  }
  const unsigned short remainder64 = attention.C % 64;
  const unsigned short remainder48 = attention.C % 48;
  if (remainder64 * 48 < remainder48 * 64) {
    return simd::ushort3 { 16, 48, revised_head };
  } else {
    return simd::ushort3 { 16, 64, revised_head };
  }
}

simd::ushort3 create_int8_block_dimensions(
    const AttentionCase& attention,
    uint16_t block_r_override = 0,
    uint16_t block_c_override = 0,
    uint16_t block_d_override = 0)
{
  const uint16_t block_r = block_r_override ? block_r_override : 16;
  const uint16_t block_c = block_c_override ? block_c_override : 64;
  const uint16_t block_d = block_d_override ? block_d_override : (attention.D >= 192 ? 64 : 32);
  return simd::ushort3 { block_r, block_c, block_d };
}

uint16_t create_baseline_execution_simd_groups(uint16_t override_value)
{
  if (override_value != 0)
    return override_value;
  return 16;
}

uint16_t create_int8_execution_simd_groups(const AttentionCase& attention, uint16_t override_value)
{
  if (override_value != 0)
    return override_value;
  return attention.D > 192 ? 16 : 4;
}

bool create_check_c_edge_1(const AttentionCase& attention, simd::ushort3 block_dimensions)
{
  return (attention.C % (block_dimensions[1] * 2)) > block_dimensions[1];
}

QuantizePipelines create_quantize_pipelines(
    MTL::Device* device,
    const AttentionCase& attention,
    simd::ushort3 block_dimensions,
    InputPrecision input_precision,
    uint16_t q_quant_threads,
    uint16_t kv_quant_threads)
{
  QuantizePipelines bundle;
  bundle.q_threads = q_quant_threads != 0 ? q_quant_threads : kDefaultQQuantizeThreads;
  bundle.kv_threads = kv_quant_threads != 0 ? kv_quant_threads : kDefaultKVQuantizeThreads;
  const std::string source_type = create_io_precision(input_precision).name();
  const bool vectorize_quantize = (attention.D % 4) == 0;
  std::ostringstream source;
  source << R"(
#include <metal_stdlib>
using namespace metal;

constant uint R = )" << attention.R << R"(;
constant uint C = )" << attention.C << R"(;
constant uint Hq = )" << attention.Hq << R"(;
constant uint Hk = )" << attention.Hk << R"(;
constant uint D = )" << attention.D << R"(;
constant uint BLOCK_R = )" << block_dimensions[0] << R"(;
constant uint BLOCK_C = )" << block_dimensions[1] << R"(;
constant uint Q_TILES = ((R + BLOCK_R - 1) / BLOCK_R);
constant uint K_TILES = ((C + BLOCK_C - 1) / BLOCK_C);
constant uint Q_QUANT_THREADS = )" << bundle.q_threads << R"(;
constant uint KV_QUANT_THREADS = )" << bundle.kv_threads << R"(;
constant uint QUANTIZE_SIMD_LANES = 32;
constant uint Q_QUANTIZE_SIMDGROUPS = Q_QUANT_THREADS / QUANTIZE_SIMD_LANES;
constant uint KV_QUANTIZE_SIMDGROUPS = KV_QUANT_THREADS / QUANTIZE_SIMD_LANES;

inline float quantize_reduce_max_q(float value,
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
    value = lane_id < Q_QUANTIZE_SIMDGROUPS ? scratch[lane_id] : 0.0f;
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

inline float quantize_reduce_max_kv(float value,
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
    value = lane_id < KV_QUANTIZE_SIMDGROUPS ? scratch[lane_id] : 0.0f;
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
  if (vectorize_quantize) {
    source << R"(

using io_vec4 = vec<)" << source_type << R"(, 4>;

kernel void quantize_q(
    device const )" << source_type << R"(* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[Q_QUANTIZE_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint row_start = tile * BLOCK_R;
  const uint rows = min(BLOCK_R, R - row_start);
  const uint vectors_per_row = D / 4;
  const uint total_vectors = rows * vectors_per_row;
  device const io_vec4* src4 = reinterpret_cast<device const io_vec4*>(src);
  device char4* dst4 = reinterpret_cast<device char4*>(dst);
  float local_max = 0.0f;
  for (uint i = tid; i < total_vectors; i += Q_QUANT_THREADS) {
    const uint row = row_start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = (((batch * R + row) * Hq + head) * D + vec_dim * 4);
    const float4 value = float4(src4[index / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max_q(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[((batch * Hq + head) * Q_TILES) + tile] = scale;
  for (uint i = tid; i < total_vectors; i += Q_QUANT_THREADS) {
    const uint row = row_start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = (((batch * R + row) * Hq + head) * D + vec_dim * 4);
    const int4 rounded = int4(rint(float4(src4[index / 4]) * inv_scale));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}

kernel void quantize_k(
    device const )" << source_type << R"(* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[KV_QUANTIZE_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint col_start = tile * BLOCK_C;
  const uint cols = min(BLOCK_C, C - col_start);
  const uint vectors_per_row = D / 4;
  const uint total_vectors = cols * vectors_per_row;
  device const io_vec4* src4 = reinterpret_cast<device const io_vec4*>(src);
  device char4* dst4 = reinterpret_cast<device char4*>(dst);
  float local_max = 0.0f;
  for (uint i = tid; i < total_vectors; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = (((batch * C + column) * Hk + head) * D + vec_dim * 4);
    const float4 value = float4(src4[index / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max_kv(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[((batch * Hk + head) * K_TILES) + tile] = scale;
  for (uint i = tid; i < total_vectors; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = (((batch * C + column) * Hk + head) * D + vec_dim * 4);
    const int4 rounded = int4(rint(float4(src4[index / 4]) * inv_scale));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}

kernel void quantize_v(
    device const )" << source_type << R"(* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[KV_QUANTIZE_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint col_start = tile * BLOCK_C;
  const uint cols = min(BLOCK_C, C - col_start);
  const uint vectors_per_row = D / 4;
  const uint total_vectors = cols * vectors_per_row;
  device const io_vec4* src4 = reinterpret_cast<device const io_vec4*>(src);
  device char4* dst4 = reinterpret_cast<device char4*>(dst);
  float local_max = 0.0f;
  for (uint i = tid; i < total_vectors; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = (((batch * C + column) * Hk + head) * D + vec_dim * 4);
    const float4 value = float4(src4[index / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max_kv(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[((batch * Hk + head) * K_TILES) + tile] = scale;
  for (uint i = tid; i < total_vectors; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = (((batch * C + column) * Hk + head) * D + vec_dim * 4);
    const int4 rounded = int4(rint(float4(src4[index / 4]) * inv_scale));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}
)";
  } else {
    source << R"(

kernel void quantize_q(
    device const )" << source_type << R"(* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[Q_QUANTIZE_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint row_start = tile * BLOCK_R;
  const uint rows = min(BLOCK_R, R - row_start);
  const uint total = rows * D;
  float local_max = 0.0f;
  for (uint i = tid; i < total; i += Q_QUANT_THREADS) {
    const uint row = row_start + i / D;
    const uint dim = i % D;
    const uint index = (((batch * R + row) * Hq + head) * D + dim);
    local_max = max(local_max, fabs((float)src[index]));
  }
  const float max_abs = quantize_reduce_max_q(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[((batch * Hq + head) * Q_TILES) + tile] = scale;
  for (uint i = tid; i < total; i += Q_QUANT_THREADS) {
    const uint row = row_start + i / D;
    const uint dim = i % D;
    const uint index = (((batch * R + row) * Hq + head) * D + dim);
    const int rounded = (int)rint((float)src[index] * inv_scale);
    dst[index] = (int8_t)clamp(rounded, -127, 127);
  }
}

kernel void quantize_k(
    device const )" << source_type << R"(* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[KV_QUANTIZE_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint col_start = tile * BLOCK_C;
  const uint cols = min(BLOCK_C, C - col_start);
  const uint total = cols * D;
  float local_max = 0.0f;
  for (uint i = tid; i < total; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / D;
    const uint dim = i % D;
    const uint index = (((batch * C + column) * Hk + head) * D + dim);
    local_max = max(local_max, fabs((float)src[index]));
  }
  const float max_abs = quantize_reduce_max_kv(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[((batch * Hk + head) * K_TILES) + tile] = scale;
  for (uint i = tid; i < total; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / D;
    const uint dim = i % D;
    const uint index = (((batch * C + column) * Hk + head) * D + dim);
    const int rounded = (int)rint((float)src[index] * inv_scale);
    dst[index] = (int8_t)clamp(rounded, -127, 127);
  }
}

kernel void quantize_v(
    device const )" << source_type << R"(* src [[buffer(0)]],
    device int8_t* dst [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  threadgroup float scratch[KV_QUANTIZE_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint col_start = tile * BLOCK_C;
  const uint cols = min(BLOCK_C, C - col_start);
  const uint total = cols * D;
  float local_max = 0.0f;
  for (uint i = tid; i < total; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / D;
    const uint dim = i % D;
    const uint index = (((batch * C + column) * Hk + head) * D + dim);
    local_max = max(local_max, fabs((float)src[index]));
  }
  const float max_abs = quantize_reduce_max_kv(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[((batch * Hk + head) * K_TILES) + tile] = scale;
  for (uint i = tid; i < total; i += KV_QUANT_THREADS) {
    const uint column = col_start + i / D;
    const uint dim = i % D;
    const uint index = (((batch * C + column) * Hk + head) * D + dim);
    const int rounded = (int)rint((float)src[index] * inv_scale);
    dst[index] = (int8_t)clamp(rounded, -127, 127);
  }
}
)";
  }

  auto string = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto create_pipeline = [&](const char* name) {
    auto function_name = NS::String::string(name, NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(library->newFunction(function_name));
    auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    descriptor->setComputeFunction(function.get());
    return NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  };

  bundle.q_pipeline = create_pipeline("quantize_q");
  CCV_NNC_MFA_CHECK_ERROR(error);
  bundle.k_pipeline = create_pipeline("quantize_k");
  CCV_NNC_MFA_CHECK_ERROR(error);
  bundle.v_pipeline = create_pipeline("quantize_v");
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

AttentionOperands<GEMMOperandPrecision> create_memory_precisions(InputPrecision input_precision)
{
  AttentionOperands<GEMMOperandPrecision> memory_precisions;
  const auto io_precision = create_io_precision(input_precision);
  memory_precisions[AttentionOperand::Q] = io_precision;
  memory_precisions[AttentionOperand::K] = io_precision;
  memory_precisions[AttentionOperand::V] = io_precision;
  memory_precisions[AttentionOperand::O] = io_precision;
  memory_precisions[AttentionOperand::dO] = io_precision;
  memory_precisions[AttentionOperand::L] = input_precision == InputPrecision::fp32 ? GEMMOperandPrecision::FP32 : io_precision;
  memory_precisions[AttentionOperand::D] = input_precision == InputPrecision::fp32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::BF16;
  memory_precisions[AttentionOperand::dV] = GEMMOperandPrecision::FP32;
  memory_precisions[AttentionOperand::dK] = GEMMOperandPrecision::FP32;
  memory_precisions[AttentionOperand::dQ] = GEMMOperandPrecision::FP32;
  return memory_precisions;
}

void upload_buffer(
    MTL::CommandQueue* command_queue,
    MTL::Buffer* source,
    MTL::Buffer* destination,
    size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void download_buffer(
    MTL::CommandQueue* command_queue,
    MTL::Buffer* source,
    MTL::Buffer* destination,
    size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

double run_quantize_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    simd::ushort3 block_dimensions,
    const QuantizePipelines& pipelines,
    MTL::Buffer* q_buffer,
    MTL::Buffer* q_int8_buffer,
    MTL::Buffer* q_scale_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* k_int8_buffer,
    MTL::Buffer* k_scale_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* v_int8_buffer,
    MTL::Buffer* v_scale_buffer)
{
  const uint32_t q_tiles = (attention.R + block_dimensions[0] - 1) / block_dimensions[0];
  const uint32_t k_tiles = (attention.C + block_dimensions[1] - 1) / block_dimensions[1];

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.q_pipeline.get());
    encoder->setBuffer(q_buffer, 0, 0);
    encoder->setBuffer(q_int8_buffer, 0, 1);
    encoder->setBuffer(q_scale_buffer, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(q_tiles, attention.Hq, attention.batch),
        MTL::Size(pipelines.q_threads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.k_pipeline.get());
    encoder->setBuffer(k_buffer, 0, 0);
    encoder->setBuffer(k_int8_buffer, 0, 1);
    encoder->setBuffer(k_scale_buffer, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(k_tiles, attention.Hk, attention.batch),
        MTL::Size(pipelines.kv_threads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.v_pipeline.get());
    encoder->setBuffer(v_buffer, 0, 0);
    encoder->setBuffer(v_int8_buffer, 0, 1);
    encoder->setBuffer(v_scale_buffer, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(k_tiles, attention.Hk, attention.batch),
        MTL::Size(pipelines.kv_threads, 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_quantize_stage_once(
    MTL::CommandQueue* command_queue,
    MTL::ComputePipelineState* pipeline,
    uint16_t threads,
    MTL::Buffer* source_buffer,
    MTL::Buffer* int8_buffer,
    MTL::Buffer* scale_buffer,
    uint32_t tiles,
    uint32_t heads,
    uint32_t batch)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(source_buffer, 0, 0);
  encoder->setBuffer(int8_buffer, 0, 1);
  encoder->setBuffer(scale_buffer, 0, 2);
  encoder->dispatchThreadgroups(
      MTL::Size(tiles, heads, batch),
      MTL::Size(threads, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_quantize_and_int8_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    simd::ushort3 block_dimensions,
    const QuantizePipelines& quantize_pipelines,
    const Int8Pipeline& bundle,
    MTL::Buffer* q_buffer,
    MTL::Buffer* q_int8_buffer,
    MTL::Buffer* q_scale_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* k_int8_buffer,
    MTL::Buffer* k_scale_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* v_int8_buffer,
    MTL::Buffer* v_scale_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  const uint32_t q_tiles = (attention.R + block_dimensions[0] - 1) / block_dimensions[0];
  const uint32_t k_tiles = (attention.C + block_dimensions[1] - 1) / block_dimensions[1];

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(quantize_pipelines.q_pipeline.get());
    encoder->setBuffer(q_buffer, 0, 0);
    encoder->setBuffer(q_int8_buffer, 0, 1);
    encoder->setBuffer(q_scale_buffer, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(q_tiles, attention.Hq, attention.batch),
        MTL::Size(quantize_pipelines.q_threads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(quantize_pipelines.k_pipeline.get());
    encoder->setBuffer(k_buffer, 0, 0);
    encoder->setBuffer(k_int8_buffer, 0, 1);
    encoder->setBuffer(k_scale_buffer, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(k_tiles, attention.Hk, attention.batch),
        MTL::Size(quantize_pipelines.kv_threads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(quantize_pipelines.v_pipeline.get());
    encoder->setBuffer(v_buffer, 0, 0);
    encoder->setBuffer(v_int8_buffer, 0, 1);
    encoder->setBuffer(v_scale_buffer, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(k_tiles, attention.Hk, attention.batch),
        MTL::Size(quantize_pipelines.kv_threads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(bundle.pipeline.get());
    const uint32_t threadgroup_memory_length = bundle.kernel->threadgroupMemoryAllocation();
    encoder->setThreadgroupMemoryLength(threadgroup_memory_length, 0);
    encoder->setBuffer(q_int8_buffer, 0, 0);
    encoder->setBuffer(k_int8_buffer, 0, 1);
    encoder->setBuffer(v_int8_buffer, 0, 2);
    encoder->setBuffer(o_buffer, 0, 3);
    encoder->setBuffer(l_buffer, 0, 4);
    encoder->setBuffer(q_scale_buffer, 0, 5);
    encoder->setBuffer(k_scale_buffer, 0, 6);
    encoder->setBuffer(v_scale_buffer, 0, 7);
    encoder->dispatchThreadgroups(
        bundle.kernel->threadgroupsPerGrid(attention.batch, attention.R),
        MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

BaselinePipeline create_baseline_pipeline(
    MTL::Device* device,
    const AttentionCase& attention,
    const VariantConfig& variant)
{
  BaselinePipeline bundle;
  bundle.descriptor.batchDimension = attention.batch;
  bundle.descriptor.Hq = attention.Hq;
  bundle.descriptor.Hk = attention.Hk;
  bundle.descriptor.lowPrecisionInputs = variant.input_precision != InputPrecision::fp32;
  bundle.descriptor.isBF16 = variant.input_precision == InputPrecision::bf16;
  bundle.descriptor.lowPrecisionIntermediates = variant.input_precision != InputPrecision::fp32;
  bundle.descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  bundle.descriptor.type = AttentionKernelType::forward;
  bundle.descriptor.scale = create_scale(attention);
  if (attention.batch > 1) {
    bundle.descriptor.batchStrides[AttentionOperand::Q] = attention.R * attention.D * attention.Hq;
    bundle.descriptor.batchStrides[AttentionOperand::K] = attention.C * attention.D * attention.Hk;
    bundle.descriptor.batchStrides[AttentionOperand::V] = attention.C * attention.D * attention.Hk;
    bundle.descriptor.batchStrides[AttentionOperand::O] = attention.R * attention.D * attention.Hq;
  }
  bundle.memory_precisions = create_memory_precisions(variant.input_precision);
  const simd::ushort3 block_dimensions = create_baseline_block_dimensions(attention);
  const uint16_t execution_simd_groups =
      create_baseline_execution_simd_groups(variant.baseline_execution_simd_groups_override);
  const bool check_c_edge_1 = create_check_c_edge_1(attention, block_dimensions);
  const NAAttentionKernelDescriptor kernel_descriptor(
      block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      execution_simd_groups,
      check_c_edge_1,
      bundle.memory_precisions,
      AttentionKernelType::forward,
      bundle.descriptor.scale);
  bundle.kernel = std::make_unique<NAAttentionKernel>(kernel_descriptor, device);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t row_dimension = attention.R;
  const uint32_t column_dimension = attention.C;
  constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, NS::UInteger(1));
  const uint32_t q_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t k_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t v_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t o_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  constants->setConstantValue(&q_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::Q).bufferIndex()));
  constants->setConstantValue(&k_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::K).bufferIndex()));
  constants->setConstantValue(&v_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::V).bufferIndex()));
  constants->setConstantValue(&o_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::O).bufferIndex()));

  NS::Error* error = nil;
  auto attention_name = NS::String::string("attention", NS::UTF8StringEncoding);
  auto attention_function = NS::TransferPtr(bundle.kernel->library->newFunction(attention_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipeline_descriptor->setComputeFunction(attention_function.get());
  bundle.pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

Int8Pipeline create_int8_pipeline(
    MTL::Device* device,
    const AttentionCase& attention,
    const VariantConfig& variant)
{
  Int8Pipeline bundle;
  bundle.block_dimensions = create_int8_block_dimensions(
      attention,
      variant.int8_block_r_override,
      variant.int8_block_c_override,
      variant.int8_block_d_override);
  bundle.execution_simd_groups =
      create_int8_execution_simd_groups(attention, variant.int8_execution_simd_groups_override);
  bundle.use_int8_qk = variant.int8_qk;
  bundle.use_qk_scales = use_qk_scales(variant);
  bundle.thread_barrier_over_c = variant.int8_thread_barrier_over_c;
  CCV_NNC_MFA_PRECONDITION(bundle.use_int8_qk);
  const bool check_c_edge_1 = create_check_c_edge_1(attention, bundle.block_dimensions);
  const NAInt8AttentionKernelDescriptor kernel_descriptor(
      bundle.block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      bundle.execution_simd_groups,
      check_c_edge_1,
      bundle.use_int8_qk,
      bundle.use_qk_scales,
      bundle.thread_barrier_over_c,
      create_io_precision(variant.input_precision),
      create_scale(attention));
  bundle.kernel = std::make_unique<NAInt8AttentionKernel>(kernel_descriptor, device);

  const uint32_t q_tiles = (attention.R + bundle.block_dimensions[0] - 1) / bundle.block_dimensions[0];
  const uint32_t k_tiles = (attention.C + bundle.block_dimensions[1] - 1) / bundle.block_dimensions[1];
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t row_dimension = attention.R;
  const uint32_t column_dimension = attention.C;
  constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, NS::UInteger(1));
  const uint32_t q_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t k_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t v_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t o_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t k_scale_batch_stride = attention.batch > 1 ? attention.Hk * k_tiles : 0;
  const uint32_t q_scale_batch_stride = attention.batch > 1 ? attention.Hq * q_tiles : 0;
  const uint32_t v_scale_batch_stride = attention.batch > 1 ? attention.Hk * k_tiles : 0;
  constants->setConstantValue(&q_batch_stride, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&k_batch_stride, MTL::DataTypeUInt, NS::UInteger(3));
  constants->setConstantValue(&v_batch_stride, MTL::DataTypeUInt, NS::UInteger(4));
  constants->setConstantValue(&o_batch_stride, MTL::DataTypeUInt, NS::UInteger(5));
  constants->setConstantValue(&q_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(6));
  constants->setConstantValue(&k_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(7));
  constants->setConstantValue(&v_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(8));

  NS::Error* error = nil;
  auto kernel_name = NS::String::string("int8_attention", NS::UTF8StringEncoding);
  auto kernel_function = NS::TransferPtr(bundle.kernel->library->newFunction(kernel_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipeline_descriptor->setComputeFunction(kernel_function.get());
  bundle.pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

QuantizedQK quantize_qk(
    const AttentionCase& attention,
    simd::ushort3 block_dimensions,
    const std::vector<float>& q_values,
    const std::vector<float>& k_values,
    const std::vector<float>& v_values)
{
  QuantizedQK quantized;
  quantized.q_int8.resize(q_values.size());
  quantized.k_int8.resize(k_values.size());
  quantized.v_int8.resize(v_values.size());
  const uint32_t q_tiles = (attention.R + block_dimensions[0] - 1) / block_dimensions[0];
  const uint32_t k_tiles = (attention.C + block_dimensions[1] - 1) / block_dimensions[1];
  quantized.q_scale_tiles = q_tiles;
  quantized.k_scale_tiles = k_tiles;
  quantized.q_scale.resize((size_t)attention.batch * attention.Hq * q_tiles);
  quantized.k_scale.resize((size_t)attention.batch * attention.Hk * k_tiles);
  quantized.v_scale.resize((size_t)attention.batch * attention.Hk * k_tiles);

  for (uint32_t batch = 0; batch < attention.batch; ++batch) {
    for (uint32_t head = 0; head < attention.Hq; ++head) {
      for (uint32_t tile = 0; tile < q_tiles; ++tile) {
        const uint32_t row_start = tile * block_dimensions[0];
        const uint32_t row_end = std::min<uint32_t>(row_start + block_dimensions[0], attention.R);
        float max_abs = 0;
        for (uint32_t row = row_start; row < row_end; ++row) {
          for (uint32_t dim = 0; dim < attention.D; ++dim) {
            max_abs = std::max(max_abs, std::fabs(q_values[q_index(attention, batch, row, head, dim)]));
          }
        }
        const float scale = max_abs > 0 ? max_abs / 127.0f : 1.0f / 127.0f;
        quantized.q_scale[((size_t)batch * attention.Hq + head) * q_tiles + tile] = scale;
        for (uint32_t row = row_start; row < row_end; ++row) {
          for (uint32_t dim = 0; dim < attention.D; ++dim) {
            const float value = q_values[q_index(attention, batch, row, head, dim)] / scale;
            const int rounded = (int)std::lrint(value);
            quantized.q_int8[q_index(attention, batch, row, head, dim)] =
                (int8_t)std::max(-127, std::min(127, rounded));
          }
        }
      }
    }
  }

  for (uint32_t batch = 0; batch < attention.batch; ++batch) {
    for (uint32_t head = 0; head < attention.Hk; ++head) {
      for (uint32_t tile = 0; tile < k_tiles; ++tile) {
        const uint32_t col_start = tile * block_dimensions[1];
        const uint32_t col_end = std::min<uint32_t>(col_start + block_dimensions[1], attention.C);
        float max_abs = 0;
        for (uint32_t column = col_start; column < col_end; ++column) {
          for (uint32_t dim = 0; dim < attention.D; ++dim) {
            max_abs = std::max(max_abs, std::fabs(k_values[kv_index(attention, batch, column, head, dim)]));
          }
        }
        const float scale = max_abs > 0 ? max_abs / 127.0f : 1.0f / 127.0f;
        quantized.k_scale[((size_t)batch * attention.Hk + head) * k_tiles + tile] = scale;
        for (uint32_t column = col_start; column < col_end; ++column) {
          for (uint32_t dim = 0; dim < attention.D; ++dim) {
            const float value = k_values[kv_index(attention, batch, column, head, dim)] / scale;
            const int rounded = (int)std::lrint(value);
            quantized.k_int8[kv_index(attention, batch, column, head, dim)] = (int8_t)std::max(-127, std::min(127, rounded));
          }
        }
      }
    }
  }
  for (uint32_t batch = 0; batch < attention.batch; ++batch) {
    for (uint32_t head = 0; head < attention.Hk; ++head) {
      for (uint32_t tile = 0; tile < k_tiles; ++tile) {
        const uint32_t col_start = tile * block_dimensions[1];
        const uint32_t col_end = std::min<uint32_t>(col_start + block_dimensions[1], attention.C);
        float max_abs = 0;
        for (uint32_t column = col_start; column < col_end; ++column) {
          for (uint32_t dim = 0; dim < attention.D; ++dim) {
            max_abs = std::max(max_abs, std::fabs(v_values[kv_index(attention, batch, column, head, dim)]));
          }
        }
        const float scale = max_abs > 0 ? max_abs / 127.0f : 1.0f / 127.0f;
        quantized.v_scale[((size_t)batch * attention.Hk + head) * k_tiles + tile] = scale;
        for (uint32_t column = col_start; column < col_end; ++column) {
          for (uint32_t dim = 0; dim < attention.D; ++dim) {
            const float value = v_values[kv_index(attention, batch, column, head, dim)] / scale;
            const int rounded = (int)std::lrint(value);
            quantized.v_int8[kv_index(attention, batch, column, head, dim)] = (int8_t)std::max(-127, std::min(127, rounded));
          }
        }
      }
    }
  }
  return quantized;
}

double run_baseline_once(
    MTL::CommandQueue* command_queue,
    const BaselinePipeline& bundle,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setThreadgroupMemoryLength(bundle.kernel->threadgroupMemoryAllocation(bundle.pipeline.get(), bundle.descriptor), 0);
  encoder->setBuffer(q_buffer, 0, 0);
  encoder->setBuffer(k_buffer, 0, 1);
  encoder->setBuffer(v_buffer, 0, 2);
  encoder->setBuffer(o_buffer, 0, 3);
  encoder->setBuffer(l_buffer, 0, 4);
  encoder->dispatchThreadgroups(
      bundle.kernel->threadgroupsPerGrid(bundle.descriptor),
      MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get(), bundle.descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_int8_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const Int8Pipeline& bundle,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* q_scale_buffer,
    MTL::Buffer* k_scale_buffer,
    MTL::Buffer* v_scale_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  const uint32_t threadgroup_memory_length = bundle.kernel->threadgroupMemoryAllocation();
  encoder->setThreadgroupMemoryLength(threadgroup_memory_length, 0);
  encoder->setBuffer(q_buffer, 0, 0);
  encoder->setBuffer(k_buffer, 0, 1);
  encoder->setBuffer(v_buffer, 0, 2);
  encoder->setBuffer(o_buffer, 0, 3);
  encoder->setBuffer(l_buffer, 0, 4);
  encoder->setBuffer(q_scale_buffer, 0, 5);
  encoder->setBuffer(k_scale_buffer, 0, 6);
  encoder->setBuffer(v_scale_buffer, 0, 7);
  encoder->dispatchThreadgroups(
      bundle.kernel->threadgroupsPerGrid(attention.batch, attention.R),
      MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get()), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

template <typename RunOnce>
bool benchmark(
    const BenchmarkConfig& config,
    RunOnce&& run_once,
    Stats* stats)
{
  std::vector<double> samples;
  samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const double seconds = run_once();
    if (!(seconds > 0) || std::isnan(seconds))
      return false;
    if (i >= config.warmup_iterations)
      samples.push_back(seconds);
  }
  stats->average_seconds = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

QuantizationValidationStats validate_quantization(
    const QuantizedQK& reference,
    const std::vector<int8_t>& q_int8,
    const std::vector<float>& q_scale,
    const std::vector<int8_t>& k_int8,
    const std::vector<float>& k_scale,
    const std::vector<float>& v_scale,
    const std::vector<int8_t>& v_int8)
{
  QuantizationValidationStats stats;
  for (size_t i = 0; i < q_scale.size(); ++i)
    stats.max_abs_q_scale = std::max(stats.max_abs_q_scale, std::fabs((double)q_scale[i] - reference.q_scale[i]));
  for (size_t i = 0; i < k_scale.size(); ++i)
    stats.max_abs_k_scale = std::max(stats.max_abs_k_scale, std::fabs((double)k_scale[i] - reference.k_scale[i]));
  for (size_t i = 0; i < v_scale.size(); ++i)
    stats.max_abs_v_scale = std::max(stats.max_abs_v_scale, std::fabs((double)v_scale[i] - reference.v_scale[i]));
  for (size_t i = 0; i < q_int8.size(); ++i)
    stats.mismatched_q += (q_int8[i] != reference.q_int8[i]);
  for (size_t i = 0; i < k_int8.size(); ++i)
    stats.mismatched_k += (k_int8[i] != reference.k_int8[i]);
  for (size_t i = 0; i < v_int8.size(); ++i)
    stats.mismatched_v += (v_int8[i] != reference.v_int8[i]);
  stats.passed = stats.max_abs_q_scale == 0 && stats.max_abs_k_scale == 0 &&
      stats.max_abs_v_scale == 0 &&
      stats.mismatched_q == 0 && stats.mismatched_k == 0 && stats.mismatched_v == 0;
  return stats;
}

void compute_int8_reference_row(
    const AttentionCase& attention,
    simd::ushort3 block_dimensions,
    const QuantizedQK& quantized,
    const std::vector<float>& v_values,
    bool use_qk_scales,
    uint32_t batch,
    uint32_t query_head,
    uint32_t row,
    float* l_value,
    std::vector<float>* o_values)
{
  const uint32_t head_ratio = attention.Hq / attention.Hk;
  const uint32_t kv_head = query_head / head_ratio;
  const uint32_t q_tiles = quantized.q_scale_tiles;
  const uint32_t k_tiles = quantized.k_scale_tiles;
  const uint32_t q_tile = q_tiles == 1 ? 0 : (row / block_dimensions[0]);
  const float q_scale = use_qk_scales ? quantized.q_scale[((size_t)batch * attention.Hq + query_head) * q_tiles + q_tile] : 1.0f;
  std::vector<float> scores(attention.C);
  float max_score = -std::numeric_limits<float>::infinity();
  for (uint32_t column = 0; column < attention.C; ++column) {
    const uint32_t k_tile = k_tiles == 1 ? 0 : (column / block_dimensions[1]);
    const float k_scale = use_qk_scales ? quantized.k_scale[((size_t)batch * attention.Hk + kv_head) * k_tiles + k_tile] : 1.0f;
    float dot = 0;
    for (uint32_t dim = 0; dim < attention.D; ++dim) {
      dot += ((float)quantized.q_int8[q_index(attention, batch, row, query_head, dim)] * q_scale) *
          ((float)quantized.k_int8[kv_index(attention, batch, column, kv_head, dim)] * k_scale);
    }
    const float score = dot * create_scale(attention);
    scores[column] = score;
    max_score = std::max(max_score, score);
  }
  float sum = 0;
  for (uint32_t column = 0; column < attention.C; ++column) {
    scores[column] = std::exp(scores[column] - max_score);
    sum += scores[column];
  }
  *l_value = (max_score + std::log(sum)) * 1.442695041f;
  const float reciprocal = 1.0f / sum;
  o_values->assign(attention.D, 0.0f);
  for (uint32_t column = 0; column < attention.C; ++column) {
    const float probability = scores[column] * reciprocal;
    for (uint32_t dim = 0; dim < attention.D; ++dim) {
      (*o_values)[dim] += probability * v_values[kv_index(attention, batch, column, kv_head, dim)];
    }
  }
}

std::vector<uint32_t> make_sample_points(uint32_t dimension, const std::vector<uint32_t>& suggestions)
{
  std::vector<uint32_t> sample_points;
  for (const auto suggestion : suggestions)
    if (suggestion < dimension)
      sample_points.push_back(suggestion);
  if (dimension > 0) {
    sample_points.push_back(dimension / 2);
    sample_points.push_back(dimension - 1);
  }
  std::sort(sample_points.begin(), sample_points.end());
  sample_points.erase(std::unique(sample_points.begin(), sample_points.end()), sample_points.end());
  return sample_points;
}

ValidationStats validate_int8_outputs(
    const AttentionCase& attention,
    simd::ushort3 block_dimensions,
    const QuantizedQK& quantized,
    const std::vector<float>& v_values,
    bool use_qk_scales,
    const std::vector<float>& o_values,
    const float* l_raw)
{
  ValidationStats stats;
  const uint64_t reference_work = (uint64_t)attention.batch * attention.Hq * attention.R * attention.C * attention.D;
  stats.full_reference = reference_work <= (1ull << 24);
  std::vector<uint32_t> batch_points;
  std::vector<uint32_t> head_points;
  std::vector<uint32_t> row_points;
  if (stats.full_reference) {
    batch_points.resize(attention.batch);
    std::iota(batch_points.begin(), batch_points.end(), 0);
    head_points.resize(attention.Hq);
    std::iota(head_points.begin(), head_points.end(), 0);
    row_points.resize(attention.R);
    std::iota(row_points.begin(), row_points.end(), 0);
  } else {
    batch_points = make_sample_points(attention.batch, { 0 });
    head_points = make_sample_points(attention.Hq, { 0, 1, 7, 15, 31 });
    row_points = make_sample_points(attention.R, { 0, 1, 15, 16, 255, 256, 4095, 4096 });
  }
  std::vector<float> reference_o;
  for (const auto batch : batch_points) {
    for (const auto head : head_points) {
      for (const auto row : row_points) {
        float reference_l = 0;
        compute_int8_reference_row(attention, block_dimensions, quantized, v_values, use_qk_scales, batch, head, row, &reference_l, &reference_o);
        const float actual_l = l_raw[l_index(attention, batch, head, row)];
        const double abs_l = std::fabs(reference_l - actual_l);
        const double rel_l = abs_l / std::max<double>(std::max(std::fabs(reference_l), std::fabs(actual_l)), 1.0);
        stats.max_abs_l = std::max(stats.max_abs_l, abs_l);
        stats.max_rel_l = std::max(stats.max_rel_l, rel_l);
        for (uint32_t dim = 0; dim < attention.D; ++dim) {
          const float actual_o = o_values[o_index(attention, batch, row, head, dim)];
          const double abs_o = std::fabs(reference_o[dim] - actual_o);
          const double rel_o = abs_o / std::max<double>(std::max(std::fabs(reference_o[dim]), std::fabs(actual_o)), 1.0);
          stats.max_abs_o = std::max(stats.max_abs_o, abs_o);
          stats.max_rel_o = std::max(stats.max_rel_o, rel_o);
        }
      }
    }
  }
  stats.checked_batches = batch_points.size();
  stats.checked_heads = head_points.size();
  stats.checked_rows = row_points.size();
  stats.passed = (stats.max_abs_o <= 7e-2 || stats.max_rel_o <= 7e-2) &&
      (stats.max_abs_l <= 7e-2 || stats.max_rel_l <= 7e-2);
  return stats;
}

double benchmark_flops(const AttentionCase& attention)
{
  const double qk_flops =
      2.0 * (double)attention.batch * attention.Hq * attention.R * attention.C * attention.D;
  return qk_flops * 2.0;
}

void print_stats(const char* label, const AttentionCase& attention, const Stats& stats)
{
  const double flops = benchmark_flops(attention);
  std::cout << std::fixed;
  std::cout << label
            << " avg_ms=" << std::setprecision(3) << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " avg_gflops=" << flops / stats.average_seconds / 1e9
            << '\n';
}

void print_time_stats(const char* label, const Stats& stats)
{
  std::cout << std::fixed;
  std::cout << label
            << " avg_ms=" << std::setprecision(3) << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << '\n';
}

void print_validation(const ValidationStats& stats)
{
  std::cout << "validation"
            << " mode=" << (stats.full_reference ? "full" : "sampled")
            << " batches=" << stats.checked_batches
            << " heads=" << stats.checked_heads
            << " rows=" << stats.checked_rows
            << " max_abs_o=" << stats.max_abs_o
            << " max_rel_o=" << stats.max_rel_o
            << " max_abs_l=" << stats.max_abs_l
            << " max_rel_l=" << stats.max_rel_l
            << '\n';
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  AttentionCase attention;
  BenchmarkConfig config;
  VariantConfig variant;
  if (argc >= 4) {
    attention.R = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    attention.C = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    attention.D = (uint32_t)std::strtoul(argv[3], nullptr, 10);
  }
  if (argc >= 7) {
    attention.batch = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    attention.Hq = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    attention.Hk = (uint32_t)std::strtoul(argv[6], nullptr, 10);
  }
  if (argc >= 9) {
    config.warmup_iterations = std::atoi(argv[7]);
    config.timed_iterations = std::atoi(argv[8]);
  }
  if (argc >= 10) {
    variant.baseline_execution_simd_groups_override =
        (uint16_t)std::strtoul(argv[9], nullptr, 10);
  }
  if (argc >= 11) {
    variant.int8_execution_simd_groups_override =
        (uint16_t)std::strtoul(argv[10], nullptr, 10);
  }
  if (argc >= 12) {
    variant.int8_block_c_override =
        (uint16_t)std::strtoul(argv[11], nullptr, 10);
  }
  if (argc >= 13) {
    variant.int8_block_r_override =
        (uint16_t)std::strtoul(argv[12], nullptr, 10);
  }
  if (argc >= 14) {
    variant.int8_block_d_override =
        (uint16_t)std::strtoul(argv[13], nullptr, 10);
  }
  if (argc >= 15) {
    if (!std::strcmp(argv[14], "quantize-only") || !std::strcmp(argv[14], "quantize_only"))
      variant.quantize_only = true;
    // Retired stripped-mode argument slot. Intentionally accepted and ignored
    // to keep older command lines usable while the production kernel stays
    // full-only.
  }
  if (argc >= 16) {
    variant.int8_qk = std::strcmp(argv[15], "fp16") != 0;
  }
  if (argc >= 17) {
    variant.int8_qk_scales = std::strtoul(argv[16], nullptr, 10) != 0;
  }
  if (argc >= 18) {
    // Retired bench-only cooperative/two-pass argument slot. Intentionally
    // accepted and ignored to keep older command lines usable.
  }
  if (argc >= 19) {
    // Retired bench-only V-precision argument slot. Intentionally accepted
    // and ignored to keep older command lines usable.
  }
  if (argc >= 20) {
    variant.capture_path = argv[19];
  }
  if (argc >= 21) {
    variant.int8_thread_barrier_over_c =
        std::strtoul(argv[20], nullptr, 10) != 0;
  }
  if (argc >= 22) {
    // Retired Morton-order override slot. Intentionally accepted and ignored
    // because Morton order is always on in the production kernel.
  }
  if (argc >= 23) {
    variant.input_precision = parse_input_precision(argv[22]);
  }
  if (argc >= 24) {
    variant.q_quant_threads_override =
        (uint16_t)std::strtoul(argv[23], nullptr, 10);
  }
  if (argc >= 25) {
    variant.kv_quant_threads_override =
        (uint16_t)std::strtoul(argv[24], nullptr, 10);
  }

  if (!variant.int8_qk) {
    std::cerr << "fp16 QK is not implemented in this scaffold; use the baseline path\n";
    return 1;
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Metal command queue unavailable.\n";
    pool->drain();
    return 1;
  }

  const auto q_values = make_data<float>(
      (size_t)attention.batch * attention.R * attention.Hq * attention.D,
      (!use_qk_scales(variant) && variant.int8_qk) ? 1.0f : 0.03125f,
      1);
  const auto k_values = make_data<float>(
      (size_t)attention.batch * attention.C * attention.Hk * attention.D,
      (!use_qk_scales(variant) && variant.int8_qk) ? 1.0f : 0.02734375f,
      2);
  const auto v_values = make_data<float>(
      (size_t)attention.batch * attention.C * attention.Hk * attention.D,
      0.0234375f,
      3);
  const auto q_encoded = encode_values(q_values, variant.input_precision);
  const auto k_encoded = encode_values(k_values, variant.input_precision);
  const auto v_encoded = encode_values(v_values, variant.input_precision);

  const simd::ushort3 block_dimensions = create_int8_block_dimensions(
      attention,
      variant.int8_block_r_override,
      variant.int8_block_c_override,
      variant.int8_block_d_override);
  BaselinePipeline baseline;
  Int8Pipeline int8_pipeline;
  if (!variant.quantize_only) {
    baseline = create_baseline_pipeline(device.get(), attention, variant);
    int8_pipeline = create_int8_pipeline(device.get(), attention, variant);
  } else {
    int8_pipeline.block_dimensions = block_dimensions;
    int8_pipeline.execution_simd_groups =
        create_int8_execution_simd_groups(attention, variant.int8_execution_simd_groups_override);
    int8_pipeline.use_int8_qk = variant.int8_qk;
    int8_pipeline.use_qk_scales = use_qk_scales(variant);
    int8_pipeline.thread_barrier_over_c = variant.int8_thread_barrier_over_c;
  }
  auto quantize_pipelines = create_quantize_pipelines(
      device.get(),
      attention,
      block_dimensions,
      variant.input_precision,
      variant.q_quant_threads_override,
      variant.kv_quant_threads_override);
  auto quantized = quantize_qk(attention, block_dimensions, q_values, k_values, v_values);

  const size_t q_bytes = q_encoded.size();
  const size_t q_int8_bytes = quantized.q_int8.size() * sizeof(int8_t);
  const size_t q_scale_bytes = quantized.q_scale.size() * sizeof(float);
  const size_t k_bytes = k_encoded.size();
  const size_t v_bytes = v_encoded.size();
  const size_t k_int8_bytes = quantized.k_int8.size() * sizeof(int8_t);
  const size_t k_scale_bytes = quantized.k_scale.size() * sizeof(float);
  const size_t v_int8_bytes = quantized.v_int8.size() * sizeof(int8_t);
  const size_t v_scale_bytes = quantized.v_scale.size() * sizeof(float);
  const size_t o_count = (size_t)attention.batch * attention.R * attention.Hq * attention.D;
  const size_t o_bytes = o_count * input_precision_size(variant.input_precision);
  const size_t l_bytes = (size_t)attention.batch * attention.Hq * attention.R * sizeof(float);
  auto q_stage = NS::TransferPtr(device->newBuffer(q_encoded.data(), q_bytes, kSharedResourceOptions));
  auto q_int8_stage = NS::TransferPtr(device->newBuffer(quantized.q_int8.data(), q_int8_bytes, kSharedResourceOptions));
  auto q_scale_stage = NS::TransferPtr(device->newBuffer(quantized.q_scale.data(), q_scale_bytes, kSharedResourceOptions));
  auto k_stage = NS::TransferPtr(device->newBuffer(k_encoded.data(), k_bytes, kSharedResourceOptions));
  auto v_stage = NS::TransferPtr(device->newBuffer(v_encoded.data(), v_bytes, kSharedResourceOptions));
  auto v_int8_stage = NS::TransferPtr(device->newBuffer(quantized.v_int8.data(), v_int8_bytes, kSharedResourceOptions));
  auto v_scale_stage = NS::TransferPtr(device->newBuffer(quantized.v_scale.data(), v_scale_bytes, kSharedResourceOptions));
  auto k_int8_stage = NS::TransferPtr(device->newBuffer(quantized.k_int8.data(), k_int8_bytes, kSharedResourceOptions));
  auto k_scale_stage = NS::TransferPtr(device->newBuffer(quantized.k_scale.data(), k_scale_bytes, kSharedResourceOptions));
  auto o_stage = NS::TransferPtr(device->newBuffer(o_bytes, kSharedResourceOptions));
  auto l_stage = NS::TransferPtr(device->newBuffer(l_bytes, kSharedResourceOptions));
  auto q_buffer = NS::TransferPtr(device->newBuffer(q_bytes, kPrivateResourceOptions));
  auto q_int8_buffer = NS::TransferPtr(device->newBuffer(q_int8_bytes, kPrivateResourceOptions));
  auto q_scale_buffer = NS::TransferPtr(device->newBuffer(q_scale_bytes, kPrivateResourceOptions));
  auto k_buffer = NS::TransferPtr(device->newBuffer(k_bytes, kPrivateResourceOptions));
  auto v_buffer = NS::TransferPtr(device->newBuffer(v_bytes, kPrivateResourceOptions));
  auto v_int8_buffer = NS::TransferPtr(device->newBuffer(v_int8_bytes, kPrivateResourceOptions));
  auto v_scale_buffer = NS::TransferPtr(device->newBuffer(v_scale_bytes, kPrivateResourceOptions));
  auto k_int8_buffer = NS::TransferPtr(device->newBuffer(k_int8_bytes, kPrivateResourceOptions));
  auto k_scale_buffer = NS::TransferPtr(device->newBuffer(k_scale_bytes, kPrivateResourceOptions));
  auto o_buffer = NS::TransferPtr(device->newBuffer(o_bytes, kPrivateResourceOptions));
  auto l_buffer = NS::TransferPtr(device->newBuffer(l_bytes, kPrivateResourceOptions));
  const bool use_int8_qk = int8_pipeline.use_int8_qk;
  auto* qk_q_buffer = use_int8_qk ? q_int8_buffer.get() : q_buffer.get();
  auto* qk_k_buffer = use_int8_qk ? k_int8_buffer.get() : k_buffer.get();
  auto* pv_v_buffer = v_int8_buffer.get();

  upload_buffer(command_queue.get(), q_stage.get(), q_buffer.get(), q_bytes);
  upload_buffer(command_queue.get(), q_int8_stage.get(), q_int8_buffer.get(), q_int8_bytes);
  upload_buffer(command_queue.get(), q_scale_stage.get(), q_scale_buffer.get(), q_scale_bytes);
  upload_buffer(command_queue.get(), k_stage.get(), k_buffer.get(), k_bytes);
  upload_buffer(command_queue.get(), v_stage.get(), v_buffer.get(), v_bytes);
  upload_buffer(command_queue.get(), v_int8_stage.get(), v_int8_buffer.get(), v_int8_bytes);
  upload_buffer(command_queue.get(), v_scale_stage.get(), v_scale_buffer.get(), v_scale_bytes);
  upload_buffer(command_queue.get(), k_int8_stage.get(), k_int8_buffer.get(), k_int8_bytes);
  upload_buffer(command_queue.get(), k_scale_stage.get(), k_scale_buffer.get(), k_scale_bytes);

  const double quantize_validation_seconds = run_quantize_once(
      command_queue.get(),
      attention,
      block_dimensions,
      quantize_pipelines,
      q_buffer.get(),
      q_int8_buffer.get(),
      q_scale_buffer.get(),
      k_buffer.get(),
      k_int8_buffer.get(),
      k_scale_buffer.get(),
      v_buffer.get(),
      v_int8_buffer.get(),
      v_scale_buffer.get());
  if (!(quantize_validation_seconds > 0)) {
    std::cerr << "quantize validation dispatch failed\n";
    pool->drain();
    return 1;
  }

  auto q_int8_download = std::vector<int8_t>(quantized.q_int8.size());
  auto q_scale_download = std::vector<float>(quantized.q_scale.size());
  auto k_int8_download = std::vector<int8_t>(quantized.k_int8.size());
  auto k_scale_download = std::vector<float>(quantized.k_scale.size());
  auto v_scale_download = std::vector<float>(quantized.v_scale.size());
  auto v_int8_download = std::vector<int8_t>(quantized.v_int8.size());
  auto q_int8_check = NS::TransferPtr(device->newBuffer(q_int8_bytes, kSharedResourceOptions));
  auto q_scale_check = NS::TransferPtr(device->newBuffer(q_scale_bytes, kSharedResourceOptions));
  auto k_int8_check = NS::TransferPtr(device->newBuffer(k_int8_bytes, kSharedResourceOptions));
  auto k_scale_check = NS::TransferPtr(device->newBuffer(k_scale_bytes, kSharedResourceOptions));
  auto v_scale_check = NS::TransferPtr(device->newBuffer(v_scale_bytes, kSharedResourceOptions));
  auto v_int8_check = NS::TransferPtr(device->newBuffer(v_int8_bytes, kSharedResourceOptions));
  download_buffer(command_queue.get(), q_int8_buffer.get(), q_int8_check.get(), q_int8_bytes);
  download_buffer(command_queue.get(), q_scale_buffer.get(), q_scale_check.get(), q_scale_bytes);
  download_buffer(command_queue.get(), k_int8_buffer.get(), k_int8_check.get(), k_int8_bytes);
  download_buffer(command_queue.get(), k_scale_buffer.get(), k_scale_check.get(), k_scale_bytes);
  download_buffer(command_queue.get(), v_scale_buffer.get(), v_scale_check.get(), v_scale_bytes);
  download_buffer(command_queue.get(), v_int8_buffer.get(), v_int8_check.get(), v_int8_bytes);
  std::memcpy(q_int8_download.data(), q_int8_check->contents(), q_int8_bytes);
  std::memcpy(q_scale_download.data(), q_scale_check->contents(), q_scale_bytes);
  std::memcpy(k_int8_download.data(), k_int8_check->contents(), k_int8_bytes);
  std::memcpy(k_scale_download.data(), k_scale_check->contents(), k_scale_bytes);
  std::memcpy(v_scale_download.data(), v_scale_check->contents(), v_scale_bytes);
  std::memcpy(v_int8_download.data(), v_int8_check->contents(), v_int8_bytes);
  const auto quant_validation = validate_quantization(
      quantized,
      q_int8_download,
      q_scale_download,
      k_int8_download,
      k_scale_download,
      v_scale_download,
      v_int8_download);
  std::cout << "quant-validation"
            << " max_abs_q_scale=" << quant_validation.max_abs_q_scale
            << " max_abs_k_scale=" << quant_validation.max_abs_k_scale
            << " max_abs_v_scale=" << quant_validation.max_abs_v_scale
            << " mismatched_q=" << quant_validation.mismatched_q
            << " mismatched_k=" << quant_validation.mismatched_k
            << " mismatched_v=" << quant_validation.mismatched_v
            << '\n';
  if (!quant_validation.passed) {
    std::cerr << "quantization validation failed\n";
    pool->drain();
    return 1;
  }

  std::cout << "shape"
            << " B=" << attention.batch
            << " R=" << attention.R
            << " C=" << attention.C
            << " Hq=" << attention.Hq
            << " Hk=" << attention.Hk
            << " D=" << attention.D
            << " warmup=" << config.warmup_iterations
            << " timed=" << config.timed_iterations
            << " inputPrecision=" << input_precision_name(variant.input_precision)
            << " qkPrecision=" << qk_precision_name(variant.int8_qk)
            << " vPrecision=int8"
            << " qkScales=" << qk_scale_mode_name(variant.int8_qk_scales)
            << " baselineSimdgroups=" << (variant.quantize_only ? 0 : baseline.kernel->executionSIMDGroups)
            << " int8Simdgroups=" << int8_pipeline.execution_simd_groups
            << " qQuantThreads=" << quantize_pipelines.q_threads
            << " kvQuantThreads=" << quantize_pipelines.kv_threads
            << " threadBarrierOverC=" << (int8_pipeline.thread_barrier_over_c ? "true" : "false")
            << '\n';
  if (!variant.quantize_only) {
    std::cout << "int8-kernel"
              << " blockR=" << int8_pipeline.block_dimensions[0]
              << " blockC=" << int8_pipeline.block_dimensions[1]
              << " blockD=" << int8_pipeline.block_dimensions[2]
              << " simdgroups=" << int8_pipeline.execution_simd_groups
              << " inputPrecision=" << input_precision_name(variant.input_precision)
              << " qkPrecision=" << qk_precision_name(int8_pipeline.use_int8_qk)
              << " vPrecision=int8"
              << " qkScales=" << qk_scale_mode_name(int8_pipeline.use_qk_scales)
              << " threadBarrierOverC=" << (int8_pipeline.thread_barrier_over_c ? "true" : "false")
              << '\n';
  }

  if (variant.quantize_only) {
    const uint32_t q_tiles = (attention.R + block_dimensions[0] - 1) / block_dimensions[0];
    const uint32_t k_tiles = (attention.C + block_dimensions[1] - 1) / block_dimensions[1];

    Stats quantize_q_stats;
    if (!benchmark(
            config,
            [&]() {
              return run_quantize_stage_once(
                  command_queue.get(),
                  quantize_pipelines.q_pipeline.get(),
                  quantize_pipelines.q_threads,
                  q_buffer.get(),
                  q_int8_buffer.get(),
                  q_scale_buffer.get(),
                  q_tiles,
                  attention.Hq,
                  attention.batch);
            },
            &quantize_q_stats)) {
      std::cerr << "quantize-q benchmark failed\n";
      pool->drain();
      return 1;
    }

    Stats quantize_k_stats;
    if (!benchmark(
            config,
            [&]() {
              return run_quantize_stage_once(
                  command_queue.get(),
                  quantize_pipelines.k_pipeline.get(),
                  quantize_pipelines.kv_threads,
                  k_buffer.get(),
                  k_int8_buffer.get(),
                  k_scale_buffer.get(),
                  k_tiles,
                  attention.Hk,
                  attention.batch);
            },
            &quantize_k_stats)) {
      std::cerr << "quantize-k benchmark failed\n";
      pool->drain();
      return 1;
    }

    Stats quantize_v_stats;
    if (!benchmark(
            config,
            [&]() {
              return run_quantize_stage_once(
                  command_queue.get(),
                  quantize_pipelines.v_pipeline.get(),
                  quantize_pipelines.kv_threads,
                  v_buffer.get(),
                  v_int8_buffer.get(),
                  v_scale_buffer.get(),
                  k_tiles,
                  attention.Hk,
                  attention.batch);
            },
            &quantize_v_stats)) {
      std::cerr << "quantize-v benchmark failed\n";
      pool->drain();
      return 1;
    }

    Stats quantize_all_stats;
    if (!benchmark(
            config,
            [&]() {
              return run_quantize_once(
                  command_queue.get(),
                  attention,
                  block_dimensions,
                  quantize_pipelines,
                  q_buffer.get(),
                  q_int8_buffer.get(),
                  q_scale_buffer.get(),
                  k_buffer.get(),
                  k_int8_buffer.get(),
                  k_scale_buffer.get(),
                  v_buffer.get(),
                  v_int8_buffer.get(),
                  v_scale_buffer.get());
            },
            &quantize_all_stats)) {
      std::cerr << "quantize-all benchmark failed\n";
      pool->drain();
      return 1;
    }

    print_time_stats("quantize-q", quantize_q_stats);
    print_time_stats("quantize-k", quantize_k_stats);
    print_time_stats("quantize-v", quantize_v_stats);
    print_time_stats("quantize-all", quantize_all_stats);
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
  }

  const double validation_seconds = run_int8_once(
      command_queue.get(),
      attention,
      int8_pipeline,
      qk_q_buffer,
      qk_k_buffer,
      pv_v_buffer,
      o_buffer.get(),
      l_buffer.get(),
      q_scale_buffer.get(),
      k_scale_buffer.get(),
      v_scale_buffer.get());
  if (!(validation_seconds > 0)) {
    std::cerr << "int8 validation dispatch failed\n";
    pool->drain();
    return 1;
  }
  download_buffer(command_queue.get(), o_buffer.get(), o_stage.get(), o_bytes);
  download_buffer(command_queue.get(), l_buffer.get(), l_stage.get(), l_bytes);
  const auto o_values = decode_values(o_stage->contents(), o_count, variant.input_precision);
  const auto validation = validate_int8_outputs(
      attention,
      block_dimensions,
      quantized,
      v_values,
      int8_pipeline.use_qk_scales,
      o_values,
      static_cast<const float*>(l_stage->contents()));
  print_validation(validation);
  if (!validation.passed) {
    std::cerr << "validation failed\n";
    pool->drain();
    return 1;
  }

  if (variant.capture_path && variant.capture_path[0]) {
    const bool started = start_metal_capture(command_queue.get(), variant.capture_path);
    if (!started) {
      pool->drain();
      return 1;
    }
    const double captured_seconds = run_int8_once(
        command_queue.get(),
        attention,
        int8_pipeline,
        qk_q_buffer,
        qk_k_buffer,
        pv_v_buffer,
        o_buffer.get(),
        l_buffer.get(),
        q_scale_buffer.get(),
        k_scale_buffer.get(),
        v_scale_buffer.get());
    stop_metal_capture();
    if (!(captured_seconds > 0)) {
      std::cerr << "captured int8 dispatch failed\n";
      pool->drain();
      return 1;
    }
    std::cout << "capture saved to " << variant.capture_path << '\n';
  }

  Stats baseline_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_baseline_once(command_queue.get(), baseline, q_buffer.get(), k_buffer.get(), v_buffer.get(), o_buffer.get(), l_buffer.get());
          },
          &baseline_stats)) {
    std::cerr << "baseline benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats int8_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_int8_once(command_queue.get(), attention, int8_pipeline, qk_q_buffer, qk_k_buffer, pv_v_buffer, o_buffer.get(), l_buffer.get(), q_scale_buffer.get(), k_scale_buffer.get(), v_scale_buffer.get());
          },
          &int8_stats)) {
    std::cerr << "int8 benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats quantize_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_quantize_once(
                command_queue.get(),
                attention,
                block_dimensions,
                quantize_pipelines,
                q_buffer.get(),
                q_int8_buffer.get(),
                q_scale_buffer.get(),
                k_buffer.get(),
                k_int8_buffer.get(),
                k_scale_buffer.get(),
                v_buffer.get(),
                v_int8_buffer.get(),
                v_scale_buffer.get());
          },
          &quantize_stats)) {
    std::cerr << "quantize benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats quantize_and_int8_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_quantize_and_int8_once(
                command_queue.get(),
                attention,
                block_dimensions,
                quantize_pipelines,
                int8_pipeline,
                q_buffer.get(),
                q_int8_buffer.get(),
                q_scale_buffer.get(),
                k_buffer.get(),
                k_int8_buffer.get(),
                k_scale_buffer.get(),
                v_buffer.get(),
                v_int8_buffer.get(),
                v_scale_buffer.get(),
                o_buffer.get(),
                l_buffer.get());
          },
          &quantize_and_int8_stats)) {
    std::cerr << "quantize+int8 benchmark failed\n";
    pool->drain();
    return 1;
  }

  print_stats("baseline", attention, baseline_stats);
  print_stats("quantize", attention, quantize_stats);
  print_stats("full", attention, int8_stats);
  print_stats("quantize+int8", attention, quantize_and_int8_stats);
  std::cout << "speedup"
            << " avg=" << baseline_stats.average_seconds / int8_stats.average_seconds
            << " median=" << baseline_stats.median_seconds / int8_stats.median_seconds
            << '\n';
  std::cout << "quantized-speedup"
            << " avg=" << baseline_stats.average_seconds / quantize_and_int8_stats.average_seconds
            << " median=" << baseline_stats.median_seconds / quantize_and_int8_stats.median_seconds
            << '\n';

  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
