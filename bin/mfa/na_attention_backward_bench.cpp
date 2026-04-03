#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/AttentionKernelType.hpp"
#include "nnc/mfa/kernels/AttentionOperand.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAAttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAAttentionKernel.hpp"
#include "nnc/mfa/kernels/NAAttentionKernelDescriptor.hpp"

namespace {

using half_float = _Float16;

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
};

struct AttentionCase {
  uint32_t batch = 1;
  uint32_t R = 4096;
  uint32_t C = 4096;
  uint32_t Hq = 32;
  uint32_t Hk = 32;
  uint32_t D = 128;
};

struct Stats {
  double average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct ForwardPipeline {
  NAAttentionDescriptor descriptor;
  std::unique_ptr<NAAttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  bool morton_order = false;
};

struct BackwardPipelines {
  NAAttentionDescriptor query_descriptor;
  NAAttentionDescriptor keyvalue_descriptor;
  std::unique_ptr<NAAttentionKernel> query_kernel;
  std::unique_ptr<NAAttentionKernel> keyvalue_kernel;
  NS::SharedPtr<MTL::ComputePipelineState> compute_d_pipeline;
  uint16_t compute_d_threads = NAAttentionKernel::computeDThreads;
  bool custom_compute_d = false;
  NS::SharedPtr<MTL::ComputePipelineState> query_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> keyvalue_pipeline;
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

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

uint32_t ceil_log2_u32(uint32_t x)
{
  uint32_t bits = 0;
  uint32_t value = 1;
  while (value < x) {
    value <<= 1;
    ++bits;
  }
  return bits;
}

std::vector<uint8_t> encode_fp16(const std::vector<float>& values)
{
  std::vector<uint8_t> bytes(values.size() * sizeof(half_float));
  auto* dst = reinterpret_cast<half_float*>(bytes.data());
  for (size_t i = 0; i < values.size(); ++i)
    dst[i] = (half_float)values[i];
  return bytes;
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

bool benchmark(const BenchmarkConfig& config, const std::function<double()>& run_once, Stats* stats)
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

void print_stats(const char* label, const Stats& stats)
{
  std::cout << std::fixed
            << label
            << " avg_ms=" << std::setprecision(4) << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << '\n';
}

MTL::Size create_attention_threadgroups_per_grid(
    const AttentionCase& attention,
    AttentionKernelType type,
    simd::ushort3 block_dimensions,
    uint16_t execution_simdgroups,
    bool morton_order)
{
  auto ceil_divide =
      [&](uint32_t target, uint32_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  const uint32_t dispatch_dimension =
      type.value == AttentionKernelType::backwardKeyValue ? attention.C : attention.R;
  const uint32_t dispatch_heads =
      type.value == AttentionKernelType::backwardKeyValue ? attention.Hk : attention.Hq;
  const uint32_t row_groups =
      ceil_divide(dispatch_dimension, block_dimensions[0] * execution_simdgroups);
  if (!morton_order)
    return MTL::Size((uint64_t)row_groups * dispatch_heads * attention.batch, 1, 1);
  const uint32_t row_bits = ceil_log2_u32(row_groups);
  const uint32_t head_bits = ceil_log2_u32(dispatch_heads);
  const uint32_t morton_codes = 1u << (row_bits + head_bits);
  return MTL::Size((uint64_t)morton_codes, 1, attention.batch);
}

std::string create_row_major_attention_source(
    const std::string& source,
    const AttentionCase& attention,
    AttentionKernelType type,
    simd::ushort3 block_dimensions,
    uint16_t execution_simdgroups)
{
  std::string output = source;
  const uint32_t dispatch_dimension =
      type.value == AttentionKernelType::backwardKeyValue ? attention.C : attention.R;
  const uint32_t dispatch_heads =
      type.value == AttentionKernelType::backwardKeyValue ? attention.Hk : attention.Hq;
  const uint32_t row_groups =
      (dispatch_dimension + block_dimensions[0] * execution_simdgroups - 1) /
      (block_dimensions[0] * execution_simdgroups);

  const size_t tgid_pos = output.find("  const uint row_group_count = ");
  const size_t if_pos = output.find("  if (tgid.x * ", tgid_pos);
  const size_t end_pos = output.find("  }\n", if_pos);
  std::ostringstream replacement;
  replacement
      << "  tgid = { (tgid.x / " << dispatch_heads << "u) % " << row_groups << "u, "
      << "tgid.x % " << dispatch_heads << "u, "
      << "tgid.x / " << dispatch_heads << "u / " << row_groups << "u };\n"
      << "  tgid.x = tgid.x * " << execution_simdgroups << " + sgid;\n"
      << "  if (tgid.x * " << block_dimensions[0] << " >= " << dispatch_dimension << "u) {\n"
      << "    return;\n"
      << "  }\n";
  if (tgid_pos != std::string::npos && if_pos != std::string::npos && end_pos != std::string::npos)
    output.replace(tgid_pos, end_pos + 4 - tgid_pos, replacement.str());
  return output;
}

simd::ushort3 create_forward_block_dimensions(const AttentionCase& attention)
{
  const unsigned short head_dimension = attention.D;
  unsigned short revised_head = (head_dimension + 15) / 16 * 16;
  if (head_dimension <= 128)
    revised_head = std::min<unsigned short>(head_dimension, revised_head);
  else
    revised_head = revised_head / std::max<unsigned short>(revised_head / 128, 2);
  if (attention.C % 64 == 0)
    return simd::ushort3 { 16, 64, revised_head };
  if (attention.C % 48 == 0)
    return simd::ushort3 { 16, 48, revised_head };
  if (attention.C % 128 > 64 && attention.C % 96 < 48)
    return simd::ushort3 { 16, 64, revised_head };
  if (attention.C % 128 < 64 && attention.C % 96 > 48)
    return simd::ushort3 { 16, 48, revised_head };
  const unsigned short remainder64 = attention.C % 64;
  const unsigned short remainder48 = attention.C % 48;
  return (remainder64 * 48 < remainder48 * 64) ?
      simd::ushort3 { 16, 48, revised_head } :
      simd::ushort3 { 16, 64, revised_head };
}

simd::ushort3 create_backward_block_dimensions(const AttentionCase& attention)
{
  auto block_dimensions = create_forward_block_dimensions(attention);
  if (attention.D == 128)
    block_dimensions[2] = 64;
  return block_dimensions;
}

bool create_backward_bypass_threadgroup_memory(const AttentionCase& attention)
{
  const auto block_dimensions = create_backward_block_dimensions(attention);
  const uint32_t min_sequence_dimension = (attention.R < attention.C) ? attention.R : attention.C;
  if (attention.D == 128 && min_sequence_dimension >= 4096)
    return false;
  switch (block_dimensions[1]) {
  case 64:
    switch (block_dimensions[2]) {
    case 32:
    case 40:
    case 64:
    case 80:
    case 96:
      return true;
    default:
      return false;
    }
  case 48:
    switch (block_dimensions[2]) {
    case 32:
    case 40:
    case 48:
    case 64:
    case 72:
    case 80:
    case 96:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

AttentionOperands<GEMMOperandPrecision> create_fp16_backward_precisions()
{
  AttentionOperands<GEMMOperandPrecision> memory_precisions;
  memory_precisions[AttentionOperand::Q] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::K] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::V] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::O] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::dO] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::dQ] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::dK] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::dV] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::L] = GEMMOperandPrecision::FP16;
  memory_precisions[AttentionOperand::D] = GEMMOperandPrecision::BF16;
  return memory_precisions;
}

NS::SharedPtr<MTL::ComputePipelineState> create_attention_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const std::string* source_override,
    const AttentionCase& attention,
    AttentionKernelType type,
    simd::ushort3 block_dimensions,
    uint16_t execution_simdgroups,
    bool morton_order)
{
  NS::SharedPtr<MTL::Library> active_library;
  if (type.value == AttentionKernelType::forward && source_override && !morton_order) {
    const auto row_major_source = create_row_major_attention_source(
        *source_override,
        attention,
        type,
        block_dimensions,
        execution_simdgroups);
    auto source_string = NS::String::string(row_major_source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    active_library = NS::TransferPtr(device->newLibrary(source_string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
  MTL::Library* pipeline_library = active_library ? active_library.get() : library;
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t row_dimension = attention.R;
  const uint32_t column_dimension = attention.C;
  constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, NS::UInteger(1));
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  const uint32_t q_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t k_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t v_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t o_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t dO_batch_stride = o_batch_stride;
  const uint32_t dQ_batch_stride = o_batch_stride;
  const uint32_t dK_batch_stride = k_batch_stride;
  const uint32_t dV_batch_stride = v_batch_stride;
  auto batch_stride_for =
      [&](AttentionOperand operand) -> uint32_t {
    switch (operand.value) {
    case AttentionOperand::Q: return q_batch_stride;
    case AttentionOperand::K: return k_batch_stride;
    case AttentionOperand::V: return v_batch_stride;
    case AttentionOperand::O: return o_batch_stride;
    case AttentionOperand::dO: return dO_batch_stride;
    case AttentionOperand::dQ: return dQ_batch_stride;
    case AttentionOperand::dK: return dK_batch_stride;
    case AttentionOperand::dV: return dV_batch_stride;
    default: return 0;
    }
  };
  for (const auto& operand : operands) {
    const uint32_t batch_stride = batch_stride_for(operand);
    constants->setConstantValue(&batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + operand.bufferIndex()));
  }
  auto function_name = NS::String::string("attention", NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto function = NS::TransferPtr(pipeline_library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipeline_descriptor->setComputeFunction(function.get());
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return pipeline;
}

NS::SharedPtr<MTL::ComputePipelineState> create_compute_d_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const AttentionCase& attention,
    uint16_t compute_d_threads)
{
  if (compute_d_threads != 0 && compute_d_threads != NAAttentionKernel::computeDThreads) {
    std::ostringstream source;
    const uint16_t simdgroups = (compute_d_threads + 31) / 32;
    source << R"(
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;

constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint Hq = )" << attention.Hq << R"(;
constant uint K_Hq = )" << (attention.D * attention.Hq) << R"(;
constant uint O_batch_stride [[function_constant(5)]];
constant uint dO_batch_stride [[function_constant(8)]];

kernel void compute_d_bench(
    device const half* O_buf [[buffer(3)]],
    device const half* dO_buf [[buffer(6)]],
    device float* D_buf [[buffer(5)]],
    threadgroup float* partial_sums [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x % R;
  const uint head = tgid.x / R;
  O_buf += tgid.z * O_batch_stride;
  dO_buf += tgid.z * dO_batch_stride;
  D_buf += (tgid.z * Hq + head) * R;

  const uint offset = row * K_Hq + head * )" << attention.D << R"(;
  float D_accumulator = 0;
  for (uint d = tid; d < )" << attention.D << R"(; d += )" << compute_d_threads << R"() {
    D_accumulator += (float)O_buf[offset + d] * (float)dO_buf[offset + d];
  }
  D_accumulator += simd_shuffle_xor(D_accumulator, 16);
  D_accumulator += simd_shuffle_xor(D_accumulator, 8);
  D_accumulator += simd_shuffle_xor(D_accumulator, 4);
  D_accumulator += simd_shuffle_xor(D_accumulator, 2);
  D_accumulator += simd_shuffle_xor(D_accumulator, 1);
  if (lane_id == 0) {
    partial_sums[sgid] = D_accumulator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    float sum = lane_id < )" << simdgroups << R"( ? partial_sums[lane_id] : 0.0f;
    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane_id == 0) {
      D_buf[row] = sum * )" << create_scale(attention) << R"(f;
    }
  }
}
)";
    auto source_string = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto custom_library = NS::TransferPtr(device->newLibrary(source_string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t row_dimension = attention.R;
    const uint32_t column_dimension = attention.C;
    const uint32_t o_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
    const uint32_t dO_batch_stride = o_batch_stride;
    constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&o_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::O).bufferIndex()));
    constants->setConstantValue(&dO_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::dO).bufferIndex()));
    auto function_name = NS::String::string("compute_d_bench", NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(custom_library->newFunction(function_name, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipeline_descriptor->setComputeFunction(function.get());
    auto pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  }
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t row_dimension = attention.R;
  const uint32_t column_dimension = attention.C;
  const uint32_t o_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t dO_batch_stride = o_batch_stride;
  constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&o_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::O).bufferIndex()));
  constants->setConstantValue(&dO_batch_stride, MTL::DataTypeUInt, NS::UInteger(2 + AttentionOperand(AttentionOperand::dO).bufferIndex()));
  auto function_name = NS::String::string("compute_d", NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto function = NS::TransferPtr(library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipeline_descriptor->setComputeFunction(function.get());
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return pipeline;
}

ForwardPipeline create_forward_pipeline(MTL::Device* device, const AttentionCase& attention, bool morton_order)
{
  ForwardPipeline bundle;
  bundle.descriptor.batchDimension = attention.batch;
  bundle.descriptor.Hq = attention.Hq;
  bundle.descriptor.Hk = attention.Hk;
  bundle.descriptor.lowPrecisionInputs = true;
  bundle.descriptor.isBF16 = false;
  bundle.descriptor.lowPrecisionIntermediates = true;
  bundle.descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  bundle.descriptor.type = AttentionKernelType::forward;
  bundle.descriptor.scale = create_scale(attention);
  if (attention.batch > 1) {
    bundle.descriptor.batchStrides[AttentionOperand::Q] = attention.R * attention.D * attention.Hq;
    bundle.descriptor.batchStrides[AttentionOperand::K] = attention.C * attention.D * attention.Hk;
    bundle.descriptor.batchStrides[AttentionOperand::V] = attention.C * attention.D * attention.Hk;
    bundle.descriptor.batchStrides[AttentionOperand::O] = attention.R * attention.D * attention.Hq;
  }
  const simd::ushort3 block_dimensions = create_forward_block_dimensions(attention);
  const AttentionOperands<GEMMOperandPrecision> memory_precisions = create_fp16_backward_precisions();
  const bool check_c_edge_1 = (attention.C % (block_dimensions[1] * 2)) > block_dimensions[1];
  const NAAttentionKernelDescriptor kernel_descriptor(
      block_dimensions, attention.D, attention.Hq, attention.Hk, 16, check_c_edge_1,
      memory_precisions, AttentionKernelType::forward, bundle.descriptor.scale);
  bundle.kernel = std::make_unique<NAAttentionKernel>(kernel_descriptor, device);
  bundle.morton_order = morton_order;
  bundle.pipeline = create_attention_pipeline(
      device,
      bundle.kernel->library.get(),
      &bundle.kernel->source,
      attention,
      AttentionKernelType::forward,
      block_dimensions,
      bundle.kernel->executionSIMDGroups,
      morton_order);
  return bundle;
}

BackwardPipelines create_backward_pipelines(
    MTL::Device* device,
    const AttentionCase& attention,
    simd::ushort3 query_block_dimensions,
    simd::ushort3 keyvalue_block_dimensions,
    uint16_t query_execution_simdgroups,
    uint16_t keyvalue_execution_simdgroups,
    bool query_bypass_threadgroup_memory,
    bool keyvalue_bypass_threadgroup_memory,
    uint16_t compute_d_threads)
{
  BackwardPipelines bundle;
  bundle.query_descriptor.batchDimension = attention.batch;
  bundle.query_descriptor.Hq = attention.Hq;
  bundle.query_descriptor.Hk = attention.Hk;
  bundle.query_descriptor.lowPrecisionInputs = true;
  bundle.query_descriptor.isBF16 = false;
  bundle.query_descriptor.lowPrecisionIntermediates = true;
  bundle.query_descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  bundle.query_descriptor.type = AttentionKernelType::backwardQuery;
  bundle.query_descriptor.scale = create_scale(attention);
  bundle.keyvalue_descriptor = bundle.query_descriptor;
  bundle.keyvalue_descriptor.type = AttentionKernelType::backwardKeyValue;
  if (attention.batch > 1) {
    bundle.query_descriptor.batchStrides[AttentionOperand::Q] = attention.R * attention.D * attention.Hq;
    bundle.query_descriptor.batchStrides[AttentionOperand::K] = attention.C * attention.D * attention.Hk;
    bundle.query_descriptor.batchStrides[AttentionOperand::V] = attention.C * attention.D * attention.Hk;
    bundle.query_descriptor.batchStrides[AttentionOperand::O] = attention.R * attention.D * attention.Hq;
    bundle.query_descriptor.batchStrides[AttentionOperand::dO] = attention.R * attention.D * attention.Hq;
    bundle.query_descriptor.batchStrides[AttentionOperand::dQ] = attention.R * attention.D * attention.Hq;
    bundle.query_descriptor.batchStrides[AttentionOperand::dK] = attention.C * attention.D * attention.Hk;
    bundle.query_descriptor.batchStrides[AttentionOperand::dV] = attention.C * attention.D * attention.Hk;
    bundle.keyvalue_descriptor.batchStrides = bundle.query_descriptor.batchStrides;
  }
  const AttentionOperands<GEMMOperandPrecision> memory_precisions = create_fp16_backward_precisions();
  const bool query_check_c_edge_1 = (attention.C % (query_block_dimensions[1] * 2)) > query_block_dimensions[1];
  const bool keyvalue_check_c_edge_1 = (attention.C % (keyvalue_block_dimensions[1] * 2)) > keyvalue_block_dimensions[1];
  const NAAttentionKernelDescriptor query_kernel_descriptor(
      query_block_dimensions, attention.D, attention.Hq, attention.Hk, query_execution_simdgroups, query_check_c_edge_1,
      memory_precisions, AttentionKernelType::backwardQuery, bundle.query_descriptor.scale, query_bypass_threadgroup_memory);
  const NAAttentionKernelDescriptor keyvalue_kernel_descriptor(
      keyvalue_block_dimensions, attention.D, attention.Hq, attention.Hk, keyvalue_execution_simdgroups, keyvalue_check_c_edge_1,
      memory_precisions, AttentionKernelType::backwardKeyValue, bundle.keyvalue_descriptor.scale, keyvalue_bypass_threadgroup_memory);
  bundle.query_kernel = std::make_unique<NAAttentionKernel>(query_kernel_descriptor, device);
  bundle.keyvalue_kernel = std::make_unique<NAAttentionKernel>(keyvalue_kernel_descriptor, device);
  bundle.compute_d_threads = compute_d_threads ? compute_d_threads : NAAttentionKernel::computeDThreads;
  bundle.custom_compute_d = (bundle.compute_d_threads != NAAttentionKernel::computeDThreads);
  bundle.compute_d_pipeline = create_compute_d_pipeline(device, bundle.query_kernel->library.get(), attention, bundle.compute_d_threads);
  bundle.query_pipeline = create_attention_pipeline(
      device,
      bundle.query_kernel->library.get(),
      nullptr,
      attention,
      AttentionKernelType::backwardQuery,
      query_block_dimensions,
      query_execution_simdgroups,
      false);
  bundle.keyvalue_pipeline = create_attention_pipeline(
      device,
      bundle.keyvalue_kernel->library.get(),
      nullptr,
      attention,
      AttentionKernelType::backwardKeyValue,
      keyvalue_block_dimensions,
      keyvalue_execution_simdgroups,
      false);
  return bundle;
}

double run_forward_once(
    MTL::CommandQueue* command_queue,
    const ForwardPipeline& pipeline,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline.pipeline.get());
  encoder->setThreadgroupMemoryLength(pipeline.kernel->threadgroupMemoryAllocation(pipeline.pipeline.get(), pipeline.descriptor), 0);
  encoder->setBuffer(q_buffer, 0, 0);
  encoder->setBuffer(k_buffer, 0, 1);
  encoder->setBuffer(v_buffer, 0, 2);
  encoder->setBuffer(o_buffer, 0, 3);
  encoder->setBuffer(l_buffer, 0, 4);
  const auto threadgroups_per_grid = create_attention_threadgroups_per_grid(
      AttentionCase {
          pipeline.descriptor.batchDimension,
          pipeline.descriptor.matrixDimensions[0],
          pipeline.descriptor.matrixDimensions[1],
          pipeline.descriptor.Hq,
          pipeline.descriptor.Hk,
          pipeline.descriptor.matrixDimensions[2]},
      AttentionKernelType::forward,
      pipeline.kernel->blockDimensions,
      pipeline.kernel->executionSIMDGroups,
      pipeline.morton_order);
  encoder->dispatchThreadgroups(
      threadgroups_per_grid,
      MTL::Size(pipeline.kernel->threadgroupSize(pipeline.pipeline.get(), pipeline.descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_compute_d_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const BackwardPipelines& pipelines,
    MTL::Buffer* o_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* d_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipelines.compute_d_pipeline.get());
  if (pipelines.custom_compute_d) {
    encoder->setThreadgroupMemoryLength(((pipelines.compute_d_threads + 31) / 32) * sizeof(float), 0);
  }
  encoder->setBuffer(o_buffer, 0, 3);
  encoder->setBuffer(dO_buffer, 0, 6);
  encoder->setBuffer(d_buffer, 0, 5);
  encoder->dispatchThreadgroups(
      MTL::Size((uint64_t)attention.R * attention.Hq, 1, attention.batch),
      MTL::Size(pipelines.compute_d_threads, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_backward_query_only_once(
    MTL::CommandQueue* command_queue,
    const BackwardPipelines& pipelines,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* dQ_buffer,
    MTL::Buffer* d_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipelines.query_pipeline.get());
  encoder->setThreadgroupMemoryLength(
      pipelines.query_kernel->threadgroupMemoryAllocation(pipelines.query_pipeline.get(), pipelines.query_descriptor), 0);
  encoder->setBuffer(q_buffer, 0, 0);
  encoder->setBuffer(k_buffer, 0, 1);
  encoder->setBuffer(v_buffer, 0, 2);
  encoder->setBuffer(l_buffer, 0, 4);
  encoder->setBuffer(d_buffer, 0, 5);
  encoder->setBuffer(dO_buffer, 0, 6);
  encoder->setBuffer(dQ_buffer, 0, 7);
  encoder->dispatchThreadgroups(
      pipelines.query_kernel->threadgroupsPerGrid(pipelines.query_descriptor),
      MTL::Size(pipelines.query_kernel->threadgroupSize(pipelines.query_pipeline.get(), pipelines.query_descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_backward_keyvalue_only_once(
    MTL::CommandQueue* command_queue,
    const BackwardPipelines& pipelines,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* dK_buffer,
    MTL::Buffer* dV_buffer,
    MTL::Buffer* d_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipelines.keyvalue_pipeline.get());
  encoder->setThreadgroupMemoryLength(
      pipelines.keyvalue_kernel->threadgroupMemoryAllocation(pipelines.keyvalue_pipeline.get(), pipelines.keyvalue_descriptor), 0);
  encoder->setBuffer(q_buffer, 0, 0);
  encoder->setBuffer(k_buffer, 0, 1);
  encoder->setBuffer(v_buffer, 0, 2);
  encoder->setBuffer(l_buffer, 0, 4);
  encoder->setBuffer(d_buffer, 0, 5);
  encoder->setBuffer(dO_buffer, 0, 6);
  encoder->setBuffer(dV_buffer, 0, 7);
  encoder->setBuffer(dK_buffer, 0, 8);
  encoder->dispatchThreadgroups(
      pipelines.keyvalue_kernel->threadgroupsPerGrid(pipelines.keyvalue_descriptor),
      MTL::Size(pipelines.keyvalue_kernel->threadgroupSize(pipelines.keyvalue_pipeline.get(), pipelines.keyvalue_descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_backward_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const BackwardPipelines& pipelines,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* dQ_buffer,
    MTL::Buffer* dK_buffer,
    MTL::Buffer* dV_buffer,
    MTL::Buffer* d_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.compute_d_pipeline.get());
    if (pipelines.custom_compute_d) {
      encoder->setThreadgroupMemoryLength(((pipelines.compute_d_threads + 31) / 32) * sizeof(float), 0);
    }
    encoder->setBuffer(o_buffer, 0, 3);
    encoder->setBuffer(dO_buffer, 0, 6);
    encoder->setBuffer(d_buffer, 0, 5);
    encoder->dispatchThreadgroups(
        MTL::Size((uint64_t)attention.R * attention.Hq, 1, attention.batch),
        MTL::Size(pipelines.compute_d_threads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.query_pipeline.get());
    encoder->setThreadgroupMemoryLength(
        pipelines.query_kernel->threadgroupMemoryAllocation(pipelines.query_pipeline.get(), pipelines.query_descriptor), 0);
    encoder->setBuffer(q_buffer, 0, 0);
    encoder->setBuffer(k_buffer, 0, 1);
    encoder->setBuffer(v_buffer, 0, 2);
    encoder->setBuffer(l_buffer, 0, 4);
    encoder->setBuffer(d_buffer, 0, 5);
    encoder->setBuffer(dO_buffer, 0, 6);
    encoder->setBuffer(dQ_buffer, 0, 7);
    encoder->dispatchThreadgroups(
        pipelines.query_kernel->threadgroupsPerGrid(pipelines.query_descriptor),
        MTL::Size(pipelines.query_kernel->threadgroupSize(pipelines.query_pipeline.get(), pipelines.query_descriptor), 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.keyvalue_pipeline.get());
    encoder->setThreadgroupMemoryLength(
        pipelines.keyvalue_kernel->threadgroupMemoryAllocation(pipelines.keyvalue_pipeline.get(), pipelines.keyvalue_descriptor), 0);
    encoder->setBuffer(q_buffer, 0, 0);
    encoder->setBuffer(k_buffer, 0, 1);
    encoder->setBuffer(v_buffer, 0, 2);
    encoder->setBuffer(l_buffer, 0, 4);
    encoder->setBuffer(d_buffer, 0, 5);
    encoder->setBuffer(dO_buffer, 0, 6);
    encoder->setBuffer(dV_buffer, 0, 7);
    encoder->setBuffer(dK_buffer, 0, 8);
    encoder->dispatchThreadgroups(
        pipelines.keyvalue_kernel->threadgroupsPerGrid(pipelines.keyvalue_descriptor),
        MTL::Size(pipelines.keyvalue_kernel->threadgroupSize(pipelines.keyvalue_pipeline.get(), pipelines.keyvalue_descriptor), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  AttentionCase attention;
  BenchmarkConfig config;
  uint16_t block_r = 0;
  uint16_t block_c = 0;
  uint16_t block_d = 0;
  uint16_t keyvalue_block_r = 0;
  uint16_t keyvalue_block_c = 0;
  uint16_t keyvalue_block_d = 0;
  uint16_t query_backward_simdgroups = 8;
  uint16_t keyvalue_backward_simdgroups = 8;
  bool query_bypass_threadgroup_memory = false;
  bool keyvalue_bypass_threadgroup_memory = false;
  uint16_t compute_d_threads = 0;
  bool forward_morton_order = false;
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
  if (argc >= 12) {
    block_r = (uint16_t)std::strtoul(argv[9], nullptr, 10);
    block_c = (uint16_t)std::strtoul(argv[10], nullptr, 10);
    block_d = (uint16_t)std::strtoul(argv[11], nullptr, 10);
  }
  if (argc >= 15) {
    keyvalue_block_r = (uint16_t)std::strtoul(argv[12], nullptr, 10);
    keyvalue_block_c = (uint16_t)std::strtoul(argv[13], nullptr, 10);
    keyvalue_block_d = (uint16_t)std::strtoul(argv[14], nullptr, 10);
  }
  if (argc >= 16) {
    query_backward_simdgroups = (uint16_t)std::strtoul(argv[15], nullptr, 10);
    keyvalue_backward_simdgroups = query_backward_simdgroups;
  }
  if (argc >= 17) {
    keyvalue_backward_simdgroups = (uint16_t)std::strtoul(argv[16], nullptr, 10);
  }
  if (argc >= 18) {
    query_bypass_threadgroup_memory = std::strtoul(argv[17], nullptr, 10) != 0;
    keyvalue_bypass_threadgroup_memory = query_bypass_threadgroup_memory;
  }
  if (argc >= 19) {
    keyvalue_bypass_threadgroup_memory = std::strtoul(argv[18], nullptr, 10) != 0;
  } else if (argc < 18) {
    query_bypass_threadgroup_memory = create_backward_bypass_threadgroup_memory(attention);
    keyvalue_bypass_threadgroup_memory = query_bypass_threadgroup_memory;
  }
  if (argc >= 20) {
    compute_d_threads = (uint16_t)std::strtoul(argv[19], nullptr, 10);
  }
  if (argc >= 21) {
    forward_morton_order = std::strtoul(argv[20], nullptr, 10) != 0;
  } else {
    forward_morton_order = true;
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

  const auto q_values = make_data<float>((size_t)attention.batch * attention.R * attention.Hq * attention.D, 0.03125f, 1);
  const auto k_values = make_data<float>((size_t)attention.batch * attention.C * attention.Hk * attention.D, 0.02734375f, 2);
  const auto v_values = make_data<float>((size_t)attention.batch * attention.C * attention.Hk * attention.D, 0.0234375f, 3);
  const auto dO_values = make_data<float>((size_t)attention.batch * attention.R * attention.Hq * attention.D, 0.01953125f, 4);
  const auto q_bytes_data = encode_fp16(q_values);
  const auto k_bytes_data = encode_fp16(k_values);
  const auto v_bytes_data = encode_fp16(v_values);
  const auto dO_bytes_data = encode_fp16(dO_values);

  const size_t q_bytes = q_bytes_data.size();
  const size_t k_bytes = k_bytes_data.size();
  const size_t v_bytes = v_bytes_data.size();
  const size_t dO_bytes = dO_bytes_data.size();
  const size_t o_bytes = (size_t)attention.batch * attention.R * attention.Hq * attention.D * sizeof(half_float);
  const size_t l_bytes = (size_t)attention.batch * attention.R * attention.Hq * sizeof(float);
  const size_t dQ_bytes = o_bytes;
  const size_t dK_bytes = (size_t)attention.batch * attention.C * attention.Hk * attention.D * sizeof(half_float);
  const size_t dV_bytes = dK_bytes;
  const size_t d_bytes = (size_t)attention.batch * attention.R * attention.Hq * sizeof(float);

  auto q_stage = NS::TransferPtr(device->newBuffer(q_bytes_data.data(), q_bytes, kSharedResourceOptions));
  auto k_stage = NS::TransferPtr(device->newBuffer(k_bytes_data.data(), k_bytes, kSharedResourceOptions));
  auto v_stage = NS::TransferPtr(device->newBuffer(v_bytes_data.data(), v_bytes, kSharedResourceOptions));
  auto dO_stage = NS::TransferPtr(device->newBuffer(dO_bytes_data.data(), dO_bytes, kSharedResourceOptions));
  auto q_buffer = NS::TransferPtr(device->newBuffer(q_bytes, kPrivateResourceOptions));
  auto k_buffer = NS::TransferPtr(device->newBuffer(k_bytes, kPrivateResourceOptions));
  auto v_buffer = NS::TransferPtr(device->newBuffer(v_bytes, kPrivateResourceOptions));
  auto o_buffer = NS::TransferPtr(device->newBuffer(o_bytes, kPrivateResourceOptions));
  auto l_buffer = NS::TransferPtr(device->newBuffer(l_bytes, kPrivateResourceOptions));
  auto dO_buffer = NS::TransferPtr(device->newBuffer(dO_bytes, kPrivateResourceOptions));
  auto dQ_buffer = NS::TransferPtr(device->newBuffer(dQ_bytes, kPrivateResourceOptions));
  auto dK_buffer = NS::TransferPtr(device->newBuffer(dK_bytes, kPrivateResourceOptions));
  auto dV_buffer = NS::TransferPtr(device->newBuffer(dV_bytes, kPrivateResourceOptions));
  auto d_buffer = NS::TransferPtr(device->newBuffer(d_bytes, kPrivateResourceOptions));

  upload_buffer(command_queue.get(), q_stage.get(), q_buffer.get(), q_bytes);
  upload_buffer(command_queue.get(), k_stage.get(), k_buffer.get(), k_bytes);
  upload_buffer(command_queue.get(), v_stage.get(), v_buffer.get(), v_bytes);
  upload_buffer(command_queue.get(), dO_stage.get(), dO_buffer.get(), dO_bytes);

  const auto forward_pipeline = create_forward_pipeline(device.get(), attention, forward_morton_order);
  const simd::ushort3 query_block_dimensions = (block_r != 0 && block_c != 0 && block_d != 0) ?
      simd::ushort3 { block_r, block_c, block_d } :
      create_backward_block_dimensions(attention);
  const simd::ushort3 keyvalue_block_dimensions = (keyvalue_block_r != 0 && keyvalue_block_c != 0 && keyvalue_block_d != 0) ?
      simd::ushort3 { keyvalue_block_r, keyvalue_block_c, keyvalue_block_d } :
      query_block_dimensions;
  const bool valid_query_block_d =
      (query_block_dimensions[2] != 0) &&
      (attention.D % (uint32_t)query_block_dimensions[2] == 0);
  const bool valid_keyvalue_block_d =
      (keyvalue_block_dimensions[2] != 0) &&
      (attention.D % (uint32_t)keyvalue_block_dimensions[2] == 0);
  if (!valid_query_block_d || !valid_keyvalue_block_d) {
    std::cerr << "invalid blockD: backwardQuery/backwardKeyValue require blockD to divide D\n";
    pool->drain();
    return 2;
  }
  const auto backward_pipelines = create_backward_pipelines(device.get(), attention, query_block_dimensions, keyvalue_block_dimensions, query_backward_simdgroups, keyvalue_backward_simdgroups, query_bypass_threadgroup_memory, keyvalue_bypass_threadgroup_memory, compute_d_threads);

  const double setup_forward_seconds = run_forward_once(
      command_queue.get(),
      forward_pipeline,
      q_buffer.get(),
      k_buffer.get(),
      v_buffer.get(),
      o_buffer.get(),
      l_buffer.get());
  if (!(setup_forward_seconds > 0)) {
    std::cerr << "forward setup failed\n";
    pool->drain();
    return 1;
  }

  const double setup_compute_d_seconds = run_compute_d_once(
      command_queue.get(),
      attention,
      backward_pipelines,
      o_buffer.get(),
      dO_buffer.get(),
      d_buffer.get());
  if (!(setup_compute_d_seconds > 0)) {
    std::cerr << "compute_d setup failed\n";
    pool->drain();
    return 1;
  }

  Stats forward_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_forward_once(
                command_queue.get(),
                forward_pipeline,
                q_buffer.get(),
                k_buffer.get(),
                v_buffer.get(),
                o_buffer.get(),
                l_buffer.get());
          },
          &forward_stats)) {
    std::cerr << "forward benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats compute_d_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_compute_d_once(
                command_queue.get(),
                attention,
                backward_pipelines,
                o_buffer.get(),
                dO_buffer.get(),
                d_buffer.get());
          },
          &compute_d_stats)) {
    std::cerr << "compute_d benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats query_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_backward_query_only_once(
                command_queue.get(),
                backward_pipelines,
                q_buffer.get(),
                k_buffer.get(),
                v_buffer.get(),
                l_buffer.get(),
                dO_buffer.get(),
                dQ_buffer.get(),
                d_buffer.get());
          },
          &query_stats)) {
    std::cerr << "query benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats keyvalue_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_backward_keyvalue_only_once(
                command_queue.get(),
                backward_pipelines,
                q_buffer.get(),
                k_buffer.get(),
                v_buffer.get(),
                l_buffer.get(),
                dO_buffer.get(),
                dK_buffer.get(),
                dV_buffer.get(),
                d_buffer.get());
          },
          &keyvalue_stats)) {
    std::cerr << "keyvalue benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats backward_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_backward_once(
                command_queue.get(),
                attention,
                backward_pipelines,
                q_buffer.get(),
                k_buffer.get(),
                v_buffer.get(),
                o_buffer.get(),
                l_buffer.get(),
                dO_buffer.get(),
                dQ_buffer.get(),
                dK_buffer.get(),
                dV_buffer.get(),
                d_buffer.get());
          },
          &backward_stats)) {
    std::cerr << "backward benchmark failed\n";
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
            << '\n';
  std::cout << "forward-kernel"
            << " blockR=" << forward_pipeline.kernel->blockDimensions[0]
            << " blockC=" << forward_pipeline.kernel->blockDimensions[1]
            << " blockD=" << forward_pipeline.kernel->blockDimensions[2]
            << " simdgroups=" << forward_pipeline.kernel->executionSIMDGroups
            << " mortonOrder=" << (forward_pipeline.morton_order ? 1 : 0)
            << " lowPrecisionIntermediates=true"
            << '\n';
  std::cout << "backward-kernel"
            << " queryBlockR=" << backward_pipelines.query_kernel->blockDimensions[0]
            << " queryBlockC=" << backward_pipelines.query_kernel->blockDimensions[1]
            << " queryBlockD=" << backward_pipelines.query_kernel->blockDimensions[2]
            << " querySimdgroups=" << backward_pipelines.query_kernel->executionSIMDGroups
            << " queryBypassTG=" << (backward_pipelines.query_kernel->bypassThreadgroupMemory ? 1 : 0)
            << " keyvalueBlockR=" << backward_pipelines.keyvalue_kernel->blockDimensions[0]
            << " keyvalueBlockC=" << backward_pipelines.keyvalue_kernel->blockDimensions[1]
            << " keyvalueBlockD=" << backward_pipelines.keyvalue_kernel->blockDimensions[2]
            << " keyvalueSimdgroups=" << backward_pipelines.keyvalue_kernel->executionSIMDGroups
            << " keyvalueBypassTG=" << (backward_pipelines.keyvalue_kernel->bypassThreadgroupMemory ? 1 : 0)
            << " computeDThreads=" << backward_pipelines.compute_d_threads
            << " computeDCustom=" << (backward_pipelines.custom_compute_d ? 1 : 0)
            << '\n';
  print_stats("forward", forward_stats);
  print_stats("compute_d", compute_d_stats);
  print_stats("query", query_stats);
  print_stats("keyvalue", keyvalue_stats);
  print_stats("backward", backward_stats);
  std::cout << std::fixed
            << "ratio"
            << " avg=" << std::setprecision(4) << backward_stats.average_seconds / forward_stats.average_seconds
            << " median=" << backward_stats.median_seconds / forward_stats.median_seconds
            << '\n';

  pool->drain();
  return 0;
}
