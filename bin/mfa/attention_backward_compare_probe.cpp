#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernel.hpp"
#include "nnc/mfa/kernels/AttentionKernelDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionOperand.hpp"
#define private public
#include "nnc/mfa/kernels/NAAttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAAttentionKernel.hpp"
#include "nnc/mfa/kernels/NAAttentionKernelDescriptor.hpp"
#undef private

namespace {

using half_float = _Float16;

struct AttentionCase {
  uint32_t batch = 1;
  uint32_t R = 1536;
  uint32_t C = 1536;
  uint32_t Hq = 24;
  uint32_t Hk = 24;
  uint32_t D = 128;
  bool low_precision_inputs = true;
  bool is_bf16 = false;
  bool low_precision_intermediates = true;
  float q_scale = 0.03125f;
  float k_scale = 0.02734375f;
  float v_scale = 0.0234375f;
  float dO_scale = 0.01953125f;
};

struct NAForwardPipeline {
  NAAttentionDescriptor descriptor;
  std::unique_ptr<PipelineValue<NAAttentionKernel>> pipeline_value;
};

struct OldBackwardPipelines {
  AttentionDescriptor query_descriptor;
  AttentionDescriptor keyvalue_descriptor;
  std::unique_ptr<PipelineValue<AttentionKernel>> query_pipeline_value;
  std::unique_ptr<PipelineValue<AttentionKernel>> keyvalue_pipeline_value;
};

struct NewBackwardPipelines {
  NAAttentionDescriptor query_descriptor;
  NAAttentionDescriptor keyvalue_descriptor;
  NAAttentionKernelDescriptor query_kernel_descriptor = NAAttentionKernelDescriptor(simd::ushort3{0, 0, 0}, 0, 0, 0, 0, false, AttentionOperands<GEMMOperandPrecision>(), AttentionKernelType::forward, 0);
  NAAttentionKernelDescriptor keyvalue_kernel_descriptor = NAAttentionKernelDescriptor(simd::ushort3{0, 0, 0}, 0, 0, 0, 0, false, AttentionOperands<GEMMOperandPrecision>(), AttentionKernelType::forward, 0);
  std::unique_ptr<PipelineValue<NAAttentionKernel>> query_pipeline_value;
  std::unique_ptr<PipelineValue<NAAttentionKernel>> keyvalue_pipeline_value;
};

struct DiffStats {
  float max_abs = 0;
  size_t max_abs_index = 0;
  float ref_at_max_abs = 0;
  float test_at_max_abs = 0;
  float max_rel = 0;
  size_t max_rel_index = 0;
  float ref_at_max_rel = 0;
  float test_at_max_rel = 0;
};

struct TensorStats {
  float max_abs = 0;
  double mean_abs = 0;
  size_t above_1e4 = 0;
  size_t above_1e3 = 0;
};

struct HeadCompareStats {
  double ref_mean_abs = 0;
  double test_mean_abs = 0;
  double diff_mean_abs = 0;
};

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

float env_or_default(const char* name, float default_value)
{
  const char* value = std::getenv(name);
  if (!value)
    return default_value;
  return std::strtof(value, nullptr);
}

template <typename T>
MTL::Buffer* create_shared_buffer(MTL::Device* device, const std::vector<T>& values)
{
  return device->newBuffer(values.data(), values.size() * sizeof(T), kSharedResourceOptions);
}

std::vector<half_float> encode_fp16(const std::vector<float>& values)
{
  std::vector<half_float> encoded(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    encoded[i] = (half_float)values[i];
  return encoded;
}

MTL::Buffer* create_shared_buffer(MTL::Device* device, size_t size)
{
  auto* buffer = device->newBuffer(size, kSharedResourceOptions);
  std::memset(buffer->contents(), 0, size);
  return buffer;
}

std::vector<float> decode_fp16(MTL::Buffer* buffer, size_t count)
{
  const auto* src = reinterpret_cast<const half_float*>(buffer->contents());
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i)
    values[i] = (float)src[i];
  return values;
}

std::vector<float> decode_bf16(MTL::Buffer* buffer, size_t count)
{
  const auto* src = reinterpret_cast<const uint16_t*>(buffer->contents());
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    const uint32_t bits = (uint32_t)src[i] << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(float));
    values[i] = value;
  }
  return values;
}

std::vector<float> decode_float_buffer(MTL::Buffer* buffer, size_t offset_bytes, size_t count)
{
  const auto* src = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(buffer->contents()) + offset_bytes);
  return std::vector<float>(src, src + count);
}

std::vector<float> cast_fp32_to_fp16_then_decode(const std::vector<float>& src)
{
  std::vector<float> values(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    const half_float h = (half_float)src[i];
    values[i] = (float)h;
  }
  return values;
}

void zero_extra_heads(
    std::vector<float>& values,
    uint32_t batch,
    uint32_t sequence,
    uint32_t heads,
    uint32_t dimension)
{
  const size_t batch_stride = (size_t)sequence * heads * dimension;
  const size_t row_stride = (size_t)heads * dimension;
  for (uint32_t b = 0; b < batch; ++b)
    for (uint32_t r = 0; r < sequence; ++r)
      for (uint32_t h = 1; h < heads; ++h)
        for (uint32_t d = 0; d < dimension; ++d) {
          const size_t index = (size_t)b * batch_stride + (size_t)r * row_stride + (size_t)h * dimension + d;
          values[index] = 0;
        }
}

uint64_t ceil_divide(uint64_t x, uint64_t y)
{
  return (x + y - 1) / y;
}

std::string format_index(size_t index, uint32_t sequence, uint32_t heads, uint32_t dimension)
{
  const size_t d = index % dimension;
  index /= dimension;
  const size_t h = index % heads;
  index /= heads;
  const size_t row = index % sequence;
  index /= sequence;
  const size_t batch = index;
  return "batch=" + std::to_string(batch) +
      " row=" + std::to_string(row) +
      " head=" + std::to_string(h) +
      " d=" + std::to_string(d);
}

DiffStats compare_vectors(const std::vector<float>& reference, const std::vector<float>& test)
{
  DiffStats stats;
  for (size_t i = 0; i < reference.size(); ++i) {
    const float ref = reference[i];
    const float value = test[i];
    const float abs_diff = std::fabs(ref - value);
    const float denom = std::max(std::max(std::fabs(ref), std::fabs(value)), 1.0f);
    const float rel_diff = abs_diff / denom;
    if (abs_diff > stats.max_abs) {
      stats.max_abs = abs_diff;
      stats.max_abs_index = i;
      stats.ref_at_max_abs = ref;
      stats.test_at_max_abs = value;
    }
    if (rel_diff > stats.max_rel) {
      stats.max_rel = rel_diff;
      stats.max_rel_index = i;
      stats.ref_at_max_rel = ref;
      stats.test_at_max_rel = value;
    }
  }
  return stats;
}

TensorStats summarize_vector(const std::vector<float>& values)
{
  TensorStats stats;
  double total_abs = 0;
  for (const float value : values) {
    const float abs_value = std::fabs(value);
    stats.max_abs = std::max(stats.max_abs, abs_value);
    total_abs += abs_value;
    if (abs_value > 1e-4f)
      ++stats.above_1e4;
    if (abs_value > 1e-3f)
      ++stats.above_1e3;
  }
  stats.mean_abs = total_abs / values.size();
  return stats;
}

HeadCompareStats summarize_head_compare(
    const std::vector<float>& reference,
    const std::vector<float>& test,
    uint32_t batch,
    uint32_t sequence,
    uint32_t heads,
    uint32_t dimension,
    uint32_t target_head)
{
  HeadCompareStats stats;
  const size_t batch_stride = (size_t)sequence * heads * dimension;
  const size_t row_stride = (size_t)heads * dimension;
  size_t count = 0;
  for (uint32_t b = 0; b < batch; ++b)
    for (uint32_t r = 0; r < sequence; ++r)
      for (uint32_t d = 0; d < dimension; ++d) {
        const size_t index = (size_t)b * batch_stride + (size_t)r * row_stride + (size_t)target_head * dimension + d;
        const float ref = reference[index];
        const float value = test[index];
        stats.ref_mean_abs += std::fabs(ref);
        stats.test_mean_abs += std::fabs(value);
        stats.diff_mean_abs += std::fabs(ref - value);
        ++count;
      }
  if (count > 0) {
    stats.ref_mean_abs /= count;
    stats.test_mean_abs /= count;
    stats.diff_mean_abs /= count;
  }
  return stats;
}

std::vector<float> collapse_heads_sum(
    const std::vector<float>& source,
    uint32_t batch,
    uint32_t sequence,
    uint32_t heads,
    uint32_t dimension)
{
  std::vector<float> collapsed(source.size());
  const size_t batch_stride = (size_t)sequence * heads * dimension;
  const size_t row_stride = (size_t)heads * dimension;
  for (uint32_t b = 0; b < batch; ++b)
    for (uint32_t r = 0; r < sequence; ++r)
      for (uint32_t h = 0; h < heads; ++h)
        for (uint32_t d = 0; d < dimension; ++d) {
          float sum = 0;
          for (uint32_t src_h = 0; src_h < heads; ++src_h) {
            const size_t src_index = (size_t)b * batch_stride + (size_t)r * row_stride + (size_t)src_h * dimension + d;
            sum += source[src_index];
          }
          const size_t dst_index = (size_t)b * batch_stride + (size_t)r * row_stride + (size_t)h * dimension + d;
          collapsed[dst_index] = sum;
        }
  return collapsed;
}

void print_head_compare(
    const char* label,
    const std::vector<float>& reference,
    const std::vector<float>& test,
    const AttentionCase& attention)
{
  const uint32_t max_heads = std::min<uint32_t>(attention.Hq, 4);
  for (uint32_t h = 0; h < max_heads; ++h) {
    const auto stats = summarize_head_compare(reference, test, attention.batch, attention.R, attention.Hq, attention.D, h);
    std::cout << label
              << " head=" << h
              << " ref_mean_abs=" << std::fixed << std::setprecision(7) << stats.ref_mean_abs
              << " test_mean_abs=" << stats.test_mean_abs
              << " diff_mean_abs=" << stats.diff_mean_abs
              << '\n';
  }
}

std::vector<float> extract_head(
    const std::vector<float>& source,
    uint32_t batch,
    uint32_t sequence,
    uint32_t heads,
    uint32_t dimension,
    uint32_t target_head)
{
  std::vector<float> extracted((size_t)batch * sequence * dimension);
  const size_t batch_stride = (size_t)sequence * heads * dimension;
  const size_t row_stride = (size_t)heads * dimension;
  size_t dst = 0;
  for (uint32_t b = 0; b < batch; ++b)
    for (uint32_t r = 0; r < sequence; ++r)
      for (uint32_t d = 0; d < dimension; ++d) {
        const size_t src = (size_t)b * batch_stride + (size_t)r * row_stride + (size_t)target_head * dimension + d;
        extracted[dst++] = source[src];
      }
  return extracted;
}

std::vector<float> head_row_to_row_head(
    const std::vector<float>& source,
    uint32_t batch,
    uint32_t sequence,
    uint32_t heads,
    uint32_t dimension)
{
  std::vector<float> reordered(source.size());
  for (uint32_t b = 0; b < batch; ++b)
    for (uint32_t h = 0; h < heads; ++h)
      for (uint32_t r = 0; r < sequence; ++r)
        for (uint32_t d = 0; d < dimension; ++d) {
          const size_t src_index = ((((size_t)b * heads + h) * sequence + r) * dimension + d);
          const size_t dst_index = ((((size_t)b * sequence + r) * heads + h) * dimension + d);
          reordered[dst_index] = source[src_index];
        }
  return reordered;
}

void print_diff(
    const char* label,
    const DiffStats& stats,
    uint32_t sequence,
    uint32_t heads,
    uint32_t dimension)
{
  std::cout << std::fixed
            << label
            << " max_abs=" << std::setprecision(7) << stats.max_abs
            << " (" << format_index(stats.max_abs_index, sequence, heads, dimension)
            << " ref=" << stats.ref_at_max_abs
            << " test=" << stats.test_at_max_abs << ")"
            << " max_rel=" << stats.max_rel
            << " (" << format_index(stats.max_rel_index, sequence, heads, dimension)
            << " ref=" << stats.ref_at_max_rel
            << " test=" << stats.test_at_max_rel << ")"
            << '\n';
}

void print_tensor_stats(const char* label, const TensorStats& stats, size_t total_count)
{
  std::cout << std::fixed
            << label
            << " max_abs=" << std::setprecision(7) << stats.max_abs
            << " mean_abs=" << stats.mean_abs
            << " nz_1e-4=" << stats.above_1e4 << "/" << total_count
            << " nz_1e-3=" << stats.above_1e3 << "/" << total_count
            << '\n';
}

AttentionOperands<unsigned int> create_batch_strides(const AttentionCase& attention)
{
  AttentionOperands<unsigned int> strides;
  if (attention.batch > 1) {
    strides[AttentionOperand::Q] = attention.R * attention.D * attention.Hq;
    strides[AttentionOperand::K] = attention.C * attention.D * attention.Hk;
    strides[AttentionOperand::V] = attention.C * attention.D * attention.Hk;
    strides[AttentionOperand::O] = attention.R * attention.D * attention.Hq;
    strides[AttentionOperand::dO] = attention.R * attention.D * attention.Hq;
    strides[AttentionOperand::dQ] = attention.R * attention.D * attention.Hq;
    strides[AttentionOperand::dK] = attention.C * attention.D * attention.Hk;
    strides[AttentionOperand::dV] = attention.C * attention.D * attention.Hk;
  }
  return strides;
}

NAForwardPipeline create_na_forward_pipeline(MTL::Device* device, const DeviceProperties& dprops, const AttentionCase& attention)
{
  NAAttentionDescriptor descriptor;
  descriptor.batchDimension = attention.batch;
  descriptor.Hq = attention.Hq;
  descriptor.Hk = attention.Hk;
  descriptor.lowPrecisionInputs = attention.low_precision_inputs;
  descriptor.isBF16 = attention.is_bf16;
  descriptor.lowPrecisionIntermediates = attention.low_precision_intermediates;
  descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  descriptor.batchStrides = create_batch_strides(attention);
  descriptor.type = AttentionKernelType::forward;
  descriptor.scale = create_scale(attention);

  static std::unordered_map<NAAttentionKernelDescriptor, std::unique_ptr<NAAttentionKernel>> cache;
  auto kernel_pair = descriptor.findKernel(device, dprops, nullptr, nullptr, "", &cache);
  auto pipeline_value = std::unique_ptr<PipelineValue<NAAttentionKernel>>(kernel_pair.second);
  return NAForwardPipeline { descriptor, std::move(pipeline_value) };
}

OldBackwardPipelines create_old_backward_pipelines(MTL::Device* device, const DeviceProperties& dprops, const AttentionCase& attention)
{
  OldBackwardPipelines pipelines;
  AttentionDescriptor descriptor;
  descriptor.batchDimension = attention.batch;
  descriptor.Hq = attention.Hq;
  descriptor.Hk = attention.Hk;
  descriptor.lowPrecisionInputs = attention.low_precision_inputs;
  descriptor.isBF16 = attention.is_bf16;
  descriptor.lowPrecisionIntermediates = attention.low_precision_intermediates;
  descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  descriptor.transposeState = simd::uchar4 { 0, 0, 0, 0 };
  descriptor.batchStrides = create_batch_strides(attention);
  descriptor.leadingDimensions = simd::uint4 {
    attention.Hq * attention.D,
    attention.Hk * attention.D,
    attention.Hk * attention.D,
    attention.Hq * attention.D,
  };
  descriptor.scale = create_scale(attention);

  static std::unordered_map<AttentionKernelDescriptor, std::unique_ptr<AttentionKernel>> cache;

  pipelines.query_descriptor = descriptor;
  pipelines.query_descriptor.type = AttentionKernelType::backwardQuery;
  auto query_pair = pipelines.query_descriptor.findKernel(device, dprops, nullptr, nullptr, "", &cache);
  pipelines.query_pipeline_value.reset(query_pair.second);

  pipelines.keyvalue_descriptor = descriptor;
  pipelines.keyvalue_descriptor.type = AttentionKernelType::backwardKeyValue;
  auto keyvalue_pair = pipelines.keyvalue_descriptor.findKernel(device, dprops, nullptr, nullptr, "", &cache);
  pipelines.keyvalue_pipeline_value.reset(keyvalue_pair.second);

  return pipelines;
}

std::unique_ptr<PipelineValue<NAAttentionKernel>> create_new_pipeline_value(
    MTL::Device* device,
    const NAAttentionDescriptor& descriptor,
    const NAAttentionKernelDescriptor& kernel_descriptor,
    std::unordered_map<NAAttentionKernelDescriptor, std::unique_ptr<NAAttentionKernel>>& cache)
{
  auto cache_it = cache.find(kernel_descriptor);
  NAAttentionKernel* kernel = nullptr;
  if (cache_it != cache.end()) {
    kernel = cache_it->second.get();
  } else {
    auto owned = std::make_unique<NAAttentionKernel>(kernel_descriptor, device);
    kernel = owned.get();
    cache.emplace(kernel_descriptor, std::move(owned));
  }

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  uint32_t row_dimension = descriptor.matrixDimensions[0];
  uint32_t column_dimension = descriptor.matrixDimensions[1];
  constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::Integer(0));
  constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, 1);

  std::vector<AttentionOperand> operands;
  switch (descriptor.type.value) {
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
  for (const auto& operand : operands) {
    const uint32_t batch_stride = descriptor.batchStrides[operand].value_or(0);
    constants->setConstantValue(&batch_stride, MTL::DataTypeUInt, 2 + operand.bufferIndex());
  }

  NS::Error* error = nil;
  auto pipeline_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  auto attention_name = NS::String::string("attention", NS::UTF8StringEncoding);
  pipeline_desc->setComputeFunction(NS::TransferPtr(kernel->library->newFunction(attention_name, constants.get(), &error)).get());
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_desc.get(), MTL::PipelineOptionNone, NULL, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  NS::SharedPtr<MTL::ComputePipelineState> second;
  if (descriptor.type.value == AttentionKernelType::backwardQuery) {
    auto compute_d_constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t o_batch_stride = descriptor.batchStrides[AttentionOperand::O].value_or(0);
    const uint32_t dO_batch_stride = descriptor.batchStrides[AttentionOperand::dO].value_or(0);
    compute_d_constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::Integer(0));
    compute_d_constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, 1);
    compute_d_constants->setConstantValue(&o_batch_stride, MTL::DataTypeUInt, 2 + AttentionOperand(AttentionOperand::O).bufferIndex());
    compute_d_constants->setConstantValue(&dO_batch_stride, MTL::DataTypeUInt, 2 + AttentionOperand(AttentionOperand::dO).bufferIndex());
    auto compute_d_name = NS::String::string("compute_d", NS::UTF8StringEncoding);
    auto compute_d_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    compute_d_desc->setComputeFunction(NS::TransferPtr(kernel->library->newFunction(compute_d_name, compute_d_constants.get(), &error)).get());
    CCV_NNC_MFA_CHECK_ERROR(error);
    second = NS::TransferPtr(device->newComputePipelineState(compute_d_desc.get(), MTL::PipelineOptionNone, NULL, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }

  auto output = std::make_unique<PipelineValue<NAAttentionKernel>>(PipelineValue<NAAttentionKernel>{ kernel, pipeline });
  output->second = second;
  return output;
}

NewBackwardPipelines create_new_backward_pipelines(
    MTL::Device* device,
    const DeviceProperties& dprops,
    const AttentionCase& attention,
    int query_bypass_override = -1,
    int keyvalue_bypass_override = -1)
{
  NewBackwardPipelines pipelines;
  NAAttentionDescriptor descriptor;
  descriptor.batchDimension = attention.batch;
  descriptor.Hq = attention.Hq;
  descriptor.Hk = attention.Hk;
  descriptor.lowPrecisionInputs = attention.low_precision_inputs;
  descriptor.isBF16 = attention.is_bf16;
  descriptor.lowPrecisionIntermediates = attention.low_precision_intermediates;
  descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  descriptor.batchStrides = create_batch_strides(attention);
  descriptor.scale = create_scale(attention);
  static std::unordered_map<NAAttentionKernelDescriptor, std::unique_ptr<NAAttentionKernel>> cache;

  pipelines.query_descriptor = descriptor;
  pipelines.query_descriptor.type = AttentionKernelType::backwardQuery;
  pipelines.query_kernel_descriptor = pipelines.query_descriptor.kernelDescriptor(device, dprops);
  if (query_bypass_override != -1)
    pipelines.query_kernel_descriptor.bypassThreadgroupMemory = (query_bypass_override != 0);
  pipelines.query_pipeline_value = create_new_pipeline_value(device, pipelines.query_descriptor, pipelines.query_kernel_descriptor, cache);

  pipelines.keyvalue_descriptor = descriptor;
  pipelines.keyvalue_descriptor.type = AttentionKernelType::backwardKeyValue;
  pipelines.keyvalue_kernel_descriptor = pipelines.keyvalue_descriptor.kernelDescriptor(device, dprops);
  if (keyvalue_bypass_override != -1)
    pipelines.keyvalue_kernel_descriptor.bypassThreadgroupMemory = (keyvalue_bypass_override != 0);
  pipelines.keyvalue_pipeline_value = create_new_pipeline_value(device, pipelines.keyvalue_descriptor, pipelines.keyvalue_kernel_descriptor, cache);

  return pipelines;
}

void zero_buffer(MTL::Buffer* buffer)
{
  std::memset(buffer->contents(), 0, buffer->length());
}

void run_na_forward_once(
    MTL::CommandQueue* command_queue,
    const NAForwardPipeline& forward,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  auto* kernel = forward.pipeline_value->kernel;
  auto* pipeline = forward.pipeline_value->pipeline.get();
  encoder->setComputePipelineState(pipeline);
  encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(pipeline, forward.descriptor), 0);
  encoder->setBuffer(q_buffer, 0, AttentionOperand(AttentionOperand::Q).bufferIndex());
  encoder->setBuffer(k_buffer, 0, AttentionOperand(AttentionOperand::K).bufferIndex());
  encoder->setBuffer(v_buffer, 0, AttentionOperand(AttentionOperand::V).bufferIndex());
  encoder->setBuffer(o_buffer, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
  encoder->setBuffer(l_buffer, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
  encoder->dispatchThreadgroups(
      kernel->threadgroupsPerGrid(forward.descriptor),
      MTL::Size(kernel->threadgroupSize(pipeline, forward.descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void run_old_backward_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const OldBackwardPipelines& old_pipelines,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* old_scratch)
{
  const size_t dQ_offset = 0;
  const size_t dK_offset = sizeof(float) * (size_t)attention.R * attention.D * attention.Hq * attention.batch;
  const size_t dV_offset = sizeof(float) * (size_t)(attention.R + attention.C) * attention.D * attention.Hq * attention.batch;
  const size_t d_offset = sizeof(float) * (size_t)(attention.R + attention.C * 2) * attention.D * attention.Hq * attention.batch;

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    auto* kernel = old_pipelines.query_pipeline_value->kernel;
    auto* pipeline = old_pipelines.query_pipeline_value->pipeline.get();
    encoder->setComputePipelineState(pipeline);
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    encoder->setBuffer(q_buffer, 0, AttentionOperand(AttentionOperand::Q).bufferIndex());
    encoder->setBuffer(k_buffer, 0, AttentionOperand(AttentionOperand::K).bufferIndex());
    encoder->setBuffer(v_buffer, 0, AttentionOperand(AttentionOperand::V).bufferIndex());
    encoder->setBuffer(o_buffer, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
    encoder->setBuffer(l_buffer, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
    encoder->setBuffer(dO_buffer, 0, AttentionOperand(AttentionOperand::dO).bufferIndex());
    encoder->setBuffer(old_scratch, dQ_offset, AttentionOperand(AttentionOperand::dQ).bufferIndex());
    encoder->setBuffer(old_scratch, d_offset, AttentionOperand(AttentionOperand::D).bufferIndex());
    encoder->dispatchThreadgroups(
        MTL::Size(ceil_divide(attention.R, kernel->blockDimensions[0]) * attention.Hq * attention.batch, 1, 1),
        MTL::Size(kernel->threadgroupSize, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    auto* kernel = old_pipelines.keyvalue_pipeline_value->kernel;
    auto* pipeline = old_pipelines.keyvalue_pipeline_value->pipeline.get();
    encoder->setComputePipelineState(pipeline);
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    encoder->setBuffer(q_buffer, 0, AttentionOperand(AttentionOperand::Q).bufferIndex());
    encoder->setBuffer(k_buffer, 0, AttentionOperand(AttentionOperand::K).bufferIndex());
    encoder->setBuffer(v_buffer, 0, AttentionOperand(AttentionOperand::V).bufferIndex());
    encoder->setBuffer(o_buffer, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
    encoder->setBuffer(l_buffer, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
    encoder->setBuffer(dO_buffer, 0, AttentionOperand(AttentionOperand::dO).bufferIndex());
    encoder->setBuffer(old_scratch, dV_offset, AttentionOperand(AttentionOperand::dV).bufferIndex());
    encoder->setBuffer(old_scratch, dK_offset, AttentionOperand(AttentionOperand::dK).bufferIndex());
    encoder->setBuffer(old_scratch, d_offset, AttentionOperand(AttentionOperand::D).bufferIndex());
    encoder->dispatchThreadgroups(
        MTL::Size(ceil_divide(attention.C, kernel->blockDimensions[0]) * attention.Hq * attention.batch, 1, 1),
        MTL::Size(kernel->threadgroupSize, 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void run_new_backward_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const NewBackwardPipelines& new_pipelines,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* new_dQ_buffer,
    MTL::Buffer* new_dK_buffer,
    MTL::Buffer* new_dV_buffer,
    MTL::Buffer* new_d_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(new_pipelines.query_pipeline_value->second.get());
    encoder->setBuffer(o_buffer, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
    encoder->setBuffer(dO_buffer, 0, AttentionOperand(AttentionOperand::dO).bufferIndex());
    encoder->setBuffer(new_d_buffer, 0, AttentionOperand(AttentionOperand::D).bufferIndex());
    encoder->dispatchThreadgroups(
        MTL::Size((uint64_t)attention.R * attention.Hq, 1, attention.batch),
        MTL::Size(NAAttentionKernel::computeDThreads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    auto* kernel = new_pipelines.query_pipeline_value->kernel;
    auto* pipeline = new_pipelines.query_pipeline_value->pipeline.get();
    encoder->setComputePipelineState(pipeline);
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(pipeline, new_pipelines.query_descriptor), 0);
    encoder->setBuffer(q_buffer, 0, AttentionOperand(AttentionOperand::Q).bufferIndex());
    encoder->setBuffer(k_buffer, 0, AttentionOperand(AttentionOperand::K).bufferIndex());
    encoder->setBuffer(v_buffer, 0, AttentionOperand(AttentionOperand::V).bufferIndex());
    encoder->setBuffer(l_buffer, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
    encoder->setBuffer(new_d_buffer, 0, AttentionOperand(AttentionOperand::D).bufferIndex());
    encoder->setBuffer(dO_buffer, 0, AttentionOperand(AttentionOperand::dO).bufferIndex());
    encoder->setBuffer(new_dQ_buffer, 0, AttentionOperand(AttentionOperand::dQ).bufferIndex());
    encoder->dispatchThreadgroups(
        kernel->threadgroupsPerGrid(new_pipelines.query_descriptor),
        MTL::Size(kernel->threadgroupSize(pipeline, new_pipelines.query_descriptor), 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    auto* kernel = new_pipelines.keyvalue_pipeline_value->kernel;
    auto* pipeline = new_pipelines.keyvalue_pipeline_value->pipeline.get();
    encoder->setComputePipelineState(pipeline);
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(pipeline, new_pipelines.keyvalue_descriptor), 0);
    encoder->setBuffer(q_buffer, 0, AttentionOperand(AttentionOperand::Q).bufferIndex());
    encoder->setBuffer(k_buffer, 0, AttentionOperand(AttentionOperand::K).bufferIndex());
    encoder->setBuffer(v_buffer, 0, AttentionOperand(AttentionOperand::V).bufferIndex());
    encoder->setBuffer(l_buffer, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
    encoder->setBuffer(new_d_buffer, 0, AttentionOperand(AttentionOperand::D).bufferIndex());
    encoder->setBuffer(dO_buffer, 0, AttentionOperand(AttentionOperand::dO).bufferIndex());
    encoder->setBuffer(new_dV_buffer, 0, AttentionOperand(AttentionOperand::dV).bufferIndex());
    encoder->setBuffer(new_dK_buffer, 0, AttentionOperand(AttentionOperand::dK).bufferIndex());
    encoder->dispatchThreadgroups(
        kernel->threadgroupsPerGrid(new_pipelines.keyvalue_descriptor),
        MTL::Size(kernel->threadgroupSize(pipeline, new_pipelines.keyvalue_descriptor), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  AttentionCase attention;
  if (argc >= 6) {
    attention.R = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    attention.C = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    attention.D = (uint32_t)std::strtoul(argv[3], nullptr, 10);
    attention.Hq = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    attention.Hk = (uint32_t)std::strtoul(argv[5], nullptr, 10);
  }
  if (argc >= 7)
    attention.low_precision_inputs = std::atoi(argv[6]) != 0;
  if (argc >= 8)
    attention.low_precision_intermediates = std::atoi(argv[7]) != 0;
  if (argc >= 9)
    attention.is_bf16 = std::atoi(argv[8]) != 0;
  if (argc >= 10)
    attention.q_scale = std::strtof(argv[9], nullptr);
  if (argc >= 11)
    attention.k_scale = std::strtof(argv[10], nullptr);
  if (argc >= 12)
    attention.v_scale = std::strtof(argv[11], nullptr);
  if (argc >= 13)
    attention.dO_scale = std::strtof(argv[12], nullptr);
  int query_bypass_override = -1;
  int keyvalue_bypass_override = -1;
  if (argc >= 14)
    query_bypass_override = std::atoi(argv[13]);
  if (argc >= 15)
    keyvalue_bypass_override = std::atoi(argv[14]);

  attention.q_scale = env_or_default("CCV_MFA_PROBE_Q_SCALE", attention.q_scale);
  attention.k_scale = env_or_default("CCV_MFA_PROBE_K_SCALE", attention.k_scale);
  attention.v_scale = env_or_default("CCV_MFA_PROBE_V_SCALE", attention.v_scale);
  attention.dO_scale = env_or_default("CCV_MFA_PROBE_DO_SCALE", attention.dO_scale);
  const float global_scale = env_or_default("CCV_MFA_PROBE_GLOBAL_SCALE", 1.0f);
  attention.q_scale *= global_scale;
  attention.k_scale *= global_scale;
  attention.v_scale *= global_scale;
  attention.dO_scale *= global_scale;

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

  DeviceProperties dprops = DeviceProperties();
  auto q_values = make_data<float>((size_t)attention.batch * attention.R * attention.Hq * attention.D, attention.q_scale, 1);
  auto k_values = make_data<float>((size_t)attention.batch * attention.C * attention.Hk * attention.D, attention.k_scale, 2);
  auto v_values = make_data<float>((size_t)attention.batch * attention.C * attention.Hk * attention.D, attention.v_scale, 3);
  auto dO_values = make_data<float>((size_t)attention.batch * attention.R * attention.Hq * attention.D, attention.dO_scale, 4);
  if (std::getenv("CCV_MFA_PROBE_ZERO_EXTRA_HEADS")) {
    zero_extra_heads(q_values, attention.batch, attention.R, attention.Hq, attention.D);
    zero_extra_heads(k_values, attention.batch, attention.C, attention.Hk, attention.D);
    zero_extra_heads(v_values, attention.batch, attention.C, attention.Hk, attention.D);
    zero_extra_heads(dO_values, attention.batch, attention.R, attention.Hq, attention.D);
  }
  NS::SharedPtr<MTL::Buffer> q_buffer;
  NS::SharedPtr<MTL::Buffer> k_buffer;
  NS::SharedPtr<MTL::Buffer> v_buffer;
  NS::SharedPtr<MTL::Buffer> dO_buffer;
  if (attention.low_precision_inputs) {
    const auto q_fp16 = encode_fp16(q_values);
    const auto k_fp16 = encode_fp16(k_values);
    const auto v_fp16 = encode_fp16(v_values);
    const auto dO_fp16 = encode_fp16(dO_values);
    q_buffer = NS::TransferPtr(create_shared_buffer(device.get(), q_fp16));
    k_buffer = NS::TransferPtr(create_shared_buffer(device.get(), k_fp16));
    v_buffer = NS::TransferPtr(create_shared_buffer(device.get(), v_fp16));
    dO_buffer = NS::TransferPtr(create_shared_buffer(device.get(), dO_fp16));
  } else {
    q_buffer = NS::TransferPtr(create_shared_buffer(device.get(), q_values));
    k_buffer = NS::TransferPtr(create_shared_buffer(device.get(), k_values));
    v_buffer = NS::TransferPtr(create_shared_buffer(device.get(), v_values));
    dO_buffer = NS::TransferPtr(create_shared_buffer(device.get(), dO_values));
  }

  const size_t o_count = (size_t)attention.batch * attention.R * attention.Hq * attention.D;
  const size_t l_count = (size_t)attention.batch * attention.R * attention.Hq;
  const size_t kv_count = (size_t)attention.batch * attention.C * attention.Hk * attention.D;
  const size_t old_output_count = (size_t)(attention.R + attention.C * 2) * attention.D * attention.Hq * attention.batch;
  const size_t io_bytes = attention.low_precision_inputs ? sizeof(half_float) : sizeof(float);
  const size_t l_bytes = attention.low_precision_intermediates ? (attention.is_bf16 ? sizeof(uint16_t) : (attention.low_precision_inputs ? sizeof(half_float) : sizeof(float))) : sizeof(float);
  auto o_buffer = NS::TransferPtr(create_shared_buffer(device.get(), o_count * io_bytes));
  auto l_buffer = NS::TransferPtr(create_shared_buffer(device.get(), l_count * l_bytes));
  auto old_scratch = NS::TransferPtr(create_shared_buffer(
      device.get(),
      sizeof(float) * old_output_count + sizeof(float) * l_count));
  auto new_dQ_buffer = NS::TransferPtr(create_shared_buffer(device.get(), o_count * io_bytes));
  auto new_dK_buffer = NS::TransferPtr(create_shared_buffer(device.get(), kv_count * io_bytes));
  auto new_dV_buffer = NS::TransferPtr(create_shared_buffer(device.get(), kv_count * io_bytes));
  auto new_d_buffer = NS::TransferPtr(create_shared_buffer(device.get(), l_count * sizeof(float)));

  const auto forward = create_na_forward_pipeline(device.get(), dprops, attention);
  const auto old_backward = create_old_backward_pipelines(device.get(), dprops, attention);
  const auto new_backward = create_new_backward_pipelines(
      device.get(), dprops, attention, query_bypass_override, keyvalue_bypass_override);

  run_na_forward_once(
      command_queue.get(),
      forward,
      q_buffer.get(),
      k_buffer.get(),
      v_buffer.get(),
      o_buffer.get(),
      l_buffer.get());

  zero_buffer(old_scratch.get());
  run_old_backward_once(
      command_queue.get(),
      attention,
      old_backward,
      q_buffer.get(),
      k_buffer.get(),
      v_buffer.get(),
      o_buffer.get(),
      l_buffer.get(),
      dO_buffer.get(),
      old_scratch.get());

  zero_buffer(new_dQ_buffer.get());
  zero_buffer(new_dK_buffer.get());
  zero_buffer(new_dV_buffer.get());
  zero_buffer(new_d_buffer.get());
  run_new_backward_once(
      command_queue.get(),
      attention,
      new_backward,
      q_buffer.get(),
      k_buffer.get(),
      v_buffer.get(),
      o_buffer.get(),
      l_buffer.get(),
      dO_buffer.get(),
      new_dQ_buffer.get(),
      new_dK_buffer.get(),
      new_dV_buffer.get(),
      new_d_buffer.get());

  const size_t old_dQ_offset = 0;
  const size_t old_dK_offset = sizeof(float) * (size_t)attention.R * attention.D * attention.Hq * attention.batch;
  const size_t old_dV_offset = sizeof(float) * (size_t)(attention.R + attention.C) * attention.D * attention.Hq * attention.batch;

  const auto old_dQ = decode_float_buffer(old_scratch.get(), old_dQ_offset, o_count);
  const auto old_dK = decode_float_buffer(old_scratch.get(), old_dK_offset, kv_count);
  const auto old_dV = decode_float_buffer(old_scratch.get(), old_dV_offset, kv_count);
  const auto new_dQ = attention.low_precision_inputs ? decode_fp16(new_dQ_buffer.get(), o_count) : decode_float_buffer(new_dQ_buffer.get(), 0, o_count);
  const auto new_dK = attention.low_precision_inputs ? decode_fp16(new_dK_buffer.get(), kv_count) : decode_float_buffer(new_dK_buffer.get(), 0, kv_count);
  const auto new_dV = attention.low_precision_inputs ? decode_fp16(new_dV_buffer.get(), kv_count) : decode_float_buffer(new_dV_buffer.get(), 0, kv_count);
  const auto old_D = decode_float_buffer(old_scratch.get(), sizeof(float) * old_output_count, l_count);
  const auto new_D = decode_float_buffer(new_d_buffer.get(), 0, l_count);

  const auto old_dQ_fp16 = cast_fp32_to_fp16_then_decode(old_dQ);
  const auto old_dK_fp16 = cast_fp32_to_fp16_then_decode(old_dK);
  const auto old_dV_fp16 = cast_fp32_to_fp16_then_decode(old_dV);
  const auto old_dQ_headsum = collapse_heads_sum(old_dQ, attention.batch, attention.R, attention.Hq, attention.D);
  const auto old_dK_headsum = collapse_heads_sum(old_dK, attention.batch, attention.C, attention.Hk, attention.D);
  const auto old_dV_headsum = collapse_heads_sum(old_dV, attention.batch, attention.C, attention.Hk, attention.D);

  const auto dQ_raw = compare_vectors(old_dQ, new_dQ);
  const auto dK_raw = compare_vectors(old_dK, new_dK);
  const auto dV_raw = compare_vectors(old_dV, new_dV);
  const auto dQ_headsum = compare_vectors(old_dQ_headsum, new_dQ);
  const auto dK_headsum = compare_vectors(old_dK_headsum, new_dK);
  const auto dV_headsum = compare_vectors(old_dV_headsum, new_dV);
  const auto dQ_head_row_layout = compare_vectors(old_dQ, head_row_to_row_head(new_dQ, attention.batch, attention.R, attention.Hq, attention.D));
  const auto dK_head_row_layout = compare_vectors(old_dK, head_row_to_row_head(new_dK, attention.batch, attention.C, attention.Hk, attention.D));
  const auto dV_head_row_layout = compare_vectors(old_dV, head_row_to_row_head(new_dV, attention.batch, attention.C, attention.Hk, attention.D));
  const auto dQ_cast = compare_vectors(old_dQ_fp16, new_dQ);
  const auto dK_cast = compare_vectors(old_dK_fp16, new_dK);
  const auto dV_cast = compare_vectors(old_dV_fp16, new_dV);
  const auto old_dQ_stats = summarize_vector(old_dQ_fp16);
  const auto old_dK_stats = summarize_vector(old_dK_fp16);
  const auto old_dV_stats = summarize_vector(old_dV_fp16);
  const auto new_dQ_stats = summarize_vector(new_dQ);
  const auto new_dK_stats = summarize_vector(new_dK);
  const auto new_dV_stats = summarize_vector(new_dV);
  const auto D_diff = compare_vectors(old_D, new_D);
  const auto old_D_stats = summarize_vector(old_D);
  const auto new_D_stats = summarize_vector(new_D);

  std::cout << "shape"
            << " B=" << attention.batch
            << " R=" << attention.R
            << " C=" << attention.C
            << " Hq=" << attention.Hq
            << " Hk=" << attention.Hk
            << " D=" << attention.D
            << " io=" << (attention.low_precision_inputs ? (attention.is_bf16 ? "bf16" : "fp16") : "fp32")
            << " lowPrecisionIntermediates=" << (attention.low_precision_intermediates ? 1 : 0)
            << " qScale=" << attention.q_scale
            << " kScale=" << attention.k_scale
            << " vScale=" << attention.v_scale
            << " dOScale=" << attention.dO_scale
            << '\n';

  std::cout << "na-forward"
            << " blockR=" << forward.pipeline_value->kernel->blockDimensions[0]
            << " blockC=" << forward.pipeline_value->kernel->blockDimensions[1]
            << " blockD=" << forward.pipeline_value->kernel->blockDimensions[2]
            << " simdgroups=" << forward.pipeline_value->kernel->executionSIMDGroups
            << '\n';

  std::cout << "old-backward"
            << " queryBlockR=" << old_backward.query_pipeline_value->kernel->blockDimensions[0]
            << " queryBlockC=" << old_backward.query_pipeline_value->kernel->blockDimensions[1]
            << " queryBlockD=" << old_backward.query_pipeline_value->kernel->blockDimensions[2]
            << " queryThreads=" << old_backward.query_pipeline_value->kernel->threadgroupSize
            << " keyvalueBlockR=" << old_backward.keyvalue_pipeline_value->kernel->blockDimensions[0]
            << " keyvalueBlockC=" << old_backward.keyvalue_pipeline_value->kernel->blockDimensions[1]
            << " keyvalueBlockD=" << old_backward.keyvalue_pipeline_value->kernel->blockDimensions[2]
            << " keyvalueThreads=" << old_backward.keyvalue_pipeline_value->kernel->threadgroupSize
            << '\n';

  std::cout << "new-backward"
            << " queryDescBlockD=" << new_backward.query_kernel_descriptor.blockDimensions[2]
            << " queryBlockR=" << new_backward.query_pipeline_value->kernel->blockDimensions[0]
            << " queryBlockC=" << new_backward.query_pipeline_value->kernel->blockDimensions[1]
            << " queryBlockD=" << new_backward.query_pipeline_value->kernel->blockDimensions[2]
            << " querySimdgroups=" << new_backward.query_pipeline_value->kernel->executionSIMDGroups
            << " keyvalueDescBlockD=" << new_backward.keyvalue_kernel_descriptor.blockDimensions[2]
            << " keyvalueBlockR=" << new_backward.keyvalue_pipeline_value->kernel->blockDimensions[0]
            << " keyvalueBlockC=" << new_backward.keyvalue_pipeline_value->kernel->blockDimensions[1]
            << " keyvalueBlockD=" << new_backward.keyvalue_pipeline_value->kernel->blockDimensions[2]
            << " keyvalueSimdgroups=" << new_backward.keyvalue_pipeline_value->kernel->executionSIMDGroups
            << '\n';

  print_diff("dQ raw", dQ_raw, attention.R, attention.Hq, attention.D);
  print_diff("dQ old_headsum", dQ_headsum, attention.R, attention.Hq, attention.D);
  print_diff("dQ new_headrow->rowhead", dQ_head_row_layout, attention.R, attention.Hq, attention.D);
  print_diff("dQ old->fp16", dQ_cast, attention.R, attention.Hq, attention.D);
  print_tensor_stats("dQ old_fp16_stats", old_dQ_stats, o_count);
  print_tensor_stats("dQ new_stats", new_dQ_stats, o_count);
  print_head_compare("dQ head_compare", old_dQ, new_dQ, attention);
  print_diff("dK raw", dK_raw, attention.C, attention.Hk, attention.D);
  print_diff("dK old_headsum", dK_headsum, attention.C, attention.Hk, attention.D);
  print_diff("dK new_headrow->rowhead", dK_head_row_layout, attention.C, attention.Hk, attention.D);
  print_diff("dK old->fp16", dK_cast, attention.C, attention.Hk, attention.D);
  print_tensor_stats("dK old_fp16_stats", old_dK_stats, kv_count);
  print_tensor_stats("dK new_stats", new_dK_stats, kv_count);
  print_head_compare("dK head_compare", old_dK, new_dK, attention);
  print_diff("dV raw", dV_raw, attention.C, attention.Hk, attention.D);
  print_diff("dV old_headsum", dV_headsum, attention.C, attention.Hk, attention.D);
  print_diff("dV new_headrow->rowhead", dV_head_row_layout, attention.C, attention.Hk, attention.D);
  print_diff("dV old->fp16", dV_cast, attention.C, attention.Hk, attention.D);
  print_tensor_stats("dV old_fp16_stats", old_dV_stats, kv_count);
  print_tensor_stats("dV new_stats", new_dV_stats, kv_count);
  print_head_compare("dV head_compare", old_dV, new_dV, attention);
  print_diff("D", D_diff, attention.R, attention.Hq, 1);
  print_tensor_stats("D old_stats", old_D_stats, l_count);
  print_tensor_stats("D new_stats", new_D_stats, l_count);
  if (attention.Hq == 2 && attention.Hk == 2) {
    const auto old_dQ_h0 = extract_head(old_dQ, attention.batch, attention.R, attention.Hq, attention.D, 0);
    const auto old_dQ_h1 = extract_head(old_dQ, attention.batch, attention.R, attention.Hq, attention.D, 1);
    const auto new_dQ_h0 = extract_head(new_dQ, attention.batch, attention.R, attention.Hq, attention.D, 0);
    const auto new_dQ_h1 = extract_head(new_dQ, attention.batch, attention.R, attention.Hq, attention.D, 1);
    const auto old_dK_h0 = extract_head(old_dK, attention.batch, attention.C, attention.Hk, attention.D, 0);
    const auto old_dK_h1 = extract_head(old_dK, attention.batch, attention.C, attention.Hk, attention.D, 1);
    const auto new_dK_h0 = extract_head(new_dK, attention.batch, attention.C, attention.Hk, attention.D, 0);
    const auto new_dK_h1 = extract_head(new_dK, attention.batch, attention.C, attention.Hk, attention.D, 1);
    const auto old_dV_h0 = extract_head(old_dV, attention.batch, attention.C, attention.Hk, attention.D, 0);
    const auto old_dV_h1 = extract_head(old_dV, attention.batch, attention.C, attention.Hk, attention.D, 1);
    const auto new_dV_h0 = extract_head(new_dV, attention.batch, attention.C, attention.Hk, attention.D, 0);
    const auto new_dV_h1 = extract_head(new_dV, attention.batch, attention.C, attention.Hk, attention.D, 1);
    print_diff("dQ new_h0_vs_new_h1", compare_vectors(new_dQ_h0, new_dQ_h1), attention.R, 1, attention.D);
    print_diff("dQ new_h0_vs_old_h0", compare_vectors(old_dQ_h0, new_dQ_h0), attention.R, 1, attention.D);
    print_diff("dQ new_h0_vs_old_h1", compare_vectors(old_dQ_h1, new_dQ_h0), attention.R, 1, attention.D);
    print_diff("dK new_h0_vs_new_h1", compare_vectors(new_dK_h0, new_dK_h1), attention.C, 1, attention.D);
    print_diff("dK new_h0_vs_old_h0", compare_vectors(old_dK_h0, new_dK_h0), attention.C, 1, attention.D);
    print_diff("dK new_h0_vs_old_h1", compare_vectors(old_dK_h1, new_dK_h0), attention.C, 1, attention.D);
    print_diff("dV new_h0_vs_new_h1", compare_vectors(new_dV_h0, new_dV_h1), attention.C, 1, attention.D);
    print_diff("dV new_h0_vs_old_h0", compare_vectors(old_dV_h0, new_dV_h0), attention.C, 1, attention.D);
    print_diff("dV new_h0_vs_old_h1", compare_vectors(old_dV_h1, new_dV_h0), attention.C, 1, attention.D);
  }

  pool->drain();
  return 0;
}
