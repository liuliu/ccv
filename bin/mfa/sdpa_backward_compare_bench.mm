#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/AttentionKernelType.hpp"
#include "nnc/mfa/kernels/AttentionOperand.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAAttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAAttentionKernel.hpp"
#include "nnc/mfa/kernels/NAAttentionKernelDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernelDescriptor.hpp"

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

struct DenseForwardPipeline {
  NAAttentionDescriptor descriptor;
  std::unique_ptr<NAAttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct DenseBackwardPipelines {
  NAAttentionDescriptor query_descriptor;
  NAAttentionDescriptor keyvalue_descriptor;
  std::unique_ptr<NAAttentionKernel> query_kernel;
  std::unique_ptr<NAAttentionKernel> keyvalue_kernel;
  NS::SharedPtr<MTL::ComputePipelineState> compute_d_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> query_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> keyvalue_pipeline;
};

struct QuantizePipelines {
  std::unique_ptr<NAInt8AttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> q_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> k_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> v_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> v_mean_pipeline;
  uint16_t q_threads = NAInt8AttentionKernel::qQuantizeThreads;
  uint16_t kv_threads = NAInt8AttentionKernel::kvQuantizeThreads;
  uint16_t v_mean_threads = NAInt8AttentionKernel::smallSequenceVMeanThreads;
  uint32_t q_tiles = 0;
  uint32_t kv_tiles = 0;
};

struct Int8ForwardPipeline {
  std::unique_ptr<NAInt8AttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct Int8BackwardPipelines {
  std::unique_ptr<NAInt8AttentionKernel> query_kernel;
  std::unique_ptr<NAInt8AttentionKernel> keyvalue_kernel;
  NS::SharedPtr<MTL::ComputePipelineState> compute_d_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> query_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> keyvalue_pipeline;
};

struct ScratchLayout {
  size_t q_int8 = 0;
  size_t k_int8 = 0;
  size_t v_int8 = 0;
  size_t dO_int8 = 0;
  size_t q_scale = 0;
  size_t k_scale = 0;
  size_t v_scale = 0;
  size_t dO_scale = 0;
  size_t v_mean = 0;
  size_t d = 0;
  size_t total = 0;
};

struct MPSGraphBackwardPipeline {
  MPSGraphExecutable* executable = nil;
  NSArray<MPSGraphTensorData*>* inputs = nil;
  NSArray<MPSGraphTensorData*>* outputs = nil;
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

std::vector<uint8_t> encode_fp16(const std::vector<float>& values)
{
  std::vector<uint8_t> bytes(values.size() * sizeof(half_float));
  auto* dst = reinterpret_cast<half_float*>(bytes.data());
  for (size_t i = 0; i < values.size(); ++i)
    dst[i] = (half_float)values[i];
  return bytes;
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

void upload_buffer_objc(
    id<MTLCommandQueue> command_queue,
    id<MTLBuffer> source,
    id<MTLBuffer> destination,
    size_t size)
{
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
  [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:size];
  [blit endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
}

size_t align_up(size_t value)
{
  return (value + 255) & ~((size_t)255);
}

size_t reserve(size_t* total, size_t size)
{
  const size_t offset = *total;
  *total = align_up(*total + size);
  return offset;
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

simd::ushort3 create_dense_block_dimensions(
    const AttentionCase& attention,
    AttentionKernelType type)
{
  unsigned short revised_head = (attention.D + 15) / 16 * 16;
  if (type.value != AttentionKernelType::forward && attention.D == 128) {
    revised_head = 64;
  } else if (attention.D <= 128) {
    revised_head = std::min<unsigned short>(attention.D, revised_head);
  } else {
    revised_head = revised_head / std::max<unsigned short>(revised_head / 128, 2);
  }
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

bool create_dense_bypass_threadgroup_memory(const AttentionCase& attention, simd::ushort3 block_dimensions)
{
  if (attention.D > 96)
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

AttentionOperands<GEMMOperandPrecision> create_fp16_dense_precisions()
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
  memory_precisions[AttentionOperand::L] = GEMMOperandPrecision::FP32;
  memory_precisions[AttentionOperand::D] = GEMMOperandPrecision::FP32;
  return memory_precisions;
}

NS::SharedPtr<MTL::ComputePipelineState> create_dense_attention_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const AttentionCase& attention,
    AttentionKernelType type)
{
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
  auto function = NS::TransferPtr(library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipeline_descriptor->setComputeFunction(function.get());
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return pipeline;
}

NS::SharedPtr<MTL::ComputePipelineState> create_dense_compute_d_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const AttentionCase& attention)
{
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

DenseForwardPipeline create_dense_forward_pipeline(MTL::Device* device, const AttentionCase& attention)
{
  DenseForwardPipeline bundle;
  bundle.descriptor.batchDimension = attention.batch;
  bundle.descriptor.Hq = attention.Hq;
  bundle.descriptor.Hk = attention.Hk;
  bundle.descriptor.lowPrecisionInputs = true;
  bundle.descriptor.isBF16 = false;
  bundle.descriptor.lowPrecisionIntermediates = false;
  bundle.descriptor.matrixDimensions = simd::uint3 { attention.R, attention.C, attention.D };
  bundle.descriptor.type = AttentionKernelType::forward;
  bundle.descriptor.scale = create_scale(attention);
  if (attention.batch > 1) {
    bundle.descriptor.batchStrides[AttentionOperand::Q] = attention.R * attention.D * attention.Hq;
    bundle.descriptor.batchStrides[AttentionOperand::K] = attention.C * attention.D * attention.Hk;
    bundle.descriptor.batchStrides[AttentionOperand::V] = attention.C * attention.D * attention.Hk;
    bundle.descriptor.batchStrides[AttentionOperand::O] = attention.R * attention.D * attention.Hq;
  }
  const simd::ushort3 block_dimensions = create_dense_block_dimensions(attention, AttentionKernelType::forward);
  const auto memory_precisions = create_fp16_dense_precisions();
  const bool check_c_edge_1 = (attention.C % (block_dimensions[1] * 2)) > block_dimensions[1];
  const NAAttentionKernelDescriptor kernel_descriptor(
      block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      check_c_edge_1,
      memory_precisions,
      AttentionKernelType::forward,
      bundle.descriptor.scale,
      false);
  bundle.kernel = std::make_unique<NAAttentionKernel>(kernel_descriptor, device);
  bundle.pipeline = create_dense_attention_pipeline(device, bundle.kernel->library.get(), attention, AttentionKernelType::forward);
  return bundle;
}

DenseBackwardPipelines create_dense_backward_pipelines(MTL::Device* device, const AttentionCase& attention)
{
  DenseBackwardPipelines bundle;
  bundle.query_descriptor.batchDimension = attention.batch;
  bundle.query_descriptor.Hq = attention.Hq;
  bundle.query_descriptor.Hk = attention.Hk;
  bundle.query_descriptor.lowPrecisionInputs = true;
  bundle.query_descriptor.isBF16 = false;
  bundle.query_descriptor.lowPrecisionIntermediates = false;
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
  const auto memory_precisions = create_fp16_dense_precisions();
  const auto query_block_dimensions = create_dense_block_dimensions(attention, AttentionKernelType::backwardQuery);
  const auto keyvalue_block_dimensions = create_dense_block_dimensions(attention, AttentionKernelType::backwardKeyValue);
  const bool query_check_c_edge_1 = (attention.C % (query_block_dimensions[1] * 2)) > query_block_dimensions[1];
  const bool keyvalue_check_c_edge_1 = (attention.C % (keyvalue_block_dimensions[1] * 2)) > keyvalue_block_dimensions[1];
  const NAAttentionKernelDescriptor query_kernel_descriptor(
      query_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      8,
      query_check_c_edge_1,
      memory_precisions,
      AttentionKernelType::backwardQuery,
      bundle.query_descriptor.scale,
      create_dense_bypass_threadgroup_memory(attention, query_block_dimensions));
  const NAAttentionKernelDescriptor keyvalue_kernel_descriptor(
      keyvalue_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      8,
      keyvalue_check_c_edge_1,
      memory_precisions,
      AttentionKernelType::backwardKeyValue,
      bundle.keyvalue_descriptor.scale,
      create_dense_bypass_threadgroup_memory(attention, keyvalue_block_dimensions));
  bundle.query_kernel = std::make_unique<NAAttentionKernel>(query_kernel_descriptor, device);
  bundle.keyvalue_kernel = std::make_unique<NAAttentionKernel>(keyvalue_kernel_descriptor, device);
  bundle.compute_d_pipeline = create_dense_compute_d_pipeline(device, bundle.query_kernel->library.get(), attention);
  bundle.query_pipeline = create_dense_attention_pipeline(device, bundle.query_kernel->library.get(), attention, AttentionKernelType::backwardQuery);
  bundle.keyvalue_pipeline = create_dense_attention_pipeline(device, bundle.keyvalue_kernel->library.get(), attention, AttentionKernelType::backwardKeyValue);
  return bundle;
}

double run_dense_forward_once(
    MTL::CommandQueue* command_queue,
    const DenseForwardPipeline& pipeline,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline.pipeline.get());
  encoder->setThreadgroupMemoryLength(pipeline.kernel->threadgroupMemoryAllocation(pipeline.pipeline.get(), pipeline.descriptor), 0);
  encoder->setBuffer(q_buffer, 0, 0);
  encoder->setBuffer(k_buffer, 0, 1);
  encoder->setBuffer(v_buffer, 0, 2);
  encoder->setBuffer(o_buffer, 0, 3);
  encoder->setBuffer(l_buffer, 0, 4);
  encoder->dispatchThreadgroups(
      pipeline.kernel->threadgroupsPerGrid(pipeline.descriptor),
      MTL::Size(pipeline.kernel->threadgroupSize(pipeline.pipeline.get(), pipeline.descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double run_dense_backward_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const DenseBackwardPipelines& pipelines,
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
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipelines.compute_d_pipeline.get());
    encoder->setBuffer(o_buffer, 0, 3);
    encoder->setBuffer(dO_buffer, 0, 6);
    encoder->setBuffer(d_buffer, 0, 5);
    encoder->dispatchThreadgroups(
        MTL::Size((uint64_t)attention.R * attention.Hq, 1, attention.batch),
        MTL::Size(NAAttentionKernel::computeDThreads, 1, 1));
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
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

NS::SharedPtr<MTL::FunctionConstantValues> create_int8_attention_constants(
    const AttentionCase& attention,
    uint32_t q_tiles,
    uint32_t kv_tiles)
{
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t row_dimension = attention.R;
  const uint32_t column_dimension = attention.C;
  const uint32_t q_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t k_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t v_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t o_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t dO_batch_stride = o_batch_stride;
  const uint32_t dV_batch_stride = v_batch_stride;
  const uint32_t dK_batch_stride = k_batch_stride;
  const uint32_t dQ_batch_stride = o_batch_stride;
  const uint32_t q_scale_batch_stride = attention.batch > 1 ? attention.Hq * q_tiles : 0;
  const uint32_t kv_scale_batch_stride = attention.batch > 1 ? attention.Hk * kv_tiles : 0;
  const uint32_t dO_scale_batch_stride = q_scale_batch_stride;
  const uint32_t v_mean_batch_stride = attention.batch > 1 ? attention.Hk * attention.D : 0;
  constants->setConstantValue(&row_dimension, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&column_dimension, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&q_batch_stride, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&k_batch_stride, MTL::DataTypeUInt, NS::UInteger(3));
  constants->setConstantValue(&v_batch_stride, MTL::DataTypeUInt, NS::UInteger(4));
  constants->setConstantValue(&o_batch_stride, MTL::DataTypeUInt, NS::UInteger(5));
  constants->setConstantValue(&dO_batch_stride, MTL::DataTypeUInt, NS::UInteger(6));
  constants->setConstantValue(&dV_batch_stride, MTL::DataTypeUInt, NS::UInteger(7));
  constants->setConstantValue(&dK_batch_stride, MTL::DataTypeUInt, NS::UInteger(8));
  constants->setConstantValue(&dQ_batch_stride, MTL::DataTypeUInt, NS::UInteger(9));
  constants->setConstantValue(&q_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(10));
  constants->setConstantValue(&kv_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(11));
  constants->setConstantValue(&kv_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(12));
  constants->setConstantValue(&dO_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(13));
  constants->setConstantValue(&v_mean_batch_stride, MTL::DataTypeUInt, NS::UInteger(14));
  return constants;
}

NS::SharedPtr<MTL::FunctionConstantValues> create_int8_quantize_constants(
    const AttentionCase& attention,
    uint32_t q_tiles,
    uint32_t kv_tiles)
{
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t q_sequence = attention.R;
  const uint32_t kv_sequence = attention.C;
  const uint32_t q_heads = attention.Hq;
  const uint32_t kv_heads = attention.Hk;
  const uint32_t q_tile_size = 16;
  const uint32_t kv_tile_size = 64;
  const uint32_t q_batch_stride = attention.batch > 1 ? attention.R * attention.D * attention.Hq : 0;
  const uint32_t k_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t v_batch_stride = attention.batch > 1 ? attention.C * attention.D * attention.Hk : 0;
  const uint32_t q_scale_batch_stride = attention.batch > 1 ? attention.Hq * q_tiles : 0;
  const uint32_t kv_scale_batch_stride = attention.batch > 1 ? attention.Hk * kv_tiles : 0;
  constants->setConstantValue(&q_sequence, MTL::DataTypeUInt, NS::UInteger(900));
  constants->setConstantValue(&kv_sequence, MTL::DataTypeUInt, NS::UInteger(901));
  constants->setConstantValue(&q_heads, MTL::DataTypeUInt, NS::UInteger(902));
  constants->setConstantValue(&kv_heads, MTL::DataTypeUInt, NS::UInteger(903));
  constants->setConstantValue(&q_tile_size, MTL::DataTypeUInt, NS::UInteger(904));
  constants->setConstantValue(&kv_tile_size, MTL::DataTypeUInt, NS::UInteger(905));
  constants->setConstantValue(&q_tiles, MTL::DataTypeUInt, NS::UInteger(906));
  constants->setConstantValue(&kv_tiles, MTL::DataTypeUInt, NS::UInteger(907));
  constants->setConstantValue(&q_batch_stride, MTL::DataTypeUInt, NS::UInteger(908));
  constants->setConstantValue(&k_batch_stride, MTL::DataTypeUInt, NS::UInteger(909));
  constants->setConstantValue(&v_batch_stride, MTL::DataTypeUInt, NS::UInteger(910));
  constants->setConstantValue(&q_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(911));
  constants->setConstantValue(&kv_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(912));
  return constants;
}

NS::SharedPtr<MTL::ComputePipelineState> create_int8_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const char* function_name_string,
    MTL::FunctionConstantValues* constants)
{
  NS::Error* error = nil;
  auto function_name = NS::String::string(function_name_string, NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(function_name, constants, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipeline_descriptor->setComputeFunction(function.get());
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(pipeline_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return pipeline;
}

QuantizePipelines create_int8_quantize_pipelines(MTL::Device* device, const AttentionCase& attention)
{
  QuantizePipelines bundle;
  const simd::ushort3 forward_block_dimensions { 16, 64, attention.D >= 192 ? (uint16_t)64 : (uint16_t)32 };
  bundle.q_tiles = (attention.R + 15) / 16;
  bundle.kv_tiles = (attention.C + 63) / 64;
  bundle.v_mean_threads =
      attention.C <= 20480 ?
      NAInt8AttentionKernel::smallSequenceVMeanThreads :
      NAInt8AttentionKernel::largeSequenceVMeanThreads;
  const NAInt8AttentionKernelDescriptor kernel_descriptor(
      forward_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      attention.D > 192 ? 16 : 4,
      bundle.v_mean_threads,
      (attention.C % forward_block_dimensions[1]) != 0,
      true,
      GEMMOperandPrecision::FP16,
      AttentionKernelType::forward,
      create_scale(attention));
  bundle.kernel = std::make_unique<NAInt8AttentionKernel>(kernel_descriptor, device);
  auto quantize_constants = create_int8_quantize_constants(attention, bundle.q_tiles, bundle.kv_tiles);
  bundle.q_pipeline = create_int8_pipeline(device, bundle.kernel->library.get(), "quantize_q", quantize_constants.get());
  bundle.k_pipeline = create_int8_pipeline(device, bundle.kernel->library.get(), "quantize_k", quantize_constants.get());
  bundle.v_pipeline = create_int8_pipeline(device, bundle.kernel->library.get(), "quantize_v", quantize_constants.get());
  bundle.v_mean_pipeline = create_int8_pipeline(device, bundle.kernel->library.get(), "compute_v_mean", quantize_constants.get());
  return bundle;
}

Int8ForwardPipeline create_int8_forward_pipeline(MTL::Device* device, const AttentionCase& attention)
{
  Int8ForwardPipeline bundle;
  const simd::ushort3 block_dimensions { 16, 64, attention.D >= 192 ? (uint16_t)64 : (uint16_t)32 };
  const NAInt8AttentionKernelDescriptor kernel_descriptor(
      block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      attention.D > 192 ? 16 : 4,
      attention.C <= 20480 ?
          NAInt8AttentionKernel::smallSequenceVMeanThreads :
          NAInt8AttentionKernel::largeSequenceVMeanThreads,
      (attention.C % block_dimensions[1]) != 0,
      true,
      GEMMOperandPrecision::FP16,
      AttentionKernelType::forward,
      create_scale(attention));
  bundle.kernel = std::make_unique<NAInt8AttentionKernel>(kernel_descriptor, device);
  auto attention_constants = create_int8_attention_constants(attention, (attention.R + 15) / 16, (attention.C + 63) / 64);
  bundle.pipeline = create_int8_pipeline(device, bundle.kernel->library.get(), "int8_attention", attention_constants.get());
  return bundle;
}

Int8BackwardPipelines create_int8_backward_pipelines(MTL::Device* device, const AttentionCase& attention)
{
  Int8BackwardPipelines bundle;
  const uint16_t v_mean_threads =
      attention.C <= 20480 ?
      NAInt8AttentionKernel::smallSequenceVMeanThreads :
      NAInt8AttentionKernel::largeSequenceVMeanThreads;
  const bool split_head_backward = attention.D == 128;
  const simd::ushort3 query_block_dimensions { 16, split_head_backward ? (uint16_t)32 : (uint16_t)64, split_head_backward ? (uint16_t)32 : (uint16_t)attention.D };
  const simd::ushort3 keyvalue_block_dimensions { 16, split_head_backward ? (uint16_t)32 : (uint16_t)16, split_head_backward ? (uint16_t)64 : (uint16_t)attention.D };
  const NAInt8AttentionKernelDescriptor query_descriptor(
      query_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      4,
      v_mean_threads,
      false,
      true,
      GEMMOperandPrecision::FP16,
      AttentionKernelType::backwardQuery,
      create_scale(attention));
  const NAInt8AttentionKernelDescriptor keyvalue_descriptor(
      keyvalue_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      split_head_backward ? (uint16_t)16 : (uint16_t)4,
      v_mean_threads,
      false,
      true,
      GEMMOperandPrecision::FP16,
      AttentionKernelType::backwardKeyValue,
      create_scale(attention));
  bundle.query_kernel = std::make_unique<NAInt8AttentionKernel>(query_descriptor, device);
  bundle.keyvalue_kernel = std::make_unique<NAInt8AttentionKernel>(keyvalue_descriptor, device);
  auto attention_constants = create_int8_attention_constants(attention, (attention.R + 15) / 16, (attention.C + 63) / 64);
  bundle.compute_d_pipeline = create_int8_pipeline(device, bundle.query_kernel->library.get(), "compute_d", attention_constants.get());
  bundle.query_pipeline = create_int8_pipeline(device, bundle.query_kernel->library.get(), "int8_backward_query", attention_constants.get());
  bundle.keyvalue_pipeline = create_int8_pipeline(device, bundle.keyvalue_kernel->library.get(), "int8_backward_keyvalue", attention_constants.get());
  return bundle;
}

ScratchLayout create_scratch_layout(const AttentionCase& attention)
{
  ScratchLayout layout;
  const uint32_t q_tiles = (attention.R + 15) / 16;
  const uint32_t kv_tiles = (attention.C + 63) / 64;
  const uint32_t q_batch_stride = attention.R * attention.D * attention.Hq;
  const uint32_t kv_batch_stride = attention.C * attention.D * attention.Hk;
  const uint32_t q_scale_batch_stride = attention.Hq * q_tiles;
  const uint32_t kv_scale_batch_stride = attention.Hk * kv_tiles;
  layout.q_int8 = reserve(&layout.total, (size_t)attention.batch * q_batch_stride * sizeof(int8_t));
  layout.k_int8 = reserve(&layout.total, (size_t)attention.batch * kv_batch_stride * sizeof(int8_t));
  layout.v_int8 = reserve(&layout.total, (size_t)attention.batch * kv_batch_stride * sizeof(int8_t));
  layout.dO_int8 = reserve(&layout.total, (size_t)attention.batch * q_batch_stride * sizeof(int8_t));
  layout.q_scale = reserve(&layout.total, (size_t)attention.batch * q_scale_batch_stride * sizeof(float));
  layout.k_scale = reserve(&layout.total, (size_t)attention.batch * kv_scale_batch_stride * sizeof(float));
  layout.v_scale = reserve(&layout.total, (size_t)attention.batch * kv_scale_batch_stride * sizeof(float));
  layout.dO_scale = reserve(&layout.total, (size_t)attention.batch * q_scale_batch_stride * sizeof(float));
  layout.v_mean = reserve(&layout.total, (size_t)attention.batch * attention.Hk * attention.D * sizeof(float));
  layout.d = reserve(&layout.total, (size_t)attention.batch * attention.Hq * attention.R * sizeof(float));
  return layout;
}

void encode_quantize(
    MTL::ComputeCommandEncoder* encoder,
    MTL::ComputePipelineState* pipeline,
    MTL::Buffer* src,
    size_t src_offset,
    MTL::Buffer* scratch,
    size_t int8_offset,
    size_t scale_offset,
    uint32_t tiles,
    uint32_t heads,
    uint16_t threads)
{
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(src, src_offset, 0);
  encoder->setBuffer(scratch, int8_offset, 1);
  encoder->setBuffer(scratch, scale_offset, 2);
  encoder->dispatchThreadgroups(MTL::Size(tiles, heads, 1), MTL::Size(threads, 1, 1));
}

void encode_compute_v_mean(
    MTL::ComputeCommandEncoder* encoder,
    const AttentionCase& attention,
    const QuantizePipelines& pipelines,
    MTL::Buffer* v_buffer,
    MTL::Buffer* scratch,
    size_t v_mean_offset)
{
  encoder->setComputePipelineState(pipelines.v_mean_pipeline.get());
  encoder->setBuffer(v_buffer, 0, 0);
  encoder->setBuffer(scratch, v_mean_offset, 1);
  const uint32_t mean_tiles = (attention.D % 4) == 0 ? (attention.D / 4) : attention.D;
  const uint32_t mean_tile_bits = ceil_log2_u32(mean_tiles);
  const uint32_t head_bits = ceil_log2_u32(attention.Hk);
  const uint32_t morton_codes = 1u << (mean_tile_bits + head_bits);
  encoder->dispatchThreadgroups(
      MTL::Size(morton_codes, 1, attention.batch),
      MTL::Size(pipelines.v_mean_threads, 1, 1));
}

void encode_quantize_v(
    MTL::ComputeCommandEncoder* encoder,
    const AttentionCase& attention,
    const QuantizePipelines& pipelines,
    MTL::Buffer* v_buffer,
    MTL::Buffer* scratch,
    const ScratchLayout& layout)
{
  encoder->setComputePipelineState(pipelines.v_pipeline.get());
  encoder->setBuffer(v_buffer, 0, 0);
  encoder->setBuffer(scratch, layout.v_int8, 1);
  encoder->setBuffer(scratch, layout.v_scale, 2);
  encoder->setBuffer(scratch, layout.v_mean, 3);
  encoder->dispatchThreadgroups(
      MTL::Size(pipelines.kv_tiles, attention.Hk, attention.batch),
      MTL::Size(pipelines.kv_threads, 1, 1));
}

double run_int8_forward_total_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const QuantizePipelines& quantize_pipelines,
    const Int8ForwardPipeline& forward_pipeline,
    const ScratchLayout& layout,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* scratch,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(encoder.get(), quantize_pipelines.q_pipeline.get(), q_buffer, 0, scratch, layout.q_int8, layout.q_scale, quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(encoder.get(), quantize_pipelines.k_pipeline.get(), k_buffer, 0, scratch, layout.k_int8, layout.k_scale, quantize_pipelines.kv_tiles, attention.Hk, quantize_pipelines.kv_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_compute_v_mean(encoder.get(), attention, quantize_pipelines, v_buffer, scratch, layout.v_mean);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize_v(encoder.get(), attention, quantize_pipelines, v_buffer, scratch, layout);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(forward_pipeline.pipeline.get());
    encoder->setThreadgroupMemoryLength(forward_pipeline.kernel->threadgroupMemoryAllocation(), 0);
    encoder->setBuffer(scratch, layout.q_int8, 0);
    encoder->setBuffer(scratch, layout.k_int8, 1);
    encoder->setBuffer(scratch, layout.v_int8, 2);
    encoder->setBuffer(o_buffer, 0, 3);
    encoder->setBuffer(l_buffer, 0, 4);
    encoder->setBuffer(scratch, layout.q_scale, 10);
    encoder->setBuffer(scratch, layout.k_scale, 11);
    encoder->setBuffer(scratch, layout.v_scale, 12);
    encoder->setBuffer(scratch, layout.v_mean, 14);
    encoder->dispatchThreadgroups(
        forward_pipeline.kernel->threadgroupsPerGrid(attention.batch, attention.R),
        MTL::Size(forward_pipeline.kernel->threadgroupSize(forward_pipeline.pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double run_int8_backward_total_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const QuantizePipelines& quantize_pipelines,
    const Int8BackwardPipelines& backward_pipelines,
    const ScratchLayout& layout,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer,
    MTL::Buffer* scratch,
    MTL::Buffer* dQ_buffer,
    MTL::Buffer* dK_buffer,
    MTL::Buffer* dV_buffer)
{
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(encoder.get(), quantize_pipelines.q_pipeline.get(), q_buffer, 0, scratch, layout.q_int8, layout.q_scale, quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(encoder.get(), quantize_pipelines.k_pipeline.get(), k_buffer, 0, scratch, layout.k_int8, layout.k_scale, quantize_pipelines.kv_tiles, attention.Hk, quantize_pipelines.kv_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_compute_v_mean(encoder.get(), attention, quantize_pipelines, v_buffer, scratch, layout.v_mean);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize_v(encoder.get(), attention, quantize_pipelines, v_buffer, scratch, layout);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(encoder.get(), quantize_pipelines.q_pipeline.get(), dO_buffer, 0, scratch, layout.dO_int8, layout.dO_scale, quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(backward_pipelines.compute_d_pipeline.get());
    encoder->setBuffer(o_buffer, 0, 3);
    encoder->setBuffer(scratch, layout.d, 5);
    encoder->setBuffer(dO_buffer, 0, 6);
    encoder->setBuffer(scratch, layout.v_mean, 14);
    encoder->dispatchThreadgroups(
        MTL::Size((uint64_t)attention.R * attention.Hq, 1, attention.batch),
        MTL::Size(NAInt8AttentionKernel::computeDThreads, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(backward_pipelines.query_pipeline.get());
    encoder->setThreadgroupMemoryLength(backward_pipelines.query_kernel->threadgroupMemoryAllocation(), 0);
    encoder->setBuffer(scratch, layout.q_int8, 0);
    encoder->setBuffer(scratch, layout.k_int8, 1);
    encoder->setBuffer(scratch, layout.v_int8, 2);
    encoder->setBuffer(l_buffer, 0, 4);
    encoder->setBuffer(scratch, layout.d, 5);
    encoder->setBuffer(scratch, layout.dO_int8, 6);
    encoder->setBuffer(dQ_buffer, 0, 9);
    encoder->setBuffer(scratch, layout.q_scale, 10);
    encoder->setBuffer(scratch, layout.k_scale, 11);
    encoder->setBuffer(scratch, layout.v_scale, 12);
    encoder->setBuffer(scratch, layout.dO_scale, 13);
    encoder->dispatchThreadgroups(
        backward_pipelines.query_kernel->threadgroupsPerGrid(attention.batch, attention.R),
        MTL::Size(backward_pipelines.query_kernel->threadgroupSize(backward_pipelines.query_pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(backward_pipelines.keyvalue_pipeline.get());
    encoder->setThreadgroupMemoryLength(backward_pipelines.keyvalue_kernel->threadgroupMemoryAllocation(), 0);
    encoder->setBuffer(scratch, layout.q_int8, 0);
    encoder->setBuffer(scratch, layout.k_int8, 1);
    encoder->setBuffer(scratch, layout.v_int8, 2);
    encoder->setBuffer(l_buffer, 0, 4);
    encoder->setBuffer(scratch, layout.d, 5);
    encoder->setBuffer(scratch, layout.dO_int8, 6);
    encoder->setBuffer(dV_buffer, 0, 7);
    encoder->setBuffer(dK_buffer, 0, 8);
    encoder->setBuffer(scratch, layout.q_scale, 10);
    encoder->setBuffer(scratch, layout.k_scale, 11);
    encoder->setBuffer(scratch, layout.v_scale, 12);
    encoder->setBuffer(scratch, layout.dO_scale, 13);
    encoder->dispatchThreadgroups(
        backward_pipelines.keyvalue_kernel->threadgroupsPerGrid(attention.batch, attention.C),
        MTL::Size(backward_pipelines.keyvalue_kernel->threadgroupSize(backward_pipelines.keyvalue_pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

NSArray<NSNumber*>* create_shape4(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3)
{
  return @[ @(d0), @(d1), @(d2), @(d3) ];
}

MPSGraphTensorData* create_tensor_data(id<MTLBuffer> buffer, NSArray<NSNumber*>* shape, MPSDataType data_type)
{
  return [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:data_type] autorelease];
}

MPSGraphBackwardPipeline create_mpsgraph_backward_pipeline(
    id<MTLDevice> device,
    const AttentionCase& attention,
    id<MTLBuffer> q_buffer,
    id<MTLBuffer> k_buffer,
    id<MTLBuffer> v_buffer,
    id<MTLBuffer> g_buffer,
    id<MTLBuffer> dq_buffer,
    id<MTLBuffer> dk_buffer,
    id<MTLBuffer> dv_buffer)
{
  MPSGraphBackwardPipeline pipeline;
  NSArray<NSNumber*>* q_shape = create_shape4(attention.batch, attention.R, attention.Hq, attention.D);
  NSArray<NSNumber*>* k_shape = create_shape4(attention.batch, attention.C, attention.Hk, attention.D);
  NSArray<NSNumber*>* v_shape = create_shape4(attention.batch, attention.C, attention.Hk, attention.D);
  NSArray<NSNumber*>* g_shape = create_shape4(attention.batch, attention.R, attention.Hq, attention.D);
  NSArray<NSNumber*>* dq_shape = create_shape4(attention.batch, attention.R, attention.Hq, attention.D);
  NSArray<NSNumber*>* dk_shape = create_shape4(attention.batch, attention.C, attention.Hk, attention.D);
  NSArray<NSNumber*>* dv_shape = create_shape4(attention.batch, attention.C, attention.Hk, attention.D);

  MPSGraph* graph = [MPSGraph new];
  graph.options = MPSGraphOptionsSynchronizeResults;

  const MPSDataType data_type = MPSDataTypeFloat16;
  MPSGraphTensor* mps_g = [graph placeholderWithShape:g_shape dataType:data_type name:nil];
  MPSGraphTensor* mps_q = [graph placeholderWithShape:q_shape dataType:data_type name:nil];
  MPSGraphTensor* mps_k = [graph placeholderWithShape:k_shape dataType:data_type name:nil];
  MPSGraphTensor* mps_v = [graph placeholderWithShape:v_shape dataType:data_type name:nil];

  NSArray<MPSGraphTensor*>* input_tensors = @[ mps_g, mps_q, mps_k, mps_v ];
  NSArray<MPSGraphShapedType*>* input_shapes = @[
    [[[MPSGraphShapedType alloc] initWithShape:g_shape dataType:data_type] autorelease],
    [[[MPSGraphShapedType alloc] initWithShape:q_shape dataType:data_type] autorelease],
    [[[MPSGraphShapedType alloc] initWithShape:k_shape dataType:data_type] autorelease],
    [[[MPSGraphShapedType alloc] initWithShape:v_shape dataType:data_type] autorelease]
  ];

  MPSGraphTensor* mps_scale = [graph constantWithScalar:create_scale(attention) dataType:data_type];
  mps_q = [graph multiplicationWithPrimaryTensor:mps_scale secondaryTensor:[graph transposeTensor:mps_q dimension:1 withDimension:2 name:nil] name:nil];
  mps_k = [graph transposeTensor:mps_k dimension:1 withDimension:2 name:nil];
  MPSGraphTensor* mps_kt = [graph transposeTensor:mps_k dimension:2 withDimension:3 name:nil];
  mps_v = [graph transposeTensor:mps_v dimension:1 withDimension:2 name:nil];
  MPSGraphTensor* mps_qk = [graph matrixMultiplicationWithPrimaryTensor:mps_q secondaryTensor:mps_kt name:nil];
  MPSGraphTensor* mps_softmax = [graph softMaxWithTensor:mps_qk axis:3 name:nil];
  mps_g = [graph transposeTensor:mps_g dimension:1 withDimension:2 name:nil];
  MPSGraphTensor* mps_softmaxt = [graph transposeTensor:mps_softmax dimension:2 withDimension:3 name:nil];
  MPSGraphTensor* mps_dv = [graph matrixMultiplicationWithPrimaryTensor:mps_softmaxt secondaryTensor:mps_g name:nil];
  mps_v = [graph transposeTensor:mps_v dimension:2 withDimension:3 name:nil];
  MPSGraphTensor* mps_dsoftmax = [graph matrixMultiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_v name:nil];
  MPSGraphTensor* mul_tensor = [graph multiplicationWithPrimaryTensor:mps_softmax secondaryTensor:mps_dsoftmax name:nil];
  MPSGraphTensor* mul_sum_tensor = [graph reductionSumWithTensor:mul_tensor axis:-1 name:nil];
  MPSGraphTensor* grad_sub_tensor = [graph subtractionWithPrimaryTensor:mps_dsoftmax secondaryTensor:mul_sum_tensor name:nil];
  MPSGraphTensor* mps_dqk = [graph multiplicationWithPrimaryTensor:mps_softmax secondaryTensor:grad_sub_tensor name:nil];
  MPSGraphTensor* mps_dq = [graph multiplicationWithPrimaryTensor:mps_scale secondaryTensor:[graph matrixMultiplicationWithPrimaryTensor:mps_dqk secondaryTensor:mps_k name:nil] name:nil];
  mps_dqk = [graph transposeTensor:mps_dqk dimension:2 withDimension:3 name:nil];
  MPSGraphTensor* mps_dk = [graph matrixMultiplicationWithPrimaryTensor:mps_dqk secondaryTensor:mps_q name:nil];
  mps_dq = [graph transposeTensor:mps_dq dimension:1 withDimension:2 name:nil];
  mps_dk = [graph transposeTensor:mps_dk dimension:1 withDimension:2 name:nil];
  mps_dv = [graph transposeTensor:mps_dv dimension:1 withDimension:2 name:nil];

  MPSGraphCompilationDescriptor* compilation_descriptor = [MPSGraphCompilationDescriptor new];
  compilation_descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
  compilation_descriptor.optimizationProfile = MPSGraphOptimizationProfilePerformance;
  NSDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
      [NSDictionary dictionaryWithObjects:input_shapes forKeys:input_tensors];
  MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
  pipeline.executable =
      [[graph compileWithDevice:graph_device
                          feeds:feeds
                  targetTensors:@[ mps_dq, mps_dk, mps_dv ]
               targetOperations:nil
           compilationDescriptor:compilation_descriptor] retain];
  pipeline.executable.options = MPSGraphOptionsSynchronizeResults;
  [compilation_descriptor release];
  [graph release];

  pipeline.inputs = [[NSArray alloc] initWithObjects:
      create_tensor_data(g_buffer, g_shape, data_type),
      create_tensor_data(q_buffer, q_shape, data_type),
      create_tensor_data(k_buffer, k_shape, data_type),
      create_tensor_data(v_buffer, v_shape, data_type), nil];
  pipeline.outputs = [[NSArray alloc] initWithObjects:
      create_tensor_data(dq_buffer, dq_shape, data_type),
      create_tensor_data(dk_buffer, dk_shape, data_type),
      create_tensor_data(dv_buffer, dv_shape, data_type), nil];
  return pipeline;
}

double run_mpsgraph_backward_once(
    id<MTLCommandQueue> command_queue,
    const MPSGraphBackwardPipeline& pipeline)
{
  const auto start = std::chrono::steady_clock::now();
  id<MTLCommandBuffer> command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];
  [pipeline.executable encodeToCommandBuffer:(MPSCommandBuffer*)command_buffer
                                 inputsArray:pipeline.inputs
                                resultsArray:pipeline.outputs
                         executionDescriptor:nil];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

void destroy_mpsgraph_backward_pipeline(MPSGraphBackwardPipeline* pipeline)
{
  if (pipeline->inputs) {
    [pipeline->inputs release];
    pipeline->inputs = nil;
  }
  if (pipeline->outputs) {
    [pipeline->outputs release];
    pipeline->outputs = nil;
  }
  if (pipeline->executable) {
    [pipeline->executable release];
    pipeline->executable = nil;
  }
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  AttentionCase attention;
  BenchmarkConfig config;
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

  auto* pool = NS::AutoreleasePool::alloc()->init();

  auto dense_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!dense_device) {
    std::cerr << "Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  auto dense_command_queue = NS::TransferPtr(dense_device->newCommandQueue());
  if (!dense_command_queue) {
    std::cerr << "Metal command queue unavailable.\n";
    pool->drain();
    return 1;
  }

  id<MTLDevice> graph_device = MTLCreateSystemDefaultDevice();
  if (!graph_device) {
    std::cerr << "MPSGraph Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  id<MTLCommandQueue> graph_command_queue = [graph_device newCommandQueue];
  if (!graph_command_queue) {
    std::cerr << "MPSGraph command queue unavailable.\n";
    [graph_device release];
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

  auto dense_q_stage = NS::TransferPtr(dense_device->newBuffer(q_bytes_data.data(), q_bytes, kSharedResourceOptions));
  auto dense_k_stage = NS::TransferPtr(dense_device->newBuffer(k_bytes_data.data(), k_bytes, kSharedResourceOptions));
  auto dense_v_stage = NS::TransferPtr(dense_device->newBuffer(v_bytes_data.data(), v_bytes, kSharedResourceOptions));
  auto dense_dO_stage = NS::TransferPtr(dense_device->newBuffer(dO_bytes_data.data(), dO_bytes, kSharedResourceOptions));
  auto dense_q = NS::TransferPtr(dense_device->newBuffer(q_bytes, kPrivateResourceOptions));
  auto dense_k = NS::TransferPtr(dense_device->newBuffer(k_bytes, kPrivateResourceOptions));
  auto dense_v = NS::TransferPtr(dense_device->newBuffer(v_bytes, kPrivateResourceOptions));
  auto dense_dO = NS::TransferPtr(dense_device->newBuffer(dO_bytes, kPrivateResourceOptions));
  auto dense_o = NS::TransferPtr(dense_device->newBuffer(o_bytes, kPrivateResourceOptions));
  auto dense_l = NS::TransferPtr(dense_device->newBuffer(l_bytes, kPrivateResourceOptions));
  auto dense_dQ = NS::TransferPtr(dense_device->newBuffer(dQ_bytes, kPrivateResourceOptions));
  auto dense_dK = NS::TransferPtr(dense_device->newBuffer(dK_bytes, kPrivateResourceOptions));
  auto dense_dV = NS::TransferPtr(dense_device->newBuffer(dV_bytes, kPrivateResourceOptions));
  auto dense_d = NS::TransferPtr(dense_device->newBuffer(d_bytes, kPrivateResourceOptions));

  upload_buffer(dense_command_queue.get(), dense_q_stage.get(), dense_q.get(), q_bytes);
  upload_buffer(dense_command_queue.get(), dense_k_stage.get(), dense_k.get(), k_bytes);
  upload_buffer(dense_command_queue.get(), dense_v_stage.get(), dense_v.get(), v_bytes);
  upload_buffer(dense_command_queue.get(), dense_dO_stage.get(), dense_dO.get(), dO_bytes);

  auto int8_q_stage = NS::TransferPtr(dense_device->newBuffer(q_bytes_data.data(), q_bytes, kSharedResourceOptions));
  auto int8_k_stage = NS::TransferPtr(dense_device->newBuffer(k_bytes_data.data(), k_bytes, kSharedResourceOptions));
  auto int8_v_stage = NS::TransferPtr(dense_device->newBuffer(v_bytes_data.data(), v_bytes, kSharedResourceOptions));
  auto int8_dO_stage = NS::TransferPtr(dense_device->newBuffer(dO_bytes_data.data(), dO_bytes, kSharedResourceOptions));
  auto int8_q = NS::TransferPtr(dense_device->newBuffer(q_bytes, kPrivateResourceOptions));
  auto int8_k = NS::TransferPtr(dense_device->newBuffer(k_bytes, kPrivateResourceOptions));
  auto int8_v = NS::TransferPtr(dense_device->newBuffer(v_bytes, kPrivateResourceOptions));
  auto int8_dO = NS::TransferPtr(dense_device->newBuffer(dO_bytes, kPrivateResourceOptions));
  auto int8_o = NS::TransferPtr(dense_device->newBuffer(o_bytes, kPrivateResourceOptions));
  auto int8_l = NS::TransferPtr(dense_device->newBuffer(l_bytes, kPrivateResourceOptions));
  auto int8_dQ = NS::TransferPtr(dense_device->newBuffer(dQ_bytes, kPrivateResourceOptions));
  auto int8_dK = NS::TransferPtr(dense_device->newBuffer(dK_bytes, kPrivateResourceOptions));
  auto int8_dV = NS::TransferPtr(dense_device->newBuffer(dV_bytes, kPrivateResourceOptions));
  const auto scratch_layout = create_scratch_layout(attention);
  auto int8_scratch = NS::TransferPtr(dense_device->newBuffer(scratch_layout.total, kPrivateResourceOptions));

  upload_buffer(dense_command_queue.get(), int8_q_stage.get(), int8_q.get(), q_bytes);
  upload_buffer(dense_command_queue.get(), int8_k_stage.get(), int8_k.get(), k_bytes);
  upload_buffer(dense_command_queue.get(), int8_v_stage.get(), int8_v.get(), v_bytes);
  upload_buffer(dense_command_queue.get(), int8_dO_stage.get(), int8_dO.get(), dO_bytes);

  id<MTLBuffer> graph_q_stage = [graph_device newBufferWithBytes:q_bytes_data.data() length:q_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> graph_k_stage = [graph_device newBufferWithBytes:k_bytes_data.data() length:k_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> graph_v_stage = [graph_device newBufferWithBytes:v_bytes_data.data() length:v_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> graph_g_stage = [graph_device newBufferWithBytes:dO_bytes_data.data() length:dO_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> graph_q = [graph_device newBufferWithLength:q_bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> graph_k = [graph_device newBufferWithLength:k_bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> graph_v = [graph_device newBufferWithLength:v_bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> graph_g = [graph_device newBufferWithLength:dO_bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> graph_dq = [graph_device newBufferWithLength:dQ_bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> graph_dk = [graph_device newBufferWithLength:dK_bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> graph_dv = [graph_device newBufferWithLength:dV_bytes options:MTLResourceStorageModePrivate];

  upload_buffer_objc(graph_command_queue, graph_q_stage, graph_q, q_bytes);
  upload_buffer_objc(graph_command_queue, graph_k_stage, graph_k, k_bytes);
  upload_buffer_objc(graph_command_queue, graph_v_stage, graph_v, v_bytes);
  upload_buffer_objc(graph_command_queue, graph_g_stage, graph_g, dO_bytes);

  const auto dense_forward = create_dense_forward_pipeline(dense_device.get(), attention);
  const auto dense_backward = create_dense_backward_pipelines(dense_device.get(), attention);
  const auto int8_quantize = create_int8_quantize_pipelines(dense_device.get(), attention);
  const auto int8_forward = create_int8_forward_pipeline(dense_device.get(), attention);
  const auto int8_backward = create_int8_backward_pipelines(dense_device.get(), attention);
  auto graph_backward = create_mpsgraph_backward_pipeline(
      graph_device, attention, graph_q, graph_k, graph_v, graph_g, graph_dq, graph_dk, graph_dv);

  if (!(run_dense_forward_once(
            dense_command_queue.get(),
            dense_forward,
            dense_q.get(),
            dense_k.get(),
            dense_v.get(),
            dense_o.get(),
            dense_l.get()) > 0)) {
    std::cerr << "dense forward setup failed\n";
    destroy_mpsgraph_backward_pipeline(&graph_backward);
    [graph_command_queue release];
    [graph_device release];
    pool->drain();
    return 1;
  }
  if (!(run_int8_forward_total_once(
            dense_command_queue.get(),
            attention,
            int8_quantize,
            int8_forward,
            scratch_layout,
            int8_q.get(),
            int8_k.get(),
            int8_v.get(),
            int8_scratch.get(),
            int8_o.get(),
            int8_l.get()) > 0)) {
    std::cerr << "int8 forward setup failed\n";
    destroy_mpsgraph_backward_pipeline(&graph_backward);
    [graph_command_queue release];
    [graph_device release];
    pool->drain();
    return 1;
  }
  if (!(run_mpsgraph_backward_once(graph_command_queue, graph_backward) > 0)) {
    std::cerr << "mpsgraph backward setup failed\n";
    destroy_mpsgraph_backward_pipeline(&graph_backward);
    [graph_command_queue release];
    [graph_device release];
    pool->drain();
    return 1;
  }

  Stats dense_backward_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_dense_backward_once(
                dense_command_queue.get(),
                attention,
                dense_backward,
                dense_q.get(),
                dense_k.get(),
                dense_v.get(),
                dense_o.get(),
                dense_l.get(),
                dense_dO.get(),
                dense_dQ.get(),
                dense_dK.get(),
                dense_dV.get(),
                dense_d.get());
          },
          &dense_backward_stats)) {
    std::cerr << "dense backward benchmark failed\n";
    destroy_mpsgraph_backward_pipeline(&graph_backward);
    [graph_command_queue release];
    [graph_device release];
    pool->drain();
    return 1;
  }

  Stats int8_backward_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_int8_backward_total_once(
                dense_command_queue.get(),
                attention,
                int8_quantize,
                int8_backward,
                scratch_layout,
                int8_q.get(),
                int8_k.get(),
                int8_v.get(),
                int8_dO.get(),
                int8_o.get(),
                int8_l.get(),
                int8_scratch.get(),
                int8_dQ.get(),
                int8_dK.get(),
                int8_dV.get());
          },
          &int8_backward_stats)) {
    std::cerr << "int8 backward benchmark failed\n";
    destroy_mpsgraph_backward_pipeline(&graph_backward);
    [graph_command_queue release];
    [graph_device release];
    pool->drain();
    return 1;
  }

  Stats graph_backward_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_mpsgraph_backward_once(graph_command_queue, graph_backward);
          },
          &graph_backward_stats)) {
    std::cerr << "mpsgraph backward benchmark failed\n";
    destroy_mpsgraph_backward_pipeline(&graph_backward);
    [graph_command_queue release];
    [graph_device release];
    pool->drain();
    return 1;
  }

  std::cout << "case"
            << " B=" << attention.batch
            << " R=" << attention.R
            << " C=" << attention.C
            << " Hq=" << attention.Hq
            << " Hk=" << attention.Hk
            << " D=" << attention.D
            << '\n';
  print_stats("na_backward", dense_backward_stats);
  print_stats("na_int8_backward_total", int8_backward_stats);
  print_stats("mpsgraph_backward", graph_backward_stats);
  std::cout << std::fixed << std::setprecision(4)
            << "na_vs_mpsgraph_speedup=" << (graph_backward_stats.median_seconds / dense_backward_stats.median_seconds)
            << '\n'
            << "na_int8_vs_na_speedup=" << (dense_backward_stats.median_seconds / int8_backward_stats.median_seconds)
            << '\n'
            << "na_int8_vs_mpsgraph_speedup=" << (graph_backward_stats.median_seconds / int8_backward_stats.median_seconds)
            << '\n';

  destroy_mpsgraph_backward_pipeline(&graph_backward);
  [graph_q_stage release];
  [graph_k_stage release];
  [graph_v_stage release];
  [graph_g_stage release];
  [graph_q release];
  [graph_k release];
  [graph_v release];
  [graph_g release];
  [graph_dq release];
  [graph_dk release];
  [graph_dv release];
  [graph_command_queue release];
  [graph_device release];
  pool->drain();
  return 0;
}
