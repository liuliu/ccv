#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/AttentionKernelType.hpp"
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

struct ForwardPipeline {
  std::unique_ptr<NAInt8AttentionKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct BackwardPipelines {
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

std::vector<uint8_t> encode_fp16(const std::vector<float>& values)
{
  std::vector<uint8_t> bytes(values.size() * sizeof(half_float));
  auto* dst = reinterpret_cast<half_float*>(bytes.data());
  for (size_t i = 0; i < values.size(); ++i)
    dst[i] = (half_float)values[i];
  return bytes;
}

float create_scale(const AttentionCase& attention)
{
  return 1.0f / std::sqrt((float)attention.D);
}

GEMMOperandPrecision create_io_precision()
{
  return GEMMOperandPrecision::FP16;
}

bool create_low_precision_intermediates()
{
  return true;
}

GEMMOperandPrecision create_l_precision()
{
  const auto io_precision = create_io_precision();
  if (!create_low_precision_intermediates())
    return GEMMOperandPrecision::FP32;
  return io_precision == GEMMOperandPrecision::BF16 ?
      GEMMOperandPrecision::BF16 :
      (io_precision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16);
}

GEMMOperandPrecision create_d_precision()
{
  const auto io_precision = create_io_precision();
  if (!create_low_precision_intermediates())
    return GEMMOperandPrecision::FP32;
  return io_precision == GEMMOperandPrecision::FP32 ?
      GEMMOperandPrecision::FP32 :
      GEMMOperandPrecision::BF16;
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

simd::ushort3 create_forward_block_dimensions(const AttentionCase& attention)
{
  return simd::ushort3 { 16, 64, attention.D >= 192 ? (uint16_t)64 : (uint16_t)32 };
}

simd::ushort3 create_backward_query_block_dimensions(
    const AttentionCase& attention,
    uint16_t block_r,
    uint16_t block_c,
    uint16_t block_d)
{
  const uint16_t default_block_d = attention.D == 128 ? (uint16_t)32 : (uint16_t)attention.D;
  const uint16_t default_block_c = attention.D == 128 ? (uint16_t)32 : (uint16_t)64;
  return simd::ushort3 {
      block_r ? block_r : (uint16_t)16,
      block_c ? block_c : default_block_c,
      block_d ? block_d : default_block_d };
}

simd::ushort3 create_backward_keyvalue_block_dimensions(
    const AttentionCase& attention,
    uint16_t block_r,
    uint16_t block_c,
    uint16_t block_d)
{
  const uint16_t default_block_d = attention.D == 128 ? (uint16_t)64 : (uint16_t)attention.D;
  const uint16_t default_block_c = attention.D == 128 ? (uint16_t)32 : (uint16_t)16;
  return simd::ushort3 {
      block_r ? block_r : (uint16_t)16,
      block_c ? block_c : default_block_c,
      block_d ? block_d : default_block_d };
}

uint16_t create_forward_execution_simdgroups(const AttentionCase& attention)
{
  return attention.D > 192 ? 16 : 4;
}

uint16_t create_backward_query_execution_simdgroups(const AttentionCase&)
{
  return 4;
}

uint16_t create_backward_keyvalue_execution_simdgroups(const AttentionCase& attention)
{
  if (attention.D == 128) {
    return 16;
  }
  return 4;
}

NS::SharedPtr<MTL::FunctionConstantValues> create_attention_constants(
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

NS::SharedPtr<MTL::FunctionConstantValues> create_quantize_constants(
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

NS::SharedPtr<MTL::ComputePipelineState> create_pipeline(
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

QuantizePipelines create_quantize_pipelines(MTL::Device* device, const AttentionCase& attention)
{
  QuantizePipelines bundle;
  const simd::ushort3 forward_block_dimensions = create_forward_block_dimensions(attention);
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
      create_forward_execution_simdgroups(attention),
      bundle.v_mean_threads,
      (attention.C % forward_block_dimensions[1]) != 0,
      true,
      create_io_precision(),
      create_low_precision_intermediates(),
      AttentionKernelType::forward,
      create_scale(attention));
  bundle.kernel = std::make_unique<NAInt8AttentionKernel>(kernel_descriptor, device);
  auto quantize_constants = create_quantize_constants(attention, bundle.q_tiles, bundle.kv_tiles);
  bundle.q_pipeline = create_pipeline(device, bundle.kernel->library.get(), "quantize_q", quantize_constants.get());
  bundle.k_pipeline = create_pipeline(device, bundle.kernel->library.get(), "quantize_k", quantize_constants.get());
  bundle.v_pipeline = create_pipeline(device, bundle.kernel->library.get(), "quantize_v", quantize_constants.get());
  bundle.v_mean_pipeline = create_pipeline(device, bundle.kernel->library.get(), "compute_v_mean", quantize_constants.get());
  return bundle;
}

ForwardPipeline create_forward_pipeline(MTL::Device* device, const AttentionCase& attention)
{
  ForwardPipeline bundle;
  const simd::ushort3 block_dimensions = create_forward_block_dimensions(attention);
  const NAInt8AttentionKernelDescriptor kernel_descriptor(
      block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      create_forward_execution_simdgroups(attention),
      attention.C <= 20480 ?
          NAInt8AttentionKernel::smallSequenceVMeanThreads :
          NAInt8AttentionKernel::largeSequenceVMeanThreads,
      (attention.C % block_dimensions[1]) != 0,
      true,
      create_io_precision(),
      create_low_precision_intermediates(),
      AttentionKernelType::forward,
      create_scale(attention));
  bundle.kernel = std::make_unique<NAInt8AttentionKernel>(kernel_descriptor, device);
  auto attention_constants = create_attention_constants(
      attention, (attention.R + 15) / 16, (attention.C + 63) / 64);
  bundle.pipeline = create_pipeline(device, bundle.kernel->library.get(), "int8_attention", attention_constants.get());
  return bundle;
}

BackwardPipelines create_backward_pipelines(
    MTL::Device* device,
    const AttentionCase& attention,
    simd::ushort3 query_block_dimensions,
    simd::ushort3 keyvalue_block_dimensions,
    uint16_t query_execution_simdgroups,
    uint16_t keyvalue_execution_simdgroups)
{
  BackwardPipelines bundle;
  const uint16_t v_mean_threads =
      attention.C <= 20480 ?
      NAInt8AttentionKernel::smallSequenceVMeanThreads :
      NAInt8AttentionKernel::largeSequenceVMeanThreads;
  const NAInt8AttentionKernelDescriptor query_descriptor(
      query_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      query_execution_simdgroups,
      v_mean_threads,
      (attention.C % query_block_dimensions[1]) != 0,
      true,
      create_io_precision(),
      create_low_precision_intermediates(),
      AttentionKernelType::backwardQuery,
      create_scale(attention));
  const NAInt8AttentionKernelDescriptor keyvalue_descriptor(
      keyvalue_block_dimensions,
      attention.D,
      attention.Hq,
      attention.Hk,
      16,
      64,
      keyvalue_execution_simdgroups,
      v_mean_threads,
      (attention.C % keyvalue_block_dimensions[1]) != 0,
      true,
      create_io_precision(),
      create_low_precision_intermediates(),
      AttentionKernelType::backwardKeyValue,
      create_scale(attention));
  bundle.query_kernel = std::make_unique<NAInt8AttentionKernel>(query_descriptor, device);
  bundle.keyvalue_kernel = std::make_unique<NAInt8AttentionKernel>(keyvalue_descriptor, device);
  auto attention_constants = create_attention_constants(
      attention, (attention.R + 15) / 16, (attention.C + 63) / 64);
  bundle.compute_d_pipeline = create_pipeline(device, bundle.query_kernel->library.get(), "compute_d", attention_constants.get());
  bundle.query_pipeline = create_pipeline(device, bundle.query_kernel->library.get(), "int8_backward_query", attention_constants.get());
  bundle.keyvalue_pipeline = create_pipeline(device, bundle.keyvalue_kernel->library.get(), "int8_backward_keyvalue", attention_constants.get());
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
  layout.d = reserve(&layout.total, (size_t)attention.batch * attention.Hq * attention.R * create_d_precision().size());
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
    size_t v_offset,
    MTL::Buffer* scratch,
    size_t v_mean_offset)
{
  encoder->setComputePipelineState(pipelines.v_mean_pipeline.get());
  encoder->setBuffer(v_buffer, v_offset, 0);
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
    size_t v_offset,
    MTL::Buffer* scratch,
    const ScratchLayout& layout)
{
  encoder->setComputePipelineState(pipelines.v_pipeline.get());
  encoder->setBuffer(v_buffer, v_offset, 0);
  encoder->setBuffer(scratch, layout.v_int8, 1);
  encoder->setBuffer(scratch, layout.v_scale, 2);
  encoder->setBuffer(scratch, layout.v_mean, 3);
  encoder->dispatchThreadgroups(
      MTL::Size(pipelines.kv_tiles, attention.Hk, attention.batch),
      MTL::Size(pipelines.kv_threads, 1, 1));
}

double run_forward_total_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const QuantizePipelines& quantize_pipelines,
    const ForwardPipeline& forward_pipeline,
    const ScratchLayout& layout,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* scratch,
    MTL::Buffer* o_buffer,
    MTL::Buffer* l_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.q_pipeline.get(),
        q_buffer, 0, scratch, layout.q_int8, layout.q_scale,
        quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.k_pipeline.get(),
        k_buffer, 0, scratch, layout.k_int8, layout.k_scale,
        quantize_pipelines.kv_tiles, attention.Hk, quantize_pipelines.kv_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_compute_v_mean(encoder.get(), attention, quantize_pipelines, v_buffer, 0, scratch, layout.v_mean);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize_v(encoder.get(), attention, quantize_pipelines, v_buffer, 0, scratch, layout);
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
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_prepare_backward_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const QuantizePipelines& quantize_pipelines,
    const BackwardPipelines& backward_pipelines,
    const ScratchLayout& layout,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* dO_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* scratch)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.q_pipeline.get(),
        q_buffer, 0, scratch, layout.q_int8, layout.q_scale,
        quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.k_pipeline.get(),
        k_buffer, 0, scratch, layout.k_int8, layout.k_scale,
        quantize_pipelines.kv_tiles, attention.Hk, quantize_pipelines.kv_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_compute_v_mean(encoder.get(), attention, quantize_pipelines, v_buffer, 0, scratch, layout.v_mean);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize_v(encoder.get(), attention, quantize_pipelines, v_buffer, 0, scratch, layout);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.q_pipeline.get(),
        dO_buffer, 0, scratch, layout.dO_int8, layout.dO_scale,
        quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
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
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_backward_query_only_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const BackwardPipelines& backward_pipelines,
    const ScratchLayout& layout,
    MTL::Buffer* scratch,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dQ_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
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
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_backward_keyvalue_only_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const BackwardPipelines& backward_pipelines,
    const ScratchLayout& layout,
    MTL::Buffer* scratch,
    MTL::Buffer* l_buffer,
    MTL::Buffer* dK_buffer,
    MTL::Buffer* dV_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
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
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_backward_total_once(
    MTL::CommandQueue* command_queue,
    const AttentionCase& attention,
    const QuantizePipelines& quantize_pipelines,
    const BackwardPipelines& backward_pipelines,
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
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.q_pipeline.get(),
        q_buffer, 0, scratch, layout.q_int8, layout.q_scale,
        quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.k_pipeline.get(),
        k_buffer, 0, scratch, layout.k_int8, layout.k_scale,
        quantize_pipelines.kv_tiles, attention.Hk, quantize_pipelines.kv_threads);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_compute_v_mean(encoder.get(), attention, quantize_pipelines, v_buffer, 0, scratch, layout.v_mean);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize_v(encoder.get(), attention, quantize_pipelines, v_buffer, 0, scratch, layout);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_quantize(
        encoder.get(), quantize_pipelines.q_pipeline.get(),
        dO_buffer, 0, scratch, layout.dO_int8, layout.dO_scale,
        quantize_pipelines.q_tiles, attention.Hq, quantize_pipelines.q_threads);
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
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  AttentionCase attention;
  BenchmarkConfig config;
  uint16_t query_block_r = 0;
  uint16_t query_block_c = 0;
  uint16_t query_block_d = 0;
  uint16_t keyvalue_block_r = 0;
  uint16_t keyvalue_block_c = 0;
  uint16_t keyvalue_block_d = 0;
  uint16_t query_execution_simdgroups = create_backward_query_execution_simdgroups(attention);
  uint16_t keyvalue_execution_simdgroups = create_backward_keyvalue_execution_simdgroups(attention);

  if (argc >= 4) {
    attention.R = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    attention.C = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    attention.D = (uint32_t)std::strtoul(argv[3], nullptr, 10);
  }
  if (argc >= 7) {
    attention.batch = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    attention.Hq = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    attention.Hk = (uint32_t)std::strtoul(argv[6], nullptr, 10);
    query_execution_simdgroups = create_backward_query_execution_simdgroups(attention);
    keyvalue_execution_simdgroups = create_backward_keyvalue_execution_simdgroups(attention);
  }
  if (argc >= 9) {
    config.warmup_iterations = std::atoi(argv[7]);
    config.timed_iterations = std::atoi(argv[8]);
  }
  if (argc >= 12) {
    query_block_r = (uint16_t)std::strtoul(argv[9], nullptr, 10);
    query_block_c = (uint16_t)std::strtoul(argv[10], nullptr, 10);
    query_block_d = (uint16_t)std::strtoul(argv[11], nullptr, 10);
  }
  if (argc >= 15) {
    keyvalue_block_r = (uint16_t)std::strtoul(argv[12], nullptr, 10);
    keyvalue_block_c = (uint16_t)std::strtoul(argv[13], nullptr, 10);
    keyvalue_block_d = (uint16_t)std::strtoul(argv[14], nullptr, 10);
  }
  if (argc >= 16) {
    query_execution_simdgroups = (uint16_t)std::strtoul(argv[15], nullptr, 10);
    keyvalue_execution_simdgroups = query_execution_simdgroups;
  }
  if (argc >= 17) {
    keyvalue_execution_simdgroups = (uint16_t)std::strtoul(argv[16], nullptr, 10);
  }

  const simd::ushort3 query_block_dimensions = create_backward_query_block_dimensions(
      attention, query_block_r, query_block_c, query_block_d);
  const simd::ushort3 keyvalue_block_dimensions = create_backward_keyvalue_block_dimensions(
      attention, keyvalue_block_r, keyvalue_block_c, keyvalue_block_d);
  const auto valid_block_d =
      [&](uint16_t block_d) {
        return block_d > 0 &&
            block_d <= attention.D &&
            (attention.D % block_d) == 0;
      };
  if (!valid_block_d(query_block_dimensions[2]) || !valid_block_d(keyvalue_block_dimensions[2])) {
    std::cerr << "invalid blockD: current int8 backward kernels require blockD to divide D\n";
    return 2;
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
  const size_t l_bytes = (size_t)attention.batch * attention.Hq * attention.R * create_l_precision().size();
  const size_t dQ_bytes = o_bytes;
  const size_t dK_bytes = (size_t)attention.batch * attention.C * attention.Hk * attention.D * sizeof(half_float);
  const size_t dV_bytes = dK_bytes;

  const auto scratch_layout = create_scratch_layout(attention);

  auto q_stage = NS::TransferPtr(device->newBuffer(q_bytes_data.data(), q_bytes, kSharedResourceOptions));
  auto k_stage = NS::TransferPtr(device->newBuffer(k_bytes_data.data(), k_bytes, kSharedResourceOptions));
  auto v_stage = NS::TransferPtr(device->newBuffer(v_bytes_data.data(), v_bytes, kSharedResourceOptions));
  auto dO_stage = NS::TransferPtr(device->newBuffer(dO_bytes_data.data(), dO_bytes, kSharedResourceOptions));
  auto q_buffer = NS::TransferPtr(device->newBuffer(q_bytes, kPrivateResourceOptions));
  auto k_buffer = NS::TransferPtr(device->newBuffer(k_bytes, kPrivateResourceOptions));
  auto v_buffer = NS::TransferPtr(device->newBuffer(v_bytes, kPrivateResourceOptions));
  auto dO_buffer = NS::TransferPtr(device->newBuffer(dO_bytes, kPrivateResourceOptions));
  auto scratch = NS::TransferPtr(device->newBuffer(scratch_layout.total, kPrivateResourceOptions));
  auto o_buffer = NS::TransferPtr(device->newBuffer(o_bytes, kPrivateResourceOptions));
  auto l_buffer = NS::TransferPtr(device->newBuffer(l_bytes, kPrivateResourceOptions));
  auto dQ_buffer = NS::TransferPtr(device->newBuffer(dQ_bytes, kPrivateResourceOptions));
  auto dK_buffer = NS::TransferPtr(device->newBuffer(dK_bytes, kPrivateResourceOptions));
  auto dV_buffer = NS::TransferPtr(device->newBuffer(dV_bytes, kPrivateResourceOptions));

  upload_buffer(command_queue.get(), q_stage.get(), q_buffer.get(), q_bytes);
  upload_buffer(command_queue.get(), k_stage.get(), k_buffer.get(), k_bytes);
  upload_buffer(command_queue.get(), v_stage.get(), v_buffer.get(), v_bytes);
  upload_buffer(command_queue.get(), dO_stage.get(), dO_buffer.get(), dO_bytes);

  const auto quantize_pipelines = create_quantize_pipelines(device.get(), attention);
  const auto forward_pipeline = create_forward_pipeline(device.get(), attention);
  const auto backward_pipelines = create_backward_pipelines(
      device.get(),
      attention,
      query_block_dimensions,
      keyvalue_block_dimensions,
      query_execution_simdgroups,
      keyvalue_execution_simdgroups);

  const double setup_forward_seconds = run_forward_total_once(
      command_queue.get(), attention, quantize_pipelines, forward_pipeline, scratch_layout,
      q_buffer.get(), k_buffer.get(), v_buffer.get(), scratch.get(), o_buffer.get(), l_buffer.get());
  if (!(setup_forward_seconds > 0)) {
    std::cerr << "forward setup failed\n";
    pool->drain();
    return 1;
  }
  const double setup_prepare_seconds = run_prepare_backward_once(
      command_queue.get(), attention, quantize_pipelines, backward_pipelines, scratch_layout,
      q_buffer.get(), k_buffer.get(), v_buffer.get(), dO_buffer.get(), o_buffer.get(), scratch.get());
  if (!(setup_prepare_seconds > 0)) {
    std::cerr << "backward prepare setup failed\n";
    pool->drain();
    return 1;
  }

  Stats forward_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_forward_total_once(
                command_queue.get(), attention, quantize_pipelines, forward_pipeline, scratch_layout,
                q_buffer.get(), k_buffer.get(), v_buffer.get(), scratch.get(), o_buffer.get(), l_buffer.get());
          },
          &forward_stats)) {
    std::cerr << "forward benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats prepare_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_prepare_backward_once(
                command_queue.get(), attention, quantize_pipelines, backward_pipelines, scratch_layout,
                q_buffer.get(), k_buffer.get(), v_buffer.get(), dO_buffer.get(), o_buffer.get(), scratch.get());
          },
          &prepare_stats)) {
    std::cerr << "prepare benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats query_stats;
  if (!benchmark(
          config,
          [&]() {
            return run_backward_query_only_once(
                command_queue.get(), attention, backward_pipelines, scratch_layout,
                scratch.get(), l_buffer.get(), dQ_buffer.get());
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
                command_queue.get(), attention, backward_pipelines, scratch_layout,
                scratch.get(), l_buffer.get(), dK_buffer.get(), dV_buffer.get());
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
            return run_backward_total_once(
                command_queue.get(), attention, quantize_pipelines, backward_pipelines, scratch_layout,
                q_buffer.get(), k_buffer.get(), v_buffer.get(), dO_buffer.get(), o_buffer.get(), l_buffer.get(),
                scratch.get(), dQ_buffer.get(), dK_buffer.get(), dV_buffer.get());
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
            << " lowPrecisionIntermediates=" << (create_low_precision_intermediates() ? "true" : "false")
            << '\n';
  std::cout << "forward-kernel"
            << " blockR=" << forward_pipeline.kernel->blockDimensions[0]
            << " blockC=" << forward_pipeline.kernel->blockDimensions[1]
            << " blockD=" << forward_pipeline.kernel->blockDimensions[2]
            << " simdgroups=" << forward_pipeline.kernel->executionSIMDGroups
            << '\n';
  std::cout << "backward-kernel"
            << " queryBlockR=" << backward_pipelines.query_kernel->blockDimensions[0]
            << " queryBlockC=" << backward_pipelines.query_kernel->blockDimensions[1]
            << " queryBlockD=" << backward_pipelines.query_kernel->blockDimensions[2]
            << " querySimdgroups=" << backward_pipelines.query_kernel->executionSIMDGroups
            << " keyvalueBlockR=" << backward_pipelines.keyvalue_kernel->blockDimensions[0]
            << " keyvalueBlockC=" << backward_pipelines.keyvalue_kernel->blockDimensions[1]
            << " keyvalueBlockD=" << backward_pipelines.keyvalue_kernel->blockDimensions[2]
            << " keyvalueSimdgroups=" << backward_pipelines.keyvalue_kernel->executionSIMDGroups
            << '\n';
  print_stats("forward_total", forward_stats);
  print_stats("backward_prepare", prepare_stats);
  print_stats("backward_query", query_stats);
  print_stats("backward_keyvalue", keyvalue_stats);
  print_stats("backward_total", backward_stats);
  std::cout << std::fixed
            << "ratio"
            << " avg=" << std::setprecision(4) << backward_stats.average_seconds / forward_stats.average_seconds
            << " median=" << backward_stats.median_seconds / forward_stats.median_seconds
            << '\n';

  pool->drain();
  return 0;
}
