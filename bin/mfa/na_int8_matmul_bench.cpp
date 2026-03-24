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
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAMatMulDescriptor.hpp"
#include "nnc/mfa/kernels/NAMatMulKernel.hpp"
#include "nnc/mfa/kernels/NAMatMulKernelDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernel.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.hpp"

namespace {

using half_float = _Float16;

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
};

struct BenchmarkCase {
  uint32_t M = 4096;
  uint32_t N = 4096;
  uint32_t K = 4096;
};

struct VariantConfig {
  simd::ushort3 block_dimensions = simd::ushort3 { 128, 128, 128 };
  uint16_t execution_simd_groups = 8;
  uint16_t activation_quant_threads = 256;
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
  size_t checked_rows = 0;
  size_t checked_cols = 0;
  double max_abs = 0;
  double max_rel = 0;
};

struct QuantizationValidationStats {
  bool passed = false;
  size_t mismatched_values = 0;
  double max_abs_scale = 0;
};

struct RowwiseQuantizedMatrix {
  std::vector<int8_t> values;
  std::vector<float> scales;
};

struct BaselinePipeline {
  NAMatMulDescriptor descriptor;
  std::unique_ptr<NAMatMulKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct DynamicPipeline {
  std::unique_ptr<NAInt8MatMulKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct QuantizePipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  uint16_t threadgroup_size = 0;
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

size_t a_index(const BenchmarkCase& bench, uint32_t row, uint32_t k)
{
  return (size_t)row * bench.K + k;
}

size_t b_index(const BenchmarkCase& bench, uint32_t row, uint32_t k)
{
  return (size_t)row * bench.K + k;
}

size_t c_index(const BenchmarkCase& bench, uint32_t row, uint32_t col)
{
  return (size_t)row * bench.N + col;
}

std::vector<half_float> make_half_matrix(uint32_t rows, uint32_t cols, float scale, int phase)
{
  std::vector<half_float> values((size_t)rows * cols);
  for (uint32_t row = 0; row < rows; ++row) {
    const float row_gain = 0.2f + 0.8f * (float)(((row * 7 + phase * 5) % 23) + 1) / 24.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      const int centered = (int)((row * 131 + col * 17 + phase * 29) % 127) - 63;
      values[(size_t)row * cols + col] = (half_float)(centered * scale * row_gain);
    }
  }
  return values;
}

std::vector<float> half_to_float_vector(const std::vector<half_float>& values)
{
  std::vector<float> output(values.size());
  std::transform(values.begin(), values.end(), output.begin(), [](half_float value) {
    return (float)value;
  });
  return output;
}

RowwiseQuantizedMatrix quantize_rowwise(
    const std::vector<float>& values,
    uint32_t rows,
    uint32_t cols)
{
  RowwiseQuantizedMatrix quantized;
  quantized.values.resize(values.size());
  quantized.scales.resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    float max_abs = 0;
    for (uint32_t col = 0; col < cols; ++col)
      max_abs = std::max(max_abs, std::fabs(values[(size_t)row * cols + col]));
    const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
    const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
    quantized.scales[row] = scale;
    for (uint32_t col = 0; col < cols; ++col) {
      const int rounded = (int)std::lrint(values[(size_t)row * cols + col] * inv_scale);
      quantized.values[(size_t)row * cols + col] = (int8_t)std::max(-127, std::min(127, rounded));
    }
  }
  return quantized;
}

uint32_t groupM(uint32_t M) noexcept
{
  return M >= 4096 ? 4096 : 0;
}

uint32_t groupN(uint32_t N) noexcept
{
  return N >= 4096 ? 4096 : 0;
}

BaselinePipeline create_baseline_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench)
{
  BaselinePipeline bundle;
  bundle.descriptor.batchDimension = 1;
  bundle.descriptor.matrixDimensions = simd::uint3 { bench.M, bench.N, bench.K };
  bundle.descriptor.memoryPrecisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  bundle.descriptor.registerPrecisionC = std::make_optional(GEMMOperandPrecision::FP16);
  bundle.descriptor.batchStrides = std::nullopt;
  bundle.descriptor.transposeState = simd::uchar3 { 0, 1, 0 };
  bundle.descriptor.useBias = false;
  bundle.descriptor.loadM = false;
  bundle.descriptor.supportIndirectCommandBuffers = false;

  const GEMMOperandPrecisions register_precisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  const NAMatMulKernelDescriptor kernel_descriptor(
      simd::ushort3 { 128, 64, 64 },
      bundle.descriptor.memoryPrecisions,
      register_precisions,
      1,
      4,
      false,
      bundle.descriptor.transposeState,
      false,
      false,
      groupM(bench.M),
      groupN(bench.N));
  bundle.kernel = std::make_unique<NAMatMulKernel>(kernel_descriptor, device);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t N = bench.N;
  const uint32_t K = bench.K;
  const bool batched = false;
  const uint32_t zero = 0;
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&batched, MTL::DataTypeBool, NS::UInteger(11));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(15));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(16));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(17));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(18));

  NS::Error* error = nil;
  auto function_name = NS::String::string("matmul", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(bundle.kernel->library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  bundle.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

DynamicPipeline create_dynamic_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    const VariantConfig& variant)
{
  DynamicPipeline bundle;
  const NAInt8MatMulKernelDescriptor kernel_descriptor(
      variant.block_dimensions,
      variant.execution_simd_groups,
      true,
      groupM(bench.M),
      groupN(bench.N));
  bundle.kernel = std::make_unique<NAInt8MatMulKernel>(kernel_descriptor, device);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t N = bench.N;
  const uint32_t K = bench.K;
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));

  NS::Error* error = nil;
  auto function_name = NS::String::string("int8_matmul", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(bundle.kernel->library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  bundle.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

QuantizePipeline create_quantize_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    const VariantConfig& variant)
{
  CCV_NNC_MFA_PRECONDITION(variant.activation_quant_threads % 32 == 0);
  const bool vectorize4 = (bench.K % 4) == 0;
  const uint16_t quant_simdgroups = variant.activation_quant_threads / 32;
  std::ostringstream source;
  source
      << "#include <metal_stdlib>\n"
      << "using namespace metal;\n"
      << "constant uint M [[function_constant(0)]];\n"
      << "constant uint K [[function_constant(1)]];\n"
      << "inline float quantize_reduce_max(float value,\n"
      << "                                 threadgroup float* scratch,\n"
      << "                                 ushort sgid,\n"
      << "                                 ushort lane_id)\n"
      << "{\n"
      << "  value = max(value, simd_shuffle_xor(value, 16));\n"
      << "  value = max(value, simd_shuffle_xor(value, 8));\n"
      << "  value = max(value, simd_shuffle_xor(value, 4));\n"
      << "  value = max(value, simd_shuffle_xor(value, 2));\n"
      << "  value = max(value, simd_shuffle_xor(value, 1));\n"
      << "  if (lane_id == 0)\n"
      << "    scratch[sgid] = value;\n"
      << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "  if (sgid == 0) {\n"
      << "    value = lane_id < " << quant_simdgroups << " ? scratch[lane_id] : 0.0f;\n"
      << "    value = max(value, simd_shuffle_xor(value, 16));\n"
      << "    value = max(value, simd_shuffle_xor(value, 8));\n"
      << "    value = max(value, simd_shuffle_xor(value, 4));\n"
      << "    value = max(value, simd_shuffle_xor(value, 2));\n"
      << "    value = max(value, simd_shuffle_xor(value, 1));\n"
      << "    if (lane_id == 0)\n"
      << "      scratch[0] = value;\n"
      << "  }\n"
      << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "  return scratch[0];\n"
      << "}\n";
  if (vectorize4) {
    source
        << "kernel void quantize_activation(\n"
        << "    device const half* src [[buffer(0)]],\n"
        << "    device int8_t* dst [[buffer(1)]],\n"
        << "    device half* scales [[buffer(2)]],\n"
        << "    uint tid [[thread_index_in_threadgroup]],\n"
        << "    ushort sgid [[simdgroup_index_in_threadgroup]],\n"
        << "    ushort lane_id [[thread_index_in_simdgroup]],\n"
        << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
        << "{\n"
        << "  threadgroup float scratch[" << quant_simdgroups << "];\n"
        << "  const uint row = tgid.x;\n"
        << "  if (row >= M)\n"
        << "    return;\n"
        << "  const uint vectors_per_row = K / 4;\n"
        << "  device const half4* src4 = reinterpret_cast<device const half4*>(src);\n"
        << "  device char4* dst4 = reinterpret_cast<device char4*>(dst);\n"
        << "  float local_max = 0.0f;\n"
        << "  const uint base = row * vectors_per_row;\n"
        << "  for (uint i = tid; i < vectors_per_row; i += " << variant.activation_quant_threads << ") {\n"
        << "    const float4 value = float4(src4[base + i]);\n"
        << "    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])), max(fabs(value[2]), fabs(value[3]))));\n"
        << "  }\n"
        << "  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);\n"
        << "  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);\n"
        << "  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;\n"
        << "  if (tid == 0)\n"
        << "    scales[row] = half(scale);\n"
        << "  for (uint i = tid; i < vectors_per_row; i += " << variant.activation_quant_threads << ") {\n"
        << "    const int4 rounded = int4(rint(float4(src4[base + i]) * inv_scale));\n"
        << "    dst4[base + i] = char4(clamp(rounded, int4(-127), int4(127)));\n"
        << "  }\n"
        << "}\n";
  } else {
    source
        << "kernel void quantize_activation(\n"
        << "    device const half* src [[buffer(0)]],\n"
        << "    device int8_t* dst [[buffer(1)]],\n"
        << "    device half* scales [[buffer(2)]],\n"
        << "    uint tid [[thread_index_in_threadgroup]],\n"
        << "    ushort sgid [[simdgroup_index_in_threadgroup]],\n"
        << "    ushort lane_id [[thread_index_in_simdgroup]],\n"
        << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
        << "{\n"
        << "  threadgroup float scratch[" << quant_simdgroups << "];\n"
        << "  const uint row = tgid.x;\n"
        << "  if (row >= M)\n"
        << "    return;\n"
        << "  float local_max = 0.0f;\n"
        << "  const uint base = row * K;\n"
        << "  for (uint i = tid; i < K; i += " << variant.activation_quant_threads << ")\n"
        << "    local_max = max(local_max, fabs((float)src[base + i]));\n"
        << "  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);\n"
        << "  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);\n"
        << "  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;\n"
        << "  if (tid == 0)\n"
        << "    scales[row] = half(scale);\n"
        << "  for (uint i = tid; i < K; i += " << variant.activation_quant_threads << ") {\n"
        << "    const int rounded = (int)rint((float)src[base + i] * inv_scale);\n"
        << "    dst[base + i] = (int8_t)clamp(rounded, -127, 127);\n"
        << "  }\n"
        << "}\n";
  }

  auto string = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t K = bench.K;
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(1));

  auto function_name = NS::String::string("quantize_activation", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());

  QuantizePipeline pipeline;
  pipeline.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  pipeline.threadgroup_size = variant.activation_quant_threads;
  return pipeline;
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

double run_baseline_once(
    MTL::CommandQueue* command_queue,
    const BaselinePipeline& bundle,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_b,
    MTL::Buffer* buffer_c)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setBuffer(buffer_a, 0, 0);
  encoder->setBuffer(buffer_b, 0, 1);
  encoder->setBuffer(buffer_c, 0, 2);
  encoder->dispatchThreadgroups(
      bundle.kernel->threadgroupsPerGrid(bundle.descriptor),
      MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get(), bundle.descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_quantize_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const QuantizePipeline& bundle,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_q,
    MTL::Buffer* buffer_scales)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setBuffer(buffer_a, 0, 0);
  encoder->setBuffer(buffer_q, 0, 1);
  encoder->setBuffer(buffer_scales, 0, 2);
  encoder->dispatchThreadgroups(
      MTL::Size(bench.M, 1, 1),
      MTL::Size(bundle.threadgroup_size, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_dynamic_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const DynamicPipeline& bundle,
    MTL::Buffer* buffer_a_q,
    MTL::Buffer* buffer_b_q,
    MTL::Buffer* buffer_c,
    MTL::Buffer* buffer_a_scale,
    MTL::Buffer* buffer_b_scale)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setBuffer(buffer_a_q, 0, 0);
  encoder->setBuffer(buffer_b_q, 0, 1);
  encoder->setBuffer(buffer_c, 0, 2);
  encoder->setBuffer(buffer_a_scale, 0, 3);
  encoder->setBuffer(buffer_b_scale, 0, 4);
  encoder->dispatchThreadgroups(
      bundle.kernel->threadgroupsPerGrid(bench.M, bench.N),
      MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get()), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_quantize_and_dynamic_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const QuantizePipeline& quantize,
    const DynamicPipeline& dynamic,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_a_q,
    MTL::Buffer* buffer_a_scale,
    MTL::Buffer* buffer_b_q,
    MTL::Buffer* buffer_b_scale,
    MTL::Buffer* buffer_c)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(quantize.pipeline.get());
    encoder->setBuffer(buffer_a, 0, 0);
    encoder->setBuffer(buffer_a_q, 0, 1);
    encoder->setBuffer(buffer_a_scale, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(bench.M, 1, 1),
        MTL::Size(quantize.threadgroup_size, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(dynamic.pipeline.get());
    encoder->setBuffer(buffer_a_q, 0, 0);
    encoder->setBuffer(buffer_b_q, 0, 1);
    encoder->setBuffer(buffer_c, 0, 2);
    encoder->setBuffer(buffer_a_scale, 0, 3);
    encoder->setBuffer(buffer_b_scale, 0, 4);
    encoder->dispatchThreadgroups(
        dynamic.kernel->threadgroupsPerGrid(bench.M, bench.N),
        MTL::Size(dynamic.kernel->threadgroupSize(dynamic.pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

template <typename RunOnce>
bool benchmark(const BenchmarkConfig& config, RunOnce&& run_once, Stats* stats)
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

QuantizationValidationStats validate_quantization(
    const RowwiseQuantizedMatrix& reference,
    const int8_t* values,
    const half_float* scales)
{
  QuantizationValidationStats stats;
  for (size_t i = 0; i < reference.values.size(); ++i)
    stats.mismatched_values += (reference.values[i] != values[i]);
  for (size_t i = 0; i < reference.scales.size(); ++i)
    stats.max_abs_scale = std::max(stats.max_abs_scale, std::fabs((double)(half_float)reference.scales[i] - (double)scales[i]));
  stats.passed = (stats.mismatched_values == 0 && stats.max_abs_scale <= 1e-5);
  return stats;
}

float compute_quantized_reference_value(
    const BenchmarkCase& bench,
    const RowwiseQuantizedMatrix& a_quantized,
    const RowwiseQuantizedMatrix& b_quantized,
    uint32_t row,
    uint32_t col)
{
  float accumulator = 0;
  for (uint32_t k = 0; k < bench.K; ++k)
    accumulator +=
        (int32_t)a_quantized.values[a_index(bench, row, k)] *
        (int32_t)b_quantized.values[b_index(bench, col, k)];
  accumulator *= a_quantized.scales[row] * b_quantized.scales[col];
  return accumulator;
}

float compute_float_reference_value(
    const BenchmarkCase& bench,
    const std::vector<float>& a_values,
    const std::vector<float>& b_values,
    uint32_t row,
    uint32_t col)
{
  float accumulator = 0;
  for (uint32_t k = 0; k < bench.K; ++k)
    accumulator +=
        a_values[a_index(bench, row, k)] *
        b_values[b_index(bench, col, k)];
  return accumulator;
}

ValidationStats validate_output(
    const BenchmarkCase& bench,
    const std::vector<float>& a_values,
    const std::vector<float>& b_values,
    const RowwiseQuantizedMatrix& a_quantized,
    const RowwiseQuantizedMatrix& b_quantized,
    const float* output,
    bool quantized_reference,
    bool half_reference)
{
  ValidationStats stats;
  const uint64_t reference_work = (uint64_t)bench.M * bench.N * bench.K;
  stats.full_reference = reference_work <= (1ull << 26);
  const auto row_points = stats.full_reference ? make_sample_points(bench.M, {}) : make_sample_points(bench.M, { 0, 1, 127, 128, 1023, 1024, 4095, 4096 });
  const auto col_points = stats.full_reference ? make_sample_points(bench.N, {}) : make_sample_points(bench.N, { 0, 1, 63, 64, 1023, 1024, 4095, 4096 });
  for (const auto row : row_points) {
    for (const auto col : col_points) {
      const float reference = quantized_reference ?
          compute_quantized_reference_value(bench, a_quantized, b_quantized, row, col) :
          compute_float_reference_value(bench, a_values, b_values, row, col);
      const float compared_reference = half_reference ? (float)(half_float)reference : reference;
      const float actual = output[c_index(bench, row, col)];
      const double abs_diff = std::fabs(compared_reference - actual);
      const double rel_diff = abs_diff / std::max<double>(std::max(std::fabs(compared_reference), std::fabs(actual)), 1.0);
      stats.max_abs = std::max(stats.max_abs, abs_diff);
      stats.max_rel = std::max(stats.max_rel, rel_diff);
    }
  }
  stats.checked_rows = row_points.size();
  stats.checked_cols = col_points.size();
  if (quantized_reference)
    stats.passed = half_reference ? (stats.max_abs <= 5e-3 || stats.max_rel <= 5e-3)
                                  : (stats.max_abs <= 1e-4 || stats.max_rel <= 1e-4);
  else
    stats.passed = true;
  return stats;
}

void print_stats(const char* label, const BenchmarkCase& bench, const Stats& stats)
{
  const double flops = 2.0 * (double)bench.M * bench.N * bench.K;
  std::cout << label
            << " avg_ms=" << std::fixed << std::setprecision(3) << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " avg_gflops=" << flops / stats.average_seconds / 1e9
            << '\n';
}

void print_quantization_validation(const QuantizationValidationStats& stats)
{
  std::cout << "quant-validation"
            << " mismatched_values=" << stats.mismatched_values
            << " max_abs_scale=" << stats.max_abs_scale
            << '\n';
}

void print_validation(const char* label, const ValidationStats& stats)
{
  std::cout << label
            << " mode=" << (stats.full_reference ? "full" : "sampled")
            << " rows=" << stats.checked_rows
            << " cols=" << stats.checked_cols
            << " max_abs=" << stats.max_abs
            << " max_rel=" << stats.max_rel
            << '\n';
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  BenchmarkCase bench;
  BenchmarkConfig config;
  VariantConfig variant;
  if (argc >= 4) {
    bench.M = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    bench.N = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    bench.K = (uint32_t)std::strtoul(argv[3], nullptr, 10);
  }
  if (argc >= 6) {
    config.warmup_iterations = std::atoi(argv[4]);
    config.timed_iterations = std::atoi(argv[5]);
  }
  if (argc >= 9) {
    variant.block_dimensions = simd::ushort3 {
      (uint16_t)std::strtoul(argv[6], nullptr, 10),
      (uint16_t)std::strtoul(argv[7], nullptr, 10),
      (uint16_t)std::strtoul(argv[8], nullptr, 10),
    };
  }
  if (argc >= 10)
    variant.execution_simd_groups = (uint16_t)std::strtoul(argv[9], nullptr, 10);
  if (argc >= 11)
    variant.activation_quant_threads = (uint16_t)std::strtoul(argv[10], nullptr, 10);

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

  const auto a_half_values = make_half_matrix(bench.M, bench.K, 0.03125f, 1);
  const auto b_half_values = make_half_matrix(bench.N, bench.K, 0.015625f, 2);
  const auto a_float_values = half_to_float_vector(a_half_values);
  const auto b_float_values = half_to_float_vector(b_half_values);
  const auto a_quantized_reference = quantize_rowwise(a_float_values, bench.M, bench.K);
  const auto b_quantized_reference = quantize_rowwise(b_float_values, bench.N, bench.K);

  const size_t a_half_bytes = a_half_values.size() * sizeof(half_float);
  const size_t b_half_bytes = b_half_values.size() * sizeof(half_float);
  const size_t a_int8_bytes = a_quantized_reference.values.size() * sizeof(int8_t);
  const size_t b_int8_bytes = b_quantized_reference.values.size() * sizeof(int8_t);
  const size_t a_scale_bytes = a_quantized_reference.scales.size() * sizeof(half_float);
  const size_t b_scale_bytes = b_quantized_reference.scales.size() * sizeof(half_float);
  const size_t c_half_bytes = (size_t)bench.M * bench.N * sizeof(half_float);
  const size_t c_dynamic_half_bytes = (size_t)bench.M * bench.N * sizeof(half_float);

  std::vector<half_float> b_half_scales(b_quantized_reference.scales.size());
  std::transform(b_quantized_reference.scales.begin(), b_quantized_reference.scales.end(), b_half_scales.begin(), [](float value) {
    return (half_float)value;
  });

  auto a_half_stage = NS::TransferPtr(device->newBuffer(a_half_values.data(), a_half_bytes, kSharedResourceOptions));
  auto b_half_stage = NS::TransferPtr(device->newBuffer(b_half_values.data(), b_half_bytes, kSharedResourceOptions));
  auto b_int8_stage = NS::TransferPtr(device->newBuffer(b_quantized_reference.values.data(), b_int8_bytes, kSharedResourceOptions));
  auto b_scale_stage = NS::TransferPtr(device->newBuffer(b_half_scales.data(), b_scale_bytes, kSharedResourceOptions));
  auto c_half_stage = NS::TransferPtr(device->newBuffer(c_half_bytes, kSharedResourceOptions));
  auto a_int8_stage = NS::TransferPtr(device->newBuffer(a_int8_bytes, kSharedResourceOptions));
  auto a_scale_stage = NS::TransferPtr(device->newBuffer(a_scale_bytes, kSharedResourceOptions));
  auto c_dynamic_half_stage = NS::TransferPtr(device->newBuffer(c_dynamic_half_bytes, kSharedResourceOptions));

  auto a_half_buffer = NS::TransferPtr(device->newBuffer(a_half_bytes, kPrivateResourceOptions));
  auto b_half_buffer = NS::TransferPtr(device->newBuffer(b_half_bytes, kPrivateResourceOptions));
  auto a_int8_buffer = NS::TransferPtr(device->newBuffer(a_int8_bytes, kPrivateResourceOptions));
  auto b_int8_buffer = NS::TransferPtr(device->newBuffer(b_int8_bytes, kPrivateResourceOptions));
  auto a_scale_buffer = NS::TransferPtr(device->newBuffer(a_scale_bytes, kPrivateResourceOptions));
  auto b_scale_buffer = NS::TransferPtr(device->newBuffer(b_scale_bytes, kPrivateResourceOptions));
  auto c_half_buffer = NS::TransferPtr(device->newBuffer(c_half_bytes, kPrivateResourceOptions));
  auto c_dynamic_half_buffer = NS::TransferPtr(device->newBuffer(c_dynamic_half_bytes, kPrivateResourceOptions));

  upload_buffer(command_queue.get(), a_half_stage.get(), a_half_buffer.get(), a_half_bytes);
  upload_buffer(command_queue.get(), b_half_stage.get(), b_half_buffer.get(), b_half_bytes);
  upload_buffer(command_queue.get(), b_int8_stage.get(), b_int8_buffer.get(), b_int8_bytes);
  upload_buffer(command_queue.get(), b_scale_stage.get(), b_scale_buffer.get(), b_scale_bytes);

  auto baseline = create_baseline_pipeline(device.get(), bench);
  auto quantize = create_quantize_pipeline(device.get(), bench, variant);
  auto dynamic = create_dynamic_pipeline(device.get(), bench, variant);

  std::cout << "shape"
            << " M=" << bench.M
            << " N=" << bench.N
            << " K=" << bench.K
            << " warmup=" << config.warmup_iterations
            << " timed=" << config.timed_iterations
            << " blockM=" << variant.block_dimensions[0]
            << " blockN=" << variant.block_dimensions[1]
            << " blockK=" << variant.block_dimensions[2]
            << " simdgroups=" << variant.execution_simd_groups
            << " quantThreads=" << variant.activation_quant_threads
            << '\n';

  const double quantize_validation_seconds =
      run_quantize_once(command_queue.get(), bench, quantize, a_half_buffer.get(), a_int8_buffer.get(), a_scale_buffer.get());
  if (!(quantize_validation_seconds > 0)) {
    std::cerr << "activation quantization dispatch failed\n";
    pool->drain();
    return 1;
  }
  download_buffer(command_queue.get(), a_int8_buffer.get(), a_int8_stage.get(), a_int8_bytes);
  download_buffer(command_queue.get(), a_scale_buffer.get(), a_scale_stage.get(), a_scale_bytes);
  const auto quantization_validation = validate_quantization(
      a_quantized_reference,
      (const int8_t*)a_int8_stage->contents(),
      (const half_float*)a_scale_stage->contents());
  print_quantization_validation(quantization_validation);
  if (!quantization_validation.passed) {
    std::cerr << "activation quantization validation failed\n";
    pool->drain();
    return 1;
  }

  const double dynamic_validation_seconds =
      run_dynamic_once(
          command_queue.get(),
          bench,
          dynamic,
          a_int8_buffer.get(),
          b_int8_buffer.get(),
          c_dynamic_half_buffer.get(),
          a_scale_buffer.get(),
          b_scale_buffer.get());
  if (!(dynamic_validation_seconds > 0)) {
    std::cerr << "dynamic int8 matmul dispatch failed\n";
    pool->drain();
    return 1;
  }
  download_buffer(command_queue.get(), c_dynamic_half_buffer.get(), c_dynamic_half_stage.get(), c_dynamic_half_bytes);
  std::vector<float> c_output((size_t)bench.M * bench.N);
  {
    const auto* c_half_output = (const half_float*)c_dynamic_half_stage->contents();
    std::transform(c_half_output, c_half_output + c_output.size(), c_output.begin(), [](half_float value) {
      return (float)value;
    });
  }
  const auto exact_validation = validate_output(
      bench,
      a_float_values,
      b_float_values,
      a_quantized_reference,
      b_quantized_reference,
      c_output.data(),
      true,
      true);
  print_validation("exact-validation", exact_validation);
  if (!exact_validation.passed) {
    std::cerr << "dynamic int8 matmul exact validation failed\n";
    pool->drain();
    return 1;
  }
  const auto accuracy_validation = validate_output(
      bench,
      a_float_values,
      b_float_values,
      a_quantized_reference,
      b_quantized_reference,
      c_output.data(),
      false,
      false);
  print_validation("float-reference", accuracy_validation);

  Stats baseline_stats;
  if (!benchmark(config, [&]() {
        return run_baseline_once(command_queue.get(), baseline, a_half_buffer.get(), b_half_buffer.get(), c_half_buffer.get());
      }, &baseline_stats)) {
    std::cerr << "baseline benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats quantize_stats;
  if (!benchmark(config, [&]() {
        return run_quantize_once(command_queue.get(), bench, quantize, a_half_buffer.get(), a_int8_buffer.get(), a_scale_buffer.get());
      }, &quantize_stats)) {
    std::cerr << "quantize benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats dynamic_stats;
  if (!benchmark(config, [&]() {
        return run_dynamic_once(
            command_queue.get(),
            bench,
            dynamic,
            a_int8_buffer.get(),
            b_int8_buffer.get(),
            c_dynamic_half_buffer.get(),
            a_scale_buffer.get(),
            b_scale_buffer.get());
      }, &dynamic_stats)) {
    std::cerr << "dynamic int8 benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats combined_stats;
  if (!benchmark(config, [&]() {
        return run_quantize_and_dynamic_once(
            command_queue.get(),
            bench,
            quantize,
            dynamic,
            a_half_buffer.get(),
            a_int8_buffer.get(),
            a_scale_buffer.get(),
            b_int8_buffer.get(),
            b_scale_buffer.get(),
            c_dynamic_half_buffer.get());
      }, &combined_stats)) {
    std::cerr << "combined benchmark failed\n";
    pool->drain();
    return 1;
  }

  print_stats("baseline-fp16", bench, baseline_stats);
  print_stats("quantize-activation", bench, quantize_stats);
  print_stats("int8-int8-inline-dequant", bench, dynamic_stats);
  print_stats("quantize-plus-int8", bench, combined_stats);
  std::cout << "speedup"
            << " kernel_avg=" << baseline_stats.average_seconds / dynamic_stats.average_seconds
            << " kernel_median=" << baseline_stats.median_seconds / dynamic_stats.median_seconds
            << " end_to_end_avg=" << baseline_stats.average_seconds / combined_stats.average_seconds
            << " end_to_end_median=" << baseline_stats.median_seconds / combined_stats.median_seconds
            << '\n';
  std::cout.flush();
  std::_Exit(0);
}
