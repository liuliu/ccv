#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
}
#include "nnc/ccv_nnc_8i_rowwise_packed_grids.inc"
#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseDescriptor.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseKernel.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseXFPDescriptor.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseXFPKernel.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseXDescriptor.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseXKernel.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernel.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.hpp"

namespace {

struct BenchmarkConfig {
  uint32_t M = 4096;
  uint32_t N = 4096;
  uint32_t K = 4096;
  int warmup_iterations = 3;
  int timed_iterations = 10;
  int dequant_repeats = 1;
  bool all_formats = false;
  bool skip_validation = false;
  bool synthetic_packed = false;
  std::string format_name = "q4_k";
};

struct FormatInfo {
  uint32_t value;
  const char* name;
};

struct Stats {
  double average_seconds = 0;
  double median_seconds = 0;
  double best3_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct TimingStats {
  Stats gpu;
  Stats wall;
};

struct ValidationStats {
  bool passed = false;
  size_t checked = 0;
  double max_abs = 0;
};

constexpr FormatInfo kFormats[] = {
  { CCV_NNC_QX_8I_ROWWISE_Q5_K, "q5_k" },
  { CCV_NNC_QX_8I_ROWWISE_Q6_K, "q6_k" },
  { CCV_NNC_QX_8I_ROWWISE_Q4_K, "q4_k" },
  { CCV_NNC_QX_8I_ROWWISE_Q3_K, "q3_k" },
  { CCV_NNC_QX_8I_ROWWISE_Q2_K, "q2_k" },
  { CCV_NNC_QX_8I_ROWWISE_IQ2_S, "iq2_s" },
  { CCV_NNC_QX_8I_ROWWISE_IQ2_XS, "iq2_xs" },
  { CCV_NNC_QX_8I_ROWWISE_IQ3_S, "iq3_s" },
  { CCV_NNC_QX_8I_ROWWISE_IQ3_XXS, "iq3_xxs" },
  { CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, "iq2_xxs" },
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

size_t align_up(const size_t value, const size_t alignment)
{
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t rowwise_8i_scale_offset(const uint32_t rows, const uint32_t cols)
{
  return align_up((size_t)rows * cols * sizeof(int8_t), 128);
}

size_t rowwise_8i_data_size(const uint32_t rows, const uint32_t cols)
{
  return rowwise_8i_scale_offset(rows, cols) + (size_t)rows * sizeof(uint16_t);
}

std::vector<uint16_t> make_half_weights(const uint32_t rows, const uint32_t cols)
{
  std::vector<float> values((size_t)rows * cols);
  for (uint32_t row = 0; row < rows; ++row) {
    const float row_gain = 0.35f + 0.65f * (float)(((row * 7) % 23) + 1) / 24.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      const int centered = (int)((row * 131 + col * 17 + 29) % 255) - 127;
      values[(size_t)row * cols + col] = centered * row_gain / 512.0f;
    }
  }
  std::vector<uint16_t> half(values.size());
  ccv_float_to_half_precision(values.data(), half.data(), half.size());
  return half;
}

std::vector<int8_t> make_int8_values(const uint32_t rows, const uint32_t cols)
{
  std::vector<int8_t> values((size_t)rows * cols);
  for (size_t i = 0; i < values.size(); ++i) {
    const int value = (int)((i * 37 + 11) % 255) - 127;
    values[i] = (int8_t)value;
  }
  return values;
}

std::vector<uint16_t> make_half_scales(const uint32_t rows)
{
  std::vector<float> scales(rows);
  for (uint32_t i = 0; i < rows; ++i)
    scales[i] = (0.75f + (float)((i * 13) % 17) / 32.0f) / 127.0f;
  std::vector<uint16_t> half(rows);
  ccv_float_to_half_precision(scales.data(), half.data(), half.size());
  return half;
}

void upload_buffer(
    MTL::CommandQueue* const command_queue,
    MTL::Buffer* const source,
    MTL::Buffer* const destination,
    const size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

NS::SharedPtr<MTL::Buffer> make_private_buffer(
    MTL::Device* const device,
    MTL::CommandQueue* const command_queue,
    const void* const data,
    const size_t size)
{
  auto stage = NS::TransferPtr(device->newBuffer(size, kSharedResourceOptions));
  auto buffer = NS::TransferPtr(device->newBuffer(size, kPrivateResourceOptions));
  std::memcpy(stage->contents(), data, size);
  upload_buffer(command_queue, stage.get(), buffer.get(), size);
  return buffer;
}

std::vector<uint8_t> download_buffer(
    MTL::Device* const device,
    MTL::CommandQueue* const command_queue,
    MTL::Buffer* const source,
    const size_t size)
{
  auto stage = NS::TransferPtr(device->newBuffer(size, kSharedResourceOptions));
  upload_buffer(command_queue, source, stage.get(), size);
  std::vector<uint8_t> data(size);
  std::memcpy(data.data(), stage->contents(), size);
  return data;
}

double run_dequant_once(
    MTL::CommandQueue* const command_queue,
    PipelineValue<Dequantize8iRowwiseXKernel>* const pipeline,
    const Dequantize8iRowwiseXDescriptor& descriptor,
    MTL::Buffer* const source,
    MTL::Buffer* const destination,
    const int repeats = 1)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline->pipeline.get());
  encoder->useResource(source, MTL::ResourceUsageRead);
  encoder->useResource(destination, MTL::ResourceUsageWrite);
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(destination, 0, 1);
  encoder->setBuffer(source, descriptor.inputScaleOffset(), 2);
  encoder->setBuffer(destination, descriptor.outputScaleOffset(), 3);
  for (int i = 0; i < repeats; ++i)
    encoder->dispatchThreadgroups(
        pipeline->kernel->gridSize(descriptor.dispatchItems()),
        pipeline->kernel->threadgroupSize);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) / repeats;
}

double run_rowwise_fp16_dequant_once(
    MTL::CommandQueue* const command_queue,
    PipelineValue<Dequantize8iRowwiseKernel>* const pipeline,
    const Dequantize8iRowwiseDescriptor& descriptor,
    MTL::Buffer* const source,
    MTL::Buffer* const destination,
    const int repeats = 1)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline->pipeline.get());
  encoder->useResource(source, MTL::ResourceUsageRead);
  encoder->useResource(destination, MTL::ResourceUsageWrite);
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(destination, 0, 1);
  encoder->setBuffer(source, rowwise_8i_scale_offset(descriptor.length / descriptor.rowLength, descriptor.rowLength), 2);
  for (int i = 0; i < repeats; ++i)
    encoder->dispatchThreadgroups(
        pipeline->kernel->gridSize(descriptor.length),
        pipeline->kernel->threadgroupSize);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) / repeats;
}

double run_x_fp_dequant_once(
    MTL::CommandQueue* const command_queue,
    PipelineValue<Dequantize8iRowwiseXFPKernel>* const pipeline,
    const Dequantize8iRowwiseXFPDescriptor& descriptor,
    MTL::Buffer* const source,
    MTL::Buffer* const destination,
    const int repeats = 1)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline->pipeline.get());
  encoder->useResource(source, MTL::ResourceUsageRead);
  encoder->useResource(destination, MTL::ResourceUsageWrite);
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(destination, 0, 1);
  encoder->setBuffer(source, descriptor.inputScaleOffset(), 2);
  for (int i = 0; i < repeats; ++i)
    encoder->dispatchThreadgroups(
        pipeline->kernel->gridSize(descriptor.totalGroups()),
        pipeline->kernel->threadgroupSize);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) / repeats;
}

double run_matmul_once(
    MTL::CommandQueue* const command_queue,
    PipelineValue<NAInt8MatMulKernel>* const pipeline,
    const BenchmarkConfig& config,
    MTL::Buffer* const a_values,
    MTL::Buffer* const a_scales,
    MTL::Buffer* const b_rowwise,
    const size_t b_scale_offset,
    MTL::Buffer* const c)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline->pipeline.get());
  encoder->useResource(a_values, MTL::ResourceUsageRead);
  encoder->useResource(a_scales, MTL::ResourceUsageRead);
  encoder->useResource(b_rowwise, MTL::ResourceUsageRead);
  encoder->useResource(c, MTL::ResourceUsageWrite);
  encoder->setBuffer(a_values, 0, 0);
  encoder->setBuffer(b_rowwise, 0, 1);
  encoder->setBuffer(c, 0, 2);
  encoder->setBuffer(a_scales, 0, 3);
  encoder->setBuffer(b_rowwise, b_scale_offset, 4);
  encoder->dispatchThreadgroups(
      pipeline->kernel->threadgroupsPerGrid(config.M, config.N, 1),
      MTL::Size(pipeline->kernel->threadgroupSize(pipeline->pipeline.get()), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_dequant_matmul_once(
    MTL::CommandQueue* const command_queue,
    PipelineValue<Dequantize8iRowwiseXKernel>* const dequant_pipeline,
    const Dequantize8iRowwiseXDescriptor& dequant_descriptor,
    PipelineValue<NAInt8MatMulKernel>* const matmul_pipeline,
    const BenchmarkConfig& config,
    MTL::Buffer* const b_x,
    MTL::Buffer* const b_rowwise_scratch,
    MTL::Buffer* const a_values,
    MTL::Buffer* const a_scales,
    const size_t b_scale_offset,
    MTL::Buffer* const c)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(dequant_pipeline->pipeline.get());
    encoder->useResource(b_x, MTL::ResourceUsageRead);
    encoder->useResource(b_rowwise_scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(b_x, 0, 0);
    encoder->setBuffer(b_rowwise_scratch, 0, 1);
    encoder->setBuffer(b_x, dequant_descriptor.inputScaleOffset(), 2);
    encoder->setBuffer(b_rowwise_scratch, dequant_descriptor.outputScaleOffset(), 3);
    encoder->dispatchThreadgroups(
        dequant_pipeline->kernel->gridSize(dequant_descriptor.dispatchItems()),
        dequant_pipeline->kernel->threadgroupSize);
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(matmul_pipeline->pipeline.get());
    encoder->useResource(a_values, MTL::ResourceUsageRead);
    encoder->useResource(a_scales, MTL::ResourceUsageRead);
    encoder->useResource(b_rowwise_scratch, MTL::ResourceUsageRead);
    encoder->useResource(c, MTL::ResourceUsageWrite);
    encoder->setBuffer(a_values, 0, 0);
    encoder->setBuffer(b_rowwise_scratch, 0, 1);
    encoder->setBuffer(c, 0, 2);
    encoder->setBuffer(a_scales, 0, 3);
    encoder->setBuffer(b_rowwise_scratch, b_scale_offset, 4);
    encoder->dispatchThreadgroups(
        matmul_pipeline->kernel->threadgroupsPerGrid(config.M, config.N, 1),
        MTL::Size(matmul_pipeline->kernel->threadgroupSize(matmul_pipeline->pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

Stats summarize_samples(std::vector<double> samples)
{
  if (samples.empty())
    return {};
  std::sort(samples.begin(), samples.end());
  Stats stats;
  stats.average_seconds = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  stats.median_seconds = samples[samples.size() / 2];
  const size_t best_count = std::min<size_t>(3, samples.size());
  stats.best3_seconds = std::accumulate(samples.begin(), samples.begin() + best_count, 0.0) / best_count;
  stats.min_seconds = samples.front();
  stats.max_seconds = samples.back();
  return stats;
}

TimingStats benchmark(const BenchmarkConfig& config, const std::function<double()>& run_once, const int wall_divisor = 1)
{
  for (int i = 0; i < config.warmup_iterations; ++i)
    (void)run_once();
  std::vector<double> gpu_samples;
  std::vector<double> wall_samples;
  gpu_samples.reserve(config.timed_iterations);
  wall_samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.timed_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const double seconds = run_once();
    const auto end = std::chrono::steady_clock::now();
    if (seconds > 0 && !std::isnan(seconds))
      gpu_samples.push_back(seconds);
    wall_samples.push_back(std::chrono::duration<double>(end - start).count() / wall_divisor);
  }
  return TimingStats { summarize_samples(std::move(gpu_samples)), summarize_samples(std::move(wall_samples)) };
}

uint32_t rowwise_x_group_size(const uint32_t format)
{
  switch (format) {
    case CCV_NNC_QX_8I_ROWWISE_Q5_K:
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
      return 16;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
      return 32;
    case CCV_NNC_QX_8I_ROWWISE_Q6_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
      return 8;
    default:
      std::abort();
  }
}

uint32_t rowwise_x_group_bits(const uint32_t format)
{
  switch (format) {
    case CCV_NNC_QX_8I_ROWWISE_Q5_K:
      return 88;
    case CCV_NNC_QX_8I_ROWWISE_Q6_K:
      return 52;
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
      return 72;
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
      return 56;
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
      return 42;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
      return 21;
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
      return 28;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
      return 64;
    default:
      std::abort();
  }
}

size_t rowwise_x_scale_offset(const uint32_t format, const uint32_t row_count, const uint32_t row_length)
{
  const uint32_t group_size = rowwise_x_group_size(format);
  const uint32_t groups_per_row = (row_length + group_size - 1) / group_size;
  const size_t payload_bits = (size_t)row_count * groups_per_row * rowwise_x_group_bits(format);
  return align_up((payload_bits + 7) / 8, 128);
}

std::vector<uint8_t> make_synthetic_packed(
    const FormatInfo& format,
    const BenchmarkConfig& config,
    const std::vector<uint8_t>& b_rowwise,
    const size_t b_scale_offset,
    const size_t x_size)
{
  std::vector<uint8_t> b_x(x_size);
  for (size_t i = 0; i < b_x.size(); ++i)
    b_x[i] = (uint8_t)((i * 131 + format.value * 17 + 23) & 0xff);
  const size_t x_scale_offset = rowwise_x_scale_offset(format.value, config.N, config.K);
  const size_t scale_bytes = (size_t)config.N * sizeof(uint16_t);
  if (x_scale_offset + scale_bytes <= b_x.size() && b_scale_offset + scale_bytes <= b_rowwise.size())
    std::memcpy(b_x.data() + x_scale_offset, b_rowwise.data() + b_scale_offset, scale_bytes);
  if (format.value == CCV_NNC_QX_8I_ROWWISE_Q6_K) {
    const uint32_t group_size = rowwise_x_group_size(format.value);
    const uint32_t group_bits = rowwise_x_group_bits(format.value);
    const uint32_t groups_per_row = (config.K + group_size - 1) / group_size;
    const uint32_t groups = config.N * groups_per_row;
    for (uint32_t g = 0; g < groups; ++g) {
      const size_t metadata_bit = (size_t)g * group_bits + 48;
      for (uint32_t b = 0; b < 4; ++b)
        b_x[(metadata_bit + b) >> 3] &= (uint8_t)~(1u << ((metadata_bit + b) & 7));
    }
  }
  return b_x;
}

uint32_t read_bits(const uint8_t* const data, const size_t bit_offset, const uint32_t bits)
{
  const size_t byte_offset = bit_offset >> 3;
  const uint32_t shift = (uint32_t)(bit_offset & 7);
  const uint32_t byte_count = (shift + bits + 7) >> 3;
  uint64_t value = 0;
  for (uint32_t i = 0; i < byte_count; ++i)
    value |= (uint64_t)data[byte_offset + i] << (i * 8);
  return (uint32_t)((value >> shift) & ((uint64_t(1) << bits) - 1));
}

int sign_extend(const uint32_t value, const uint32_t bits)
{
  const uint32_t sign = 1u << (bits - 1);
  return (value & sign) ? (int)value - (int)(1u << bits) : (int)value;
}

int iq2_value(const uint64_t* const grid, const uint32_t index, const uint32_t lane)
{
  const int v = (int)((grid[index] >> (lane * 8)) & 0xffu);
  if (v == 8)
    return 1;
  if (v == 25)
    return 3;
  return 5;
}

int iq2xxs_value(const uint32_t index, const uint32_t lane)
{
  return (int)(1 + (((ccv_nnc_8i_rowwise_packed_iq2xxs_grid[index] >> (lane * 2)) & 3u) * 2));
}

int iq3xxs_value(const uint32_t index, const uint32_t lane)
{
  const int v = (int)((ccv_nnc_8i_rowwise_packed_iq3xxs_grid[index] >> (lane * 8)) & 0xffu);
  return v >> 2;
}

int iq3s_value(const uint32_t index, const uint32_t lane)
{
  return (int)((ccv_nnc_8i_rowwise_packed_iq3s_grid[index] >> (lane * 8)) & 0xffu);
}

void decode_group(const uint8_t* const input, const size_t group_index, const uint32_t format, int* const q8)
{
  static const int q2_xs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
  size_t bit = group_index * (size_t)rowwise_x_group_bits(format);
  switch (format) {
    case CCV_NNC_QX_8I_ROWWISE_Q5_K: {
      int q[16];
      for (uint32_t j = 0; j < 16; ++j, bit += 5)
        q[j] = (int)read_bits(input, bit, 5) - 16;
      const int m = (int)read_bits(input, bit, 3) + 1;
      const int b = (int)read_bits(input, bit + 3, 5) - 16;
      for (uint32_t j = 0; j < 16; ++j)
        q8[j] = q[j] * m + b;
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_Q6_K: {
      int q[8];
      for (uint32_t j = 0; j < 8; ++j, bit += 6)
        q[j] = sign_extend(read_bits(input, bit, 6), 6);
      const int m = (int)read_bits(input, bit, 2) + 1;
      const int b = sign_extend(read_bits(input, bit + 2, 2), 2);
      for (uint32_t j = 0; j < 8; ++j)
        q8[j] = q[j] * m + b;
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_Q4_K: {
      int q[16];
      for (uint32_t j = 0; j < 16; ++j, bit += 4)
        q[j] = (int)read_bits(input, bit, 4) - 8;
      const int m = (int)read_bits(input, bit, 4) + 1;
      const int b = (int)read_bits(input, bit + 4, 4) - 8;
      for (uint32_t j = 0; j < 16; ++j)
        q8[j] = q[j] * m + b;
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_Q3_K: {
      int q[16];
      for (uint32_t j = 0; j < 16; ++j, bit += 3)
        q[j] = (int)read_bits(input, bit, 3) - 4;
      const int m = (int)read_bits(input, bit, 5) + 1;
      const int b = ((int)read_bits(input, bit + 5, 3) - 4) << 1;
      for (uint32_t j = 0; j < 16; ++j)
        q8[j] = q[j] * m + b;
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_Q2_K: {
      int q[16];
      for (uint32_t j = 0; j < 16; ++j, bit += 2)
        q[j] = (int)read_bits(input, bit, 2);
      const int m = (int)read_bits(input, bit, 6) + 1;
      const int z = (int)read_bits(input, bit + 6, 4) << 3;
      for (uint32_t j = 0; j < 16; ++j)
        q8[j] = q[j] * m - z;
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S: {
      const uint32_t grid0 = read_bits(input, bit, 10);
      const uint32_t grid1 = read_bits(input, bit + 10, 10);
      const uint32_t signs = read_bits(input, bit + 20, 16);
      const int scale = (int)read_bits(input, bit + 36, 6) + 1;
      for (uint32_t j = 0; j < 8; ++j) {
        const int mag0 = std::min(iq2_value(ccv_nnc_8i_rowwise_packed_iq2s_grid, grid0, j) * scale, 127);
        const int mag1 = std::min(iq2_value(ccv_nnc_8i_rowwise_packed_iq2s_grid, grid1, j) * scale, 127);
        q8[j] = (signs & (1u << j)) ? -mag0 : mag0;
        q8[8 + j] = (signs & (1u << (8 + j))) ? -mag1 : mag1;
      }
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS: {
      const uint32_t grid0 = read_bits(input, bit, 9);
      const uint32_t signs = read_bits(input, bit + 9, 8);
      const int scale = q2_xs_scales[read_bits(input, bit + 17, 4)];
      for (uint32_t j = 0; j < 8; ++j) {
        const int mag = std::min(iq2_value(ccv_nnc_8i_rowwise_packed_iq2xs_grid, grid0, j) * scale, 127);
        q8[j] = (signs & (1u << j)) ? -mag : mag;
      }
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS: {
      uint32_t grid[4];
      for (uint32_t j = 0; j < 4; ++j)
        grid[j] = read_bits(input, bit + j * 8, 8);
      const uint32_t sign_codes = read_bits(input, bit + 32, 28);
      const int scale = q2_xs_scales[read_bits(input, bit + 60, 4)];
      for (uint32_t sg = 0; sg < 4; ++sg) {
        const uint32_t signs = ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[(sign_codes >> (sg * 7)) & 0x7f];
        for (uint32_t j = 0; j < 8; ++j) {
          const uint32_t lane = sg * 8 + j;
          const int mag = std::min(iq2xxs_value(grid[sg], j) * scale, 127);
          q8[lane] = (signs & (1u << j)) ? -mag : mag;
        }
      }
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S: {
      uint32_t grid[4];
      for (uint32_t j = 0; j < 4; ++j)
        grid[j] = read_bits(input, bit + j * 9, 9);
      const uint32_t signs = read_bits(input, bit + 36, 16);
      const int scale = (int)read_bits(input, bit + 52, 4) + 1;
      for (uint32_t sg = 0; sg < 4; ++sg) {
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t lane = sg * 4 + j;
          const int mag = std::min(iq3s_value(grid[sg], j) * scale, 127);
          q8[lane] = (signs & (1u << lane)) ? -mag : mag;
        }
      }
      break;
    }
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS: {
      const uint32_t grid0 = read_bits(input, bit, 8);
      const uint32_t grid1 = read_bits(input, bit + 8, 8);
      const uint32_t signs = read_bits(input, bit + 16, 8);
      const int scale = (int)read_bits(input, bit + 24, 4) + 1;
      for (uint32_t j = 0; j < 4; ++j) {
        const int mag0 = std::min(iq3xxs_value(grid0, j) * scale, 127);
        const int mag1 = std::min(iq3xxs_value(grid1, j) * scale, 127);
        q8[j] = (signs & (1u << j)) ? -mag0 : mag0;
        q8[4 + j] = (signs & (1u << (4 + j))) ? -mag1 : mag1;
      }
      break;
    }
    default:
      std::abort();
  }
}

ValidationStats validate_dequant(
    const std::vector<uint8_t>& gpu_rowwise,
    const std::vector<uint8_t>& packed_x,
    const uint32_t format,
    const BenchmarkConfig& config)
{
  const uint32_t group_size = rowwise_x_group_size(format);
  const uint32_t group_bits = rowwise_x_group_bits(format);
  const uint32_t groups_per_row = (config.K + group_size - 1) / group_size;
  const size_t input_scale_offset = align_up(((size_t)config.N * groups_per_row * group_bits + 7) / 8, 128);
  const size_t output_scale_offset = rowwise_8i_scale_offset(config.N, config.K);
  const size_t scale_bytes = (size_t)config.N * sizeof(uint16_t);
  const int8_t* const q = reinterpret_cast<const int8_t*>(gpu_rowwise.data());
  ValidationStats stats;
  stats.passed = true;
  if (std::memcmp(gpu_rowwise.data() + output_scale_offset, packed_x.data() + input_scale_offset, scale_bytes) != 0) {
    stats.passed = false;
    return stats;
  }
  for (uint32_t row = 0; row < config.N; ++row) {
    for (uint32_t group = 0; group < groups_per_row; ++group) {
      int expected_q[32] = {};
      decode_group(packed_x.data(), (size_t)row * groups_per_row + group, format, expected_q);
      for (uint32_t j = 0; j < group_size; ++j) {
        const uint32_t col = group * group_size + j;
        if (col >= config.K)
          continue;
        const int actual = (int)q[(size_t)row * config.K + col];
        const int expected = expected_q[j];
        const double diff = std::fabs((double)actual - expected);
        stats.max_abs = std::max(stats.max_abs, diff);
        stats.checked += 1;
        if (actual != expected) {
          stats.passed = false;
          return stats;
        }
      }
    }
  }
  return stats;
}

uint32_t format_from_name(const std::string& name)
{
  for (const auto& format : kFormats)
    if (name == format.name)
      return format.value;
  return 0;
}

void print_usage(const char* const argv0)
{
  std::cerr << "usage: " << argv0 << " [--m M] [--n N] [--k K] [--format q4_k] [--all-formats] [--warmup N] [--iters N] [--dequant-repeats N] [--synthetic-packed] [--skip-validation]\n";
}

bool parse_args(int argc, char** argv, BenchmarkConfig* const config)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* const name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value.\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--m") {
      const char* value = require_value("--m");
      if (!value) return false;
      config->M = (uint32_t)std::strtoul(value, nullptr, 10);
    } else if (arg == "--n") {
      const char* value = require_value("--n");
      if (!value) return false;
      config->N = (uint32_t)std::strtoul(value, nullptr, 10);
    } else if (arg == "--k") {
      const char* value = require_value("--k");
      if (!value) return false;
      config->K = (uint32_t)std::strtoul(value, nullptr, 10);
    } else if (arg == "--format") {
      const char* value = require_value("--format");
      if (!value) return false;
      config->format_name = value;
    } else if (arg == "--all-formats") {
      config->all_formats = true;
    } else if (arg == "--skip-validation") {
      config->skip_validation = true;
    } else if (arg == "--synthetic-packed") {
      config->synthetic_packed = true;
      config->skip_validation = true;
    } else if (arg == "--warmup") {
      const char* value = require_value("--warmup");
      if (!value) return false;
      config->warmup_iterations = std::atoi(value);
    } else if (arg == "--iters") {
      const char* value = require_value("--iters");
      if (!value) return false;
      config->timed_iterations = std::atoi(value);
    } else if (arg == "--dequant-repeats") {
      const char* value = require_value("--dequant-repeats");
      if (!value) return false;
      config->dequant_repeats = std::atoi(value);
    } else if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return false;
    }
  }
  return config->M > 0 && config->N > 0 && config->K > 0 && config->warmup_iterations >= 0 && config->timed_iterations > 0 && config->dequant_repeats > 0;
}

void print_stats(const char* const label, const Stats& stats)
{
  std::cout << " " << label
            << "_avg_ms=" << std::fixed << std::setprecision(4) << stats.average_seconds * 1e3
            << " " << label << "_median_ms=" << stats.median_seconds * 1e3
            << " " << label << "_best3_ms=" << stats.best3_seconds * 1e3
            << " " << label << "_min_ms=" << stats.min_seconds * 1e3
            << " " << label << "_max_ms=" << stats.max_seconds * 1e3;
}

} // namespace

int main(int argc, char** argv)
{
  BenchmarkConfig config;
  if (!parse_args(argc, argv, &config)) {
    print_usage(argv[0]);
    return 1;
  }
  ccv_nnc_init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "No Metal device available.\n";
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Failed to create command queue.\n";
    return 1;
  }

  std::vector<FormatInfo> formats;
  if (config.all_formats) {
    formats.assign(std::begin(kFormats), std::end(kFormats));
  } else {
    const uint32_t format = format_from_name(config.format_name);
    if (!format) {
      std::cerr << "Unknown format: " << config.format_name << "\n";
      return 1;
    }
    formats.push_back({ format, config.format_name.c_str() });
  }

  const size_t b_rowwise_size = rowwise_8i_data_size(config.N, config.K);
  const size_t b_scale_offset = rowwise_8i_scale_offset(config.N, config.K);
  const size_t dense_weight_bytes = (size_t)config.N * config.K * sizeof(uint16_t);
  const size_t c_bytes = (size_t)config.M * config.N * sizeof(uint16_t);
  auto a_values_host = make_int8_values(config.M, config.K);
  auto a_scales_host = make_half_scales(config.M);
  auto weights_half = make_half_weights(config.N, config.K);
  std::vector<uint8_t> b_rowwise(b_rowwise_size);
  const size_t rowwise_written = ccv_nnc_quantize_8i_rowwise(
      weights_half.data(),
      CCV_16F,
      CCV_TENSOR_CPU_MEMORY,
      weights_half.size(),
      config.K,
      nullptr,
      0,
      b_rowwise.data(),
      b_rowwise.size());
  if (rowwise_written > b_rowwise.size()) {
    std::cerr << "rowwise quantization overflowed output buffer.\n";
    return 1;
  }

  auto a_values = make_private_buffer(device.get(), command_queue.get(), a_values_host.data(), a_values_host.size() * sizeof(int8_t));
  auto a_scales = make_private_buffer(device.get(), command_queue.get(), a_scales_host.data(), a_scales_host.size() * sizeof(uint16_t));
  auto b_rowwise_buffer = make_private_buffer(device.get(), command_queue.get(), b_rowwise.data(), b_rowwise.size());
  auto b_x_dequant = NS::TransferPtr(device->newBuffer(b_rowwise_size, kPrivateResourceOptions));
  auto b_fp16_dequant = NS::TransferPtr(device->newBuffer(dense_weight_bytes, kPrivateResourceOptions));
  auto c_buffer = NS::TransferPtr(device->newBuffer(c_bytes, kPrivateResourceOptions));
  if (!a_values || !a_scales || !b_rowwise_buffer || !b_x_dequant || !b_fp16_dequant || !c_buffer) {
    std::cerr << "Failed to allocate Metal buffers.\n";
    return 1;
  }

  NAInt8MatMulDescriptor matmul_descriptor;
  matmul_descriptor.batchDimension = 1;
  matmul_descriptor.ioPrecision = GEMMOperandPrecision::FP16;
  matmul_descriptor.matrixDimensions = simd::uint3 { config.M, config.N, config.K };
  matmul_descriptor.batchStrides = std::nullopt;
  matmul_descriptor.useBias = false;
  std::unordered_map<NAInt8MatMulKernelDescriptor, std::unique_ptr<NAInt8MatMulKernel>> matmul_cache;
  auto matmul_pipeline = std::unique_ptr<PipelineValue<NAInt8MatMulKernel>>(
      matmul_descriptor.findKernel(device.get(), DeviceProperties(), nullptr, nullptr, "", &matmul_cache).second);

  Dequantize8iRowwiseDescriptor rowwise_dequant_descriptor;
  rowwise_dequant_descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  rowwise_dequant_descriptor.rowLength = config.K;
  rowwise_dequant_descriptor.length = config.N * config.K;
  std::unordered_map<Dequantize8iRowwiseKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseKernel>> rowwise_dequant_cache;
  auto rowwise_dequant_pipeline = std::unique_ptr<PipelineValue<Dequantize8iRowwiseKernel>>(
      rowwise_dequant_descriptor.findKernel(device.get(), DeviceProperties(), nullptr, nullptr, "", &rowwise_dequant_cache).second);
  const TimingStats rowwise_fp16_dequant_stats = benchmark(config, [&]() {
    return run_rowwise_fp16_dequant_once(
        command_queue.get(),
        rowwise_dequant_pipeline.get(),
        rowwise_dequant_descriptor,
        b_rowwise_buffer.get(),
        b_fp16_dequant.get(),
        config.dequant_repeats);
  }, config.dequant_repeats);

  const TimingStats rowwise_stats = benchmark(config, [&]() {
    return run_matmul_once(
        command_queue.get(),
        matmul_pipeline.get(),
        config,
        a_values.get(),
        a_scales.get(),
        b_rowwise_buffer.get(),
        b_scale_offset,
        c_buffer.get());
  });

  for (const auto& format : formats) {
    const size_t x_size = ccv_nnc_8i_rowwise_x_data_size(format.value, CCV_16F, weights_half.size(), config.K);
    std::vector<uint8_t> b_x;
    if (config.synthetic_packed) {
      b_x = make_synthetic_packed(format, config, b_rowwise, b_scale_offset, x_size);
    } else {
      b_x.resize(x_size);
      const size_t x_written = ccv_nnc_quantize_8i_rowwise_x(
          weights_half.data(),
          CCV_16F,
          CCV_TENSOR_CPU_MEMORY,
          weights_half.size(),
          config.K,
          format.value,
          nullptr,
          0,
          b_x.data(),
          b_x.size());
      if (x_written > b_x.size()) {
        std::cerr << "packed-x quantization overflowed output buffer for " << format.name << ".\n";
        return 1;
      }
    }
    auto b_x_buffer = make_private_buffer(device.get(), command_queue.get(), b_x.data(), b_x.size());
    Dequantize8iRowwiseXDescriptor dequant_descriptor;
    dequant_descriptor.format = format.value;
    dequant_descriptor.scaleSize = sizeof(uint16_t);
    dequant_descriptor.rowLength = config.K;
    dequant_descriptor.length = config.N * config.K;
    std::unordered_map<Dequantize8iRowwiseXKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXKernel>> dequant_cache;
    auto dequant_pipeline = std::unique_ptr<PipelineValue<Dequantize8iRowwiseXKernel>>(
        dequant_descriptor.findKernel(device.get(), DeviceProperties(), nullptr, nullptr, "", &dequant_cache).second);

    Dequantize8iRowwiseXFPDescriptor fp_dequant_descriptor;
    fp_dequant_descriptor.format = format.value;
    fp_dequant_descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
    fp_dequant_descriptor.rowLength = config.K;
    fp_dequant_descriptor.length = config.N * config.K;
    std::unordered_map<Dequantize8iRowwiseXFPKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXFPKernel>> fp_dequant_cache;
    auto fp_dequant_pipeline = std::unique_ptr<PipelineValue<Dequantize8iRowwiseXFPKernel>>(
        fp_dequant_descriptor.findKernel(device.get(), DeviceProperties(), nullptr, nullptr, "", &fp_dequant_cache).second);

    ValidationStats validation;
    if (!config.skip_validation) {
      (void)run_dequant_once(command_queue.get(), dequant_pipeline.get(), dequant_descriptor, b_x_buffer.get(), b_x_dequant.get());
      const std::vector<uint8_t> gpu_rowwise = download_buffer(device.get(), command_queue.get(), b_x_dequant.get(), b_rowwise_size);
      validation = validate_dequant(gpu_rowwise, b_x, format.value, config);
      if (!validation.passed) {
        std::cerr << "dequant validation failed for " << format.name
                  << " checked=" << validation.checked
                  << " max_abs=" << validation.max_abs << "\n";
        return 1;
      }
    } else {
      validation.passed = true;
      (void)run_dequant_once(command_queue.get(), dequant_pipeline.get(), dequant_descriptor, b_x_buffer.get(), b_x_dequant.get());
    }

    const TimingStats x_rowwise_stats = benchmark(config, [&]() {
      return run_matmul_once(
          command_queue.get(),
          matmul_pipeline.get(),
          config,
          a_values.get(),
          a_scales.get(),
          b_x_dequant.get(),
          b_scale_offset,
          c_buffer.get());
    });
    const TimingStats dequant_stats = benchmark(config, [&]() {
      return run_dequant_once(
          command_queue.get(),
          dequant_pipeline.get(),
          dequant_descriptor,
          b_x_buffer.get(),
          b_x_dequant.get(),
          config.dequant_repeats);
    }, config.dequant_repeats);
    const TimingStats fp_dequant_stats = benchmark(config, [&]() {
      return run_x_fp_dequant_once(
          command_queue.get(),
          fp_dequant_pipeline.get(),
          fp_dequant_descriptor,
          b_x_buffer.get(),
          b_fp16_dequant.get(),
          config.dequant_repeats);
    }, config.dequant_repeats);
    const TimingStats combined_stats = benchmark(config, [&]() {
      return run_dequant_matmul_once(
          command_queue.get(),
          dequant_pipeline.get(),
          dequant_descriptor,
          matmul_pipeline.get(),
          config,
          b_x_buffer.get(),
          b_x_dequant.get(),
          a_values.get(),
          a_scales.get(),
          b_scale_offset,
          c_buffer.get());
    });

    const double rowwise_bytes = (double)b_rowwise_size;
    const double packed_bytes = (double)b_x.size();
    std::cout << "format=" << format.name
              << " M=" << config.M
              << " N=" << config.N
              << " K=" << config.K
              << " synthetic_packed=" << (config.synthetic_packed ? 1 : 0)
              << " dequant_repeats=" << config.dequant_repeats
              << " rowwise_bytes=" << (uint64_t)b_rowwise_size
              << " dense_fp16_bytes=" << (uint64_t)dense_weight_bytes
              << " packed_bytes=" << (uint64_t)b_x.size()
              << " compression=" << std::setprecision(4) << rowwise_bytes / packed_bytes
              << " validation_checked=" << validation.checked
              << " validation_skipped=" << (config.skip_validation ? 1 : 0);
    print_stats("rowwise_matmul_gpu", rowwise_stats.gpu);
    print_stats("rowwise_matmul_wall", rowwise_stats.wall);
    print_stats("rowwise_fp16_dequant_gpu", rowwise_fp16_dequant_stats.gpu);
    print_stats("rowwise_fp16_dequant_wall", rowwise_fp16_dequant_stats.wall);
    print_stats("x_rowwise_matmul_gpu", x_rowwise_stats.gpu);
    print_stats("x_rowwise_matmul_wall", x_rowwise_stats.wall);
    print_stats("x_dequant_gpu", dequant_stats.gpu);
    print_stats("x_dequant_wall", dequant_stats.wall);
    print_stats("x_fp_dequant_gpu", fp_dequant_stats.gpu);
    print_stats("x_fp_dequant_wall", fp_dequant_stats.wall);
    print_stats("x_dequant_matmul_gpu", combined_stats.gpu);
    print_stats("x_dequant_matmul_wall", combined_stats.wall);
    if (rowwise_stats.wall.median_seconds > 0 && combined_stats.wall.median_seconds > 0) {
      const double overhead = combined_stats.wall.median_seconds - rowwise_stats.wall.median_seconds;
      std::cout << " wall_median_overhead_ms=" << overhead * 1e3
                << " wall_median_overhead_pct=" << (overhead / rowwise_stats.wall.median_seconds) * 100.0;
    }
    if (rowwise_stats.gpu.median_seconds > 0 && combined_stats.gpu.median_seconds > 0) {
      const double overhead = combined_stats.gpu.median_seconds - rowwise_stats.gpu.median_seconds;
      std::cout << " gpu_median_overhead_ms=" << overhead * 1e3
                << " gpu_median_overhead_pct=" << (overhead / rowwise_stats.gpu.median_seconds) * 100.0;
    }
    if (rowwise_fp16_dequant_stats.gpu.median_seconds > 0 && dequant_stats.gpu.median_seconds > 0) {
      std::cout << " x_dequant_vs_rowwise_fp16_gpu_pct="
                << (dequant_stats.gpu.median_seconds / rowwise_fp16_dequant_stats.gpu.median_seconds) * 100.0;
    }
    if (rowwise_fp16_dequant_stats.wall.median_seconds > 0 && dequant_stats.wall.median_seconds > 0) {
      std::cout << " x_dequant_vs_rowwise_fp16_wall_pct="
                << (dequant_stats.wall.median_seconds / rowwise_fp16_dequant_stats.wall.median_seconds) * 100.0;
    }
    if (rowwise_fp16_dequant_stats.gpu.median_seconds > 0 && fp_dequant_stats.gpu.median_seconds > 0) {
      std::cout << " x_fp_dequant_vs_rowwise_fp16_gpu_pct="
                << (fp_dequant_stats.gpu.median_seconds / rowwise_fp16_dequant_stats.gpu.median_seconds) * 100.0;
    }
    if (rowwise_fp16_dequant_stats.wall.median_seconds > 0 && fp_dequant_stats.wall.median_seconds > 0) {
      std::cout << " x_fp_dequant_vs_rowwise_fp16_wall_pct="
                << (fp_dequant_stats.wall.median_seconds / rowwise_fp16_dequant_stats.wall.median_seconds) * 100.0;
    }
    if (x_rowwise_stats.wall.median_seconds > 0 && combined_stats.wall.median_seconds > 0) {
      const double overhead = combined_stats.wall.median_seconds - x_rowwise_stats.wall.median_seconds;
      std::cout << " same_data_wall_median_overhead_ms=" << overhead * 1e3
                << " same_data_wall_median_overhead_pct=" << (overhead / x_rowwise_stats.wall.median_seconds) * 100.0;
    }
    if (x_rowwise_stats.gpu.median_seconds > 0 && combined_stats.gpu.median_seconds > 0) {
      const double overhead = combined_stats.gpu.median_seconds - x_rowwise_stats.gpu.median_seconds;
      std::cout << " same_data_gpu_median_overhead_ms=" << overhead * 1e3
                << " same_data_gpu_median_overhead_pct=" << (overhead / x_rowwise_stats.gpu.median_seconds) * 100.0;
    }
    std::cout << "\n";
  }
  return 0;
}
