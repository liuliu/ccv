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
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <QuartzCore/QuartzCore.h>

extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
}
#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseDescriptor.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseKernel.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseXDescriptor.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseXKernel.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/GemvDescriptor.hpp"
#include "nnc/mfa/kernels/GemvKernel.hpp"
#include "nnc/mfa/kernels/Int8GemvDescriptor.hpp"
#include "nnc/mfa/kernels/Int8GemvKernel.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

using half_float = _Float16;

struct bfloat16_value {
  uint16_t bits = 0;

  bfloat16_value() = default;
  bfloat16_value(const float value) {
    uint32_t u = 0;
    std::memcpy(&u, &value, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1;
    bits = (uint16_t)((u + 0x7fff + lsb) >> 16);
  }

  operator float() const {
    const uint32_t u = (uint32_t)bits << 16;
    float value = 0;
    std::memcpy(&value, &u, sizeof(value));
    return value;
  }
};

constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;

struct Shape {
  uint32_t rows;
  uint32_t cols;
};

struct Config {
  int warmup_iterations = 3;
  int timed_iterations = 20;
  int duplicated_dispatches = 5;
  int sleep_ms = 200;
  uint32_t mrows = 1;
  bool fused_bias = true;
  bool all_formats = false;
  std::string dtype = "fp16";
  std::string format = "rowwise";
  std::vector<Shape> shapes;
};

struct FormatInfo {
  uint32_t value;
  const char* name;
};

struct Stats {
  double average_seconds = 0;
  double best3_average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct ValidationStats {
  double max_abs = 0;
  double max_rel = 0;
};

struct Pipelines {
  NS::SharedPtr<MTL::ComputePipelineState> gemv;
  Dequantize8iRowwiseKernel* dequant_kernel = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> dequant;
  NS::SharedPtr<MTL::ComputePipelineState> scaled;
  Dequantize8iRowwiseXKernel* dequant_x_kernel = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> dequant_x;
  NS::SharedPtr<MTL::ComputePipelineState> scaled_x;
  uint32_t dequant_x_dispatch_items = 0;
  uint32_t gemv_rows_per_threadgroup = 0;
};

constexpr FormatInfo kRowwiseFormat = { 0, "rowwise" };

constexpr FormatInfo kPackedFormats[] = {
  { CCV_NNC_QX_8I_ROWWISE_Q4_K, "q4_k" },
  { CCV_NNC_QX_8I_ROWWISE_Q3_K, "q3_k" },
  { CCV_NNC_QX_8I_ROWWISE_Q2_K, "q2_k" },
  { CCV_NNC_QX_8I_ROWWISE_IQ2_S, "iq2_s" },
  { CCV_NNC_QX_8I_ROWWISE_IQ2_XS, "iq2_xs" },
  { CCV_NNC_QX_8I_ROWWISE_IQ3_S, "iq3_s" },
  { CCV_NNC_QX_8I_ROWWISE_IQ3_XXS, "iq3_xxs" },
};

uint32_t ceil_div(const uint32_t x, const uint32_t y)
{
  return (x + y - 1) / y;
}

size_t align_up(const size_t value, const size_t alignment)
{
  return (value + alignment - 1) & ~(alignment - 1);
}

bool parse_uint32(const char* text, uint32_t* value)
{
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (text == end || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max())
    return false;
  *value = (uint32_t)parsed;
  return true;
}

bool parse_int(const char* text, int* value)
{
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (text == end || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max())
    return false;
  *value = (int)parsed;
  return true;
}

template <typename T>
std::vector<T> make_matrix(const uint32_t rows, const uint32_t cols, const float scale, const int phase)
{
  std::vector<T> values((size_t)rows * cols);
  for (uint32_t row = 0; row < rows; ++row) {
    const float row_gain = 0.2f + 0.8f * (float)(((row * 7 + phase * 5) % 23) + 1) / 24.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      const int centered = (int)((row * 131 + col * 17 + phase * 29) % 127) - 63;
      values[(size_t)row * cols + col] = (T)(centered * scale * row_gain);
    }
  }
  return values;
}

template <typename T>
std::vector<T> make_bias(const uint32_t length)
{
  std::vector<T> values(length);
  for (uint32_t i = 0; i < length; ++i) {
    const int centered = (int)((i * 11 + 7) % 31) - 15;
    values[i] = (T)(centered * 0.0078125f);
  }
  return values;
}

template <typename T>
std::vector<uint8_t> quantize_rowwise(
    const std::vector<T>& source,
    const uint32_t rows,
    const uint32_t cols,
    std::vector<T>* dense_dequantized)
{
  const size_t q_bytes = (size_t)rows * cols * sizeof(int8_t);
  const size_t scale_offset = align_up(q_bytes, 128);
  const size_t total_bytes = scale_offset + (size_t)rows * sizeof(T);
  std::vector<uint8_t> quantized(total_bytes, 0);
  int8_t* const q = (int8_t*)quantized.data();
  T* const scales = (T*)(quantized.data() + scale_offset);
  dense_dequantized->resize((size_t)rows * cols);
  for (uint32_t row = 0; row < rows; ++row) {
    float max_abs = 0;
    for (uint32_t col = 0; col < cols; ++col)
      max_abs = std::max(max_abs, std::fabs((float)source[(size_t)row * cols + col]));
    const float scale = max_abs > 0 ? max_abs / 127.0f : 0.0f;
    const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 0.0f;
    scales[row] = (T)scale;
    for (uint32_t col = 0; col < cols; ++col) {
      const size_t index = (size_t)row * cols + col;
      const int rounded = max_abs > 0 ? (int)std::lrint((float)source[index] * inv_scale) : 0;
      const int8_t qv = (int8_t)std::max(-127, std::min(127, rounded));
      q[index] = qv;
      (*dense_dequantized)[index] = (T)((float)qv * scale);
    }
  }
  return quantized;
}

GEMMOperandPrecision precision_for_dtype(const std::string& dtype)
{
  if (dtype == "fp32")
    return GEMMOperandPrecision::FP32;
  if (dtype == "bf16")
    return GEMMOperandPrecision::BF16;
  return GEMMOperandPrecision::FP16;
}

uint32_t format_from_name(const std::string& name)
{
  if (name == kRowwiseFormat.name)
    return kRowwiseFormat.value;
  for (const FormatInfo format : kPackedFormats)
    if (name == format.name)
      return format.value;
  return UINT32_MAX;
}

std::vector<FormatInfo> selected_formats(const Config& config)
{
  std::vector<FormatInfo> formats;
  if (config.all_formats) {
    formats.push_back(kRowwiseFormat);
    formats.insert(formats.end(), std::begin(kPackedFormats), std::end(kPackedFormats));
    return formats;
  }
  const uint32_t value = format_from_name(config.format);
  if (value == kRowwiseFormat.value) {
    formats.push_back(kRowwiseFormat);
  } else {
    for (const FormatInfo format : kPackedFormats)
      if (format.value == value)
        formats.push_back(format);
  }
  return formats;
}

uint32_t rowwise_x_group_size(const uint32_t format)
{
  switch (format) {
    case CCV_NNC_QX_8I_ROWWISE_Q4_K:
    case CCV_NNC_QX_8I_ROWWISE_Q3_K:
    case CCV_NNC_QX_8I_ROWWISE_Q2_K:
    case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
      return 16;
    case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
    case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
      return 8;
    default:
      return 0;
  }
}

uint32_t rowwise_x_group_bits(const uint32_t format)
{
  switch (format) {
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
    default:
      return 0;
  }
}

template <typename T>
std::vector<uint8_t> make_packed_rowwise_x(
    const std::vector<uint8_t>& rowwise,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t format)
{
  const uint32_t group_size = rowwise_x_group_size(format);
  const uint32_t group_bits = rowwise_x_group_bits(format);
  const uint32_t groups_per_row = ceil_div(cols, group_size);
  const size_t payload_bits = (size_t)rows * groups_per_row * group_bits;
  const size_t payload_bytes = (payload_bits + 7) / 8;
  const size_t scale_offset = align_up(payload_bytes, 128);
  const size_t rowwise_scale_offset = align_up((size_t)rows * cols, 128);
  std::vector<uint8_t> packed(scale_offset + (size_t)rows * sizeof(T), 0);
  for (size_t i = 0; i < scale_offset; ++i)
    packed[i] = (uint8_t)((i * 131 + format * 17 + 23) & 0xff);
  if (format == CCV_NNC_QX_8I_ROWWISE_Q4_K || format == CCV_NNC_QX_8I_ROWWISE_Q3_K) {
    const uint32_t groups_per_row = ceil_div(cols, group_size);
    const uint32_t group_bytes = format == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 9 : 7;
    for (uint32_t g = 0; g < rows * groups_per_row; ++g)
      packed[(size_t)g * group_bytes + group_bytes - 1] = 0x80;
  } else if (format == CCV_NNC_QX_8I_ROWWISE_Q2_K) {
    const uint32_t groups_per_row = ceil_div(cols, group_size);
    const uint32_t groups = rows * groups_per_row;
    for (uint32_t g = 0; g < groups; ++g) {
      const size_t metadata_bit = (size_t)g * group_bits + 32;
      for (uint32_t b = 0; b < 10; ++b)
        packed[(metadata_bit + b) >> 3] &= (uint8_t)~(1u << ((metadata_bit + b) & 7));
    }
  }
  std::memcpy(packed.data() + scale_offset, rowwise.data() + rowwise_scale_offset, (size_t)rows * sizeof(T));
  return packed;
}

Pipelines create_pipelines(
    MTL::Device* const device,
    ShaderCache& shader_cache,
    const Shape shape,
    const GEMMOperandPrecision precision,
    const uint32_t mrows,
    const bool fused_bias,
    const uint32_t packed_format)
{
  DeviceProperties dprops{};

  GemvDescriptor gemv_desc;
  gemv_desc.fusedBias = fused_bias ? 1 : 0;
  gemv_desc.mrows = (uint8_t)mrows;
  gemv_desc.memoryPrecision = precision;
  gemv_desc.nrows = shape.rows;
  gemv_desc.ncols = shape.cols;
  auto gemv_value = shader_cache.findKernel<GemvKernel, GemvDescriptor, GemvKernelDescriptor>(
      gemv_desc, device, dprops);

  Dequantize8iRowwiseDescriptor dequant_desc;
  dequant_desc.memoryPrecision = precision;
  dequant_desc.rowLength = shape.cols;
  dequant_desc.length = shape.rows * shape.cols;
  auto dequant_value = shader_cache.findKernel<Dequantize8iRowwiseKernel, Dequantize8iRowwiseDescriptor, Dequantize8iRowwiseKernelDescriptor>(
      dequant_desc, device, dprops);

  Int8GemvDescriptor scaled_desc;
  scaled_desc.fusedBias = fused_bias ? 1 : 0;
  scaled_desc.mrows = (uint8_t)mrows;
  scaled_desc.format = 0;
  scaled_desc.memoryPrecision = precision;
  scaled_desc.nrows = shape.rows;
  scaled_desc.ncols = shape.cols;
  auto scaled_value = shader_cache.findKernel<Int8GemvKernel, Int8GemvDescriptor, Int8GemvKernelDescriptor>(
      scaled_desc, device, dprops);

  PipelineValue<Dequantize8iRowwiseXKernel>* dequant_x_value = nullptr;
  PipelineValue<Int8GemvKernel>* scaled_x_value = nullptr;
  if (packed_format != 0) {
    Dequantize8iRowwiseXDescriptor dequant_x_desc;
    dequant_x_desc.format = packed_format;
    dequant_x_desc.scaleSize = precision == GEMMOperandPrecision::FP32 ? 4 : 2;
    dequant_x_desc.rowLength = shape.cols;
    dequant_x_desc.length = shape.rows * shape.cols;
    dequant_x_value = shader_cache.findKernel<Dequantize8iRowwiseXKernel, Dequantize8iRowwiseXDescriptor, Dequantize8iRowwiseXKernelDescriptor>(
        dequant_x_desc, device, dprops);

    Int8GemvDescriptor scaled_x_desc;
    scaled_x_desc.fusedBias = fused_bias ? 1 : 0;
    scaled_x_desc.mrows = (uint8_t)mrows;
    scaled_x_desc.format = packed_format;
    scaled_x_desc.memoryPrecision = precision;
    scaled_x_desc.nrows = shape.rows;
    scaled_x_desc.ncols = shape.cols;
    scaled_x_value = shader_cache.findKernel<Int8GemvKernel, Int8GemvDescriptor, Int8GemvKernelDescriptor>(
        scaled_x_desc, device, dprops);
  }

  Pipelines pipelines;
  pipelines.gemv = gemv_value->pipeline;
  pipelines.dequant_kernel = dequant_value->kernel;
  pipelines.dequant = dequant_value->pipeline;
  pipelines.scaled = scaled_value->pipeline;
  if (dequant_x_value) {
    pipelines.dequant_x_kernel = dequant_x_value->kernel;
    pipelines.dequant_x = dequant_x_value->pipeline;
    Dequantize8iRowwiseXDescriptor dequant_x_desc;
    dequant_x_desc.format = packed_format;
    dequant_x_desc.scaleSize = precision == GEMMOperandPrecision::FP32 ? 4 : 2;
    dequant_x_desc.rowLength = shape.cols;
    dequant_x_desc.length = shape.rows * shape.cols;
    pipelines.dequant_x_dispatch_items = dequant_x_desc.dispatchItems();
  }
  if (scaled_x_value)
    pipelines.scaled_x = scaled_x_value->pipeline;
  pipelines.gemv_rows_per_threadgroup = GemvDescriptor::rowsPerThreadgroup(device);
  return pipelines;
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

void download_buffer(
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

double finish_and_time(NS::SharedPtr<MTL::CommandBuffer>& command_buffer, const int duplicated_dispatches)
{
  const double start = CACurrentMediaTime();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const double end = CACurrentMediaTime();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
    std::cerr << "command buffer failed with status="
              << static_cast<int>(command_buffer->status());
    if (auto* error = command_buffer->error()) {
      auto* description = error->localizedDescription();
      if (description)
        std::cerr << " error=" << description->utf8String();
    }
    std::cerr << '\n';
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double gpu_start = command_buffer->GPUStartTime();
  const double gpu_end = command_buffer->GPUEndTime();
  if (gpu_end > gpu_start)
    return (gpu_end - gpu_start) / duplicated_dispatches;
  return (end - start) / duplicated_dispatches;
}

void encode_gemv(
    MTL::ComputeCommandEncoder* const encoder,
    const Pipelines& pipelines,
    const Shape shape,
    MTL::Buffer* const matrix,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  encoder->setComputePipelineState(pipelines.gemv.get());
  encoder->setBuffer(matrix, 0, 0);
  encoder->setBuffer(vector, 0, 1);
  encoder->setBuffer(output, 0, 2);
  if (bias)
    encoder->setBuffer(bias, 0, 3);
  encoder->dispatchThreadgroups(
      MTL::Size(ceil_div(shape.rows, pipelines.gemv_rows_per_threadgroup), 1, 1),
      MTL::Size(pipelines.gemv_rows_per_threadgroup * 32, 1, 1));
}

void encode_dequant(
    MTL::ComputeCommandEncoder* const encoder,
    const Pipelines& pipelines,
    const Shape shape,
    MTL::Buffer* const quantized,
    MTL::Buffer* const dense)
{
  encoder->setComputePipelineState(pipelines.dequant.get());
  encoder->setBuffer(quantized, 0, 0);
  encoder->setBuffer(dense, 0, 1);
  encoder->dispatchThreadgroups(
      pipelines.dequant_kernel->gridSize(shape.rows * shape.cols),
      MTL::Size(256, 1, 1));
}

void encode_scaled_gemv(
    MTL::ComputeCommandEncoder* const encoder,
    const Pipelines& pipelines,
    const Shape shape,
    MTL::Buffer* const quantized,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  encoder->setComputePipelineState(pipelines.scaled.get());
  encoder->setBuffer(quantized, 0, 0);
  encoder->setBuffer(vector, 0, 1);
  encoder->setBuffer(output, 0, 2);
  if (bias)
    encoder->setBuffer(bias, 0, 3);
  encoder->dispatchThreadgroups(
      MTL::Size(ceil_div(shape.rows, kInt8GemvRowsPerThreadgroup), 1, 1),
      MTL::Size(kInt8GemvSIMDGroupsPerThreadgroup * 32, 1, 1));
}

void encode_dequant_x(
    MTL::ComputeCommandEncoder* const encoder,
    const Pipelines& pipelines,
    const Shape shape,
    MTL::Buffer* const quantized,
    MTL::Buffer* const rowwise)
{
  encoder->setComputePipelineState(pipelines.dequant_x.get());
  encoder->setBuffer(quantized, 0, 0);
  encoder->setBuffer(rowwise, 0, 1);
  encoder->dispatchThreadgroups(
      pipelines.dequant_x_kernel->gridSize(pipelines.dequant_x_dispatch_items),
      MTL::Size(256, 1, 1));
}

void encode_scaled_x_gemv(
    MTL::ComputeCommandEncoder* const encoder,
    const Pipelines& pipelines,
    const Shape shape,
    MTL::Buffer* const quantized,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  encoder->setComputePipelineState(pipelines.scaled_x.get());
  encoder->setBuffer(quantized, 0, 0);
  encoder->setBuffer(vector, 0, 1);
  encoder->setBuffer(output, 0, 2);
  if (bias)
    encoder->setBuffer(bias, 0, 3);
  encoder->dispatchThreadgroups(
      MTL::Size(ceil_div(shape.rows, kInt8GemvRowsPerThreadgroup), 1, 1),
      MTL::Size(kInt8GemvSIMDGroupsPerThreadgroup * 32, 1, 1));
}

double run_raw_once(
    MTL::CommandQueue* const command_queue,
    const Pipelines& pipelines,
    const Shape shape,
    const int duplicated_dispatches,
    MTL::Buffer* const dense,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int i = 0; i < duplicated_dispatches; ++i) {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_gemv(encoder.get(), pipelines, shape, dense, vector, output, bias);
    encoder->endEncoding();
  }
  return finish_and_time(command_buffer, duplicated_dispatches);
}

double run_dequant_gemv_once(
    MTL::CommandQueue* const command_queue,
    const Pipelines& pipelines,
    const Shape shape,
    const int duplicated_dispatches,
    MTL::Buffer* const quantized,
    MTL::Buffer* const dense,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int i = 0; i < duplicated_dispatches; ++i) {
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encode_dequant(encoder.get(), pipelines, shape, quantized, dense);
      encoder->endEncoding();
    }
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encode_gemv(encoder.get(), pipelines, shape, dense, vector, output, bias);
      encoder->endEncoding();
    }
  }
  return finish_and_time(command_buffer, duplicated_dispatches);
}

double run_scaled_once(
    MTL::CommandQueue* const command_queue,
    const Pipelines& pipelines,
    const Shape shape,
    const int duplicated_dispatches,
    MTL::Buffer* const quantized,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int i = 0; i < duplicated_dispatches; ++i) {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_scaled_gemv(encoder.get(), pipelines, shape, quantized, vector, output, bias);
    encoder->endEncoding();
  }
  return finish_and_time(command_buffer, duplicated_dispatches);
}

double run_dequant_x_scaled_once(
    MTL::CommandQueue* const command_queue,
    const Pipelines& pipelines,
    const Shape shape,
    const int duplicated_dispatches,
    MTL::Buffer* const quantized,
    MTL::Buffer* const rowwise,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int i = 0; i < duplicated_dispatches; ++i) {
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encode_dequant_x(encoder.get(), pipelines, shape, quantized, rowwise);
      encoder->endEncoding();
    }
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encode_scaled_gemv(encoder.get(), pipelines, shape, rowwise, vector, output, bias);
      encoder->endEncoding();
    }
  }
  return finish_and_time(command_buffer, duplicated_dispatches);
}

double run_scaled_x_once(
    MTL::CommandQueue* const command_queue,
    const Pipelines& pipelines,
    const Shape shape,
    const int duplicated_dispatches,
    MTL::Buffer* const quantized,
    MTL::Buffer* const vector,
    MTL::Buffer* const output,
    MTL::Buffer* const bias)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int i = 0; i < duplicated_dispatches; ++i) {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encode_scaled_x_gemv(encoder.get(), pipelines, shape, quantized, vector, output, bias);
    encoder->endEncoding();
  }
  return finish_and_time(command_buffer, duplicated_dispatches);
}

bool benchmark(const Config& config, const std::function<double()>& run_once, Stats* const stats)
{
  std::vector<double> samples;
  samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const double elapsed = run_once();
    if (!(elapsed >= 0))
      return false;
    if (i >= config.warmup_iterations)
      samples.push_back(elapsed);
    if (config.sleep_ms > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(config.sleep_ms));
  }
  if (samples.empty())
    return false;
  stats->average_seconds =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const size_t best_count = std::min<size_t>(3, samples.size());
  stats->best3_average_seconds =
      std::accumulate(samples.begin(), samples.begin() + best_count, 0.0) / best_count;
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

template <typename T>
ValidationStats validate_output(
    const std::vector<T>& reference,
    const std::vector<T>& actual)
{
  ValidationStats stats;
  for (size_t i = 0; i < reference.size(); ++i) {
    const float lhs = (float)reference[i];
    const float rhs = (float)actual[i];
    const float abs = std::fabs(lhs - rhs);
    const float rel = abs / std::max(1.0f, std::max(std::fabs(lhs), std::fabs(rhs)));
    stats.max_abs = std::max(stats.max_abs, (double)abs);
    stats.max_rel = std::max(stats.max_rel, (double)rel);
  }
  return stats;
}

void print_stats(
    const char* const label,
    const Stats& stats,
    const double flops,
    const double bytes)
{
  std::cout << "  " << label
            << " avg_ms=" << std::fixed << std::setprecision(4) << stats.average_seconds * 1e3
            << " best3_avg_ms=" << stats.best3_average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " tflops=" << flops / stats.average_seconds / 1e12
            << " effective_tb_s=" << bytes / stats.average_seconds / 1e12
            << '\n';
}

void print_validation(const char* const label, const ValidationStats& stats)
{
  std::cout << "  " << label
            << " max_abs=" << std::scientific << std::setprecision(3) << stats.max_abs
            << " max_rel=" << stats.max_rel << std::fixed << std::setprecision(4)
            << '\n';
}

template <typename T>
int run_shape(
    MTL::Device* const device,
    MTL::CommandQueue* const command_queue,
    ShaderCache& shader_cache,
    const Config& config,
    const Shape shape,
    const FormatInfo format,
    const GEMMOperandPrecision precision)
{
  const size_t element_bytes = sizeof(T);
  const size_t dense_bytes = (size_t)shape.rows * shape.cols * element_bytes;
  const size_t output_bytes = (size_t)config.mrows * shape.rows * element_bytes;
  const size_t vector_bytes = (size_t)config.mrows * shape.cols * element_bytes;
  const size_t bias_bytes = (size_t)shape.rows * element_bytes;
  const size_t row_groups = ceil_div(shape.rows, GemvDescriptor::rowsPerThreadgroup(device));

  auto original_matrix = make_matrix<T>(shape.rows, shape.cols, 0.015625f, 2);
  std::vector<T> dense_matrix;
  auto quantized = quantize_rowwise(original_matrix, shape.rows, shape.cols, &dense_matrix);
  std::vector<uint8_t> packed_quantized;
  if (format.value != 0) {
    packed_quantized = make_packed_rowwise_x<T>(quantized, shape.rows, shape.cols, format.value);
    if (packed_quantized.empty())
      return 1;
  }
  auto vector = make_matrix<T>(config.mrows, shape.cols, 0.03125f, 3);
  auto bias = make_bias<T>(shape.rows);

  auto quantized_stage = NS::TransferPtr(device->newBuffer(quantized.data(), quantized.size(), kSharedResourceOptions));
  NS::SharedPtr<MTL::Buffer> packed_quantized_stage;
  if (format.value != 0)
    packed_quantized_stage = NS::TransferPtr(device->newBuffer(packed_quantized.data(), packed_quantized.size(), kSharedResourceOptions));
  auto dense_stage = NS::TransferPtr(device->newBuffer(dense_matrix.data(), dense_bytes, kSharedResourceOptions));
  auto vector_stage = NS::TransferPtr(device->newBuffer(vector.data(), vector_bytes, kSharedResourceOptions));
  auto bias_stage = NS::TransferPtr(device->newBuffer(bias.data(), bias_bytes, kSharedResourceOptions));
  auto raw_output_stage = NS::TransferPtr(device->newBuffer(output_bytes, kSharedResourceOptions));
  auto dequant_output_stage = NS::TransferPtr(device->newBuffer(output_bytes, kSharedResourceOptions));
  auto scaled_output_stage = NS::TransferPtr(device->newBuffer(output_bytes, kSharedResourceOptions));
  NS::SharedPtr<MTL::Buffer> dequant_x_output_stage;
  NS::SharedPtr<MTL::Buffer> scaled_x_output_stage;
  if (format.value != 0) {
    dequant_x_output_stage = NS::TransferPtr(device->newBuffer(output_bytes, kSharedResourceOptions));
    scaled_x_output_stage = NS::TransferPtr(device->newBuffer(output_bytes, kSharedResourceOptions));
  }

  auto quantized_buffer = NS::TransferPtr(device->newBuffer(quantized.size(), kPrivateResourceOptions));
  NS::SharedPtr<MTL::Buffer> packed_quantized_buffer;
  if (format.value != 0)
    packed_quantized_buffer = NS::TransferPtr(device->newBuffer(packed_quantized.size(), kPrivateResourceOptions));
  auto dense_buffer = NS::TransferPtr(device->newBuffer(dense_bytes, kPrivateResourceOptions));
  auto dequant_dense_buffer = NS::TransferPtr(device->newBuffer(dense_bytes, kPrivateResourceOptions));
  auto dequant_x_rowwise_buffer = NS::TransferPtr(device->newBuffer(quantized.size(), kPrivateResourceOptions));
  auto vector_buffer = NS::TransferPtr(device->newBuffer(vector_bytes, kPrivateResourceOptions));
  auto bias_buffer = NS::TransferPtr(device->newBuffer(bias_bytes, kPrivateResourceOptions));
  auto raw_output_buffer = NS::TransferPtr(device->newBuffer(output_bytes, kPrivateResourceOptions));
  auto dequant_output_buffer = NS::TransferPtr(device->newBuffer(output_bytes, kPrivateResourceOptions));
  auto scaled_output_buffer = NS::TransferPtr(device->newBuffer(output_bytes, kPrivateResourceOptions));
  NS::SharedPtr<MTL::Buffer> dequant_x_output_buffer;
  NS::SharedPtr<MTL::Buffer> scaled_x_output_buffer;
  if (format.value != 0) {
    dequant_x_output_buffer = NS::TransferPtr(device->newBuffer(output_bytes, kPrivateResourceOptions));
    scaled_x_output_buffer = NS::TransferPtr(device->newBuffer(output_bytes, kPrivateResourceOptions));
  }

  if (!quantized_stage || !dense_stage || !vector_stage || !bias_stage ||
      !raw_output_stage || !dequant_output_stage || !scaled_output_stage ||
      !quantized_buffer || !dense_buffer || !dequant_dense_buffer || !vector_buffer ||
      !bias_buffer || !raw_output_buffer || !dequant_output_buffer || !scaled_output_buffer ||
      (format.value != 0 && (!packed_quantized_stage || !dequant_x_output_stage || !scaled_x_output_stage ||
          !packed_quantized_buffer || !dequant_x_rowwise_buffer || !dequant_x_output_buffer || !scaled_x_output_buffer))) {
    std::cerr << "failed to allocate buffers for shape rows=" << shape.rows
              << " cols=" << shape.cols << '\n';
    return 1;
  }

  upload_buffer(command_queue, quantized_stage.get(), quantized_buffer.get(), quantized.size());
  if (format.value != 0)
    upload_buffer(command_queue, packed_quantized_stage.get(), packed_quantized_buffer.get(), packed_quantized.size());
  upload_buffer(command_queue, dense_stage.get(), dense_buffer.get(), dense_bytes);
  upload_buffer(command_queue, vector_stage.get(), vector_buffer.get(), vector_bytes);
  upload_buffer(command_queue, bias_stage.get(), bias_buffer.get(), bias_bytes);

  auto pool = NS::AutoreleasePool::alloc()->init();
  Pipelines pipelines = create_pipelines(device, shader_cache, shape, precision, config.mrows, config.fused_bias, format.value);
  pool->drain();

  MTL::Buffer* const active_bias = config.fused_bias ? bias_buffer.get() : nullptr;

  (void)run_raw_once(command_queue, pipelines, shape, 1, dense_buffer.get(), vector_buffer.get(), raw_output_buffer.get(), active_bias);
  (void)run_dequant_gemv_once(command_queue, pipelines, shape, 1, quantized_buffer.get(), dequant_dense_buffer.get(), vector_buffer.get(), dequant_output_buffer.get(), active_bias);
  (void)run_scaled_once(command_queue, pipelines, shape, 1, quantized_buffer.get(), vector_buffer.get(), scaled_output_buffer.get(), active_bias);
  if (format.value != 0) {
    (void)run_dequant_x_scaled_once(command_queue, pipelines, shape, 1, packed_quantized_buffer.get(), dequant_x_rowwise_buffer.get(), vector_buffer.get(), dequant_x_output_buffer.get(), active_bias);
    (void)run_scaled_x_once(command_queue, pipelines, shape, 1, packed_quantized_buffer.get(), vector_buffer.get(), scaled_x_output_buffer.get(), active_bias);
  }

  download_buffer(command_queue, raw_output_buffer.get(), raw_output_stage.get(), output_bytes);
  download_buffer(command_queue, dequant_output_buffer.get(), dequant_output_stage.get(), output_bytes);
  download_buffer(command_queue, scaled_output_buffer.get(), scaled_output_stage.get(), output_bytes);
  if (format.value != 0) {
    download_buffer(command_queue, dequant_x_output_buffer.get(), dequant_x_output_stage.get(), output_bytes);
    download_buffer(command_queue, scaled_x_output_buffer.get(), scaled_x_output_stage.get(), output_bytes);
  }
  std::vector<T> raw_output((size_t)config.mrows * shape.rows);
  std::vector<T> dequant_output((size_t)config.mrows * shape.rows);
  std::vector<T> scaled_output((size_t)config.mrows * shape.rows);
  std::memcpy(raw_output.data(), raw_output_stage->contents(), output_bytes);
  std::memcpy(dequant_output.data(), dequant_output_stage->contents(), output_bytes);
  std::memcpy(scaled_output.data(), scaled_output_stage->contents(), output_bytes);
  const ValidationStats dequant_validation = validate_output(raw_output, dequant_output);
  const ValidationStats scaled_validation = validate_output(raw_output, scaled_output);
  ValidationStats scaled_x_validation;
  if (format.value != 0) {
    std::vector<T> dequant_x_output((size_t)config.mrows * shape.rows);
    std::vector<T> scaled_x_output((size_t)config.mrows * shape.rows);
    std::memcpy(dequant_x_output.data(), dequant_x_output_stage->contents(), output_bytes);
    std::memcpy(scaled_x_output.data(), scaled_x_output_stage->contents(), output_bytes);
    scaled_x_validation = validate_output(dequant_x_output, scaled_x_output);
  }

  Stats raw_stats;
  Stats dequant_stats;
  Stats scaled_stats;
  Stats dequant_x_stats;
  Stats scaled_x_stats;
  if (!benchmark(config, [&]() {
        return run_raw_once(command_queue, pipelines, shape, config.duplicated_dispatches,
            dense_buffer.get(), vector_buffer.get(), raw_output_buffer.get(), active_bias);
      }, &raw_stats))
    return 1;
  if (!benchmark(config, [&]() {
        return run_dequant_gemv_once(command_queue, pipelines, shape, config.duplicated_dispatches,
            quantized_buffer.get(), dequant_dense_buffer.get(), vector_buffer.get(), dequant_output_buffer.get(), active_bias);
      }, &dequant_stats))
    return 1;
  if (!benchmark(config, [&]() {
        return run_scaled_once(command_queue, pipelines, shape, config.duplicated_dispatches,
            quantized_buffer.get(), vector_buffer.get(), scaled_output_buffer.get(), active_bias);
      }, &scaled_stats))
    return 1;
  if (format.value != 0) {
    if (!benchmark(config, [&]() {
          return run_dequant_x_scaled_once(command_queue, pipelines, shape, config.duplicated_dispatches,
              packed_quantized_buffer.get(), dequant_x_rowwise_buffer.get(), vector_buffer.get(), dequant_x_output_buffer.get(), active_bias);
        }, &dequant_x_stats))
      return 1;
    if (!benchmark(config, [&]() {
          return run_scaled_x_once(command_queue, pipelines, shape, config.duplicated_dispatches,
              packed_quantized_buffer.get(), vector_buffer.get(), scaled_x_output_buffer.get(), active_bias);
        }, &scaled_x_stats))
      return 1;
  }

  const double flops = 2.0 * (double)config.mrows * shape.rows * shape.cols;
  const double bias_read_bytes = config.fused_bias ? (double)bias_bytes : 0.0;
  const double raw_bytes = (double)dense_bytes + (double)row_groups * vector_bytes + output_bytes + bias_read_bytes;
  const double qx_scale_bytes = (double)shape.rows * element_bytes;
  const double scaled_bytes = (double)shape.rows * shape.cols + qx_scale_bytes +
      (double)row_groups * vector_bytes + output_bytes + bias_read_bytes;
  const double dequant_bytes = (double)shape.rows * shape.cols + qx_scale_bytes +
      dense_bytes + raw_bytes;
  const double packed_scaled_bytes = format.value != 0 ? (double)packed_quantized.size() +
      (double)row_groups * vector_bytes + output_bytes + bias_read_bytes : 0;
  const double packed_dequant_bytes = format.value != 0 ? (double)packed_quantized.size() +
      (double)quantized.size() + scaled_bytes : 0;

  std::cout << "shape mrows=" << config.mrows
            << " rows=" << shape.rows << " cols=" << shape.cols
            << " dtype=" << config.dtype
            << " format=" << format.name
            << " bias=" << (config.fused_bias ? 1 : 0)
            << " dispatches=" << config.duplicated_dispatches
            << " runs=" << config.timed_iterations
            << " sleep_ms=" << config.sleep_ms;
  if (format.value != 0) {
    std::cout << " rowwise_bytes=" << quantized.size()
              << " packed_bytes=" << packed_quantized.size()
              << " compression=" << (double)quantized.size() / (double)packed_quantized.size();
  }
  std::cout << '\n';
  print_validation("dequant_vs_raw", dequant_validation);
  print_validation("scaled_vs_raw", scaled_validation);
  if (format.value != 0)
    print_validation("packed_direct_vs_packed_dequant", scaled_x_validation);
  print_stats("raw_gemv", raw_stats, flops, raw_bytes);
  print_stats("dequant_gemv", dequant_stats, flops, dequant_bytes);
  print_stats("rowwise_int8_gemv", scaled_stats, flops, scaled_bytes);
  if (format.value != 0) {
    print_stats("packed_dequant_int8_gemv", dequant_x_stats, flops, packed_dequant_bytes);
    print_stats("packed_direct_int8_gemv", scaled_x_stats, flops, packed_scaled_bytes);
    std::cout << "  packed_direct_vs_rowwise_best3="
              << scaled_stats.best3_average_seconds / scaled_x_stats.best3_average_seconds
              << " packed_direct_vs_dequant_best3="
              << dequant_x_stats.best3_average_seconds / scaled_x_stats.best3_average_seconds
              << " packed_direct_time_over_rowwise="
              << scaled_x_stats.best3_average_seconds / scaled_stats.best3_average_seconds
              << "\n\n";
  } else {
    std::cout << "  scaled_vs_raw_speedup="
              << raw_stats.best3_average_seconds / scaled_stats.best3_average_seconds
              << " scaled_vs_dequant_speedup="
              << dequant_stats.best3_average_seconds / scaled_stats.best3_average_seconds
              << "\n\n";
  }
  return 0;
}

void print_usage(const char* const argv0)
{
  std::cerr << "usage: " << argv0 << " [--dtype fp16|bf16|fp32] "
            << "[--format rowwise|q4_k|q3_k|q2_k|iq2_s|iq2_xs|iq3_s|iq3_xxs] [--all-formats] "
            << "[--bias 0|1] [--mrows 1|2|3] [--shape rows cols] [--runs 20] [--warmup 3] "
            << "[--dispatches 5] [--sleep-ms 200]\n";
}

bool parse_args(int argc, char** argv, Config* const config)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dtype" && i + 1 < argc) {
      config->dtype = argv[++i];
      if (config->dtype != "fp16" && config->dtype != "bf16" && config->dtype != "fp32")
        return false;
    } else if (arg == "--format" && i + 1 < argc) {
      config->format = argv[++i];
      if (format_from_name(config->format) == UINT32_MAX)
        return false;
    } else if (arg == "--all-formats") {
      config->all_formats = true;
    } else if (arg == "--bias" && i + 1 < argc) {
      int value = 0;
      if (!parse_int(argv[++i], &value))
        return false;
      config->fused_bias = value != 0;
    } else if (arg == "--mrows" && i + 1 < argc) {
      uint32_t value = 0;
      if (!parse_uint32(argv[++i], &value))
        return false;
      config->mrows = value;
    } else if (arg == "--shape" && i + 2 < argc) {
      uint32_t rows = 0;
      uint32_t cols = 0;
      if (!parse_uint32(argv[++i], &rows) || !parse_uint32(argv[++i], &cols))
        return false;
      config->shapes.push_back(Shape{ rows, cols });
    } else if (arg == "--runs" && i + 1 < argc) {
      if (!parse_int(argv[++i], &config->timed_iterations))
        return false;
    } else if (arg == "--warmup" && i + 1 < argc) {
      if (!parse_int(argv[++i], &config->warmup_iterations))
        return false;
    } else if (arg == "--dispatches" && i + 1 < argc) {
      if (!parse_int(argv[++i], &config->duplicated_dispatches))
        return false;
    } else if (arg == "--sleep-ms" && i + 1 < argc) {
      if (!parse_int(argv[++i], &config->sleep_ms))
        return false;
    } else {
      return false;
    }
  }
  if ((config->mrows != 1 && config->mrows != 2 && config->mrows != 3) ||
      config->timed_iterations <= 0 || config->warmup_iterations < 0 ||
      config->duplicated_dispatches <= 0 || config->sleep_ms < 0)
    return false;
  if (config->shapes.empty()) {
    config->shapes = {
      Shape{ 1024, 1024 },
      Shape{ 2560, 1024 },
      Shape{ 1024, 2560 },
      Shape{ 2560, 9216 },
      Shape{ 9216, 2560 },
      Shape{ 2560, 2560 },
      Shape{ 4096, 3072 },
      Shape{ 3072, 4096 },
      Shape{ 5120, 5120 },
      Shape{ 5120, 17408 },
      Shape{ 17408, 5120 },
    };
  }
  for (const Shape shape : config->shapes) {
    if (shape.rows == 0 || shape.cols == 0 || (shape.cols % 4) != 0)
      return false;
  }
  const std::vector<FormatInfo> formats = selected_formats(*config);
  if (formats.empty())
    return false;
  for (const FormatInfo format : formats) {
    if (format.value == 0)
      continue;
    for (const Shape shape : config->shapes)
      if ((shape.rows % 256) != 0 || (shape.cols % 256) != 0)
        return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  Config config;
  if (!parse_args(argc, argv, &config)) {
    print_usage(argv[0]);
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

  ShaderCache shader_cache;
  const GEMMOperandPrecision precision = precision_for_dtype(config.dtype);
  const std::vector<FormatInfo> formats = selected_formats(config);
  int status = 0;
  for (const Shape shape : config.shapes) {
    for (const FormatInfo format : formats) {
      if (config.dtype == "fp32")
        status = run_shape<float>(device.get(), command_queue.get(), shader_cache, config, shape, format, precision);
      else if (config.dtype == "bf16")
        status = run_shape<bfloat16_value>(device.get(), command_queue.get(), shader_cache, config, shape, format, precision);
      else
        status = run_shape<half_float>(device.get(), command_queue.get(), shader_cache, config, shape, format, precision);
      if (status != 0)
        break;
    }
    if (status != 0)
      break;
  }

  fflush(stdout);
  fflush(stderr);
  std::_Exit(status);
}
