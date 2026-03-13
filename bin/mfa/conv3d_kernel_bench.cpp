#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <QuartzCore/QuartzCore.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/Conv3DDescriptor.hpp"
#include "nnc/mfa/kernels/Conv3DKernel.hpp"
#include "nnc/mfa/kernels/Conv3DKernelDescriptor.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

using half_float = _Float16;

struct Conv3DShape {
  uint32_t batch;
  uint32_t input_d;
  uint32_t input_h;
  uint32_t input_w;
  uint32_t input_c;
  uint32_t output_d;
  uint32_t output_h;
  uint32_t output_w;
  uint32_t output_c;
  uint32_t kernel_d;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t pad_h;
  uint32_t pad_w;
};

size_t input_index(
    const Conv3DShape& shape,
    uint32_t n,
    uint32_t d,
    uint32_t h,
    uint32_t w,
    uint32_t c)
{
  return ((((size_t)n * shape.input_d + d) * shape.input_h + h) * shape.input_w + w) *
      shape.input_c + c;
}

size_t weight_index_oidhw(
    const Conv3DShape& shape,
    uint32_t o,
    uint32_t i,
    uint32_t kd,
    uint32_t kh,
    uint32_t kw)
{
  return ((((size_t)o * shape.input_c + i) * shape.kernel_d + kd) * shape.kernel_h + kh) *
      shape.kernel_w + kw;
}

size_t output_index(
    const Conv3DShape& shape,
    uint32_t n,
    uint32_t d,
    uint32_t h,
    uint32_t w,
    uint32_t o)
{
  return ((((size_t)n * shape.output_d + d) * shape.output_h + h) * shape.output_w + w) *
      shape.output_c + o;
}

template <typename T>
std::vector<T> make_data(size_t size, float scale)
{
  std::vector<T> values(size);
  for (size_t i = 0; i < size; ++i)
    values[i] = static_cast<T>(
        static_cast<float>(static_cast<int>(i % 17) - 8) * scale);
  return values;
}

std::vector<float> cpu_conv3d(
    const Conv3DShape& shape,
    const std::vector<half_float>& input,
    const std::vector<half_float>& weights)
{
  std::vector<float> output(
      (size_t)shape.batch * shape.output_d * shape.output_h * shape.output_w * shape.output_c,
      0.0f);
  for (uint32_t n = 0; n < shape.batch; ++n)
    for (uint32_t od = 0; od < shape.output_d; ++od)
      for (uint32_t oh = 0; oh < shape.output_h; ++oh)
        for (uint32_t ow = 0; ow < shape.output_w; ++ow)
          for (uint32_t oc = 0; oc < shape.output_c; ++oc)
          {
            float sum = 0.0f;
            for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
              for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
                for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
                {
                  const uint32_t id = od + kd;
                  const int ih = static_cast<int>(oh) + static_cast<int>(kh) -
                      static_cast<int>(shape.pad_h);
                  const int iw = static_cast<int>(ow) + static_cast<int>(kw) -
                      static_cast<int>(shape.pad_w);
                  if (ih < 0 || ih >= static_cast<int>(shape.input_h) || iw < 0 ||
                      iw >= static_cast<int>(shape.input_w))
                    continue;
                  for (uint32_t ic = 0; ic < shape.input_c; ++ic)
                    sum += static_cast<float>(
                               input[input_index(shape, n, id, (uint32_t)ih, (uint32_t)iw, ic)]) *
                        static_cast<float>(
                            weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)]);
                }
            output[output_index(shape, n, od, oh, ow, oc)] = sum;
          }
  return output;
}

Conv3DDescriptor make_descriptor(const Conv3DShape& shape, simd::ushort2 block_dimensions)
{
  Conv3DDescriptor descriptor;
  descriptor.dataType = 16;
  descriptor.batchDimension = shape.batch;
  descriptor.inputChannels = shape.input_c;
  descriptor.outputChannels = shape.output_c;
  descriptor.blockDimensions = block_dimensions;
  descriptor.paddingLeft = shape.pad_w;
  descriptor.paddingRight = shape.pad_w;
  descriptor.paddingTop = shape.pad_h;
  descriptor.paddingBottom = shape.pad_h;
  descriptor.matrixDimensions = simd::uint3{
      shape.output_d, shape.output_h, shape.output_w};
  descriptor.kernelDimensions = simd::uint3{
      shape.kernel_d, shape.kernel_h, shape.kernel_w};
  descriptor.useBias = false;
  return descriptor;
}

void encode_permutation(
    MTL::CommandBuffer* command_buffer,
    Conv3DKernel* kernel,
    MTL::Buffer* weights_oidhw,
    MTL::Buffer* weights_ok,
    const Conv3DShape& shape)
{
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(kernel->permutationPipeline.get());
  encoder->setBuffer(weights_oidhw, 0, 0);
  encoder->setBuffer(weights_ok, 0, 1);
  uint32_t output_channels = shape.output_c;
  uint32_t input_channels = shape.input_c;
  uint32_t kernel_depth = shape.kernel_d;
  uint32_t kernel_height = shape.kernel_h;
  uint32_t kernel_width = shape.kernel_w;
  encoder->setBytes(&output_channels, sizeof(output_channels), 2);
  encoder->setBytes(&input_channels, sizeof(input_channels), 3);
  encoder->setBytes(&kernel_depth, sizeof(kernel_depth), 4);
  encoder->setBytes(&kernel_height, sizeof(kernel_height), 5);
  encoder->setBytes(&kernel_width, sizeof(kernel_width), 6);
  const uint32_t element_count =
      shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h * shape.kernel_w;
  const auto threadgroup_size =
      kernel->permutationThreadgroupSize(kernel->permutationPipeline.get());
  encoder->dispatchThreadgroups(
      MTL::Size((element_count + threadgroup_size - 1) / threadgroup_size, 1, 1),
      MTL::Size(threadgroup_size, 1, 1));
  encoder->endEncoding();
}

void encode_conv(
    MTL::CommandBuffer* command_buffer,
    PipelineValue<Conv3DKernel>* pipeline_value,
    const Conv3DDescriptor& descriptor,
    MTL::Buffer* input,
    MTL::Buffer* weights_ok,
    MTL::Buffer* output)
{
  auto* kernel = pipeline_value->kernel;
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline_value->pipeline.get());
  encoder->setBuffer(input, 0, 0);
  encoder->setBuffer(weights_ok, 0, 1);
  encoder->setBuffer(output, 0, 2);
  encoder->dispatchThreadgroups(
      kernel->threadgroupsPerGrid(descriptor),
      MTL::Size(kernel->threadgroupSize(pipeline_value->pipeline.get(), descriptor), 1, 1));
  encoder->endEncoding();
}

void validate_small(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    ShaderCache& shader_cache,
    const DeviceProperties& dprops)
{
  const Conv3DShape shape{
      .batch = 1,
      .input_d = 4,
      .input_h = 9,
      .input_w = 65,
      .input_c = 16,
      .output_d = 2,
      .output_h = 9,
      .output_w = 65,
      .output_c = 32,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
      .pad_h = 1,
      .pad_w = 1,
  };
  const auto descriptor = make_descriptor(shape, simd::ushort2 { 32, 32 });
  auto pipeline_value =
      shader_cache.findKernel<Conv3DKernel, Conv3DDescriptor, Conv3DKernelDescriptor>(
          descriptor, device, dprops);

  const auto input = make_data<half_float>(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.input_c,
      0.03125f);
  const auto weights_oidhw = make_data<half_float>(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h *
          shape.kernel_w,
      -0.015625f);
  const auto reference = cpu_conv3d(shape, input, weights_oidhw);
  std::vector<half_float> output(reference.size(), half_float(0));

  auto input_buffer = NS::TransferPtr(device->newBuffer(
      input.data(),
      input.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weights_oidhw_buffer = NS::TransferPtr(device->newBuffer(
      weights_oidhw.data(),
      weights_oidhw.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weights_ok_buffer = NS::TransferPtr(device->newBuffer(
      weights_oidhw.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto output_buffer = NS::TransferPtr(device->newBuffer(
      output.data(),
      output.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  encode_permutation(command_buffer.get(), pipeline_value->kernel, weights_oidhw_buffer.get(), weights_ok_buffer.get(), shape);
  encode_conv(command_buffer.get(), pipeline_value, descriptor, input_buffer.get(), weights_ok_buffer.get(), output_buffer.get());
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  const auto* raw = static_cast<const half_float*>(output_buffer->contents());
  float max_abs_error = 0.0f;
  size_t max_error_index = 0;
  for (size_t i = 0; i < output.size(); ++i)
  {
    const float abs_error =
        std::fabs(static_cast<float>(raw[i]) - reference[i]);
    if (abs_error > max_abs_error)
    {
      max_abs_error = abs_error;
      max_error_index = i;
    }
  }
  std::fprintf(
      stderr,
      "conv3d kernel validation: output=%ux%ux%u channels=%u max_abs_error=%.6f max_error_index=%zu\n",
      shape.output_d,
      shape.output_h,
      shape.output_w,
      shape.output_c,
      max_abs_error,
      max_error_index);
  if (!(max_abs_error <= 0.05f))
    std::_Exit(2);
}

} // namespace

int main(int argc, char** argv)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    std::fprintf(stderr, "Metal device unavailable.\n");
    std::_Exit(1);
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue)
  {
    std::fprintf(stderr, "Metal command queue unavailable.\n");
    std::_Exit(1);
  }

  ShaderCache shader_cache;
  DeviceProperties dprops{};
  validate_small(device.get(), command_queue.get(), shader_cache, dprops);

  uint32_t channels = 256;
  if (argc >= 2)
  {
    const long parsed = std::strtol(argv[1], nullptr, 10);
    if (parsed <= 0)
    {
      std::fprintf(stderr, "Invalid channel count: %s\n", argv[1]);
      std::_Exit(2);
    }
    channels = static_cast<uint32_t>(parsed);
  }
  simd::ushort2 block_dimensions {
      static_cast<unsigned short>(
          std::min<uint32_t>(((channels + 31u) / 32u) * 32u, 128u)),
      32,
  };
  if (argc >= 3)
  {
    const long parsed_block_n = std::strtol(argv[2], nullptr, 10);
    long parsed_block_m = 32;
    if (argc >= 4)
      parsed_block_m = std::strtol(argv[3], nullptr, 10);
    if (parsed_block_n <= 0 || parsed_block_m != 32 || (parsed_block_n % 32) != 0)
    {
      std::fprintf(
          stderr,
          "Invalid block dimensions: %s x %s (M tile must stay 32, N tile must be a multiple of 32)\n",
          argv[2],
          argc >= 4 ? argv[3] : "32");
      std::_Exit(2);
    }
    block_dimensions = simd::ushort2 {
        static_cast<unsigned short>(parsed_block_n),
        static_cast<unsigned short>(parsed_block_m),
    };
  }

  const Conv3DShape shape{
      .batch = 1,
      .input_d = 16,
      .input_h = 128,
      .input_w = 128,
      .input_c = channels,
      .output_d = 14,
      .output_h = 128,
      .output_w = 128,
      .output_c = channels,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
      .pad_h = 1,
      .pad_w = 1,
  };
  const auto descriptor = make_descriptor(shape, block_dimensions);
  auto pipeline_value =
      shader_cache.findKernel<Conv3DKernel, Conv3DDescriptor, Conv3DKernelDescriptor>(
          descriptor, device.get(), dprops);

  const auto input = make_data<half_float>(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.input_c,
      0.03125f);
  const auto weights_oidhw = make_data<half_float>(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h *
          shape.kernel_w,
      -0.015625f);
  const size_t output_count =
      (size_t)shape.batch * shape.output_d * shape.output_h * shape.output_w * shape.output_c;
  std::vector<half_float> output(output_count, half_float(0));

  auto input_buffer = NS::TransferPtr(device->newBuffer(
      input.data(),
      input.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weights_oidhw_buffer = NS::TransferPtr(device->newBuffer(
      weights_oidhw.data(),
      weights_oidhw.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weights_ok_buffer = NS::TransferPtr(device->newBuffer(
      weights_oidhw.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto output_buffer = NS::TransferPtr(device->newBuffer(
      output.data(),
      output.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  {
    auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
    encode_permutation(command_buffer.get(), pipeline_value->kernel, weights_oidhw_buffer.get(), weights_ok_buffer.get(), shape);
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
  }

  const int warmup_iterations = 2;
  const int timed_iterations = 10;
  double total_seconds = 0.0;

  for (int iteration = 0; iteration < warmup_iterations + timed_iterations; ++iteration)
  {
    auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
    encode_conv(command_buffer.get(), pipeline_value, descriptor, input_buffer.get(), weights_ok_buffer.get(), output_buffer.get());

    const double start_time = CACurrentMediaTime();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    const double end_time = CACurrentMediaTime();

    if (iteration >= warmup_iterations)
      total_seconds += end_time - start_time;
  }

  const double average_seconds = total_seconds / timed_iterations;
  const double flops =
      2.0 * static_cast<double>(shape.batch) * static_cast<double>(shape.output_d) *
      static_cast<double>(shape.output_h) * static_cast<double>(shape.output_w) *
      static_cast<double>(shape.output_c) * static_cast<double>(shape.kernel_d) *
      static_cast<double>(shape.kernel_h) * static_cast<double>(shape.kernel_w) *
      static_cast<double>(shape.input_c);
  const double tflops = flops / average_seconds / 1e12;

  const auto* raw_output = static_cast<const half_float*>(output_buffer->contents());
  double checksum = 0.0;
  const size_t checksum_count = std::min<size_t>(1024, output.size());
  for (size_t i = 0; i < checksum_count; ++i)
    checksum += static_cast<double>(raw_output[i]);

  const auto grid = pipeline_value->kernel->threadgroupsPerGrid(descriptor);
  std::fprintf(
      stderr,
      "conv3d kernel bench: device=%s input=%ux%ux%ux%ux%u output=%ux%ux%ux%ux%u "
      "kernel=%ux%ux%u block=%u x %u padding=(0,%u,%u) grid=%llu x %llu x %llu warmup=%d timed=%d "
      "avg_ms=%.3f tflops=%.3f checksum1024=%.6f\n",
      device->name()->utf8String(),
      shape.batch,
      shape.input_d,
      shape.input_h,
      shape.input_w,
      shape.input_c,
      shape.batch,
      shape.output_d,
      shape.output_h,
      shape.output_w,
      shape.output_c,
      shape.kernel_d,
      shape.kernel_h,
      shape.kernel_w,
      descriptor.blockDimensions[0],
      descriptor.blockDimensions[1],
      shape.pad_h,
      shape.pad_w,
      static_cast<unsigned long long>(grid.width),
      static_cast<unsigned long long>(grid.height),
      static_cast<unsigned long long>(grid.depth),
      warmup_iterations,
      timed_iterations,
      average_seconds * 1e3,
      tflops,
      checksum);
  std::fflush(stderr);

  pool->drain();
  std::_Exit(0);
}
