#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>

#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernel.hpp"
#include "nnc/mfa/kernels/GEMMKernelDescriptor.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

struct Conv3DShape {
  uint32_t batch;
  uint32_t input_d;
  uint32_t input_h;
  uint32_t input_w;
  uint32_t input_c;
  uint32_t output_c;
  uint32_t kernel_d;
  uint32_t kernel_h;
  uint32_t kernel_w;
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
    uint32_t output_d,
    uint32_t output_h,
    uint32_t output_w,
    uint32_t n,
    uint32_t d,
    uint32_t h,
    uint32_t w,
    uint32_t o)
{
  return ((((size_t)n * output_d + d) * output_h + h) * output_w + w) * shape.output_c + o;
}

std::vector<float> make_data(size_t size, float bias)
{
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i)
    values[i] = bias + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0625f;
  return values;
}

std::vector<float> cpu_conv3d(
    const Conv3DShape& shape,
    const std::vector<float>& input,
    const std::vector<float>& weights)
{
  const uint32_t output_d = shape.input_d - shape.kernel_d + 1;
  const uint32_t output_h = shape.input_h - shape.kernel_h + 1;
  const uint32_t output_w = shape.input_w - shape.kernel_w + 1;
  std::vector<float> output(
      (size_t)shape.batch * output_d * output_h * output_w * shape.output_c, 0.0f);
  for (uint32_t n = 0; n < shape.batch; ++n)
    for (uint32_t od = 0; od < output_d; ++od)
      for (uint32_t oh = 0; oh < output_h; ++oh)
        for (uint32_t ow = 0; ow < output_w; ++ow)
          for (uint32_t oc = 0; oc < shape.output_c; ++oc)
          {
            float sum = 0.0f;
            for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
              for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
                for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
                  for (uint32_t ic = 0; ic < shape.input_c; ++ic)
                    sum += input[input_index(shape, n, od + kd, oh + kh, ow + kw, ic)] *
                        weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)];
            output[output_index(shape, output_d, output_h, output_w, n, od, oh, ow, oc)] = sum;
          }
  return output;
}

std::vector<float> permute_weights_to_ok(
    const Conv3DShape& shape,
    const std::vector<float>& weights)
{
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;
  std::vector<float> permuted((size_t)shape.output_c * gemm_k);
  for (uint32_t oc = 0; oc < shape.output_c; ++oc)
  {
    uint32_t column = 0;
    for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
      for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
        for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
          for (uint32_t ic = 0; ic < shape.input_c; ++ic, ++column)
            permuted[(size_t)oc * gemm_k + column] =
                weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)];
  }
  return permuted;
}

std::vector<float> unfold_input_to_mk(
    const Conv3DShape& shape,
    uint32_t output_d,
    uint32_t output_h,
    uint32_t output_w,
    const std::vector<float>& input)
{
  const uint32_t gemm_m = shape.batch * output_d * output_h * output_w;
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;
  std::vector<float> unfolded((size_t)gemm_m * gemm_k);
  uint32_t row = 0;
  for (uint32_t n = 0; n < shape.batch; ++n)
    for (uint32_t od = 0; od < output_d; ++od)
      for (uint32_t oh = 0; oh < output_h; ++oh)
        for (uint32_t ow = 0; ow < output_w; ++ow, ++row)
        {
          uint32_t column = 0;
          for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
            for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
              for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
                for (uint32_t ic = 0; ic < shape.input_c; ++ic, ++column)
                  unfolded[(size_t)row * gemm_k + column] =
                      input[input_index(shape, n, od + kd, oh + kh, ow + kw, ic)];
        }
  return unfolded;
}

GEMMDescriptor make_descriptor(uint32_t m, uint32_t n, uint32_t k)
{
  GEMMDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint3{m, n, k};
  descriptor.memoryPrecisions = {
      .A = GEMMOperandPrecision::FP32,
      .B = GEMMOperandPrecision::FP32,
      .C = GEMMOperandPrecision::FP32,
      .bias = GEMMOperandPrecision::FP32,
  };
  descriptor.registerPrecisionC = std::nullopt;
  descriptor.leadingDimensions = std::nullopt;
  descriptor.batchStrides = std::nullopt;
  descriptor.transposeState = simd::uchar3{0, 1, 0};
  descriptor.loadPreviousC = false;
  descriptor.useBias = false;
  descriptor.loadM = false;
  descriptor.supportIndirectCommandBuffers = false;
  return descriptor;
}

std::vector<float> run_gemm(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    ShaderCache& shader_cache,
    const std::vector<float>& a,
    const std::vector<float>& b,
    uint32_t m,
    uint32_t n,
    uint32_t k)
{
  DeviceProperties dprops{};
  const auto descriptor = make_descriptor(m, n, k);
  auto pipeline_value =
      shader_cache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(
          descriptor, device, dprops);

  std::vector<float> c((size_t)m * n, 0.0f);
  auto buffer_a = NS::TransferPtr(device->newBuffer(
      a.data(),
      a.size() * sizeof(float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto buffer_b = NS::TransferPtr(device->newBuffer(
      b.data(),
      b.size() * sizeof(float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto buffer_c = NS::TransferPtr(device->newBuffer(
      c.data(),
      c.size() * sizeof(float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  const auto ceil_divide =
      [](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  const auto grid_size = MTL::Size(
      ceil_divide(n, pipeline_value->kernel->blockDimensions[1]),
      ceil_divide(m, pipeline_value->kernel->blockDimensions[0]),
      1);
  const auto group_size = MTL::Size(pipeline_value->kernel->threadgroupSize, 1, 1);

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline_value->pipeline.get());
  encoder->setThreadgroupMemoryLength(
      pipeline_value->kernel->threadgroupMemoryAllocation, 0);
  encoder->setBuffer(buffer_a.get(), 0, 0);
  encoder->setBuffer(buffer_b.get(), 0, 1);
  encoder->setBuffer(buffer_c.get(), 0, 2);
  encoder->useResource(buffer_a.get(), MTL::ResourceUsageRead);
  encoder->useResource(buffer_b.get(), MTL::ResourceUsageRead);
  encoder->useResource(buffer_c.get(), MTL::ResourceUsageWrite);
  encoder->dispatchThreadgroups(grid_size, group_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  const auto* raw = static_cast<const float*>(buffer_c->contents());
  return std::vector<float>(raw, raw + c.size());
}

} // namespace

int main()
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    fprintf(stderr, "Metal device unavailable.\n");
    std::_Exit(1);
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue)
  {
    fprintf(stderr, "Metal command queue unavailable.\n");
    std::_Exit(1);
  }

  const Conv3DShape shape{
      .batch = 1,
      .input_d = 4,
      .input_h = 4,
      .input_w = 4,
      .input_c = 4,
      .output_c = 8,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
  };
  const uint32_t output_d = shape.input_d - shape.kernel_d + 1;
  const uint32_t output_h = shape.input_h - shape.kernel_h + 1;
  const uint32_t output_w = shape.input_w - shape.kernel_w + 1;
  const uint32_t gemm_m = shape.batch * output_d * output_h * output_w;
  const uint32_t gemm_n = shape.output_c;
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;

  const auto input = make_data(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.input_c,
      0.25f);
  const auto weights = make_data(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h * shape.kernel_w,
      -0.125f);
  const auto reference = cpu_conv3d(shape, input, weights);
  const auto unfolded = unfold_input_to_mk(shape, output_d, output_h, output_w, input);
  const auto permuted_weights = permute_weights_to_ok(shape, weights);

  ShaderCache shader_cache;
  const auto gemm_output = run_gemm(
      device.get(), command_queue.get(), shader_cache, unfolded, permuted_weights, gemm_m, gemm_n, gemm_k);

  float max_abs_error = 0.0f;
  size_t max_error_index = 0;
  for (size_t i = 0; i < gemm_output.size(); ++i)
  {
    const float abs_error = std::fabs(gemm_output[i] - reference[i]);
    if (abs_error > max_abs_error)
    {
      max_abs_error = abs_error;
      max_error_index = i;
    }
  }

  fprintf(
      stderr,
      "conv3d layout scaffold: M=%u N=%u K=%u output=%ux%ux%u max_abs_error=%.6f max_error_index=%zu\n",
      gemm_m,
      gemm_n,
      gemm_k,
      output_d,
      output_h,
      output_w,
      max_abs_error,
      max_error_index);
  fflush(stderr);
  fflush(stdout);
  std::_Exit(max_abs_error <= 1e-4f ? 0 : 2);
}
