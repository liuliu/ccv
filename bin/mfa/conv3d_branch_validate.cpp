#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/Conv3DDescriptor.hpp"
#include "nnc/mfa/kernels/Conv3DKernel.hpp"
#include "nnc/mfa/kernels/Conv3DKernelDescriptor.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/NAConv3DDescriptor.hpp"
#include "nnc/mfa/kernels/NAConv3DKernel.hpp"
#include "nnc/mfa/kernels/NAConv3DKernelDescriptor.hpp"
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
    const uint32_t n,
    const uint32_t d,
    const uint32_t h,
    const uint32_t w,
    const uint32_t c)
{
  return ((((size_t)n * shape.input_d + d) * shape.input_h + h) * shape.input_w + w) *
      shape.input_c + c;
}

size_t weight_index_oidhw(
    const Conv3DShape& shape,
    const uint32_t o,
    const uint32_t i,
    const uint32_t kd,
    const uint32_t kh,
    const uint32_t kw)
{
  return ((((size_t)o * shape.input_c + i) * shape.kernel_d + kd) * shape.kernel_h + kh) *
      shape.kernel_w + kw;
}

size_t output_index(
    const Conv3DShape& shape,
    const uint32_t n,
    const uint32_t d,
    const uint32_t h,
    const uint32_t w,
    const uint32_t o)
{
  return ((((size_t)n * shape.output_d + d) * shape.output_h + h) * shape.output_w + w) *
      shape.output_c + o;
}

template <typename T>
std::vector<T> make_data(const size_t size, const float scale)
{
  std::vector<T> values(size);
  for (size_t i = 0; i < size; ++i)
    values[i] =
        static_cast<T>(static_cast<float>(static_cast<int>(i % 17) - 8) * scale);
  return values;
}

std::vector<float> cpu_conv3d(
    const Conv3DShape& shape,
    const std::vector<half_float>& input,
    const std::vector<half_float>& weights)
{
  std::vector<float> output(
      (size_t)shape.batch * shape.output_d * shape.output_h * shape.output_w *
          shape.output_c,
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
                               input[input_index(shape, n, id, (uint32_t)ih,
                                                 (uint32_t)iw, ic)]) *
                        static_cast<float>(
                            weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)]);
                }
            output[output_index(shape, n, od, oh, ow, oc)] = sum;
          }
  return output;
}

Conv3DDescriptor make_generic_descriptor(const Conv3DShape& shape)
{
  Conv3DDescriptor descriptor;
  descriptor.dataType = 16;
  descriptor.batchDimension = shape.batch;
  descriptor.inputChannels = shape.input_c;
  descriptor.outputChannels = shape.output_c;
  descriptor.blockDimensions = simd::ushort2 {
      static_cast<unsigned short>(
          std::min<uint32_t>(
              std::max<uint32_t>(((shape.output_c + 31u) / 32u) * 32u, 32u), 128u)),
      32,
  };
  descriptor.paddingLeft = shape.pad_w;
  descriptor.paddingRight = shape.pad_w;
  descriptor.paddingTop = shape.pad_h;
  descriptor.paddingBottom = shape.pad_h;
  descriptor.matrixDimensions =
      simd::uint3 { shape.output_d, shape.output_h, shape.output_w };
  descriptor.kernelDimensions =
      simd::uint3 { shape.kernel_d, shape.kernel_h, shape.kernel_w };
  descriptor.useBias = false;
  return descriptor;
}

NAConv3DDescriptor make_na_descriptor(const Conv3DShape& shape)
{
  NAConv3DDescriptor descriptor;
  descriptor.dataType = 16;
  descriptor.batchDimension = shape.batch;
  descriptor.inputChannels = shape.input_c;
  descriptor.outputChannels = shape.output_c;
  descriptor.paddingLeft = shape.pad_w;
  descriptor.paddingRight = shape.pad_w;
  descriptor.paddingTop = shape.pad_h;
  descriptor.paddingBottom = shape.pad_h;
  descriptor.matrixDimensions =
      simd::uint3 { shape.output_d, shape.output_h, shape.output_w };
  descriptor.kernelDimensions =
      simd::uint3 { shape.kernel_d, shape.kernel_h, shape.kernel_w };
  descriptor.useBias = false;
  return descriptor;
}

void encode_generic_permutation(
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
  const uint32_t element_count = shape.output_c * shape.input_c * shape.kernel_d *
      shape.kernel_h * shape.kernel_w;
  const auto threadgroup_size =
      kernel->permutationThreadgroupSize(kernel->permutationPipeline.get());
  encoder->dispatchThreadgroups(
      MTL::Size((element_count + threadgroup_size - 1) / threadgroup_size, 1, 1),
      MTL::Size(threadgroup_size, 1, 1));
  encoder->endEncoding();
}

void encode_generic_conv(
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
      MTL::Size(kernel->threadgroupSize(pipeline_value->pipeline.get(), descriptor), 1,
                1));
  encoder->endEncoding();
}

void encode_na_permutation(
    MTL::CommandBuffer* command_buffer,
    NAConv3DKernel* kernel,
    MTL::Buffer* weights_oidhw,
    MTL::Buffer* weights_dhwio,
    const Conv3DShape& shape)
{
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(kernel->permutationPipeline.get());
  encoder->setBuffer(weights_oidhw, 0, 0);
  encoder->setBuffer(weights_dhwio, 0, 1);
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
  const uint32_t element_count = shape.output_c * shape.input_c * shape.kernel_d *
      shape.kernel_h * shape.kernel_w;
  const auto threadgroup_size =
      kernel->permutationThreadgroupSize(kernel->permutationPipeline.get());
  encoder->dispatchThreadgroups(
      MTL::Size((element_count + threadgroup_size - 1) / threadgroup_size, 1, 1),
      MTL::Size(threadgroup_size, 1, 1));
  encoder->endEncoding();
}

void encode_na_conv(
    MTL::CommandBuffer* command_buffer,
    PipelineValue<NAConv3DKernel>* pipeline_value,
    const NAConv3DDescriptor& descriptor,
    const Conv3DShape& shape,
    MTL::Buffer* input,
    MTL::Buffer* weights_dhwio,
    MTL::Buffer* output)
{
  auto* kernel = pipeline_value->kernel;
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  const uint32_t weight_slice_element_count =
      shape.kernel_h * shape.kernel_w * shape.input_c * shape.output_c;
  const uint32_t input_slice_element_count =
      shape.input_h * shape.input_w * shape.input_c;
  const uint32_t element_size = sizeof(half_float);
  const MTL::Size convolution_grid_size = kernel->threadgroupsPerGrid(descriptor);
  const MTL::Size convolution_group_size(
      kernel->threadgroupSize(pipeline_value->pipeline.get(), descriptor), 1, 1);
  encoder->setBuffer(output, 0, 2);
  for (uint32_t kernel_slice = 0; kernel_slice < shape.kernel_d; ++kernel_slice)
  {
    const uint32_t activation_base = kernel_slice * input_slice_element_count;
    const uint32_t weights_base = kernel_slice * weight_slice_element_count;
    encoder->setComputePipelineState(
        kernel_slice == 0 ? pipeline_value->pipeline.get()
                          : pipeline_value->second.get());
    encoder->setBuffer(input, (size_t)activation_base * element_size, 0);
    encoder->setBuffer(weights_dhwio, (size_t)weights_base * element_size, 1);
    encoder->dispatchThreadgroups(convolution_grid_size, convolution_group_size);
  }
  encoder->endEncoding();
}

struct ValidationResult {
  float max_abs_error;
  size_t max_error_index;
};

ValidationResult compare_output(
    const std::vector<float>& reference,
    const half_float* output,
    const size_t output_size)
{
  ValidationResult result{0.0f, 0};
  for (size_t i = 0; i < output_size; ++i)
  {
    const float abs_error = std::fabs(static_cast<float>(output[i]) - reference[i]);
    if (abs_error > result.max_abs_error)
    {
      result.max_abs_error = abs_error;
      result.max_error_index = i;
    }
  }
  return result;
}

ValidationResult run_generic_case(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    ShaderCache& shader_cache,
    const DeviceProperties& dprops,
    const Conv3DShape& shape,
    const std::vector<half_float>& input,
    const std::vector<half_float>& weights_oidhw,
    const std::vector<float>& reference)
{
  const auto descriptor = make_generic_descriptor(shape);
  auto pipeline_value =
      shader_cache.findKernel<Conv3DKernel, Conv3DDescriptor, Conv3DKernelDescriptor>(
          descriptor, device, dprops);
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
  encode_generic_permutation(command_buffer.get(), pipeline_value->kernel,
                             weights_oidhw_buffer.get(), weights_ok_buffer.get(), shape);
  encode_generic_conv(command_buffer.get(), pipeline_value, descriptor, input_buffer.get(),
                      weights_ok_buffer.get(), output_buffer.get());
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  return compare_output(reference,
                        static_cast<const half_float*>(output_buffer->contents()),
                        output.size());
}

ValidationResult run_na_case(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    ShaderCache& shader_cache,
    const DeviceProperties& dprops,
    const Conv3DShape& shape,
    const std::vector<half_float>& input,
    const std::vector<half_float>& weights_oidhw,
    const std::vector<float>& reference)
{
  const auto descriptor = make_na_descriptor(shape);
  auto pipeline_value = shader_cache
                            .findKernel<NAConv3DKernel, NAConv3DDescriptor,
                                        NAConv3DKernelDescriptor>(
                                descriptor, device, dprops);
  std::vector<half_float> output(reference.size(), half_float(0));

  auto input_buffer = NS::TransferPtr(device->newBuffer(
      input.data(),
      input.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weights_oidhw_buffer = NS::TransferPtr(device->newBuffer(
      weights_oidhw.data(),
      weights_oidhw.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weights_dhwio_buffer = NS::TransferPtr(device->newBuffer(
      weights_oidhw.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto output_buffer = NS::TransferPtr(device->newBuffer(
      output.data(),
      output.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  encode_na_permutation(command_buffer.get(), pipeline_value->kernel,
                        weights_oidhw_buffer.get(), weights_dhwio_buffer.get(), shape);
  encode_na_conv(command_buffer.get(), pipeline_value, descriptor, shape,
                 input_buffer.get(), weights_dhwio_buffer.get(), output_buffer.get());
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  return compare_output(reference,
                        static_cast<const half_float*>(output_buffer->contents()),
                        output.size());
}

int validate_channel_count(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    ShaderCache& shader_cache,
    const DeviceProperties& dprops,
    const uint32_t channels)
{
  const Conv3DShape shape{
      .batch = 1,
      .input_d = 4,
      .input_h = 33,
      .input_w = 35,
      .input_c = channels,
      .output_d = 2,
      .output_h = 33,
      .output_w = 35,
      .output_c = channels,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
      .pad_h = 1,
      .pad_w = 1,
  };
  const auto input = make_data<half_float>(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w *
          shape.input_c,
      0.03125f);
  const auto weights_oidhw = make_data<half_float>(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h *
          shape.kernel_w,
      -0.015625f);
  const auto reference = cpu_conv3d(shape, input, weights_oidhw);

  const auto generic_result = run_generic_case(device, command_queue, shader_cache,
                                               dprops, shape, input, weights_oidhw,
                                               reference);
  std::fprintf(stderr,
               "use_neural_accelerators=0 channels=%u max_abs_error=%.6f "
               "max_error_index=%zu\n",
               channels, generic_result.max_abs_error,
               generic_result.max_error_index);

  const auto na_result = run_na_case(device, command_queue, shader_cache, dprops, shape,
                                     input, weights_oidhw, reference);
  std::fprintf(stderr,
               "use_neural_accelerators=1 channels=%u max_abs_error=%.6f "
               "max_error_index=%zu\n",
               channels, na_result.max_abs_error, na_result.max_error_index);

  if (!(generic_result.max_abs_error <= 0.05f))
    return 2;
  if (!(na_result.max_abs_error <= 0.05f))
    return 3;
  return 0;
}

} // namespace

int main()
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

  int status = validate_channel_count(device.get(), command_queue.get(), shader_cache,
                                      dprops, 16);
  if (status == 0)
    status = validate_channel_count(device.get(), command_queue.get(), shader_cache,
                                    dprops, 48);
  pool->drain();
  return status;
}
