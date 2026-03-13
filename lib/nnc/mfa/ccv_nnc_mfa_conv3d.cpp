#include "ccv_nnc_mfa.hpp"

using namespace ccv::nnc;

#include <algorithm>

#include "kernels/ShaderCache.hpp"
#include "kernels/NAConv3DKernel.hpp"
#include "kernels/NAConv3DKernelDescriptor.hpp"
#include "kernels/NAConv3DDescriptor.hpp"
#include "kernels/Conv3DKernel.hpp"
#include "kernels/Conv3DKernelDescriptor.hpp"
#include "kernels/Conv3DDescriptor.hpp"

size_t ccv_nnc_mfa_conv3d_reserved_scratch_size(ccv_nnc_mfa_conv3d_params_t params)
{
  const uint32_t element_size = params.data_type == 3 ? 4 : 2;
  return (size_t)params.filter_dimensions[0] * params.filter_dimensions[1] * params.filter_dimensions[2] * params.input_channels * params.output_channels * element_size;
}

void ccv_nnc_mfa_prepare_conv3d(mfa::context* context, ccv_nnc_mfa_conv3d_params_t params)
{
  // No-op.
}

void ccv_nnc_mfa_encode_conv3d(mfa::context* context, ccv_nnc_mfa_conv3d_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    num_tensors += 1;
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3 || num_tensors == 4);
  CCV_NNC_MFA_PRECONDITION(params.fused_bias == (num_tensors == 4));
  CCV_NNC_MFA_PRECONDITION(params.groups == 1);
  CCV_NNC_MFA_PRECONDITION(params.format == CCV_TENSOR_FORMAT_NHWC);
  CCV_NNC_MFA_PRECONDITION(params.filter_dimensions[1] == params.filter_dimensions[2]);
  CCV_NNC_MFA_PRECONDITION((params.filter_dimensions[1] % 2) == 1);
  CCV_NNC_MFA_PRECONDITION(params.stride_dimensions[0] == 1 && params.stride_dimensions[1] == 1 && params.stride_dimensions[2] == 1);
  CCV_NNC_MFA_PRECONDITION(params.dilation_dimensions[0] == 1 && params.dilation_dimensions[1] == 1 && params.dilation_dimensions[2] == 1);
  const int32_t total_padding_height = (int32_t)params.output_dimensions[1] + (int32_t)params.filter_dimensions[1] - 1 - (int32_t)params.input_dimensions[1];
  const int32_t total_padding_width = (int32_t)params.output_dimensions[2] + (int32_t)params.filter_dimensions[2] - 1 - (int32_t)params.input_dimensions[2];
  CCV_NNC_MFA_PRECONDITION(params.output_dimensions[0] + params.filter_dimensions[0] - 1 == params.input_dimensions[0]);
  CCV_NNC_MFA_PRECONDITION(total_padding_height >= (int32_t)params.padding_bottom);
  CCV_NNC_MFA_PRECONDITION(total_padding_width >= (int32_t)params.padding_right);
  const uint32_t padding_right = params.padding_right;
  const uint32_t padding_left = (uint32_t)(total_padding_width - (int32_t)padding_right);
  const uint32_t padding_bottom = params.padding_bottom;
  const uint32_t padding_top = (uint32_t)(total_padding_height - (int32_t)padding_bottom);
  if (params.use_neural_accelerators) {
    NAConv3DDescriptor conv3d_desc;
    conv3d_desc.dataType = params.data_type;
    conv3d_desc.batchDimension = params.batch_size;
    conv3d_desc.inputChannels = params.input_channels;
    conv3d_desc.outputChannels = params.output_channels;
    conv3d_desc.paddingLeft = padding_left;
    conv3d_desc.paddingRight = padding_right;
    conv3d_desc.paddingTop = padding_top;
    conv3d_desc.paddingBottom = padding_bottom;
    conv3d_desc.matrixDimensions = simd::uint3 { params.output_dimensions[0], params.output_dimensions[1], params.output_dimensions[2] };
    conv3d_desc.kernelDimensions = simd::uint3 { params.filter_dimensions[0], params.filter_dimensions[1], params.filter_dimensions[2] };
    conv3d_desc.useBias = params.fused_bias;
    auto pool = NS::AutoreleasePool::alloc()->init();
    auto& shader_cache = context->kernel_cache;
    DeviceProperties dprops = DeviceProperties();
    auto pipeline_value = shader_cache.findKernel<NAConv3DKernel, NAConv3DDescriptor, NAConv3DKernelDescriptor>(conv3d_desc, context->device.get(), dprops);
    pool->drain();

    auto kernel = pipeline_value->kernel;
    auto multiply_pipeline = pipeline_value->pipeline;
    auto accumulate_pipeline = pipeline_value->second;
    auto permutation_pipeline = kernel->permutationPipeline;

    const uint32_t kernel_depth = conv3d_desc.kernelDimensions[0];
    const uint32_t kernel_height = conv3d_desc.kernelDimensions[1];
    const uint32_t kernel_width = conv3d_desc.kernelDimensions[2];
    const uint32_t input_height = params.input_dimensions[1];
    const uint32_t input_width = params.input_dimensions[2];

    const uint32_t weight_slice_element_count = kernel_height * kernel_width * conv3d_desc.inputChannels * conv3d_desc.outputChannels;
    const uint32_t weight_element_count = kernel_depth * weight_slice_element_count;
    const uint32_t input_slice_element_count = input_height * input_width * conv3d_desc.inputChannels;
    const uint32_t element_size = params.data_type == 3 ? 4 : 2;

    CCV_NNC_MFA_PRECONDITION(ccv_nnc_mfa_conv3d_reserved_scratch_size(params) == (size_t)weight_element_count * element_size);
    auto permuted_weights = context->request_scratch(ccv_nnc_mfa_conv3d_reserved_scratch_size(params));
    auto encoder = command_batch->startCommand();
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    if (params.fused_bias)
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(permuted_weights, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);

    encoder->setComputePipelineState(permutation_pipeline.get());
    encoder->setBuffer(tensors[1], tensor_offsets[1], 0);
    encoder->setBuffer(permuted_weights, 0, 1);
    uint32_t output_channels = conv3d_desc.outputChannels;
    uint32_t input_channels = conv3d_desc.inputChannels;
    encoder->setBytes(&output_channels, sizeof(output_channels), 2);
    encoder->setBytes(&input_channels, sizeof(input_channels), 3);
    encoder->setBytes(&kernel_depth, sizeof(kernel_depth), 4);
    encoder->setBytes(&kernel_height, sizeof(kernel_height), 5);
    encoder->setBytes(&kernel_width, sizeof(kernel_width), 6);
    const auto permutation_threadgroup_size = kernel->permutationThreadgroupSize(permutation_pipeline.get());
    const MTL::Size permutation_grid_size((weight_element_count + permutation_threadgroup_size - 1) / permutation_threadgroup_size, 1, 1);
    const MTL::Size permutation_group_size(permutation_threadgroup_size, 1, 1);
    encoder->dispatchThreadgroups(permutation_grid_size, permutation_group_size);

    const MTL::Size convolution_grid_size = kernel->threadgroupsPerGrid(conv3d_desc);
    const MTL::Size convolution_group_size(kernel->threadgroupSize(multiply_pipeline.get(), conv3d_desc), 1, 1);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
    for (uint32_t kernel_slice = 0; kernel_slice < kernel_depth; ++kernel_slice) {
      const uint32_t activation_base = kernel_slice * input_slice_element_count;
      const uint32_t weights_base = kernel_slice * weight_slice_element_count;
      encoder->setComputePipelineState(kernel_slice == 0 ? multiply_pipeline.get() : accumulate_pipeline.get());
      encoder->setBuffer(tensors[0], tensor_offsets[0] + (size_t)activation_base * element_size, 0);
      encoder->setBuffer(permuted_weights, (size_t)weights_base * element_size, 1);
      if (kernel_slice == 0 && params.fused_bias)
        encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
      encoder->dispatchThreadgroups(convolution_grid_size, convolution_group_size);
    }
    command_batch->finishCommand(encoder);
    return;
  }

  Conv3DDescriptor conv3d_desc;
  conv3d_desc.dataType = params.data_type;
  conv3d_desc.batchDimension = params.batch_size;
  conv3d_desc.inputChannels = params.input_channels;
  conv3d_desc.outputChannels = params.output_channels;
  const uint32_t max_block_n = 128u;
  const uint32_t rounded_output_channels =
      ((params.output_channels + 31u) / 32u) * 32u;
  const uint32_t block_n =
      std::min<uint32_t>(std::max<uint32_t>(rounded_output_channels, 32u), max_block_n);
  conv3d_desc.blockDimensions = simd::ushort2 {
      (unsigned short)block_n,
      32,
  };
  conv3d_desc.paddingLeft = padding_left;
  conv3d_desc.paddingRight = padding_right;
  conv3d_desc.paddingTop = padding_top;
  conv3d_desc.paddingBottom = padding_bottom;
  conv3d_desc.matrixDimensions = simd::uint3 { params.output_dimensions[0], params.output_dimensions[1], params.output_dimensions[2] };
  conv3d_desc.kernelDimensions = simd::uint3 { params.filter_dimensions[0], params.filter_dimensions[1], params.filter_dimensions[2] };
  conv3d_desc.useBias = params.fused_bias;
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shader_cache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipeline_value = shader_cache.findKernel<Conv3DKernel, Conv3DDescriptor, Conv3DKernelDescriptor>(conv3d_desc, context->device.get(), dprops);
  pool->drain();

  auto kernel = pipeline_value->kernel;
  auto conv_pipeline = pipeline_value->pipeline;
  auto permutation_pipeline = kernel->permutationPipeline;

  const uint32_t kernel_depth = conv3d_desc.kernelDimensions[0];
  const uint32_t kernel_height = conv3d_desc.kernelDimensions[1];
  const uint32_t kernel_width = conv3d_desc.kernelDimensions[2];

  const uint32_t weight_element_count = kernel_depth * kernel_height * kernel_width * conv3d_desc.inputChannels * conv3d_desc.outputChannels;
  const uint32_t element_size = params.data_type == 3 ? 4 : 2;

  CCV_NNC_MFA_PRECONDITION(ccv_nnc_mfa_conv3d_reserved_scratch_size(params) == (size_t)weight_element_count * element_size);
  auto permuted_weights = context->request_scratch(ccv_nnc_mfa_conv3d_reserved_scratch_size(params));
  auto encoder = command_batch->startCommand();
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  if (params.fused_bias)
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  encoder->useResource(permuted_weights, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);

  encoder->setComputePipelineState(permutation_pipeline.get());
  encoder->setBuffer(tensors[1], tensor_offsets[1], 0);
  encoder->setBuffer(permuted_weights, 0, 1);
  uint32_t output_channels = conv3d_desc.outputChannels;
  uint32_t input_channels = conv3d_desc.inputChannels;
  encoder->setBytes(&output_channels, sizeof(output_channels), 2);
  encoder->setBytes(&input_channels, sizeof(input_channels), 3);
  encoder->setBytes(&kernel_depth, sizeof(kernel_depth), 4);
  encoder->setBytes(&kernel_height, sizeof(kernel_height), 5);
  encoder->setBytes(&kernel_width, sizeof(kernel_width), 6);
  const auto permutation_threadgroup_size = kernel->permutationThreadgroupSize(permutation_pipeline.get());
  const MTL::Size permutation_grid_size((weight_element_count + permutation_threadgroup_size - 1) / permutation_threadgroup_size, 1, 1);
  const MTL::Size permutation_group_size(permutation_threadgroup_size, 1, 1);
  encoder->dispatchThreadgroups(permutation_grid_size, permutation_group_size);

  const MTL::Size convolution_grid_size = kernel->threadgroupsPerGrid(conv3d_desc);
  const MTL::Size convolution_group_size(kernel->threadgroupSize(conv_pipeline.get(), conv3d_desc), 1, 1);
  encoder->setComputePipelineState(conv_pipeline.get());
  encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
  encoder->setBuffer(permuted_weights, 0, 1);
  encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
  if (params.fused_bias)
    encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
  encoder->dispatchThreadgroups(convolution_grid_size, convolution_group_size);
  command_batch->finishCommand(encoder);
}
