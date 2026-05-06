#include "ccv_nnc_mfa.hpp"
#include "kernels/NormalizationDescriptor.hpp"
#include "kernels/NormalizationKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_normalization(mfa::context* context, ccv_nnc_mfa_normalization_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_normalization(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_normalization_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 6 || num_tensors == 4 || num_tensors == 3);

  // Simple broadcasting rules; not yet support for NumPy broadcasting rules.
  simd::ushort2 num_batch_dims(0);
  simd::ulong2 batch_sizes(1);
  for (uint16_t operand = 0; operand < 2; ++operand) {
    uint32_t* batch_dims;
    if (operand == 0) {
      batch_dims = params.batch_dims_data;
    } else if (operand == 1) {
      if (params.scale_translation_batched) {
        batch_dims = params.batch_dims_scale_translation;
      } else {
        continue;
      }
    }

    for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
      if (batch_dims[i] == 0) {
        break;
      }
      num_batch_dims[operand] += 1;
      batch_sizes[operand] *= batch_dims[i];
    }

    bool dims_match_data = true;
    if (num_batch_dims[0] != num_batch_dims[operand]) {
      dims_match_data = false;
    } else if (batch_sizes[0] != batch_sizes[operand]) {
      dims_match_data = false;
    } else {
      for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
        if (params.batch_dims_data[i] != batch_dims[i]) {
          dims_match_data = false;
        }
      }
    }

    if (!dims_match_data) {
      CCV_NNC_MFA_PRECONDITION(batch_sizes[operand] == 1);
    }
  }

  if (params.scale_translation_batched) {
    uint16_t data_type_size = 0;
    switch (params.data_type) {
      case MTL::DataTypeHalf: {
        data_type_size = 2;
        break;
      }
      case MTL::DataTypeBFloat: {
        data_type_size = 2;
        break;
      }
      case MTL::DataTypeFloat: {
        data_type_size = 4;
        break;
      }
      default:
        CCV_NNC_MFA_PRECONDITION(false);
        break;
    }

    uint64_t byte_stride_scale_translation = 0;
    if (batch_sizes[1] > 1) {
      byte_stride_scale_translation = params.channel_count * data_type_size;
    }

    simd::ulong4 scale_translation_offsets[batch_sizes[0]];
    for (int i = 0; i < batch_sizes[0]; ++i) {
      scale_translation_offsets[i] = simd::ulong4 {
        i * byte_stride_scale_translation,
        i * byte_stride_scale_translation,
        0,
        0,
      };
    }
    encoder->setBytes(scale_translation_offsets, batch_sizes[0] * 32, 10);
  }

  NormalizationDescriptor descriptor = {
    .dataType = params.data_type,
    .channelCount = params.channel_count,
    .channelGroups = params.channel_groups,
    .sequenceCount = params.sequence_count,
    .epsilon = params.epsilon,
    .scale = params.scale,
    .elementwiseAffine = params.elementwise_affine,
    .scaleTranslationBatched = params.scale_translation_batched,
    .normalizationType = params.normalization_type,
    .reuseSavedStatistics = params.reuse_saved_statistics,
    .srcBatchStride = params.src_batch_stride,
    .dstBatchStride = params.dst_batch_stride,
  };

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<NormalizationKernel, NormalizationDescriptor, NormalizationKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  if (num_tensors == 6) { // This is for layer norm.
    if (params.reuse_saved_statistics) {
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    } else {
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
    }
    encoder->useResource(tensors[4], MTL::ResourceUsageRead);
    encoder->useResource(tensors[5], MTL::ResourceUsageRead);
  } else if (num_tensors == 4) {
    if (params.elementwise_affine) { // This is for RMSNorm.
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      if (params.reuse_saved_statistics) {
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      } else {
        encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      }
    } else { // This is for layer norm without elementwise affine.
      if (params.reuse_saved_statistics) {
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
        encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      } else {
        encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
        encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      }
    }
  } else { // This is for RMSNorm.
    if (params.reuse_saved_statistics) {
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    } else {
      encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    }
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  }

  auto grid_size = kernel->gridSize;
  grid_size.depth = batch_sizes[0];
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, kernel->groupSize);
  command_batch->finishCommand(encoder);
}
