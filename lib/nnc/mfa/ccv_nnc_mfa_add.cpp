#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "kernels/AddDescriptor.hpp"
#include "kernels/AddKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_add(mfa::context* context, ccv_nnc_mfa_add_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_add_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(!(params.negative_mask | params.broadcast | params.scaled_mask) || params.args <= 8);
  CCV_NNC_MFA_PRECONDITION(!params.channel_broadcast ||
    ((params.channel_broadcast == 1 || params.channel_broadcast == 2) && params.args == 2 && !params.broadcast && params.channel_count > 0 && params.channel_length > 0));
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 1 + params.args);

  // Larger 16-bit unary / binary workloads benefit from wider threadgroups.
  const unsigned int threadgroup_width = params.args <= 2 && params.data_type != MTL::DataTypeFloat && params.length >= 262144 ? 512 : 256;

  AddDescriptor descriptor;
  descriptor.args = params.args;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else if (params.data_type == MTL::DataTypeBFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.length = params.length;
  descriptor.loadM = params.loadM;
  descriptor.channel_broadcast = params.channel_broadcast;
  descriptor.channel_count = params.channel_broadcast ? params.channel_count : 0;
  descriptor.channel_length = params.channel_broadcast ? params.channel_length : 0;
  descriptor.negative_mask = params.negative_mask;
  descriptor.broadcast = params.broadcast;
  descriptor.scaled_mask = params.scaled_mask;

  const bool vectorized = params.length % 4 == 0 && (!params.channel_broadcast || params.channel_length % 4 == 0);
  if (vectorized && !params.loadM && params.length % (4 * threadgroup_width) == 0) {
    descriptor.value = 0;
  } else if (vectorized) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<AddKernel, AddDescriptor, AddKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  if (params.scaled_mask) {
    if (params.data_type == MTL::DataTypeFloat) {
      encoder->setBytes(params.scales, sizeof(float) * params.args, NS::UInteger(params.args + 1));
    } else {
      uint16_t scales[8];
      if (params.data_type == MTL::DataTypeBFloat)
        ccv_float_to_bfloat(params.scales, scales, params.args);
      else
        ccv_float_to_half_precision(params.scales, scales, params.args);
      encoder->setBytes(scales, sizeof(uint16_t) * params.args, NS::UInteger(params.args + 1));
    }
  }
  
  int i;
  int flag = 0;
  for (i = 0; i < params.args; i++) {
    if (tensors[i] == tensors[params.args]) {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      flag = 1;
	} else {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
	}
  }
  if (!flag) {
    encoder->useResource(tensors[params.args], MTL::ResourceUsageWrite);
  }

  unsigned int count;
  if (vectorized) {
    count = params.length / 4;
  } else {
    count = params.length;
  }
  if (params.loadM)
    encoder->setBytes(&count, sizeof(count), NS::UInteger(num_tensors + (params.scaled_mask ? 1 : 0)));
  const size_t num_blocks = ((size_t)count + threadgroup_width - 1) / threadgroup_width;
  MTL::Size gridSize = MTL::Size(num_blocks, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, MTL::Size(threadgroup_width, 1, 1));
  command_batch->finishCommand(encoder);
}
