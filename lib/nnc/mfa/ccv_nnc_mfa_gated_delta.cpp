#include "ccv_nnc_mfa.hpp"
#include "kernels/GatedDeltaDescriptor.hpp"
#include "kernels/GatedDeltaKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_gated_delta(mfa::context* context, ccv_nnc_mfa_gated_delta_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_gated_delta(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gated_delta_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 8);

  GatedDeltaDescriptor descriptor;
  descriptor.batchSize = params.batch_size;
  descriptor.sequenceLength = params.sequence_length;
  descriptor.keyHeadCount = params.key_head_count;
  descriptor.valueHeadCount = params.value_head_count;
  descriptor.keyDim = params.key_dim;
  descriptor.valueDim = params.value_dim;
  descriptor.stateCheckpointCount = params.state_checkpoint_count;
  CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat || params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeBFloat);
  CCV_NNC_MFA_PRECONDITION(params.beta_data_type == MTL::DataTypeFloat || params.beta_data_type == MTL::DataTypeHalf || params.beta_data_type == MTL::DataTypeBFloat);
  CCV_NNC_MFA_PRECONDITION(params.state_checkpoint_count < params.sequence_length);
  descriptor.inputMemoryPrecision = params.data_type == MTL::DataTypeHalf ? GEMMOperandPrecision::FP16 : (params.data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32);
  descriptor.betaMemoryPrecision = params.beta_data_type == MTL::DataTypeHalf ? GEMMOperandPrecision::FP16 : (params.beta_data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32);
  descriptor.logDecay = params.log_decay != 0;
  descriptor.loadM = params.loadM != 0;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<GatedDeltaKernel, GatedDeltaDescriptor, GatedDeltaKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  if (params.loadM) {
    encoder->setBytes(&params.sequence_length, sizeof(params.sequence_length), 8);
  }
  encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
  int i;
  for (i = 0; i < 6; i++) {
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  }
  encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
  if (tensors[5] == tensors[7]) {
    encoder->useResource(tensors[7], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[7], MTL::ResourceUsageWrite);
  }

  const uint32_t value_dim_threadgroups = (params.value_dim + 3) / 4;
  MTL::Size gridSize = MTL::Size(1, value_dim_threadgroups, params.batch_size * params.value_head_count);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
