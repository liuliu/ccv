#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "kernels/RMSNormGatedDescriptor.hpp"
#include "kernels/RMSNormGatedKernel.hpp"
using namespace ccv::nnc;

namespace {

GEMMOperandPrecision precision_from_mtl_data_type(const uint64_t data_type)
{
  if (data_type == MTL::DataTypeFloat) {
    return GEMMOperandPrecision::FP32;
  } else if (data_type == MTL::DataTypeBFloat) {
    return GEMMOperandPrecision::BF16;
  }
  return GEMMOperandPrecision::FP16;
}

}

void ccv_nnc_mfa_prepare_rmsnorm_gated(mfa::context* context, ccv_nnc_mfa_rmsnorm_gated_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_rmsnorm_gated(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_gated_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 4);

  RMSNormGatedDescriptor descriptor;
  descriptor.epsilon = params.epsilon;
  descriptor.aPrecision = precision_from_mtl_data_type(params.a_data_type);
  descriptor.gatePrecision = precision_from_mtl_data_type(params.gate_data_type);
  descriptor.scalePrecision = precision_from_mtl_data_type(params.scale_data_type);
  descriptor.columnCount = params.column_count;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<RMSNormGatedKernel, RMSNormGatedDescriptor, RMSNormGatedKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());

  if (tensors[0] == tensors[3]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  }
  if (tensors[1] == tensors[3]) {
    encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  }
  if (tensors[2] == tensors[3]) {
    encoder->useResource(tensors[2], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  }
  if (tensors[0] != tensors[3] && tensors[1] != tensors[3] && tensors[2] != tensors[3]) {
    encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
  }

  MTL::Size gridSize = MTL::Size(params.row_count, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->groupSize);

  command_batch->finishCommand(encoder);
}
