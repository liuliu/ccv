#include "ccv_nnc_mfa.hpp"
#include "kernels/GemvDescriptor.hpp"
#include "kernels/GemvKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_gemv(mfa::context* context, ccv_nnc_mfa_gemv_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3 || num_tensors == 4);
  CCV_NNC_MFA_PRECONDITION((params.fused_bias && num_tensors == 4) || (!params.fused_bias && num_tensors == 3));
  CCV_NNC_MFA_PRECONDITION(params.mrows == 1 || params.mrows == 2 || params.mrows == 3);

  GemvDescriptor descriptor;
  descriptor.fusedBias = params.fused_bias ? 1 : 0;
  descriptor.mrows = (uint8_t)params.mrows;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else if (params.data_type == MTL::DataTypeBFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.nrows = params.nrows;
  descriptor.ncols = params.ncols;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<GemvKernel, GemvDescriptor, GemvKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  if (num_tensors == 4) {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  }

  const uint32_t rowsPerThreadgroup = GemvDescriptor::rowsPerThreadgroup(context->device.get());
  MTL::Size gridSize = MTL::Size((params.nrows + rowsPerThreadgroup - 1) / rowsPerThreadgroup, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, MTL::Size(rowsPerThreadgroup * 32, 1, 1));
  command_batch->finishCommand(encoder);
}
