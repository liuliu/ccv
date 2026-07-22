#include "ccv_nnc_mfa.hpp"
#include "kernels/ConformDataFormatDescriptor.hpp"
#include "kernels/ConformDataFormatKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_conform_data_format(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_conform_data_format_params_t)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_conform_data_format(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_conform_data_format_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();
  int numTensors = 0;
  while (tensors[numTensors] != nullptr) {
    encoder->setBuffer(tensors[numTensors], tensor_offsets[numTensors], NS::UInteger(numTensors));
    numTensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(numTensors == 2);

  ConformDataFormatDescriptor descriptor;
  descriptor.rowCount = params.row_count;
  descriptor.headDim = params.head_dim;
  descriptor.preservedTail = params.preserved_tail;
  descriptor.loadM = params.loadM;
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<ConformDataFormatKernel, ConformDataFormatDescriptor, ConformDataFormatKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  if (params.loadM)
    encoder->setBytes(&params.row_count, sizeof(params.row_count), numTensors);
  if (tensors[0] == tensors[1]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  }

  const MTL::Size gridSize = kernel->gridSize(params.row_count, params.head_dim, params.preserved_tail);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  CCV_NNC_MFA_PRECONDITION(kernel->threadgroupSize.width <= pipeline->maxTotalThreadsPerThreadgroup());
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
