#include "ccv_nnc_mfa.hpp"
#include "kernels/FillIfLessThanDescriptor.hpp"
#include "kernels/FillIfLessThanKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_fill_if_less_than(mfa::context* context, ccv_nnc_mfa_fill_if_less_than_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_fill_if_less_than(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_fill_if_less_than_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 4);

  FillIfLessThanDescriptor descriptor;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else if (params.data_type == MTL::DataTypeBFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.length = params.length;
  descriptor.loadM = params.loadM;
  if (!params.loadM && params.length % (4 * 256) == 0) {
    descriptor.value = 0;
  } else if (params.length % 4 == 0) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<FillIfLessThanKernel, FillIfLessThanDescriptor, FillIfLessThanKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  encoder->setBytes(&params.fill, sizeof(params.fill), NS::UInteger(4));

  int output_alias = 0;
  for (int i = 0; i < 3; i++) {
    if (tensors[i] == tensors[3]) {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      output_alias = 1;
    } else {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    }
  }
  if (!output_alias) {
    encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
  }

  const uint32_t count = params.length % 4 == 0 ? params.length / 4 : params.length;
  if (params.loadM) {
    encoder->setBytes(&count, sizeof(count), NS::UInteger(5));
  }
  const uint32_t num_blocks = (count + 255) / 256;
  MTL::Size gridSize = MTL::Size(num_blocks, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

  command_batch->finishCommand(encoder);
}
