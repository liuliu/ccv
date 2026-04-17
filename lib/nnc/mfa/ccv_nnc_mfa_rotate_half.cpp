#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "kernels/RotateHalfDescriptor.hpp"
#include "kernels/RotateHalfKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_rotate_half(mfa::context* context, ccv_nnc_mfa_rotate_half_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_rotate_half(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rotate_half_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 2);

  RotateHalfDescriptor descriptor;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.rowCount = params.row_count;
  descriptor.dim = params.dim;
  descriptor.value = (params.dim % 8 == 0) ? 0 : 1;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<RotateHalfKernel, RotateHalfDescriptor, RotateHalfKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  if (tensors[0] == tensors[1]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  }

  const int vectorized = (descriptor.value == 0);
  const uint32_t dim_units = vectorized ? (params.dim / 4) : params.dim;
  const uint32_t count = params.row_count * dim_units;
  const int num_blocks = (count + 255) / 256;
  MTL::Size gridSize = MTL::Size(num_blocks, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
