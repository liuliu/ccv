#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "kernels/CMulDescriptor.hpp"
#include "kernels/CMulKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

static GEMMOperandPrecision _ccv_nnc_mfa_cmul_precision(const uint64_t data_type)
{
  if (data_type == MTL::DataTypeFloat) {
    return GEMMOperandPrecision::FP32;
  } else if (data_type == MTL::DataTypeBFloat) {
    return GEMMOperandPrecision::BF16;
  }
  return GEMMOperandPrecision::FP16;
}

void ccv_nnc_mfa_prepare_cmul(mfa::context* context, ccv_nnc_mfa_cmul_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cmul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3);

  CMulDescriptor descriptor;
  descriptor.conjugate = params.conjugate ? 1 : 0;
  descriptor.memoryPrecisionA = _ccv_nnc_mfa_cmul_precision(params.data_type_a);
  descriptor.memoryPrecisionB = _ccv_nnc_mfa_cmul_precision(params.data_type_b);
  descriptor.memoryPrecisionC = _ccv_nnc_mfa_cmul_precision(params.data_type_c);
  descriptor.stridesA[0] = params.astride[0];
  descriptor.stridesA[1] = params.astride[1];
  descriptor.stridesA[2] = params.astride[2];
  descriptor.stridesB[0] = params.bstride[0];
  descriptor.stridesB[1] = params.bstride[1];
  descriptor.stridesB[2] = params.bstride[2];
  descriptor.stridesC[0] = params.cstride[0];
  descriptor.stridesC[1] = params.cstride[1];
  descriptor.stridesC[2] = params.cstride[2];

  descriptor.dimensions[0] = params.dim[0];
  descriptor.dimensions[1] = params.dim[1];
  descriptor.dimensions[2] = params.dim[2];

  if (params.dim[3] == 0 && params.dim[2] == 0 && params.dim[1] == 0) {
    descriptor.value = 0;
  } else if (params.dim[3] == 0 && params.dim[2] == 0) {
    descriptor.value = 1;
  } else if (params.dim[3] == 0) {
    descriptor.value = 2;
  } else {
    descriptor.value = 3;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<CMulKernel, CMulDescriptor, CMulKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());

  if (tensors[0] == tensors[2]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  } else if (tensors[1] == tensors[2]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  }

  MTL::Size gridSize;
  if (params.dim[3] == 0 && params.dim[2] == 0 && params.dim[1] == 0) {
    const int num_blocks = (params.dim[0] / 2 + 255) / 256;
    gridSize = MTL::Size(num_blocks, 1, 1);
  } else if (params.dim[3] == 0 && params.dim[2] == 0) {
    gridSize = MTL::Size((params.dim[0] / 2 + 31) / 32, (params.dim[1] + 7) / 8, 1);
  } else if (params.dim[3] == 0) {
    gridSize = MTL::Size((params.dim[0] / 2 + 31) / 32, (params.dim[1] + 7) / 8, params.dim[2]);
  } else {
    gridSize = MTL::Size((params.dim[0] / 2 + 31) / 32, (params.dim[1] + 7) / 8, params.dim[2] * params.dim[3]);
  }
  CCV_NNC_MFA_PRECONDITION(gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

  command_batch->finishCommand(encoder);
}
