#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "kernels/SwishMulDescriptor.hpp"
#include "kernels/SwishMulKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

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

// MARK: - C

void ccv_nnc_mfa_prepare_swish_mul(mfa::context* context, ccv_nnc_mfa_swish_mul_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_swish_mul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_swish_mul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  if (params.gradient) {
    CCV_NNC_MFA_PRECONDITION(params.output_mask >= 1 && params.output_mask <= 3);
    const int expected_tensors = params.output_mask == 1 ? 3 : params.output_mask == 2 ? 4 : 5;
    CCV_NNC_MFA_PRECONDITION(num_tensors == expected_tensors);
  } else {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 3);
  }

  SwishMulDescriptor descriptor;
  descriptor.gradient = params.gradient ? 1 : 0;
  descriptor.outputMask = params.output_mask;
  descriptor.beta = params.beta;
  descriptor.scale = params.scale;
  descriptor.gPrecision = precision_from_mtl_data_type(params.g_data_type);
  descriptor.aPrecision = precision_from_mtl_data_type(params.a_data_type);
  descriptor.bPrecision = precision_from_mtl_data_type(params.b_data_type);
  descriptor.daPrecision = precision_from_mtl_data_type(params.da_data_type);
  descriptor.dbPrecision = precision_from_mtl_data_type(params.db_data_type);
  descriptor.length = params.length;

  if (params.length % (4 * 256) == 0) {
    descriptor.value = 0;
  } else if (params.length % 4 == 0) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<SwishMulKernel, SwishMulDescriptor, SwishMulKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());

  if (params.gradient) {
    int read_indices[3];
    int read_count;
    int write_indices[2];
    int write_count;
    if (params.output_mask == 1) {
      read_indices[0] = 0;
      read_indices[1] = 1;
      read_count = 2;
      write_indices[0] = 2;
      write_count = 1;
    } else if (params.output_mask == 2) {
      read_indices[0] = 0;
      read_indices[1] = 1;
      read_indices[2] = 2;
      read_count = 3;
      write_indices[0] = 3;
      write_count = 1;
    } else {
      read_indices[0] = 0;
      read_indices[1] = 1;
      read_indices[2] = 2;
      read_count = 3;
      write_indices[0] = 3;
      write_indices[1] = 4;
      write_count = 2;
    }
    for (int i = 0; i < read_count; i++) {
      bool written = false;
      for (int j = 0; j < write_count; j++)
        written = written || (tensors[read_indices[i]] == tensors[write_indices[j]]);
      encoder->useResource(tensors[read_indices[i]], written ? MTL::ResourceUsageRead | MTL::ResourceUsageWrite : MTL::ResourceUsageRead);
    }
    for (int i = 0; i < write_count; i++) {
      bool read = false;
      for (int j = 0; j < read_count; j++)
        read = read || (tensors[write_indices[i]] == tensors[read_indices[j]]);
      if (!read)
        encoder->useResource(tensors[write_indices[i]], MTL::ResourceUsageWrite);
    }
  } else if (tensors[0] == tensors[2]) {
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

  unsigned int count;
  if (params.length % 4 == 0) {
    count = params.length / 4;
  } else {
    count = params.length;
  }
  const int num_blocks = (count + 255) / 256;
  MTL::Size gridSize = MTL::Size(num_blocks, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

  command_batch->finishCommand(encoder);
}
