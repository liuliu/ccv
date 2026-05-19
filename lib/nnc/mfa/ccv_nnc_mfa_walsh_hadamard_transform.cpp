#include "ccv_nnc_mfa.hpp"
#include "kernels/WalshHadamardTransformDescriptor.hpp"
#include "kernels/WalshHadamardTransformKernel.hpp"
using namespace ccv::nnc;

#include <algorithm>

// MARK: - C

static inline uint32_t _ccv_nnc_mfa_wht_rows_per_threadgroup(MTL::Device* const device, const uint32_t strategy, const uint32_t dim)
{
  if (strategy != 2)
    return 1;
  const uint32_t maxThreads = (uint32_t)device->maxThreadsPerThreadgroup().width;
  return std::max<uint32_t>(1, std::min<uint32_t>(8, maxThreads / dim));
}

void ccv_nnc_mfa_prepare_walsh_hadamard_transform(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_walsh_hadamard_transform_params_t)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_walsh_hadamard_transform(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_walsh_hadamard_transform_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 2);

  WalshHadamardTransformDescriptor descriptor;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else if (params.data_type == MTL::DataTypeBFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.rowCount = params.row_count;
  descriptor.dim = params.dim;
  descriptor.scale = params.scale;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<WalshHadamardTransformKernel, WalshHadamardTransformDescriptor, WalshHadamardTransformKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  if (tensors[0] == tensors[1]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  }

  const uint32_t strategy = (params.dim >= 32 && params.dim <= 128) ? 2 : ((params.dim <= 128) ? 1 : 0);
  const uint32_t maxRadix = (params.dim > 1) ? std::min<uint32_t>(params.dim, 16) : 1;
  const uint32_t rowsPerThreadgroup = _ccv_nnc_mfa_wht_rows_per_threadgroup(context->device.get(), strategy, params.dim);
  const uint32_t numThreads = (strategy == 2) ? params.dim * rowsPerThreadgroup : ((strategy == 1) ? params.dim : std::max<uint32_t>(params.dim / maxRadix, 1));
  CCV_NNC_MFA_PRECONDITION(numThreads <= pipeline->maxTotalThreadsPerThreadgroup());
  encoder->setThreadgroupMemoryLength(NS::UInteger(params.dim * rowsPerThreadgroup * sizeof(float)), NS::UInteger(0));
  MTL::Size gridSize = MTL::Size((params.row_count + rowsPerThreadgroup - 1) / rowsPerThreadgroup, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, MTL::Size(numThreads, 1, 1));
  command_batch->finishCommand(encoder);
}
