#include "ccv_nnc_mfa.hpp"
#include "kernels/SegmentedInt8GemvDescriptor.hpp"
#include "kernels/SegmentedInt8GemvKernel.hpp"
using namespace ccv::nnc;

namespace {

static GEMMOperandPrecision io_precision(const uint64_t data_type) noexcept
{
  switch (data_type) {
    case MTL::DataTypeHalf:
      return GEMMOperandPrecision::FP16;
    case MTL::DataTypeBFloat:
      return GEMMOperandPrecision::BF16;
    case MTL::DataTypeFloat:
      return GEMMOperandPrecision::FP32;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      return GEMMOperandPrecision::FP16;
  }
}

}

void ccv_nnc_mfa_prepare_segmented_int8_gemv(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_segmented_int8_gemv_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_segmented_int8_gemv(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_segmented_int8_gemv_params_t params,
  mtl_command_batch_t* command_batch,
  mtl_buffer_t** tensors,
  size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    num_tensors += 1;
  CCV_NNC_MFA_PRECONDITION(num_tensors == 5 || num_tensors == 6);
  CCV_NNC_MFA_PRECONDITION(
    (params.fused_bias && num_tensors == 6) ||
    (!params.fused_bias && num_tensors == 5));
  CCV_NNC_MFA_PRECONDITION(params.M > 0 && params.N > 0 && params.K > 0);
  CCV_NNC_MFA_PRECONDITION(params.expert_count > 0 && params.bincount > 0);
  CCV_NNC_MFA_PRECONDITION(params.M == params.bincount);
  CCV_NNC_MFA_PRECONDITION((params.K % 4) == 0);
  CCV_NNC_MFA_PRECONDITION(
    params.format == 0 ||
    ((params.K % 256) == 0 && (params.N % 256) == 0));
  CCV_NNC_MFA_PRECONDITION(
    (uint64_t)params.expert_count * params.N <= UINT32_MAX);

  SegmentedInt8GemvDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint2 { params.N, params.K };
  descriptor.expertCount = params.expert_count;
  descriptor.binCount = params.bincount;
  descriptor.format = params.format;
  descriptor.memoryPrecision = io_precision(params.data_type);
  descriptor.useBias = params.fused_bias;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto pipelineValue = context->kernel_cache.findKernel<
    SegmentedInt8GemvKernel,
    SegmentedInt8GemvDescriptor,
    SegmentedInt8GemvKernelDescriptor>(
      descriptor, context->device.get(), DeviceProperties());
  auto pipeline = pipelineValue->pipeline;
  pool->drain();

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipeline.get());
  encoder->setBuffer(tensors[3], tensor_offsets[3], 0);
  encoder->setBuffer(tensors[0], tensor_offsets[0], 1);
  encoder->setBuffer(tensors[4], tensor_offsets[4], 2);
  encoder->setBuffer(
    tensors[3],
    tensor_offsets[3] + descriptor.inputScaleOffset(),
    3);
  if (params.fused_bias)
    encoder->setBuffer(tensors[5], tensor_offsets[5], 4);
  encoder->setBuffer(tensors[1], tensor_offsets[1], 5);
  encoder->setBuffer(tensors[2], tensor_offsets[2], 6);

  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
  if (params.fused_bias)
    encoder->useResource(tensors[5], MTL::ResourceUsageRead);

  const NS::UInteger threadgroupsPerRoute =
    (params.N + kSegmentedInt8GemvRowsPerThreadgroup - 1) /
      kSegmentedInt8GemvRowsPerThreadgroup;
  CCV_NNC_MFA_PRECONDITION(
    (uint64_t)threadgroupsPerRoute * params.M <= UINT32_MAX);
  const MTL::Size gridSize(threadgroupsPerRoute * params.M, 1, 1);
  const MTL::Size threadgroupSize(
    kSegmentedInt8GemvSIMDGroupsPerThreadgroup * 32, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadgroupSize);
  command_batch->finishCommand(encoder);
}
