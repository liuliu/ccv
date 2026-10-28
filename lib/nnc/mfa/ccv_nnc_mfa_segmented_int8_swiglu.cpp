#include "ccv_nnc_mfa.hpp"
#include "kernels/SegmentedInt8SwiGLUDescriptor.hpp"

using namespace ccv::nnc;

namespace {

static GEMMOperandPrecision ioPrecision(const uint64_t dataType) noexcept
{
  switch (dataType) {
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

void ccv_nnc_mfa_prepare_segmented_int8_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_segmented_int8_swiglu_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_segmented_int8_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_segmented_int8_swiglu_params_t params,
  mtl_command_batch_t* commandBatch,
  mtl_buffer_t** tensors,
  size_t* tensorOffsets)
{
  int tensorCount = 0;
  while (tensors[tensorCount] != nullptr)
    ++tensorCount;
  // Tensors: gate, up, activation, indices, counts, route weights, output, then with a shared
  // expert its gate, up, activation and output.
  CCV_NNC_MFA_PRECONDITION(tensorCount == (params.shared_rows > 0 ? 11 : 7));
  CCV_NNC_MFA_PRECONDITION(params.shared_rows == 0 || (params.K % 4) == 0);
  CCV_NNC_MFA_PRECONDITION(
    params.M > 0 && params.N > 0 && params.K > 0 &&
    params.expert_count > 0 && params.bincount > 0);
  CCV_NNC_MFA_PRECONDITION(params.K % 256 == 0 && params.N % 256 == 0);
  CCV_NNC_MFA_PRECONDITION(
    params.format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ||
    params.format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ||
    params.format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS ||
    params.format == CCV_NNC_QX_8I_ROWWISE_Q2_K ||
    params.format == CCV_NNC_QX_8I_ROWWISE_IQ1_S ||
    params.format == CCV_NNC_QX_8I_ROWWISE_IQ1_M);

  SegmentedInt8SwiGLUDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint2 { params.N, params.K };
  descriptor.expertCount = params.expert_count;
  descriptor.routeCount = params.bincount;
  descriptor.format = params.format;
  descriptor.broadcastInput = params.broadcast_input;
  descriptor.clamp = params.clamp;
  descriptor.memoryPrecision = ioPrecision(params.data_type);
  const NS::UInteger threadgroupsPerRoute = (params.N + 3) / 4;
  const NS::UInteger routedThreadgroups = threadgroupsPerRoute * params.M;
  descriptor.sharedRows = params.shared_rows;
  descriptor.routedThreadgroups = params.shared_rows > 0 ? (uint32_t)routedThreadgroups : 0;
  auto pipelineValue = context->kernel_cache.findKernel<
    SegmentedInt8SwiGLUKernel,
    SegmentedInt8SwiGLUDescriptor,
    SegmentedInt8SwiGLUKernelDescriptor>(
      descriptor, context->device.get(), DeviceProperties());
  auto pipeline = pipelineValue->pipeline;

  auto encoder = commandBatch->startCommand();
  encoder->setComputePipelineState(pipeline.get());
  encoder->setBuffer(tensors[0], tensorOffsets[0], 0);
  encoder->setBuffer(tensors[1], tensorOffsets[1], 1);
  encoder->setBuffer(tensors[2], tensorOffsets[2], 2);
  encoder->setBuffer(tensors[6], tensorOffsets[6], 3);
  encoder->setBuffer(
    tensors[0], tensorOffsets[0] + descriptor.inputScaleOffset(), 4);
  encoder->setBuffer(
    tensors[1], tensorOffsets[1] + descriptor.inputScaleOffset(), 5);
  encoder->setBuffer(tensors[3], tensorOffsets[3], 6);
  encoder->setBuffer(tensors[4], tensorOffsets[4], 7);
  encoder->setBuffer(tensors[5], tensorOffsets[5], 8);
  for (int i = 0; i < 6; ++i)
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
  NS::UInteger sharedThreadgroups = 0;
  if (params.shared_rows > 0) {
    const uint64_t sharedScaleOffset = ((uint64_t)params.shared_rows * params.K + 127) & ~UINT64_C(127);
    encoder->setBuffer(tensors[7], tensorOffsets[7], 9);
    encoder->setBuffer(tensors[8], tensorOffsets[8], 10);
    encoder->setBuffer(tensors[9], tensorOffsets[9], 11);
    encoder->setBuffer(tensors[10], tensorOffsets[10], 12);
    encoder->setBuffer(tensors[7], tensorOffsets[7] + sharedScaleOffset, 13);
    encoder->setBuffer(tensors[8], tensorOffsets[8] + sharedScaleOffset, 14);
    for (int i = 7; i < 10; ++i)
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    encoder->useResource(tensors[10], MTL::ResourceUsageWrite);
    sharedThreadgroups = (params.shared_rows + 1) / 2;
  }

  const MTL::Size gridSize(routedThreadgroups + sharedThreadgroups, 1, 1);
  const MTL::Size threadgroupSize(128, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadgroupSize);
  commandBatch->finishCommand(encoder);
}
