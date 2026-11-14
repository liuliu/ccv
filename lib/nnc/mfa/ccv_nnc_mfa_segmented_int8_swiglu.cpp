#include "ccv_nnc_mfa.hpp"
#include "kernels/SegmentedInt8SwiGLUDescriptor.hpp"
#include "kernels/SegmentedInt8GemvDescriptor.hpp"

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
  CCV_NNC_MFA_PRECONDITION(params.shared_format == 0 || params.shared_format == CCV_NNC_QX_8I_ROWWISE_Q5_K || params.shared_format == CCV_NNC_QX_8I_ROWWISE_Q6_K);
  CCV_NNC_MFA_PRECONDITION(params.shared_format == 0 || (params.shared_rows % 256) == 0);
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
    params.format == CCV_NNC_QX_8I_ROWWISE_IQ1_M ||
    segmentedTMCQRate(params.format) != 0);

  // TMCQ tiles are read as 32-bit words.
  CCV_NNC_MFA_PRECONDITION(!segmentedTMCQRate(params.format) ||
    ((tensorOffsets[0] % 4) == 0 && (tensorOffsets[1] % 4) == 0));

  SegmentedInt8SwiGLUDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint2 { params.N, params.K };
  descriptor.expertCount = params.expert_count;
  descriptor.routeCount = params.bincount;
  descriptor.format = params.format;
  descriptor.broadcastInput = params.broadcast_input;
  descriptor.clamp = params.clamp;
  descriptor.memoryPrecision = ioPrecision(params.data_type);
  // Four rows per threadgroup (one per SIMD group), or for TMCQ one 16-row tile row whose SIMD groups
  // split K.
  const NS::UInteger rowsPerThreadgroup = segmentedTMCQRate(params.format) ? 16 : 4;
  const NS::UInteger threadgroupsPerRoute = (params.N + rowsPerThreadgroup - 1) / rowsPerThreadgroup;
  const NS::UInteger routedThreadgroups = threadgroupsPerRoute * params.M;
  descriptor.sharedRows = params.shared_rows;
  descriptor.routedThreadgroups = params.shared_rows > 0 ? (uint32_t)routedThreadgroups : 0;
  descriptor.sharedFormat = params.shared_rows > 0 ? params.shared_format : 0;
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
    // Q8_0 rows are K bytes; Q5_K packs 16 weights in 11 bytes, Q6_K in 13.
    const uint64_t sharedPayloadBytes = params.shared_format == 0 ? (uint64_t)params.shared_rows * params.K :
      (uint64_t)params.shared_rows * params.K / 16 * (params.shared_format == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 11 : 13);
    const uint64_t sharedScaleOffset = (sharedPayloadBytes + 127) & ~UINT64_C(127);
    encoder->setBuffer(tensors[7], tensorOffsets[7], 9);
    encoder->setBuffer(tensors[8], tensorOffsets[8], 10);
    encoder->setBuffer(tensors[9], tensorOffsets[9], 11);
    encoder->setBuffer(tensors[10], tensorOffsets[10], 12);
    encoder->setBuffer(tensors[7], tensorOffsets[7] + sharedScaleOffset, 13);
    encoder->setBuffer(tensors[8], tensorOffsets[8] + sharedScaleOffset, 14);
    for (int i = 7; i < 10; ++i)
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    encoder->useResource(tensors[10], MTL::ResourceUsageWrite);
    // Two Q8_0 rows per threadgroup, or a packed row per half SIMD group.
    const NS::UInteger sharedRowsPerThreadgroup = params.shared_format == 0 ? 2 : 8;
    sharedThreadgroups = (params.shared_rows + sharedRowsPerThreadgroup - 1) / sharedRowsPerThreadgroup;
  }

  const MTL::Size gridSize(routedThreadgroups + sharedThreadgroups, 1, 1);
  // 128 threads, or for TMCQ one SIMD group per K split.
  const NS::UInteger threadgroupThreads = segmentedTMCQRate(params.format) ?
    32 * segmentedInt8SwiGLUTMCQSIMDGroups(params.shared_rows > 0) : 128;
  const MTL::Size threadgroupSize(threadgroupThreads, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadgroupSize);
  commandBatch->finishCommand(encoder);
}
