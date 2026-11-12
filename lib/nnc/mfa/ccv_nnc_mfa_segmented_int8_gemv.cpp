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
  CCV_NNC_MFA_PRECONDITION((params.K % 4) == 0);
  // TMCQ has its own N % 16 / K % 256 precondition below.
  CCV_NNC_MFA_PRECONDITION(
    params.format == 0 || segmentedTMCQRate(params.format) ||
    ((params.K % 256) == 0 && (params.N % 256) == 0));
  // TMCQ tiles span 16 rows and its kernel takes one activation row per route; it has no
  // modifier bits (no Hadamard rotation).
  CCV_NNC_MFA_PRECONDITION(
    !segmentedTMCQRate(params.format) || ((params.N % 16) == 0 && (params.K % 256) == 0));
  // TMCQ tiles are read as 32-bit words: the weight offset and the expert stride must keep that.
  CCV_NNC_MFA_PRECONDITION(
    !segmentedTMCQRate(params.format) || (tensor_offsets[3] % 4) == 0);
  CCV_NNC_MFA_PRECONDITION(
    (uint64_t)params.expert_count * params.N <= UINT32_MAX);

  SegmentedInt8GemvDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint2 { params.N, params.K };
  descriptor.expertCount = params.expert_count;
  descriptor.binCount = params.bincount;
  descriptor.format = params.format;
  descriptor.memoryPrecision = io_precision(params.data_type);
  descriptor.useBias = params.fused_bias;
  descriptor.broadcastInput = params.broadcast_input;
  const uint32_t simdgroups = segmentedInt8GemvSIMDGroupsPerThreadgroup(params.format);
  // Two rows per SIMD group, or for TMCQ one 16-row tile row per threadgroup.
  const uint32_t rowsPerThreadgroup = segmentedInt8GemvRowsPerThreadgroup(params.format);
  const NS::UInteger threadgroupsPerRoute =
    (params.N + rowsPerThreadgroup - 1) / rowsPerThreadgroup;
  // The shared rows take two per threadgroup, after the routed ones in the grid's count.
  CCV_NNC_MFA_PRECONDITION(params.shared_rows == 0 || (params.format == CCV_NNC_QX_8I_ROWWISE_Q2_K && !params.fused_bias));
  descriptor.sharedRows = params.shared_rows;
  descriptor.routedThreadgroups = params.shared_rows > 0 ? (uint32_t)(threadgroupsPerRoute * params.M) : 0;

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

  CCV_NNC_MFA_PRECONDITION(
    (uint64_t)threadgroupsPerRoute * params.M <= UINT32_MAX);
  if (params.shared_rows > 0) {
    // Shared weights, activation, output, then the weights' row scales after the padded payload.
    encoder->setBuffer(tensors[7], tensor_offsets[7], 7);
    encoder->setBuffer(tensors[8], tensor_offsets[8], 8);
    encoder->setBuffer(tensors[9], tensor_offsets[9], 9);
    encoder->setBuffer(tensors[7], tensor_offsets[7] + (((uint64_t)params.shared_rows * params.K + 127) & ~UINT64_C(127)), 10);
    encoder->useResource(tensors[7], MTL::ResourceUsageRead);
    encoder->useResource(tensors[8], MTL::ResourceUsageRead);
    encoder->useResource(tensors[9], MTL::ResourceUsageWrite);
  }
  const MTL::Size gridSize(threadgroupsPerRoute * params.M + (params.shared_rows + 1) / 2, 1, 1);
  const MTL::Size threadgroupSize(simdgroups * 32, 1, 1);
  encoder->dispatchThreadgroups(gridSize, threadgroupSize);
  command_batch->finishCommand(encoder);
}
