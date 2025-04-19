#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include "v2/ShaderCache.hpp"
#include "v2/SegmentedGEMMPrologueKernel.hpp"
#include "v2/SegmentedGEMMPrologueKernelDescriptor.hpp"
#include "v2/SegmentedGEMMPrologueDescriptor.hpp"
#include "v2/GEMMKernel.hpp"
#include "v2/GEMMKernelDescriptor.hpp"
#include "v2/GEMMDescriptor.hpp"
#include <string>

// MARK: - C

void ccv_nnc_mfa_encode_segmented_gemm(mfa::context* context, ccv_nnc_mfa_segmented_gemm_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION((num_tensors == 5) || (num_tensors == 6))
  SegmentedGEMMPrologueDescriptor prologueDesc;
  prologueDesc.matrixDimensions = simd::uint3 {
    params.segments,
    params.N,
    params.K,
  };
  switch (params.data_type) {
    case MTL::DataTypeHalf: {
      prologueDesc.memoryPrecisions = {
        .A = GEMMOperandPrecision::FP16,
        .B = GEMMOperandPrecision::FP16,
        .C = GEMMOperandPrecision::FP16,
        .bias = GEMMOperandPrecision::FP16,
      };
      break;
    }
    case MTL::DataTypeFloat: {
      prologueDesc.memoryPrecisions = {
        .A = GEMMOperandPrecision::FP32,
        .B = GEMMOperandPrecision::FP32,
        .C = GEMMOperandPrecision::FP32,
        .bias = GEMMOperandPrecision::FP32,
      };
      break;
    }
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
  }
  prologueDesc.useBias = params.fused_bias;

  // Branch on whether to use the new kernel.
  GEMMDescriptor gemmDesc;
  gemmDesc.matrixDimensions = simd::uint3 {
    params.M,
    params.N,
    params.K,
  };
  gemmDesc.memoryPrecisions = prologueDesc.memoryPrecisions;
  gemmDesc.transposeState = simd::uchar3 { params.A_trans, params.B_trans, params.D_trans };
  gemmDesc.registerPrecisionC = (params.register_float) ? std::optional(GEMMOperandPrecision::FP32) : std::nullopt;
  gemmDesc.leadingDimensions = std::nullopt;
  gemmDesc.loadPreviousC = false;
  gemmDesc.useBias = params.fused_bias;
  gemmDesc.loadM = true;
  gemmDesc.supportIndirectCommandBuffers = true;

  gemmDesc.batchDimension = 1;
  gemmDesc.batchStrides = std::nullopt;

  // Instantiate the prologue.
  auto &shaderCache = context->v2_cache;
  DeviceProperties dprops = DeviceProperties();

  auto gemmPipelineValue = shaderCache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(gemmDesc, context->device.get(), dprops);
  auto gemmKernel = gemmPipelineValue->kernel;
  auto gemmPipeline = gemmPipelineValue->pipeline;

  // Allocate indirect command buffer.
  auto icbDesc = NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
  icbDesc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
  icbDesc->setInheritPipelineState(false);
  icbDesc->setInheritBuffers(false);
  icbDesc->setMaxKernelBufferBindCount(5);
  auto indirectCommandBuffer = NS::TransferPtr(context->device->newIndirectCommandBuffer(icbDesc.get(), params.segments, MTL::ResourceStorageModePrivate));

  prologueDesc.threadgroupSize = gemmKernel->threadgroupSize;
  prologueDesc.threadgroupMemoryAllocation = gemmKernel->threadgroupMemoryAllocation;
  prologueDesc.blockDimensions = gemmKernel->blockDimensions;
  auto pipelineValue = shaderCache.findKernel<SegmentedGEMMPrologueKernel, SegmentedGEMMPrologueDescriptor, SegmentedGEMMPrologueKernelDescriptor>(prologueDesc, context->device.get(), dprops);
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  // Allocate a new command.
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipeline.get());
  encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);

  // Bind the function arguments.
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
  if (num_tensors >= 6) {
    encoder->useResource(tensors[5], MTL::ResourceUsageRead);
  }
  for (int i = 0; i < num_tensors; ++i) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], i);
  }
  encoder->useResource(indirectCommandBuffer.get(), MTL::ResourceUsageWrite);
  auto argumentEncoder = NS::TransferPtr(kernel->function->newArgumentEncoder(num_tensors));
  auto argumentBuffer = NS::TransferPtr(context->device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceStorageModeShared));
  argumentEncoder->setArgumentBuffer(argumentBuffer.get(), 0);
  argumentEncoder->setIndirectCommandBuffer(indirectCommandBuffer.get(), 0);
  argumentEncoder->setComputePipelineState(gemmPipeline.get(), 1);
  encoder->useResource(argumentBuffer.get(), MTL::ResourceUsageRead);
  encoder->setBuffer(argumentBuffer.get(), 0, num_tensors);
  MTL::Size gridSize(1, 1, 1);
  MTL::Size groupSize(int64_t(params.segments), 1, 1);
  // Dispatch the required number of threads.
  encoder->dispatchThreadgroups(gridSize, groupSize);
  // Finish the command.
  command_batch->finishCommand(encoder);

  auto gemmEncoder = command_batch->startCommand();
  gemmEncoder->useResource(tensors[0], MTL::ResourceUsageRead);
  gemmEncoder->useResource(tensors[2], MTL::ResourceUsageRead);
  gemmEncoder->useResource(tensors[3], MTL::ResourceUsageRead);
  gemmEncoder->useResource(tensors[4], MTL::ResourceUsageWrite);
  gemmEncoder->useResource(indirectCommandBuffer.get(), MTL::ResourceUsageRead);
  gemmEncoder->executeCommandsInBuffer(indirectCommandBuffer.get(), NS::Range::Make(0, params.segments));
  command_batch->finishCommand(gemmEncoder);
}

