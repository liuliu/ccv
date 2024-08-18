#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include "v2/ShaderCache.hpp"
#include "v2/GEMMKernel.hpp"
#include "v2/GEMMKernelDescriptor.hpp"
#include "v2/GEMMDescriptor.hpp"
#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params)
{
  // No-op.
}

void ccv_nnc_mfa_encode_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION((num_tensors == 3) || (num_tensors == 4))

  // Branch on whether to use the new kernel.
  GEMMDescriptor gemmDesc;
  gemmDesc.matrixDimensions = simd::uint3 {
    params.M,
    params.N,
    params.K,
  };
  switch (params.data_type) {
    case MTL::DataTypeHalf: {
      gemmDesc.memoryPrecisions = {
        .A = GEMMOperandPrecision::FP16,
        .B = GEMMOperandPrecision::FP16,
        .C = GEMMOperandPrecision::FP16,
        .bias = GEMMOperandPrecision::FP16,
      };
      break;
    }
    case MTL::DataTypeFloat: {
      gemmDesc.memoryPrecisions = {
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
  gemmDesc.transposeState = simd::uchar3 { params.A_trans, params.B_trans, params.D_trans };
  gemmDesc.registerPrecisionC = (params.register_float) ? std::optional(GEMMOperandPrecision::FP32) : std::nullopt;
  gemmDesc.leadingDimensions = std::nullopt;
  gemmDesc.loadPreviousC = false;
  gemmDesc.useBias = params.fused_bias;

  gemmDesc.batchDimension = params.batch_dimension;
  if (params.batch_dimension > 1) {
	simd::uint4 batchStrides;
    batchStrides[0] = params.batch_stride_a;
    batchStrides[1] = params.batch_stride_b;
    batchStrides[2] = params.batch_stride_c;
    batchStrides[3] = params.batch_stride_d;
	gemmDesc.batchStrides = batchStrides;
  } else {
    gemmDesc.batchStrides = std::nullopt;
  }

  // Instantiate the kernel.
  //
  // TODO: Remove the autoreleasepool, once you confirm the caller always
  // makes one. Or find a different solution, like spawning a pool inside
  // of 'fetchKernel' when a new kernel variant is compiled.
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->v2_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(gemmDesc, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  // Allocate a new command.
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipeline.get());
  encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);

  // Bind the function arguments.
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  if (num_tensors >= 4) {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  }
  for (int i = 0; i < num_tensors; ++i) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], i);
  }

  // Calculate the grid size.
  auto ceilDivide =
  [=](int64_t target, uint16_t granularity) -> int64_t {
    return (target + int64_t(granularity) - 1) / int64_t(granularity);
  };
  MTL::Size gridSize
  (ceilDivide(int64_t(params.N), kernel->blockDimensions[1]),
   ceilDivide(int64_t(params.M), kernel->blockDimensions[0]),
   gemmDesc.batchDimension);
  MTL::Size groupSize
  (int64_t(kernel->threadgroupSize), 1, 1);

  // Dispatch the required number of threads.
  encoder->dispatchThreadgroups(gridSize, groupSize);

  // Finish the command.
  command_batch->finishCommand(encoder);
}

