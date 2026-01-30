#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include "v2/ShaderCache.hpp"
#include "v2/GEMMKernel.hpp"
#include "v2/GEMMKernelDescriptor.hpp"
#include "v2/GEMMDescriptor.hpp"
#include "v2/NAMatMulKernel.hpp"
#include "v2/NAMatMulKernelDescriptor.hpp"
#include "v2/NAMatMulDescriptor.hpp"
#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params)
{
  // No-op.
}

size_t ccv_nnc_mfa_gemm_reserved_scratch_size(ccv_nnc_mfa_gemm_params_t params)
{
  if (params.use_neural_accelerators) {
    // Branch on whether to use the new kernel.
    NAMatMulDescriptor gemmDesc;
    gemmDesc.matrixDimensions = simd::uint3 {
      params.M,
      params.N,
      params.K,
    };
    size_t datatype_size = 0;
    switch (params.data_type) {
      case MTL::DataTypeHalf: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::FP16,
          .B = GEMMOperandPrecision::FP16,
          .C = GEMMOperandPrecision::FP16,
          .bias = GEMMOperandPrecision::FP16,
        };
        datatype_size = 2;
        break;
      }
      case MTL::DataTypeBFloat: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::BF16,
          .B = GEMMOperandPrecision::BF16,
          .C = GEMMOperandPrecision::BF16,
          .bias = GEMMOperandPrecision::BF16,
        };
        datatype_size = 2;
        break;
      }
      case MTL::DataTypeFloat: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::FP32,
          .B = GEMMOperandPrecision::FP32,
          .C = GEMMOperandPrecision::FP32,
          .bias = GEMMOperandPrecision::FP32,
        };
        datatype_size = 4;
        break;
      }
      default:
        CCV_NNC_MFA_PRECONDITION(false);
        break;
    }
    gemmDesc.transposeState = simd::uchar3 { params.A_trans, params.B_trans, params.D_trans };
    gemmDesc.registerPrecisionC = (params.register_float) ? std::optional(GEMMOperandPrecision::FP32) : std::nullopt;
    gemmDesc.useBias = params.fused_bias;
    gemmDesc.dispatchMMajor = NAMatMulDescriptor::preferDispatchMMajor(params.M, params.N, params.K);
    gemmDesc.loadM = true;
    gemmDesc.supportIndirectCommandBuffers = false;

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
    return datatype_size * params.M * params.N * gemmDesc.splitK() * params.batch_dimension;
  } else {
    return 0;
  }
}

void ccv_nnc_mfa_encode_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION((num_tensors == 3) || (num_tensors == 4))
  if (params.use_neural_accelerators && params.K < 65536) {
    // Branch on whether to use the new kernel.
    NAMatMulDescriptor gemmDesc;
    gemmDesc.matrixDimensions = simd::uint3 {
      params.M,
      params.N,
      params.K,
    };
    size_t datatype_size;
    switch (params.data_type) {
      case MTL::DataTypeHalf: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::FP16,
          .B = GEMMOperandPrecision::FP16,
          .C = GEMMOperandPrecision::FP16,
          .bias = GEMMOperandPrecision::FP16,
        };
        datatype_size = 2;
        break;
      }
      case MTL::DataTypeBFloat: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::BF16,
          .B = GEMMOperandPrecision::BF16,
          .C = GEMMOperandPrecision::BF16,
          .bias = GEMMOperandPrecision::BF16,
        };
        datatype_size = 2;
        break;
      }
      case MTL::DataTypeFloat: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::FP32,
          .B = GEMMOperandPrecision::FP32,
          .C = GEMMOperandPrecision::FP32,
          .bias = GEMMOperandPrecision::FP32,
        };
        datatype_size = 4;
        break;
      }
      default:
        CCV_NNC_MFA_PRECONDITION(false);
        break;
    }
    gemmDesc.transposeState = simd::uchar3 { params.A_trans, params.B_trans, params.D_trans };
    gemmDesc.registerPrecisionC = (params.register_float) ? std::optional(GEMMOperandPrecision::FP32) : std::nullopt;
    gemmDesc.useBias = params.fused_bias;
    gemmDesc.dispatchMMajor = NAMatMulDescriptor::preferDispatchMMajor(params.M, params.N, params.K);
    gemmDesc.loadM = true;
    gemmDesc.supportIndirectCommandBuffers = false;
  
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
    auto pipelineValue = shaderCache.findKernel<NAMatMulKernel, NAMatMulDescriptor, NAMatMulKernelDescriptor>(gemmDesc, context->device.get(), dprops);
    pool->drain();
    auto kernel = pipelineValue->kernel;
    auto pipeline = pipelineValue->pipeline;
  
    // Allocate a new command.
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline.get());

    // Bind the function arguments.
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    MTL::Buffer *scratch = NULL;
    if (kernel->splitK > 1) {
      scratch = context->request_scratch(datatype_size * params.M * params.N * kernel->splitK * params.batch_dimension);
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
    }
    if (num_tensors >= 4) {
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    }
    for (int i = 0; i < num_tensors; ++i) {
      if (kernel->splitK > 1 && i == 2) {
        encoder->setBuffer(scratch, 0, i);
	  } else {
        encoder->setBuffer(tensors[i], tensor_offsets[i], i);
	  }
    }
    encoder->setBytes(&params.M, sizeof(params.M), num_tensors);
  
    // Calculate the grid size.
    MTL::Size gridSize = kernel->threadgroupsPerGrid(gemmDesc);
    MTL::Size groupSize(int64_t(kernel->threadgroupSize(pipeline.get(), gemmDesc)), 1, 1);

    // Dispatch the required number of threads.
    encoder->dispatchThreadgroups(gridSize, groupSize);
  
    // Finish the command.
    command_batch->finishCommand(encoder);
    if (kernel->splitK > 1) { // reduce_sum kernel.
      auto encoder = command_batch->startCommand();
      auto second = pipelineValue->second;
      encoder->setComputePipelineState(second.get());
      encoder->setBuffer(scratch, 0, 0);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 1);
      encoder->setBytes(&params.M, sizeof(params.M), 2);
      encoder->useResource(scratch, MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      if ((params.N % 2) == 0) {
        encoder->dispatchThreadgroups(MTL::Size((params.M * params.N / 2 + 255) / 256, params.batch_dimension, 1), MTL::Size(256, 1, 1));
      } else {
        encoder->dispatchThreadgroups(MTL::Size((params.M * params.N + 255) / 256, params.batch_dimension, 1), MTL::Size(256, 1, 1));
      }
      command_batch->finishCommand(encoder);
    }
  } else {
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
      case MTL::DataTypeBFloat: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::BF16,
          .B = GEMMOperandPrecision::BF16,
          .C = GEMMOperandPrecision::BF16,
          .bias = GEMMOperandPrecision::BF16,
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
    gemmDesc.loadM = false;
    gemmDesc.supportIndirectCommandBuffers = false;
  
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
}

