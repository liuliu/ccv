#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include "kernels/ShaderCache.hpp"
#include "kernels/GEMMKernel.hpp"
#include "kernels/GEMMKernelDescriptor.hpp"
#include "kernels/GEMMDescriptor.hpp"
#include "kernels/NAMatMulKernel.hpp"
#include "kernels/NAMatMulKernelDescriptor.hpp"
#include "kernels/NAMatMulDescriptor.hpp"
#include "kernels/NAMatMulSmallMKernel.hpp"
#include "kernels/NAMatMulSmallMKernelDescriptor.hpp"
#include "kernels/NAMatMulSmallMDescriptor.hpp"
#include <string>

static constexpr uint32_t kNAMatMulSmallMLowKMaxM = 48;
static constexpr uint32_t kNAMatMulSmallMReducedMaxM = 16;
static constexpr uint32_t kNAMatMulSmallMHighK = 16384;
static constexpr uint32_t kNAMatMulSmallMVectorK = 5120;
static constexpr uint32_t kNAMatMulSmallMPack = 8;

static bool _ccv_nnc_mfa_gemm_memory_precisions(const ccv_nnc_mfa_gemm_params_t params, GEMMOperandPrecisions* const precisions) noexcept
{
  switch (params.data_type) {
    case MTL::DataTypeHalf: {
      *precisions = {
        .A = GEMMOperandPrecision::FP16,
        .B = GEMMOperandPrecision::FP16,
        .C = GEMMOperandPrecision::FP16,
        .bias = GEMMOperandPrecision::FP16,
      };
      return true;
    }
    case MTL::DataTypeBFloat: {
      *precisions = {
        .A = GEMMOperandPrecision::BF16,
        .B = GEMMOperandPrecision::BF16,
        .C = GEMMOperandPrecision::BF16,
        .bias = GEMMOperandPrecision::BF16,
      };
      return true;
    }
    case MTL::DataTypeFloat: {
      *precisions = {
        .A = GEMMOperandPrecision::FP32,
        .B = GEMMOperandPrecision::FP32,
        .C = GEMMOperandPrecision::FP32,
        .bias = GEMMOperandPrecision::FP32,
      };
      return true;
    }
    default:
      return false;
  }
}

static bool _ccv_nnc_mfa_use_na_matmul_small_m(const ccv_nnc_mfa_gemm_params_t params) noexcept
{
  // This variant targets small M over transposed weights:
  // C[M, N] = A[M, K] * B[N, K]^T. The packed diagonal trick requires NAX.
  // At larger K, only widen M when the SmallM split-K path can use full packed tiles.
  NAMatMulSmallMDescriptor splitDesc;
  splitDesc.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  const uint16_t splitK = splitDesc.splitK();
  const uint32_t high_k_max_m = splitK > 1 ? kNAMatMulSmallMReducedMaxM : 8;
  const uint32_t max_m = params.K >= kNAMatMulSmallMHighK ? high_k_max_m : kNAMatMulSmallMLowKMaxM;
  return params.use_neural_accelerators &&
    (params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeBFloat) &&
    params.M <= max_m &&
    (params.M != 1 || params.K < kNAMatMulSmallMVectorK) &&
    params.batch_dimension == 1 &&
    !params.A_trans &&
    params.B_trans &&
    !params.D_trans &&
    (params.K % kNAMatMulSmallMPack) == 0 &&
    params.K < 65536;
}

static NAMatMulSmallMDescriptor _ccv_nnc_mfa_make_na_matmul_small_m_descriptor(const ccv_nnc_mfa_gemm_params_t params) noexcept
{
  NAMatMulSmallMDescriptor desc;
  desc.matrixDimensions = simd::uint3 {
    params.M,
    params.N,
    params.K,
  };
  CCV_NNC_MFA_PRECONDITION(_ccv_nnc_mfa_gemm_memory_precisions(params, &desc.memoryPrecisions));
  desc.useBias = params.fused_bias;
  desc.batchDimension = params.batch_dimension;
  desc.loadM = params.loadM;
  return desc;
}

static void _ccv_nnc_mfa_encode_na_matmul_small_m(
  mfa::context* context,
  const ccv_nnc_mfa_gemm_params_t params,
  MTL::CommandBatch* command_batch,
  MTL::Buffer** tensors,
  size_t* tensor_offsets,
  const int num_tensors)
{
  CCV_NNC_MFA_PRECONDITION((params.fused_bias && num_tensors == 4) || (!params.fused_bias && num_tensors == 3));
  NAMatMulSmallMDescriptor desc = _ccv_nnc_mfa_make_na_matmul_small_m_descriptor(params);
  const NAMatMulSmallMScratchOffsets offsets = desc.scratchOffsets();

  if (METAL_LOG_LEVEL(context) >= 1) {
    ccv_nnc_mfa_log_message("Using NAX small-M MatMul.");
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<NAMatMulSmallMKernel, NAMatMulSmallMDescriptor, NAMatMulSmallMKernelDescriptor>(desc, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  MTL::Buffer* const scratch = context->request_scratch(offsets.total);

  const uint64_t reduce_threads = (uint64_t)params.M * params.N;
  const MTL::Size linear_group_size(256, 1, 1);
  auto dispatch_linear =
  [&](MTL::ComputeCommandEncoder* const encoder, const uint64_t threads) {
    encoder->dispatchThreadgroups(
      MTL::Size((int64_t)((threads + 255) / 256), 1, 1),
      linear_group_size);
  };

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelineValue->pipeline.get());
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 0);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 1);
    encoder->setBuffer(scratch, offsets.partials, 2);
    if (desc.loadM) {
      encoder->setBytes(&params.M, sizeof(params.M), 3);
    }
    encoder->dispatchThreadgroups(
      kernel->threadgroupsPerGrid(desc),
      MTL::Size(kernel->threadgroupSize(pipelineValue->pipeline.get()), 1, 1));
    command_batch->finishCommand(encoder);
  }
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelineValue->second.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
    if (params.fused_bias) {
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    }
    encoder->setBuffer(scratch, offsets.partials, 0);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 1);
    if (params.fused_bias) {
      encoder->setBuffer(tensors[3], tensor_offsets[3], 2);
    }
    if (desc.loadM) {
      encoder->setBytes(&params.M, sizeof(params.M), params.fused_bias ? 3 : 2);
    }
    dispatch_linear(encoder, reduce_threads);
    command_batch->finishCommand(encoder);
  }
}

// MARK: - C

void ccv_nnc_mfa_prepare_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params)
{
  // No-op.
}

size_t ccv_nnc_mfa_gemm_reserved_scratch_size(ccv_nnc_mfa_gemm_params_t params)
{
  if (_ccv_nnc_mfa_use_na_matmul_small_m(params)) {
    NAMatMulSmallMDescriptor desc = _ccv_nnc_mfa_make_na_matmul_small_m_descriptor(params);
    return desc.scratchOffsets().total;
  }
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
  if (_ccv_nnc_mfa_use_na_matmul_small_m(params)) {
    _ccv_nnc_mfa_encode_na_matmul_small_m(context, params, command_batch, tensors, tensor_offsets, num_tensors);
    return;
  }
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
    if (params.leading_dimension_a || params.leading_dimension_c) {
      CCV_NNC_MFA_PRECONDITION(params.leading_dimension_a && params.leading_dimension_c);
      CCV_NNC_MFA_PRECONDITION(!params.A_trans && params.B_trans && !params.D_trans && !params.fused_bias);
      gemmDesc.leadingDimensions = simd::uint2 {
        params.leading_dimension_a,
        params.leading_dimension_c,
      };
    } else {
      gemmDesc.leadingDimensions = std::nullopt;
    }
    gemmDesc.useBias = params.fused_bias;
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
    auto &shaderCache = context->kernel_cache;
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
    if (params.leading_dimension_a || params.leading_dimension_c) {
      gemmDesc.leadingDimensions = simd::uint3 {
        params.leading_dimension_a,
        0,
        params.leading_dimension_c,
      };
    } else {
      gemmDesc.leadingDimensions = std::nullopt;
    }
    gemmDesc.loadPreviousC = false;
    gemmDesc.useBias = params.fused_bias;
    gemmDesc.loadM = params.loadM;
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
    auto &shaderCache = context->kernel_cache;
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
    if (gemmDesc.loadM) {
      encoder->setBytes(&params.M, sizeof(params.M), num_tensors);
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
