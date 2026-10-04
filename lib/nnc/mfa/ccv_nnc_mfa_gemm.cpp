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
#include "kernels/NAMatMulTuning.hpp"
#include "kernels/NARegisterMatMulKernel.hpp"
#include "kernels/NAMatMulSmallMKernel.hpp"
#include "kernels/NAMatMulSmallMKernelDescriptor.hpp"
#include "kernels/NAMatMulSmallMDescriptor.hpp"
#include <string>

static constexpr uint32_t kNAMatMulSmallMLowKMaxM = 48;
static constexpr uint32_t kNAMatMulSmallMReducedMaxM = 16;
static constexpr uint32_t kNAMatMulSmallMHighK = 16384;
static constexpr uint32_t kNAMatMulSmallMVectorK = 5120;
static constexpr uint32_t kNAMatMulSmallMPack = 8;

static bool _ccv_nnc_mfa_register_matmul_supported(const ccv_nnc_mfa_gemm_params_t params) noexcept
{
  // FP32 accumulation benefits from register fragments even on smaller grids.
  // Half accumulation needs more rows and the former minimum output area.
  // Preserve the wide MPP profile's advantage on compact moderate reductions;
  // like NAMatMulDescriptor::useWideTile(), that profile needs full N/K tiles.
  const bool native_moderate_reduction = params.K <= 8192 &&
      params.K >= uint64_t(2) * params.N && params.N % 128 == 0 && params.K % 512 == 0;
  const bool register_grid = params.M >= 4096 || (params.M >= 512 && params.register_float) ||
      (params.M >= 2048 && uint64_t(params.M) * params.N >= 4096ull * 1536 &&
       !native_moderate_reduction);
  return params.use_neural_accelerators && params.data_type == MTL::DataTypeHalf &&
      (!params.output_data_type || params.output_data_type == MTL::DataTypeHalf) &&
      params.batch_dimension == 1 && !params.A_trans && params.B_trans && !params.D_trans &&
      !params.leading_dimension_a && !params.leading_dimension_c &&
      register_grid && params.N >= 1536 &&
      params.M <= INT32_MAX - 127 && params.N <= INT32_MAX - 127;
}

static bool _ccv_nnc_mfa_register_matmul_split_k(const ccv_nnc_mfa_gemm_params_t params) noexcept
{
  // Wide outputs with deep reductions benefit from partition-major traversal.
  // Keep each FP32 partial output within the measured 256 MiB working set and
  // at most eight 4096-element partitions. Tall outputs favor the unsplit path.
  return _ccv_nnc_mfa_register_matmul_supported(params) && params.M < params.N &&
      params.K >= uint64_t(2) * params.N && params.K <= 32768 &&
      uint64_t(params.M) * params.N * sizeof(float) <= (256ull << 20);
}

static bool _ccv_nnc_mfa_gemm_memory_precisions(const ccv_nnc_mfa_gemm_params_t params, GEMMOperandPrecisions* const precisions) noexcept
{
  GEMMOperandPrecision input_precision;
  GEMMOperandPrecision output_precision;
  switch (params.data_type) {
    case MTL::DataTypeHalf:
      input_precision = GEMMOperandPrecision::FP16;
      break;
    case MTL::DataTypeBFloat:
      input_precision = GEMMOperandPrecision::BF16;
      break;
    case MTL::DataTypeFloat:
      input_precision = GEMMOperandPrecision::FP32;
      break;
    default:
      return false;
  }
  const uint64_t output_data_type = params.output_data_type ? params.output_data_type : params.data_type;
  switch (output_data_type) {
    case MTL::DataTypeHalf:
      output_precision = GEMMOperandPrecision::FP16;
      break;
    case MTL::DataTypeBFloat:
      output_precision = GEMMOperandPrecision::BF16;
      break;
    case MTL::DataTypeFloat:
      output_precision = GEMMOperandPrecision::FP32;
      break;
    default:
      return false;
  }
  *precisions = {
    .A = input_precision,
    .B = input_precision,
    .C = output_precision,
    .bias = output_precision,
  };
  return true;
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
  if (_ccv_nnc_mfa_register_matmul_split_k(params)) {
    // This API has no device argument. Conservatively reserve for both paths
    // so packed / palettized weights placed after this region remain intact.
    NAMatMulDescriptor descriptor;
    descriptor.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
    const size_t partitions = (params.K + 4095) / 4096;
    return size_t(params.M) * params.N *
        std::max(partitions * sizeof(float), size_t(descriptor.splitK()) * sizeof(uint16_t));
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
  const uint32_t dimensions[] = {
    params.M, params.batch_stride_a, params.batch_stride_b, params.batch_stride_c, params.batch_stride_d,
  };
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION((num_tensors == 3) || (num_tensors == 4))
  if (_ccv_nnc_mfa_use_na_matmul_small_m(params)) {
    _ccv_nnc_mfa_encode_na_matmul_small_m(context, params, command_batch, tensors, tensor_offsets, num_tensors);
    return;
  }
  // Register fragments avoid tensor-view staging and, on sufficiently large
  // grids, repeated full-output writes. Keep the existing path for narrow
  // outputs and deep reductions with large weight working sets. The bounds
  // are measured neighboring-shape limits, not assumed hardware cache sizes.
  const bool register_split_k = _ccv_nnc_mfa_register_matmul_split_k(params);
  const bool register_unsplit = params.K >= 2048 && (params.K <= 8192 ||
      (params.K <= 16384 && params.K < uint64_t(3) * params.N &&
       (params.M < params.N || params.M >= uint64_t(2) * params.N) &&
       uint64_t(params.N) * params.K * sizeof(uint16_t) <= (192ull << 20)));
  if (_ccv_nnc_mfa_register_matmul_supported(params) &&
      useNeuralAcceleratorMatMulTuning(context->device.get()) && (register_split_k || register_unsplit)) {
    const bool wide_m = params.N > params.M;
    const int32_t block_m = wide_m ? 128 : 64, block_n = wide_m ? 64 : 128;
    const NARegisterMatMulDescriptor descriptor {
      params.M % block_m == 0, params.N % block_n == 0, params.K % 512 == 0,
      bool(params.fused_bias), false, wide_m, register_split_k
    };
    auto pool = NS::AutoreleasePool::alloc()->init();
    auto* value = context->kernel_cache.findKernel<NARegisterMatMulKernel,
        NARegisterMatMulDescriptor, NARegisterMatMulKernelDescriptor>(
            descriptor, context->device.get(), DeviceProperties());
    pool->drain();
    const int32_t M = params.M, N = params.N, K = params.K;
    const int32_t tiles_n = (N + block_n - 1) / block_n, tiles_m = (M + block_m - 1) / block_m;
    if (register_split_k) {
      const int32_t partitions = (K + 4095) / 4096;
      const NARegisterMatMulSplitKParams arguments {
        M, N, K, K, K, N, tiles_n, tiles_m, partitions, M * N, 4096, 1, 8
      };
      auto* scratch = context->request_scratch(size_t(M) * N * partitions * sizeof(float));
      auto* encoder = command_batch->startCommand();
      encoder->setComputePipelineState(value->pipeline.get());
      encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
      encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
      encoder->setBuffer(scratch, 0, 2);
      encoder->setBytes(&arguments, sizeof(arguments), 3);
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
      // Partition-major 1D grid preserves the intended traversal order.
      encoder->dispatchThreadgroups(MTL::Size(tiles_n * 2 * ((tiles_m + 1) / 2) * partitions, 1, 1),
          MTL::Size(32, 2, 4));
      command_batch->finishCommand(encoder);
      encoder = command_batch->startCommand();
      encoder->setComputePipelineState(value->second.get());
      encoder->setBuffer(scratch, 0, 0);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 1);
      encoder->setBytes(&arguments, sizeof(arguments), 2);
      encoder->useResource(scratch, MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      if (params.fused_bias) {
        CCV_NNC_MFA_PRECONDITION(num_tensors == 4);
        encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
        encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      }
      encoder->dispatchThreads(MTL::Size(N, M, 1), MTL::Size(32, 8, 1));
      command_batch->finishCommand(encoder);
      return;
    }
    const int32_t swizzle_log = wide_m ? 2 : 0;
    const NARegisterMatMulParams arguments {
      M, N, K, K, K, N, tiles_n, tiles_m,
      0, 0, 0, swizzle_log, K / 512, 1
    };
    auto* encoder = command_batch->startCommand();
    encoder->setComputePipelineState(value->pipeline.get());
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 3);
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
    encoder->setBytes(&arguments, sizeof(arguments), 4);
    if (params.fused_bias) {
      CCV_NNC_MFA_PRECONDITION(num_tensors == 4);
      const NARegisterMatMulBiasParams bias_arguments {0, 1, 0, 1, 1};
      encoder->setBuffer(tensors[3], tensor_offsets[3], 2);
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      encoder->setBytes(&bias_arguments, sizeof(bias_arguments), 5);
    }
    encoder->dispatchThreadgroups(MTL::Size(tiles_n << swizzle_log,
        (tiles_m + (1 << swizzle_log) - 1) >> swizzle_log, 1),
        MTL::Size(32, block_n / 32, block_m / 32));
    command_batch->finishCommand(encoder);
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
    encoder->setBytes(dimensions, sizeof(dimensions), num_tensors);
  
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
      encoder->setBytes(dimensions, sizeof(dimensions), 2);
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
    CCV_NNC_MFA_PRECONDITION(_ccv_nnc_mfa_gemm_memory_precisions(params, &gemmDesc.memoryPrecisions));
    gemmDesc.transposeState = simd::uchar3 { params.A_trans, params.B_trans, params.D_trans };
    // The generic fallback does not split long reductions. Half accumulation
    // can stop changing well before K=65536, even when the result is finite.
    gemmDesc.registerPrecisionC = (params.register_float || params.K >= 65536) ?
        std::optional(GEMMOperandPrecision::FP32) : std::nullopt;
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
      encoder->setBytes(dimensions, sizeof(dimensions), num_tensors);
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
