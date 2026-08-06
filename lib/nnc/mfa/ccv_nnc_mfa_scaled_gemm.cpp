#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include "kernels/ShaderCache.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"
#include "kernels/NAInt8MatMulKernelDescriptor.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"
#include "kernels/NAInt8MatMulSmallMKernel.hpp"
#include "kernels/NAInt8MatMulSmallMKernelDescriptor.hpp"
#include "kernels/NAInt8MatMulSmallMDescriptor.hpp"

namespace {

static constexpr uint32_t kNAInt8MatMulSmallMLowKMaxM = 16;
static constexpr uint32_t kNAInt8MatMulSmallMReducedMaxM = 16;
static constexpr uint32_t kNAInt8MatMulSmallMHighK = 16384;
static constexpr uint32_t kNAInt8MatMulSmallMPack = 8;

static size_t align_up(const size_t value, const size_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

typedef struct {
  size_t q_bytes;
  size_t scale_offset;
  size_t scale_bytes;
  size_t scratch_bytes;
} ccv_nnc_mfa_activation_quant_layout_t;

static GEMMOperandPrecision io_precision(uint64_t data_type) noexcept {
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

static ccv_nnc_mfa_activation_quant_layout_t activation_quant_layout(ccv_nnc_mfa_scaled_gemm_params_t params) noexcept
{
  const size_t q_bytes = (size_t)params.batch_dimension * params.M * params.K * sizeof(int8_t);
  const size_t scale_offset = align_up(q_bytes, 256);
  const size_t scale_bytes = (size_t)params.batch_dimension * params.M * io_precision(params.data_type).size();
  return (ccv_nnc_mfa_activation_quant_layout_t){
    .q_bytes = q_bytes,
    .scale_offset = scale_offset,
    .scale_bytes = scale_bytes,
    .scratch_bytes = align_up(scale_offset + scale_bytes, 256),
  };
}

static size_t rowwise_8i_scale_offset(const size_t rows, const size_t cols) noexcept
{
  return align_up(rows * cols * sizeof(int8_t), 128);
}

static NAInt8MatMulSmallMDescriptor make_na_int8_matmul_small_m_descriptor(ccv_nnc_mfa_scaled_gemm_params_t params) noexcept
{
  NAInt8MatMulSmallMDescriptor desc;
  desc.batchDimension = params.batch_dimension;
  desc.ioPrecision = io_precision(params.data_type);
  desc.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  desc.useBias = params.fused_bias;
  desc.loadM = params.loadM;
  return desc;
}

static bool use_na_int8_matmul_small_m(ccv_nnc_mfa_scaled_gemm_params_t params) noexcept
{
  NAInt8MatMulSmallMDescriptor splitDesc = make_na_int8_matmul_small_m_descriptor(params);
  const uint16_t splitK = splitDesc.splitK();
  const uint32_t high_k_max_m = splitK > 1 ? kNAInt8MatMulSmallMReducedMaxM : 8;
  const uint32_t max_m = params.K >= kNAInt8MatMulSmallMHighK ? high_k_max_m : kNAInt8MatMulSmallMLowKMaxM;
  return params.use_neural_accelerators &&
    !params.leading_dimension_a &&
    !params.leading_dimension_c &&
    params.M <= max_m &&
    params.batch_dimension == 1 &&
    (params.K % kNAInt8MatMulSmallMPack) == 0 &&
    params.K < 65536;
}

}

void ccv_nnc_mfa_prepare_scaled_gemm(mfa::context* context, ccv_nnc_mfa_scaled_gemm_params_t params)
{
  (void)context;
  (void)params;
}

size_t ccv_nnc_mfa_scaled_gemm_reserved_scratch_size(ccv_nnc_mfa_scaled_gemm_params_t params)
{
  if (!params.use_neural_accelerators)
    return 0;
  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  if (use_na_int8_matmul_small_m(params)) {
    const NAInt8MatMulSmallMDescriptor desc = make_na_int8_matmul_small_m_descriptor(params);
    return align_up(align_up(a_layout.scratch_bytes, 256) + desc.scratchOffsets().total, 256);
  }
  return a_layout.scratch_bytes;
}

void ccv_nnc_mfa_encode_scaled_gemm(mfa::context* context, ccv_nnc_mfa_scaled_gemm_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    ++num_tensors;
  CCV_NNC_MFA_PRECONDITION((num_tensors == 3) || (num_tensors == 4));
  CCV_NNC_MFA_PRECONDITION(params.use_neural_accelerators);

  NAInt8MatMulDescriptor matmulDesc;
  matmulDesc.batchDimension = params.batch_dimension;
  matmulDesc.ioPrecision = io_precision(params.data_type);
  matmulDesc.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  matmulDesc.loadM = params.loadM;
  if (params.leading_dimension_a || params.leading_dimension_c) {
    CCV_NNC_MFA_PRECONDITION(params.leading_dimension_a && params.leading_dimension_c);
    matmulDesc.leadingDimensions = simd::uint2 {
      params.leading_dimension_a,
      params.leading_dimension_c,
    };
  } else {
    matmulDesc.leadingDimensions = std::nullopt;
  }
  if (params.batch_dimension > 1) {
    simd::uint4 batchStrides;
    batchStrides[0] = params.batch_stride_a;
    batchStrides[1] = params.batch_stride_b;
    batchStrides[2] = params.batch_stride_c;
    batchStrides[3] = params.batch_stride_d;
    matmulDesc.batchStrides = batchStrides;
    matmulDesc.packedABatchStride = params.M * params.K;
    matmulDesc.aScaleBatchStride = params.M;
  } else {
    matmulDesc.batchStrides = std::nullopt;
  }
  matmulDesc.useBias = params.fused_bias;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(matmulDesc, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto matmulPipeline = pipelineValue->pipeline;
  auto quantizePipeline = pipelineValue->second;

  const bool useSmallM = use_na_int8_matmul_small_m(params);
  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  size_t scratch_bytes = a_layout.scratch_bytes;
  NAInt8MatMulSmallMDescriptor smallDesc;
  NAInt8MatMulSmallMScratchOffsets smallOffsets = { 0, 0 };
  size_t small_scratch_base = 0;
  if (useSmallM) {
    smallDesc = make_na_int8_matmul_small_m_descriptor(params);
    smallOffsets = smallDesc.scratchOffsets();
    small_scratch_base = align_up(a_layout.scratch_bytes, 256);
    scratch_bytes = small_scratch_base + smallOffsets.total;
  }
  auto scratch = context->request_scratch(scratch_bytes);
  const uint32_t b_batches = (params.batch_dimension > 1 && params.batch_stride_b > 0) ? params.batch_dimension : 1;
  const size_t b_scale_offset = rowwise_8i_scale_offset((size_t)b_batches * params.N, params.K);

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(quantizePipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBuffer(scratch, a_layout.scale_offset, 2);
    if (matmulDesc.loadM) {
      encoder->setBytes(&params.M, sizeof(params.M), 3);
    }
    encoder->dispatchThreadgroups(
        MTL::Size(params.M, 1, params.batch_dimension),
        MTL::Size(kernel->activationQuantizeThreads, 1, 1));
    command_batch->finishCommand(encoder);
  }

  if (useSmallM) {
    CCV_NNC_MFA_PRECONDITION((params.fused_bias && num_tensors == 4) || (!params.fused_bias && num_tensors == 3));
    if (METAL_LOG_LEVEL(context) >= 1) {
      ccv_nnc_mfa_log_message("Using NAX small-M Int8 MatMul.");
    }
    auto smallPool = NS::AutoreleasePool::alloc()->init();
    auto smallPipelineValue = shaderCache.findKernel<NAInt8MatMulSmallMKernel, NAInt8MatMulSmallMDescriptor, NAInt8MatMulSmallMKernelDescriptor>(smallDesc, context->device.get(), dprops);
    smallPool->drain();
    auto smallKernel = smallPipelineValue->kernel;
    const size_t partials_offset = small_scratch_base + smallOffsets.partials;
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(smallPipelineValue->pipeline.get());
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->setBuffer(tensors[1], tensor_offsets[1], 0);
      encoder->setBuffer(scratch, 0, 1);
      encoder->setBuffer(scratch, partials_offset, 2);
      if (smallDesc.loadM) {
        encoder->setBytes(&params.M, sizeof(params.M), 3);
      }
      encoder->dispatchThreadgroups(
          smallKernel->threadgroupsPerGrid(smallDesc),
          MTL::Size(smallKernel->threadgroupSize(smallPipelineValue->pipeline.get()), 1, 1));
      command_batch->finishCommand(encoder);
    }
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(smallPipelineValue->second.get());
      encoder->useResource(scratch, MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      if (params.fused_bias) {
        encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      }
      encoder->setBuffer(scratch, partials_offset, 0);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 1);
      encoder->setBuffer(scratch, a_layout.scale_offset, 2);
      encoder->setBuffer(tensors[1], tensor_offsets[1] + b_scale_offset, 3);
      if (params.fused_bias) {
        encoder->setBuffer(tensors[3], tensor_offsets[3], 4);
      }
      if (smallDesc.loadM) {
        encoder->setBytes(&params.M, sizeof(params.M), params.fused_bias ? 5 : 4);
      }
      encoder->dispatchThreadgroups(
          MTL::Size((int64_t)(((uint64_t)params.M * params.N + 255) / 256), 1, 1),
          MTL::Size(256, 1, 1));
      command_batch->finishCommand(encoder);
    }
    return;
  }

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(matmulPipeline.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
    if (num_tensors >= 4)
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
    encoder->setBuffer(scratch, a_layout.scale_offset, 3);
    encoder->setBuffer(tensors[1], tensor_offsets[1] + b_scale_offset, 4);
    if (num_tensors >= 4)
      encoder->setBuffer(tensors[3], tensor_offsets[3], 5);
    if (matmulDesc.loadM)
      encoder->setBytes(&params.M, sizeof(params.M), params.fused_bias ? 6 : 5);
    encoder->dispatchThreadgroups(
        kernel->threadgroupsPerGrid(params.M, params.N, params.batch_dimension),
        MTL::Size(kernel->threadgroupSize(matmulPipeline.get()), 1, 1));
    command_batch->finishCommand(encoder);
  }
}
