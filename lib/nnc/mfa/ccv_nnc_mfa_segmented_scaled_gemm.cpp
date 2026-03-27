#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include "kernels/ShaderCache.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"
#include "kernels/NAInt8MatMulKernelDescriptor.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"
#include "kernels/SegmentedScaledGEMMPrologueKernel.hpp"
#include "kernels/SegmentedScaledGEMMPrologueKernelDescriptor.hpp"
#include "kernels/SegmentedScaledGEMMPrologueDescriptor.hpp"

namespace {

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
  return (value + alignment - 1) & ~(alignment - 1);
}

typedef struct {
  size_t q_bytes;
  size_t scale_offset;
  size_t scale_bytes;
  size_t scratch_bytes;
} ccv_nnc_mfa_activation_quant_layout_t;

typedef struct {
  uint32_t a_scale_offset;
  uint32_t b_scale_offset;
} ccv_nnc_mfa_segmented_scaled_gemm_offsets_t;

static GEMMOperandPrecision io_precision(uint64_t data_type) noexcept
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

static ccv_nnc_mfa_activation_quant_layout_t activation_quant_layout(ccv_nnc_mfa_segmented_scaled_gemm_params_t params) noexcept
{
  const size_t q_bytes = (size_t)params.originalM * params.K * sizeof(int8_t);
  const size_t scale_offset = align_up(q_bytes, 256);
  const size_t scale_bytes = (size_t)params.originalM * io_precision(params.data_type).size();
  return (ccv_nnc_mfa_activation_quant_layout_t){
    .q_bytes = q_bytes,
    .scale_offset = scale_offset,
    .scale_bytes = scale_bytes,
    .scratch_bytes = align_up(scale_offset + scale_bytes, 256),
  };
}

static size_t rowwise_8i_scale_offset(const uint32_t rows, const uint32_t cols) noexcept
{
  return align_up((size_t)rows * cols * sizeof(int8_t), 128);
}

}

void ccv_nnc_mfa_prepare_segmented_scaled_gemm(mfa::context* context, ccv_nnc_mfa_segmented_scaled_gemm_params_t params)
{
  (void)context;
  (void)params;
}

size_t ccv_nnc_mfa_segmented_scaled_gemm_reserved_scratch_size(ccv_nnc_mfa_segmented_scaled_gemm_params_t params)
{
  if (!params.use_neural_accelerators)
    return 0;
  return activation_quant_layout(params).scratch_bytes;
}

void ccv_nnc_mfa_encode_segmented_scaled_gemm(
    mfa::context* context,
    ccv_nnc_mfa_segmented_scaled_gemm_params_t params,
    MTL::CommandBatch* command_batch,
    MTL::Buffer** tensors,
    size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    ++num_tensors;
  CCV_NNC_MFA_PRECONDITION((num_tensors == 5) || (num_tensors == 6));
  CCV_NNC_MFA_PRECONDITION(params.use_neural_accelerators);

  NAInt8MatMulDescriptor matmulDesc;
  matmulDesc.batchDimension = 1;
  matmulDesc.ioPrecision = io_precision(params.data_type);
  matmulDesc.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  matmulDesc.batchStrides = std::nullopt;
  matmulDesc.useBias = params.fused_bias;
  matmulDesc.loadM = true;
  matmulDesc.supportIndirectCommandBuffers = true;

  NAInt8MatMulDescriptor quantizeDesc = matmulDesc;
  quantizeDesc.matrixDimensions[0] = params.originalM;
  quantizeDesc.loadM = false;
  quantizeDesc.supportIndirectCommandBuffers = false;

  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto matmulPipelineValue = shaderCache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(matmulDesc, context->device.get(), dprops);
  auto matmulKernel = matmulPipelineValue->kernel;
  auto matmulPipeline = matmulPipelineValue->pipeline;
  auto quantizePipelineValue = shaderCache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(quantizeDesc, context->device.get(), dprops);
  auto quantizePipeline = quantizePipelineValue->second;

  SegmentedScaledGEMMPrologueDescriptor prologueDesc;
  prologueDesc.matrixDimensions = simd::uint3 { params.segments, params.N, params.K };
  prologueDesc.blockDimensions = matmulKernel->blockDimensions;
  prologueDesc.ioPrecision = matmulDesc.ioPrecision;
  prologueDesc.useBias = params.fused_bias;
  prologueDesc.threadgroupSize = matmulKernel->threadgroupSize(matmulPipeline.get());
  auto prologuePipelineValue = shaderCache.findKernel<SegmentedScaledGEMMPrologueKernel, SegmentedScaledGEMMPrologueDescriptor, SegmentedScaledGEMMPrologueKernelDescriptor>(prologueDesc, context->device.get(), dprops);
  auto prologuePipeline = prologuePipelineValue->pipeline;
  auto indirectCommandBuffer = prologuePipelineValue->indirect1;

  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  auto scratch = context->request_scratch(a_layout.scratch_bytes);
  const size_t b_scale_offset = rowwise_8i_scale_offset(params.segments * params.N, params.K);

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(quantizePipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBuffer(scratch, a_layout.scale_offset, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(params.originalM, 1, 1),
        MTL::Size(matmulKernel->activationQuantizeThreads, 1, 1));
    command_batch->finishCommand(encoder);
  }

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(prologuePipeline.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
    if (num_tensors >= 6)
      encoder->useResource(tensors[5], MTL::ResourceUsageRead);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
    encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
    encoder->setBuffer(tensors[4], tensor_offsets[4], 4);
    const ccv_nnc_mfa_segmented_scaled_gemm_offsets_t offsets = {
      .a_scale_offset = (uint32_t)a_layout.scale_offset,
      .b_scale_offset = (uint32_t)b_scale_offset,
    };
    encoder->setBytes(&offsets, sizeof(offsets), 5);
    const NS::UInteger argsIndex = params.fused_bias ? 7 : 6;
    if (num_tensors >= 6)
      encoder->setBuffer(tensors[5], tensor_offsets[5], 6);
    encoder->useResource(indirectCommandBuffer.get(), MTL::ResourceUsageWrite);
    auto argumentEncoder = NS::TransferPtr(prologuePipelineValue->function->newArgumentEncoder(argsIndex));
    auto argumentBuffer = NS::TransferPtr(context->device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceStorageModeShared));
    argumentEncoder->setArgumentBuffer(argumentBuffer.get(), 0);
    argumentEncoder->setIndirectCommandBuffer(indirectCommandBuffer.get(), 0);
    argumentEncoder->setComputePipelineState(matmulPipeline.get(), 1);
    encoder->useResource(argumentBuffer.get(), MTL::ResourceUsageRead);
    encoder->setBuffer(argumentBuffer.get(), 0, argsIndex);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(params.segments, 1, 1));
    command_batch->finishCommand(encoder);
  }

  {
    auto encoder = command_batch->startCommand();
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
    if (num_tensors >= 6)
      encoder->useResource(tensors[5], MTL::ResourceUsageRead);
    encoder->useResource(indirectCommandBuffer.get(), MTL::ResourceUsageRead);
    encoder->executeCommandsInBuffer(indirectCommandBuffer.get(), NS::Range::Make(0, params.segments));
    command_batch->finishCommand(encoder);
  }
}
