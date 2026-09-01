#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include "kernels/ShaderCache.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"
#include "kernels/NAInt8MatMulKernelDescriptor.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"
#include "kernels/SegmentedScaledGEMMKernel.hpp"
#include "kernels/SegmentedScaledGEMMKernelDescriptor.hpp"
#include "kernels/SegmentedScaledGEMMDescriptor.hpp"
#include "kernels/Int8MatMulDescriptor.hpp"
#include "kernels/Int8MatMulKernel.hpp"

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
  size_t records_offset;
  size_t dispatch_offset;
  size_t scratch_bytes;
} ccv_nnc_mfa_segmented_scaled_gemm_plan_layout_t;

typedef struct {
  size_t activation_offset;
  size_t scale_offset;
  size_t scratch_bytes;
} ccv_nnc_mfa_segmented_int8_matmul_layout_t;

static constexpr uint32_t kSegmentedScaledGEMMBlockM = 128;

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

static size_t rowwise_8i_scale_offset(const size_t rows, const size_t cols) noexcept
{
  return align_up(rows * cols * sizeof(int8_t), 128);
}

static ccv_nnc_mfa_segmented_int8_matmul_layout_t int8_matmul_layout(ccv_nnc_mfa_segmented_scaled_gemm_params_t params) noexcept
{
  const size_t activation_offset = 0;
  const size_t activation_bytes = (size_t)params.originalM * params.K * sizeof(uint16_t);
  const size_t scale_offset = align_up(activation_offset + activation_bytes, 128);
  const size_t scale_bytes = (size_t)params.originalM * sizeof(float);
  return (ccv_nnc_mfa_segmented_int8_matmul_layout_t){
    .activation_offset = activation_offset,
    .scale_offset = scale_offset,
    .scratch_bytes = scale_offset + scale_bytes,
  };
}

static ccv_nnc_mfa_segmented_scaled_gemm_plan_layout_t plan_layout(ccv_nnc_mfa_segmented_scaled_gemm_params_t params) noexcept
{
  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  const size_t records_offset = align_up(a_layout.scratch_bytes, 256);
  const size_t records_per_bin = (params.originalM + kSegmentedScaledGEMMBlockM - 1) / kSegmentedScaledGEMMBlockM;
  const size_t records = (size_t)params.bincount * (records_per_bin > 0 ? records_per_bin : (size_t)1);
  const size_t records_bytes = align_up(records * sizeof(simd::uint4), 256);
  const size_t dispatch_offset = records_offset + records_bytes;
  return (ccv_nnc_mfa_segmented_scaled_gemm_plan_layout_t){
    .records_offset = records_offset,
    .dispatch_offset = dispatch_offset,
    .scratch_bytes = align_up(dispatch_offset + sizeof(simd::uint4), 256),
  };
}

}

void ccv_nnc_mfa_prepare_segmented_scaled_gemm(mfa::context* context, ccv_nnc_mfa_segmented_scaled_gemm_params_t params)
{
  (void)context;
  (void)params;
}

size_t ccv_nnc_mfa_segmented_scaled_gemm_reserved_scratch_size(ccv_nnc_mfa_segmented_scaled_gemm_params_t params)
{
  if (!params.use_neural_accelerators) {
    CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat && !params.fused_bias);
    return int8_matmul_layout(params).scratch_bytes;
  }
  return plan_layout(params).scratch_bytes;
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
  if (!params.use_neural_accelerators) {
    CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat && !params.fused_bias && num_tensors == 5);
    CCV_NNC_MFA_PRECONDITION((params.N % 8) == 0 && (params.K % 8) == 0);
    CCV_NNC_MFA_PRECONDITION((uint64_t)params.expert_count * params.N * params.K <= UINT32_MAX);
    const Int8MatMulDescriptor descriptor = {
      .M = params.originalM,
      .N = params.N,
      .K = params.K,
      .expertCount = params.expert_count,
      .binCount = params.bincount,
      .operation = Int8MatMulQuantizeActivation,
    };
    auto int8_matmul_pipeline = [&](const Int8MatMulOperation operation) {
      Int8MatMulDescriptor operation_descriptor = descriptor;
      operation_descriptor.operation = operation;
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto pipeline = context->kernel_cache.findKernel<Int8MatMulKernel, Int8MatMulDescriptor, Int8MatMulKernelDescriptor>(
        operation_descriptor, context->device.get(), DeviceProperties());
      pool->drain();
      return pipeline;
    };
    auto quantize_pipeline = int8_matmul_pipeline(Int8MatMulQuantizeActivation);
    auto segmented_pipeline = int8_matmul_pipeline(Int8MatMulSegmented);
    auto dequantize_pipeline = int8_matmul_pipeline(Int8MatMulDequantizeSegmentedOutput);
    const ccv_nnc_mfa_segmented_int8_matmul_layout_t layout = int8_matmul_layout(params);
    auto scratch = context->request_scratch(layout.scratch_bytes);
    const size_t weight_scale_offset = rowwise_8i_scale_offset((size_t)params.expert_count * params.N, params.K);
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(quantize_pipeline->pipeline.get());
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
      encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
      encoder->setBuffer(scratch, layout.activation_offset, 1);
      encoder->setBuffer(scratch, layout.scale_offset, 2);
      encoder->dispatchThreadgroups(MTL::Size(params.originalM, 1, 1), quantize_pipeline->kernel->threadgroupSize);
      command_batch->finishCommand(encoder);
    }
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(segmented_pipeline->pipeline.get());
      encoder->useResource(scratch, MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
      encoder->setBuffer(scratch, layout.activation_offset, 0);
      encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
      encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
      encoder->setBuffer(tensors[4], tensor_offsets[4], 4);
      encoder->dispatchThreadgroups(MTL::Size((params.N + 7) / 8, params.bincount, 1), MTL::Size(32, 1, 1));
      command_batch->finishCommand(encoder);
    }
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(dequantize_pipeline->pipeline.get());
      encoder->useResource(tensors[4], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->useResource(scratch, MTL::ResourceUsageRead);
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->setBuffer(tensors[4], tensor_offsets[4], 0);
      encoder->setBuffer(scratch, layout.scale_offset, 1);
      encoder->setBuffer(tensors[3], tensor_offsets[3] + weight_scale_offset, 2);
      encoder->setBuffer(tensors[1], tensor_offsets[1], 3);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 4);
      encoder->dispatchThreadgroups(
        MTL::Size((params.N + 255) / 256, params.bincount, 1),
        dequantize_pipeline->kernel->threadgroupSize);
      command_batch->finishCommand(encoder);
    }
    return;
  }

  NAInt8MatMulDescriptor quantizeDesc;
  quantizeDesc.batchDimension = 1;
  quantizeDesc.ioPrecision = io_precision(params.data_type);
  quantizeDesc.matrixDimensions = simd::uint3 { params.originalM, params.N, params.K };
  quantizeDesc.batchStrides = std::nullopt;
  quantizeDesc.useBias = params.fused_bias;
  quantizeDesc.loadM = params.loadM;
  quantizeDesc.supportIndirectCommandBuffers = false;

  SegmentedScaledGEMMDescriptor segmentedDesc;
  segmentedDesc.ioPrecision = io_precision(params.data_type);
  segmentedDesc.matrixDimensions = simd::uint3 { params.originalM, params.N, params.K };
  segmentedDesc.expertCount = params.expert_count;
  segmentedDesc.binCount = params.bincount;
  segmentedDesc.useBias = params.fused_bias;
  segmentedDesc.loadM = params.loadM;

  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto quantizePipelineValue = shaderCache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(quantizeDesc, context->device.get(), dprops);
  auto quantizeKernel = quantizePipelineValue->kernel;
  auto quantizePipeline = quantizePipelineValue->second;
  auto segmentedPipelineValue = shaderCache.findKernel<SegmentedScaledGEMMKernel, SegmentedScaledGEMMDescriptor, SegmentedScaledGEMMKernelDescriptor>(segmentedDesc, context->device.get(), dprops);
  auto segmentedKernel = segmentedPipelineValue->kernel;
  auto segmentedPipeline = segmentedPipelineValue->pipeline;
  auto planPipeline = segmentedPipelineValue->second;

  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  const ccv_nnc_mfa_segmented_scaled_gemm_plan_layout_t p_layout = plan_layout(params);
  auto scratch = context->request_scratch(p_layout.scratch_bytes);
  const size_t b_scale_offset = rowwise_8i_scale_offset((size_t)params.expert_count * params.N, params.K);

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(quantizePipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBuffer(scratch, a_layout.scale_offset, 2);
    if (quantizeDesc.loadM)
      encoder->setBytes(&params.originalM, sizeof(params.originalM), 3);
    encoder->dispatchThreadgroups(
        MTL::Size(params.originalM, 1, 1),
        MTL::Size(quantizeKernel->activationQuantizeThreads, 1, 1));
    command_batch->finishCommand(encoder);
  }

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(planPipeline.get());
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 0);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 1);
    encoder->setBuffer(scratch, p_layout.records_offset, 2);
    encoder->setBuffer(scratch, p_layout.dispatch_offset, 3);
    if (segmentedDesc.loadM) {
      const uint32_t maxTileRecords = segmentedKernel->maxTileRecords(params.originalM, params.bincount);
      encoder->setBytes(&maxTileRecords, sizeof(maxTileRecords), 4);
    }
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
    command_batch->finishCommand(encoder);
  }

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(segmentedPipeline.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
    if (num_tensors >= 6)
      encoder->useResource(tensors[5], MTL::ResourceUsageRead);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(tensors[3], tensor_offsets[3], 1);
    encoder->setBuffer(tensors[4], tensor_offsets[4], 2);
    encoder->setBuffer(scratch, a_layout.scale_offset, 3);
    encoder->setBuffer(tensors[3], tensor_offsets[3] + b_scale_offset, 4);
    encoder->setBuffer(scratch, p_layout.records_offset, 5);
    if (num_tensors >= 6)
      encoder->setBuffer(tensors[5], tensor_offsets[5], 6);
    encoder->dispatchThreadgroups(
        scratch,
        p_layout.dispatch_offset,
        MTL::Size(segmentedKernel->threadgroupSize(segmentedPipeline.get()), 1, 1));
    command_batch->finishCommand(encoder);
  }
}
