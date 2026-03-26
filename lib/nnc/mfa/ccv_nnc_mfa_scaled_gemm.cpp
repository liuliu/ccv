#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include "kernels/ShaderCache.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"
#include "kernels/NAInt8MatMulKernelDescriptor.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"

namespace {

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

static size_t rowwise_8i_scale_offset(const uint32_t rows, const uint32_t cols) noexcept
{
  return align_up((size_t)rows * cols * sizeof(int8_t), 128);
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
  return activation_quant_layout(params).scratch_bytes;
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
  if (params.batch_dimension > 1) {
    simd::uint4 batchStrides;
    batchStrides[0] = params.batch_stride_a;
    batchStrides[1] = params.batch_stride_b;
    batchStrides[2] = params.batch_stride_c;
    batchStrides[3] = params.batch_stride_d;
    matmulDesc.batchStrides = batchStrides;
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

  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  auto scratch = context->request_scratch(a_layout.scratch_bytes);
  const uint32_t b_batches = (params.batch_dimension > 1 && params.batch_stride_b > 0) ? params.batch_dimension : 1;
  const size_t b_scale_offset = rowwise_8i_scale_offset(b_batches * params.N, params.K);

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(quantizePipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBuffer(scratch, a_layout.scale_offset, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(params.M, 1, params.batch_dimension),
        MTL::Size(kernel->activationQuantizeThreads, 1, 1));
    command_batch->finishCommand(encoder);
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
    encoder->dispatchThreadgroups(
        kernel->threadgroupsPerGrid(params.M, params.N, params.batch_dimension),
        MTL::Size(kernel->threadgroupSize(matmulPipeline.get()), 1, 1));
    command_batch->finishCommand(encoder);
  }
}
