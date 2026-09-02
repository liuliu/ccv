#include "ccv_nnc_mfa.hpp"
using namespace ccv::nnc;

#include "kernels/Int8SwiGLUDescriptor.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"
#include "kernels/NAInt8MatMulKernelDescriptor.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"
#include "kernels/Int8MatMulDescriptor.hpp"
#include "kernels/Int8MatMulKernel.hpp"

namespace {

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
  return (value + alignment - 1) & ~(alignment - 1);
}

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

typedef struct {
  size_t scale_offset;
  size_t scratch_bytes;
} ccv_nnc_mfa_swiglu_activation_layout_t;

typedef struct {
  size_t activation_offset;
  size_t scale_offset;
  size_t weight_offset;
  size_t intermediate_offset;
  size_t scratch_bytes;
} ccv_nnc_mfa_swiglu_int8_matmul_layout_t;

static ccv_nnc_mfa_swiglu_activation_layout_t activation_layout(
  const ccv_nnc_mfa_scaled_swiglu_params_t params) noexcept
{
  const size_t quantized_bytes = (size_t)params.M * params.K * sizeof(int8_t);
  const size_t scale_offset = align_up(quantized_bytes, 256);
  const size_t scale_bytes =
    (size_t)params.M * io_precision(params.data_type).size();
  return (ccv_nnc_mfa_swiglu_activation_layout_t){
    .scale_offset = scale_offset,
    .scratch_bytes = align_up(scale_offset + scale_bytes, 256),
  };
}

static size_t rowwise_8i_scale_offset(
  const size_t rows, const size_t cols) noexcept
{
  return align_up(rows * cols * sizeof(int8_t), 128);
}

static ccv_nnc_mfa_gemm_params_t int8_matmul_gemm_params(
  const ccv_nnc_mfa_scaled_swiglu_params_t params) noexcept
{
  return (ccv_nnc_mfa_gemm_params_t){
    .data_type = MTL::DataTypeHalf,
    .M = params.M,
    .N = params.N,
    .K = params.K,
    .A_trans = 0,
    .B_trans = 1,
    .D_trans = 0,
    .fused_bias = 0,
    .register_float = 1,
    .use_neural_accelerators = 0,
    .batch_dimension = 1,
    .loadM = params.loadM,
    .output_data_type = MTL::DataTypeFloat,
  };
}

static ccv_nnc_mfa_swiglu_int8_matmul_layout_t int8_matmul_layout(
  const ccv_nnc_mfa_scaled_swiglu_params_t params) noexcept
{
  const ccv_nnc_mfa_gemm_params_t gemm_params =
    int8_matmul_gemm_params(params);
  const size_t activation_offset = align_up(
    ccv_nnc_mfa_gemm_reserved_scratch_size(gemm_params), 128);
  const size_t activation_bytes =
    (size_t)params.M * params.K * sizeof(uint16_t);
  const size_t scale_offset = align_up(
    activation_offset + activation_bytes, 128);
  const size_t scale_bytes = (size_t)params.M * sizeof(float);
  const size_t weight_offset = align_up(scale_offset + scale_bytes, 128);
  const size_t weight_bytes =
    (size_t)params.N * params.K * sizeof(uint16_t);
  const size_t intermediate_offset = align_up(
    weight_offset + weight_bytes, 128);
  const size_t intermediate_bytes =
    (size_t)params.M * params.N * sizeof(float);
  return (ccv_nnc_mfa_swiglu_int8_matmul_layout_t){
    .activation_offset = activation_offset,
    .scale_offset = scale_offset,
    .weight_offset = weight_offset,
    .intermediate_offset = intermediate_offset,
    .scratch_bytes = intermediate_offset + intermediate_bytes,
  };
}

static ccv_nnc_mfa_swish_mul_params_t swish_params(
  const ccv_nnc_mfa_scaled_swiglu_params_t params) noexcept
{
  return (ccv_nnc_mfa_swish_mul_params_t){
    .beta = 1,
    .scale = 1,
    .clamp = params.clamp,
    .a_data_type = params.data_type,
    .b_data_type = params.data_type,
    .weight_data_type = params.data_type,
    .length = params.M * params.N,
    .weighted = 0,
    .loadM = params.loadM,
  };
}

}

void ccv_nnc_mfa_prepare_int8_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_int8_swiglu_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_int8_swiglu(
  mfa::context* context,
  ccv_nnc_mfa_int8_swiglu_params_t params,
  MTL::CommandBatch* command_batch,
  MTL::Buffer** tensors,
  size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    ++num_tensors;
  CCV_NNC_MFA_PRECONDITION(num_tensors == 4);
  CCV_NNC_MFA_PRECONDITION(params.N > 0 && params.K > 0);
  CCV_NNC_MFA_PRECONDITION((params.K % 4) == 0);

  Int8SwiGLUDescriptor descriptor;
  descriptor.N = params.N;
  descriptor.K = params.K;
  descriptor.clamp = params.clamp;
  descriptor.memoryPrecision = io_precision(params.data_type);
  auto pipelineValue = context->kernel_cache.findKernel<
    Int8SwiGLUKernel, Int8SwiGLUDescriptor, Int8SwiGLUKernelDescriptor>(
      descriptor, context->device.get(), DeviceProperties());
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
  encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
  encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
  encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
  const size_t scale_offset = rowwise_8i_scale_offset(params.N, params.K);
  encoder->setBuffer(tensors[0], tensor_offsets[0] + scale_offset, 4);
  encoder->setBuffer(tensors[1], tensor_offsets[1] + scale_offset, 5);
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
  encoder->dispatchThreadgroups(
    MTL::Size(
      (params.N + kInt8SwiGLURowsPerThreadgroup - 1) /
        kInt8SwiGLURowsPerThreadgroup,
      1, 1),
    MTL::Size(kInt8SwiGLUSIMDGroupsPerThreadgroup * 32, 1, 1));
  command_batch->finishCommand(encoder);
}

void ccv_nnc_mfa_prepare_scaled_swiglu(
  mfa::context* context,
  ccv_nnc_mfa_scaled_swiglu_params_t params)
{
  if (params.format) {
    const ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t decode_params = {
      .data_type = params.data_type,
      .format = params.format,
      .row_length = params.K,
      .length = (uint64_t)params.N * params.K,
    };
    ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x(context, decode_params);
  }
  ccv_nnc_mfa_prepare_swish_mul(context, swish_params(params));
}

void ccv_nnc_mfa_encode_scaled_swiglu(
  mfa::context* context,
  ccv_nnc_mfa_scaled_swiglu_params_t params,
  MTL::CommandBatch* command_batch,
  MTL::Buffer** tensors,
  size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    ++num_tensors;
  CCV_NNC_MFA_PRECONDITION(num_tensors == 4);
  CCV_NNC_MFA_PRECONDITION(params.M > 0 && params.N > 0 && params.K > 0);
  CCV_NNC_MFA_PRECONDITION(
    params.format == 0 || ((params.N % 256) == 0 && (params.K % 256) == 0));

  if (!params.use_neural_accelerators) {
    CCV_NNC_MFA_PRECONDITION(
      params.data_type == MTL::DataTypeFloat && params.format == 0 &&
      (params.N % 8) == 0 && (params.K % 8) == 0);
    CCV_NNC_MFA_PRECONDITION((uint64_t)params.N * params.K <= UINT32_MAX);
    CCV_NNC_MFA_PRECONDITION((uint64_t)params.M * params.N <= UINT32_MAX);
    const Int8MatMulDescriptor descriptor = {
      .M = params.M,
      .N = params.N,
      .K = params.K,
      .expertCount = 1,
      .binCount = 0,
      .operation = Int8MatMulQuantizeActivation,
    };
    auto int8_matmul_pipeline = [&](const Int8MatMulOperation operation) {
      Int8MatMulDescriptor operation_descriptor = descriptor;
      operation_descriptor.operation = operation;
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto pipeline = context->kernel_cache.findKernel<
        Int8MatMulKernel, Int8MatMulDescriptor, Int8MatMulKernelDescriptor>(
          operation_descriptor, context->device.get(), DeviceProperties());
      pool->drain();
      return pipeline;
    };
    auto quantize_pipeline = int8_matmul_pipeline(
      Int8MatMulQuantizeActivation);
    auto cast_pipeline = int8_matmul_pipeline(Int8MatMulCastWeights);
    auto dequantize_pipeline = int8_matmul_pipeline(
      Int8MatMulDequantizeOutput);
    const ccv_nnc_mfa_swiglu_int8_matmul_layout_t layout =
      int8_matmul_layout(params);
    MTL::Buffer* const scratch = context->request_scratch(layout.scratch_bytes);
    const size_t weight_scale_offset = rowwise_8i_scale_offset(
      params.N, params.K);
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(quantize_pipeline->pipeline.get());
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
      encoder->setBuffer(scratch, layout.activation_offset, 1);
      encoder->setBuffer(scratch, layout.scale_offset, 2);
      encoder->dispatchThreadgroups(
        MTL::Size(params.M, 1, 1), quantize_pipeline->kernel->threadgroupSize);
      command_batch->finishCommand(encoder);
    }
    const ccv_nnc_mfa_gemm_params_t gemm_params =
      int8_matmul_gemm_params(params);
    for (int projection = 0; projection < 2; ++projection) {
      {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(cast_pipeline->pipeline.get());
        encoder->useResource(tensors[projection], MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageWrite);
        encoder->setBuffer(
          tensors[projection], tensor_offsets[projection], 0);
        encoder->setBuffer(scratch, layout.weight_offset, 1);
        const uint64_t weight_count = (uint64_t)params.N * params.K;
        encoder->dispatchThreadgroups(
          MTL::Size((weight_count + 255) / 256, 1, 1),
          cast_pipeline->kernel->threadgroupSize);
        command_batch->finishCommand(encoder);
      }
      MTL::Buffer* const destination =
        projection == 0 ? scratch : tensors[3];
      const size_t destination_offset = projection == 0 ?
        layout.intermediate_offset : tensor_offsets[3];
      MTL::Buffer* gemm_tensors[4] = {
        scratch, scratch, destination, nullptr,
      };
      size_t gemm_offsets[3] = {
        layout.activation_offset, layout.weight_offset, destination_offset,
      };
      ccv_nnc_mfa_encode_gemm(
        context, gemm_params, command_batch, gemm_tensors, gemm_offsets);
      {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(dequantize_pipeline->pipeline.get());
        encoder->useResource(
          destination, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        encoder->useResource(scratch, MTL::ResourceUsageRead);
        encoder->useResource(tensors[projection], MTL::ResourceUsageRead);
        encoder->setBuffer(destination, destination_offset, 0);
        encoder->setBuffer(scratch, layout.scale_offset, 1);
        encoder->setBuffer(
          tensors[projection],
          tensor_offsets[projection] + weight_scale_offset, 2);
        const uint64_t output_count = (uint64_t)params.M * params.N;
        encoder->dispatchThreadgroups(
          MTL::Size((output_count + 255) / 256, 1, 1),
          dequantize_pipeline->kernel->threadgroupSize);
        command_batch->finishCommand(encoder);
      }
    }
    MTL::Buffer* swish_tensors[4] = {
      tensors[3], scratch, tensors[3], nullptr,
    };
    size_t swish_offsets[3] = {
      tensor_offsets[3], layout.intermediate_offset, tensor_offsets[3],
    };
    ccv_nnc_mfa_encode_swish_mul(
      context, swish_params(params), command_batch,
      swish_tensors, swish_offsets);
    return;
  }

  NAInt8MatMulDescriptor descriptor;
  descriptor.batchDimension = 1;
  descriptor.ioPrecision = io_precision(params.data_type);
  descriptor.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  descriptor.batchStrides = std::nullopt;
  descriptor.useBias = false;
  descriptor.loadM = params.loadM;
  descriptor.supportIndirectCommandBuffers = false;
  auto pipelineValue = context->kernel_cache.findKernel<
    NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(
      descriptor, context->device.get(), DeviceProperties());
  auto* const kernel = pipelineValue->kernel;
  const ccv_nnc_mfa_swiglu_activation_layout_t a_layout =
    activation_layout(params);
  const size_t decoded_weight_offset = a_layout.scratch_bytes;
  const size_t decoded_weight_size = params.format ?
    rowwise_8i_scale_offset(params.N, params.K) +
      (size_t)params.N * descriptor.ioPrecision.size() : 0;
  const size_t intermediate_offset = align_up(
    decoded_weight_offset + decoded_weight_size, 256);
  const size_t intermediate_size =
    (size_t)params.M * params.N * descriptor.ioPrecision.size();
  MTL::Buffer* const scratch = context->request_scratch(
    intermediate_offset + intermediate_size);

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelineValue->second.get());
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBuffer(scratch, a_layout.scale_offset, 2);
    if (descriptor.loadM)
      encoder->setBytes(&params.M, sizeof(params.M), 3);
    encoder->dispatchThreadgroups(
      MTL::Size(params.M, 1, 1),
      MTL::Size(kernel->activationQuantizeThreads, 1, 1));
    command_batch->finishCommand(encoder);
  }

  const ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t decode_params = {
    .data_type = params.data_type,
    .format = params.format,
    .row_length = params.K,
    .length = (uint64_t)params.N * params.K,
  };
  const size_t weight_scale_offset = rowwise_8i_scale_offset(params.N, params.K);
  for (int projection = 0; projection < 2; ++projection) {
    MTL::Buffer* weight = tensors[projection];
    size_t weight_offset = tensor_offsets[projection];
    if (params.format) {
      MTL::Buffer* decode_tensors[3] = { weight, scratch, nullptr };
      size_t decode_offsets[2] = { weight_offset, decoded_weight_offset };
      ccv_nnc_mfa_encode_dequantize_8i_rowwise_x(
        context, decode_params, command_batch, decode_tensors, decode_offsets);
      weight = scratch;
      weight_offset = decoded_weight_offset;
    }
    MTL::Buffer* const destination = projection == 0 ? scratch : tensors[3];
    const size_t destination_offset = projection == 0 ?
      intermediate_offset : tensor_offsets[3];
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelineValue->pipeline.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead |
      (destination == scratch ? MTL::ResourceUsageWrite : 0));
    if (weight != scratch)
      encoder->useResource(weight, MTL::ResourceUsageRead);
    if (destination != scratch)
      encoder->useResource(destination, MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(weight, weight_offset, 1);
    encoder->setBuffer(destination, destination_offset, 2);
    encoder->setBuffer(scratch, a_layout.scale_offset, 3);
    encoder->setBuffer(weight, weight_offset + weight_scale_offset, 4);
    if (descriptor.loadM)
      encoder->setBytes(&params.M, sizeof(params.M), 5);
    encoder->dispatchThreadgroups(
      kernel->threadgroupsPerGrid(params.M, params.N, 1),
      MTL::Size(
        kernel->threadgroupSize(pipelineValue->pipeline.get()), 1, 1));
    command_batch->finishCommand(encoder);
  }

  MTL::Buffer* swish_tensors[4] = {
    tensors[3], scratch, tensors[3], nullptr,
  };
  size_t swish_offsets[3] = {
    tensor_offsets[3], intermediate_offset, tensor_offsets[3],
  };
  ccv_nnc_mfa_encode_swish_mul(
    context, swish_params(params), command_batch, swish_tensors, swish_offsets);
}
