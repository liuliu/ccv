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
} ccv_nnc_mfa_segmented_scaled_swiglu_activation_layout_t;

typedef struct {
  size_t records_offset;
  size_t dispatch_offset;
  size_t scratch_bytes;
} ccv_nnc_mfa_segmented_scaled_swiglu_plan_layout_t;

typedef struct {
  NAInt8MatMulDescriptor quantize_desc;
  SegmentedScaledGEMMDescriptor segmented_desc;
  PipelineValue<NAInt8MatMulKernel>* quantize_pipeline_value;
  PipelineValue<SegmentedScaledGEMMKernel>* segmented_pipeline_value;
} ccv_nnc_mfa_segmented_scaled_swiglu_execution_t;

static constexpr uint32_t kSegmentedScaledSwiGLUBlockM = 128;

static ccv_nnc_mfa_segmented_scaled_swiglu_activation_layout_t activation_layout(
    const ccv_nnc_mfa_segmented_scaled_swiglu_params_t params) noexcept
{
  const size_t quantized_bytes = (size_t)params.M * params.K * sizeof(int8_t);
  const size_t scale_offset = align_up(quantized_bytes, 256);
  const size_t scale_bytes = (size_t)params.M * io_precision(params.data_type).size();
  return (ccv_nnc_mfa_segmented_scaled_swiglu_activation_layout_t){
    .scale_offset = scale_offset,
    .scratch_bytes = align_up(scale_offset + scale_bytes, 256),
  };
}

static size_t rowwise_8i_scale_offset(const size_t rows, const size_t cols) noexcept
{
  return align_up(rows * cols * sizeof(int8_t), 128);
}

static ccv_nnc_mfa_segmented_scaled_swiglu_plan_layout_t plan_layout(
    const ccv_nnc_mfa_segmented_scaled_swiglu_params_t params) noexcept
{
  const ccv_nnc_mfa_segmented_scaled_swiglu_activation_layout_t a_layout =
    activation_layout(params);
  const size_t records_offset = align_up(a_layout.scratch_bytes, 256);
  const size_t records_per_bin =
    (params.M + kSegmentedScaledSwiGLUBlockM - 1) / kSegmentedScaledSwiGLUBlockM;
  const size_t records = (size_t)params.bincount *
    (records_per_bin > 0 ? records_per_bin : (size_t)1);
  const size_t records_bytes = align_up(records * sizeof(simd::uint4), 256);
  const size_t dispatch_offset = records_offset + records_bytes;
  return (ccv_nnc_mfa_segmented_scaled_swiglu_plan_layout_t){
    .records_offset = records_offset,
    .dispatch_offset = dispatch_offset,
    .scratch_bytes = align_up(dispatch_offset + sizeof(simd::uint4), 256),
  };
}

static ccv_nnc_mfa_swish_mul_params_t swish_params(
    const ccv_nnc_mfa_segmented_scaled_swiglu_params_t params) noexcept
{
  return (ccv_nnc_mfa_swish_mul_params_t){
    .beta = 1,
    .scale = 1,
    .clamp = params.clamp,
    .a_data_type = params.data_type,
    .b_data_type = params.data_type,
    .weight_data_type = params.data_type,
    .length = params.M * params.N,
    .weight_count = params.M,
    .weighted = 1,
    .loadM = params.loadM,
  };
}

static ccv_nnc_mfa_segmented_scaled_swiglu_execution_t execution(
    mfa::context* const context,
    const ccv_nnc_mfa_segmented_scaled_swiglu_params_t params)
{
  NAInt8MatMulDescriptor quantize_desc;
  quantize_desc.batchDimension = 1;
  quantize_desc.ioPrecision = io_precision(params.data_type);
  quantize_desc.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  quantize_desc.batchStrides = std::nullopt;
  quantize_desc.useBias = false;
  quantize_desc.loadM = params.loadM;
  quantize_desc.supportIndirectCommandBuffers = false;

  SegmentedScaledGEMMDescriptor segmented_desc;
  segmented_desc.ioPrecision = io_precision(params.data_type);
  segmented_desc.matrixDimensions = simd::uint3 { params.M, params.N, params.K };
  segmented_desc.expertCount = params.expert_count;
  segmented_desc.binCount = params.bincount;
  segmented_desc.useBias = false;
  segmented_desc.loadM = params.loadM;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shader_cache = context->kernel_cache;
  const DeviceProperties device_properties = DeviceProperties();
  auto quantize_pipeline_value = shader_cache.findKernel<
    NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(
      quantize_desc, context->device.get(), device_properties);
  auto segmented_pipeline_value = shader_cache.findKernel<
    SegmentedScaledGEMMKernel, SegmentedScaledGEMMDescriptor,
    SegmentedScaledGEMMKernelDescriptor>(
      segmented_desc, context->device.get(), device_properties);
  pool->drain();
  return (ccv_nnc_mfa_segmented_scaled_swiglu_execution_t){
    .quantize_desc = quantize_desc,
    .segmented_desc = segmented_desc,
    .quantize_pipeline_value = quantize_pipeline_value,
    .segmented_pipeline_value = segmented_pipeline_value,
  };
}

static void encode_prologue(
    const ccv_nnc_mfa_segmented_scaled_swiglu_execution_t& execution,
    const ccv_nnc_mfa_segmented_scaled_swiglu_params_t params,
    MTL::CommandBatch* const command_batch,
    MTL::Buffer* const activation,
    const size_t activation_offset,
    MTL::Buffer* const indices,
    const size_t indices_offset,
    MTL::Buffer* const counts,
    const size_t counts_offset,
    MTL::Buffer* const scratch)
{
  const ccv_nnc_mfa_segmented_scaled_swiglu_activation_layout_t a_layout =
    activation_layout(params);
  const ccv_nnc_mfa_segmented_scaled_swiglu_plan_layout_t p_layout =
    plan_layout(params);

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(execution.quantize_pipeline_value->second.get());
    encoder->useResource(activation, MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(activation, activation_offset, 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBuffer(scratch, a_layout.scale_offset, 2);
    if (execution.quantize_desc.loadM)
      encoder->setBytes(&params.M, sizeof(params.M), 3);
    encoder->dispatchThreadgroups(
      MTL::Size(params.M, 1, 1),
      MTL::Size(execution.quantize_pipeline_value->kernel->activationQuantizeThreads, 1, 1));
    command_batch->finishCommand(encoder);
  }

  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(execution.segmented_pipeline_value->second.get());
    encoder->useResource(indices, MTL::ResourceUsageRead);
    encoder->useResource(counts, MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(indices, indices_offset, 0);
    encoder->setBuffer(counts, counts_offset, 1);
    encoder->setBuffer(scratch, p_layout.records_offset, 2);
    encoder->setBuffer(scratch, p_layout.dispatch_offset, 3);
    if (execution.segmented_desc.loadM) {
      const uint32_t max_tile_records =
        execution.segmented_pipeline_value->kernel->maxTileRecords(params.M, params.bincount);
      encoder->setBytes(&max_tile_records, sizeof(max_tile_records), 4);
    }
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
    command_batch->finishCommand(encoder);
  }
}

static void encode_projection(
    const ccv_nnc_mfa_segmented_scaled_swiglu_execution_t& execution,
    const ccv_nnc_mfa_segmented_scaled_swiglu_params_t params,
    MTL::CommandBatch* const command_batch,
    MTL::Buffer* const weight,
    const size_t weight_offset,
    MTL::Buffer* const destination,
    const size_t destination_offset,
    MTL::Buffer* const scratch)
{
  const ccv_nnc_mfa_segmented_scaled_swiglu_activation_layout_t a_layout =
    activation_layout(params);
  const ccv_nnc_mfa_segmented_scaled_swiglu_plan_layout_t p_layout =
    plan_layout(params);
  const size_t weight_scale_offset = rowwise_8i_scale_offset(
    (size_t)params.expert_count * params.N, params.K);
  auto segmented_kernel = execution.segmented_pipeline_value->kernel;
  auto segmented_pipeline = execution.segmented_pipeline_value->pipeline;

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(segmented_pipeline.get());
  encoder->useResource(scratch, destination == scratch ?
    MTL::ResourceUsageRead | MTL::ResourceUsageWrite : MTL::ResourceUsageRead);
  if (weight != scratch)
    encoder->useResource(weight, MTL::ResourceUsageRead);
  if (destination != scratch)
    encoder->useResource(destination, MTL::ResourceUsageWrite);
  encoder->setBuffer(scratch, 0, 0);
  encoder->setBuffer(weight, weight_offset, 1);
  encoder->setBuffer(destination, destination_offset, 2);
  encoder->setBuffer(scratch, a_layout.scale_offset, 3);
  encoder->setBuffer(weight, weight_offset + weight_scale_offset, 4);
  encoder->setBuffer(scratch, p_layout.records_offset, 5);
  encoder->dispatchThreadgroups(
    scratch,
    p_layout.dispatch_offset,
    MTL::Size(segmented_kernel->threadgroupSize(segmented_pipeline.get()), 1, 1));
  command_batch->finishCommand(encoder);
}

}

void ccv_nnc_mfa_prepare_segmented_scaled_swiglu(
    mfa::context* context,
    ccv_nnc_mfa_segmented_scaled_swiglu_params_t params)
{
  ccv_nnc_mfa_prepare_swish_mul(context, swish_params(params));
}

void ccv_nnc_mfa_encode_segmented_scaled_swiglu(
    mfa::context* context,
    ccv_nnc_mfa_segmented_scaled_swiglu_params_t params,
    MTL::CommandBatch* command_batch,
    MTL::Buffer** tensors,
    size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr)
    ++num_tensors;
  CCV_NNC_MFA_PRECONDITION(num_tensors == 7);
  CCV_NNC_MFA_PRECONDITION(
    params.M > 0 && params.N > 0 && params.K > 0 &&
    params.expert_count > 0 && params.bincount > 0);
  CCV_NNC_MFA_PRECONDITION(
    params.format == 0 || ((params.N % 256) == 0 && (params.K % 256) == 0));

  if (!params.use_neural_accelerators) {
    CCV_NNC_MFA_PRECONDITION(
      params.data_type == MTL::DataTypeFloat && params.format == 0 &&
      (params.N % 8) == 0 && (params.K % 8) == 0);
    CCV_NNC_MFA_PRECONDITION(
      (uint64_t)params.expert_count * params.N * params.K <= UINT32_MAX);
    const Int8MatMulDescriptor descriptor = {
      .M = params.M,
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
      auto pipeline = context->kernel_cache.findKernel<
        Int8MatMulKernel, Int8MatMulDescriptor, Int8MatMulKernelDescriptor>(
          operation_descriptor, context->device.get(), DeviceProperties());
      pool->drain();
      return pipeline;
    };
    auto quantize_pipeline = int8_matmul_pipeline(Int8MatMulQuantizeActivation);
    auto segmented_pipeline = int8_matmul_pipeline(Int8MatMulSegmented);
    auto dequantize_pipeline = int8_matmul_pipeline(Int8MatMulDequantizeSegmentedOutput);
    const size_t activation_offset = 0;
    const size_t activation_bytes = (size_t)params.M * params.K * sizeof(uint16_t);
    const size_t scale_offset = align_up(activation_offset + activation_bytes, 128);
    const size_t scale_bytes = (size_t)params.M * sizeof(float);
    const size_t intermediate_offset = align_up(scale_offset + scale_bytes, 128);
    const size_t intermediate_bytes = (size_t)params.M * params.N * sizeof(float);
    MTL::Buffer* const scratch = context->request_scratch(
      intermediate_offset + intermediate_bytes);
    const size_t weight_scale_offset = rowwise_8i_scale_offset(
      (size_t)params.expert_count * params.N, params.K);
    {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(quantize_pipeline->pipeline.get());
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
      encoder->setBuffer(scratch, activation_offset, 1);
      encoder->setBuffer(scratch, scale_offset, 2);
      encoder->dispatchThreadgroups(
        MTL::Size(params.M, 1, 1), quantize_pipeline->kernel->threadgroupSize);
      command_batch->finishCommand(encoder);
    }
    for (int projection = 0; projection < 2; ++projection) {
      MTL::Buffer* const destination = projection == 0 ? scratch : tensors[6];
      const size_t destination_offset = projection == 0 ?
        intermediate_offset : tensor_offsets[6];
      {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(segmented_pipeline->pipeline.get());
        encoder->useResource(scratch, destination == scratch ?
          MTL::ResourceUsageRead | MTL::ResourceUsageWrite : MTL::ResourceUsageRead);
        encoder->useResource(tensors[3], MTL::ResourceUsageRead);
        encoder->useResource(tensors[4], MTL::ResourceUsageRead);
        encoder->useResource(tensors[projection], MTL::ResourceUsageRead);
        if (destination != scratch)
          encoder->useResource(destination, MTL::ResourceUsageWrite);
        encoder->setBuffer(scratch, activation_offset, 0);
        encoder->setBuffer(tensors[3], tensor_offsets[3], 1);
        encoder->setBuffer(tensors[4], tensor_offsets[4], 2);
        encoder->setBuffer(tensors[projection], tensor_offsets[projection], 3);
        encoder->setBuffer(destination, destination_offset, 4);
        encoder->dispatchThreadgroups(
          MTL::Size((params.N + 7) / 8, params.bincount, 1), MTL::Size(32, 1, 1));
        command_batch->finishCommand(encoder);
      }
      {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(dequantize_pipeline->pipeline.get());
        encoder->useResource(destination, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        if (destination != scratch)
          encoder->useResource(scratch, MTL::ResourceUsageRead);
        encoder->useResource(tensors[projection], MTL::ResourceUsageRead);
        encoder->useResource(tensors[3], MTL::ResourceUsageRead);
        encoder->useResource(tensors[4], MTL::ResourceUsageRead);
        encoder->setBuffer(destination, destination_offset, 0);
        encoder->setBuffer(scratch, scale_offset, 1);
        encoder->setBuffer(
          tensors[projection], tensor_offsets[projection] + weight_scale_offset, 2);
        encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
        encoder->setBuffer(tensors[4], tensor_offsets[4], 4);
        encoder->dispatchThreadgroups(
          MTL::Size((params.N + 255) / 256, params.bincount, 1),
          dequantize_pipeline->kernel->threadgroupSize);
        command_batch->finishCommand(encoder);
      }
    }
    MTL::Buffer* swish_tensors[5] = {
      tensors[6], scratch, tensors[5], tensors[6], nullptr,
    };
    size_t swish_tensor_offsets[4] = {
      tensor_offsets[6], intermediate_offset, tensor_offsets[5], tensor_offsets[6],
    };
    ccv_nnc_mfa_encode_swish_mul(
      context, swish_params(params), command_batch, swish_tensors, swish_tensor_offsets);
    return;
  }

  const ccv_nnc_mfa_segmented_scaled_swiglu_execution_t execution_state =
    execution(context, params);
  const ccv_nnc_mfa_segmented_scaled_swiglu_plan_layout_t p_layout =
    plan_layout(params);
  const size_t row_count = (size_t)params.expert_count * params.N;
  const size_t decoded_weight_size = params.format ?
    rowwise_8i_scale_offset(row_count, params.K) +
      row_count * io_precision(params.data_type).size() : 0;
  ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t decode_params = {};
  size_t decode_scratch_size = 0;
  if (params.format) {
    decode_params = (ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t){
      .data_type = params.data_type,
      .format = params.format,
      .row_length = params.K,
      .rows_per_expert = params.N,
      .expert_count = params.expert_count,
      .bincount = params.bincount,
    };
    decode_scratch_size =
      ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(decode_params);
  }
  const size_t decode_scratch_offset = p_layout.scratch_bytes;
  const size_t decoded_weight_offset = align_up(
    decode_scratch_offset + decode_scratch_size, 256);
  const size_t intermediate_offset = align_up(
    decoded_weight_offset + decoded_weight_size, 256);
  const size_t intermediate_size =
    (size_t)params.M * params.N * io_precision(params.data_type).size();
  MTL::Buffer* const scratch = context->request_scratch(
    intermediate_offset + intermediate_size);

  encode_prologue(
    execution_state, params, command_batch,
    tensors[2], tensor_offsets[2], tensors[3], tensor_offsets[3],
    tensors[4], tensor_offsets[4], scratch);

  for (int projection = 0; projection < 2; ++projection) {
    MTL::Buffer* weight = tensors[projection];
    size_t weight_offset = tensor_offsets[projection];
    if (params.format) {
      MTL::Buffer* decode_tensors[6] = {
        weight, tensors[3], tensors[4], scratch, scratch, nullptr,
      };
      size_t decode_tensor_offsets[5] = {
        weight_offset, tensor_offsets[3], tensor_offsets[4],
        decoded_weight_offset, decode_scratch_offset,
      };
      ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(
        context, decode_params, command_batch, decode_tensors, decode_tensor_offsets);
      weight = scratch;
      weight_offset = decoded_weight_offset;
    }
    MTL::Buffer* const destination = projection == 0 ? scratch : tensors[6];
    const size_t destination_offset = projection == 0 ?
      intermediate_offset : tensor_offsets[6];
    encode_projection(
      execution_state, params, command_batch,
      weight, weight_offset, destination, destination_offset, scratch);
  }

  MTL::Buffer* swish_tensors[5] = {
    tensors[6], scratch, tensors[5], tensors[6], nullptr,
  };
  size_t swish_tensor_offsets[4] = {
    tensor_offsets[6], intermediate_offset, tensor_offsets[5], tensor_offsets[6],
  };
  ccv_nnc_mfa_encode_swish_mul(
    context, swish_params(params), command_batch, swish_tensors, swish_tensor_offsets);
}
