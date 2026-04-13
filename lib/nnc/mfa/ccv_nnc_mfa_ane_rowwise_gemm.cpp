#include "ccv.h"
#include "ccv_nnc_mfa_ane_rowwise_internal.hpp"
#include "ccv_nnc_mfa_ane_rowwise_gemm.hpp"
#include "ccv_nnc_mfa_ane_rowwise_coreml.hpp"
#include "ccv_nnc_mfa_error.hpp"

#include "kernels/ANERowwiseTransformDescriptor.hpp"
#include "kernels/ANERowwiseTransformKernel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

using namespace ccv::nnc;

typedef struct ccv_nnc_stream_context_s ccv_nnc_stream_context_t;

extern "C" {
void ccv_nnc_mfa_log_message(const char* message);
mtl_command_batch_t* ccv_nnc_stream_context_start_command_batch(ccv_nnc_stream_context_t* const stream_context);
}

namespace {

constexpr uint64_t kPrivateQuantCommitActivationElementsThreshold = (1ULL << 22);
constexpr uint32_t kANERowAlignment = 128;

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
  return (value + alignment - 1) & ~(alignment - 1);
}

static void log_ane_rowwise_error(ccv_nnc_mfa_context_t* const context, const std::string& error)
{
  if (error.empty())
    return;
  if (METAL_LOG_LEVEL(context) >= 1)
    ccv_nnc_mfa_log_message(error.c_str());
}

static std::string bridge_error(const char* const error_buffer)
{
  return (error_buffer && error_buffer[0]) ? std::string(error_buffer) : std::string();
}

static uint32_t pad_ane_rows(const uint32_t rows)
{
  return (uint32_t)align_up(rows, kANERowAlignment);
}

static uint32_t rowwise_batch_dimension(const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  return params.batch_dimension ? params.batch_dimension : 1;
}

static uint32_t rowwise_total_rows(const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  return params.M * rowwise_batch_dimension(params);
}

static uint32_t rowwise_padded_total_rows(const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  return pad_ane_rows(rowwise_total_rows(params));
}

static size_t rowwise_8i_scale_offset(const uint32_t rows, const uint32_t cols)
{
  return align_up((size_t)rows * cols * sizeof(int8_t), 128);
}

static ccv_nnc_mfa_ane_rowwise_coreml_cache_t* get_or_create_cache(
    ccv_nnc_mfa_context_t* const context,
    std::string* const error_out)
{
  if (ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context))
    return (ccv_nnc_mfa_ane_rowwise_coreml_cache_t*)ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context);
  auto* const cache = ccv_nnc_mfa_ane_rowwise_coreml_cache_create(ccv_nnc_mfa_context_device(context));
  if (!cache) {
    if (error_out)
      *error_out = "failed to create ANE rowwise CoreML cache";
    return nullptr;
  }
  ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(context, cache);
  return cache;
}

static PipelineValue<ANERowwiseTransformKernel>* find_transform_pipeline(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  ANERowwiseTransformDescriptor descriptor;
  if (params.data_type == MTL::DataTypeHalf) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  } else {
    CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeBFloat);
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  }
  descriptor.M = params.M;
  descriptor.paddedM = rowwise_padded_total_rows(params);
  descriptor.batchDimension = rowwise_batch_dimension(params);
  descriptor.N = params.N;
  descriptor.K = params.K;
  descriptor.batchStrideA = params.batch_stride_a;
  descriptor.batchStrideC = params.batch_stride_c;
  return ccv_nnc_mfa_prepare_ane_rowwise_transform(context, descriptor);
}

static bool ensure_shared_scratch(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache,
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    std::string* const error_out)
{
  char error_buffer[1024] = {};
  const int ok = ccv_nnc_mfa_ane_rowwise_coreml_cache_ensure_scratch(
      cache,
      rowwise_padded_total_rows(params),
      params.N,
      params.K,
      error_buffer,
      sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer);
  return ok != 0;
}

static bool run_quantize_activation(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache,
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline,
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program,
    mtl_buffer_t* const activation,
    const size_t activation_offset,
    mtl_buffer_t* const weight,
    const size_t weight_offset,
    ccv_nnc_stream_context_t* const stream_context,
    std::string* const error_out)
{
  auto* const kernel = transform_pipeline->kernel;
  const uint32_t program_M = ccv_nnc_mfa_ane_rowwise_coreml_program_M(program);
  const uint32_t program_N = ccv_nnc_mfa_ane_rowwise_coreml_program_N(program);
  const uint32_t program_K = ccv_nnc_mfa_ane_rowwise_coreml_program_K(program);
  const MTL::Size activation_scale_grid_size = kernel->activationScaleGridSize(program_M);
  const MTL::Size activation_scale_threadgroup_size = kernel->activationScaleThreadgroupSize();
  const MTL::Size activation_quantize_grid_size = kernel->activationQuantizeGridSize(program_M, program_K);
  const MTL::Size activation_quantize_threadgroup_size = kernel->activationQuantizeThreadgroupSize();

  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(transform_pipeline->pipeline.get());
  encoder->useResource(activation, MTL::ResourceUsageRead);
  encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), MTL::ResourceUsageWrite);
  encoder->setBuffer(activation, activation_offset, 0);
  encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), 0, 1);
  encoder->dispatchThreadgroups(activation_scale_grid_size, activation_scale_threadgroup_size);
  command_batch->finishCommand(encoder);

  encoder = command_batch->startCommand();
  encoder->setComputePipelineState(transform_pipeline->second.get());
  encoder->useResource(activation, MTL::ResourceUsageRead);
  encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), MTL::ResourceUsageRead);
  encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), MTL::ResourceUsageWrite);
  encoder->setBuffer(activation, activation_offset, 0);
  encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), 0, 1);
  encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), 0, 2);
  encoder->dispatchThreadgroups(activation_quantize_grid_size, activation_quantize_threadgroup_size);
  command_batch->finishCommand(encoder);
  char error_buffer[1024] = {};
  const int weight_upload_ok = ccv_nnc_mfa_ane_rowwise_coreml_append_weight_upload(
      cache,
      command_batch,
      weight,
      weight_offset,
      (size_t)program_N * program_K * sizeof(int8_t),
      error_buffer,
      sizeof(error_buffer));
  if (!weight_upload_ok) {
    if (error_out)
      *error_out = bridge_error(error_buffer).empty() ? "failed to append weight upload to quantize command batch" : bridge_error(error_buffer);
    return false;
  }
  const int use_mps_wrapper =
      (stream_context && (uint64_t)program_M * program_K <= kPrivateQuantCommitActivationElementsThreshold) ? 1 : 0;
  const int ok = ccv_nnc_mfa_ane_rowwise_finish_command_batch_and_wait(
      stream_context,
      command_batch,
      use_mps_wrapper,
      error_buffer,
      sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer).empty() ? "activation quantize command failed" : bridge_error(error_buffer);
  return ok != 0;
}

static bool evaluate_program(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache,
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program,
    std::string* const error_out)
{
  char error_buffer[1024] = {};
  const int ok = ccv_nnc_mfa_ane_rowwise_coreml_evaluate(cache, program, error_buffer, sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer);
  return ok != 0;
}

static bool run_dequantize_output(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache,
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline,
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program,
    mtl_buffer_t* const weight_buffer,
    const size_t weight_scale_offset,
    mtl_buffer_t* const bias_buffer,
    const size_t bias_offset,
    mtl_buffer_t* const output,
    const size_t output_offset,
    const uint32_t fused_bias,
    ccv_nnc_stream_context_t* const stream_context,
    std::string* const error_out)
{
  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  auto* const kernel = transform_pipeline->kernel;
  mtl_buffer_t* const coreml_output_buffer = ccv_nnc_mfa_ane_rowwise_coreml_cache_output_surface_buffer(cache);
  if (!coreml_output_buffer) {
    if (error_out)
      *error_out = "CoreML output buffer is not available for dequantize";
    return false;
  }
  encoder->setComputePipelineState(fused_bias ? transform_pipeline->fourth.get() : transform_pipeline->third.get());
  encoder->useResource(coreml_output_buffer, MTL::ResourceUsageRead);
  encoder->useResource(output, MTL::ResourceUsageWrite);
  encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), MTL::ResourceUsageRead);
  encoder->useResource(weight_buffer, MTL::ResourceUsageRead);
  encoder->setBuffer(coreml_output_buffer, 0, 0);
  encoder->setBuffer(output, output_offset, 1);
  encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), 0, 2);
  encoder->setBuffer(weight_buffer, weight_scale_offset, 3);
  if (fused_bias) {
    if (!bias_buffer) {
      if (error_out)
        *error_out = "bias buffer is not available for output bias add";
      return false;
    }
    encoder->useResource(bias_buffer, MTL::ResourceUsageRead);
    encoder->setBuffer(bias_buffer, bias_offset, 4);
  }
  encoder->dispatchThreadgroups(
      kernel->outputDequantizeGridSize(
          ccv_nnc_mfa_ane_rowwise_coreml_program_M(program),
          ccv_nnc_mfa_ane_rowwise_coreml_program_N(program)),
      kernel->outputDequantizeThreadgroupSize());
  command_batch->finishCommand(encoder);
  char error_buffer[1024] = {};
  const int ok = ccv_nnc_mfa_ane_rowwise_finish_command_batch_and_wait(
      stream_context,
      command_batch,
      0,
      error_buffer,
      sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer).empty() ? "output dequantize command failed" : bridge_error(error_buffer);
  return ok != 0;
}

} // namespace

int ccv_nnc_mfa_run_ane_rowwise_gemm(
    ccv_nnc_mfa_context_t* const context,
    ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* const stream_context)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensor_offsets != nullptr);
  std::string error;
  ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache = get_or_create_cache(context, &error);
  if (!cache) {
    log_ane_rowwise_error(context, error);
    return 0;
  }
  mtl_buffer_t* const activation = tensors[0];
  mtl_buffer_t* const weight = tensors[1];
  mtl_buffer_t* const output = tensors[2];
  mtl_buffer_t* const bias = params.fused_bias ? tensors[3] : nullptr;
  PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline = find_transform_pipeline(context, params);
  char error_buffer[1024] = {};
  ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program =
      ccv_nnc_mfa_ane_rowwise_coreml_find_or_create_program(
          rowwise_padded_total_rows(params),
          params.N,
          params.K,
          error_buffer,
          sizeof(error_buffer));
  if (!program) {
    log_ane_rowwise_error(context, bridge_error(error_buffer));
    return 0;
  }
  if (!ensure_shared_scratch(cache, params, &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }
  const size_t weight_scale_offset = tensor_offsets[1] + rowwise_8i_scale_offset(params.N, params.K);
  const size_t bias_offset = params.fused_bias ? tensor_offsets[3] : 0;
  if (!run_quantize_activation(cache, transform_pipeline, program, activation, tensor_offsets[0], weight, tensor_offsets[1], stream_context, &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }
  if (!evaluate_program(cache, program, &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }
  if (!run_dequantize_output(cache, transform_pipeline, program, weight, weight_scale_offset, bias, bias_offset, output, tensor_offsets[2], params.fused_bias, stream_context, &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }
  ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
  return 1;
}

void ccv_nnc_mfa_ane_rowwise_gemm_cleanup(ccv_nnc_mfa_context_t* const context)
{
  if (!context || !ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context))
    return;
  auto* const cache = (ccv_nnc_mfa_ane_rowwise_coreml_cache_t*)ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context);
  ccv_nnc_mfa_ane_rowwise_coreml_cache_destroy(cache);
  ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(context, nullptr);
}
