#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_ane_rowwise_internal.hpp"
#include "ccv_nnc_mfa_ane_rowwise_gemm.hpp"
#include "ccv_nnc_mfa_ane_rowwise_coreml.hpp"
#include "ccv_nnc_mfa_error.hpp"

#include "kernels/ANERowwiseTransformDescriptor.hpp"
#include "kernels/ANERowwiseTransformKernel.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"
#include "kernels/Int8MatMulDescriptor.hpp"
#include "kernels/Int8MatMulKernelDescriptor.hpp"
#include "kernels/Int8MatMulKernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>

using namespace ccv::nnc;

typedef struct ccv_nnc_stream_context_s ccv_nnc_stream_context_t;

extern "C" {
void ccv_nnc_mfa_log_message(const char* message);
mtl_command_batch_t* ccv_nnc_stream_context_start_command_batch(ccv_nnc_stream_context_t* const stream_context);
}

namespace {

constexpr uint64_t kPrivateQuantCommitActivationElementsThreshold = (1ULL << 22);
// The ANE computes rows in 256-row units: an odd multiple of 128 rows costs
// about as much as the next multiple of 256 (measured on M3 Ultra and M4 Pro).
constexpr uint32_t kANERowAlignment = 256;
// Core ML compile time grows with rows. Without neural accelerators, a split
// runs its ANE rows as programs of at most this many rows.
constexpr uint32_t kHybridMaxANEProgramRows = 4096;
constexpr uint32_t kHybridProbeN = 4096;
constexpr uint32_t kHybridProbeK = 4096;
constexpr uint32_t kHybridPrepFenceIndex = 0;

// Each side's kernel takes fixed + rows * N * K * slope seconds, fitted from
// two probed row counts: one size cannot separate the fixed cost from the
// rate, and the fixed cost decides small splits. Elementwise passes move bytes
// at the probed bandwidth, and each ANE program waits for a fence round trip.
typedef struct {
  bool initialized;
  bool valid;
  double ane_fixed;
  double ane_slope;
  double gpu_fixed;
  double gpu_slope;
  double bandwidth;
  double fence_seconds;
} ccv_nnc_mfa_hybrid_cost_model_t;

// One model for each GPU kernel: Int8MatMul, and NAInt8MatMul.
static std::mutex g_hybrid_cost_model_mutex;
static ccv_nnc_mfa_hybrid_cost_model_t g_hybrid_cost_models[2] = {};

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

static size_t rowwise_8i_scale_offset(const size_t rows, const size_t cols)
{
  return align_up(rows * cols * sizeof(int8_t), 128);
}

typedef struct {
  size_t q_bytes;
  size_t scale_offset;
  size_t scale_bytes;
  size_t scratch_bytes;
} ccv_nnc_mfa_activation_quant_layout_t;

static uint32_t rowwise_batch_dimension(const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params)
{
  return params.batch_dimension ? params.batch_dimension : 1;
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

static ccv_nnc_mfa_activation_quant_layout_t activation_quant_layout(
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params) noexcept
{
  const size_t q_bytes = (size_t)rowwise_batch_dimension(params) * params.M * params.K * sizeof(int8_t);
  const size_t scale_offset = align_up(q_bytes, 256);
  const size_t scale_bytes = (size_t)rowwise_batch_dimension(params) * params.M * io_precision(params.data_type).size();
  return (ccv_nnc_mfa_activation_quant_layout_t){
    .q_bytes = q_bytes,
    .scale_offset = scale_offset,
    .scale_bytes = scale_bytes,
    .scratch_bytes = align_up(scale_offset + scale_bytes, 256),
  };
}

// With neural accelerators, the ANE rows copy codes NAInt8MatMul already
// quantized (and rotated, with activation_hadamard_256), so their transform
// does not rotate again.
static ccv_nnc_mfa_ane_rowwise_gemm_params_t ane_params_from_hybrid(
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t ane_m) noexcept
{
  return (ccv_nnc_mfa_ane_rowwise_gemm_params_t){
    .M = ane_m,
    .N = params.N,
    .K = params.K,
    .data_type = (uint32_t)params.data_type,
    .fused_bias = params.fused_bias,
    .batch_dimension = rowwise_batch_dimension(params),
    .batch_stride_a = params.batch_stride_a,
    .batch_stride_c = params.batch_stride_c,
  };
}

static PipelineValue<NAInt8MatMulKernel>* find_na_int8_pipeline(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t matrix_m,
    const std::optional<uint32_t> packed_a_batch_stride,
    const std::optional<uint32_t> a_scale_batch_stride);

static void upload_buffer(
    MTL::CommandQueue* const command_queue,
    MTL::Buffer* const source,
    MTL::Buffer* const destination,
    const size_t size)
{
  auto command_buffer = NS::RetainPtr(command_queue->commandBuffer());
  auto blit = NS::RetainPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

static void fill_scale_bytes(const uint64_t data_type, void* const contents, const uint32_t count)
{
  if (data_type == MTL::DataTypeFloat) {
    float* const values = (float*)contents;
    for (uint32_t i = 0; i < count; ++i)
      values[i] = 1.0f;
    return;
  }
  uint16_t one_bits = 0x3c00;
  if (data_type == MTL::DataTypeBFloat)
    one_bits = 0x3f80;
  uint16_t* const values = (uint16_t*)contents;
  for (uint32_t i = 0; i < count; ++i)
    values[i] = one_bits;
}

static bool probe_na_int8_matmul_seconds(
    ccv_nnc_mfa_context_t* const context,
    const uint32_t M,
    double* const seconds_out,
    std::string* const error_out)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  constexpr uint32_t N = kHybridProbeN;
  constexpr uint32_t K = kHybridProbeK;
  constexpr uint32_t kWarmupIterations = 2;
  constexpr uint32_t kTimedIterations = 3;
  constexpr uint32_t kDispatchPerIteration = 2;
  constexpr uint64_t data_type = MTL::DataTypeHalf;
  const GEMMOperandPrecision precision = io_precision(data_type);
  const size_t a_q_bytes = (size_t)M * K * sizeof(int8_t);
  const size_t a_scale_offset = align_up(a_q_bytes, 256);
  const size_t a_scale_bytes = (size_t)M * precision.size();
  const size_t a_scratch_bytes = align_up(a_scale_offset + a_scale_bytes, 256);
  const size_t b_q_bytes = (size_t)N * K * sizeof(int8_t);
  const size_t b_scale_offset = rowwise_8i_scale_offset(N, K);
  const size_t b_scale_bytes = (size_t)N * precision.size();
  const size_t b_buffer_bytes = align_up(b_scale_offset + b_scale_bytes, 256);
  const size_t c_bytes = (size_t)M * N * precision.size();

  ccv_nnc_mfa_ane_na_rowwise_gemm_params_t probe_params = {
    .data_type = data_type,
    .M = M,
    .N = N,
    .K = K,
    .fused_bias = 0,
    .batch_dimension = 1,
    .batch_stride_a = 0,
    .batch_stride_b = 0,
    .batch_stride_c = 0,
    .batch_stride_d = 0,
  };
  PipelineValue<NAInt8MatMulKernel>* const pipeline =
      find_na_int8_pipeline(context, probe_params, M, std::nullopt, std::nullopt);
  if (!pipeline || !pipeline->kernel || !pipeline->pipeline) {
    if (error_out)
      *error_out = "failed to prepare NAInt8MatMul pipeline for hybrid split probe";
    return false;
  }

  auto command_queue = NS::TransferPtr(context->device->newCommandQueue());
  if (!command_queue) {
    if (error_out)
      *error_out = "failed to create command queue for hybrid split probe";
    return false;
  }

  auto a_stage = NS::TransferPtr(context->device->newBuffer(a_scratch_bytes,
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto b_stage = NS::TransferPtr(context->device->newBuffer(b_buffer_bytes,
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto a_buffer = NS::TransferPtr(context->device->newBuffer(a_scratch_bytes,
      MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked));
  auto b_buffer = NS::TransferPtr(context->device->newBuffer(b_buffer_bytes,
      MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked));
  auto c_buffer = NS::TransferPtr(context->device->newBuffer(c_bytes,
      MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked));
  if (!a_stage || !b_stage || !a_buffer || !b_buffer || !c_buffer) {
    if (error_out)
      *error_out = "failed to allocate buffers for hybrid split probe";
    return false;
  }

  std::memset(a_stage->contents(), 0, a_scratch_bytes);
  std::memset(b_stage->contents(), 0, b_buffer_bytes);
  std::memset(a_stage->contents(), 1, a_q_bytes);
  std::memset(b_stage->contents(), 1, b_q_bytes);
  fill_scale_bytes(data_type, (uint8_t*)a_stage->contents() + a_scale_offset, M);
  fill_scale_bytes(data_type, (uint8_t*)b_stage->contents() + b_scale_offset, N);
  upload_buffer(command_queue.get(), a_stage.get(), a_buffer.get(), a_scratch_bytes);
  upload_buffer(command_queue.get(), b_stage.get(), b_buffer.get(), b_buffer_bytes);

  const auto run_once = [&]() {
    auto* const command_batch = ccv_nnc_start_command_batch(command_queue.get());
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline->pipeline.get());
    encoder->useResource(a_buffer.get(), MTL::ResourceUsageRead);
    encoder->useResource(b_buffer.get(), MTL::ResourceUsageRead);
    encoder->useResource(c_buffer.get(), MTL::ResourceUsageWrite);
    encoder->setBuffer(a_buffer.get(), 0, 0);
    encoder->setBuffer(b_buffer.get(), 0, 1);
    encoder->setBuffer(c_buffer.get(), 0, 2);
    encoder->setBuffer(a_buffer.get(), a_scale_offset, 3);
    encoder->setBuffer(b_buffer.get(), b_scale_offset, 4);
    for (uint32_t i = 0; i < kDispatchPerIteration; i++) {
      encoder->dispatchThreadgroups( pipeline->kernel->threadgroupsPerGrid(M, N, 1), pipeline->kernel->threadsPerThreadgroup(pipeline->pipeline.get()));
    }
    command_batch->finishCommand(encoder);
    auto command_buffer = NS::RetainPtr(command_batch->commandBuffer);
    ccv_nnc_finish_command_batch(command_batch);
    command_buffer->waitUntilCompleted();
    return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
  };

  for (uint32_t i = 0; i < kWarmupIterations; ++i)
    (void)run_once();
  double best_seconds = 0;
  for (uint32_t i = 0; i < kTimedIterations; ++i) {
    const double seconds = run_once();
    if (i == 0 || seconds < best_seconds)
      best_seconds = seconds;
  }
  if (!(best_seconds > 0)) {
    if (error_out)
      *error_out = "hybrid split probe produced invalid GPU timing";
    return false;
  }
  *seconds_out = best_seconds / kDispatchPerIteration;
  return true;
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
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    const uint32_t source_row_offset = 0,
    const uint32_t output_row_offset = 0,
    const uint32_t activation_scale_batch_stride = 0,
    const uint32_t activation_scale_row_offset = 0)
{
  ANERowwiseTransformDescriptor descriptor;
  descriptor.memoryPrecision = io_precision(params.data_type);
  descriptor.activationHadamard256 = params.activation_hadamard_256;
  descriptor.M = params.M;
  descriptor.paddedM = rowwise_padded_total_rows(params);
  descriptor.batchDimension = rowwise_batch_dimension(params);
  descriptor.N = params.N;
  descriptor.K = params.K;
  descriptor.batchStrideA = params.batch_stride_a;
  descriptor.batchStrideC = params.batch_stride_c;
  descriptor.sourceRowOffset = source_row_offset;
  descriptor.outputRowOffset = output_row_offset;
  descriptor.activationScaleBatchStride = activation_scale_batch_stride ? activation_scale_batch_stride : params.M;
  descriptor.activationScaleRowOffset = activation_scale_row_offset;
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
    std::string* const error_out,
    const uint32_t prep_fence_value = 0)
{
  auto* const kernel = transform_pipeline->kernel;
  const uint32_t program_M = ccv_nnc_mfa_ane_rowwise_coreml_program_M(program);
  const uint32_t program_N = ccv_nnc_mfa_ane_rowwise_coreml_program_N(program);
  const uint32_t program_K = ccv_nnc_mfa_ane_rowwise_coreml_program_K(program);
  const bool fused = program_K <= 8192;
  const bool row_prepare = kernel->activationHadamard256 || fused;
  const MTL::Size activation_prepare_grid_size = row_prepare ? kernel->activationPrepareGridSize(program_M, program_K) : MTL::Size(program_M, 1, 1);
  const MTL::Size activation_prepare_threadgroup_size = row_prepare ? kernel->activationPrepareThreadgroupSize(program_K) : MTL::Size(256, 1, 1);

  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(transform_pipeline->pipeline.get());
  encoder->useResource(activation, MTL::ResourceUsageRead);
  encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), MTL::ResourceUsageWrite);
  encoder->setBuffer(activation, activation_offset, 0);
  encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), 0, 1);
  if (fused) {
    encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), MTL::ResourceUsageWrite);
    encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), 0, 2);
  }
  encoder->dispatchThreadgroups(activation_prepare_grid_size, activation_prepare_threadgroup_size);
  command_batch->finishCommand(encoder);
  if (!fused) {
    encoder = command_batch->startCommand();
    encoder->setComputePipelineState(transform_pipeline->second.get());
    encoder->useResource(activation, MTL::ResourceUsageRead);
    encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), MTL::ResourceUsageRead);
    encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), MTL::ResourceUsageWrite);
    encoder->setBuffer(activation, activation_offset, 0);
    encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), 0, 1);
    encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), 0, 2);
    encoder->dispatchThreadgroups(kernel->activationQuantizeGridSize(program_M, program_K), kernel->activationQuantizeThreadgroupSize());
    command_batch->finishCommand(encoder);
  }
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
  // A split signals the CPU with a fence when the ANE inputs are ready, and
  // leaves the stream free for the GPU rows.
  if (prep_fence_value) {
    const int ok =
        ccv_nnc_mfa_ane_rowwise_fast_fence_append_update(cache, command_batch, kHybridPrepFenceIndex, prep_fence_value, error_buffer, sizeof(error_buffer)) &&
        ccv_nnc_mfa_ane_rowwise_finish_command_batch_async(stream_context, command_batch, error_buffer, sizeof(error_buffer));
    if (!ok && error_out)
      *error_out = bridge_error(error_buffer).empty() ? "split GEMM prep command failed" : bridge_error(error_buffer);
    return ok != 0;
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
    mtl_buffer_t* const activation_scales_buffer,
    const size_t activation_scales_offset,
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
  encoder->useResource(activation_scales_buffer, MTL::ResourceUsageRead);
  encoder->useResource(weight_buffer, MTL::ResourceUsageRead);
  encoder->setBuffer(coreml_output_buffer, 0, 0);
  encoder->setBuffer(output, output_offset, 1);
  encoder->setBuffer(activation_scales_buffer, activation_scales_offset, 2);
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
  const int ok = ccv_nnc_mfa_ane_rowwise_finish_command_batch_async(
      stream_context,
      command_batch,
      error_buffer,
      sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer).empty() ? "output dequantize command failed" : bridge_error(error_buffer);
  return ok != 0;
}

static PipelineValue<NAInt8MatMulKernel>* find_na_int8_pipeline(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t matrix_m,
    const std::optional<uint32_t> packed_a_batch_stride = std::nullopt,
    const std::optional<uint32_t> a_scale_batch_stride = std::nullopt)
{
  NAInt8MatMulDescriptor descriptor;
  descriptor.batchDimension = rowwise_batch_dimension(params);
  descriptor.ioPrecision = io_precision(params.data_type);
  descriptor.matrixDimensions = simd::uint3 { matrix_m, params.N, params.K };
  if (descriptor.batchDimension > 1) {
    descriptor.batchStrides = simd::uint4 {
      params.batch_stride_a,
      params.batch_stride_b,
      params.batch_stride_c,
      params.batch_stride_d,
    };
  } else {
    descriptor.batchStrides = std::nullopt;
  }
  descriptor.packedABatchStride = packed_a_batch_stride;
  descriptor.aScaleBatchStride = a_scale_batch_stride;
  descriptor.useBias = params.fused_bias != 0;
  descriptor.activationHadamard256 = params.activation_hadamard_256 != 0;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  const DeviceProperties& dprops = context->device_properties;
  auto pipeline = shaderCache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(
      descriptor,
      context->device.get(),
      dprops);
  pool->drain();
  return pipeline;
}

static bool run_hybrid_prepare_inputs(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache,
    PipelineValue<NAInt8MatMulKernel>* const quantize_pipeline,
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline,
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    mtl_buffer_t* const activation,
    const size_t activation_offset,
    mtl_buffer_t* const weight,
    const size_t weight_offset,
    ccv_nnc_stream_context_t* const stream_context,
    mtl_buffer_t* const scratch,
    const ccv_nnc_mfa_activation_quant_layout_t a_layout,
    const uint32_t fast_fence_value,
    std::string* const error_out)
{
  auto* const quantize_kernel = quantize_pipeline->kernel;
  auto* const transform_kernel = transform_pipeline->kernel;
  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(quantize_pipeline->second.get());
  encoder->useResource(activation, MTL::ResourceUsageRead);
  encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  encoder->setBuffer(activation, activation_offset, 0);
  encoder->setBuffer(scratch, 0, 1);
  encoder->setBuffer(scratch, a_layout.scale_offset, 2);
  encoder->dispatchThreadgroups(
      MTL::Size(params.M, 1, rowwise_batch_dimension(params)),
      MTL::Size(quantize_kernel->activationQuantizeThreads, 1, 1));
  command_batch->finishCommand(encoder);

  encoder = command_batch->startCommand();
  encoder->setComputePipelineState(transform_pipeline->fifth.get());
  encoder->useResource(scratch, MTL::ResourceUsageRead);
  encoder->useResource(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), MTL::ResourceUsageWrite);
  encoder->setBuffer(scratch, 0, 0);
  encoder->setBuffer(ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(cache), 0, 1);
  encoder->dispatchThreadgroups(
      transform_kernel->activationQuantizeGridSize(
          ccv_nnc_mfa_ane_rowwise_coreml_program_M(program),
          ccv_nnc_mfa_ane_rowwise_coreml_program_K(program)),
      transform_kernel->activationQuantizeThreadgroupSize());
  command_batch->finishCommand(encoder);

  char error_buffer[1024] = {};
  const int weight_upload_ok = ccv_nnc_mfa_ane_rowwise_coreml_append_weight_upload(
      cache,
      command_batch,
      weight,
      weight_offset,
      (size_t)ccv_nnc_mfa_ane_rowwise_coreml_program_N(program) *
          ccv_nnc_mfa_ane_rowwise_coreml_program_K(program) * sizeof(int8_t),
      error_buffer,
      sizeof(error_buffer));
  if (!weight_upload_ok) {
    if (error_out)
      *error_out = bridge_error(error_buffer).empty() ? "failed to append weight upload to split GEMM prep batch" : bridge_error(error_buffer);
    return false;
  }
  const int fence_ok = ccv_nnc_mfa_ane_rowwise_fast_fence_append_update(
      cache,
      command_batch,
      kHybridPrepFenceIndex,
      fast_fence_value,
      error_buffer,
      sizeof(error_buffer));
  if (!fence_ok) {
    if (error_out)
      *error_out = bridge_error(error_buffer).empty() ? "failed to append split GEMM prep fence update" : bridge_error(error_buffer);
    return false;
  }
  const int ok = ccv_nnc_mfa_ane_rowwise_finish_command_batch_async(
      stream_context,
      command_batch,
      error_buffer,
      sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer).empty() ? "split GEMM prep command failed" : bridge_error(error_buffer);
  return ok != 0;
}

static bool run_partial_na_matmul_async(
    PipelineValue<NAInt8MatMulKernel>* const matmul_pipeline,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t na_m,
    mtl_buffer_t* const scratch,
    const ccv_nnc_mfa_activation_quant_layout_t a_layout,
    mtl_buffer_t* const weight,
    const size_t weight_offset,
    const size_t weight_scale_offset,
    mtl_buffer_t* const output,
    const size_t output_offset,
    mtl_buffer_t* const bias,
    const size_t bias_offset,
    ccv_nnc_stream_context_t* const stream_context,
    std::string* const error_out)
{
  auto* const kernel = matmul_pipeline->kernel;
  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(matmul_pipeline->pipeline.get());
  encoder->useResource(scratch, MTL::ResourceUsageRead);
  encoder->useResource(weight, MTL::ResourceUsageRead);
  encoder->useResource(output, MTL::ResourceUsageWrite);
  if (bias)
    encoder->useResource(bias, MTL::ResourceUsageRead);
  encoder->setBuffer(scratch, 0, 0);
  encoder->setBuffer(weight, weight_offset, 1);
  encoder->setBuffer(output, output_offset, 2);
  encoder->setBuffer(scratch, a_layout.scale_offset, 3);
  encoder->setBuffer(weight, weight_scale_offset, 4);
  if (bias)
    encoder->setBuffer(bias, bias_offset, 5);
  encoder->dispatchThreadgroups(
      kernel->threadgroupsPerGrid(na_m, params.N, rowwise_batch_dimension(params)),
      kernel->threadsPerThreadgroup(matmul_pipeline->pipeline.get()));
  command_batch->finishCommand(encoder);

  char error_buffer[1024] = {};
  const int ok = ccv_nnc_mfa_ane_rowwise_finish_command_batch_async(
      stream_context,
      command_batch,
      error_buffer,
      sizeof(error_buffer));
  if (!ok && error_out)
    *error_out = bridge_error(error_buffer).empty() ? "partial NAInt8MatMul command failed" : bridge_error(error_buffer);
  return ok != 0;
}

// Without neural accelerators, the GPU rows run Int8MatMul: direct loads
// (Apple9), K in multiples of 8, and one batch of FP16 or FP32 rows.
static bool int8_matmul_split_supported(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params)
{
  return context->device->supportsFamily(MTL::GPUFamily(1009)) &&
      (params.K % 8) == 0 && rowwise_batch_dimension(params) == 1 && !params.fused_bias &&
      (params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeFloat) &&
      (uint64_t)params.M * params.K <= UINT32_MAX &&
      (uint64_t)params.N * params.K <= UINT32_MAX &&
      (uint64_t)params.M * params.N <= UINT32_MAX;
}

static ccv_nnc_mfa_scaled_gemm_params_t int8_matmul_split_gpu_params(
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t gpu_m)
{
  return (ccv_nnc_mfa_scaled_gemm_params_t){
    .data_type = params.data_type,
    .M = gpu_m,
    .N = params.N,
    .K = params.K,
    .fused_bias = 0,
    .use_neural_accelerators = 0,
    .batch_dimension = 1,
    .activation_hadamard_256 = params.activation_hadamard_256,
  };
}

static bool probe_int8_matmul_seconds(
    ccv_nnc_mfa_context_t* const context,
    const uint32_t M,
    double* const seconds_out,
    std::string* const error_out)
{
  constexpr uint32_t N = kHybridProbeN;
  constexpr uint32_t K = kHybridProbeK;
  constexpr uint32_t kWarmupIterations = 1;
  constexpr uint32_t kTimedIterations = 3;
  const Int8MatMulDescriptor descriptor = {
    .M = M,
    .N = N,
    .K = K,
    .expertCount = 1,
    .binCount = 0,
    .operation = Int8MatMulDense,
    .ioPrecision = GEMMOperandPrecision::FP16,
  };
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto* const pipeline = context->kernel_cache.findKernel<Int8MatMulKernel, Int8MatMulDescriptor, Int8MatMulKernelDescriptor>(
      descriptor, context->device.get(), context->device_properties);
  pool->drain();
  auto command_queue = NS::TransferPtr(context->device->newCommandQueue());
  const size_t b_scale_offset = rowwise_8i_scale_offset(N, K);
  auto a = NS::TransferPtr(context->device->newBuffer((size_t)M * K * sizeof(uint16_t), MTL::ResourceStorageModeShared));
  auto b = NS::TransferPtr(context->device->newBuffer(b_scale_offset + (size_t)N * sizeof(uint16_t), MTL::ResourceStorageModeShared));
  auto c = NS::TransferPtr(context->device->newBuffer((size_t)M * N * sizeof(uint16_t), MTL::ResourceStorageModePrivate));
  if (!pipeline || !command_queue || !a || !b || !c) {
    if (error_out)
      *error_out = "failed to prepare Int8MatMul for hybrid split probe";
    return false;
  }
  // Nonzero activations, as a GEMM sees them.
  std::memset(a->contents(), 0x20, a->length());
  std::memset(b->contents(), 1, b->length());
  const auto run_once = [&]() {
    auto* const command_batch = ccv_nnc_start_command_batch(command_queue.get());
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline->pipeline.get());
    encoder->setBuffer(a.get(), 0, 0);
    encoder->setBuffer(b.get(), 0, 1);
    encoder->setBuffer(c.get(), 0, 2);
    encoder->setBuffer(b.get(), b_scale_offset, 4);
    encoder->dispatchThreadgroups(pipeline->kernel->threadgroupsPerGrid(descriptor), pipeline->kernel->threadgroupSize);
    command_batch->finishCommand(encoder);
    auto command_buffer = NS::RetainPtr(command_batch->commandBuffer);
    ccv_nnc_finish_command_batch(command_batch);
    command_buffer->waitUntilCompleted();
    return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
  };
  for (uint32_t i = 0; i < kWarmupIterations; ++i)
    (void)run_once();
  double best_seconds = 0;
  for (uint32_t i = 0; i < kTimedIterations; ++i) {
    const double seconds = run_once();
    if (i == 0 || seconds < best_seconds)
      best_seconds = seconds;
  }
  *seconds_out = best_seconds;
  return true;
}

// Core ML evaluation alone, at M rows of the probe shape. A short program
// needs about ten evaluations before its time settles (measured on M3 Ultra:
// with two, 1024 rows varied from 2.7 to 4.3 ms between processes).
static bool probe_ane_rowwise_seconds(
    ccv_nnc_mfa_context_t* const context,
    const uint32_t M,
    double* const seconds_out,
    std::string* const error_out)
{
  constexpr uint32_t kWarmupIterations = 10;
  constexpr uint32_t kTimedIterations = 3;
  auto* const cache = get_or_create_cache(context, error_out);
  if (!cache)
    return false;
  const ccv_nnc_mfa_ane_rowwise_gemm_params_t params = {
    .M = M,
    .N = kHybridProbeN,
    .K = kHybridProbeK,
    .data_type = MTL::DataTypeHalf,
    .batch_dimension = 1,
  };
  if (!ensure_shared_scratch(cache, params, error_out))
    return false;
  char error_buffer[1024] = {};
  auto* const program = ccv_nnc_mfa_ane_rowwise_coreml_find_or_create_program(
      cache, M, kHybridProbeN, kHybridProbeK, error_buffer, sizeof(error_buffer));
  if (!program) {
    if (error_out)
      *error_out = bridge_error(error_buffer);
    return false;
  }
  using clock = std::chrono::steady_clock;
  double best_seconds = 0;
  bool ok = true;
  for (uint32_t i = 0; ok && i < kWarmupIterations + kTimedIterations; ++i) {
    const auto start = clock::now();
    ok = evaluate_program(cache, program, error_out);
    const double seconds = std::chrono::duration<double>(clock::now() - start).count();
    if (i >= kWarmupIterations && (best_seconds == 0 || seconds < best_seconds))
      best_seconds = seconds;
  }
  ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
  *seconds_out = best_seconds;
  return ok;
}

// Int8MatMul's activation quantize reads and writes 4 bytes per FP16 element.
static bool probe_bandwidth(
    ccv_nnc_mfa_context_t* const context,
    double* const bandwidth_out,
    std::string* const error_out)
{
  constexpr uint32_t M = 8192;
  constexpr uint32_t K = kHybridProbeK;
  constexpr uint32_t kWarmupIterations = 2;
  constexpr uint32_t kTimedIterations = 3;
  const Int8MatMulDescriptor descriptor = {
    .M = M,
    .N = kHybridProbeN,
    .K = K,
    .expertCount = 1,
    .binCount = 0,
    .operation = Int8MatMulQuantizeActivation,
    .ioPrecision = GEMMOperandPrecision::FP16,
  };
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto* const pipeline = context->kernel_cache.findKernel<Int8MatMulKernel, Int8MatMulDescriptor, Int8MatMulKernelDescriptor>(
      descriptor, context->device.get(), context->device_properties);
  pool->drain();
  auto command_queue = NS::TransferPtr(context->device->newCommandQueue());
  auto source = NS::TransferPtr(context->device->newBuffer((size_t)M * K * sizeof(uint16_t), MTL::ResourceStorageModePrivate));
  auto destination = NS::TransferPtr(context->device->newBuffer((size_t)M * K * sizeof(uint16_t) + (size_t)M * sizeof(float), MTL::ResourceStorageModePrivate));
  if (!pipeline || !command_queue || !source || !destination) {
    if (error_out)
      *error_out = "failed to prepare quantize for hybrid split probe";
    return false;
  }
  double best_seconds = 0;
  for (uint32_t i = 0; i < kWarmupIterations + kTimedIterations; ++i) {
    auto* const command_batch = ccv_nnc_start_command_batch(command_queue.get());
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline->pipeline.get());
    encoder->setBuffer(source.get(), 0, 0);
    encoder->setBuffer(destination.get(), 0, 1);
    encoder->setBuffer(destination.get(), (size_t)M * K * sizeof(uint16_t), 2);
    encoder->dispatchThreadgroups(pipeline->kernel->threadgroupsPerGrid(descriptor), pipeline->kernel->threadgroupSize);
    command_batch->finishCommand(encoder);
    auto command_buffer = NS::RetainPtr(command_batch->commandBuffer);
    ccv_nnc_finish_command_batch(command_batch);
    command_buffer->waitUntilCompleted();
    const double seconds = command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
    if (i >= kWarmupIterations && (best_seconds == 0 || seconds < best_seconds))
      best_seconds = seconds;
  }
  *bandwidth_out = best_seconds > 0 ? 4.0 * M * K / best_seconds : 0;
  return true;
}

// Commit a fence update and spin until the CPU sees it.
static bool probe_fence_seconds(
    ccv_nnc_mfa_context_t* const context,
    double* const seconds_out,
    std::string* const error_out)
{
  constexpr uint32_t kWarmupIterations = 2;
  constexpr uint32_t kTimedIterations = 3;
  auto* const cache = get_or_create_cache(context, error_out);
  char error_buffer[1024] = {};
  if (!cache)
    return false;
  if (!ccv_nnc_mfa_ane_rowwise_fast_fence_prepare(cache, error_buffer, sizeof(error_buffer))) {
    if (error_out)
      *error_out = bridge_error(error_buffer);
    return false;
  }
  auto command_queue = NS::TransferPtr(context->device->newCommandQueue());
  using clock = std::chrono::steady_clock;
  double best_seconds = 0;
  for (uint32_t i = 0; i < kWarmupIterations + kTimedIterations; ++i) {
    const uint32_t value = ccv_nnc_mfa_ane_rowwise_fast_fence_next_value(cache, kHybridPrepFenceIndex);
    auto* const command_batch = ccv_nnc_start_command_batch(command_queue.get());
    if (!ccv_nnc_mfa_ane_rowwise_fast_fence_append_update(cache, command_batch, kHybridPrepFenceIndex, value, error_buffer, sizeof(error_buffer))) {
      if (error_out)
        *error_out = bridge_error(error_buffer);
      return false;
    }
    const auto start = clock::now();
    ccv_nnc_finish_command_batch(command_batch);
    if (!ccv_nnc_mfa_ane_rowwise_fast_fence_cpu_wait(cache, kHybridPrepFenceIndex, value, error_buffer, sizeof(error_buffer))) {
      if (error_out)
        *error_out = bridge_error(error_buffer);
      return false;
    }
    const double seconds = std::chrono::duration<double>(clock::now() - start).count();
    if (i >= kWarmupIterations && (best_seconds == 0 || seconds < best_seconds))
      best_seconds = seconds;
  }
  *seconds_out = best_seconds;
  return true;
}

static bool select_cached_hybrid_cost_model(
    ccv_nnc_mfa_context_t* const context,
    const bool neural_accelerators,
    ccv_nnc_mfa_hybrid_cost_model_t* const model_out,
    std::string* const error_out)
{
  std::lock_guard<std::mutex> lock(g_hybrid_cost_model_mutex);
  ccv_nnc_mfa_hybrid_cost_model_t& cached_model = g_hybrid_cost_models[neural_accelerators ? 1 : 0];
  if (!cached_model.initialized) {
    const auto probe_gpu_seconds = [&](const uint32_t M, double* const seconds_out, std::string* const error_out) {
      return neural_accelerators ?
          probe_na_int8_matmul_seconds(context, M, seconds_out, error_out) :
          probe_int8_matmul_seconds(context, M, seconds_out, error_out);
    };
    // About 300 ms in all on M3 Ultra and M4 Pro, mostly the ANE.
    constexpr uint32_t kANERows[2] = { 1024, 4096 };
    constexpr uint32_t kGPURows[2] = { 1024, 4096 };
    double ane_seconds[2] = {};
    double gpu_seconds[2] = {};
    ccv_nnc_mfa_hybrid_cost_model_t model = {};
    std::string probe_error;
    // The larger size of each side first, which also warms it up.
    const bool ok =
        probe_ane_rowwise_seconds(context, kANERows[1], &ane_seconds[1], &probe_error) &&
        probe_ane_rowwise_seconds(context, kANERows[0], &ane_seconds[0], &probe_error) &&
        probe_gpu_seconds(kGPURows[1], &gpu_seconds[1], &probe_error) &&
        probe_gpu_seconds(kGPURows[0], &gpu_seconds[0], &probe_error) &&
        probe_bandwidth(context, &model.bandwidth, &probe_error) &&
        probe_fence_seconds(context, &model.fence_seconds, &probe_error);
    const double work = (double)kHybridProbeN * kHybridProbeK;
    model.ane_slope = (ane_seconds[1] - ane_seconds[0]) / ((kANERows[1] - kANERows[0]) * work);
    model.ane_fixed = std::max(0.0, ane_seconds[0] - kANERows[0] * work * model.ane_slope);
    model.gpu_slope = (gpu_seconds[1] - gpu_seconds[0]) / ((kGPURows[1] - kGPURows[0]) * work);
    model.gpu_fixed = std::max(0.0, gpu_seconds[0] - kGPURows[0] * work * model.gpu_slope);
    model.initialized = true;
    model.valid = ok && model.ane_slope > 0 && model.gpu_slope > 0 && model.bandwidth > 0;
    cached_model = model;
    if (!model.valid && error_out)
      *error_out = probe_error.empty() ? "hybrid split probe produced invalid timing" : probe_error;
    if (model.valid && METAL_LOG_LEVEL(context) >= 1) {
      char message[256];
      std::snprintf(message, sizeof(message),
          "Hybrid ANE+GPU rowwise GEMM model: ANE %.3f ms + %.2f TFLOPs, GPU %.3f ms + %.2f TFLOPs, %.0f GB/s, fence %.3f ms.",
          model.ane_fixed * 1e3, 2e-12 / model.ane_slope, model.gpu_fixed * 1e3, 2e-12 / model.gpu_slope,
          model.bandwidth * 1e-9, model.fence_seconds * 1e3);
      ccv_nnc_mfa_log_message(message);
    }
  }
  if (!cached_model.valid) {
    if (error_out && error_out->empty())
      *error_out = "failed to determine hybrid split cost model";
    return false;
  }
  *model_out = cached_model;
  return true;
}

// Predicts the GPU alone, the ANE alone, and the split where both sides finish
// together, and returns the ANE rows per batch of the fastest: 0 for the GPU
// alone, M for the ANE alone. Returns false if the split does not apply.
static bool choose_hybrid_ane_rows_per_batch(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    uint32_t* const ane_rows_out)
{
  const bool neural_accelerators = params.use_neural_accelerators != 0;
  if (!neural_accelerators && !int8_matmul_split_supported(context, params))
    return false;
  ccv_nnc_mfa_hybrid_cost_model_t model;
  std::string error;
  if (!select_cached_hybrid_cost_model(context, neural_accelerators, &model, &error)) {
    log_ane_rowwise_error(context, error);
    return false;
  }
  const double batch = rowwise_batch_dimension(params);
  const double M = params.M;
  const double N = params.N;
  const double K = params.K;
  const double work = N * K;
  const double io = io_precision(params.data_type).size();
  // The GPU quantizes its own rows, except that Int8MatMul multiplies FP16
  // rows as they are, or only rotates them for H256 weights. Each ANE program
  // waits for the GPU to prepare its rows and a copy of the weights, and for
  // the GPU to dequantize its output afterwards; these passes do not overlap
  // the kernels.
  const bool gpu_prepares_rows = neural_accelerators || params.data_type != MTL::DataTypeHalf || params.activation_hadamard_256;
  const double gpu_per_row = work * model.gpu_slope + (gpu_prepares_rows ? K * (io + 2) / model.bandwidth : 0);
  const auto gpu_seconds = [&](const double rows) {
    return rows > 0 ? model.gpu_fixed + rows * gpu_per_row : 0.0;
  };
  const auto ane_serial_seconds = [&](const double rows) {
    return model.fence_seconds + (rows * K * (io + 1) + 2 * N * K + rows * N * (io + 2)) / model.bandwidth;
  };
  const auto ane_kernel_seconds = [&](const double rows) {
    return model.ane_fixed + rows * work * model.ane_slope;
  };
  // Without neural accelerators, a split runs as tiles of the whole GEMM, one
  // after another, each with one ANE program of at most kHybridMaxANEProgramRows.
  const auto tiles_for = [&](const double ane_rows) {
    return neural_accelerators ? 1.0 : std::max(1.0, std::ceil(ane_rows / kHybridMaxANEProgramRows));
  };
  const auto split_seconds = [&](const double ane_rows) {
    const double tiles = tiles_for(ane_rows);
    const double tile_ane_rows = ane_rows / tiles;
    const double tile_gpu_rows = (M * batch - ane_rows) / tiles;
    return tiles * (ane_serial_seconds(tile_ane_rows) + std::max(ane_kernel_seconds(tile_ane_rows), gpu_seconds(tile_gpu_rows)));
  };
  // The ANE rows of a program where both sides finish together.
  const auto balanced_rows = [&](const double rows) {
    return (rows * gpu_per_row + model.gpu_fixed - model.ane_fixed) / (work * model.ane_slope + gpu_per_row);
  };
  const auto nearest_rows = [](const double rows) -> uint32_t {
    return rows > 0 ? (uint32_t)std::llround(rows / kANERowAlignment) * kANERowAlignment : 0;
  };
  const double tiles = tiles_for(balanced_rows(M * batch));
  uint32_t ane_m = nearest_rows(balanced_rows(M * batch / tiles) / batch) * (uint32_t)tiles;
  double split = ane_m > 0 && ane_m < params.M ? split_seconds((double)ane_m * batch) : INFINITY;
  // Another tile adds a whole program's fixed and serial costs, so full
  // programs with one tile fewer can be faster than balanced tiles.
  if (tiles > 1) {
    const uint32_t fewer_tiles_ane_m = kHybridMaxANEProgramRows * ((uint32_t)tiles - 1);
    const double fewer_tiles_split = fewer_tiles_ane_m < params.M ? split_seconds(fewer_tiles_ane_m) : INFINITY;
    if (fewer_tiles_split < split) {
      ane_m = fewer_tiles_ane_m;
      split = fewer_tiles_split;
    }
  }
  const double gpu_alone = gpu_seconds(M * batch);
  const double ane_alone = ane_serial_seconds(M * batch) + ane_kernel_seconds(M * batch);
  if (gpu_alone <= std::min(ane_alone, split))
    *ane_rows_out = 0;
  else
    *ane_rows_out = ane_alone <= split ? params.M : ane_m;
  if (METAL_LOG_LEVEL(context) >= 2) {
    char message[256];
    std::snprintf(message, sizeof(message),
        "Hybrid ANE+GPU rowwise GEMM M=%u N=%u K=%u predicted GPU %.3f ms, ANE %.3f ms, split with %u ANE rows %.3f ms.",
        params.M, params.N, params.K, gpu_alone * 1e3, ane_alone * 1e3, ane_m, split * 1e3);
    ccv_nnc_mfa_log_message(message);
  }
  return true;
}

// Splits a GEMM without neural accelerators into tiles of the whole GEMM, one
// after another: Int8MatMul computes the first rows of each tile while one ANE
// program computes the rest. The CPU waits for the ANE inputs before it
// submits the GPU rows, so the stream does not delay the ANE.
static int run_int8_matmul_split_gemm(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t ane_m,
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* const stream_context)
{
  const uint32_t tile_count = (ane_m + kHybridMaxANEProgramRows - 1) / kHybridMaxANEProgramRows;
  const uint32_t tile_ane_m = tile_count ? ane_m / tile_count : 0;
  if (!int8_matmul_split_supported(context, params) || !ane_m || ane_m >= params.M ||
      tile_ane_m * tile_count != ane_m || tile_ane_m % kANERowAlignment != 0 ||
      params.M / tile_count <= tile_ane_m)
    return 0;
  // These ANE rows quantize (and rotate) their own activations.
  const ccv_nnc_mfa_ane_rowwise_gemm_params_t ane_params = {
    .M = tile_ane_m,
    .N = params.N,
    .K = params.K,
    .data_type = (uint32_t)params.data_type,
    .fused_bias = 0,
    .activation_hadamard_256 = params.activation_hadamard_256,
    .batch_dimension = 1,
  };
  std::string error;
  ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache = get_or_create_cache(context, &error);
  if (!cache || !ensure_shared_scratch(cache, ane_params, &error)) {
    log_ane_rowwise_error(context, error);
    return 0;
  }
  char error_buffer[1024] = {};
  ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program =
      ccv_nnc_mfa_ane_rowwise_coreml_find_or_create_program(cache, tile_ane_m, params.N, params.K, error_buffer, sizeof(error_buffer));
  if (!program) {
    log_ane_rowwise_error(context, bridge_error(error_buffer));
    return 0;
  }
  if (!ccv_nnc_mfa_ane_rowwise_fast_fence_prepare(cache, error_buffer, sizeof(error_buffer))) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, bridge_error(error_buffer));
    return 0;
  }
  const size_t element_size = io_precision(params.data_type).size();
  const size_t weight_scale_offset = tensor_offsets[1] + rowwise_8i_scale_offset(params.N, params.K);
  for (uint32_t tile = 0; tile < tile_count; tile++) {
    const uint32_t start = (uint32_t)((uint64_t)params.M * tile / tile_count);
    const uint32_t end = (uint32_t)((uint64_t)params.M * (tile + 1) / tile_count);
    const uint32_t gpu_m = end - start - tile_ane_m;
    mtl_buffer_t* gpu_tensors[4] = { tensors[0], tensors[1], tensors[2], nullptr };
    size_t gpu_offsets[3] = {
      tensor_offsets[0] + (size_t)start * params.K * element_size,
      tensor_offsets[1],
      tensor_offsets[2] + (size_t)start * params.N * element_size,
    };
    // The tile's ANE rows follow its GPU rows.
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline = find_transform_pipeline(context, ane_params, gpu_m, gpu_m);
    const uint32_t fence = ccv_nnc_mfa_ane_rowwise_fast_fence_next_value(cache, kHybridPrepFenceIndex);
    if (!run_quantize_activation(cache, transform_pipeline, program, tensors[0], gpu_offsets[0], tensors[1], tensor_offsets[1], stream_context, &error, fence)) {
      ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
      log_ane_rowwise_error(context, error);
      return tile ? -1 : 0;
    }
    ccv_nnc_stream_context_commit(stream_context);
    bool ok = ccv_nnc_mfa_ane_rowwise_fast_fence_cpu_wait(cache, kHybridPrepFenceIndex, fence, error_buffer, sizeof(error_buffer));
    if (ok) {
      mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
      ccv_nnc_mfa_encode_scaled_gemm(context, int8_matmul_split_gpu_params(params, gpu_m), command_batch, gpu_tensors, gpu_offsets);
      ok = ccv_nnc_mfa_ane_rowwise_finish_command_batch_async(stream_context, command_batch, error_buffer, sizeof(error_buffer));
      ccv_nnc_stream_context_commit(stream_context);
    }
    ok = ok &&
        evaluate_program(cache, program, &error) &&
        run_dequantize_output(cache, transform_pipeline, program,
            ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache), 0,
            tensors[1], weight_scale_offset, nullptr, 0,
            tensors[2], gpu_offsets[2], 0, stream_context, &error);
    if (!ok) {
      ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
      log_ane_rowwise_error(context, error.empty() ? bridge_error(error_buffer) : error);
      return -1;
    }
  }
  ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
  if (METAL_LOG_LEVEL(context) >= 1) {
    char message[192];
    std::snprintf(message, sizeof(message), "Hybrid ANE+Int8MatMul M=%u N=%u K=%u ANE=%u in %u tiles.",
        params.M, params.N, params.K, ane_m, tile_count);
    ccv_nnc_mfa_log_message(message);
  }
  return 1;
}

} // namespace

size_t ccv_nnc_mfa_ane_rowwise_gemm_reserved_scratch_size(
    ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  (void)params;
  return 0;
}

size_t ccv_nnc_mfa_ane_na_rowwise_split_gemm_reserved_scratch_size(
    ccv_nnc_mfa_context_t* const context,
    ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params)
{
  // Without neural accelerators, Int8MatMul prepares the GPU rows in MFA
  // scratch (unless they are FP16 without H256), ahead of any weights the
  // caller decodes there.
  if (!params.use_neural_accelerators)
    return int8_matmul_split_supported(context, params) ?
        ccv_nnc_mfa_scaled_gemm_reserved_scratch_size(context, int8_matmul_split_gpu_params(params, params.M)) : 0;
  return activation_quant_layout(params).scratch_bytes;
}

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
  if (params.activation_hadamard_256 && (!params.K || params.K > 65536 || params.K % 256 != 0))
    return 0;
  std::string error;
  ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache = get_or_create_cache(context, &error);
  if (!cache) {
    log_ane_rowwise_error(context, error);
    return 0;
  }
  if (!ensure_shared_scratch(cache, params, &error)) {
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
          cache,
          rowwise_padded_total_rows(params),
          params.N,
          params.K,
          error_buffer,
          sizeof(error_buffer));
  if (!program) {
    log_ane_rowwise_error(context, bridge_error(error_buffer));
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
  if (!run_dequantize_output(
          cache,
          transform_pipeline,
          program,
          ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(cache),
          0,
          weight,
          weight_scale_offset,
          bias,
          bias_offset,
          output,
          tensor_offsets[2],
          params.fused_bias,
          stream_context,
          &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }
  ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
  return 1;
}

int ccv_nnc_mfa_run_ane_na_rowwise_split_gemm(
    ccv_nnc_mfa_context_t* const context,
    ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    const uint32_t ane_m,
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* const stream_context)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensor_offsets != nullptr);
  if (!params.use_neural_accelerators)
    return run_int8_matmul_split_gemm(context, params, ane_m, tensors, tensor_offsets, stream_context);
  if (!ane_m || ane_m >= params.M)
    return 0;
  const uint32_t na_m = params.M - ane_m;
  const ccv_nnc_mfa_ane_rowwise_gemm_params_t ane_params = ane_params_from_hybrid(params, ane_m);
  if (rowwise_padded_total_rows(ane_params) != rowwise_total_rows(ane_params))
    return 0;
  ccv_nnc_mfa_ane_rowwise_gemm_params_t hybrid_transform_params = ane_params;
  hybrid_transform_params.batch_stride_a =
      rowwise_batch_dimension(params) > 1 ? params.M * params.K : 0;
  const ccv_nnc_mfa_activation_quant_layout_t a_layout = activation_quant_layout(params);
  std::string error;
  ccv_nnc_mfa_ane_rowwise_coreml_cache_t* const cache = get_or_create_cache(context, &error);
  if (!cache) {
    log_ane_rowwise_error(context, error);
    return 0;
  }
  if (!ensure_shared_scratch(cache, ane_params, &error)) {
    log_ane_rowwise_error(context, error);
    return 0;
  }
  PipelineValue<NAInt8MatMulKernel>* const full_quantize_pipeline =
      find_na_int8_pipeline(context, params, params.M);
  PipelineValue<NAInt8MatMulKernel>* const partial_matmul_pipeline =
      find_na_int8_pipeline(
          context,
          params,
          na_m,
          rowwise_batch_dimension(params) > 1 ? std::optional<uint32_t>(params.M * params.K) : std::nullopt,
          rowwise_batch_dimension(params) > 1 ? std::optional<uint32_t>(params.M) : std::nullopt);
  PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline =
      find_transform_pipeline(context, hybrid_transform_params, na_m, na_m, params.M, na_m);
  char error_buffer[1024] = {};
  ccv_nnc_mfa_ane_rowwise_coreml_program_t* const program =
      ccv_nnc_mfa_ane_rowwise_coreml_find_or_create_program(
          cache,
          rowwise_padded_total_rows(ane_params),
          params.N,
          params.K,
          error_buffer,
          sizeof(error_buffer));
  if (!program) {
    log_ane_rowwise_error(context, bridge_error(error_buffer));
    return 0;
  }

  mtl_buffer_t* const activation = tensors[0];
  mtl_buffer_t* const weight = tensors[1];
  mtl_buffer_t* const output = tensors[2];
  mtl_buffer_t* const bias = params.fused_bias ? tensors[3] : nullptr;
  mtl_buffer_t* const scratch = context->request_scratch(a_layout.scratch_bytes);
  const uint32_t b_batches = (rowwise_batch_dimension(params) > 1 && params.batch_stride_b > 0) ? rowwise_batch_dimension(params) : 1;
  const size_t weight_scale_offset = tensor_offsets[1] + rowwise_8i_scale_offset((size_t)b_batches * params.N, params.K);
  const size_t bias_offset = params.fused_bias ? tensor_offsets[3] : 0;

  char fence_error_buffer[1024] = {};
  const int fast_fence_ok = ccv_nnc_mfa_ane_rowwise_fast_fence_prepare(
      cache,
      fence_error_buffer,
      sizeof(fence_error_buffer));
  if (!fast_fence_ok) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, bridge_error(fence_error_buffer));
    return 0;
  }
  const uint32_t prep_fence_value = ccv_nnc_mfa_ane_rowwise_fast_fence_next_value(cache, kHybridPrepFenceIndex);

  if (!run_hybrid_prepare_inputs(
          cache,
          full_quantize_pipeline,
          transform_pipeline,
          program,
          params,
          activation,
          tensor_offsets[0],
          weight,
          tensor_offsets[1],
          stream_context,
          scratch,
          a_layout,
          prep_fence_value,
          &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }

  if (!run_partial_na_matmul_async(
          partial_matmul_pipeline,
          params,
          na_m,
          scratch,
          a_layout,
          weight,
          tensor_offsets[1],
          weight_scale_offset,
          output,
          tensor_offsets[2],
          bias,
          bias_offset,
          stream_context,
          &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return 0;
  }

  ccv_nnc_stream_context_commit(stream_context);
  const int prep_ok = ccv_nnc_mfa_ane_rowwise_fast_fence_cpu_wait(
      cache,
      kHybridPrepFenceIndex,
      prep_fence_value,
      fence_error_buffer,
      sizeof(fence_error_buffer));
  if (!prep_ok) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(
        context,
        bridge_error(fence_error_buffer).empty() ? "split GEMM prep fence wait failed" : bridge_error(fence_error_buffer));
    return 0;
  }
  const bool evaluated = evaluate_program(cache, program, &error);
  if (!evaluated) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return -1;
  }
  if (!run_dequantize_output(
          cache,
          transform_pipeline,
          program,
          scratch,
          a_layout.scale_offset,
          weight,
          weight_scale_offset,
          bias,
          bias_offset,
          output,
          tensor_offsets[2],
          params.fused_bias,
          stream_context,
          &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
    return -1;
  }
  ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
  return 1;
}

int ccv_nnc_mfa_choose_ane_na_rowwise_split_rows_per_batch(
    ccv_nnc_mfa_context_t* const context,
    ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    uint32_t* const ane_rows)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  CCV_NNC_MFA_PRECONDITION(ane_rows != nullptr);
  return choose_hybrid_ane_rows_per_batch(context, params, ane_rows) ? 1 : 0;
}

void ccv_nnc_mfa_ane_rowwise_gemm_cleanup(ccv_nnc_mfa_context_t* const context)
{
  if (!context || !ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context))
    return;
  auto* const cache = (ccv_nnc_mfa_ane_rowwise_coreml_cache_t*)ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context);
  ccv_nnc_mfa_ane_rowwise_coreml_cache_destroy(cache);
  ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(context, nullptr);
}
