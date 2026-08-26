#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_ane_rowwise_internal.hpp"
#include "ccv_nnc_mfa_ane_rowwise_gemm.hpp"
#include "ccv_nnc_mfa_ane_rowwise_coreml.hpp"
#include "ccv_nnc_mfa_error.hpp"

#include "kernels/ANERowwiseTransformDescriptor.hpp"
#include "kernels/ANERowwiseTransformKernel.hpp"
#include "kernels/NAInt8MatMulDescriptor.hpp"
#include "kernels/NAInt8MatMulKernel.hpp"

#include <algorithm>
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
constexpr uint32_t kANERowAlignment = 128;
constexpr uint32_t kHybridProbeM = 8192;
constexpr uint32_t kHybridProbeN = 4096;
constexpr uint32_t kHybridProbeK = 4096;
constexpr double kANEReferenceTFlops = 20.0;
constexpr uint32_t kHybridDisableRatioThreshold = 7;
constexpr uint64_t kHybridLargeANEWorkThreshold = 3ULL * 1024 * 1024;
constexpr uint32_t kHybridMinANERowsForLargeWork = 512;
constexpr uint32_t kHybridMinANERowsForSmallWork = 1024;
constexpr uint32_t kHybridPrepFenceIndex = 0;

typedef struct {
  bool initialized;
  bool valid;
  uint32_t gpu_to_ane_ratio;
  double na_tflops;
} ccv_nnc_mfa_hybrid_probe_cache_t;

static std::mutex g_hybrid_probe_mutex;
static ccv_nnc_mfa_hybrid_probe_cache_t g_hybrid_probe_cache = {
  .initialized = false,
  .valid = false,
  .gpu_to_ane_ratio = 0,
  .na_tflops = 0,
};

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

static uint32_t align_rows_up(const uint32_t rows) noexcept
{
  return (uint32_t)align_up(rows, kANERowAlignment);
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

static bool probe_na_int8_matmul_tflops(
    ccv_nnc_mfa_context_t* const context,
    double* const tflops_out,
    std::string* const error_out)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  constexpr uint32_t M = kHybridProbeM;
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
      encoder->dispatchThreadgroups( pipeline->kernel->threadgroupsPerGrid(M, N, 1), MTL::Size(pipeline->kernel->threadgroupSize(pipeline->pipeline.get()), 1, 1));
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
  *tflops_out = (4.0 * (double)M * (double)N * (double)K) / best_seconds / 1.0e12;
  return true;
}

static bool select_cached_hybrid_gpu_to_ane_ratio(
    ccv_nnc_mfa_context_t* const context,
    uint32_t* const ratio_out,
    double* const tflops_out,
    std::string* const error_out)
{
  std::lock_guard<std::mutex> lock(g_hybrid_probe_mutex);
  if (!g_hybrid_probe_cache.initialized) {
    double na_tflops = 0;
    std::string probe_error;
    const bool ok = probe_na_int8_matmul_tflops(context, &na_tflops, &probe_error);
    g_hybrid_probe_cache.initialized = true;
    g_hybrid_probe_cache.valid = ok;
    g_hybrid_probe_cache.na_tflops = na_tflops;
    g_hybrid_probe_cache.gpu_to_ane_ratio =
        ok ? std::max<uint32_t>(1, (uint32_t)llround(na_tflops / kANEReferenceTFlops + 1)) : 0;
    if (!ok && error_out)
      *error_out = probe_error;
    if (ok && METAL_LOG_LEVEL(context) >= 1) {
      char message[256];
      std::snprintf(
          message,
          sizeof(message),
          "Hybrid ANE+NA rowwise GEMM probe selected GPU:ANE ratio %u:1 from %.2f TFLOPs.",
          g_hybrid_probe_cache.gpu_to_ane_ratio,
          g_hybrid_probe_cache.na_tflops);
      ccv_nnc_mfa_log_message(message);
      if (g_hybrid_probe_cache.gpu_to_ane_ratio >= kHybridDisableRatioThreshold) {
        std::snprintf(
            message,
            sizeof(message),
            "Hybrid ANE+NA rowwise GEMM disabled because cached GPU:ANE ratio %u:1 meets cutoff %u:1.",
            g_hybrid_probe_cache.gpu_to_ane_ratio,
            kHybridDisableRatioThreshold);
        ccv_nnc_mfa_log_message(message);
      }
    }
  }
  if (!g_hybrid_probe_cache.valid) {
    if (error_out && error_out->empty())
      *error_out = "failed to determine cached hybrid GPU:ANE split ratio";
    return false;
  }
  if (ratio_out)
    *ratio_out = g_hybrid_probe_cache.gpu_to_ane_ratio;
  if (tflops_out)
    *tflops_out = g_hybrid_probe_cache.na_tflops;
  return true;
}

static uint32_t choose_hybrid_ane_rows_per_batch(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params)
{
  if (params.M < kANERowAlignment * 2)
    return 0;
  uint32_t gpu_to_ane_ratio = 0;
  std::string error;
  if (!select_cached_hybrid_gpu_to_ane_ratio(context, &gpu_to_ane_ratio, nullptr, &error)) {
    log_ane_rowwise_error(context, error);
    return 0;
  }
  if (gpu_to_ane_ratio >= kHybridDisableRatioThreshold)
    return 0;
  const uint32_t denominator = gpu_to_ane_ratio + 1;
  uint32_t ane_m = align_rows_up((uint32_t)llround((double)params.M / denominator));
  ane_m = std::max<uint32_t>(kANERowAlignment, ane_m);
  if (ane_m >= params.M)
    ane_m = params.M > kANERowAlignment ? params.M - kANERowAlignment : 0;
  ane_m -= ane_m % kANERowAlignment;
  if (ane_m == 0 || ane_m >= params.M)
    return 0;
  const uint64_t work_per_row = (uint64_t)params.N * params.K;
  const uint32_t min_ane_m =
      work_per_row >= kHybridLargeANEWorkThreshold ? kHybridMinANERowsForLargeWork : kHybridMinANERowsForSmallWork;
  if (ane_m < min_ane_m)
    return 0;
  return ane_m;
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

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
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
      MTL::Size(kernel->threadgroupSize(matmul_pipeline->pipeline.get()), 1, 1));
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

} // namespace

size_t ccv_nnc_mfa_ane_rowwise_gemm_reserved_scratch_size(
    ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  (void)params;
  return 0;
}

size_t ccv_nnc_mfa_ane_na_rowwise_split_gemm_reserved_scratch_size(
    ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params)
{
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
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* const stream_context)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensor_offsets != nullptr);
  const uint32_t ane_m = choose_hybrid_ane_rows_per_batch(context, params);
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
  if (!ensure_shared_scratch(cache, ane_params, &error)) {
    ccv_nnc_mfa_ane_rowwise_coreml_program_release(program);
    log_ane_rowwise_error(context, error);
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

uint32_t ccv_nnc_mfa_choose_ane_na_rowwise_split_rows_per_batch(
    ccv_nnc_mfa_context_t* const context,
    ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params)
{
  CCV_NNC_MFA_PRECONDITION(context != nullptr);
  return choose_hybrid_ane_rows_per_batch(context, params);
}

void ccv_nnc_mfa_ane_rowwise_gemm_cleanup(ccv_nnc_mfa_context_t* const context)
{
  if (!context || !ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context))
    return;
  auto* const cache = (ccv_nnc_mfa_ane_rowwise_coreml_cache_t*)ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context);
  ccv_nnc_mfa_ane_rowwise_coreml_cache_destroy(cache);
  ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(context, nullptr);
}
