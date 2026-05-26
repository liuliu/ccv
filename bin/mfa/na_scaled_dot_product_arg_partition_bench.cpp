#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
}
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAScaledDotProductArgPartitionDescriptor.hpp"
#include "nnc/mfa/kernels/NAScaledDotProductArgPartitionKernel.hpp"
#include "nnc/mfa/kernels/NAScaledDotProductArgPartitionKernelDescriptor.hpp"
#include "nnc/mfa/kernels/ScaledDotProductArgPartitionDescriptor.hpp"
#include "nnc/mfa/kernels/ScaledDotProductArgPartitionKernel.hpp"
#include "nnc/mfa/kernels/ScaledDotProductArgPartitionKernelDescriptor.hpp"

using namespace ccv::nnc;

namespace {

struct Config {
  int T = 512;
  int C = 8192;
  int H = 64;
  int D = 128;
  int kth = 512;
  int is_causal = 1;
  int compression_ratio = 4;
  int warmup = 2;
  int timed = 5;
  int scan_score_tiles = 0;
};

struct Stats {
  double average_ms = 0;
  double median_ms = 0;
  double min_ms = 0;
  double max_ms = 0;
};

struct StageProfile {
  Stats score;
  Stats topk_tile;
  Stats topk_merge;
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

double average_ms(const std::vector<double>& values)
{
  double sum = 0;
  for (const double value : values)
    sum += value;
  return sum / values.size();
}

Stats compute_stats(std::vector<double> values)
{
  Stats stats;
  stats.average_ms = average_ms(values);
  std::sort(values.begin(), values.end());
  stats.min_ms = values.front();
  stats.max_ms = values.back();
  stats.median_ms = values[values.size() / 2];
  return stats;
}

void print_stats(const char* const name, const Stats& stats)
{
  std::cout << name
            << " avg_ms=" << stats.average_ms
            << " median_ms=" << stats.median_ms
            << " min_ms=" << stats.min_ms
            << " max_ms=" << stats.max_ms
            << "\n";
}

void restore_flag(const uint64_t saved_flags, const uint64_t flag)
{
  if (saved_flags & flag)
    ccv_nnc_enable_flag(flag);
  else
    ccv_nnc_disable_flag(flag);
}

void fill_sdpap_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& head_w, const Config& config)
{
  for (int t = 0; t < config.T; ++t)
    for (int h = 0; h < config.H; ++h)
      for (int d = 0; d < config.D; ++d)
      {
        const int centered = (t * 17 + h * 13 + d * 5) % 31 - 15;
        q[(t * config.H + h) * config.D + d] = (float)centered / 64.0f;
      }
  for (int c = 0; c < config.C; ++c)
    for (int d = 0; d < config.D; ++d)
    {
      const int centered = (c * 19 + d * 7) % 37 - 18;
      k[c * config.D + d] = (float)centered / 64.0f;
    }
  for (int t = 0; t < config.T; ++t)
    for (int h = 0; h < config.H; ++h)
      head_w[t * config.H + h] = 0.75f + (float)((t * 11 + h * 3) % 17) / 32.0f;
}

void fill_sdpa_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v, const std::vector<float>& q_sdpap, const std::vector<float>& k_sdpap, const Config& config)
{
  for (int t = 0; t < config.T; ++t)
    for (int h = 0; h < config.H; ++h)
      for (int d = 0; d < config.D; ++d)
        q[(t * config.H + h) * config.D + d] = q_sdpap[(t * config.H + h) * config.D + d];
  for (int c = 0; c < config.C; ++c)
    for (int h = 0; h < config.H; ++h)
      for (int d = 0; d < config.D; ++d)
      {
        k[(c * config.H + h) * config.D + d] = k_sdpap[c * config.D + d];
        const int centered = (c * 23 + h * 5 + d * 3) % 43 - 21;
        v[(c * config.H + h) * config.D + d] = (float)centered / 64.0f;
      }
}

Stats benchmark_sdpap(
  const ccv_nnc_cmd_t cmd,
  ccv_nnc_tensor_t* const q,
  ccv_nnc_tensor_t* const k,
  ccv_nnc_tensor_t* const head_w,
  ccv_nnc_tensor_t* const selected,
  ccv_nnc_stream_context_t* const stream,
  const int warmup,
  const int timed)
{
  for (int i = 0; i < warmup; ++i)
  {
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, head_w), TENSOR_LIST(selected), stream);
    ccv_nnc_stream_context_wait(stream);
  }
  std::vector<double> samples;
  samples.reserve(timed);
  for (int i = 0; i < timed; ++i)
  {
    const auto start = std::chrono::steady_clock::now();
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, head_w), TENSOR_LIST(selected), stream);
    ccv_nnc_stream_context_wait(stream);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return compute_stats(samples);
}

Stats benchmark_sdpa(
  const ccv_nnc_cmd_t cmd,
  ccv_nnc_tensor_t* const q,
  ccv_nnc_tensor_t* const k,
  ccv_nnc_tensor_t* const v,
  ccv_nnc_tensor_t* const o,
  ccv_nnc_stream_context_t* const stream,
  const int warmup,
  const int timed)
{
  for (int i = 0; i < warmup; ++i)
  {
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v), TENSOR_LIST(o), stream);
    ccv_nnc_stream_context_wait(stream);
  }
  std::vector<double> samples;
  samples.reserve(timed);
  for (int i = 0; i < timed; ++i)
  {
    const auto start = std::chrono::steady_clock::now();
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v), TENSOR_LIST(o), stream);
    ccv_nnc_stream_context_wait(stream);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return compute_stats(samples);
}

void upload_buffer(
  MTL::CommandQueue* const command_queue,
  MTL::Buffer* const source,
  MTL::Buffer* const destination,
  const size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

NS::SharedPtr<MTL::Buffer> make_private_buffer(
  MTL::Device* const device,
  MTL::CommandQueue* const command_queue,
  const void* const data,
  const size_t size)
{
  auto stage = NS::TransferPtr(device->newBuffer(size, kSharedResourceOptions));
  auto buffer = NS::TransferPtr(device->newBuffer(size, kPrivateResourceOptions));
  std::memcpy(stage->contents(), data, size);
  upload_buffer(command_queue, stage.get(), buffer.get(), size);
  return buffer;
}

double run_score_stage_once(
  MTL::CommandQueue* const command_queue,
  MTL::ComputePipelineState* const pipeline,
  MTL::Size threadgroup_size,
  MTL::Buffer* const q,
  MTL::Buffer* const k,
  MTL::Buffer* const head_w,
  MTL::Buffer* const scores,
  const Config& config,
  const uint32_t score_block_m,
  const uint32_t score_block_n)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline);
  encoder->useResource(q, MTL::ResourceUsageRead);
  encoder->useResource(k, MTL::ResourceUsageRead);
  encoder->useResource(head_w, MTL::ResourceUsageRead);
  encoder->useResource(scores, MTL::ResourceUsageWrite);
  encoder->setBuffer(q, 0, 0);
  encoder->setBuffer(k, 0, 1);
  encoder->setBuffer(head_w, 0, 2);
  encoder->setBuffer(scores, 0, 3);
  encoder->dispatchThreadgroups(MTL::Size((config.C + score_block_n - 1) / score_block_n, (config.T + score_block_m - 1) / score_block_m, 1), threadgroup_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1000.0;
}

double run_topk_tile_stage_once(
  MTL::CommandQueue* const command_queue,
  MTL::ComputePipelineState* const pipeline,
  MTL::Size threadgroup_size,
  MTL::Buffer* const scores,
  MTL::Buffer* const candidate_scores,
  MTL::Buffer* const candidate_indices,
  const Config& config)
{
  const uint32_t topk_tile_c = 2048;
  const uint32_t topk_tiles = (config.C + topk_tile_c - 1) / topk_tile_c;
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline);
  encoder->useResource(scores, MTL::ResourceUsageRead);
  encoder->useResource(candidate_scores, MTL::ResourceUsageWrite);
  encoder->useResource(candidate_indices, MTL::ResourceUsageWrite);
  encoder->setBuffer(scores, 0, 0);
  encoder->setBuffer(candidate_scores, 0, 1);
  encoder->setBuffer(candidate_indices, 0, 2);
  encoder->dispatchThreadgroups(MTL::Size(topk_tiles, config.T, 1), threadgroup_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1000.0;
}

double run_topk_merge_stage_once(
  MTL::CommandQueue* const command_queue,
  MTL::ComputePipelineState* const pipeline,
  MTL::Size threadgroup_size,
  MTL::Buffer* const candidate_scores,
  MTL::Buffer* const candidate_indices,
  MTL::Buffer* const selected,
  const Config& config)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline);
  encoder->useResource(candidate_scores, MTL::ResourceUsageRead);
  encoder->useResource(candidate_indices, MTL::ResourceUsageRead);
  encoder->useResource(selected, MTL::ResourceUsageWrite);
  encoder->setBuffer(candidate_scores, 0, 0);
  encoder->setBuffer(candidate_indices, 0, 1);
  encoder->setBuffer(selected, 0, 2);
  encoder->dispatchThreadgroups(MTL::Size(config.T, 1, 1), threadgroup_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted)
    return 0;
  return (command_buffer->GPUEndTime() - command_buffer->GPUStartTime()) * 1000.0;
}

template<typename Descriptor, typename Kernel, typename KernelDescriptor>
StageProfile benchmark_mfa_stages(
  const Config& config,
  const ccv_nnc_tensor_t* const hq,
  const ccv_nnc_tensor_t* const hk,
  const ccv_nnc_tensor_t* const hhead_w,
  const float scale)
{
  auto device_owner = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  ccv_nnc_mfa_context_t* const context = ccv_nnc_init_mfa_context(device_owner.get());
  MTL::Device* const device = context->device.get();
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  Descriptor descriptor;
  descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  descriptor.T = (uint32_t)config.T;
  descriptor.C = (uint32_t)config.C;
  descriptor.H = (uint32_t)config.H;
  descriptor.D = (uint32_t)config.D;
  descriptor.kth = (uint32_t)config.kth;
  descriptor.compressionRatio = (uint32_t)config.compression_ratio;
  descriptor.scale = scale;
  descriptor.isCausal = config.is_causal != 0;
  DeviceProperties dprops = DeviceProperties();
  auto pipeline_value = context->kernel_cache.findKernel<Kernel, Descriptor, KernelDescriptor>(descriptor, device, dprops);
  auto kernel = pipeline_value->kernel;
  const size_t q_bytes = (size_t)config.T * config.H * config.D * sizeof(uint16_t);
  const size_t k_bytes = (size_t)config.C * config.D * sizeof(uint16_t);
  const size_t head_w_bytes = (size_t)config.T * config.H * sizeof(uint16_t);
  const size_t score_bytes = std::max<size_t>((size_t)config.T * config.C * sizeof(float), sizeof(float));
  const size_t selected_bytes = (size_t)config.T * config.kth * sizeof(int32_t);
  const uint32_t topk_tile_c = 2048;
  const uint32_t topk_tiles = (config.C + topk_tile_c - 1) / topk_tile_c;
  const size_t candidate_count = (size_t)config.T * topk_tiles * config.kth;
  auto q_buffer = make_private_buffer(device, command_queue.get(), hq->data.f16, q_bytes);
  auto k_buffer = make_private_buffer(device, command_queue.get(), hk->data.f16, k_bytes);
  auto head_w_buffer = make_private_buffer(device, command_queue.get(), hhead_w->data.f16, head_w_bytes);
  auto scores = NS::TransferPtr(device->newBuffer(score_bytes, kPrivateResourceOptions));
  auto candidate_scores = NS::TransferPtr(device->newBuffer(std::max<size_t>(candidate_count * sizeof(float), sizeof(float)), kPrivateResourceOptions));
  auto candidate_indices = NS::TransferPtr(device->newBuffer(std::max<size_t>(candidate_count * sizeof(int32_t), sizeof(int32_t)), kPrivateResourceOptions));
  auto selected = NS::TransferPtr(device->newBuffer(selected_bytes, kPrivateResourceOptions));

  for (int i = 0; i < config.warmup; ++i)
    run_score_stage_once(command_queue.get(), pipeline_value->pipeline.get(), kernel->scoreThreadgroupSize, q_buffer.get(), k_buffer.get(), head_w_buffer.get(), scores.get(), config, kernel->scoreBlockM, kernel->scoreBlockN);
  std::vector<double> score_samples;
  score_samples.reserve(config.timed);
  for (int i = 0; i < config.timed; ++i)
    score_samples.push_back(run_score_stage_once(command_queue.get(), pipeline_value->pipeline.get(), kernel->scoreThreadgroupSize, q_buffer.get(), k_buffer.get(), head_w_buffer.get(), scores.get(), config, kernel->scoreBlockM, kernel->scoreBlockN));

  run_score_stage_once(command_queue.get(), pipeline_value->pipeline.get(), kernel->scoreThreadgroupSize, q_buffer.get(), k_buffer.get(), head_w_buffer.get(), scores.get(), config, kernel->scoreBlockM, kernel->scoreBlockN);
  for (int i = 0; i < config.warmup; ++i)
    run_topk_tile_stage_once(command_queue.get(), pipeline_value->third.get(), kernel->topKTileThreadgroupSize, scores.get(), candidate_scores.get(), candidate_indices.get(), config);
  std::vector<double> topk_tile_samples;
  topk_tile_samples.reserve(config.timed);
  for (int i = 0; i < config.timed; ++i)
    topk_tile_samples.push_back(run_topk_tile_stage_once(command_queue.get(), pipeline_value->third.get(), kernel->topKTileThreadgroupSize, scores.get(), candidate_scores.get(), candidate_indices.get(), config));

  run_topk_tile_stage_once(command_queue.get(), pipeline_value->third.get(), kernel->topKTileThreadgroupSize, scores.get(), candidate_scores.get(), candidate_indices.get(), config);
  for (int i = 0; i < config.warmup; ++i)
    run_topk_merge_stage_once(command_queue.get(), pipeline_value->fourth.get(), kernel->topKMergeThreadgroupSize, candidate_scores.get(), candidate_indices.get(), selected.get(), config);
  std::vector<double> topk_merge_samples;
  topk_merge_samples.reserve(config.timed);
  for (int i = 0; i < config.timed; ++i)
    topk_merge_samples.push_back(run_topk_merge_stage_once(command_queue.get(), pipeline_value->fourth.get(), kernel->topKMergeThreadgroupSize, candidate_scores.get(), candidate_indices.get(), selected.get(), config));

  StageProfile profile = { compute_stats(score_samples), compute_stats(topk_tile_samples), compute_stats(topk_merge_samples) };
  ccv_nnc_deinit_mfa_context(context);
  return profile;
}

Stats benchmark_score_stage_variant(
  const Config& config,
  const ccv_nnc_tensor_t* const hq,
  const ccv_nnc_tensor_t* const hk,
  const ccv_nnc_tensor_t* const hhead_w,
  const float scale,
  const uint16_t score_block_m,
  const uint16_t score_block_n,
  const uint16_t score_simdgroups)
{
  auto device_owner = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  ccv_nnc_mfa_context_t* const context = ccv_nnc_init_mfa_context(device_owner.get());
  MTL::Device* const device = context->device.get();
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  NAScaledDotProductArgPartitionDescriptor descriptor;
  descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  descriptor.T = (uint32_t)config.T;
  descriptor.C = (uint32_t)config.C;
  descriptor.H = (uint32_t)config.H;
  descriptor.D = (uint32_t)config.D;
  descriptor.kth = (uint32_t)config.kth;
  descriptor.compressionRatio = (uint32_t)config.compression_ratio;
  descriptor.scale = scale;
  descriptor.isCausal = config.is_causal != 0;
  descriptor.scoreBlockM = score_block_m;
  descriptor.scoreBlockN = score_block_n;
  descriptor.scoreSIMDGroups = score_simdgroups;
  DeviceProperties dprops = DeviceProperties();
  auto pipeline_value = context->kernel_cache.findKernel<NAScaledDotProductArgPartitionKernel, NAScaledDotProductArgPartitionDescriptor, NAScaledDotProductArgPartitionKernelDescriptor>(descriptor, device, dprops);
  auto kernel = pipeline_value->kernel;
  const size_t q_bytes = (size_t)config.T * config.H * config.D * sizeof(uint16_t);
  const size_t k_bytes = (size_t)config.C * config.D * sizeof(uint16_t);
  const size_t head_w_bytes = (size_t)config.T * config.H * sizeof(uint16_t);
  const size_t score_bytes = std::max<size_t>((size_t)config.T * config.C * sizeof(float), sizeof(float));
  auto q_buffer = make_private_buffer(device, command_queue.get(), hq->data.f16, q_bytes);
  auto k_buffer = make_private_buffer(device, command_queue.get(), hk->data.f16, k_bytes);
  auto head_w_buffer = make_private_buffer(device, command_queue.get(), hhead_w->data.f16, head_w_bytes);
  auto scores = NS::TransferPtr(device->newBuffer(score_bytes, kPrivateResourceOptions));
  for (int i = 0; i < config.warmup; ++i)
    run_score_stage_once(command_queue.get(), pipeline_value->pipeline.get(), kernel->scoreThreadgroupSize, q_buffer.get(), k_buffer.get(), head_w_buffer.get(), scores.get(), config, kernel->scoreBlockM, kernel->scoreBlockN);
  std::vector<double> samples;
  samples.reserve(config.timed);
  for (int i = 0; i < config.timed; ++i)
    samples.push_back(run_score_stage_once(command_queue.get(), pipeline_value->pipeline.get(), kernel->scoreThreadgroupSize, q_buffer.get(), k_buffer.get(), head_w_buffer.get(), scores.get(), config, kernel->scoreBlockM, kernel->scoreBlockN));
  Stats stats = compute_stats(samples);
  ccv_nnc_deinit_mfa_context(context);
  return stats;
}

bool parse_positive(const char* const text, int* const value)
{
  const int parsed = std::atoi(text);
  if (parsed <= 0)
    return false;
  *value = parsed;
  return true;
}

void print_usage(const char* const argv0)
{
  std::cerr << "usage: " << argv0 << " [T C H D kth causal compression_ratio warmup timed [scan_score_tiles]]\n";
}

} // namespace

int main(int argc, char** argv)
{
  ccv_nnc_init();
  Config config;
  if (argc > 1 && argc != 10 && argc != 11)
  {
    print_usage(argv[0]);
    return 1;
  }
  if (argc >= 10)
  {
    if (!parse_positive(argv[1], &config.T) ||
        !parse_positive(argv[2], &config.C) ||
        !parse_positive(argv[3], &config.H) ||
        !parse_positive(argv[4], &config.D) ||
        !parse_positive(argv[5], &config.kth) ||
        !parse_positive(argv[7], &config.compression_ratio) ||
        !parse_positive(argv[8], &config.warmup) ||
        !parse_positive(argv[9], &config.timed))
    {
      print_usage(argv[0]);
      return 1;
    }
    config.is_causal = std::atoi(argv[6]) != 0;
    if (argc == 11)
      config.scan_score_tiles = std::atoi(argv[10]) != 0;
  }
  if (config.H != 64 || config.D != 128 || config.kth > 1024)
  {
    std::cerr << "MFA path requires H=64, D=128, kth<=1024 for this first DS4-native implementation\n";
    return 1;
  }
  if (ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) <= 0)
  {
    std::cerr << "no GPU stream context available\n";
    return 1;
  }
  if (!ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS) ||
      !ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS))
  {
    std::cerr << "required MPS commands are not available\n";
    return 1;
  }

  const float sdpap_scale = 1.0f / std::sqrt((float)(config.H * config.D));
  const float sdpa_scale = 1.0f / std::sqrt((float)config.D);
  const int q_count = config.T * config.H * config.D;
  const int k_count = config.C * config.D;
  const int head_w_count = config.T * config.H;
  const int sdpa_kv_count = config.C * config.H * config.D;

  std::vector<float> q_f32(q_count);
  std::vector<float> k_f32(k_count);
  std::vector<float> head_w_f32(head_w_count);
  std::vector<float> sdpa_q_f32(q_count);
  std::vector<float> sdpa_k_f32(sdpa_kv_count);
  std::vector<float> sdpa_v_f32(sdpa_kv_count);
  fill_sdpap_inputs(q_f32, k_f32, head_w_f32, config);
  fill_sdpa_inputs(sdpa_q_f32, sdpa_k_f32, sdpa_v_f32, q_f32, k_f32, config);

  ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, config.C, config.D), 0);
  ccv_nnc_tensor_t* const hhead_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, config.T, config.H), 0);
  ccv_nnc_tensor_t* const hsdpa_q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const hsdpa_k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, config.C, config.H, config.D), 0);
  ccv_nnc_tensor_t* const hsdpa_v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, config.C, config.H, config.D), 0);
  ccv_float_to_half_precision(q_f32.data(), (uint16_t*)hq->data.f16, q_count);
  ccv_float_to_half_precision(k_f32.data(), (uint16_t*)hk->data.f16, k_count);
  ccv_float_to_half_precision(head_w_f32.data(), (uint16_t*)hhead_w->data.f16, head_w_count);
  ccv_float_to_half_precision(sdpa_q_f32.data(), (uint16_t*)hsdpa_q->data.f16, q_count);
  ccv_float_to_half_precision(sdpa_k_f32.data(), (uint16_t*)hsdpa_k->data.f16, sdpa_kv_count);
  ccv_float_to_half_precision(sdpa_v_f32.data(), (uint16_t*)hsdpa_v->data.f16, sdpa_kv_count);

  ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, config.C, config.D), 0);
  ccv_nnc_tensor_t* const head_w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, config.T, config.H), 0);
  ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, config.T, config.kth), 0);
  ccv_nnc_tensor_t* const sdpa_q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const sdpa_k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.C, config.H, config.D), 0);
  ccv_nnc_tensor_t* const sdpa_v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.C, config.H, config.D), 0);
  ccv_nnc_tensor_t* const sdpa_o = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
  ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hhead_w, hsdpa_q, hsdpa_k, hsdpa_v), TENSOR_LIST(q, k, head_w, sdpa_q, sdpa_k, sdpa_v), stream);
  ccv_nnc_stream_context_wait(stream);

  ccv_nnc_cmd_t sdpap_cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(config.kth, sdpap_scale, config.is_causal, config.compression_ratio);
  ccv_nnc_cmd_t sdpa_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(sdpa_scale, config.is_causal);
  sdpa_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;

  std::cout << "shape"
            << " T=" << config.T
            << " C=" << config.C
            << " H=" << config.H
            << " D=" << config.D
            << " kth=" << config.kth
            << " causal=" << config.is_causal
            << " compression_ratio=" << config.compression_ratio
            << " warmup=" << config.warmup
            << " timed=" << config.timed
            << " dtype=fp16\n";

  const uint64_t saved_flags = ccv_nnc_flags();
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  print_stats("sdpap_mfa", benchmark_sdpap(sdpap_cmd, q, k, head_w, selected, stream, config.warmup, config.timed));
  const StageProfile current_mfa_profile = benchmark_mfa_stages<NAScaledDotProductArgPartitionDescriptor, NAScaledDotProductArgPartitionKernel, NAScaledDotProductArgPartitionKernelDescriptor>(config, hq, hk, hhead_w, sdpap_scale);
  print_stats("sdpap_mfa_score_stage_gpu", current_mfa_profile.score);
  print_stats("sdpap_mfa_topk_tile_stage_gpu", current_mfa_profile.topk_tile);
  print_stats("sdpap_mfa_topk_merge_stage_gpu", current_mfa_profile.topk_merge);
  if (config.scan_score_tiles)
  {
    struct ScoreTile {
      uint16_t block_m;
      uint16_t block_n;
      uint16_t simdgroups;
    };
    const ScoreTile tiles[] = {
      { 8, 32, 4 },
      { 8, 48, 4 },
      { 8, 64, 4 },
      { 16, 32, 4 },
      { 16, 48, 4 },
      { 16, 64, 4 },
      { 8, 32, 8 },
      { 8, 64, 8 },
      { 16, 32, 8 },
      { 16, 64, 8 },
    };
    for (const ScoreTile& tile : tiles)
    {
      const Stats stats = benchmark_score_stage_variant(config, hq, hk, hhead_w, sdpap_scale, tile.block_m, tile.block_n, tile.simdgroups);
      std::cout << "score_tile_m" << tile.block_m << "_n" << tile.block_n << "_sg" << tile.simdgroups;
      std::cout << " avg_ms=" << stats.average_ms
                << " median_ms=" << stats.median_ms
                << " min_ms=" << stats.min_ms
                << " max_ms=" << stats.max_ms
                << "\n";
    }
  }

  ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  print_stats("sdpap_mfa_generic", benchmark_sdpap(sdpap_cmd, q, k, head_w, selected, stream, config.warmup, config.timed));
  const StageProfile generic_mfa_profile = benchmark_mfa_stages<ScaledDotProductArgPartitionDescriptor, ScaledDotProductArgPartitionKernel, ScaledDotProductArgPartitionKernelDescriptor>(config, hq, hk, hhead_w, sdpap_scale);
  print_stats("sdpap_mfa_generic_score_stage_gpu", generic_mfa_profile.score);
  print_stats("sdpap_mfa_generic_topk_tile_stage_gpu", generic_mfa_profile.topk_tile);
  print_stats("sdpap_mfa_generic_topk_merge_stage_gpu", generic_mfa_profile.topk_merge);

  ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
  print_stats("sdpap_mpsgraph", benchmark_sdpap(sdpap_cmd, q, k, head_w, selected, stream, config.warmup, config.timed));

  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ATTENTION);
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  print_stats("sdpa_na_attention", benchmark_sdpa(sdpa_cmd, sdpa_q, sdpa_k, sdpa_v, sdpa_o, stream, config.warmup, config.timed));

  restore_flag(saved_flags, CCV_NNC_DISABLE_MFA);
  restore_flag(saved_flags, CCV_NNC_DISABLE_MFA_ATTENTION);
  restore_flag(saved_flags, CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);

  ccv_nnc_stream_context_free(stream);
  ccv_nnc_tensor_free(sdpa_o);
  ccv_nnc_tensor_free(sdpa_v);
  ccv_nnc_tensor_free(sdpa_k);
  ccv_nnc_tensor_free(sdpa_q);
  ccv_nnc_tensor_free(selected);
  ccv_nnc_tensor_free(head_w);
  ccv_nnc_tensor_free(k);
  ccv_nnc_tensor_free(q);
  ccv_nnc_tensor_free(hsdpa_v);
  ccv_nnc_tensor_free(hsdpa_k);
  ccv_nnc_tensor_free(hsdpa_q);
  ccv_nnc_tensor_free(hhead_w);
  ccv_nnc_tensor_free(hk);
  ccv_nnc_tensor_free(hq);
  return 0;
}
