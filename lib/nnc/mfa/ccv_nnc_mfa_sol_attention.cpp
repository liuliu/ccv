#include "ccv_nnc_mfa.hpp"
#include "kernels/NAInt8SolAttentionDescriptor.hpp"
#include "kernels/NAInt8SolAttentionKernel.hpp"
#include "kernels/SolAttentionKernel.hpp"
#include <cmath>
#include <cstddef>

void ccv_nnc_mfa_encode_sol_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_sol_attention_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.N > 0 && params.T > 0 && params.H > 0);
  CCV_NNC_MFA_PRECONDITION(params.block_size == 16 || params.block_size == 32 || params.block_size == 64);
  CCV_NNC_MFA_PRECONDITION(params.query_block_size == 16 || params.query_block_size == 32 || params.query_block_size == 64);
  CCV_NNC_MFA_PRECONDITION(params.query_block_size <= params.block_size && params.local_block_radius >= 1);
  CCV_NNC_MFA_PRECONDITION(params.approximation_start <= params.approximation_end && params.approximation_end <= params.T);
  CCV_NNC_MFA_PRECONDITION(std::isfinite(params.scale) && std::isfinite(params.tau));
  // Only the first 40 bytes are Metal's Params; backend selection is host-only.
  constexpr size_t metal_params_size = offsetof(ccv_nnc_mfa_sol_attention_params_t, use_neural_accelerators);
  static_assert(metal_params_size == 40 && offsetof(ccv_nnc_mfa_sol_attention_params_t, query_block_size) == 32, "Sol Metal parameter layout");
  if (params.use_neural_accelerators) {
    const uint32_t J = (params.T + params.block_size - 1) / params.block_size, NH = params.N * params.H;
    const uint32_t QJ = (params.T + params.query_block_size - 1) / params.query_block_size;
    const uint32_t JP = (J + 63) / 64 * 64, TP = (params.T + 63) / 64 * 64;
    const size_t count = size_t(NH) * JP * 128;
    // Pooled queries/keys/centered values, routing statistics, and byte routes.
    const size_t qc = 0, kc = size_t(NH) * QJ * 128 * 4, vc = kc + count * 2;
    const size_t stats = vc + count * 4, routes = stats + size_t(NH) * 128 * 8;
    const size_t qi = (routes + size_t(NH) * QJ * J + 255) / 256 * 256;
    const size_t token_bytes = size_t(NH) * TP * 128;
    const size_t ki = qi + token_bytes, vi = ki + token_bytes;
    const size_t qs = vi + token_bytes, ks = qs + size_t(NH) * TP / 16 * 4;
    const size_t vs = ks + size_t(NH) * TP / 64 * 4;
    // The single-pass reduction writes float4 means, including for odd head counts.
    const size_t mean = (vs + size_t(NH) * TP / 64 * 4 + 15) / 16 * 16;
    const size_t kci = (mean + size_t(NH) * 128 * 4 + 255) / 256 * 256;
    const size_t vch = kci + count, kcs = vch + count * 2, vcs = kcs + size_t(NH) * (JP / 64) * 4;
    const size_t route_bits = vcs + size_t(NH) * (JP / 64) * 4;
    auto scratch = context->request_scratch(route_bits + size_t(NH) * QJ * ((J + 31) / 32) * 4);
    const size_t offsets[] = { qc, kc, vc, stats, routes, qi, ki, vi, qs, ks, vs, mean, kci, vch, kcs, vcs, route_bits };
    using Stage = NAInt8SolAttentionStage;
    for (Stage stage : { Stage::VMean, Stage::Quantize, Stage::Pool, Stage::PrepareSummaries, Stage::Route, Stage::Attention }) {
      // B64 pooling is fused into token quantization. Consumers only read real
      // KC/VC entries; PrepareSummaries initializes its own padded KCI/VCH tail.
      if (stage == Stage::Pool && params.block_size == 64)
        continue;
      const NAInt8SolAttentionDescriptor descriptor = { stage, params.block_size, params.N, params.T, params.H, params.query_block_size };
      auto value = context->kernel_cache.findKernel<NAInt8SolAttentionKernel, NAInt8SolAttentionDescriptor, NAInt8SolAttentionKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(value->pipeline.get());
      for (int i = 0; i < 4; ++i) {
        encoder->setBuffer(tensors[i], tensor_offsets[i], i);
        encoder->useResource(tensors[i], i == 3 ? MTL::ResourceUsageWrite : MTL::ResourceUsageRead);
      }
      for (int i = 0; i < 17; ++i)
        encoder->setBuffer(scratch, offsets[i], i + 4);
      encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->setBytes(&params, metal_params_size, 21);
      encoder->dispatchThreadgroups(value->kernel->threadgroupsPerGrid(descriptor), MTL::Size(value->kernel->threadgroupSize(descriptor), 1, 1));
      command_batch->finishCommand(encoder);
    }
    return;
  }
  const uint32_t J = (params.T + params.block_size - 1) / params.block_size, NH = params.N * params.H;
  const uint32_t QJ = (params.T + params.query_block_size - 1) / params.query_block_size;
  const uint32_t JP = (J + 63) / 64 * 64;
  const size_t count = size_t(NH) * JP * 128;
  const size_t qc = 0, kc = size_t(NH) * QJ * 128 * 4, vc = kc + count * 2;
  const size_t stats = vc + count * 2, routes = stats + size_t(NH) * 128 * 8;
  const size_t counts = (routes + size_t(NH) * QJ * J + 255) / 256 * 256;
  // Bit traversal and periodic SIMD-group synchronization help long B64/Q64
  // mixed workloads. Shorter sequences retain the cheaper byte traversal.
  const bool use_route_bits = params.T >= 32768 && params.block_size == 64 && params.query_block_size == 64 && (J + 31) / 32 <= 128;
  // Routing consumes each pooled-query row before replacing it with bits.
  // Limit bit traversal to maps fitting that row; no extra scratch is needed.
  const size_t route_bits = qc;
  CCV_NNC_MFA_PRECONDITION(size_t(NH) * params.T * 128 <= UINT32_MAX);
  // Store the FP32 register accumulator directly to the FP16 output.
  const size_t l = (counts + size_t(NH) * QJ * 4 + 255) / 256 * 256;
  auto scratch = context->request_scratch(l + size_t(NH) * params.T * 4);
  const size_t offsets[] = {qc, kc, vc, stats, routes};
  for (uint32_t entry : {0u, 1u, 2u}) {
    auto value = context->kernel_cache.findKernel<SolAttentionKernel, SolAttentionPreparationDescriptor, SolAttentionKernelKey>(
      {entry, params.block_size, params.N, params.T, params.H, params.query_block_size, use_route_bits}, context->device.get(), DeviceProperties());
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(value->pipeline.get());
    for (int i = 0; i < 3; ++i) {
      encoder->setBuffer(tensors[i], tensor_offsets[i], i);
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    }
    for (int i = 0; i < 5; ++i) encoder->setBuffer(scratch, offsets[i], i + 4);
    encoder->setBuffer(scratch, counts, 29);
    if (use_route_bits) encoder->setBuffer(scratch, route_bits, 30);
    encoder->setBytes(&params, metal_params_size, 10);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->dispatchThreadgroups(entry == 1 ? MTL::Size(NH, 1, 1) : MTL::Size(entry == 0 ? JP : QJ, NH, 1), MTL::Size(128, 1, 1));
    command_batch->finishCommand(encoder);
  }
  auto value = context->kernel_cache.findKernel<SolAttentionKernel, SolAttentionDescriptor, SolAttentionKernelKey>(
    {params.N, params.T, params.H, params.block_size, params.query_block_size, params.scale, use_route_bits}, context->device.get(), DeviceProperties());
  for (int phase = 0; phase < 2; ++phase) {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(phase == 0 ? value->second.get() : value->pipeline.get());
    encoder->setThreadgroupMemoryLength(phase == 0 ? value->kernel->denseThreadgroupMemoryAllocation : value->kernel->threadgroupMemoryAllocation, 0);
    for (int i = 0; i < 3; ++i) {
      encoder->setBuffer(tensors[i], tensor_offsets[i], i);
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    }
    encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
    encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, l, 4);
    encoder->setBuffer(scratch, kc, 5);
    encoder->setBuffer(scratch, vc, 6);
    encoder->setBuffer(scratch, routes, 8);
    encoder->setBuffer(scratch, counts, 9);
    if (use_route_bits) encoder->setBuffer(scratch, route_bits, 10);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->dispatchThreadgroups(MTL::Size(size_t(NH) * ((params.T + value->kernel->blockDimensions[0] - 1) / value->kernel->blockDimensions[0]), 1, 1), MTL::Size(value->kernel->threadgroupSize, 1, 1));
    command_batch->finishCommand(encoder);
  }
}
