#include "ccv_nnc_mfa.hpp"
#include "kernels/NAInt8SolAttentionDescriptor.hpp"
#include "kernels/NAInt8SolAttentionKernel.hpp"
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
  // This struct is also Metal's Params. Keep the scalar ABI explicit.
  static_assert(sizeof(params) == 40 && offsetof(ccv_nnc_mfa_sol_attention_params_t, query_block_size) == 32, "Sol Metal parameter layout");
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
    encoder->setBytes(&params, sizeof(params), 21);
    encoder->dispatchThreadgroups(value->kernel->threadgroupsPerGrid(descriptor), MTL::Size(value->kernel->threadgroupSize(descriptor), 1, 1));
    command_batch->finishCommand(encoder);
  }
}
