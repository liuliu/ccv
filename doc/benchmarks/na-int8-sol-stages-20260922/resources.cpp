#include <array>
#include <cstdio>
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "SolExperimentAttentionKernel.hpp"
#include "NAInt8SolAttentionKernel.hpp"
int main()
{
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  auto context = ccv_nnc_init_mfa_context(device.get());
  using Stage = SolExperimentAttentionStage;
  for (auto shape : {std::array<uint32_t,5>{2,257,1,16,16}, {2,20480,56,32,16},
      {2,20481,64,32,32}, {1,32769,3,64,16}, {1,32769,56,64,32},
      {1,32769,64,64,64}, {1,65536,56,64,64}, {1,103982,56,64,64}}) {
          const uint32_t N = shape[0], T = shape[1], H = shape[2], B = shape[3], Q = shape[4];
          for (auto stage : {NAInt8SolAttentionStage::VMean, NAInt8SolAttentionStage::Quantize,
              NAInt8SolAttentionStage::Pool, NAInt8SolAttentionStage::PrepareSummaries,
              NAInt8SolAttentionStage::Route, NAInt8SolAttentionStage::Attention}) {
            if (stage == NAInt8SolAttentionStage::Pool && B == 64) continue;
            const NAInt8SolAttentionDescriptor descriptor = {stage, B, N, T, H, Q};
            const auto value = context->kernel_cache.findKernel<NAInt8SolAttentionKernel, NAInt8SolAttentionDescriptor, NAInt8SolAttentionKernelDescriptor>(descriptor, device.get(), DeviceProperties());
            printf("PRODUCTION N=%u T=%u H=%u B=%u Q=%u stage=%u threads=%u memory=%zu cap=%zu\n",
              N, T, H, B, Q, uint32_t(stage), value->kernel->threadgroupSize(descriptor),
              size_t(value->pipeline->staticThreadgroupMemoryLength()), size_t(device->maxThreadgroupMemoryLength()));
          }
          for (Stage stage : {Stage::VMean, Stage::Quantize, Stage::Pool, Stage::Prepared, Stage::Route, Stage::Fused, Stage::Routed}) {
            if (stage == Stage::Pool && B == 64) continue;
            const SolExperimentAttentionDescriptor descriptor = {stage, B, N, T, H, Q};
            context->kernel_cache.findKernel<SolExperimentAttentionKernel, SolExperimentAttentionDescriptor, SolExperimentAttentionKernelDescriptor>(descriptor, device.get(), DeviceProperties());
          }
        }
  ccv_nnc_deinit_mfa_context(context);
  printf("All resource caps passed.\n");
}
