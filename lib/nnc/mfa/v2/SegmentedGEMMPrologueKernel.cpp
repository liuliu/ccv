#include "SegmentedGEMMPrologueKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

SegmentedGEMMPrologueKernel::SegmentedGEMMPrologueKernel(SegmentedGEMMPrologueKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecisions = descriptor.memoryPrecisions;
  useBias = descriptor.useBias;
  threadgroupSize = 1;
  
  source = createSource();

  threadgroupMemoryAllocation = 0;

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

#pragma mark - Source

std::string SegmentedGEMMPrologueKernel::createSource() const noexcept {
  CodeWriter source;

  source.SetValue("MEMORY_NAME_A", memoryPrecisions.A.name());
  source.SetValue("MEMORY_NAME_B", memoryPrecisions.B.name());
  source.SetValue("MEMORY_NAME_C", memoryPrecisions.C.name());
  source.SetValue("MEMORY_NAME_BIAS", memoryPrecisions.bias.name());

  source += R"(

using namespace metal;

struct Arguments {
  command_buffer icb [[id(0)]]; // Assign an explicit ID for the encoder
  compute_pipeline_state pipeline [[id(1)]];
};

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

constant uint M_block [[function_constant(3)]];
constant uint N_block [[function_constant(4)]];

constant uint threadgroup_size [[function_constant(5)]];
constant uint threadgroup_memory_allocation [[function_constant(6)]];

kernel void segmented_gemm_prologue(device {{MEMORY_NAME_A}} *A [[buffer(0)]],
                 device int *indices [[buffer(1)]],
                 device int *counts [[buffer(2)]],
                 device {{MEMORY_NAME_B}} *B [[buffer(3)]],
                 device {{MEMORY_NAME_C}} *C [[buffer(4)]],
)";
  if (useBias) {
    source += R"(
                 device {{MEMORY_NAME_BIAS}} *bias [[buffer(5)]],
                 device Arguments *args [[buffer(6)]],
)";
  } else {
    source += R"(
                 device Arguments *args [[buffer(5)]],
)";
  }
  source += R"(
                 uint gid [[thread_position_in_grid]]
)
{
  if (gid >= M)
    return;
  if (counts[gid] <= 0)
    return;
  int offset = 0;
  for (int i = 0; i < gid; i++)
    offset += counts[i];
  compute_command cmd = compute_command(args->icb, gid);
  const int idx = indices[gid];
  const int count = counts[gid];
  cmd.reset();
  cmd.set_compute_pipeline_state(args->pipeline);
  cmd.set_threadgroup_memory_length(threadgroup_memory_allocation, 0);
  cmd.set_kernel_buffer(A + offset * K, 0);
  cmd.set_kernel_buffer(B + idx * (N * K), 1);
  cmd.set_kernel_buffer(C + offset * N, 2);
)";
  if (useBias) {
    source += R"(
  cmd.set_kernel_buffer(bias + idx * N, 3);
  cmd.set_kernel_buffer(counts + gid, 4);
)";
  } else {
    source += R"(
  cmd.set_kernel_buffer(counts + gid, 3);
)";
  }
    source += R"(
  cmd.concurrent_dispatch_threadgroups(uint3((N + N_block - 1) / N_block, (count + M_block - 1) / M_block, 1), uint3(threadgroup_size, 1, 1));
}
)";

  return source.ToString();
}
