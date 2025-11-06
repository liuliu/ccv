#include "SegmentedGEMMPrologueKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

SegmentedGEMMPrologueKernel::SegmentedGEMMPrologueKernel(SegmentedGEMMPrologueKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecisions = descriptor.memoryPrecisions;
  useBias = descriptor.useBias;
  splitK = descriptor.splitK;

  source = createSource();

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
  source.SetValue("SPLIT_K", std::to_string(splitK));

  source += R"(

using namespace metal;

struct Arguments {
  command_buffer icb1 [[id(0)]]; // Assign an explicit ID for the encoder
  compute_pipeline_state pipeline1 [[id(1)]];
  command_buffer icb2 [[id(2)]]; // Assign an explicit ID for the encoder
  compute_pipeline_state pipeline2 [[id(3)]];
};

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

constant uint M_block [[function_constant(3)]];
constant uint N_block [[function_constant(4)]];

constant uint threadgroup_size [[function_constant(5)]];
constant uint threadgroup_memory_allocation [[function_constant(6)]];
constant bool dispatch_m_major [[function_constant(7)]];

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
    if (splitK > 1) {
      source += R"(
                 device {{MEMORY_NAME_C}} *D [[buffer(7)]],
)";
    }
  } else {
    source += R"(
                 device Arguments *args [[buffer(5)]],
)";
    if (splitK > 1) {
      source += R"(
                 device {{MEMORY_NAME_C}} *D [[buffer(6)]],
)";
    }
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
  for (uint i = 0; i < gid; i++)
    offset += counts[i];
  compute_command cmd = compute_command(args->icb1, gid);
  const int idx = indices[gid];
  const int count = counts[gid];
  cmd.reset();
  cmd.set_compute_pipeline_state(args->pipeline1);
  cmd.set_threadgroup_memory_length(threadgroup_memory_allocation, 0);
  cmd.set_kernel_buffer(A + offset * K, 0);
  cmd.set_kernel_buffer(B + idx * (N * K), 1);
)";
  if (splitK > 1) {
    source += R"(
  cmd.set_kernel_buffer(D + offset * N * {{SPLIT_K}}, 2);
)";
  } else {
    source += R"(
  cmd.set_kernel_buffer(C + offset * N, 2);
)";
  }
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
)";
  if (splitK > 1) {
    source += R"(
  if (dispatch_m_major) {
    cmd.concurrent_dispatch_threadgroups(uint3((count + M_block - 1) / M_block * {{SPLIT_K}}, (N + N_block - 1) / N_block, 1), uint3(threadgroup_size, 1, 1));
  } else {
    cmd.concurrent_dispatch_threadgroups(uint3((N + N_block - 1) / N_block * {{SPLIT_K}}, (count + M_block - 1) / M_block, 1), uint3(threadgroup_size, 1, 1));
  }
  compute_command reduce_sum = compute_command(args->icb2, gid);
  reduce_sum.reset();
  reduce_sum.set_compute_pipeline_state(args->pipeline2);
  reduce_sum.set_kernel_buffer(D + offset * N * {{SPLIT_K}}, 0);
  reduce_sum.set_kernel_buffer(C + offset * N, 1);
  reduce_sum.set_kernel_buffer(counts + gid, 2);
  if ((N % 2) == 0) {
    reduce_sum.concurrent_dispatch_threadgroups(uint3((count * N / 2 + 255) / 256, 1, 1), uint3(256, 1, 1));
  } else {
    reduce_sum.concurrent_dispatch_threadgroups(uint3((count * N + 255) / 256, 1, 1), uint3(256, 1, 1));
  }
)";
  } else {
    source += R"(
  if (dispatch_m_major) {
    cmd.concurrent_dispatch_threadgroups(uint3((count + M_block - 1) / M_block, (N + N_block - 1) / N_block, 1), uint3(threadgroup_size, 1, 1));
  } else {
    cmd.concurrent_dispatch_threadgroups(uint3((N + N_block - 1) / N_block, (count + M_block - 1) / M_block, 1), uint3(threadgroup_size, 1, 1));
  }
)";
  }
  source += R"(
}
)";

  return source.ToString();
}
