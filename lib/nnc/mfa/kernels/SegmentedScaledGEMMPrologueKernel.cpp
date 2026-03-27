#include "SegmentedScaledGEMMPrologueKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

SegmentedScaledGEMMPrologueKernel::SegmentedScaledGEMMPrologueKernel(SegmentedScaledGEMMPrologueKernelDescriptor descriptor, MTL::Device *const device)
{
  ioPrecision = descriptor.ioPrecision;
  useBias = descriptor.useBias;
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string SegmentedScaledGEMMPrologueKernel::createSource() const noexcept
{
  CodeWriter source;
  source.SetValue("MEMORY_NAME", ioPrecision.name());
  source += R"(

using namespace metal;

struct Arguments {
  command_buffer icb1 [[id(0)]];
  compute_pipeline_state pipeline1 [[id(1)]];
};

struct Offsets {
  uint A_scale_offset;
  uint B_scale_offset;
};

constant uint segments [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];
constant uint M_block [[function_constant(3)]];
constant uint N_block [[function_constant(4)]];
constant uint threadgroup_size [[function_constant(5)]];

kernel void segmented_scaled_gemm_prologue(device char *A_storage [[buffer(0)]],
                 device int *indices [[buffer(1)]],
                 device int *counts [[buffer(2)]],
                 device char *B_storage [[buffer(3)]],
                 device {{MEMORY_NAME}} *C [[buffer(4)]],
                 constant Offsets& offsets [[buffer(5)]],
)";
  if (useBias) {
    source += R"(
                 device {{MEMORY_NAME}} *bias [[buffer(6)]],
                 device Arguments *args [[buffer(7)]],
)";
  } else {
    source += R"(
                 device Arguments *args [[buffer(6)]],
)";
  }
  source += R"(
                 uint gid [[thread_position_in_grid]])
{
  if (gid >= segments)
    return;
  if (counts[gid] <= 0)
    return;
  int offset = 0;
  for (uint i = 0; i < gid; i++)
    offset += counts[i];
  compute_command cmd = compute_command(args->icb1, gid);
  const int idx = indices[gid];
  const int count = counts[gid];
  device int8_t *A = reinterpret_cast<device int8_t *>(A_storage);
  device int8_t *B = reinterpret_cast<device int8_t *>(B_storage);
  device {{MEMORY_NAME}} *A_scale = reinterpret_cast<device {{MEMORY_NAME}} *>(A_storage + offsets.A_scale_offset);
  device {{MEMORY_NAME}} *B_scale = reinterpret_cast<device {{MEMORY_NAME}} *>(B_storage + offsets.B_scale_offset);
  cmd.reset();
  cmd.set_compute_pipeline_state(args->pipeline1);
  cmd.set_threadgroup_memory_length(0, 0);
  cmd.set_kernel_buffer(A + offset * K, 0);
  cmd.set_kernel_buffer(B + idx * (N * K), 1);
  cmd.set_kernel_buffer(C + offset * N, 2);
  cmd.set_kernel_buffer(A_scale + offset, 3);
  cmd.set_kernel_buffer(B_scale + idx * N, 4);
)";
  if (useBias) {
    source += R"(
  cmd.set_kernel_buffer(bias + idx * N, 5);
  cmd.set_kernel_buffer(counts + gid, 6);
)";
  } else {
    source += R"(
  cmd.set_kernel_buffer(counts + gid, 5);
)";
  }
  source += R"(
  const uint M_blocks = (count + M_block - 1) / M_block;
  const uint N_blocks = (N + N_block - 1) / N_block;
  const uint M_block_bits = M_blocks <= 1 ? 0 : 32 - clz(M_blocks - 1);
  const uint N_block_bits = N_blocks <= 1 ? 0 : 32 - clz(N_blocks - 1);
  cmd.concurrent_dispatch_threadgroups(uint3(1u << (M_block_bits + N_block_bits), 1, 1), uint3(threadgroup_size, 1, 1));
}
)";
  return source.ToString();
}
