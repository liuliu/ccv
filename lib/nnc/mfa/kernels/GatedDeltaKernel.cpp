#include "GatedDeltaKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

GatedDeltaKernel::GatedDeltaKernel(GatedDeltaKernelDescriptor descriptor, MTL::Device* const device) {
  value = descriptor.value;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = MTL::Size(256, 1, 1);

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short GatedDeltaKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 256 * sizeof(float);
}

std::string GatedDeltaKernel::createSource() const noexcept {
  std::string shader = R"(
#include <metal_stdlib>
using namespace metal;

constant uint B [[function_constant(0)]];
constant uint T [[function_constant(1)]];
constant uint Hk [[function_constant(2)]];
constant uint Hv [[function_constant(3)]];
constant uint Dk [[function_constant(4)]];
constant uint Dv [[function_constant(5)]];
constant uint hv_per_hk = Hv / Hk;
constant uint threadgroup_size = 256;

kernel void gated_delta(
  device const float* q [[buffer(0)]],
  device const float* k [[buffer(1)]],
  device const float* v [[buffer(2)]],
  device const float* log_decay [[buffer(3)]],
  device const float* beta [[buffer(4)]],
  device const float* state_in [[buffer(5)]],
  device float* y [[buffer(6)]],
  device float* state_out [[buffer(7)]],
  threadgroup float* partials [[threadgroup(0)]],

  uint group [[threadgroup_position_in_grid]],
  uint tid [[thread_position_in_threadgroup]]
) {
  const uint dv = group % Dv;
  const uint hv = (group / Dv) % Hv;
  const uint b = group / (Dv * Hv);
  const uint hk = hv / hv_per_hk;
  const uint state_offset = ((b * Hv + hv) * Dv + dv) * Dk;

  for (uint dk = tid; dk < Dk; dk += threadgroup_size) {
    state_out[state_offset + dk] = state_in[state_offset + dk];
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint t = 0; t < T; t++) {
    const uint qk_offset = ((b * T + t) * Hk + hk) * Dk;
    const uint gate_offset = (b * T + t) * Hv + hv;
    const float decay = precise::exp(log_decay[gate_offset]);
    float memory = 0.0f;
    for (uint dk = tid; dk < Dk; dk += threadgroup_size) {
      const uint idx = state_offset + dk;
      const float decayed = state_out[idx] * decay;
      state_out[idx] = decayed;
      memory += decayed * k[qk_offset + dk];
    }
    partials[tid] = memory;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threadgroup_size / 2; offset > 0; offset /= 2) {
      if (tid < offset) {
        partials[tid] += partials[tid + offset];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float kv_mem = partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint v_offset = ((b * T + t) * Hv + hv) * Dv + dv;
    const float delta = (v[v_offset] - kv_mem) * beta[gate_offset];
    float out = 0.0f;
    for (uint dk = tid; dk < Dk; dk += threadgroup_size) {
      const uint idx = state_offset + dk;
      const float next = state_out[idx] + delta * k[qk_offset + dk];
      state_out[idx] = next;
      out += next * q[qk_offset + dk];
    }
    partials[tid] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = threadgroup_size / 2; offset > 0; offset /= 2) {
      if (tid < offset) {
        partials[tid] += partials[tid + offset];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
      y[v_offset] = partials[0];
    }
  }
}
  )";
  return shader;
}
