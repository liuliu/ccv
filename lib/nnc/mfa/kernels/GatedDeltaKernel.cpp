#include "GatedDeltaKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

GatedDeltaKernel::GatedDeltaKernel(GatedDeltaKernelDescriptor descriptor, MTL::Device* const device) {
  stateElementsPerLane = descriptor.stateElementsPerLane;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = MTL::Size(32, 4, 1);

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short GatedDeltaKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
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
constant uint dv_per_threadgroup = 4;
constant uint state_elements_per_lane = )" + std::to_string(stateElementsPerLane) + R"(;

kernel void gated_delta(
  device const float* q [[buffer(0)]],
  device const float* k [[buffer(1)]],
  device const float* v [[buffer(2)]],
  device const float* log_decay [[buffer(3)]],
  device const float* beta [[buffer(4)]],
  device const float* state_in [[buffer(5)]],
  device float* y [[buffer(6)]],
  device float* state_out [[buffer(7)]],

  uint3 group [[threadgroup_position_in_grid]],
  uint3 tid [[thread_position_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  const uint dv = group.x * dv_per_threadgroup + tid.y;
  if (dv >= Dv) {
    return;
  }
  const uint hv = group.y;
  const uint b = group.z;
  const uint hk = hv / hv_per_hk;
  const uint state_offset = ((b * Hv + hv) * Dv + dv) * Dk;

  float state[state_elements_per_lane];
  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    state[i] = (dk < Dk) ? state_in[state_offset + dk] : 0.0f;
  }

  device const float* q_ptr = q + (b * T * Hk + hk) * Dk;
  device const float* k_ptr = k + (b * T * Hk + hk) * Dk;
  device const float* v_ptr = v + (b * T * Hv + hv) * Dv + dv;
  device const float* log_decay_ptr = log_decay + (b * T * Hv + hv);
  device const float* beta_ptr = beta + (b * T * Hv + hv);
  device float* y_ptr = y + (b * T * Hv + hv) * Dv + dv;

  for (uint t = 0; t < T; t++) {
    float decay = (lane_id == 0) ? precise::exp(log_decay_ptr[0]) : 0.0f;
    decay = simd_broadcast_first(decay);
    float memory = 0.0f;
    for (uint i = 0; i < state_elements_per_lane; i++) {
      const uint dk = state_elements_per_lane * lane_id + i;
      if (dk < Dk) {
        state[i] *= decay;
        memory += state[i] * k_ptr[dk];
      }
    }
    memory = simd_sum(memory);

    const float delta = (v_ptr[0] - memory) * beta_ptr[0];
    float out = 0.0f;
    for (uint i = 0; i < state_elements_per_lane; i++) {
      const uint dk = state_elements_per_lane * lane_id + i;
      if (dk < Dk) {
        state[i] += delta * k_ptr[dk];
        out += state[i] * q_ptr[dk];
      }
    }
    out = simd_sum(out);
    if (lane_id == 0) {
      y_ptr[0] = out;
    }
    q_ptr += Hk * Dk;
    k_ptr += Hk * Dk;
    v_ptr += Hv * Dv;
    y_ptr += Hv * Dv;
    log_decay_ptr += Hv;
    beta_ptr += Hv;
  }

  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    if (dk < Dk) {
      state_out[state_offset + dk] = state[i];
    }
  }
}
  )";
  return shader;
}
