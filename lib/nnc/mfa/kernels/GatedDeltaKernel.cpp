#include "GatedDeltaKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

GatedDeltaKernel::GatedDeltaKernel(GatedDeltaKernelDescriptor descriptor, MTL::Device* const device) {
  stateElementsPerLane = descriptor.stateElementsPerLane;
  inputMemoryPrecision = descriptor.inputMemoryPrecision;
  betaMemoryPrecision = descriptor.betaMemoryPrecision;

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

typedef )" + inputMemoryPrecision.name() + R"( input_t;
typedef )" + betaMemoryPrecision.name() + R"( beta_t;

constant uint B [[function_constant(0)]];
constant uint T [[function_constant(1)]];
constant uint Hk [[function_constant(2)]];
constant uint Hv [[function_constant(3)]];
constant uint Dk [[function_constant(4)]];
constant uint Dv [[function_constant(5)]];
constant bool log_decay_input [[function_constant(6)]];
constant bool key_dim_multiple_of_32 [[function_constant(7)]];
constant bool value_dim_multiple_of_4 [[function_constant(8)]];
constant uint hv_per_hk = Hv / Hk;
constant uint dv_per_threadgroup = 4;

kernel void gated_delta(
  device const input_t* q [[buffer(0)]],
  device const input_t* k [[buffer(1)]],
  device const input_t* v [[buffer(2)]],
  device const float* log_decay [[buffer(3)]],
  device const beta_t* beta [[buffer(4)]],
  device const float* state_in [[buffer(5)]],
  device input_t* y [[buffer(6)]],
  device float* state_out [[buffer(7)]],

  uint3 group [[threadgroup_position_in_grid]],
  uint3 tid [[thread_position_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  const uint dv = group.y * dv_per_threadgroup + tid.y;
  if (!value_dim_multiple_of_4 && dv >= Dv) {
    return;
  }
  const uint n = group.z;
  const uint b = n / Hv;
  const uint hv = n - b * Hv;
  const uint hk = hv / hv_per_hk;
  const uint state_offset = ((b * Hv + hv) * Dv + dv) * Dk;
  constexpr int state_elements_per_lane = )" + std::to_string(stateElementsPerLane) + R"(;

  float state[state_elements_per_lane];
  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    state[i] = (key_dim_multiple_of_32 || dk < Dk) ? state_in[state_offset + dk] : 0.0f;
  }

  device const input_t* q_ptr = q + (b * T * Hk + hk) * Dk;
  device const input_t* k_ptr = k + (b * T * Hk + hk) * Dk;
  device const input_t* v_ptr = v + (b * T * Hv + hv) * Dv + dv;
  device const float* decay_ptr = log_decay + (b * T * Hv + hv);
  device const beta_t* beta_ptr = beta + (b * T * Hv + hv);
  device input_t* y_ptr = y + (b * T * Hv + hv) * Dv + dv;

  for (uint t = 0; t < T; t++) {
    const float decay = log_decay_input ? simd_broadcast_first((lane_id == 0) ? precise::exp(decay_ptr[0]) : 0.0f) : decay_ptr[0];
    float memory = 0.0f;
    for (uint i = 0; i < state_elements_per_lane; i++) {
      const uint dk = state_elements_per_lane * lane_id + i;
      if (key_dim_multiple_of_32 || dk < Dk) {
        state[i] *= decay;
        memory += state[i] * static_cast<float>(k_ptr[dk]);
      }
    }
    memory = simd_sum(memory);

    const float delta = (static_cast<float>(v_ptr[0]) - memory) * static_cast<float>(beta_ptr[0]);
    float out = 0.0f;
    for (uint i = 0; i < state_elements_per_lane; i++) {
      const uint dk = state_elements_per_lane * lane_id + i;
      if (key_dim_multiple_of_32 || dk < Dk) {
        const float k_value = static_cast<float>(k_ptr[dk]);
        state[i] += delta * k_value;
        out += state[i] * static_cast<float>(q_ptr[dk]);
      }
    }
    out = simd_sum(out);
    if (lane_id == 0) {
      y_ptr[0] = static_cast<input_t>(out);
    }
    q_ptr += Hk * Dk;
    k_ptr += Hk * Dk;
    v_ptr += Hv * Dv;
    y_ptr += Hv * Dv;
    decay_ptr += Hv;
    beta_ptr += Hv;
  }

  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    if (key_dim_multiple_of_32 || dk < Dk) {
      state_out[state_offset + dk] = state[i];
    }
  }
}
  )";
  return shader;
}
