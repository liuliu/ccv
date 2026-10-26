#include "GatedDeltaKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

GatedDeltaKernel::GatedDeltaKernel(GatedDeltaKernelDescriptor descriptor, MTL::Device* const device) {
  stateElementsPerLane = descriptor.stateElementsPerLane;
  stateCheckpointing = descriptor.stateCheckpointing;
  inputMemoryPrecision = descriptor.inputMemoryPrecision;
  betaMemoryPrecision = descriptor.betaMemoryPrecision;
  loadM = descriptor.loadM;
  blocked = descriptor.blocked;
  channelDecay = descriptor.channelDecay;
  timeBlock = descriptor.timeBlock;
  simdGroups = descriptor.simdGroups;
  columnsPerThreadgroup = blocked ? 4 * simdGroups : 4;

  source = blocked ? createBlockedSource() : createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = blocked ? MTL::Size(32, simdGroups, 1) : MTL::Size(32, 4, 1);

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
  const std::string sequenceLengthConstant = loadM ? "" : R"(constant uint T [[function_constant(1)]];
)";
  const std::string loadMArgument = loadM ? R"(  const device uint* loadM [[buffer(8)]],
)" : "";
  const std::string loadMValue = loadM ? R"(  const uniform<uint> T = make_uniform(loadM[0]);
)" : "";
  const std::string checkpointConstants = stateCheckpointing ? R"(constant uint state_checkpoint_count [[function_constant(9)]];
)" : "";
  const std::string historyCountConstant = stateCheckpointing ? R"(constant uint state_history_count = state_checkpoint_count + 1;
)" : "";
  const std::string stateOffsetLine = stateCheckpointing
    ? "  const uint state_in_offset = ((b * Hv + hv) * Dv + dv) * Dk;\n"
    : "  const uint state_offset = ((b * Hv + hv) * Dv + dv) * Dk;\n";
  const std::string stateInputOffset = stateCheckpointing ? "state_in_offset" : "state_offset";
  const std::string checkpointStore = stateCheckpointing ? R"(    const uint history_index = T - 1 - t;
    if (history_index > 0 && history_index <= state_checkpoint_count) {
      const uint state_out_offset = ((b * state_history_count * Hv + history_index * Hv + hv) * Dv + dv) * Dk;
      for (uint i = 0; i < state_elements_per_lane; i++) {
        const uint dk = state_elements_per_lane * lane_id + i;
        if (key_dim_multiple_of_32 || dk < Dk) {
          state_out[state_out_offset + dk] = state[i];
        }
      }
    }
)" : "";
  const std::string finalStateStore = stateCheckpointing ? R"(
  const uint state_out_offset = ((b * state_history_count * Hv + hv) * Dv + dv) * Dk;
  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    if (key_dim_multiple_of_32 || dk < Dk) {
      state_out[state_out_offset + dk] = state[i];
    }
  }
)" : R"(
  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    if (key_dim_multiple_of_32 || dk < Dk) {
      state_out[state_offset + dk] = state[i];
    }
  }
)";
  std::string shader = R"(
#include <metal_stdlib>
using namespace metal;

typedef )" + inputMemoryPrecision.name() + R"( input_t;
typedef )" + betaMemoryPrecision.name() + R"( beta_t;

constant uint B [[function_constant(0)]];
)" + sequenceLengthConstant + R"(constant uint Hk [[function_constant(2)]];
constant uint Hv [[function_constant(3)]];
constant uint Dk [[function_constant(4)]];
constant uint Dv [[function_constant(5)]];
constant bool log_decay_input [[function_constant(6)]];
constant bool key_dim_multiple_of_32 [[function_constant(7)]];
constant bool value_dim_multiple_of_4 [[function_constant(8)]];
constant bool channel_decay [[function_constant(10)]];
constant uint decay_stride = channel_decay ? Dk : 1;
)" + checkpointConstants + R"(constant uint hv_per_hk = Hv / Hk;
constant uint dv_per_threadgroup = 4;
)" + historyCountConstant + R"(
kernel void gated_delta(
  device const input_t* q [[buffer(0)]],
  device const input_t* k [[buffer(1)]],
  device const input_t* v [[buffer(2)]],
  device const float* log_decay [[buffer(3)]],
  device const beta_t* beta [[buffer(4)]],
  device const float* state_in [[buffer(5)]],
  device input_t* y [[buffer(6)]],
  device float* state_out [[buffer(7)]],
)" + loadMArgument + R"(
  uint3 group [[threadgroup_position_in_grid]],
  uint3 tid [[thread_position_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
)" + loadMValue + R"(  const uint dv = group.y * dv_per_threadgroup + tid.y;
  if (!value_dim_multiple_of_4 && dv >= Dv) {
    return;
  }
  const uint n = group.z;
  const uint b = n / Hv;
  const uint hv = n - b * Hv;
  const uint hk = hv / hv_per_hk;
)" + stateOffsetLine + R"(  constexpr int state_elements_per_lane = )" + std::to_string(stateElementsPerLane) + R"(;

  float state[state_elements_per_lane];
  for (uint i = 0; i < state_elements_per_lane; i++) {
    const uint dk = state_elements_per_lane * lane_id + i;
    state[i] = (key_dim_multiple_of_32 || dk < Dk) ? state_in[)" + stateInputOffset + R"( + dk] : 0.0f;
  }

  device const input_t* q_ptr = q + (b * T * Hk + hk) * Dk;
  device const input_t* k_ptr = k + (b * T * Hk + hk) * Dk;
  device const input_t* v_ptr = v + (b * T * Hv + hv) * Dv + dv;
  device const float* decay_ptr = log_decay + (b * T * Hv + hv) * decay_stride;
  device const beta_t* beta_ptr = beta + (b * T * Hv + hv);
  device input_t* y_ptr = y + (b * T * Hv + hv) * Dv + dv;

  for (uint t = 0; t < T; t++) {
    const float head_decay = channel_decay ? 0.0f : (log_decay_input ? simd_broadcast_first((lane_id == 0) ? precise::exp(decay_ptr[0]) : 0.0f) : decay_ptr[0]);
    float memory = 0.0f;
    for (uint i = 0; i < state_elements_per_lane; i++) {
      const uint dk = state_elements_per_lane * lane_id + i;
      if (key_dim_multiple_of_32 || dk < Dk) {
        // Per key channel decay (KDA) gives each lane its own gates for the state slice it owns.
        state[i] *= channel_decay ? (log_decay_input ? precise::exp(decay_ptr[dk]) : decay_ptr[dk]) : head_decay;
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
)" + checkpointStore + R"(    q_ptr += Hk * Dk;
    k_ptr += Hk * Dk;
    v_ptr += Hv * Dv;
    y_ptr += Hv * Dv;
    decay_ptr += Hv * decay_stride;
    beta_ptr += Hv;
  }
)" + finalStateStore + R"(
}
  )";
  return shader;
}

// The recurrence is sequential in time, so the work per step is what matters. Each SIMD
// group runs four value columns with eight lanes per column, each lane owning keyDim / 8
// state elements (keyDim / 32 float4 chunks; chunk c of lane j covers keys c * 32 + j * 4
// ... + 3, so the eight lanes read 32 consecutive keys). Reductions are three xor-shuffle
// stages. The threadgroup stages a block of timesteps of q, k, the decay (exponentiated
// once here rather than once per value column), v and beta in threadgroup memory, and
// writes the block's outputs coalesced.
std::string GatedDeltaKernel::createBlockedSource() const noexcept {
  const unsigned keyDim = 8 * stateElementsPerLane;
  const unsigned chunks = stateElementsPerLane / 4;
  const unsigned decayWidth = channelDecay ? keyDim : 1;
  const unsigned columns = 4 * simdGroups;
  const unsigned threads = 32 * simdGroups;
  const std::string sequenceLengthConstant = loadM ? "" : R"(constant uint T [[function_constant(1)]];
)";
  const std::string loadMArgument = loadM ? R"(  const device uint* loadM [[buffer(8)]],
)" : "";
  const std::string loadMValue = loadM ? R"(  const uniform<uint> T = make_uniform(loadM[0]);
)" : "";
  const std::string checkpointConstants = stateCheckpointing ? R"(constant uint state_checkpoint_count [[function_constant(9)]];
constant uint state_history_count = state_checkpoint_count + 1;
)" : "";
  const std::string stateInOffset = "((b * Hv + hv) * Dv + dv) * DK";
  const std::string stateOutOffset = stateCheckpointing ? "((b * state_history_count * Hv + hv) * Dv + dv) * DK" : stateInOffset;
  const std::string checkpointStore = stateCheckpointing ? R"(      const uint history_index = T - 1 - (t0 + tt);
      if (active && history_index > 0 && history_index <= state_checkpoint_count) {
        const uint offset = ((b * state_history_count * Hv + history_index * Hv + hv) * Dv + dv) * DK;
        for (uint c = 0; c < CHUNKS; c++)
          for (uint e = 0; e < 4; e++)
            state_out[offset + c * 32 + j * 4 + e] = state[c][e];
      }
)" : "";
  const std::string decayRead = channelDecay ? "tdecay[tt * (DK / 4) + c * 8 + j]" : "float4(tdecay[tt])";
  std::string shader = R"(
#include <metal_stdlib>
using namespace metal;

typedef )" + inputMemoryPrecision.name() + R"( input_t;
typedef vec<input_t, 4> input4_t;
typedef )" + betaMemoryPrecision.name() + R"( beta_t;

constant uint B [[function_constant(0)]];
)" + sequenceLengthConstant + R"(constant uint Hk [[function_constant(2)]];
constant uint Hv [[function_constant(3)]];
constant uint Dv [[function_constant(5)]];
constant bool log_decay_input [[function_constant(6)]];
)" + checkpointConstants + R"(constant uint hv_per_hk = Hv / Hk;

#define DK )" + std::to_string(keyDim) + R"(
#define CHUNKS )" + std::to_string(chunks) + R"(
#define TB )" + std::to_string((unsigned)timeBlock) + R"(
#define COLUMNS )" + std::to_string(columns) + R"(
#define THREADS )" + std::to_string(threads) + R"(
#define DECAY_WIDTH )" + std::to_string(decayWidth) + R"(

kernel void gated_delta(
  device const input_t* q [[buffer(0)]],
  device const input_t* k [[buffer(1)]],
  device const input_t* v [[buffer(2)]],
  device const float* decay [[buffer(3)]],
  device const beta_t* beta [[buffer(4)]],
  device const float* state_in [[buffer(5)]],
  device input_t* y [[buffer(6)]],
  device float* state_out [[buffer(7)]],
)" + loadMArgument + R"(
  uint3 group [[threadgroup_position_in_grid]],
  ushort sg [[simdgroup_index_in_threadgroup]],
  ushort lane [[thread_index_in_simdgroup]],
  ushort tid [[thread_index_in_threadgroup]]
) {
)" + loadMValue + R"(  threadgroup input4_t tq[TB * DK / 4];
  threadgroup input4_t tk[TB * DK / 4];
  threadgroup )" + (channelDecay ? "float4 tdecay[TB * DK / 4]" : "float tdecay[TB]") + R"(;
  threadgroup input_t tv[TB * COLUMNS];
  threadgroup float tbeta[TB];
  threadgroup input_t ty[TB * COLUMNS];

  const uint n = group.z;
  const uint b = n / Hv;
  const uint hv = n - b * Hv;
  const uint hk = hv / hv_per_hk;
  const uint dv_base = group.y * COLUMNS;
  const uint column = sg * 4 + lane / 8;
  const uint j = lane % 8;
  const uint dv = dv_base + column;
  const bool active = dv < Dv;

  float4 state[CHUNKS];
  {
    const uint offset = )" + stateInOffset + R"(;
    for (uint c = 0; c < CHUNKS; c++)
      for (uint e = 0; e < 4; e++)
        state[c][e] = active ? state_in[offset + c * 32 + j * 4 + e] : 0.0f;
  }

  for (uint t0 = 0; t0 < T; t0 += TB) {
    const uint steps = min((uint)TB, T - t0);
    threadgroup input_t* const q_stage = (threadgroup input_t*)tq;
    threadgroup input_t* const k_stage = (threadgroup input_t*)tk;
    threadgroup float* const decay_stage = (threadgroup float*)tdecay;
    for (uint i = tid; i < TB * DK; i += THREADS) {
      const uint tt = i / DK;
      const uint d = i - tt * DK;
      input_t q_value = 0;
      input_t k_value = 0;
      if (tt < steps) {
        const uint offset = ((b * T + t0 + tt) * Hk + hk) * DK + d;
        q_value = q[offset];
        k_value = k[offset];
      }
      q_stage[i] = q_value;
      k_stage[i] = k_value;
    }
    for (uint i = tid; i < TB * DECAY_WIDTH; i += THREADS) {
      const uint tt = i / DECAY_WIDTH;
      const uint d = i - tt * DECAY_WIDTH;
      const float g = tt < steps ? decay[((b * T + t0 + tt) * Hv + hv) * DECAY_WIDTH + d] : 0.0f;
      decay_stage[i] = log_decay_input ? precise::exp(g) : g;
    }
    for (uint i = tid; i < TB * COLUMNS; i += THREADS) {
      const uint tt = i / COLUMNS;
      const uint c = i - tt * COLUMNS;
      tv[i] = (tt < steps && dv_base + c < Dv) ? v[((b * T + t0 + tt) * Hv + hv) * Dv + dv_base + c] : input_t(0);
    }
    if (tid < TB)
      tbeta[tid] = tid < steps ? static_cast<float>(beta[(b * T + t0 + tid) * Hv + hv]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tt = 0; tt < steps; tt++) {
      float4 keys[CHUNKS];
      float memory = 0.0f;
      for (uint c = 0; c < CHUNKS; c++) {
        keys[c] = float4(tk[tt * (DK / 4) + c * 8 + j]);
        state[c] *= )" + decayRead + R"(;
        memory += dot(state[c], keys[c]);
      }
      memory += simd_shuffle_xor(memory, 1);
      memory += simd_shuffle_xor(memory, 2);
      memory += simd_shuffle_xor(memory, 4);
      const float delta = (static_cast<float>(tv[tt * COLUMNS + column]) - memory) * tbeta[tt];
      float out = 0.0f;
      for (uint c = 0; c < CHUNKS; c++) {
        state[c] += delta * keys[c];
        out += dot(state[c], float4(tq[tt * (DK / 4) + c * 8 + j]));
      }
      out += simd_shuffle_xor(out, 1);
      out += simd_shuffle_xor(out, 2);
      out += simd_shuffle_xor(out, 4);
      if (j == 0)
        ty[tt * COLUMNS + column] = static_cast<input_t>(out);
)" + checkpointStore + R"(    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TB * COLUMNS; i += THREADS) {
      const uint tt = i / COLUMNS;
      const uint c = i - tt * COLUMNS;
      if (tt < steps && dv_base + c < Dv)
        y[((b * T + t0 + tt) * Hv + hv) * Dv + dv_base + c] = ty[i];
    }
  }

  if (active) {
    const uint offset = )" + stateOutOffset + R"(;
    for (uint c = 0; c < CHUNKS; c++)
      for (uint e = 0; e < 4; e++)
        state_out[offset + c * 32 + j * 4 + e] = state[c][e];
  }
}
)";
  return shader;
}
