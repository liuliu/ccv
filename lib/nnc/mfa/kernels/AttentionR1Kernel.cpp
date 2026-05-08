#include "AttentionR1Kernel.hpp"

#include "../ccv_nnc_mfa.hpp"

AttentionR1Kernel::AttentionR1Kernel(AttentionR1KernelDescriptor descriptor, MTL::Device* const device) {
  memoryPrecision = descriptor.memoryPrecision;
  loadC = descriptor.loadC;
  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

static uint32_t alignUp(uint32_t value, uint32_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

uint32_t AttentionR1Kernel::threadgroupMemoryAllocation(const AttentionR1Descriptor& descriptor) const noexcept {
  const uint32_t queryBytes = alignUp(descriptor.D * (uint32_t)descriptor.memoryPrecision.size(), sizeof(float));
  const uint32_t partialBytes = descriptor.simdgroups * (descriptor.D + 2) * sizeof(float);
  return alignUp(queryBytes + partialBytes, 16);
}

uint32_t AttentionR1Kernel::threadgroupSize(const AttentionR1Descriptor& descriptor) const noexcept {
  return 32 * descriptor.simdgroups;
}

std::string AttentionR1Kernel::createSource() const noexcept {
  std::string source = createConstants();
  source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void attention_r1_direct(
    device const real* Q [[buffer(0)]],
    device const real* K [[buffer(1)]],
    device const real* V [[buffer(2)]],
    device real* O [[buffer(3)]],
)";
  if (loadC) {
    source += R"(
    const device uint* loadC [[buffer(4)]],
)";
  }
  source += R"(
    threadgroup uchar* scratch [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint hq = tgid.x;
  const uint batch = tgid.y;
  const uint hk = hq / (Hq / Hk);
)";
  if (loadC) {
    source += R"(
  const uniform<uint> C_LEN = make_uniform(loadC[0]);
)";
  }
  source += R"(
  const uint q_shared_bytes = ((D_LEN * (uint)sizeof(real) + 3) / 4) * 4;
  threadgroup real* q_shared = (threadgroup real*)scratch;
  threadgroup float* partial_o = (threadgroup float*)(scratch + q_shared_bytes);
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;

  if (sgid == 0) {
    device const real* q_row = Q + (batch * Hq + hq) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      q_shared[d] = q_row[d];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[8];
  for (uint i = 0; i < 8; ++i) {
    acc[i] = 0;
  }
  float row_m = -INFINITY;
  float row_s = 0;

  for (uint c = sgid; c < C_LEN; c += NSG) {
    float dot_acc = 0;
    device const real* k_row = K + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      dot_acc += (float)q_shared[d] * (float)k_row[d];
    }
    const float score = simd_sum(dot_acc) * scale_log2e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float p = fast::exp2(score - new_m);
    device const real* v_row = V + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint i = 0; i < 8; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        acc[i] = acc[i] * old_scale + p * (float)v_row[d];
      }
    }
    row_s = row_s * old_scale + p;
    row_m = new_m;
  }

  for (uint i = 0; i < 8; ++i) {
    const uint d = lane_id + i * 32;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float global_m = -INFINITY;
    for (uint i = 0; i < NSG; ++i) {
      global_m = max(global_m, partial_m[i]);
    }
    float global_s = 0;
    for (uint i = 0; i < NSG; ++i) {
      global_s += partial_s[i] * fast::exp2(partial_m[i] - global_m);
    }
    device real* o_row = O + (batch * Hq + hq) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      float numerator = 0;
      for (uint i = 0; i < NSG; ++i) {
        numerator += partial_o[i * D_LEN + d] * fast::exp2(partial_m[i] - global_m);
      }
      o_row[d] = (real)(numerator / global_s);
    }
  }
}

kernel void attention_r1_split_partials(
    device const real* Q [[buffer(0)]],
    device const real* K [[buffer(1)]],
    device const real* V [[buffer(2)]],
    device float* partial [[buffer(3)]],
)";
  if (loadC) {
    source += R"(
    const device uint* loadC [[buffer(4)]],
)";
  }
  source += R"(
    threadgroup uchar* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint hq = tgid.x;
  const uint batch = tgid.y;
  const uint iwg = tgid.z;
  const uint hk = hq / (Hq / Hk);
)";
  if (loadC) {
    source += R"(
  const uniform<uint> C_LEN = make_uniform(loadC[0]);
)";
  }
  source += R"(
  const uint q_shared_bytes = ((D_LEN * (uint)sizeof(real) + 3) / 4) * 4;
  threadgroup real* q_shared = (threadgroup real*)scratch;
  threadgroup float* partial_o = (threadgroup float*)(scratch + q_shared_bytes);
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;

  if (sgid == 0) {
    device const real* q_row = Q + (batch * Hq + hq) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      q_shared[d] = q_row[d];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[8];
  for (uint i = 0; i < 8; ++i) {
    acc[i] = 0;
  }
  float row_m = -INFINITY;
  float row_s = 0;

  for (uint c = iwg * NSG + sgid; c < C_LEN; c += NWG * NSG) {
    float dot_acc = 0;
    device const real* k_row = K + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      dot_acc += (float)q_shared[d] * (float)k_row[d];
    }
    const float score = simd_sum(dot_acc) * scale_log2e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float p = fast::exp2(score - new_m);
    device const real* v_row = V + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint i = 0; i < 8; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        acc[i] = acc[i] * old_scale + p * (float)v_row[d];
      }
    }
    row_s = row_s * old_scale + p;
    row_m = new_m;
  }

  for (uint i = 0; i < 8; ++i) {
    const uint d = lane_id + i * 32;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float global_m = -INFINITY;
    for (uint i = 0; i < NSG; ++i) {
      global_m = max(global_m, partial_m[i]);
    }
    float global_s = 0;
    for (uint i = 0; i < NSG; ++i) {
      global_s += partial_s[i] * fast::exp2(partial_m[i] - global_m);
    }
    device float* row = partial + (((batch * Hq + hq) * NWG + iwg) * (D_LEN + 2));
    for (uint d = lane_id; d < D_LEN; d += 32) {
      float numerator = 0;
      for (uint i = 0; i < NSG; ++i) {
        numerator += partial_o[i * D_LEN + d] * fast::exp2(partial_m[i] - global_m);
      }
      row[d] = numerator;
    }
    if (lane_id == 0) {
      row[D_LEN] = global_s;
      row[D_LEN + 1] = global_m;
    }
  }
}

kernel void attention_r1_split_reduce(
    device const float* partial [[buffer(0)]],
    device real* O [[buffer(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint hq = tgid.x;
  const uint batch = tgid.y;
  device const float* base = partial + ((batch * Hq + hq) * NWG * (D_LEN + 2));
  float local_m = -INFINITY;
  if (lane_id < NWG) {
    local_m = base[lane_id * (D_LEN + 2) + D_LEN + 1];
  }
  const float global_m = simd_max(local_m);
  float local_s = 0;
  if (lane_id < NWG) {
    local_s = base[lane_id * (D_LEN + 2) + D_LEN] *
        fast::exp2(base[lane_id * (D_LEN + 2) + D_LEN + 1] - global_m);
  }
  const float global_s = simd_sum(local_s);
  device real* o_row = O + (batch * Hq + hq) * D_LEN;
  for (uint d = lane_id; d < D_LEN; d += 32) {
    float numerator = 0;
    for (uint i = 0; i < NWG; ++i) {
      device const float* row = base + i * (D_LEN + 2);
      numerator += row[d] * fast::exp2(row[D_LEN + 1] - global_m);
    }
    o_row[d] = (real)(numerator / global_s);
  }
}
)";
  return source;
}

std::string AttentionR1Kernel::createConstants() const noexcept {
  std::string defines;
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float real;\n";
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    defines += "typedef bfloat real;\n";
  } else {
    defines += "typedef half real;\n";
  }
  if (!loadC) {
    defines += "constant uint C_LEN [[function_constant(0)]];\n";
  }
  defines += "constant uint Hq [[function_constant(1)]];\n";
  defines += "constant uint Hk [[function_constant(2)]];\n";
  defines += "constant uint D_LEN [[function_constant(3)]];\n";
  defines += "constant uint NSG [[function_constant(4)]];\n";
  defines += "constant uint NWG [[function_constant(5)]];\n";
  defines += "constant float dot_product_scale [[function_constant(6)]];\n";
  defines += "constant float scale_log2e = dot_product_scale * 1.442695041;\n";
  return defines;
}
