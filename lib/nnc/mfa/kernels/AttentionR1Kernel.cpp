#include "AttentionR1Kernel.hpp"

#include "../ccv_nnc_mfa.hpp"

AttentionR1Kernel::AttentionR1Kernel(AttentionR1KernelDescriptor descriptor, MTL::Device* const device) {
  memoryPrecision = descriptor.memoryPrecision;
  loadC = descriptor.loadC;
  attentionSinks = descriptor.attentionSinks;
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint32_t AttentionR1Kernel::threadgroupMemoryAllocation(const AttentionR1Descriptor& descriptor) const noexcept {
  return descriptor.mode == AttentionR1Descriptor::Mode::cooperative
      ? descriptor.simdgroups * (descriptor.D + 2) * sizeof(float) : 0;
}

uint32_t AttentionR1Kernel::threadgroupSize(const AttentionR1Descriptor& descriptor) const noexcept {
  return 32 * descriptor.simdgroups;
}

std::string AttentionR1Kernel::createSource() const noexcept {
  std::string source = createConstants();
  source += R"(
#include <metal_stdlib>
using namespace metal;
)";
  for (const bool partial : {false, true}) {
    source += partial ? "kernel void attention_r1_split_partials(\n" : "kernel void attention_r1_direct(\n";
    source += R"(
    device const real* Q [[buffer(0)]],
    device const real* K [[buffer(1)]],
    device const real* V [[buffer(2)]],
)";
    source += partial ? "    device float* O [[buffer(3)]],\n" : "    device real* O [[buffer(3)]],\n";
    if (loadC)
      source += "    constant uint& C_LEN [[buffer(4)]],\n";
    if (attentionSinks)
      source += "    device const real* Sinks [[buffer(19)]],\n    constant uint& Sink_head_stride [[buffer(20)]],\n";
    source += R"(
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 grid [[threadgroups_per_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint simds_per_row = NSG / R_LEN;
  const uint r = sgid / simds_per_row;
  const uint hq = (tgid.x * simds_per_row + sgid % simds_per_row) * HEADS_PER_SIMD;
  const uint hk = hq / (Hq / Hk);
  const uint batch = tgid.y;
  const uint partition = tgid.z;
  const uint elements = D_LEN / 32;
  const uint d = lane * elements;
  const uint cols = CAUSAL ? uint(max(0, int(C_LEN) - int(R_LEN) + int(r) + 1)) : C_LEN;
  device const real* q_row = Q + ((size_t(batch) * R_LEN + r) * Hq + hq) * D_LEN + d;
  device const real* k_row = K + ((size_t(batch) * C_LEN + partition) * Hk + hk) * D_LEN + d;
  device const real* v_row = V + ((size_t(batch) * C_LEN + partition) * Hk + hk) * D_LEN + d;
  float q[2][8];
  float acc[2][8] = {{0}};
  // A finite empty maximum makes completely empty partitions safe to merge.
  float row_m[2] = {-MAXFLOAT, -MAXFLOAT};
  float row_s[2] = {0};
  #pragma clang loop unroll(full)
  for (uint j = 0; j < HEADS_PER_SIMD; ++j)
    for (uint i = 0; i < elements; ++i)
      q[j][i] = float(q_row[j * D_LEN + i]) * scale_log2e;
)";
    if (attentionSinks)
      source += R"(
  if (partition == 0) {
    #pragma clang loop unroll(full)
    for (uint j = 0; j < HEADS_PER_SIMD; ++j) {
      row_m[j] = float(Sinks[(hq + j) * Sink_head_stride]) * 1.442695041;
      row_s[j] = 1;
    }
  }
)";
    source += R"(
  for (uint c = partition; c < cols; c += NWG) {
    float scores[2] = {0}, correction[2], weights[2];
    for (uint i = 0; i < elements; ++i) {
      const float key = float(k_row[i]);
      #pragma clang loop unroll(full)
      for (uint j = 0; j < HEADS_PER_SIMD; ++j)
        scores[j] += q[j][i] * key;
    }
    #pragma clang loop unroll(full)
    for (uint j = 0; j < HEADS_PER_SIMD; ++j) {
      scores[j] = simd_sum(scores[j]);
      const float next_m = max(row_m[j], scores[j]);
      correction[j] = fast::exp2(row_m[j] - next_m);
      weights[j] = fast::exp2(scores[j] - next_m);
      row_s[j] = row_s[j] * correction[j] + weights[j];
      row_m[j] = next_m;
    }
    for (uint i = 0; i < elements; ++i) {
      const float value = float(v_row[i]);
      #pragma clang loop unroll(full)
      for (uint j = 0; j < HEADS_PER_SIMD; ++j)
        acc[j][i] = acc[j][i] * correction[j] + weights[j] * value;
    }
    k_row += size_t(NWG) * Hk * D_LEN;
    v_row += size_t(NWG) * Hk * D_LEN;
  }
  #pragma clang loop unroll(full)
  for (uint j = 0; j < HEADS_PER_SIMD; ++j) {
    const size_t row = (size_t(batch) * R_LEN + r) * Hq + hq + j;
)";
    if (partial)
      source += R"(
    const size_t index = row * NWG + partition;
    const size_t count = size_t(grid.y) * R_LEN * Hq * NWG;
    for (uint i = 0; i < elements; ++i)
      O[index * D_LEN + d + i] = acc[j][i];
    if (lane == 0) {
      O[count * D_LEN + index] = row_s[j];
      O[count * (D_LEN + 1) + index] = row_m[j];
    }
  }
}
)";
    else
      source += R"(
    for (uint i = 0; i < elements; ++i)
      O[row * D_LEN + d + i] = real(row_s[j] > 0 ? acc[j][i] / row_s[j] : 0);
  }
}
)";
  }
  source += R"(
// For short sequences, token streams cooperate within a single workgroup.
// This avoids a second dispatch while retaining parallel KV traversal.
kernel void attention_r1_cooperative(
    device const real* Q [[buffer(0)]],
    device const real* K [[buffer(1)]],
    device const real* V [[buffer(2)]],
    device real* O [[buffer(3)]],
)";
  if (loadC)
    source += "    constant uint& C_LEN [[buffer(4)]],\n";
  if (attentionSinks)
    source += "    device const real* Sinks [[buffer(19)]],\n    constant uint& Sink_head_stride [[buffer(20)]],\n";
  source += R"(
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint hq = tgid.x, batch = tgid.y, r = tgid.z;
  const uint hk = hq / (Hq / Hk);
  const uint elements = D_LEN / 32;
  const uint d = lane * elements;
  const uint cols = CAUSAL ? uint(max(0, int(C_LEN) - int(R_LEN) + int(r) + 1)) : C_LEN;
  const size_t row = (size_t(batch) * R_LEN + r) * Hq + hq;
  device const real* qp = Q + row * D_LEN + d;
  device const real* kp = K + ((size_t(batch) * C_LEN + sgid) * Hk + hk) * D_LEN + d;
  device const real* vp = V + ((size_t(batch) * C_LEN + sgid) * Hk + hk) * D_LEN + d;
  float q[8], acc[8] = {0};
  for (uint i = 0; i < elements; ++i)
    q[i] = float(qp[i]) * scale_log2e;
  float maximum = -MAXFLOAT, denominator = 0;
)";
  if (attentionSinks)
    source += R"(
  if (sgid == 0) {
    maximum = float(Sinks[hq * Sink_head_stride]) * 1.442695041;
    denominator = 1;
  }
)";
  source += R"(
  for (uint c = sgid; c < cols; c += NSG) {
    float score = 0;
    for (uint i = 0; i < elements; ++i)
      score += q[i] * float(kp[i]);
    score = simd_sum(score);
    const float next_m = max(maximum, score);
    const float correction = fast::exp2(maximum - next_m);
    const float weight = fast::exp2(score - next_m);
    for (uint i = 0; i < elements; ++i)
      acc[i] = acc[i] * correction + weight * float(vp[i]);
    denominator = denominator * correction + weight;
    maximum = next_m;
    kp += size_t(NSG) * Hk * D_LEN;
    vp += size_t(NSG) * Hk * D_LEN;
  }
  threadgroup float* sums = scratch + NSG * D_LEN;
  threadgroup float* maxs = sums + NSG;
  for (uint i = 0; i < elements; ++i)
    scratch[sgid * D_LEN + d + i] = acc[i];
  if (lane == 0) {
    sums[sgid] = denominator;
    maxs[sgid] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float local_m = lane < NSG ? maxs[lane] : -MAXFLOAT;
  maximum = simd_max(local_m);
  const float factor = lane < NSG ? fast::exp2(local_m - maximum) : 0;
  denominator = simd_sum(lane < NSG ? sums[lane] * factor : 0);
  // SIMD shuffles distribute each stream's rescale without another barrier.
  for (uint out_d = sgid * 32 + lane; out_d < D_LEN; out_d += NSG * 32) {
    float value = 0;
    for (uint s = 0; s < NSG; ++s)
      value += scratch[s * D_LEN + out_d] * simd_shuffle(factor, s);
    O[row * D_LEN + out_d] = real(denominator > 0 ? value / denominator : 0);
  }
}
)";
  source += R"(
kernel void attention_r1_split_reduce(
    device const float* partial [[buffer(0)]],
    device real* O [[buffer(1)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 grid [[threadgroups_per_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const size_t row = (size_t(tgid.y) * R_LEN + tgid.z) * Hq + tgid.x;
  const size_t count = size_t(grid.y) * R_LEN * Hq * NWG;
  device const float* sums = partial + count * D_LEN + row * NWG;
  device const float* maxs = partial + count * (D_LEN + 1) + row * NWG;
  float maximum = -MAXFLOAT;
  for (uint p = lane; p < NWG; p += 32)
    maximum = max(maximum, maxs[p]);
  maximum = simd_max(maximum);
  float denominator = 0;
  for (uint p = lane; p < NWG; p += 32)
    denominator += sums[p] * fast::exp2(maxs[p] - maximum);
  denominator = simd_sum(denominator);
  const uint elements = D_LEN / 32;
  const uint d = lane * elements;
  float acc[8] = {0};
  for (uint p = sgid; p < NWG; p += REDUCE_SG) {
    const float factor = fast::exp2(maxs[p] - maximum);
    device const float* src = partial + (row * NWG + p) * D_LEN + d;
    for (uint i = 0; i < elements; ++i)
      acc[i] += factor * src[i];
  }
  for (uint i = 0; i < elements; ++i)
    scratch[sgid * D_LEN + d + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Different SIMD groups finish different output components. Each shared
  // memory load is contiguous across lanes, and the merge does not serialize
  // every component through SIMD group zero.
  for (uint out_d = sgid * 32 + lane; out_d < D_LEN; out_d += REDUCE_SG * 32) {
    float value = 0;
    for (uint s = 0; s < REDUCE_SG; ++s)
      value += scratch[s * D_LEN + out_d];
    O[row * D_LEN + out_d] = real(denominator > 0 ? value / denominator : 0);
  }
}
)";
  return source;
}

std::string AttentionR1Kernel::createConstants() const noexcept {
  std::string defines;
  if (memoryPrecision == GEMMOperandPrecision::FP32)
    defines += "typedef float real;\n";
  else if (memoryPrecision == GEMMOperandPrecision::BF16)
    defines += "typedef bfloat real;\n";
  else
    defines += "typedef half real;\n";
  if (!loadC)
    defines += "constant uint C_LEN [[function_constant(0)]];\n";
  defines += "constant uint Hq [[function_constant(1)]];\n";
  defines += "constant uint Hk [[function_constant(2)]];\n";
  defines += "constant uint D_LEN [[function_constant(3)]];\n";
  defines += "constant uint NSG [[function_constant(4)]];\n";
  defines += "constant uint NWG [[function_constant(5)]];\n";
  defines += "constant float dot_product_scale [[function_constant(6)]];\n";
  defines += "constant float scale_log2e = dot_product_scale * 1.442695041;\n";
  defines += "constant uint R_LEN [[function_constant(7)]];\n";
  defines += "constant bool CAUSAL [[function_constant(8)]];\n";
  defines += "constant uint REDUCE_SG [[function_constant(9)]];\n";
  defines += "constant uint HEADS_PER_SIMD [[function_constant(10)]];\n";
  return defines;
}
