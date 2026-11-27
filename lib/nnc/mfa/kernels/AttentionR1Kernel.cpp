#include "AttentionR1Kernel.hpp"

#include "../ccv_nnc_mfa.hpp"
#include <iomanip>
#include <limits>
#include <sstream>

AttentionR1Kernel::AttentionR1Kernel(AttentionR1KernelDescriptor descriptor, MTL::Device* const device) {
  memoryPrecision = descriptor.memoryPrecision;
  loadC = descriptor.loadC;
  attentionSinks = descriptor.attentionSinks;
  sinksPerQuery = descriptor.sinksPerQuery;
  relativeBias = descriptor.relativeBias;
  masked = descriptor.masked;
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = ccv::nnc::mfa::compile_source(device, string, &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
}

static uint32_t alignUp(uint32_t value, uint32_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

uint32_t AttentionR1Kernel::threadgroupMemoryAllocation(const AttentionR1Descriptor& descriptor) const noexcept {
  return descriptor.mode == AttentionR1Descriptor::Mode::cooperative
      ? alignUp(descriptor.simdgroups * (descriptor.D + 2) * sizeof(float), 16) : 0;
}

uint32_t AttentionR1Kernel::threadgroupSize(const AttentionR1Descriptor& descriptor) const noexcept {
  return 32 * descriptor.simdgroups;
}

std::string AttentionR1Kernel::createSource() const noexcept {
  std::string source = createConstants();
  source += R"(
#include <metal_stdlib>
using namespace metal;

// Adds the partner's upper value into the lower one, after swapping the two where `swap` is all
// ones. The swap is integer arithmetic rather than a select between two slots, which would keep
// the slots in memory.
inline float butterfly(float lower, float upper, uint swap, ushort mask) {
  const uint exchange = (as_type<uint>(lower) ^ as_type<uint>(upper)) & swap;
  return as_type<float>(as_type<uint>(lower) ^ exchange) + simd_shuffle_xor(as_type<float>(as_type<uint>(upper) ^ exchange), mask);
}
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
      source += "    constant uint& C [[buffer(4)]],\n";
    if (attentionSinks)
      source += "    device const real* Sinks [[buffer(19)]],\n    constant uint& Sink_head_stride [[buffer(20)]],\n";
    if (relativeBias)
      source += "    device const real* Relative_bias [[buffer(23)]],\n";
    if (masked)
      source += "    device const real* Mask [[buffer(15)]],\n";
    source += R"(
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 grid [[threadgroups_per_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  // The SIMD groups of a workgroup: HeadsPerSIMD query heads for each of RowGroups groups of
  // RowsPerSIMD rows. Query t of a SIMD group is head hq + t / RowsPerSIMD of row
  // r0 + t % RowsPerSIMD; rows past R repeat the last row and are not written.
  const uint simds_per_rows = SIMDGroups / RowGroups;
  const uint r0 = sgid / simds_per_rows * RowsPerSIMD;
  const uint hq = (tgid.x * simds_per_rows + sgid % simds_per_rows) * HeadsPerSIMD;
  const uint hk = hq / (Hq / Hk);
  const uint batch = tgid.y;
  const uint partition = tgid.z;
  const uint elements = D / 32;
  const uint d = lane * elements;
  uint rows[4], cols[4], first[4];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < RowsPerSIMD; ++i) {
    rows[i] = min(r0 + i, R - 1);
    cols[i] = Causal ? uint(max(0, int(C) - int(R) + int(rows[i]) + 1)) : C;
    // A sliding-window row reads its last SlidingWindow keys.
    first[i] = SlidingWindow > 0 && cols[i] > SlidingWindow ? cols[i] - SlidingWindow : 0;
  }
  // Later rows end and start later. Keys go in blocks of 16, and partition p takes blocks p,
  // p + Workgroups, ...
  const uint end = cols[RowsPerSIMD - 1];
  const uint first_block = first[0] / 16;
  const uint start_block = first_block + (partition + Workgroups - first_block % Workgroups) % Workgroups;
  const uint key_stride = Hk * D;
  device const real* k_base = K + size_t(batch) * C * key_stride + hk * D + d;
  device const real* v_base = V + size_t(batch) * C * key_stride + hk * D + d;
  // Each lane ends up with the score of key `own` of a block (both halves of the SIMD hold the
  // same 16). Its products for key s ^ key_xor go to slot s, and each butterfly step below adds
  // the partner's upper slots into its own lower ones. With scattered keys (key_xor = own), a
  // load spreads over 16 keys; with contiguous ones (key_xor = 0), the SIMD group reads one key
  // at a time, and a lane whose bit of a step is set first swaps the step's two halves.
  const uint own = lane & 15;
  const uint key_xor = ContiguousKeys ? 0 : own;
  const uint swap8 = ContiguousKeys ? 0u - ((uint(lane) >> 3) & 1u) : 0;
  const uint swap4 = ContiguousKeys ? 0u - ((uint(lane) >> 2) & 1u) : 0;
  const uint swap2 = ContiguousKeys ? 0u - ((uint(lane) >> 1) & 1u) : 0;
  const uint swap1 = ContiguousKeys ? 0u - (uint(lane) & 1u) : 0;
  float q[4][8];
  float acc[4][8] = {{0}};
  // A finite empty maximum makes completely empty partitions safe to merge.
  float row_m[4] = {-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT};
  float row_s[4] = {0};
  #pragma clang loop unroll(full)
  for (uint t = 0; t < QueriesPerSIMD; ++t) {
    device const real* q_row = Q + ((size_t(batch) * R + rows[t % RowsPerSIMD]) * Hq + hq + t / RowsPerSIMD) * D + d;
    for (uint i = 0; i < elements; ++i)
      q[t][i] = float(q_row[i]) * ScaleLog2e;
  }
)";
    if (attentionSinks) {
      source += R"(
  if (partition == 0) {
    #pragma clang loop unroll(full)
    for (uint t = 0; t < QueriesPerSIMD; ++t) {
)";
      // One sink for each query and head, [batch, R, Hq], or one for each head.
      if (sinksPerQuery)
        source += "      row_m[t] = float(Sinks[(size_t(batch) * R + rows[t % RowsPerSIMD]) * Hq + hq + t / RowsPerSIMD]) * 1.442695041;\n";
      else
        source += "      row_m[t] = float(Sinks[(hq + t / RowsPerSIMD) * Sink_head_stride]) * 1.442695041;\n";
      source += R"(      row_s[t] = 1;
    }
  }
)";
    }
    source += R"(
  for (uint block = start_block; block * 16 < end; block += Workgroups) {
    const uint c0 = block * 16;
    // The last block may run past the keys; its extra keys read the last one and are masked.
    const uint last = C - 1 - c0;
    device const real* k_block = k_base + size_t(c0) * key_stride;
    device const real* v_block = v_base + size_t(c0) * key_stride;
    float part[4][16];
    #pragma clang loop unroll(full)
    for (uint s = 0; s < 16; ++s) {
      device const real* k_row = k_block + min(s ^ key_xor, last) * key_stride;
      float key[8];
      for (uint i = 0; i < elements; ++i)
        key[i] = float(k_row[i]);
      #pragma clang loop unroll(full)
      for (uint t = 0; t < QueriesPerSIMD; ++t) {
        float sum = 0;
        for (uint i = 0; i < elements; ++i)
          sum += q[t][i] * key[i];
        part[t][s] = sum;
      }
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < QueriesPerSIMD; ++t) {
      #pragma clang loop unroll(full)
      for (uint s = 0; s < 8; ++s)
        part[t][s] = butterfly(part[t][s], part[t][s + 8], swap8, 8);
      #pragma clang loop unroll(full)
      for (uint s = 0; s < 4; ++s)
        part[t][s] = butterfly(part[t][s], part[t][s + 4], swap4, 4);
      #pragma clang loop unroll(full)
      for (uint s = 0; s < 2; ++s)
        part[t][s] = butterfly(part[t][s], part[t][s + 2], swap2, 2);
      part[t][0] = butterfly(part[t][0], part[t][1], swap1, 1);
      part[t][0] += simd_shuffle_xor(part[t][0], 16);
    }
    const uint c = c0 + own;
    float weight[4];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < QueriesPerSIMD; ++t) {
      const uint i = t % RowsPerSIMD;
      bool visible = c >= first[i] && c < cols[i];
      float score = part[t][0];
)";
    if (masked)
      source += R"(      // The mask adds to the score, and closes the key at or below MaskedThreshold.
      if (visible) {
        const float addend = float(Mask[size_t(rows[i]) * C + c]);
        visible = addend > MaskedThreshold;
        score += addend * 1.442695041;
      }
)";
    if (relativeBias)
      source += R"(      // The key this far before the query's own, the last visible one, adds the query's bias at
      // that distance.
      if (visible && cols[i] - 1 - c < RelativeBiasExtent)
        score += float(Relative_bias[((size_t(batch) * R + rows[i]) * Hq + hq + t / RowsPerSIMD) * RelativeBiasExtent + cols[i] - 1 - c]) * 1.442695041;
)";
    source += R"(      const float next_m = max(row_m[t], simd_max(visible ? score : -MAXFLOAT));
      const float correction = fast::exp2(row_m[t] - next_m);
      weight[t] = visible ? fast::exp2(score - next_m) : 0;
      row_s[t] = row_s[t] * correction + simd_sum(lane < 16 ? weight[t] : 0);
      row_m[t] = next_m;
      for (uint e = 0; e < elements; ++e)
        acc[t][e] *= correction;
    }
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 16; ++k) {
      device const real* v_row = v_block + min(k, last) * key_stride;
      float value[8];
      for (uint i = 0; i < elements; ++i)
        value[i] = float(v_row[i]);
      #pragma clang loop unroll(full)
      for (uint t = 0; t < QueriesPerSIMD; ++t) {
        const float w = simd_shuffle(weight[t], ushort(k));
        for (uint i = 0; i < elements; ++i)
          acc[t][i] += w * value[i];
      }
    }
  }
  #pragma clang loop unroll(full)
  for (uint t = 0; t < QueriesPerSIMD; ++t) {
    if (r0 + t % RowsPerSIMD >= R)
      continue;
    const size_t row = (size_t(batch) * R + r0 + t % RowsPerSIMD) * Hq + hq + t / RowsPerSIMD;
)";
    if (partial)
      source += R"(
    const size_t index = row * Workgroups + partition;
    const size_t count = size_t(grid.y) * R * Hq * Workgroups;
    for (uint i = 0; i < elements; ++i)
      O[index * D + d + i] = acc[t][i];
    if (lane == 0) {
      O[count * D + index] = row_s[t];
      O[count * (D + 1) + index] = row_m[t];
    }
  }
}
)";
    else
      source += R"(
    for (uint i = 0; i < elements; ++i)
      O[row * D + d + i] = real(row_s[t] > 0 ? acc[t][i] / row_s[t] : 0);
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
    source += "    constant uint& C [[buffer(4)]],\n";
  if (attentionSinks)
    source += "    device const real* Sinks [[buffer(19)]],\n    constant uint& Sink_head_stride [[buffer(20)]],\n";
  if (relativeBias)
    source += "    device const real* Relative_bias [[buffer(23)]],\n";
  if (masked)
    source += "    device const real* Mask [[buffer(15)]],\n";
  source += R"(
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint hq = tgid.x, batch = tgid.y, r = tgid.z;
  const uint hk = hq / (Hq / Hk);
  const uint elements = D / 32;
  const uint d = lane * elements;
  const uint cols = Causal ? uint(max(0, int(C) - int(R) + int(r) + 1)) : C;
  // A sliding-window row reads its last SlidingWindow keys; the stream keeps its stride.
  const uint first = SlidingWindow > 0 && cols > SlidingWindow ? cols - SlidingWindow : 0;
  const uint start = first + (sgid + SIMDGroups - first % SIMDGroups) % SIMDGroups;
  const size_t row = (size_t(batch) * R + r) * Hq + hq;
  device const real* qp = Q + row * D + d;
  device const real* kp = K + ((size_t(batch) * C + start) * Hk + hk) * D + d;
  device const real* vp = V + ((size_t(batch) * C + start) * Hk + hk) * D + d;
  float q[8], acc[8] = {0};
  for (uint i = 0; i < elements; ++i)
    q[i] = float(qp[i]) * ScaleLog2e;
  float maximum = -MAXFLOAT, denominator = 0;
)";
  if (attentionSinks) {
    source += R"(
  if (sgid == 0) {
)";
    // One sink for each query and head, [batch, R, Hq], or one for each head.
    if (sinksPerQuery)
      source += "    maximum = float(Sinks[row]) * 1.442695041;\n";
    else
      source += "    maximum = float(Sinks[hq * Sink_head_stride]) * 1.442695041;\n";
    source += R"(    denominator = 1;
  }
)";
  }
  source += R"(
  for (uint c = start; c < cols; c += SIMDGroups) {
    float score = 0;
    for (uint i = 0; i < elements; ++i)
      score += q[i] * float(kp[i]);
    score = simd_sum(score);
)";
  if (relativeBias)
    source += R"(    if (cols - 1 - c < RelativeBiasExtent)
      score += float(Relative_bias[row * RelativeBiasExtent + cols - 1 - c]) * 1.442695041;
)";
  if (masked)
    source += R"(    // The mask adds to the score, and closes the key at or below MaskedThreshold.
    const float addend = float(Mask[size_t(r) * C + c]);
    const bool open = addend > MaskedThreshold;
    score += open ? addend * 1.442695041 : 0;
)";
  else
    source += "    const bool open = true;\n";
  source += R"(    const float next_m = open ? max(maximum, score) : maximum;
    // A closed key leaves the sums alone: after a -inf sink, maximum - next_m would be NaN.
    const float correction = open ? fast::exp2(maximum - next_m) : 1;
    const float weight = open ? fast::exp2(score - next_m) : 0;
    for (uint i = 0; i < elements; ++i)
      acc[i] = acc[i] * correction + weight * float(vp[i]);
    denominator = denominator * correction + weight;
    maximum = next_m;
    kp += size_t(SIMDGroups) * Hk * D;
    vp += size_t(SIMDGroups) * Hk * D;
  }
  threadgroup float* sums = scratch + SIMDGroups * D;
  threadgroup float* maxs = sums + SIMDGroups;
  for (uint i = 0; i < elements; ++i)
    scratch[sgid * D + d + i] = acc[i];
  if (lane == 0) {
    sums[sgid] = denominator;
    maxs[sgid] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float local_m = lane < SIMDGroups ? maxs[lane] : -MAXFLOAT;
  maximum = simd_max(local_m);
  const float factor = lane < SIMDGroups ? fast::exp2(local_m - maximum) : 0;
  denominator = simd_sum(lane < SIMDGroups ? sums[lane] * factor : 0);
  // SIMD shuffles distribute each stream's rescale without another barrier.
  for (uint out_d = sgid * 32 + lane; out_d < D; out_d += SIMDGroups * 32) {
    float value = 0;
    for (uint s = 0; s < SIMDGroups; ++s)
      value += scratch[s * D + out_d] * simd_shuffle(factor, s);
    O[row * D + out_d] = real(denominator > 0 ? value / denominator : 0);
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
  const size_t row = (size_t(tgid.y) * R + tgid.z) * Hq + tgid.x;
  const size_t count = size_t(grid.y) * R * Hq * Workgroups;
  device const float* sums = partial + count * D + row * Workgroups;
  device const float* maxs = partial + count * (D + 1) + row * Workgroups;
  float maximum = -MAXFLOAT;
  for (uint p = lane; p < Workgroups; p += 32)
    maximum = max(maximum, maxs[p]);
  maximum = simd_max(maximum);
  float denominator = 0;
  for (uint p = lane; p < Workgroups; p += 32)
    denominator += sums[p] * fast::exp2(maxs[p] - maximum);
  denominator = simd_sum(denominator);
  const uint elements = D / 32;
  const uint d = lane * elements;
  float acc[8] = {0};
  for (uint p = sgid; p < Workgroups; p += ReductionSIMDGroups) {
    const float factor = fast::exp2(maxs[p] - maximum);
    device const float* src = partial + (row * Workgroups + p) * D + d;
    for (uint i = 0; i < elements; ++i)
      acc[i] += factor * src[i];
  }
  for (uint i = 0; i < elements; ++i)
    scratch[sgid * D + d + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Different SIMD groups finish different output components. Each shared
  // memory load is contiguous across lanes, and the merge does not serialize
  // every component through SIMD group zero.
  for (uint out_d = sgid * 32 + lane; out_d < D; out_d += ReductionSIMDGroups * 32) {
    float value = 0;
    for (uint s = 0; s < ReductionSIMDGroups; ++s)
      value += scratch[s * D + out_d];
    O[row * D + out_d] = real(denominator > 0 ? value / denominator : 0);
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
    defines += "constant uint C [[function_constant(0)]];\n";
  }
  defines += "constant uint Hq [[function_constant(1)]];\n";
  defines += "constant uint Hk [[function_constant(2)]];\n";
  defines += "constant uint D [[function_constant(3)]];\n";
  defines += "constant uint SIMDGroups [[function_constant(4)]];\n";
  defines += "constant uint Workgroups [[function_constant(5)]];\n";
  defines += "constant float DotProductScale [[function_constant(6)]];\n";
  defines += "constant float ScaleLog2e = DotProductScale * 1.442695041;\n";
  defines += "constant uint R [[function_constant(7)]];\n";
  defines += "constant bool Causal [[function_constant(8)]];\n";
  defines += "constant uint ReductionSIMDGroups [[function_constant(9)]];\n";
  defines += "constant uint HeadsPerSIMD [[function_constant(10)]];\n";
  defines += "constant uint SlidingWindow [[function_constant(12)]];\n";
  defines += "constant uint RowsPerSIMD [[function_constant(13)]];\n";
  defines += "constant uint RowGroups = (R + RowsPerSIMD - 1) / RowsPerSIMD;\n";
  defines += "constant uint QueriesPerSIMD = HeadsPerSIMD * RowsPerSIMD;\n";
  // Contiguous keys load faster, but their swaps cost each query. Only a SIMD group of four D=128
  // queries is faster with scattered keys; three measured even, and D=256 holds at most two.
  defines += "constant bool ContiguousKeys = QueriesPerSIMD < 4;\n";
  if (relativeBias)
    defines += "constant uint RelativeBiasExtent [[function_constant(11)]];\n";
  if (masked) {
    // A mask value at or below the general attention kernel's threshold closes the key there, so
    // it closes it here too: half the most negative FP16 value with FP16 inputs, else of FP32.
    std::ostringstream threshold;
    threshold << std::setprecision(std::numeric_limits<float>::max_digits10)
        << (memoryPrecision == GEMMOperandPrecision::FP16 ? -65504.0f * 0.5f : -std::numeric_limits<float>::max() * 0.5f);
    defines += "constant float MaskedThreshold = float(" + threshold.str() + ");\n";
  }
  return defines;
}
