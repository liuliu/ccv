#include "NAInt8SolAttentionKernel.hpp"
#include "CodeWriter.hpp"
#include <algorithm>
#include "../ccv_nnc_mfa_error.hpp"

NAInt8SolAttentionKernel::NAInt8SolAttentionKernel(
    NAInt8SolAttentionKernelDescriptor descriptor,
    MTL::Device *const device)
{
  blockSize = descriptor.blockSize;
  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nullptr;
  library = NS::TransferPtr(device->newLibrary(string, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint32_t NAInt8SolAttentionKernel::threadgroupSize(const NAInt8SolAttentionDescriptor& descriptor) const noexcept {
  switch (descriptor.stage) {
    case NAInt8SolAttentionStage::VMean: return vMeanThreadgroupSize(descriptor.T, descriptor.H);
    case NAInt8SolAttentionStage::Attention: return descriptor.queryBlockSize * 2;
    default: return 128;
  }
}

MTL::Size NAInt8SolAttentionKernel::threadgroupsPerGrid(const NAInt8SolAttentionDescriptor& descriptor) const noexcept {
  // Runtime geometry must not be retained in the source-only kernel cache.
  const uint32_t J = (descriptor.T + blockSize - 1) / blockSize;
  const uint32_t JP = (J + 63) / 64 * 64;
  const uint32_t QJ = (descriptor.T + descriptor.queryBlockSize - 1) / descriptor.queryBlockSize;
  MTL::Size grid(QJ, descriptor.N * descriptor.H, 1);
  switch (descriptor.stage) {
    case NAInt8SolAttentionStage::VMean: return vMeanThreadgroupsPerGrid(descriptor.N, descriptor.T, descriptor.H);
    case NAInt8SolAttentionStage::Quantize: grid.width = (descriptor.T + 63) / 64; break;
    case NAInt8SolAttentionStage::Pool: grid.width = JP; break;
    case NAInt8SolAttentionStage::PrepareSummaries: grid.width = JP / 64 + 1; break;
    case NAInt8SolAttentionStage::Route: break;
    case NAInt8SolAttentionStage::Attention: {
      // Match native attention scheduling without rearranging token/head data.
      uint64_t paddedQueries = 1, paddedHeads = 1;
      while (paddedQueries < QJ) paddedQueries *= 2;
      while (paddedHeads < descriptor.H) paddedHeads *= 2;
      return MTL::Size(paddedQueries * paddedHeads, 1, descriptor.N);
    }
  }
  return grid;
}

std::string NAInt8SolAttentionKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("BLOCK_SIZE", std::to_string(blockSize));
  createConstants(source);
  createMortonUtilities(source);
  createVMean(source);
  createQuantize(source);
  if (blockSize != 64)
    createPool(source);
  createPrepareSummaries(source);
  createRoute(source);
  createAttention(source);
  return source.ToString();
}

// INT8 Sol attention with pooled routing and a shared softmax normalization.
// Exact tokens use INT8 QK/PV; summaries use INT8 QK and normalized FP16 PV.
// No dense T x T score matrix is materialized.
void NAInt8SolAttentionKernel::createConstants(CodeWriter& source) const noexcept {
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant uint SOL_N [[function_constant(0)]];
constant uint SOL_T [[function_constant(1)]];
constant uint SOL_H [[function_constant(2)]];
constant uint SOL_QB [[function_constant(3)]];
struct Params {
  uint N, T, H, B, begin, end;
  float scale, tau;
  uint QB;
  uint local_block_radius;
};
)";
}

uint32_t NAInt8SolAttentionKernel::vMeanVectorsPerTile(uint32_t T, uint32_t H) noexcept {
  const uint32_t maxTile = T <= 20480 ? 4 : 8;
  uint32_t tile = 1;
  while (tile < maxTile && H >= tile * 8)
    tile *= 2;
  return tile;
}

uint32_t NAInt8SolAttentionKernel::vMeanThreadgroupSize(uint32_t T, uint32_t H) noexcept {
  return std::max(T <= 20480 ? 256u : 128u, 32 * vMeanVectorsPerTile(T, H));
}

MTL::Size NAInt8SolAttentionKernel::vMeanThreadgroupsPerGrid(uint32_t N, uint32_t T, uint32_t H) noexcept {
  uint32_t paddedHeads = 1;
  while (paddedHeads < H)
    paddedHeads *= 2;
  return MTL::Size(uint64_t(paddedHeads) * (32 / vMeanVectorsPerTile(T, H)), 1, N);
}

void NAInt8SolAttentionKernel::createMortonUtilities(CodeWriter& source) const noexcept {
  source += R"(
inline uint compact_morton_even_bits(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

inline uint2 morton_decode_2d(uint code) {
  return uint2(compact_morton_even_bits(code),
               compact_morton_even_bits(code >> 1));
}

inline uint lower_bits_mask(uint bit_count) {
  if (bit_count == 0)
    return 0;
  return (1u << bit_count) - 1;
}

inline uint2 morton_decode_rectangular_2d(uint code,
                                         uint x_bits,
                                         uint y_bits) {
  const uint paired_bits = min(x_bits, y_bits);
  const uint paired_code = code & lower_bits_mask(paired_bits * 2);
  uint2 tile = morton_decode_2d(paired_code);
  uint tail = code >> (paired_bits * 2);
  if (x_bits > paired_bits) {
    const uint x_extra_bits = x_bits - paired_bits;
    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;
    tail >>= x_extra_bits;
  }
  if (y_bits > paired_bits) {
    tile.y |= tail << paired_bits;
  }
  return tile;
}

)";
}

void NAInt8SolAttentionKernel::createVMean(CodeWriter& source) const noexcept {
  source += R"(
kernel void sol_v_mean(device const packed_half4* V [[buffer(2)]], device float4* mean [[buffer(15)]],
  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint3 group [[threadgroup_position_in_grid]]) {
  const uint threads = SOL_T <= 20480 ? 256 : 128;
  const uint max_tile = SOL_T <= 20480 ? 4 : 8;
  uint tile = 1;
  while (tile < max_tile && SOL_H >= tile * 8) tile *= 2;
  const uint dispatch_threads = max(threads, 32 * tile);
  const uint2 morton = morton_decode_rectangular_2d(group.x,
      32 - clz(32 / tile - 1), SOL_H <= 1 ? 0 : 32 - clz(SOL_H - 1));
  const uint h = morton.y, n = group.z;
  if (h >= SOL_H) return;
  const uint nh = n * SOL_H + h;
  // The fixed 16 KiB array also fits the validator's doubled allocation.
  // Reuse it for the SIMD sums after every reader has consumed the partials.
  threadgroup float4 partials[1024];
  const uint vec_dim = morton.x * tile + tid % tile;
  const uint rows_per_part = dispatch_threads / tile;
  for (uint part = 0; part < threads / rows_per_part; ++part) {
    const uint row = tid / tile + part * rows_per_part;
    float4 sum = 0;
    for (uint t = row; t < SOL_T; t += threads)
      sum += float4(V[((ulong(n) * SOL_T + t) * SOL_H + h) * 32 + vec_dim]);
    partials[(tid % tile) * threads + row] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row = tid % threads, sg = row / 32;
  for (uint vec = tid / threads; vec < tile; vec += dispatch_threads / threads) {
    float4 sum = partials[vec * threads + row];
    for (uint offset = 16; offset > 0; offset /= 2)
      for (uint component = 0; component < 4; ++component)
        sum[component] += simd_shuffle_xor(sum[component], offset);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[vec * threads + sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float4 reduced = lane < threads / 32 ? partials[vec * threads + lane] : float4(0);
      for (uint offset = 16; offset > 0; offset /= 2)
        for (uint component = 0; component < 4; ++component)
          reduced[component] += simd_shuffle_xor(reduced[component], offset);
      if (lane == 0) mean[nh * 32 + morton.x * tile + vec] = reduced * (1.0f / float(SOL_T));
    }
  }
}
)";
}

void NAInt8SolAttentionKernel::createQuantize(CodeWriter& source) const noexcept {
  source += R"(
kernel void sol_quantize(device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]],
  device const half* V [[buffer(2)]], device int8_t* QI [[buffer(9)]], device int8_t* KI [[buffer(10)]],
  device int8_t* VI [[buffer(11)]], device float* QS [[buffer(12)]], device float* KS [[buffer(13)]],
  device float* VS [[buffer(14)]], device const float* mean [[buffer(15)]], constant Params& p [[buffer(21)]],
  device float* QC [[buffer(4)]], device half* KC [[buffer(5)]], device float* VC [[buffer(6)]],
  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint TP = (SOL_T + 63) / 64 * 64, J = TP / 64, j = group.x, nh = group.y, n = nh / SOL_H, h = nh % SOL_H;
  float qm[4] = {0, 0, 0, 0}; float km = 0, vm = 0;
)";
  if (blockSize == 64) {
    source += R"(
  float qsum[4] = {0, 0, 0, 0}; float ksum = 0, vsum = 0;
)";
  }
  source += R"(
  const float average = mean[nh * 128 + tid];
  for (uint r = 0; r < 64 && j * 64 + r < SOL_T; ++r) {
    const ulong index = ((ulong(n) * SOL_T + j * 64 + r) * SOL_H + h) * 128 + tid;
    qm[r / 16] = max(qm[r / 16], abs(float(Q[index])));
    km = max(km, abs(float(K[index]))); vm = max(vm, abs(float(V[index]) - average));
)";
  if (blockSize == 64) {
    source += R"(
    qsum[r / SOL_QB] += float(Q[index]); ksum += float(K[index]); vsum += float(V[index]);
)";
  }
  source += R"(
  }
)";
  if (blockSize == 64) {
    source += R"(
  const uint JP = (J + 63) / 64 * 64, QJ = (SOL_T + SOL_QB - 1) / SOL_QB;
  const uint count = min(64u, SOL_T - j * 64);
  KC[(ulong(nh) * JP + j) * 128 + tid] = half(ksum / count);
  VC[(ulong(nh) * JP + j) * 128 + tid] = vsum / count - average;
  for (uint part = 0; part < 64 / SOL_QB; ++part) {
    const uint qb = j * 64 / SOL_QB + part;
    if (qb < QJ) QC[(ulong(nh) * QJ + qb) * 128 + tid] = qsum[part] / min(SOL_QB, SOL_T - qb * SOL_QB);
  }
)";
  }
  source += R"(
  threadgroup float partials[24];
  for (uint i = 0; i < 4; ++i) { const float m = simd_max(qm[i]); if (lane == 0) partials[sg * 6 + i] = m; }
  km = simd_max(km); vm = simd_max(vm);
  if (lane == 0) { partials[sg * 6 + 4] = km; partials[sg * 6 + 5] = vm; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scales[6], inverse_scales[6];
  for (uint i = 0; i < 6; ++i) {
    float m = 0; for (uint s = 0; s < 4; ++s) m = max(m, partials[s * 6 + i]);
    scales[i] = m > 0 ? m / 127.0f : 1.0f / 127.0f;
    inverse_scales[i] = m > 0 ? 127.0f / m : 127.0f;
  }
  if (tid == 0) {
    for (uint i = 0; i < 4; ++i) QS[nh * J * 4 + j * 4 + i] = scales[i];
    KS[nh * J + j] = scales[4]; VS[nh * J + j] = scales[5];
  }
  for (uint r = 0; r < 64; ++r) {
    const uint t = j * 64 + r;
    // Preserve token/head order in INT8 scratch; only the sequence tail is padded.
    const ulong dst = ((ulong(n) * TP + t) * SOL_H + h) * 128 + tid;
    const ulong src = ((ulong(n) * SOL_T + t) * SOL_H + h) * 128 + tid;
    QI[dst] = t < SOL_T ? int8_t(clamp(rint(float(Q[src]) * inverse_scales[r / 16]), -127.0f, 127.0f)) : 0;
    KI[dst] = t < SOL_T ? int8_t(clamp(rint(float(K[src]) * inverse_scales[4]), -127.0f, 127.0f)) : 0;
    VI[dst] = t < SOL_T ? int8_t(clamp(rint((float(V[src]) - average) * inverse_scales[5]), -127.0f, 127.0f)) : 0;
  }
}
)";
}

void NAInt8SolAttentionKernel::createPool(CodeWriter& source) const noexcept {
  source += R"(
kernel void sol_pool(device const half* Q [[buffer(0)]],
  device const half* K [[buffer(1)]], device const half* V [[buffer(2)]],
  device float* QC [[buffer(4)]], device half* KC [[buffer(5)]],
  device float* VC [[buffer(6)]], constant Params& p [[buffer(21)]],
  device const float* mean [[buffer(15)]],
  uint d [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint J = (SOL_T + {{BLOCK_SIZE}} - 1) / {{BLOCK_SIZE}};
  const uint j = group.x, nh = group.y, n = nh / SOL_H, h = nh % SOL_H;
  const uint JP = (J + 63) / 64 * 64;
  if (j >= J) {
    KC[(ulong(nh) * JP + j) * 128 + d] = 0;
    VC[(ulong(nh) * JP + j) * 128 + d] = 0;
    return;
  }
)";
  source += R"(
  const uint count = min(uint({{BLOCK_SIZE}}), SOL_T - j * {{BLOCK_SIZE}});
  const uint QJ = (SOL_T + SOL_QB - 1) / SOL_QB;
  float q[4] = {0, 0, 0, 0}; float k = 0, v = 0;
  for (uint t = j * {{BLOCK_SIZE}}; t < j * {{BLOCK_SIZE}} + count; ++t) {
    const ulong index = ((ulong(n) * SOL_T + t) * SOL_H + h) * 128 + d;
    q[(t - j * {{BLOCK_SIZE}}) / SOL_QB] += float(Q[index]); k += float(K[index]); v += float(V[index]);
  }
  for (uint part = 0; part < {{BLOCK_SIZE}} / SOL_QB; ++part) {
    const uint qb = j * {{BLOCK_SIZE}} / SOL_QB + part;
    if (qb < QJ) QC[(ulong(nh) * QJ + qb) * 128 + d] = q[part] / min(SOL_QB, SOL_T - qb * SOL_QB);
  }
  KC[(ulong(nh) * JP + j) * 128 + d] = half(k / count);
  VC[(ulong(nh) * JP + j) * 128 + d] = float(v / count
    - mean[nh * 128 + d]
  );
}
)";
}

void NAInt8SolAttentionKernel::createPrepareSummaries(CodeWriter& source) const noexcept {
  source += R"(
kernel void sol_prepare_summaries(device const half* KC [[buffer(5)]],
  device float2* stats [[buffer(7)]], device const float* VC [[buffer(6)]], device int8_t* KCI [[buffer(16)]],
  device half* VCH [[buffer(17)]], device float* KCS [[buffer(18)]],
  device float* VCS [[buffer(19)]], constant Params& p [[buffer(21)]],
  uint tid [[thread_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
  uint2 group [[threadgroup_position_in_grid]]) {
  const uint J = (SOL_T + {{BLOCK_SIZE}} - 1) / {{BLOCK_SIZE}};
  const uint JP = (J + 63) / 64 * 64, nh = group.y, first = group.x * 64;

  // Independent head-statistics jobs share this launch with summary jobs.
  if (group.x == JP / 64) {
    float m = 0, v = 0;
    for (uint j = 0; j < J; ++j) {
      const float k = float(KC[(ulong(nh) * JP + j) * 128 + tid]);
      m += k; v += k * k;
    }
    m /= J;
    stats[nh * 128 + tid] = float2(m, max(0.0f, v / J - m * m));
    return;
  }

  float km = 0, vm = 0;
  for (uint r = 0; r < 64 && first + r < J; ++r) {
    // Protected blocks never contribute summaries; do not let their values
    // reduce quantization precision for eligible video blocks in this group.
    const uint block = first + r;
    if (block * {{BLOCK_SIZE}} < p.begin || min((block + 1) * {{BLOCK_SIZE}}, SOL_T) > p.end) continue;
    const ulong index = (ulong(nh) * JP + first + r) * 128 + tid;
    km = max(km, abs(float(KC[index])));
    vm = max(vm, abs(float(VC[index])));
  }
  threadgroup float partials[8];
  km = simd_max(km); vm = simd_max(vm);
  if (lane == 0) { partials[sg * 2] = km; partials[sg * 2 + 1] = vm; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  km = 0; vm = 0;
  for (uint s = 0; s < 4; ++s) {
    km = max(km, partials[s * 2]); vm = max(vm, partials[s * 2 + 1]);
  }
  const float ki = km > 0 ? 127.0f / km : 127.0f;
  const float vi = vm > 0 ? 1.0f / vm : 1.0f;
  if (tid == 0) {
    KCS[nh * (JP / 64) + group.x] = km > 0 ? km / 127.0f : 1.0f / 127.0f;
    VCS[nh * (JP / 64) + group.x] = vm > 0 ? vm : 1.0f;
  }
  for (uint r = 0; r < 64; ++r) {
    const uint block = first + r;
    const bool eligible = block < J && block * {{BLOCK_SIZE}} >= p.begin && min((block + 1) * {{BLOCK_SIZE}}, SOL_T) <= p.end;
    const ulong index = (ulong(nh) * JP + first + r) * 128 + tid;
    KCI[index] = eligible ? int8_t(clamp(rint(float(KC[index]) * ki), -127.0f, 127.0f)) : 0;
    VCH[index] = eligible ? half(float(VC[index]) * vi) : half(0);
  }
}
)";
}

void NAInt8SolAttentionKernel::createRoute(CodeWriter& source) const noexcept {
  source += R"(
kernel void sol_route(device const float* QC [[buffer(4)]],
  device const half* KC [[buffer(5)]], device const float2* stats [[buffer(7)]],
  device uchar* routes [[buffer(8)]], constant Params& p [[buffer(21)]],
  device uint* route_bits [[buffer(20)]],
  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint J = (SOL_T + {{BLOCK_SIZE}} - 1) / {{BLOCK_SIZE}}, qb = group.x, nh = group.y;
  const uint QJ = (SOL_T + SOL_QB - 1) / SOL_QB;
  QC += (ulong(nh) * QJ + qb) * 128;
  KC += ulong(nh) * ((J + 63) / 64 * 64) * 128;
  routes += (ulong(nh) * QJ + qb) * J;
  const float g = p.scale * 1.4426950408889634f;
  float mu = 0, variance = 0;
  // Each SIMD group derives the same threshold independently.
  for (uint d = lane; d < 128; d += 32) {
    const float q = QC[d];
    const float2 s = stats[nh * 128 + d];
    mu += q * s.x; variance += q * q * s.y;
  }
  const float threshold = g * simd_sum(mu) + p.tau * sqrt(g * g * simd_sum(variance) + 1e-6f);
  const bool exact_query = qb * SOL_QB < p.begin || min((qb + 1) * SOL_QB, SOL_T) > p.end;
  for (uint j = sg; j < J; j += 4) {
    float dot = 0;
    for (uint d = lane; d < 128; d += 32) dot += QC[d] * float(KC[j * 128 + d]);
    const float score = simd_sum(dot) * g;
    if (lane == 0) routes[j] = exact_query || j * {{BLOCK_SIZE}} < p.begin || min((j + 1) * {{BLOCK_SIZE}}, SOL_T) > p.end || abs(int(qb * SOL_QB / {{BLOCK_SIZE}}) - int(j)) <= p.local_block_radius || score > threshold;
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint words = (J + 31) / 32;
  for (uint word = tid; word < words; word += 128) {
    uint bits = 0;
    for (uint bit = 0; bit < 32 && word * 32 + bit < J; ++bit)
      bits |= uint(routes[word * 32 + bit] != 0) << bit;
    route_bits[(ulong(nh) * QJ + qb) * words + word] = bits;
  }
}
)";
}

void NAInt8SolAttentionKernel::createAttention(CodeWriter& source) const noexcept {
  source += R"(
kernel void sol_attention(device half* O_buf [[buffer(3)]],
  device const uchar* routes [[buffer(8)]],
  device int8_t* QI_buf [[buffer(9)]], device int8_t* KI_buf [[buffer(10)]], device int8_t* VI_buf [[buffer(11)]],
  device const float* QS [[buffer(12)]], device const float* KS [[buffer(13)]], device const float* VS [[buffer(14)]],
  device const float* V_mean [[buffer(15)]],
  device int8_t* KCI_buf [[buffer(16)]], device half* VCH_buf [[buffer(17)]],
  device const float* KCS [[buffer(18)]], device const float* VCS [[buffer(19)]],
  device const uint* route_bits [[buffer(20)]],
  constant Params& p [[buffer(21)]],
  uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint query_groups = (SOL_T + SOL_QB - 1) / SOL_QB;
  const uint2 tile = morton_decode_rectangular_2d(group.x,
      query_groups <= 1 ? 0 : 32 - clz(query_groups - 1),
      SOL_H <= 1 ? 0 : 32 - clz(SOL_H - 1));
  if (tile.x >= query_groups || tile.y >= SOL_H || group.z >= SOL_N) return;
  const uint n = group.z, h = tile.y, nh = n * SOL_H + h;
  const uint J = (SOL_T + {{BLOCK_SIZE}} - 1) / {{BLOCK_SIZE}};
  const uint row = tile.x * SOL_QB + sg * 16;
  // Padded SIMD groups must participate in the exact-pass threadgroup barriers.
  // Their Q tiles are zero padded and all output accesses are row guarded.
  const uint qb = row / SOL_QB, QJ = (SOL_T + SOL_QB - 1) / SOL_QB;
  const uint TP = (SOL_T + 63) / 64 * 64, JP = (J + 63) / 64 * 64;
  O_buf += ulong(n) * SOL_T * SOL_H * 128;
  routes += (ulong(nh) * QJ + qb) * J;
  const float g = p.scale * 1.4426950408889634f;

  auto QI = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(QI_buf + ulong(n) * TP * SOL_H * 128, dextents<int32_t, 2>(SOL_H * 128, TP));
  auto KI = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(KI_buf + ulong(n) * TP * SOL_H * 128, dextents<int32_t, 2>(SOL_H * 128, TP));
  auto VI = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(VI_buf + ulong(n) * TP * SOL_H * 128, dextents<int32_t, 2>(SOL_H * 128, TP));
  auto KCI = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(KCI_buf + ulong(nh) * JP * 128, dextents<int32_t, 2>(128, JP));
  auto VCH = tensor<device half, dextents<int32_t, 2>, tensor_inline>(VCH_buf + ulong(nh) * JP * 128, dextents<int32_t, 2>(128, JP));
  constexpr uint qkD = 32;
  constexpr auto qk_desc = matmul2d_descriptor(16, {{BLOCK_SIZE}}, qkD, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto pv_desc = matmul2d_descriptor(16, 32, {{BLOCK_SIZE}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> qk;
  matmul2d<pv_desc, execution_simdgroups<1>> pv;
  // Keep one FP32 numerator and softmax state across summary and exact tokens.
  using query_tile_t = decltype(VCH.slice<qkD, 16>(0, 0));
  using key_value_tile_t = decltype(VCH.slice<32, {{BLOCK_SIZE}}>(0, 0));
  auto S = qk.get_destination_cooperative_tensor<query_tile_t, key_value_tile_t, float>();
  auto M = qk.get_row_reduction_destination_cooperative_tensor<query_tile_t, key_value_tile_t, float>();
  auto L = qk.get_row_reduction_destination_cooperative_tensor<query_tile_t, key_value_tile_t, float>();
  auto correction = qk.get_row_reduction_destination_cooperative_tensor<query_tile_t, key_value_tile_t, float>();
  auto P = pv.get_left_input_cooperative_tensor<half, half, float>();
  auto O0 = pv.get_destination_cooperative_tensor<decltype(P), key_value_tile_t, float>();
  auto O1 = pv.get_destination_cooperative_tensor<decltype(P), key_value_tile_t, float>();
  auto O2 = pv.get_destination_cooperative_tensor<decltype(P), key_value_tile_t, float>();
  auto O3 = pv.get_destination_cooperative_tensor<decltype(P), key_value_tile_t, float>();
  auto mq8 = QI.slice<qkD, 16>(h * 128, row);
  auto mk8 = KI.slice<qkD, {{BLOCK_SIZE}}>(h * 128, 0);
  auto mv8 = VI.slice<32, {{BLOCK_SIZE}}>(h * 128, 0);
  auto S8 = qk.get_destination_cooperative_tensor<decltype(mq8), decltype(mk8), int>();
  constexpr auto pv8_desc = matmul2d_descriptor(16, 32, {{BLOCK_SIZE}}, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<pv8_desc, execution_simdgroups<1>> pv8;
  auto P8 = pv8.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto O8 = pv8.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), int>();
  auto CQ0 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto CQ1 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto CQ2 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto CQ3 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  CQ0.load(QI.slice<32, 16>(h * 128, row)); CQ1.load(QI.slice<32, 16>(h * 128 + 32, row));
  CQ2.load(QI.slice<32, 16>(h * 128 + 64, row)); CQ3.load(QI.slice<32, 16>(h * 128 + 96, row));
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < M.get_capacity(); ++i) if (M.is_valid_element(i)) { M[i] = -INFINITY; L[i] = 0; }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < O0.get_capacity(); ++i) if (O0.is_valid_element(i)) { O0[i] = 0; O1[i] = 0; O2[i] = 0; O3[i] = 0; }
)";

  // Emit distinct loops so each traversal has compile-time tensor/precision choices.
  loopAttention(source, true);
  loopAttention(source, false);
  source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < O0.get_capacity(); ++i) if (O0.is_valid_element(i)) {
    const auto idx = O0.get_multidimensional_index(i);
    if (row + idx[1] < SOL_T) {
      const float reciprocal = 1 / *L.map_iterator(O0.get_iterator(i));
      const ulong index = (ulong(row + idx[1]) * SOL_H + h) * 128 + idx[0];
)";
  for (uint32_t slice = 0; slice < 4; ++slice) {
    source.SetValue("SLICE", std::to_string(slice));
    source.SetValue("CHANNEL", std::to_string(slice * 32));
    source += R"(
      O_buf[index + {{CHANNEL}}] = half(O{{SLICE}}[i] * reciprocal
        + V_mean[nh * 128 + idx[0] + {{CHANNEL}}]);
)";
  }
  source += R"(
    }
  }
}
)";
}

void NAInt8SolAttentionKernel::loopAttention(CodeWriter& source, bool summary) const noexcept {
  source.SetValue("KEY_TENSOR", summary ? "KCI" : "KI");
  source.SetValue("KEY_HEAD_OFFSET", summary ? "0" : "h * 128");
  source.SetValue("KEY_SCALE", summary ? "KCS[nh * (JP / 64) + c / 64]" : "KS[nh * TP / 64 + c / 64]");
  source.SetValue("VALID_KEY", summary ? "column < J && !routes[column]" : "column < SOL_T");
  if (summary) {
    source += R"(
  {
    const bool protected_query = qb * SOL_QB < p.begin || min((qb + 1) * SOL_QB, SOL_T) > p.end;
    const uint summaries = protected_query ? 0 : (J + {{BLOCK_SIZE}} - 1) / {{BLOCK_SIZE}};
    for (uint tile = 0; tile < summaries; ++tile) {
      const uint c = tile * {{BLOCK_SIZE}};
)";
  } else {
    source += R"(
  {
    const uint words = (J + 31) / 32;
    route_bits += (ulong(nh) * QJ + qb) * words;
    uint visited_blocks = 0;
    for (uint word = 0; word < words; ++word) {
      uint selected = route_bits[word];
      while (selected != 0) {
        const uint block = word * 32 + ctz(selected);
        selected &= selected - 1;
        const uint c = block * {{BLOCK_SIZE}};
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S8.get_capacity(); ++i) if (S8.is_valid_element(i)) S8[i] = 0;
    auto k0 = {{KEY_TENSOR}}.slice<32, {{BLOCK_SIZE}}>({{KEY_HEAD_OFFSET}} + 0, c); auto k1 = {{KEY_TENSOR}}.slice<32, {{BLOCK_SIZE}}>({{KEY_HEAD_OFFSET}} + 32, c);
    auto k2 = {{KEY_TENSOR}}.slice<32, {{BLOCK_SIZE}}>({{KEY_HEAD_OFFSET}} + 64, c); auto k3 = {{KEY_TENSOR}}.slice<32, {{BLOCK_SIZE}}>({{KEY_HEAD_OFFSET}} + 96, c);
    qk.run(CQ0, k0, S8); qk.run(CQ1, k1, S8);
    qk.run(CQ2, k2, S8); qk.run(CQ3, k3, S8);
    const float scale = QS[nh * TP / 16 + row / 16] * {{KEY_SCALE}} * g;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) S[i] = float(S8[i]) * scale;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) {
      const auto idx = S.get_multidimensional_index(i);
      const uint column = c + idx[0];
      S[i] = {{VALID_KEY}} ? S[i] : -INFINITY;
    }
    auto nextM = qk.get_row_reduction_destination_cooperative_tensor<query_tile_t, key_value_tile_t, float>();
    reduce_rows(S, nextM, reduction_operation::max, -INFINITY);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < M.get_capacity(); ++i) if (M.is_valid_element(i)) {
      const float next = max(M[i], nextM[i]);
      correction[i] = next == -INFINITY ? 1 : fast::exp2(M[i] - next);
      M[i] = next;
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) {
      auto maximum = M.map_iterator(S.get_iterator(i));
      S[i] = S[i] == -INFINITY ? 0 : fast::exp2(S[i] - *maximum);
)";
  if (summary) {
    source += R"(
      // Multiplicity belongs in numerator and denominator, but not in the
      // running maximum used to quantize exact-token probabilities to INT8.
      const uint column = c + S.get_multidimensional_index(i)[0];
      if (column < J) S[i] *= float(min(uint({{BLOCK_SIZE}}), SOL_T - column * {{BLOCK_SIZE}}));
)";
  }
  source += R"(
    }

    auto sum = qk.get_row_reduction_destination_cooperative_tensor<query_tile_t, key_value_tile_t, float>();
    reduce_rows(S, sum, reduction_operation::sum, 0.0f);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < L.get_capacity(); ++i) if (L.is_valid_element(i)) L[i] = L[i] * correction[i] + sum[i];
)";
  if (summary) {
    source += R"(
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < O0.get_capacity(); ++i) if (O0.is_valid_element(i)) {
      const float factor = *correction.map_iterator(O0.get_iterator(i));
      O0[i] *= factor; O1[i] *= factor; O2[i] *= factor; O3[i] *= factor;
    }
)";
  }
  source += R"(
    simdgroup_barrier(mem_flags::mem_none);
)";
  accumulateAttention(source, summary);
  if (!summary) {
    source += R"(
    // Keep SIMD groups near the same K/V block on long traversals.
    if (SOL_T >= 32768 && SOL_QB == 64 && {{BLOCK_SIZE}} == 64 &&
        (++visited_blocks % 2) == 0 && c + {{BLOCK_SIZE}} < SOL_T)
      threadgroup_barrier(mem_flags::mem_none);
      }
)";
  }
  source += R"(
    }
  }
)";
}

void NAInt8SolAttentionKernel::accumulateAttention(CodeWriter& source, bool summary) const noexcept {
  source += "{";
  if (summary) {
    source += R"(
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) P[i] = half(S[i]);
      // Summary V/P stay FP16: INT8 rounding loses cancellation accuracy even
      // for uniform attention. Normalize centered V before the FP16 cast so
      // finite input values cannot overflow, then restore its scale in FP32.
      const float scale = VCS[nh * (JP / 64) + c / 64];
      auto v0 = VCH.slice<32, {{BLOCK_SIZE}}>(0, c); auto v1 = VCH.slice<32, {{BLOCK_SIZE}}>(32, c);
      auto v2 = VCH.slice<32, {{BLOCK_SIZE}}>(64, c); auto v3 = VCH.slice<32, {{BLOCK_SIZE}}>(96, c);
      constexpr auto summary_pv_desc = matmul2d_descriptor(16, 32, {{BLOCK_SIZE}}, false, false, true, matmul2d_descriptor::mode::multiply);
      matmul2d<summary_pv_desc, execution_simdgroups<1>> summary_pv;
      auto temporary = summary_pv.get_destination_cooperative_tensor<decltype(P), decltype(v0), float>();
)";
  } else {
    source += R"(
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) P8[i] = int8_t(clamp(rint(S[i] * 127.0f), 0.0f, 127.0f));
      const float scale = VS[nh * TP / 64 + c / 64] / 127.0f;
      auto v0 = VI.slice<32, {{BLOCK_SIZE}}>(h * 128, c); auto v1 = VI.slice<32, {{BLOCK_SIZE}}>(h * 128 + 32, c);
      auto v2 = VI.slice<32, {{BLOCK_SIZE}}>(h * 128 + 64, c); auto v3 = VI.slice<32, {{BLOCK_SIZE}}>(h * 128 + 96, c);
      // Rescale immediately before accumulating each PV result. Explicit FMA
      // preserves the rounded rescale as its addend instead of reassociating it.
)";
  }
  for (uint32_t slice = 0; slice < 4; ++slice) {
    source.SetValue("SLICE", std::to_string(slice));
    if (summary) {
      source += R"(
      summary_pv.run(P, v{{SLICE}}, temporary);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < O{{SLICE}}.get_capacity(); ++i) if (O{{SLICE}}.is_valid_element(i)) O{{SLICE}}[i] += temporary[i] * scale;
)";
    } else {
      source += R"(
      pv8.run(P8, v{{SLICE}}, O8);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < O{{SLICE}}.get_capacity(); ++i) if (O{{SLICE}}.is_valid_element(i)) {
        O{{SLICE}}[i] = fma(float(O8[i]), scale, O{{SLICE}}[i] * *correction.map_iterator(O{{SLICE}}.get_iterator(i)));
      }
)";
    }
  }
  source += "}";
}
