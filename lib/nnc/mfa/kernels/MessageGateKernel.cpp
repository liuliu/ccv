#include "MessageGateKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "CodeWriter.hpp"

MessageGateKernel::MessageGateKernel(MessageGateKernelDescriptor descriptor, MTL::Device* const device) {
  memoryPrecision = descriptor.memoryPrecision;
  downPrecision = descriptor.downPrecision;
  weightPrecision = descriptor.weightPrecision;
  downQuantized = descriptor.downQuantized;
  quantized = descriptor.quantized;
  rank = descriptor.rank;
  // LanesPerChannel lanes share a channel and each takes VectorsPerLane vectors of 4 of its R
  // weights: the largest power of two up to 8 that divides R / 4, as the lanes add their sums by xor
  // shuffles, so a row's channels still spread over many threadgroups when there is one row.
  lanesPerChannel = std::min(8u, (rank / 4) & (0u - rank / 4));
  threadgroupSize = MTL::Size(256, 1, 1);
  channelsPerThreadgroup = 256 / lanesPerChannel;
  rowsPerThreadgroup = 8;
  mixRows = descriptor.mixRows;
  source = createSource();

  NS::Error* error = nil;
  library = ccv::nnc::mfa::compile_source(device, NS::String::string(source.c_str(), NS::UTF8StringEncoding), &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string MessageGateKernel::createSource() const noexcept {
  CodeWriter source;
  // The types use the Metal library's vectors.
  source += "#include <metal_stdlib>\nusing namespace metal;\n\n" + createConstants() + "\n";
  source.SetValue("SCALES", quantized ?
    "  const float forget_scale = float(forget_scales[c]);\n  const float gain_scale = float(gain_scales[c]);\n" :
    "  const float forget_scale = 1;\n  const float gain_scale = 1;\n");
  source.SetValue("ROW_SCALE", downQuantized ? "float(down_scales[r])" : "1.0f");
  source += R"(
// mix[t, r] = down[r, :C] . state[t] / rms(state[t]) + down[r, C:] . message[t] / rms(message[t]), with
// rms the root of a row's mean square plus epsilon: the normalizations fold into the dot products, and
// a quantized row's scale into the result.
//
// The mix of a few rows: a threadgroup takes rank r of MixRows rows, and its lanes split the columns.
// Each lane keeps MixValues = 4 MixRows sums: the rows' state dot products, their message dot
// products, then the sums of squares of their states and of their messages. A SIMD group reduces
// them so that lane i holds sum i (the lanes first add across the lanes that keep the same sums,
// then halve the sums each keeps with every exchange), and the first MixRows threads add the SIMD
// groups' sums.
kernel void message_gate_mix(
  device const real* state [[buffer(0)]],
  device const real* message [[buffer(1)]],
  device const dtype* down [[buffer(2)]],
  device float* mix [[buffer(3)]],
  device const dreal* down_scales [[buffer(4)]],
  constant uint& row_count [[buffer(5)]],
  uint2 tgid [[threadgroup_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane [[thread_index_in_simdgroup]])
{
  const uniform<uint> rows = make_uniform(row_count);
  threadgroup float partials[(ThreadsPerThreadgroup / 32) * MixValues];
  const uint r = tgid.x;
  const uint t0 = tgid.y * MixRows;
  float v[MixValues];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < MixValues; i++)
    v[i] = 0;
  // Row r of the down projection, and its message half C / 4 vectors later.
  device const dtype4* const down_row = (device const dtype4*)(down + size_t(r) * 2 * C);
  for (uint k = lid; k < C / 4; k += ThreadsPerThreadgroup) {
    const float4 ws = float4(down_row[k]);
    const float4 wm = float4(down_row[C / 4 + k]);
    #pragma clang loop unroll(full)
    for (uint t = 0; t < MixRows; t++) {
      // Rows past the last read the last again; their mixes are never stored.
      const size_t offset = size_t(min(t0 + t, rows - 1)) * (C / 4) + k;
      const float4 s = float4(((device const real4*)state)[offset]);
      const float4 m = float4(((device const real4*)message)[offset]);
      v[t] += dot(ws, s);
      v[MixRows + t] += dot(wm, m);
      v[2 * MixRows + t] += dot(s, s);
      v[3 * MixRows + t] += dot(m, m);
    }
  }
  #pragma clang loop unroll(full)
  for (uint offset = 16; offset >= MixValues; offset /= 2) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < MixValues; i++)
      v[i] += simd_shuffle_xor(v[i], ushort(offset));
  }
  #pragma clang loop unroll(full)
  for (uint half_count = MixValues / 2; half_count >= 1; half_count /= 2) {
    const bool upper = (lane & half_count) != 0;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < half_count; i++) {
      const float keep = upper ? v[i + half_count] : v[i];
      const float give = upper ? v[i] : v[i + half_count];
      v[i] = keep + simd_shuffle_xor(give, ushort(half_count));
    }
  }
  if (lane < MixValues)
    partials[sidx * MixValues + lane] = v[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid >= MixRows || t0 + lid >= rows)
    return;
  float sums[4] = { 0, 0, 0, 0 };
  #pragma clang loop unroll(full)
  for (uint g = 0; g < ThreadsPerThreadgroup / 32; g++) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++)
      sums[i] += partials[g * MixValues + i * MixRows + lid];
  }
  mix[size_t(t0 + lid) * R + r] = {{ROW_SCALE}} * (sums[0] * precise::rsqrt(sums[2] / C + Epsilon) + sums[1] * precise::rsqrt(sums[3] / C + Epsilon));
}

// Many rows: each threadgroup writes one row's RMS-normalized state and message side by side, [2C]
// in FP32, for the down GEMM.
kernel void message_gate_normalize(
  device const real* state [[buffer(0)]],
  device const real* message [[buffer(1)]],
  device float* normalized [[buffer(2)]],
  uint t [[threadgroup_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane [[thread_index_in_simdgroup]])
{
  threadgroup float partials[2 * (ThreadsPerThreadgroup / 32)];
  device const real4* const s = (device const real4*)(state + size_t(t) * C);
  device const real4* const m = (device const real4*)(message + size_t(t) * C);
  float squares[2] = { 0, 0 };
  for (uint k = lid; k < C / 4; k += ThreadsPerThreadgroup) {
    const float4 a = float4(s[k]);
    const float4 b = float4(m[k]);
    squares[0] += dot(a, a);
    squares[1] += dot(b, b);
  }
  squares[0] = simd_sum(squares[0]);
  squares[1] = simd_sum(squares[1]);
  if (lane == 0) {
    partials[sidx] = squares[0];
    partials[ThreadsPerThreadgroup / 32 + sidx] = squares[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  squares[0] = 0;
  squares[1] = 0;
  #pragma clang loop unroll(full)
  for (uint g = 0; g < ThreadsPerThreadgroup / 32; g++) {
    squares[0] += partials[g];
    squares[1] += partials[ThreadsPerThreadgroup / 32 + g];
  }
  const float inv_s = precise::rsqrt(squares[0] / C + Epsilon);
  const float inv_m = precise::rsqrt(squares[1] / C + Epsilon);
  device float4* const out = (device float4*)(normalized + size_t(t) * 2 * C);
  for (uint k = lid; k < C / 4; k += ThreadsPerThreadgroup) {
    out[k] = float4(s[k]) * inv_s;
    out[C / 4 + k] = float4(m[k]) * inv_m;
  }
}

// y = state - tanh(mix forget^T) state + tanh(mix gain^T) message, for each row. A threadgroup
// takes RowsPerThreadgroup rows and ChannelsPerThreadgroup channels of them. LanesPerChannel lanes
// of a SIMD group share a channel: each reads its slice of the channel's forget and gain rows, so
// the group's loads are contiguous, and the lanes add their partial sums with shuffles. The rows'
// mixes are in threadgroup memory.
kernel void message_gate(
  device const real* state [[buffer(0)]],
  device const real* message [[buffer(1)]],
  device const float* mix [[buffer(2)]],
  device const wtype* forget [[buffer(3)]],
  device const wtype* gain [[buffer(4)]],
  device real* y [[buffer(5)]],
  device const wreal* forget_scales [[buffer(6)]],
  device const wreal* gain_scales [[buffer(7)]],
  constant uint& row_count [[buffer(8)]],
  uint3 tgid [[threadgroup_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]])
{
  const uniform<uint> rows = make_uniform(row_count);
  threadgroup float4 mixes[RowsPerThreadgroup * R / 4];
  const uint t0 = tgid.x * RowsPerThreadgroup;
  const uint count = min(RowsPerThreadgroup, rows - t0);
  // Rows past the last are zeros; their sums are never stored.
  for (uint i = lid; i < RowsPerThreadgroup * R; i += ThreadsPerThreadgroup)
    ((threadgroup float*)mixes)[i] = i < count * R ? mix[size_t(t0) * R + i] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tgid.y * ChannelsPerThreadgroup + lid / LanesPerChannel;
  const uint slice = lid % LanesPerChannel;
  if (c >= C)
    return;
  // This lane's VectorsPerLane vectors of 4 of the channel's R projection weights.
  device const wtype4* const forget_row = (device const wtype4*)(forget + size_t(c) * R) + slice * VectorsPerLane;
  device const wtype4* const gain_row = (device const wtype4*)(gain + size_t(c) * R) + slice * VectorsPerLane;
  float f[RowsPerThreadgroup];
  float g[RowsPerThreadgroup];
  #pragma clang loop unroll(full)
  for (uint t = 0; t < RowsPerThreadgroup; t++) {
    f[t] = 0;
    g[t] = 0;
  }
  #pragma clang loop unroll(full)
  for (uint v = 0; v < VectorsPerLane; v++) {
    const float4 wf = float4(forget_row[v]);
    const float4 wg = float4(gain_row[v]);
    #pragma clang loop unroll(full)
    for (uint t = 0; t < RowsPerThreadgroup; t++) {
      const float4 m = mixes[t * (R / 4) + slice * VectorsPerLane + v];
      f[t] += dot(wf, m);
      g[t] += dot(wg, m);
    }
  }
  // The channel's lanes are adjacent, so xor shuffles below LanesPerChannel stay within them.
  #pragma clang loop unroll(full)
  for (uint offset = 1; offset < LanesPerChannel; offset *= 2) {
    #pragma clang loop unroll(full)
    for (uint t = 0; t < RowsPerThreadgroup; t++) {
      f[t] += simd_shuffle_xor(f[t], ushort(offset));
      g[t] += simd_shuffle_xor(g[t], ushort(offset));
    }
  }
{{SCALES}}  // Each lane of the channel writes every LanesPerChannel-th row.
  #pragma clang loop unroll(full)
  for (uint t = 0; t < RowsPerThreadgroup; t++) {
    if (t % LanesPerChannel != slice || t >= count)
      continue;
    const size_t i = size_t(t0 + t) * C + c;
    const float x = float(state[i]);
    y[i] = real(x - precise::tanh(f[t] * forget_scale) * x + precise::tanh(g[t] * gain_scale) * float(message[i]));
  }
}
)";
  return source.ToString();
}

static std::string _message_gate_type(const GEMMOperandPrecision precision) {
  return precision == GEMMOperandPrecision::FP32 ? "float" : (precision == GEMMOperandPrecision::BF16 ? "bfloat" : "half");
}

std::string MessageGateKernel::createConstants() const noexcept {
  std::string defines = "";
  defines += "typedef " + _message_gate_type(memoryPrecision) + " real;\n";
  defines += "typedef vec<real, 4> real4;\n";
  defines += "typedef " + _message_gate_type(downPrecision) + " dreal;\n";
  defines += "typedef " + _message_gate_type(weightPrecision) + " wreal;\n";
  // A quantized projection is its signed bytes, and its row scales are in its base type.
  defines += downQuantized ? "typedef char dtype;\ntypedef char4 dtype4;\n" : "typedef dreal dtype;\ntypedef vec<dreal, 4> dtype4;\n";
  defines += quantized ? "typedef char wtype;\ntypedef char4 wtype4;\n" : "typedef wreal wtype;\ntypedef vec<wreal, 4> wtype4;\n";
  defines += "constant uint R = " + std::to_string(rank) + ";\n";
  defines += "constant uint RowsPerThreadgroup = " + std::to_string(rowsPerThreadgroup) + ";\n";
  defines += "constant uint ThreadsPerThreadgroup = " + std::to_string(threadgroupSize.width) + ";\n";
  defines += "constant uint LanesPerChannel = " + std::to_string(lanesPerChannel) + ";\n";
  defines += "constant uint VectorsPerLane = " + std::to_string(rank / 4 / lanesPerChannel) + ";\n";
  defines += "constant uint ChannelsPerThreadgroup = " + std::to_string(channelsPerThreadgroup) + ";\n";
  const uint32_t tr = std::max(mixRows, 1u);
  defines += "constant uint MixRows = " + std::to_string(tr) + ";\n";
  defines += "constant uint MixValues = " + std::to_string(4 * tr) + ";\n";
  defines += "constant uint C [[function_constant(0)]];\n";
  defines += "constant float Epsilon [[function_constant(1)]];\n";
  return defines;
}
