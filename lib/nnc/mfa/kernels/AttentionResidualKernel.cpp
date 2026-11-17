#include "AttentionResidualKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "CodeWriter.hpp"

AttentionResidualKernel::AttentionResidualKernel(AttentionResidualKernelDescriptor descriptor, MTL::Device* const device) {
  queryCount = descriptor.queryCount;
  memoryPrecision = descriptor.memoryPrecision;
  // One threadgroup mixes a row: a thread's loads are serial, so a row's channels are spread over
  // as many threads as a threadgroup has.
  threadgroupSize = MTL::Size(1024, 1, 1);
  source = createSource();

  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string AttentionResidualKernel::createSource() const noexcept {
  CodeWriter source;
  source += createConstants() + "\n";
  std::string outputs = "";
  std::string stores = "";
  for (uint32_t q = 0; q < queryCount; q++) {
    outputs += "  device real* y" + std::to_string(q) + " [[buffer(" + std::to_string(3 + q) + ")]],\n";
    stores += "    y" + std::to_string(q) + "[size_t(row) * C + c] = real(y[" + std::to_string(q) + "]);\n";
  }
  source.SetValue("OUTPUT_BUFFERS", outputs);
  source.SetValue("OUTPUT_STORES", stores);
  source += R"(
#include <metal_stdlib>
using namespace metal;

// The states of a row are its block states, [block, row, channel], and then its partial state.
// Each query's output is a softmax mixture of the RMS-normalized states, scored by the query's
// dot product with them. A threadgroup takes a row, and each thread every THREADS-th channel.
kernel void attention_residual(
  device const real* blocks [[buffer(0)]],
  device const real* partial [[buffer(1)]],
  device const real* queries [[buffer(2)]],
{{OUTPUT_BUFFERS}}  uint3 tgid [[threadgroup_position_in_grid]],
  uint3 tgpg [[threadgroups_per_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane [[thread_index_in_simdgroup]],
  uint lid [[thread_index_in_threadgroup]])
{
  const uint row = tgid.x;
  const uint rows = tgpg.x;
  const uint state_count = block_count + 1;
  // Each SIMD group's share of a state's sum of squares and of its dot product with each query.
  threadgroup float partials[MAX_STATES * (Q + 1)][SIMD_GROUPS];
  // A state's weight in each query's mixture, with the state's normalization.
  threadgroup float mix[MAX_STATES * Q];
  for (uint s = 0; s < state_count; s++) {
    device const real* const state = s < block_count ? blocks + (size_t(s) * rows + row) * C : partial + size_t(row) * C;
    float sum = 0;
    float dot[Q] = {0};
    for (uint c = lid; c < C; c += THREADS) {
      const float x = float(state[c]);
      sum += x * x;
      #pragma clang loop unroll(full)
      for (uint q = 0; q < Q; q++)
        dot[q] += x * float(queries[size_t(q) * C + c]);
    }
    sum = simd_sum(sum);
    #pragma clang loop unroll(full)
    for (uint q = 0; q < Q; q++)
      dot[q] = simd_sum(dot[q]);
    if (lane == 0) {
      partials[s * (Q + 1)][sidx] = sum;
      #pragma clang loop unroll(full)
      for (uint q = 0; q < Q; q++)
        partials[s * (Q + 1) + 1 + q][sidx] = dot[q];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    // Each lane brings one SIMD group's share.
    float reciprocals[MAX_STATES];
    float weights[MAX_STATES][Q];
    float maximum[Q];
    for (uint s = 0; s < state_count; s++) {
      const float sum = simd_sum(lane < SIMD_GROUPS ? partials[s * (Q + 1)][lane] : 0);
      reciprocals[s] = rsqrt(sum / float(C) + epsilon);
      #pragma clang loop unroll(full)
      for (uint q = 0; q < Q; q++) {
        const float score = simd_sum(lane < SIMD_GROUPS ? partials[s * (Q + 1) + 1 + q][lane] : 0) * reciprocals[s] * scale;
        weights[s][q] = score;
        maximum[q] = s == 0 ? score : max(maximum[q], score);
      }
    }
    if (lane == 0) {
      // The softmax weights, times the normalization of the states they mix.
      float total[Q] = {0};
      for (uint s = 0; s < state_count; s++) {
        #pragma clang loop unroll(full)
        for (uint q = 0; q < Q; q++) {
          weights[s][q] = exp(weights[s][q] - maximum[q]);
          total[q] += weights[s][q];
        }
      }
      for (uint s = 0; s < state_count; s++) {
        #pragma clang loop unroll(full)
        for (uint q = 0; q < Q; q++)
          mix[s * Q + q] = weights[s][q] * reciprocals[s] / total[q];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = lid; c < C; c += THREADS) {
    float y[Q] = {0};
    for (uint s = 0; s < state_count; s++) {
      const float x = s < block_count ? float(blocks[(size_t(s) * rows + row) * C + c]) : float(partial[size_t(row) * C + c]);
      #pragma clang loop unroll(full)
      for (uint q = 0; q < Q; q++)
        y[q] += mix[s * Q + q] * x;
    }
{{OUTPUT_STORES}}  }
}
)";
  return source.ToString();
}

std::string AttentionResidualKernel::createConstants() const noexcept {
  const std::string type = memoryPrecision == GEMMOperandPrecision::FP32 ? "float" :
    (memoryPrecision == GEMMOperandPrecision::BF16 ? "bfloat" : "half");
  std::string defines = "typedef " + type + " real;\n";
  defines += "constant uint Q = " + std::to_string(queryCount) + ";\n";
  defines += "constant uint MAX_STATES = " + std::to_string(CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_STATES) + ";\n";
  defines += "constant uint THREADS = " + std::to_string(threadgroupSize.width) + ";\n";
  defines += "constant uint SIMD_GROUPS = " + std::to_string(threadgroupSize.width / 32) + ";\n";
  defines += "constant uint C [[function_constant(0)]];\n";
  defines += "constant uint block_count [[function_constant(1)]];\n";
  defines += "constant float epsilon [[function_constant(2)]];\n";
  defines += "constant float scale [[function_constant(3)]];\n";
  return defines;
}
