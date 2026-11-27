#include "CausalConvolutionKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "CodeWriter.hpp"

CausalConvolutionKernel::CausalConvolutionKernel(CausalConvolutionKernelDescriptor descriptor, MTL::Device* const device) {
  silu = descriptor.silu;
  dynamicWeights = descriptor.dynamicWeights;
  memoryPrecision = descriptor.memoryPrecision;
  threadgroupSize = MTL::Size(256, 1, 1);
  source = createSource();

  NS::Error* error = nil;
  library = ccv::nnc::mfa::compile_source(device, NS::String::string(source.c_str(), NS::UTF8StringEncoding), &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string CausalConvolutionKernel::createSource() const noexcept {
  CodeWriter source;
  source += createConstants() + "\n";
  if (dynamicWeights) {
    // Per-token dynamic weights [B, T, G, K], which each token adds to the weights of the channels in each group.
    source.SetValue("DYNAMIC_WEIGHTS_BUFFER", "  device const real* dynamic_weights [[buffer(5)]],\n");
    source.SetValue("DYNAMIC_OFFSET", "    const uint dynamic_offset = ((b * T + row) * G + c / (C / G)) * K;\n");
    source.SetValue("WEIGHT", "(float(weight[c * K + j]) + float(dynamic_weights[dynamic_offset + j]))");
  } else {
    source.SetValue("DYNAMIC_WEIGHTS_BUFFER", "");
    source.SetValue("DYNAMIC_OFFSET", "");
    source.SetValue("WEIGHT", "float(weight[c * K + j])");
  }
  source += R"(
#include <metal_stdlib>
using namespace metal;

// The context of batch b is its K - 1 previous rows followed by its T rows of x.
inline float context_value(device const real* x, device const real* previous, uint b, uint r, uint c, uint T) {
  return r < K - 1 ? float(previous[(b * (K - 1) + r) * C + c]) : float(x[(b * T + r - (K - 1)) * C + c]);
}

kernel void causal_convolution(
  device const real* x [[buffer(0)]],
  device const real* previous [[buffer(1)]],
  device const real* weight [[buffer(2)]],
  device real* y [[buffer(3)]],
  device real* next [[buffer(4)]],
{{DYNAMIC_WEIGHTS_BUFFER}}  uint3 tpig [[thread_position_in_grid]],
  uint3 grid [[threadgroups_per_grid]])
{
  // The grid has a row for each of the T tokens and then for each of the R next rows.
  const uniform<uint> T = make_uniform(grid.y - R);
  const uint c = tpig.x;
  const uint row = tpig.y;
  const uint b = tpig.z;
  if (c >= C)
    return;
  if (row < T) {
    float sum = 0;
{{DYNAMIC_OFFSET}}    for (uint j = 0; j < K; j++)
      sum += {{WEIGHT}} * context_value(x, previous, b, row + j, c, T);
    if (silu)
      sum = sum / (1.0f + exp(-sum));
    y[(b * T + row) * C + c] = real(sum);
  } else {
    // The next rows are the context's last R rows.
    const uint j = row - T;
    next[(b * R + j) * C + c] = real(context_value(x, previous, b, K - 1 + T - R + j, c, T));
  }
}
)";
  return source.ToString();
}

std::string CausalConvolutionKernel::createConstants() const noexcept {
  std::string defines = "typedef " + memoryPrecision.name() + " real;\n";
  defines += silu ? "constant bool silu = true;\n" : "constant bool silu = false;\n";
  defines += "constant uint C [[function_constant(0)]];\n";
  defines += "constant uint K [[function_constant(1)]];\n";
  defines += "constant uint R [[function_constant(2)]];\n";
  if (dynamicWeights)
    defines += "constant uint G [[function_constant(3)]];\n";
  return defines;
}
