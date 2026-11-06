#include "CausalConvolutionKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

CausalConvolutionKernel::CausalConvolutionKernel(CausalConvolutionKernelDescriptor descriptor, MTL::Device* const device) {
  silu = descriptor.silu;
  memoryPrecision = descriptor.memoryPrecision;
  threadgroupSize = MTL::Size(256, 1, 1);
  source = createSource();

  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string CausalConvolutionKernel::createSource() const noexcept {
  std::string shader = "#include <metal_stdlib>\nusing namespace metal;\ntypedef " + memoryPrecision.name() + " real;\n";
  shader += silu ? "constant bool silu = true;\n" : "constant bool silu = false;\n";
  shader += R"(
// The context of batch b is its kernel_size - 1 previous rows followed by its T rows of x.
inline float context_value(device const real* x, device const real* previous, uint b, uint r, uint c, uint T, uint C, uint history) {
  return r < history ? float(previous[(b * history + r) * C + c]) : float(x[(b * T + r - history) * C + c]);
}

kernel void causal_convolution(
  device const real* x [[buffer(0)]],
  device const real* previous [[buffer(1)]],
  device const real* weight [[buffer(2)]],
  device real* y [[buffer(3)]],
  device real* next [[buffer(4)]],
  constant uint* sizes [[buffer(5)]],
  uint3 tpig [[thread_position_in_grid]])
{
  const uint T = sizes[1];
  const uint C = sizes[2];
  const uint K = sizes[3];
  const uint R = sizes[4];
  const uint c = tpig.x;
  const uint row = tpig.y;
  const uint b = tpig.z;
  if (c >= C)
    return;
  const uint history = K - 1;
  if (row < T) {
    float sum = 0;
    for (uint j = 0; j < K; j++)
      sum += float(weight[c * K + j]) * context_value(x, previous, b, row + j, c, T, C, history);
    if (silu)
      sum = sum / (1.0f + exp(-sum));
    y[(b * T + row) * C + c] = real(sum);
  } else {
    // The next rows are the context's last R rows.
    const uint j = row - T;
    next[(b * R + j) * C + c] = real(context_value(x, previous, b, history + T - R + j, c, T, C, history));
  }
}
)";
  return shader;
}
