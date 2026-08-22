#include "LogKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

LogKernel::LogKernel(LogKernelDescriptor descriptor, MTL::Device* const device) {
  value = descriptor.value;
  loadM = descriptor.loadM;
  memoryPrecision = descriptor.memoryPrecision;
  source = createSource();
  threadgroupSize = MTL::Size(256, 1, 1);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string LogKernel::createSource() const noexcept {
  std::string shader = createConstants() + R"(
#include <metal_stdlib>
using namespace metal;
)";
  if (value == 0) {
    shader += R"(
kernel void log_forward(
  device const real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],
  uint3 tpig [[thread_position_in_grid]]) {
  const uint idx = tpig.x;
  destination[idx] = (real4)(precise::log((float4)src[idx]));
}
)";
  } else if (value == 1) {
    shader += R"(
kernel void log_forward(
  device const real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],
  uint3 tpig [[thread_position_in_grid]]) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = (real4)(precise::log((float4)src[idx]));
}
)";
  } else {
    shader += R"(
kernel void log_forward(
  device const real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],
  uint3 tpig [[thread_position_in_grid]]) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = (real)(precise::log((float)src[idx]));
}
)";
  }
  if (loadM) {
    const std::string::size_type argumentPosition = shader.find("  uint3 tpig [[thread_position_in_grid]]");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint *loadM [[buffer(2)]],\n");
    const std::string::size_type countPosition = shader.find("  const uint idx = tpig.x;");
    CCV_NNC_MFA_PRECONDITION(countPosition != std::string::npos);
    shader.insert(countPosition, "  const uniform<uint> count = make_uniform(loadM[0]);\n");
  }
  return shader;
}

std::string LogKernel::createConstants() const noexcept {
  std::string defines;
  if (value == 0 || value == 1)
    defines = memoryPrecision == GEMMOperandPrecision::FP32 ? "typedef float4 real4;\n" : "typedef half4 real4;\n";
  else
    defines = memoryPrecision == GEMMOperandPrecision::FP32 ? "typedef float real;\n" : "typedef half real;\n";
  if (value != 0 && !loadM)
    defines += "constant uint count [[function_constant(0)]];\n";
  return defines;
}
