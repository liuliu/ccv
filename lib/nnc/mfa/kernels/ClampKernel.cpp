#include "ClampKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ClampKernel::ClampKernel(ClampKernelDescriptor descriptor, MTL::Device* const device) {
  value = descriptor.value;
  bounds = descriptor.bounds;
  loadM = descriptor.loadM;
  memoryPrecision = descriptor.memoryPrecision;
  source = createSource();
  threadgroupSize = MTL::Size(256, 1, 1);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string ClampKernel::createSource() const noexcept {
  const bool vectorized = value == 0 || value == 1;
  const std::string realType = vectorized ? "real4" : "real";
  const std::string floatType = vectorized ? "float4" : "float";
  std::string operation = "  " + floatType + " x = (" + floatType + ")src[idx];\n";
  if (bounds & 1)
    operation += "  x = max(x, " + floatType + "(min_value));\n";
  if (bounds & 2)
    operation += "  x = min(x, " + floatType + "(max_value));\n";
  operation += "  destination[idx] = (" + realType + ")x;\n";

  std::string shader = createConstants() + R"(
#include <metal_stdlib>
using namespace metal;
kernel void clamp_forward(
  device const )" + realType + R"( *src [[buffer(0)]],
  device )" + realType + R"( *destination [[buffer(1)]],
  constant float& min_value [[buffer(2)]],
  constant float& max_value [[buffer(3)]],
  uint3 tpig [[thread_position_in_grid]]) {
  const uint idx = tpig.x;
)";
  if (value != 0)
    shader += "  if (idx >= count)\n    return;\n";
  shader += operation + "}\n";
  if (loadM) {
    const std::string::size_type argumentPosition = shader.find("  uint3 tpig [[thread_position_in_grid]]");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint *loadM [[buffer(4)]],\n");
    const std::string::size_type countPosition = shader.find("  const uint idx = tpig.x;");
    CCV_NNC_MFA_PRECONDITION(countPosition != std::string::npos);
    shader.insert(countPosition, "  const uniform<uint> count = make_uniform(loadM[0]);\n");
  }
  return shader;
}

std::string ClampKernel::createConstants() const noexcept {
  std::string defines;
  if (value == 0 || value == 1)
    defines = memoryPrecision == GEMMOperandPrecision::FP32 ? "typedef float4 real4;\n" : "typedef half4 real4;\n";
  else
    defines = memoryPrecision == GEMMOperandPrecision::FP32 ? "typedef float real;\n" : "typedef half real;\n";
  if (value != 0 && !loadM)
    defines += "constant uint count [[function_constant(0)]];\n";
  return defines;
}
