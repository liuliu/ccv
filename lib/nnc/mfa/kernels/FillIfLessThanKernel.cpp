#include "FillIfLessThanKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

FillIfLessThanKernel::FillIfLessThanKernel(FillIfLessThanKernelDescriptor descriptor, MTL::Device* const device) {
  value = descriptor.value;
  loadM = descriptor.loadM;
  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();
  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = MTL::Size(256, 1, 1);

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

unsigned short FillIfLessThanKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string FillIfLessThanKernel::createSource() const noexcept {
  const std::string loadMArgument = loadM ? "  const device uint *loadM [[buffer(5)]],\n" : "";
  const std::string loadMValue = loadM ? "  const uniform<uint> count = make_uniform(loadM[0]);\n" : "";
  std::string shader = createConstants() + R"(
#include <metal_stdlib>
using namespace metal;
)";
  if (value == 0 || value == 1) {
    shader += R"(
kernel void fill_if_less_than(
  device const real4 *src [[buffer(0)]],
  device const real4 *selector [[buffer(1)]],
  device const real *threshold [[buffer(2)]],
  device real4 *destination [[buffer(3)]],
  constant float& fill [[buffer(4)]],

)" + loadMArgument + R"(  uint3 tpig [[thread_position_in_grid]]
) {
)" + loadMValue + R"(  const uint idx = tpig.x;
)";
    if (value == 1) {
      shader += R"(  if (idx >= count)
    return;
)";
    }
    shader += R"(  const float threshold_value = (float)threshold[0];
  const float4 source_value = (float4)src[idx];
  const bool4 predicate = (float4)selector[idx] < threshold_value;
  destination[idx] = (real4)select(source_value, float4(fill), predicate);
}
)";
  } else {
    shader += R"(
kernel void fill_if_less_than(
  device const real *src [[buffer(0)]],
  device const real *selector [[buffer(1)]],
  device const real *threshold [[buffer(2)]],
  device real *destination [[buffer(3)]],
  constant float& fill [[buffer(4)]],

)" + loadMArgument + R"(  uint3 tpig [[thread_position_in_grid]]
) {
)" + loadMValue + R"(  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float source_value = (float)src[idx];
  destination[idx] = (real)(((float)selector[idx] < (float)threshold[0]) ? fill : source_value);
}
)";
  }
  return shader;
}

std::string FillIfLessThanKernel::createConstants() const noexcept {
  std::string defines;
  if (value == 0 || value == 1) {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += "typedef float real;\ntypedef float4 real4;\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += "typedef bfloat real;\ntypedef bfloat4 real4;\n";
    } else {
      defines += "typedef half real;\ntypedef half4 real4;\n";
    }
  } else {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += "typedef float real;\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += "typedef bfloat real;\n";
    } else {
      defines += "typedef half real;\n";
    }
  }
  if (value != 0 && !loadM) {
    defines += "constant uint count [[function_constant(0)]];\n";
  }
  return defines;
}
