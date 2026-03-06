#include "AddKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "CodeWriter.hpp"

#include <algorithm>

AddKernel::AddKernel(AddKernelDescriptor descriptor, MTL::Device *const device) {

  args = descriptor.args;

  value = descriptor.value;

  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();

  threadgroupSize = MTL::Size(256, 1, 1);

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short AddKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string AddKernel::createSource() const noexcept {
  CodeWriter source;
  source += createConstants() + "\n";
  std::string buffers = "";
  if (value == 0 || value == 1) {
    for (int i = 1; i < args; i++) {
      buffers += "device real4 *src" + std::to_string(i) + " [[buffer(" + std::to_string(i) + ")]],\n";
    }
  } else {
    for (int i = 1; i < args; i++) {
      buffers += "device real *src" + std::to_string(i) + " [[buffer(" + std::to_string(i) + ")]],\n";
    }
  }
  source.SetValue("OTHER_SOURCE_BUFFERS", buffers);
  source.SetValue("DESTINATION_INDEX", std::to_string(args));
  std::string items = " + src1[idx]";
  for (int i = 2; i < args; i++) {
    items += " + src" + std::to_string(i) + "[idx]";
  }
  source.SetValue("OTHER_SOURCE_ITEMS", items);
  if (value == 0) {
    source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device real4 *src0 [[buffer(0)]],
  {{OTHER_SOURCE_BUFFERS}}
  device real4 *destination [[buffer({{DESTINATION_INDEX}})]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  destination[idx] = src0[idx]{{OTHER_SOURCE_ITEMS}};
}
  )";
  } else if (value == 1) {
    source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device real4 *src0 [[buffer(0)]],
  {{OTHER_SOURCE_BUFFERS}}
  device real4 *destination [[buffer({{DESTINATION_INDEX}})]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = src0[idx]{{OTHER_SOURCE_ITEMS}};
}
  )";
  } else {
    source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device real *src0 [[buffer(0)]],
  {{OTHER_SOURCE_BUFFERS}}
  device real *destination [[buffer({{DESTINATION_INDEX}})]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = src0[idx]{{OTHER_SOURCE_ITEMS}};
}
  )";
  }
  return source.ToString();
}

std::string AddKernel::createConstants() const noexcept {

  std::string defines = "";
  if (value == 0 || value == 1) {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float4 real4;");
      defines += "\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat4 real4;");
      defines += "\n";
    } else {
      defines += std::string("typedef half4 real4;");
      defines += "\n";
    }
  } else {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float real;");
      defines += "\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat real;");
      defines += "\n";
    } else {
      defines += std::string("typedef half real;");
      defines += "\n";
    }
  }
  if (value != 0) {
    defines += "constant uint count [[function_constant(0)]];";
    defines += "\n";
  }
  return defines;
}
