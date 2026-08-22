#include "AddKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "CodeWriter.hpp"

#include <algorithm>

AddKernel::AddKernel(AddKernelDescriptor descriptor, MTL::Device *const device) {

  args = descriptor.args;

  value = descriptor.value;

  loadM = descriptor.loadM;

  negative_mask = descriptor.negative_mask;

  broadcast = descriptor.broadcast;

  scaled_mask = descriptor.scaled_mask;

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
  const bool vectorized = value == 0 || value == 1;
  for (int i = 0; i < args; i++) {
    const bool scalar = i < 8 && (broadcast & (1u << i));
    buffers += scalar || !vectorized ? "device const real *src" : "device const real4 *src";
    buffers += std::to_string(i) + " [[buffer(" + std::to_string(i) + ")]],\n";
  }
  source.SetValue("SOURCE_BUFFERS", buffers);
  source.SetValue("DESTINATION_INDEX", std::to_string(args));
  std::string items;
  for (int i = 0; i < args; i++) {
    const bool scalar = i < 8 && (broadcast & (1u << i));
    const bool subtract = i < 8 && (negative_mask & (1u << i));
    std::string item = "src" + std::to_string(i) + (scalar ? "[0]" : "[idx]");
    if (i < 8 && (scaled_mask & (1u << i)))
      item = "(" + (vectorized ? std::string("real4") : std::string("real")) + "(scales[" + std::to_string(i) + "]) * " + item + ")";
    if (i == 0)
      items += subtract ? "-" + item : item;
    else
      items += subtract ? " - " + item : " + " + item;
  }
  source.SetValue("SOURCE_ITEMS", items);
  source.SetValue("SCALES", scaled_mask ? "  constant real *scales [[buffer(" + std::to_string(args + 1) + ")]],\n" : "");
  if (value == 0) {
    source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  {{SOURCE_BUFFERS}}
  device real4 *destination [[buffer({{DESTINATION_INDEX}})]],

{{SCALES}}

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  destination[idx] = {{SOURCE_ITEMS}};
}
  )";
  } else if (value == 1) {
    source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  {{SOURCE_BUFFERS}}
  device real4 *destination [[buffer({{DESTINATION_INDEX}})]],

{{SCALES}}

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = {{SOURCE_ITEMS}};
}
  )";
  } else {
    source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  {{SOURCE_BUFFERS}}
  device real *destination [[buffer({{DESTINATION_INDEX}})]],

{{SCALES}}

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = {{SOURCE_ITEMS}};
}
  )";
  }
  std::string shader = source.ToString();
  if (loadM) {
    const std::string::size_type argumentPosition = shader.find("  uint3 tpig [[thread_position_in_grid]]");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint *loadM [[buffer(" + std::to_string(args + 1 + (scaled_mask ? 1 : 0)) + ")]],\n");
    const std::string::size_type countPosition = shader.find("  const uint idx = tpig.x;");
    CCV_NNC_MFA_PRECONDITION(countPosition != std::string::npos);
    shader.insert(countPosition, "  const uniform<uint> count = make_uniform(loadM[0]);\n");
  }
  return shader;
}

std::string AddKernel::createConstants() const noexcept {

  std::string defines = "";
  if (value == 0 || value == 1) {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float real;");
      defines += "\n";
      defines += std::string("typedef float4 real4;");
      defines += "\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat real;");
      defines += "\n";
      defines += std::string("typedef bfloat4 real4;");
      defines += "\n";
    } else {
      defines += std::string("typedef half real;");
      defines += "\n";
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
  if (value != 0 && !loadM) {
    defines += "constant uint count [[function_constant(0)]];";
    defines += "\n";
  }
  return defines;
}
