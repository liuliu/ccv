#include "CastKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

CastKernel::CastKernel(CastKernelDescriptor descriptor, MTL::Device *const device) {

  value = descriptor.value;

  fromMemoryPrecision = descriptor.fromMemoryPrecision;

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

unsigned short CastKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string CastKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (value == 0) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast(
  device original_real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  destination[idx] = (real4)(src[idx]);
}
  )";
  } else if (value == 1) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast(
  device original_real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = (real4)(src[idx]);
}
  )";
  } else {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast(
  device original_real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = (real)(src[idx]);
}
  )";
  }
  return shader;
}

std::string CastKernel::createConstants() const noexcept {

  std::string defines = "";
  if (value == 0 || value == 1) {
    if (fromMemoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float4 original_real4;");
      defines += "\n";
    } else if (fromMemoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat4 original_real4;");
      defines += "\n";
    } else {
      defines += std::string("typedef half4 original_real4;");
      defines += "\n";
    }
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
    if (fromMemoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float original_real;");
      defines += "\n";
    } else if (fromMemoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat original_real;");
      defines += "\n";
    } else {
      defines += std::string("typedef half original_real;");
      defines += "\n";
    }
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
