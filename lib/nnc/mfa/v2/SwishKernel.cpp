#include "SwishKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

SwishKernel::SwishKernel(SwishKernelDescriptor descriptor, MTL::Device *const device) {

  gradient = descriptor.gradient;

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

unsigned short SwishKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string SwishKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (gradient) {
    if (value == 0) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void swish(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  const float4 y = 1. / (1. + exp(-x));
  const float4 y_sq = y * y;
  destination[idx] = (real4)((float4)g[idx] * (x * (y - y_sq) + y));
}
    )";
    } else if (value == 1) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void swish(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  const float4 y = 1. / (1. + exp(-x));
  const float4 y_sq = y * y;
  destination[idx] = (real4)((float4)g[idx] * (x * (y - y_sq) + y));
}
      )";
    } else {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void swish(
  device real *g [[buffer(0)]],
  device real *src [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  const float y = 1. / (1. + exp(-x));
  const float y_sq = y * y;
  destination[idx] = (real)((float)g[idx] * (x * (y - y_sq) + y));
}
    )";
    }
  } else {
    if (value == 0) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void swish(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(x / (1 + exp(-x)));
}
    )";
    } else if (value == 1) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void swish(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(x / (1 + exp(-x)));
}
    )";
    } else {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void swish(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(x / (1 + exp(-x)));
}
    )";
    }
  }
  return shader;
}

std::string SwishKernel::createConstants() const noexcept {

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
