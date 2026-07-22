#include "SigmoidKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

SigmoidKernel::SigmoidKernel(SigmoidKernelDescriptor descriptor, MTL::Device* const device) {
  gradient = descriptor.gradient;
  value = descriptor.value;

  loadM = descriptor.loadM;
  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = MTL::Size(256, 1, 1);

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short SigmoidKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string SigmoidKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float stable_sigmoid(float z)
{
  const float tail = 1.0f / (1.0f + exp(abs(z)));
  const float positive = step(0.0f, z);
  return tail + positive * (1.0f - 2.0f * tail);
}

inline float4 stable_sigmoid(float4 z)
{
  const float4 tail = 1.0f / (1.0f + exp(abs(z)));
  const float4 positive = step(0.0f, z);
  return tail + positive * (1.0f - 2.0f * tail);
}
  )";
  if (gradient) {
    if (value == 0) {
      shader += R"(
kernel void sigmoid(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)((float4)g[idx] * x * (1.0f - x));
}
      )";
    } else if (value == 1) {
      shader += R"(
kernel void sigmoid(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)((float4)g[idx] * x * (1.0f - x));
}
      )";
    } else {
      shader += R"(
kernel void sigmoid(
  device real *g [[buffer(0)]],
  device real *src [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)((float)g[idx] * x * (1.0f - x));
}
      )";
    }
  } else {
    if (value == 0) {
      shader += R"(
kernel void sigmoid(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(stable_sigmoid(x));
}
      )";
    } else if (value == 1) {
      shader += R"(
kernel void sigmoid(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(stable_sigmoid(x));
}
      )";
    } else {
      shader += R"(
kernel void sigmoid(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(stable_sigmoid(x));
}
      )";
    }
  }
  if (loadM) {
    const std::string::size_type argumentPosition = shader.find("  uint3 tpig [[thread_position_in_grid]]");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint *loadM [[buffer(" + std::to_string(gradient ? 3 : 2) + ")]],\n");
    const std::string::size_type countPosition = shader.find("  const uint idx = tpig.x;");
    CCV_NNC_MFA_PRECONDITION(countPosition != std::string::npos);
    shader.insert(countPosition, "  const uniform<uint> count = make_uniform(loadM[0]);\n");
  }
  return shader;
}

std::string SigmoidKernel::createConstants() const noexcept {
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
  if (value != 0 && !loadM) {
    defines += "constant uint count [[function_constant(0)]];";
    defines += "\n";
  }
  return defines;
}
