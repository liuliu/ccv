#include "SoftplusKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

SoftplusKernel::SoftplusKernel(SoftplusKernelDescriptor descriptor, MTL::Device* const device) {
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

unsigned short SoftplusKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string SoftplusKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;

inline float stable_log1p(float x)
{
  const float xp1 = 1.0f + x;
  if (xp1 == 1.0f)
    return x;
  return x * (log(xp1) / (xp1 - 1.0f));
}

inline float4 stable_log1p(float4 x)
{
  const float4 xp1 = 1.0f + x;
  const float4 y = x * (log(xp1) / (xp1 - 1.0f));
  return select(y, x, xp1 == 1.0f);
}

inline float stable_softplus(float x)
{
  const float positive = (x > 0.0f) ? x : 0.0f;
  return positive + stable_log1p(precise::exp(-abs(x)));
}

inline float4 stable_softplus(float4 x)
{
  const float4 positive = select((float4)(0.0f), x, x > 0.0f);
  return positive + stable_log1p(precise::exp(-abs(x)));
}
  )";
  if (value == 0) {
    shader += R"(
kernel void softplus_forward(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(stable_softplus(x));
}
      )";
  } else if (value == 1) {
    shader += R"(
kernel void softplus_forward(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(stable_softplus(x));
}
      )";
  } else {
    shader += R"(
kernel void softplus_forward(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(stable_softplus(x));
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

std::string SoftplusKernel::createConstants() const noexcept {
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
