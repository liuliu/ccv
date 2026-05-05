#include "SwishMulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

SwishMulKernel::SwishMulKernel(SwishMulKernelDescriptor descriptor, MTL::Device* const device) {

  value = descriptor.value;

  beta = descriptor.beta;

  scale = descriptor.scale;

  aPrecision = descriptor.aPrecision;

  bPrecision = descriptor.bPrecision;

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

unsigned short SwishMulKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string SwishMulKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  const bool beta_is_one = (beta == 1);
  const bool scale_is_one = (scale == 1);
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
  if (value == 0 || value == 1) {
    shader += R"(
kernel void swish_mul(
  device realA4 *a [[buffer(0)]],
  device realB4 *b [[buffer(1)]],
  device realA4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
)";
    if (value == 1)
      shader += "  if (idx >= count)\n    return;\n";
    shader += R"(
  const float4 av = (float4)(a[idx]);
  const float4 bv = (float4)(b[idx]);
)";
    if (beta_is_one)
      shader += "  float4 result = av * bv * stable_sigmoid(bv);\n";
    else
      shader += "  float4 result = av * bv * stable_sigmoid(beta * bv);\n";
    if (!scale_is_one)
      shader += "  result *= scale;\n";
    shader += R"(
  destination[idx] = (realA4)result;
}
    )";
  } else {
    shader += R"(
kernel void swish_mul(
  device realA *a [[buffer(0)]],
  device realB *b [[buffer(1)]],
  device realA *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float av = (float)(a[idx]);
  const float bv = (float)(b[idx]);
)";
    if (beta_is_one)
      shader += "  float result = av * bv * stable_sigmoid(bv);\n";
    else
      shader += "  float result = av * bv * stable_sigmoid(beta * bv);\n";
    if (!scale_is_one)
      shader += "  result *= scale;\n";
    shader += R"(
  destination[idx] = (realA)result;
}
    )";
  }
  return shader;
}

std::string SwishMulKernel::createConstants() const noexcept {

  std::string defines = "";
  if (value == 0 || value == 1) {
    if (aPrecision == GEMMOperandPrecision::FP32) {
      defines += "typedef float4 realA4;";
    } else if (aPrecision == GEMMOperandPrecision::BF16) {
      defines += "typedef bfloat4 realA4;";
    } else {
      defines += "typedef half4 realA4;";
    }
    defines += "\n";
    if (bPrecision == GEMMOperandPrecision::FP32) {
      defines += "typedef float4 realB4;";
    } else if (bPrecision == GEMMOperandPrecision::BF16) {
      defines += "typedef bfloat4 realB4;";
    } else {
      defines += "typedef half4 realB4;";
    }
    defines += "\n";
  } else {
    if (aPrecision == GEMMOperandPrecision::FP32) {
      defines += "typedef float realA;";
    } else if (aPrecision == GEMMOperandPrecision::BF16) {
      defines += "typedef bfloat realA;";
    } else {
      defines += "typedef half realA;";
    }
    defines += "\n";
    if (bPrecision == GEMMOperandPrecision::FP32) {
      defines += "typedef float realB;";
    } else if (bPrecision == GEMMOperandPrecision::BF16) {
      defines += "typedef bfloat realB;";
    } else {
      defines += "typedef half realB;";
    }
    defines += "\n";
  }
  if (value != 0) {
    defines += "constant uint count [[function_constant(0)]];";
    defines += "\n";
  }
  if (beta != 1) {
    defines += "constant float beta [[function_constant(1)]];";
    defines += "\n";
  }
  if (scale != 1) {
    defines += "constant float scale [[function_constant(2)]];";
    defines += "\n";
  }
  return defines;
}
