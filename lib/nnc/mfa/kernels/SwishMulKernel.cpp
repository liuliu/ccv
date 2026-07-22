#include "SwishMulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

SwishMulKernel::SwishMulKernel(SwishMulKernelDescriptor descriptor, MTL::Device* const device) {

  gradient = descriptor.gradient;

  outputMask = descriptor.outputMask;

  value = descriptor.value;

  loadM = descriptor.loadM;

  beta = descriptor.beta;

  scale = descriptor.scale;

  gPrecision = descriptor.gPrecision;

  aPrecision = descriptor.aPrecision;

  bPrecision = descriptor.bPrecision;

  daPrecision = descriptor.daPrecision;

  dbPrecision = descriptor.dbPrecision;

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
  const bool value_gradient = gradient && ((outputMask & 1) != 0);
  const bool gate_gradient = gradient && ((outputMask & 2) != 0);
  const bool vectorized = (value == 0 || value == 1);
  shader += R"(
#include <metal_stdlib>
using namespace metal;
  )";
  if (!gradient || value_gradient) {
    if (vectorized)
      shader += R"(
inline float4 stable_sigmoid(float4 z)
{
  const float4 tail = 1.0f / (1.0f + exp(abs(z)));
  const float4 positive = step(0.0f, z);
  return tail + positive * (1.0f - 2.0f * tail);
}
      )";
    else
      shader += R"(
inline float stable_sigmoid(float z)
{
  const float tail = 1.0f / (1.0f + exp(abs(z)));
  const float positive = step(0.0f, z);
  return tail + positive * (1.0f - 2.0f * tail);
}
      )";
  }
  if (gate_gradient) {
    if (vectorized)
      shader += R"(
inline float4 stable_swish_gradient(float4 z)
{
  const float4 t = exp(-abs(z));
  const float4 tail = t / (1.0f + t); // min(sigmoid, 1 - sigmoid)
  const float4 sigmoid = tail + step(0.0f, z) * (1.0f - 2.0f * tail);
  const float4 slope = tail * ((float4)(1.0f) - tail);
  const float4 correction = select((float4)(0.0f), z * slope, isfinite(z));
  return sigmoid + correction;
}
      )";
    else
      shader += R"(
inline float stable_swish_gradient(float z)
{
  const float t = exp(-abs(z));
  const float tail = t / (1.0f + t); // min(sigmoid, 1 - sigmoid)
  const float sigmoid = tail + step(0.0f, z) * (1.0f - 2.0f * tail);
  const float slope = tail * (1.0f - tail);
  const float correction = isfinite(z) ? (z * slope) : 0.0f;
  return sigmoid + correction;
}
      )";
  }
  if (gradient) {
    shader += "kernel void swish_mul(\n";
    if (vectorized)
      shader += "  device realG4 *g [[buffer(0)]],\n";
    else
      shader += "  device realG *g [[buffer(0)]],\n";
    if (value_gradient && gate_gradient)
    {
      if (vectorized)
        shader += "  device realA4 *a [[buffer(1)]],\n  device realB4 *b [[buffer(2)]],\n  device realDA4 *dvalue [[buffer(3)]],\n  device realDB4 *dgate [[buffer(4)]],\n";
      else
        shader += "  device realA *a [[buffer(1)]],\n  device realB *b [[buffer(2)]],\n  device realDA *dvalue [[buffer(3)]],\n  device realDB *dgate [[buffer(4)]],\n";
    } else if (value_gradient) {
      if (vectorized)
        shader += "  device realB4 *b [[buffer(1)]],\n  device realDA4 *dvalue [[buffer(2)]],\n";
      else
        shader += "  device realB *b [[buffer(1)]],\n  device realDA *dvalue [[buffer(2)]],\n";
    } else {
      if (vectorized)
        shader += "  device realA4 *a [[buffer(1)]],\n  device realB4 *b [[buffer(2)]],\n  device realDB4 *dgate [[buffer(3)]],\n";
      else
        shader += "  device realA *a [[buffer(1)]],\n  device realB *b [[buffer(2)]],\n  device realDB *dgate [[buffer(3)]],\n";
    }
    shader += R"(
  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
)";
    if (value == 1 || value == 2)
      shader += "  if (idx >= count)\n    return;\n";
    if (vectorized)
    {
      shader += R"(
  const float4 gv = (float4)(g[idx]);
  const float4 bv = (float4)(b[idx]);
)";
      if (beta_is_one)
        shader += "  const float4 z = bv;\n";
      else
        shader += "  const float4 z = beta * bv;\n";
      shader += "  float4 scaled_g = gv;\n";
      if (!scale_is_one)
        shader += "  scaled_g *= scale;\n";
      if (value_gradient)
        shader += "  dvalue[idx] = (realDA4)(scaled_g * bv * stable_sigmoid(z));\n";
      if (gate_gradient)
        shader += "  const float4 av = (float4)(a[idx]);\n  dgate[idx] = (realDB4)(scaled_g * av * stable_swish_gradient(z));\n";
    } else {
      shader += R"(
  const float gv = (float)(g[idx]);
  const float bv = (float)(b[idx]);
)";
      if (beta_is_one)
        shader += "  const float z = bv;\n";
      else
        shader += "  const float z = beta * bv;\n";
      shader += "  float scaled_g = gv;\n";
      if (!scale_is_one)
        shader += "  scaled_g *= scale;\n";
      if (value_gradient)
        shader += "  dvalue[idx] = (realDA)(scaled_g * bv * stable_sigmoid(z));\n";
      if (gate_gradient)
        shader += "  const float av = (float)(a[idx]);\n  dgate[idx] = (realDB)(scaled_g * av * stable_swish_gradient(z));\n";
    }
    shader += R"(
}
    )";
  } else if (value == 0 || value == 1) {
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
  if (loadM) {
    const uint8_t countBufferIndex = gradient ? 2 + ((outputMask & 1) ? 1 : 0) + ((outputMask & 2) ? 2 : 0) : 3;
    const std::string::size_type argumentPosition = shader.find("  uint3 tpig [[thread_position_in_grid]]");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint *loadM [[buffer(" + std::to_string(countBufferIndex) + ")]],\n");
    const std::string::size_type countPosition = shader.find("  const uint idx = tpig.x;");
    CCV_NNC_MFA_PRECONDITION(countPosition != std::string::npos);
    shader.insert(countPosition, "  const uniform<uint> count = make_uniform(loadM[0]);\n");
  }
  return shader;
}

std::string SwishMulKernel::createConstants() const noexcept {

  std::string defines = "";
  auto define_type = [&](const char* name, const GEMMOperandPrecision precision, const bool vectorized) {
    defines += "typedef ";
    if (precision == GEMMOperandPrecision::FP32) {
      defines += vectorized ? "float4 " : "float ";
    } else if (precision == GEMMOperandPrecision::BF16) {
      defines += vectorized ? "bfloat4 " : "bfloat ";
    } else {
      defines += vectorized ? "half4 " : "half ";
    }
    defines += name;
    defines += ";";
    defines += "\n";
  };
  const bool vectorized = (value == 0 || value == 1);
  const bool value_gradient = gradient && ((outputMask & 1) != 0);
  const bool gate_gradient = gradient && ((outputMask & 2) != 0);
  if (gradient) {
    if (vectorized) {
      define_type("realG4", gPrecision, true);
      if (gate_gradient)
        define_type("realA4", aPrecision, true);
      define_type("realB4", bPrecision, true);
      if (value_gradient)
        define_type("realDA4", daPrecision, true);
      if (gate_gradient)
        define_type("realDB4", dbPrecision, true);
    } else {
      define_type("realG", gPrecision, false);
      if (gate_gradient)
        define_type("realA", aPrecision, false);
      define_type("realB", bPrecision, false);
      if (value_gradient)
        define_type("realDA", daPrecision, false);
      if (gate_gradient)
        define_type("realDB", dbPrecision, false);
    }
  } else if (vectorized) {
    define_type("realA4", aPrecision, true);
    define_type("realB4", bPrecision, true);
  } else {
    define_type("realA", aPrecision, false);
    define_type("realB", bPrecision, false);
  }
  if (value != 0 && !loadM) {
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
