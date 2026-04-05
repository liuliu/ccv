#include "SwishKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

SwishKernel::SwishKernel(SwishKernelDescriptor descriptor, MTL::Device *const device) {

  gradient = descriptor.gradient;

  value = descriptor.value;

  beta = descriptor.beta;

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
  const bool beta_is_one = (beta == 1);
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

inline float stable_swish_gradient(float z)
{
  const float t = exp(-abs(z));
  const float tail = t / (1.0f + t); // min(sigmoid, 1 - sigmoid)
  const float sigmoid = tail + step(0.0f, z) * (1.0f - 2.0f * tail);
  const float slope = tail * (1.0f - tail);
  const float correction = isfinite(z) ? (z * slope) : 0.0f;
  return sigmoid + correction;
}

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
  if (gradient) {
    if (value == 0) {
      if (beta_is_one) {
        shader += R"(
kernel void swish(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)((float4)g[idx] * stable_swish_gradient(x));
}
    )";
      } else {
        shader += R"(
kernel void swish(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  const float4 z = beta * x;
  destination[idx] = (real4)((float4)g[idx] * stable_swish_gradient(z));
}
    )";
      }
    } else if (value == 1) {
      if (beta_is_one) {
        shader += R"(
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
  destination[idx] = (real4)((float4)g[idx] * stable_swish_gradient(x));
}
      )";
      } else {
        shader += R"(
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
  const float4 z = beta * x;
  destination[idx] = (real4)((float4)g[idx] * stable_swish_gradient(z));
}
      )";
      }
    } else {
      if (beta_is_one) {
        shader += R"(
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
  destination[idx] = (real)((float)g[idx] * stable_swish_gradient(x));
}
    )";
      } else {
        shader += R"(
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
  const float z = beta * x;
  destination[idx] = (real)((float)g[idx] * stable_swish_gradient(z));
}
    )";
      }
    }
  } else {
    if (value == 0) {
      if (beta_is_one) {
        shader += R"(
kernel void swish(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(x * stable_sigmoid(x));
}
    )";
      } else {
        shader += R"(
kernel void swish(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(x * stable_sigmoid(beta * x));
}
    )";
      }
    } else if (value == 1) {
      if (beta_is_one) {
        shader += R"(
kernel void swish(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(x * stable_sigmoid(x));
}
    )";
      } else {
        shader += R"(
kernel void swish(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(x * stable_sigmoid(beta * x));
}
    )";
      }
    } else {
      if (beta_is_one) {
        shader += R"(
kernel void swish(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(x * stable_sigmoid(x));
}
    )";
      } else {
        shader += R"(
kernel void swish(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(x * stable_sigmoid(beta * x));
}
    )";
      }
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
  if (beta != 1) {
    defines += "constant float beta [[function_constant(1)]];";
    defines += "\n";
  }
  return defines;
}
