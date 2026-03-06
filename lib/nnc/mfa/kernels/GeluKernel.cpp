#include "GeluKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

GeluKernel::GeluKernel(GeluKernelDescriptor descriptor, MTL::Device *const device) {

  gradient = descriptor.gradient;

  tanh = descriptor.tanh;

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

unsigned short GeluKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string GeluKernel::createErf() const noexcept {
  return R"(
float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = metal::fma(r, t, -t);
    // TODO, replace with expm1 when implemented
    r = 1.0f - metal::exp(r);
    r = metal::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = metal::fma(r, a, a);
  }
  return r;
}
  )";
}

std::string GeluKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (gradient) {
    if (tanh) {
      if (value == 0) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  const float4 x_sq = x * x;
  const float4 x_cube = x_sq * x;
  const float4 inner = 0.797884560802865355 * (x + 0.044715 * x_cube);
  const float4 tanh_inner = precise::tanh(inner);
  const float4 left = 0.5 * x;
  const float4 right = 1 + tanh_inner;
  const float4 left_derivative = 0.5 * right;
  const float4 tanh_derivative = 1 - tanh_inner * tanh_inner;
  const float4 inner_derivative = 0.797884560802865355 * (1 + 3 * 0.044715 * x_sq);
  const float4 right_derivative = left * tanh_derivative * inner_derivative;
  destination[idx] = (real4)((float4)g[idx] * (left_derivative + right_derivative));
}
      )";
      } else if (value == 1) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real4 *g [[buffer(0)]],
  device real4 *src [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  const float4 x_sq = x * x;
  const float4 x_cube = x_sq * x;
  const float4 inner = 0.797884560802865355 * (x + 0.044715 * x_cube);
  const float4 tanh_inner = precise::tanh(inner);
  const float4 left = 0.5 * x;
  const float4 right = 1 + tanh_inner;
  const float4 left_derivative = 0.5 * right;
  const float4 tanh_derivative = 1 - tanh_inner * tanh_inner;
  const float4 inner_derivative = 0.797884560802865355 * (1 + 3 * 0.044715 * x_sq);
  const float4 right_derivative = left * tanh_derivative * inner_derivative;
  destination[idx] = (real4)((float4)g[idx] * (left_derivative + right_derivative));
}
      )";
      } else {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real *g [[buffer(0)]],
  device real *src [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float4)(src[idx]);
  const float x_sq = x * x;
  const float x_cube = x_sq * x;
  const float inner = 0.797884560802865355 * (x + 0.044715 * x_cube);
  const float tanh_inner = precise::tanh(inner);
  const float left = 0.5 * x;
  const float right = 1 + tanh_inner;
  const float left_derivative = 0.5 * right;
  const float tanh_derivative = 1 - tanh_inner * tanh_inner;
  const float inner_derivative = 0.797884560802865355 * (1 + 3 * 0.044715 * x_sq);
  const float right_derivative = left * tanh_derivative * inner_derivative;
  destination[idx] = (real)((float)g[idx] * (left_derivative + right_derivative));
}
      )";
      }
    } else {
      shader += createErf() + "\n";
      if (value == 0) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real *g [[buffer(0)]],
  device real *src [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float x = (float)(src[idx]);
  const float cdf = 0.5 * (1. + erf(x * 0.70710678118654752440));
  const float pdf = exp(-0.5 * x * x) * 0.797884560802865355;
  destination[idx] = (real)(g[idx] * (cdf + x * pdf));
}
      )";
      } else {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real *g [[buffer(0)]],
  device real *src [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  const float cdf = 0.5 * (1. + erf(x * 0.70710678118654752440));
  const float pdf = exp(-0.5 * x * x) * 0.797884560802865355;
  destination[idx] = (real)(g[idx] * (cdf + x * pdf));
}
      )";
      }
    }
  } else {
    if (tanh) {
      if (value == 0) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(0.5 * x * (1 + precise::tanh(0.797884560802865355 * (x + 0.044715 * x * x * x))));
}
      )";
      } else if (value == 1) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float4 x = (float4)(src[idx]);
  destination[idx] = (real4)(0.5 * x * (1 + precise::tanh(0.797884560802865355 * (x + 0.044715 * x * x * x))));
}
      )";
      } else {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(0.5 * x * (1 + precise::tanh(0.797884560802865355 * (x + 0.044715 * x * x * x))));
}
      )";
      }
    } else {
      shader += createErf() + "\n";
      if (value == 0) {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(x * 0.5 * (1. + erf(x * 0.70710678118654752440)));
}
      )";
      } else {
        shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const float x = (float)(src[idx]);
  destination[idx] = (real)(x * 0.5 * (1. + erf(x * 0.70710678118654752440)));
}
      )";
      }
    }
  }
  return shader;
}

std::string GeluKernel::createConstants() const noexcept {

  std::string defines = "";
  if (tanh && (value == 0 || value == 1)) {
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
