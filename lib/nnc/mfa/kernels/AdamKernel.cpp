#include "AdamKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

AdamKernel::AdamKernel(AdamKernelDescriptor descriptor, MTL::Device* const device) {
  adamw = descriptor.adamw;
  amsgrad = descriptor.amsgrad;
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

MTL::Size AdamKernel::gridSize(uint32_t length) const noexcept {
  return MTL::Size((length + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);
}

unsigned short AdamKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string AdamKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (adamw) {
    if (amsgrad) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  device real *vm [[buffer(7)]],
  device real *new_vm [[buffer(8)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  const float vel_hat = v * inv_bias_correction2;
  const float vel_max_hat = max((float)vm[idx], vel_hat);
  destination[idx] = (real)(a - rate_decay * a - (m * rate_inv_bias_correction1) / (sqrt(vel_max_hat) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
  new_vm[idx] = (real)vel_max_hat;
}
      )";
    } else {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  destination[idx] = (real)(a - rate_decay * a - (m * rate_inv_bias_correction1) / (sqrt(v * inv_bias_correction2) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
}
      )";
    }
  } else {
    if (amsgrad) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  device real *vm [[buffer(7)]],
  device real *new_vm [[buffer(8)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  grad += weight_decay * a;
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  const float vel_hat = v * inv_bias_correction2;
  const float vel_max_hat = max((float)vm[idx], vel_hat);
  destination[idx] = (real)(a - (m * rate_inv_bias_correction1) / (sqrt(vel_max_hat) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
  new_vm[idx] = (real)vel_max_hat;
}
      )";
    } else {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  grad += weight_decay * a;
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  destination[idx] = (real)(a - (m * rate_inv_bias_correction1) / (sqrt(v * inv_bias_correction2) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
}
      )";
    }
  }
  return shader;
}

std::string AdamKernel::createConstants() const noexcept {
  std::string defines = "";
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float real;\n";
  } else {
    defines += "typedef half real;\n";
  }
  defines += "constant ushort threadgroup_size = 256;\n";
  defines += "constant uint tensor_length [[function_constant(0)]];\n";
  if (adamw) {
    defines += "constant float rate_decay [[function_constant(1)]];\n";
  } else {
    defines += "constant float weight_decay [[function_constant(1)]];\n";
  }
  defines += "constant float scale [[function_constant(2)]];\n";
  defines += "constant float beta1 [[function_constant(3)]];\n";
  defines += "constant float beta2 [[function_constant(4)]];\n";
  defines += "constant float epsilon [[function_constant(5)]];\n";
  return defines;
}
