#include "RMSNormGatedKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace {

std::string high_precision_to_string(const float value)
{
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

}

RMSNormGatedKernel::RMSNormGatedKernel(RMSNormGatedKernelDescriptor descriptor, MTL::Device* const device) {
  epsilon = descriptor.epsilon;
  aPrecision = descriptor.aPrecision;
  gatePrecision = descriptor.gatePrecision;
  scalePrecision = descriptor.scalePrecision;
  columnCount = descriptor.columnCount;

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  groupSize = MTL::Size(columnCount <= 384 ? 128 : 256, 1, 1);

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

unsigned short RMSNormGatedKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string RMSNormGatedKernel::createSource() const noexcept {
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

kernel void rmsnorm_gated(
  device realA *source [[buffer(0)]],
  device realGate *gate [[buffer(1)]],
  device realScale *scale [[buffer(2)]],
  device realA *destination [[buffer(3)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  uint lid [[thread_index_in_threadgroup]]
) {
  source += tgid.x * column_count;
  gate += tgid.x * column_count;
  destination += tgid.x * column_count;

  float variance = 0;
  for (uint i = lid; i < column_count; i += threadgroup_size) {
    const float x = float(source[i]);
    variance += x * x;
  }
  threadgroup float partials[threadgroup_size / 32];
  partials[sidx] = simd_sum(variance);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < (threadgroup_size / 32)) {
    float variance = quad_sum(partials[lid]);
    if (threadgroup_size >= 256) {
      variance += simd_shuffle_xor(variance, 4);
    }
    partials[lid] = rsqrt(variance / float(column_count) + epsilon);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_std = partials[sidx];

  for (uint i = lid; i < column_count; i += threadgroup_size) {
    const float x = float(source[i]);
    const float z = float(gate[i]);
    const float gated = z * stable_sigmoid(z);
    const float result = x * inv_std * float(scale[i]) * gated;
    destination[i] = realA(result);
  }
}
  )";
  return shader;
}

std::string RMSNormGatedKernel::createConstants() const noexcept {
  std::string defines = "";
  if (aPrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float realA;";
  } else if (aPrecision == GEMMOperandPrecision::BF16) {
    defines += "typedef bfloat realA;";
  } else {
    defines += "typedef half realA;";
  }
  defines += "\n";
  if (gatePrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float realGate;";
  } else if (gatePrecision == GEMMOperandPrecision::BF16) {
    defines += "typedef bfloat realGate;";
  } else {
    defines += "typedef half realGate;";
  }
  defines += "\n";
  if (scalePrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float realScale;";
  } else if (scalePrecision == GEMMOperandPrecision::BF16) {
    defines += "typedef bfloat realScale;";
  } else {
    defines += "typedef half realScale;";
  }
  defines += "\n";
  defines += "constant uint column_count = ";
  defines += std::to_string(columnCount) + ";";
  defines += "\n";
  defines += "constant float epsilon = ";
  defines += high_precision_to_string(epsilon) + ";";
  defines += "\n";
  defines += "constant ushort threadgroup_size = ";
  defines += std::to_string(groupSize.width) + ";";
  defines += "\n";
  return defines;
}
