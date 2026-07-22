#include "RotateHalfKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

RotateHalfKernel::RotateHalfKernel(RotateHalfKernelDescriptor descriptor, MTL::Device *const device) {

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

unsigned short RotateHalfKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string RotateHalfKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (value == 0) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void rotate_half(
  device real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const uint x = idx % dim;
  const uint y = idx / dim;
  const uint source = y * dim + ((x < half_dim) ? (x + half_dim) : (x - half_dim));
  destination[idx] = src[source];
}
    )";
  } else {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void rotate_half(
  device real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  const uint x = idx % dim;
  const uint y = idx / dim;
  const uint source = y * dim + ((x < half_dim) ? (x + half_dim) : (x - half_dim));
  destination[idx] = src[source];
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

std::string RotateHalfKernel::createConstants() const noexcept {

  std::string defines = "";
  if (value == 0) {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef uint4 real4;");
      defines += "\n";
    } else {
      defines += std::string("typedef uint2 real4;");
      defines += "\n";
    }
  } else {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef uint real;");
      defines += "\n";
    } else {
      defines += std::string("typedef ushort real;");
      defines += "\n";
    }
  }
  if (!loadM) {
    defines += "constant uint count [[function_constant(0)]];";
    defines += "\n";
  }
  defines += "constant uint dim [[function_constant(1)]];";
  defines += "\n";
  defines += "constant uint half_dim [[function_constant(2)]];";
  defines += "\n";
  return defines;
}
