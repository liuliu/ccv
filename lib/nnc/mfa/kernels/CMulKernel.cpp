#include "CMulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

CMulKernel::CMulKernel(CMulKernelDescriptor descriptor, MTL::Device *const device) {

  conjugate = descriptor.conjugate;

  value = descriptor.value;

  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();

  if (value == 0) {
    threadgroupSize = MTL::Size(256, 1, 1);
  } else if (value == 1) {
    threadgroupSize = MTL::Size(32, 8, 1);
  } else if (value == 2) {
    threadgroupSize = MTL::Size(32, 8, 1);
  } else {
    threadgroupSize = MTL::Size(32, 8, 1);
  }

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short CMulKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string CMulKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (conjugate) {
    if (value == 0) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= dim0)
    return;
  const float a0 = (float)src0[idx * 2];
  const float a1 = (float)src0[idx * 2 + 1];
  const float b0 = (float)src1[idx * 2];
  const float b1 = (float)src1[idx * 2 + 1];
  destination[idx * 2] = (real)(a0 * b0 + a1 * b1);
  destination[idx * 2 + 1] = (real)(-a0 * b1 + a1 * b0);
}
    )";
    } else if (value == 1) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  if (y >= dim1 || x >= dim0)
    return;
  const uint ida = y * astride0 + x * 2;
  const uint idb = y * bstride0 + x * 2;
  const uint idc = y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 + a1 * b1);
  destination[idc + 1] = (real)(-a0 * b1 + a1 * b0);
}
    )";
    } else if (value == 2) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  const uint z = tpig.z;
  if (y >= dim1 || x >= dim0)
    return;
  const uint ida = z * astride1 + y * astride0 + x * 2;
  const uint idb = z * bstride1 + y * bstride0 + x * 2;
  const uint idc = z * cstride1 + y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 + a1 * b1);
  destination[idc + 1] = (real)(-a0 * b1 + a1 * b0);
}
    )";
    } else {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  const uint z = tpig.z;
  if (y >= dim1 || x >= dim0)
    return;
  const int u = z % dim2;
  const int v = z / dim2;
  const uint ida = v * astride2 + u * astride1 + y * astride0 + x * 2;
  const uint idb = v * bstride2 + u * bstride1 + y * bstride0 + x * 2;
  const uint idc = v * cstride2 + u * cstride1 + y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 + a1 * b1);
  destination[idc + 1] = (real)(-a0 * b1 + a1 * b0);
}
    )";
    }
  } else {
    if (value == 0) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= dim0)
    return;
  const float a0 = (float)src0[idx * 2];
  const float a1 = (float)src0[idx * 2 + 1];
  const float b0 = (float)src1[idx * 2];
  const float b1 = (float)src1[idx * 2 + 1];
  destination[idx * 2] = (real)(a0 * b0 - a1 * b1);
  destination[idx * 2 + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
    } else if (value == 1) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  if (y >= dim1 || x >= dim0)
    return;
  const uint ida = y * astride0 + x * 2;
  const uint idb = y * bstride0 + x * 2;
  const uint idc = y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 - a1 * b1);
  destination[idc + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
    } else if (value == 2) {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  const uint z = tpig.z;
  if (y >= dim1 || x >= dim0)
    return;
  const uint ida = z * astride1 + y * astride0 + x * 2;
  const uint idb = z * bstride1 + y * bstride0 + x * 2;
  const uint idc = z * cstride1 + y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 - a1 * b1);
  destination[idc + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
    } else {
      shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  const uint z = tpig.z;
  if (y >= dim1 || x >= dim0)
    return;
  const int u = z % dim2;
  const int v = z / dim2;
  const uint ida = v * astride2 + u * astride1 + y * astride0 + x * 2;
  const uint idb = v * bstride2 + u * bstride1 + y * bstride0 + x * 2;
  const uint idc = v * cstride2 + u * cstride1 + y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 - a1 * b1);
  destination[idc + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
    }
  }
  return shader;
}

std::string CMulKernel::createConstants() const noexcept {

  std::string defines = "";
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

  defines += "constant uint dim0 [[function_constant(0)]];";
  defines += "\n";
  if (value != 0)
  {
    defines += "constant uint dim1 [[function_constant(1)]];";
    defines += "\n";
    defines += "constant uint astride0 [[function_constant(2)]];";
    defines += "\n";
    defines += "constant uint bstride0 [[function_constant(3)]];";
    defines += "\n";
    defines += "constant uint cstride0 [[function_constant(4)]];";
    defines += "\n";
  }
  if (value != 0 && value != 1)
  {
    defines += "constant uint astride1 [[function_constant(5)]];";
    defines += "\n";
    defines += "constant uint bstride1 [[function_constant(6)]];";
    defines += "\n";
    defines += "constant uint cstride1 [[function_constant(7)]];";
    defines += "\n";
  }
  if (value != 0 && value != 1 && value != 2)
  {
    defines += "constant uint dim2 [[function_constant(8)]];";
    defines += "\n";
    defines += "constant uint astride2 [[function_constant(9)]];";
    defines += "\n";
    defines += "constant uint bstride2 [[function_constant(10)]];";
    defines += "\n";
    defines += "constant uint cstride2 [[function_constant(11)]];";
    defines += "\n";
  }
  return defines;
}
