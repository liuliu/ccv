#include "GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

GemvKernel::GemvKernel(GemvKernelDescriptor descriptor, MTL::Device* const device) {
  fusedBias = descriptor.fusedBias;
  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = MTL::Size(32, 1, 1);

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short GemvKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string GemvKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (fusedBias) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],
  device const real *bias [[buffer(3)]],

  uint tgpig [[threadgroup_position_in_grid]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  const uint rb = tgpig * N;
  device const real* y = (device const real*)src1;

  if (ncols < 128) {
    for (uint row = 0; row < N; ++row) {
      const uint r1 = rb + row;
      if (r1 >= nrows)
        break;
      device const real* x = (device const real*)src0 + r1 * ncols;
      float sumf = 0;
      for (uint i = tiisg; i < ncols; i += 32) {
        sumf += (real)x[i] * (real)y[i];
      }

      const float all_sum = simd_sum(sumf);
      if (tiisg == 0)
        dst[r1] = bias[r1] + (real)all_sum;
    }
  } else {
    device const real4* y4 = (device const real4*)y;
    for (uint row = 0; row < N; ++row) {
      const uint r1 = rb + row;
      if (r1 >= nrows)
        break;

      device const real* x = (device const real*)src0 + r1 * ncols;
      device const real4* x4 = (device const real4*)x;

      float sumf = 0;
      for (uint i = tiisg; i < ncols / 4; i += 32) {
        sumf += (real)x4[i][0] * y4[i][0];
        sumf += (real)x4[i][1] * y4[i][1];
        sumf += (real)x4[i][2] * y4[i][2];
        sumf += (real)x4[i][3] * y4[i][3];
      }

      const float all_sum = simd_sum(sumf);
      if (tiisg == 0)
        dst[r1] = bias[r1] + (real)all_sum;
    }
  }
}
    )";
  } else {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],

  uint tgpig [[threadgroup_position_in_grid]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  const uint rb = tgpig * N;
  device const real* y = (device const real*)src1;

  if (ncols < 128) {
    for (uint row = 0; row < N; ++row) {
      const uint r1 = rb + row;
      if (r1 >= nrows)
        break;
      device const real* x = (device const real*)src0 + r1 * ncols;
      float sumf = 0;
      for (uint i = tiisg; i < ncols; i += 32) {
        sumf += (real)x[i] * (real)y[i];
      }

      const float all_sum = simd_sum(sumf);
      if (tiisg == 0)
        dst[r1] = (real)all_sum;
    }
  } else {
    device const real4* y4 = (device const real4*)y;
    for (uint row = 0; row < N; ++row) {
      const uint r1 = rb + row;
      if (r1 >= nrows)
        break;

      device const real* x = (device const real*)src0 + r1 * ncols;
      device const real4* x4 = (device const real4*)x;

      float sumf = 0;
      for (uint i = tiisg; i < ncols / 4; i += 32) {
        sumf += (real)x4[i][0] * y4[i][0];
        sumf += (real)x4[i][1] * y4[i][1];
        sumf += (real)x4[i][2] * y4[i][2];
        sumf += (real)x4[i][3] * y4[i][3];
      }

      const float all_sum = simd_sum(sumf);
      if (tiisg == 0)
        dst[r1] = (real)all_sum;
    }
  }
}
    )";
  }
  return shader;
}

std::string GemvKernel::createConstants() const noexcept {
  std::string defines = "";
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += std::string("typedef float real;");
    defines += "\n";
    defines += std::string("typedef float4 real4;");
    defines += "\n";
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    defines += std::string("typedef bfloat real;");
    defines += "\n";
    defines += std::string("typedef bfloat4 real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
    defines += std::string("typedef half4 real4;");
    defines += "\n";
  }
  defines += "constant uint N [[function_constant(0)]];";
  defines += "\n";
  defines += "constant uint ncols [[function_constant(1)]];";
  defines += "\n";
  defines += "constant uint nrows [[function_constant(2)]];";
  defines += "\n";
  return defines;
}
