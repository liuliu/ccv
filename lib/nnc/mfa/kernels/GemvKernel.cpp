#include "GemvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

GemvKernel::GemvKernel(GemvKernelDescriptor descriptor, MTL::Device* const device) {
  fusedBias = descriptor.fusedBias;
  mrows = descriptor.mrows;
  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

std::string GemvKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (mrows == 3) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],
)";
    if (fusedBias) {
      shader += R"(
  device const real *bias [[buffer(3)]],
)";
    }
    shader += R"(
  uint tgpig [[threadgroup_position_in_grid]],
  uint tiitg [[thread_index_in_threadgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  constexpr uint TILE_COLS = 256;
  const uint rb = tgpig * N;
  const uint row = rb + sgitg;
  device const real* y0 = (device const real*)src1;
  device const real* y1 = y0 + ncols;
  device const real* y2 = y1 + ncols;
  threadgroup real y0_shared[TILE_COLS];
  threadgroup real y1_shared[TILE_COLS];
  threadgroup real y2_shared[TILE_COLS];
  const bool active = row < nrows;
  device const real* x = active ? ((device const real*)src0 + row * ncols) : (device const real*)src0;

  float sum0 = 0;
  float sum1 = 0;
  float sum2 = 0;
  uint k = 0;
  for (; k + TILE_COLS <= ncols; k += TILE_COLS) {
    for (uint i = tiitg; i < TILE_COLS; i += N * 32) {
      y0_shared[i] = y0[k + i];
      y1_shared[i] = y1[k + i];
      y2_shared[i] = y2[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint i = tiisg * 4; i + 3 < TILE_COLS; i += 32 * 4) {
        const float x0 = (float)x[k + i + 0];
        const float x1 = (float)x[k + i + 1];
        const float x2 = (float)x[k + i + 2];
        const float x3 = (float)x[k + i + 3];
        sum0 += x0 * (float)y0_shared[i + 0];
        sum0 += x1 * (float)y0_shared[i + 1];
        sum0 += x2 * (float)y0_shared[i + 2];
        sum0 += x3 * (float)y0_shared[i + 3];
        sum1 += x0 * (float)y1_shared[i + 0];
        sum1 += x1 * (float)y1_shared[i + 1];
        sum1 += x2 * (float)y1_shared[i + 2];
        sum1 += x3 * (float)y1_shared[i + 3];
        sum2 += x0 * (float)y2_shared[i + 0];
        sum2 += x1 * (float)y2_shared[i + 1];
        sum2 += x2 * (float)y2_shared[i + 2];
        sum2 += x3 * (float)y2_shared[i + 3];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k < ncols) {
    const uint tile = ncols - k;
    for (uint i = tiitg; i < tile; i += N * 32) {
      y0_shared[i] = y0[k + i];
      y1_shared[i] = y1[k + i];
      y2_shared[i] = y2[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      const uint tile4 = tile & ~3u;
      for (uint i = tiisg * 4; i + 3 < tile4; i += 32 * 4) {
        const float x0 = (float)x[k + i + 0];
        const float x1 = (float)x[k + i + 1];
        const float x2 = (float)x[k + i + 2];
        const float x3 = (float)x[k + i + 3];
        sum0 += x0 * (float)y0_shared[i + 0];
        sum0 += x1 * (float)y0_shared[i + 1];
        sum0 += x2 * (float)y0_shared[i + 2];
        sum0 += x3 * (float)y0_shared[i + 3];
        sum1 += x0 * (float)y1_shared[i + 0];
        sum1 += x1 * (float)y1_shared[i + 1];
        sum1 += x2 * (float)y1_shared[i + 2];
        sum1 += x3 * (float)y1_shared[i + 3];
        sum2 += x0 * (float)y2_shared[i + 0];
        sum2 += x1 * (float)y2_shared[i + 1];
        sum2 += x2 * (float)y2_shared[i + 2];
        sum2 += x3 * (float)y2_shared[i + 3];
      }
      for (uint i = tile4 + tiisg; i < tile; i += 32) {
        const float xv = (float)x[k + i];
        sum0 += xv * (float)y0_shared[i];
        sum1 += xv * (float)y1_shared[i];
        sum2 += xv * (float)y2_shared[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float all_sum0 = simd_sum(sum0);
  const float all_sum1 = simd_sum(sum1);
  const float all_sum2 = simd_sum(sum2);
  if (active && tiisg == 0) {
)";
    if (fusedBias) {
      shader += R"(
    const real biasv = bias[row];
    dst[row] = biasv + (real)all_sum0;
    dst[nrows + row] = biasv + (real)all_sum1;
    dst[nrows * 2 + row] = biasv + (real)all_sum2;
)";
    } else {
      shader += R"(
    dst[row] = (real)all_sum0;
    dst[nrows + row] = (real)all_sum1;
    dst[nrows * 2 + row] = (real)all_sum2;
)";
  }
    shader += R"(
  }
}
    )";
  } else if (mrows == 2) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],
)";
    if (fusedBias) {
      shader += R"(
  device const real *bias [[buffer(3)]],
)";
    }
    shader += R"(

  uint tgpig [[threadgroup_position_in_grid]],
  uint tiitg [[thread_index_in_threadgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  constexpr uint TILE_COLS = 256;
  const uint rb = tgpig * N;
  const uint row = rb + sgitg;
  device const real* y0 = (device const real*)src1;
  device const real* y1 = y0 + ncols;
  threadgroup real y0_shared[TILE_COLS];
  threadgroup real y1_shared[TILE_COLS];
  const bool active = row < nrows;
  device const real* x = active ? ((device const real*)src0 + row * ncols) : (device const real*)src0;

  float sum0 = 0;
  float sum1 = 0;
  uint k = 0;
  for (; k + TILE_COLS <= ncols; k += TILE_COLS) {
    for (uint i = tiitg; i < TILE_COLS; i += N * 32) {
      y0_shared[i] = y0[k + i];
      y1_shared[i] = y1[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint i = tiisg * 4; i + 3 < TILE_COLS; i += 32 * 4) {
        const float x0 = (float)x[k + i + 0];
        const float x1 = (float)x[k + i + 1];
        const float x2 = (float)x[k + i + 2];
        const float x3 = (float)x[k + i + 3];
        sum0 += x0 * (float)y0_shared[i + 0];
        sum0 += x1 * (float)y0_shared[i + 1];
        sum0 += x2 * (float)y0_shared[i + 2];
        sum0 += x3 * (float)y0_shared[i + 3];
        sum1 += x0 * (float)y1_shared[i + 0];
        sum1 += x1 * (float)y1_shared[i + 1];
        sum1 += x2 * (float)y1_shared[i + 2];
        sum1 += x3 * (float)y1_shared[i + 3];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k < ncols) {
    const uint tile = ncols - k;
    for (uint i = tiitg; i < tile; i += N * 32) {
      y0_shared[i] = y0[k + i];
      y1_shared[i] = y1[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      const uint tile4 = tile & ~3u;
      for (uint i = tiisg * 4; i + 3 < tile4; i += 32 * 4) {
        const float x0 = (float)x[k + i + 0];
        const float x1 = (float)x[k + i + 1];
        const float x2 = (float)x[k + i + 2];
        const float x3 = (float)x[k + i + 3];
        sum0 += x0 * (float)y0_shared[i + 0];
        sum0 += x1 * (float)y0_shared[i + 1];
        sum0 += x2 * (float)y0_shared[i + 2];
        sum0 += x3 * (float)y0_shared[i + 3];
        sum1 += x0 * (float)y1_shared[i + 0];
        sum1 += x1 * (float)y1_shared[i + 1];
        sum1 += x2 * (float)y1_shared[i + 2];
        sum1 += x3 * (float)y1_shared[i + 3];
      }
      for (uint i = tile4 + tiisg; i < tile; i += 32) {
        const float xv = (float)x[k + i];
        sum0 += xv * (float)y0_shared[i];
        sum1 += xv * (float)y1_shared[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float all_sum0 = simd_sum(sum0);
  const float all_sum1 = simd_sum(sum1);
  if (active && tiisg == 0) {
)";
    if (fusedBias) {
      shader += R"(
    const real biasv = bias[row];
    dst[row] = biasv + (real)all_sum0;
    dst[nrows + row] = biasv + (real)all_sum1;
)";
    } else {
      shader += R"(
    dst[row] = (real)all_sum0;
    dst[nrows + row] = (real)all_sum1;
)";
    }
    shader += R"(
  }
}
      )";
  } else if (fusedBias) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],
  device const real *bias [[buffer(3)]],

  uint tgpig [[threadgroup_position_in_grid]],
  uint tiitg [[thread_index_in_threadgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  constexpr uint TILE_COLS = 256;
  const uint rb = tgpig * N;
  const uint row = rb + sgitg;
  device const real* y = (device const real*)src1;
  threadgroup real y_shared[TILE_COLS];
  const bool active = row < nrows;
  device const real* x = active ? ((device const real*)src0 + row * ncols) : (device const real*)src0;

  float sumf = 0;
  uint k = 0;
  for (; k + TILE_COLS <= ncols; k += TILE_COLS) {
    for (uint i = tiitg; i < TILE_COLS; i += N * 32) {
      y_shared[i] = y[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint i = tiisg * 4; i + 3 < TILE_COLS; i += 32 * 4) {
        sumf += (float)x[k + i + 0] * (float)y_shared[i + 0];
        sumf += (float)x[k + i + 1] * (float)y_shared[i + 1];
        sumf += (float)x[k + i + 2] * (float)y_shared[i + 2];
        sumf += (float)x[k + i + 3] * (float)y_shared[i + 3];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k < ncols) {
    const uint tile = ncols - k;
    for (uint i = tiitg; i < tile; i += N * 32) {
      y_shared[i] = y[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      const uint tile4 = tile & ~3u;
      for (uint i = tiisg * 4; i + 3 < tile4; i += 32 * 4) {
        sumf += (float)x[k + i + 0] * (float)y_shared[i + 0];
        sumf += (float)x[k + i + 1] * (float)y_shared[i + 1];
        sumf += (float)x[k + i + 2] * (float)y_shared[i + 2];
        sumf += (float)x[k + i + 3] * (float)y_shared[i + 3];
      }
      for (uint i = tile4 + tiisg; i < tile; i += 32) {
        sumf += (float)x[k + i] * (float)y_shared[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float all_sum = simd_sum(sumf);
  if (active && tiisg == 0) {
    dst[row] = bias[row] + (real)all_sum;
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
  uint tiitg [[thread_index_in_threadgroup]],
  uint sgitg [[simdgroup_index_in_threadgroup]],
  uint tiisg [[thread_index_in_simdgroup]]
) {
  constexpr uint TILE_COLS = 256;
  const uint rb = tgpig * N;
  const uint row = rb + sgitg;
  device const real* y = (device const real*)src1;
  threadgroup real y_shared[TILE_COLS];
  const bool active = row < nrows;
  device const real* x = active ? ((device const real*)src0 + row * ncols) : (device const real*)src0;

  float sumf = 0;
  uint k = 0;
  for (; k + TILE_COLS <= ncols; k += TILE_COLS) {
    for (uint i = tiitg; i < TILE_COLS; i += N * 32) {
      y_shared[i] = y[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint i = tiisg * 4; i + 3 < TILE_COLS; i += 32 * 4) {
        sumf += (float)x[k + i + 0] * (float)y_shared[i + 0];
        sumf += (float)x[k + i + 1] * (float)y_shared[i + 1];
        sumf += (float)x[k + i + 2] * (float)y_shared[i + 2];
        sumf += (float)x[k + i + 3] * (float)y_shared[i + 3];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k < ncols) {
    const uint tile = ncols - k;
    for (uint i = tiitg; i < tile; i += N * 32) {
      y_shared[i] = y[k + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      const uint tile4 = tile & ~3u;
      for (uint i = tiisg * 4; i + 3 < tile4; i += 32 * 4) {
        sumf += (float)x[k + i + 0] * (float)y_shared[i + 0];
        sumf += (float)x[k + i + 1] * (float)y_shared[i + 1];
        sumf += (float)x[k + i + 2] * (float)y_shared[i + 2];
        sumf += (float)x[k + i + 3] * (float)y_shared[i + 3];
      }
      for (uint i = tile4 + tiisg; i < tile; i += 32) {
        sumf += (float)x[k + i] * (float)y_shared[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float all_sum = simd_sum(sumf);
  if (active && tiisg == 0) {
    dst[row] = (real)all_sum;
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
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    defines += std::string("typedef bfloat real;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
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
