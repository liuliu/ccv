#include "WalshHadamardTransformKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

WalshHadamardTransformKernel::WalshHadamardTransformKernel(WalshHadamardTransformKernelDescriptor descriptor, MTL::Device* const device) {

  memoryPrecision = descriptor.memoryPrecision;
  loadM = descriptor.loadM;

  const std::string source = createSource();

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

std::string WalshHadamardTransformKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  shader += R"(
#include <metal_stdlib>
using namespace metal;

inline void walsh_hadamard_radix(thread accum* x, const uint radix)
{
  #pragma clang loop unroll(full)
  for (uint stride = 1; stride < radix; stride <<= 1) {
    #pragma clang loop unroll(full)
    for (uint base = 0; base < radix; base += stride << 1) {
      #pragma clang loop unroll(full)
      for (uint i = 0; i < stride; i++) {
        const accum a = x[base + i];
        const accum b = x[base + i + stride];
        x[base + i] = a + b;
        x[base + i + stride] = a - b;
      }
    }
  }
}

kernel void walsh_hadamard_transform(
  device const real* source [[buffer(0)]],
  device real* destination [[buffer(1)]],
  threadgroup accum* buffer [[threadgroup(0)]],
  uint row_group [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
  if (strategy == 2) {
    const uint local_row = tid / dim;
    const uint lane = tid - local_row * dim;
    const uint row = row_group * rows_per_threadgroup + local_row;
    const bool valid = row < row_count;
    const uint row_base = row * dim;
    threadgroup accum* row_buffer = buffer + local_row * dim;
    accum v = valid ? (accum)(source[row_base + lane]) : (accum)0;
    #pragma clang loop unroll(full)
    for (uint stride = 1; stride < 32; stride <<= 1) {
      const accum other = simd_shuffle_xor(v, (ushort)stride);
      v = ((lane & stride) == 0) ? (v + other) : (other - v);
    }
    row_buffer[lane] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint stride = 32; stride < dim; stride <<= 1) {
      if ((lane & stride) == 0) {
        const uint base = (lane & ~((stride << 1) - 1)) + (lane & (stride - 1));
        const accum a = row_buffer[base];
        const accum b = row_buffer[base + stride];
        row_buffer[base] = a + b;
        row_buffer[base + stride] = a - b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid)
      destination[row_base + lane] = (real)(row_buffer[lane] * (accum)scale);
    return;
  }

  const uint row_base = row_group * dim;
  if (strategy == 1) {
    buffer[tid] = (accum)(source[row_base + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint stride = 1; stride < dim; stride <<= 1) {
      if ((tid & stride) == 0) {
        const uint base = (tid & ~((stride << 1) - 1)) + (tid & (stride - 1));
        const accum a = buffer[base];
        const accum b = buffer[base + stride];
        buffer[base] = a + b;
        buffer[base + stride] = a - b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    destination[row_base + tid] = (real)(buffer[tid] * (accum)scale);
    return;
  }

  #pragma clang loop unroll(full)
  for (uint i = 0; i < max_radix; i++) {
    const uint x = i * num_threads + tid;
    buffer[x] = (accum)(source[row_base + x]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  accum x[16];
  uint h = 1;
  #pragma clang loop unroll(full)
  for (uint step = 0; step < num_steps; step++) {
    const uint k = tid & (h - 1);
    const uint j = ((tid - k) * max_radix) + k;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < max_radix; i++)
      x[i] = buffer[j + h * i];
    walsh_hadamard_radix(x, max_radix);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < max_radix; i++)
      buffer[j + h * i] = x[i];
    h *= max_radix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (final_radix > 1) {
    const uint chunks = max_radix / final_radix;
    #pragma clang loop unroll(full)
    for (uint t = 0; t < chunks; t++) {
      const uint index = tid + t * num_threads;
      const uint k = index & (h - 1);
      const uint j = ((index - k) * final_radix) + k;
      #pragma clang loop unroll(full)
      for (uint i = 0; i < final_radix; i++)
        x[i] = buffer[j + h * i];
      walsh_hadamard_radix(x, final_radix);
      #pragma clang loop unroll(full)
      for (uint i = 0; i < final_radix; i++)
        buffer[j + h * i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (uint i = 0; i < max_radix; i++) {
    const uint x = i * num_threads + tid;
    destination[row_base + x] = (real)(buffer[x] * (accum)scale);
  }
}
  )";
  if (loadM) {
    const std::string::size_type argumentPosition = shader.find("  threadgroup accum* buffer [[threadgroup(0)]],");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint* loadM [[buffer(2)]],\n");
    const std::string::size_type rowCountPosition = shader.find("  if (strategy == 2) {");
    CCV_NNC_MFA_PRECONDITION(rowCountPosition != std::string::npos);
    shader.insert(rowCountPosition, "  const uniform<uint> row_count = make_uniform(loadM[0]);\n");
  }
  return shader;
}

std::string WalshHadamardTransformKernel::createConstants() const noexcept {

  std::string defines = "";
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += std::string("typedef float real;");
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    defines += std::string("typedef bfloat real;");
  } else {
    defines += std::string("typedef half real;");
  }
  defines += "\n";
  defines += std::string("typedef float accum;");
  defines += "\n";
  defines += "constant uint dim [[function_constant(0)]];";
  defines += "\n";
  defines += "constant uint max_radix [[function_constant(1)]];";
  defines += "\n";
  defines += "constant uint num_threads [[function_constant(2)]];";
  defines += "\n";
  defines += "constant uint num_steps [[function_constant(3)]];";
  defines += "\n";
  defines += "constant uint final_radix [[function_constant(4)]];";
  defines += "\n";
  defines += "constant float scale [[function_constant(5)]];";
  defines += "\n";
  if (!loadM) {
    defines += "constant uint row_count [[function_constant(6)]];";
    defines += "\n";
  }
  defines += "constant uint strategy [[function_constant(7)]];";
  defines += "\n";
  defines += "constant uint rows_per_threadgroup [[function_constant(8)]];";
  defines += "\n";
  return defines;
}
