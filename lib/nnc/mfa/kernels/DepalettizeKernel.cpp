#include "DepalettizeKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

DepalettizeKernel::DepalettizeKernel(DepalettizeKernelDescriptor descriptor, MTL::Device* const device) {
  qbits = descriptor.qbits;
  partial = descriptor.partial;
  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
  threadgroupSize = MTL::Size(qbits == 5 ? 128 : 256, 1, 1);

  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

MTL::Size DepalettizeKernel::gridSize(uint32_t length, uint32_t numberInBlocks) const noexcept {
  if (qbits == 5)
    return MTL::Size(numberInBlocks / (128 * 8), length / numberInBlocks, 1);
  if (qbits == 6)
    return MTL::Size(numberInBlocks / (256 * 4), length / numberInBlocks, 1);
  if (partial)
    return MTL::Size(length / (256 * 4), 1, 1);
  return MTL::Size(numberInBlocks / (256 * 4), (length + numberInBlocks - 1) / numberInBlocks, 1);
}

unsigned short DepalettizeKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string DepalettizeKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (qbits == 5) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 5) * tgid.y;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = tgid.x * threadgroup_size + lid;
  device const uchar *ui1 = (device const uchar*)(ui0 + sizeof(real) * palette_size);
  const uchar u0 = ui1[x * 5];
  const uchar u1 = ui1[x * 5 + 1];
  const uchar u2 = ui1[x * 5 + 2];
  const uchar u3 = ui1[x * 5 + 3];
  const uchar u4 = ui1[x * 5 + 4];
  const real4 d0 = real4(palette[u0 >> 3], palette[((u0 & 7) << 2) | (u1 >> 6)], palette[(u1 >> 1) & 31], palette[((u1 & 1) << 4) | (u2 >> 4)]);
  const real4 d1 = real4(palette[((u2 & 15) << 1) | (u3 >> 7)], palette[(u3 >> 2) & 31], palette[((u3 & 3) << 3) | (u4 >> 5)], palette[u4 & 31]);
  destination[(number_in_blocks * tgid.y + x) * 2] = d0;
  destination[(number_in_blocks * tgid.y + x) * 2 + 1] = d1;
}
    )";
  } else if (qbits == 6) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 3) * tgid.y;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = tgid.x * threadgroup_size + lid;
  device const packed_uchar3 *ui1 = (device const packed_uchar3*)(ui0 + sizeof(real) * palette_size);
  const packed_uchar3 u = ui1[x];
  const real4 d = real4(palette[u.x >> 2], palette[((u.x & 3) << 4) | (u.y >> 4)], palette[((u.y & 15) << 2) | (u.z >> 6)], palette[u.z & 63]);
  destination[number_in_blocks * tgid.y + x] = d;
}
    )";
  } else if (!partial) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 4) * tgid.y;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = tgid.x * threadgroup_size + lid;
  device const uchar4 *ui1 = (device const uchar4*)(ui0 + sizeof(real) * palette_size);
  const uchar4 u = ui1[x];
  const real4 d = real4(palette[u.x], palette[u.y], palette[u.z], palette[u.w]);
  destination[number_in_blocks * tgid.y + x] = d;
}
    )";
  } else {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint block_idx = tgid / number_of_elements_per_segment;
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 4) * block_idx;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = (tgid % number_of_elements_per_segment) * threadgroup_size + lid;
  device const uchar4 *ui1 = (device const uchar4*)(ui0 + sizeof(real) * palette_size);
  const uchar4 u = ui1[x];
  const real4 d = real4(palette[u.x], palette[u.y], palette[u.z], palette[u.w]);
  destination[number_in_blocks * block_idx + x] = d;
}
    )";
  }
  return shader;
}

std::string DepalettizeKernel::createConstants() const noexcept {
  std::string defines = "";
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float real;\n";
    defines += "typedef float4 real4;\n";
  } else {
    defines += "typedef half real;\n";
    defines += "typedef half4 real4;\n";
  }
  if (qbits == 5) {
    defines += "constant ushort threadgroup_size = 128;\n";
    defines += "constant ushort palette_size = 32;\n";
    defines += "constant uint number_in_blocks [[function_constant(0)]];\n";
  } else if (qbits == 6) {
    defines += "constant ushort threadgroup_size = 256;\n";
    defines += "constant ushort palette_size = 64;\n";
    defines += "constant uint number_in_blocks [[function_constant(0)]];\n";
  } else {
    defines += "constant ushort threadgroup_size = 256;\n";
    defines += "constant ushort palette_size = 256;\n";
    defines += "constant uint number_in_blocks [[function_constant(0)]];\n";
    if (partial)
      defines += "constant uint number_of_elements_per_segment [[function_constant(1)]];\n";
  }
  return defines;
}
