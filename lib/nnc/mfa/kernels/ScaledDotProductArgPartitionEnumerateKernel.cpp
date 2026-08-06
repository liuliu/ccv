#include "ScaledDotProductArgPartitionEnumerateKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ScaledDotProductArgPartitionEnumerateKernel::ScaledDotProductArgPartitionEnumerateKernel(ScaledDotProductArgPartitionEnumerateKernelDescriptor descriptor, MTL::Device* const device) {
  (void)descriptor;
  threadgroupSize = MTL::Size(256, 1, 1);
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

MTL::Size ScaledDotProductArgPartitionEnumerateKernel::gridSize(const uint32_t T, const uint32_t kth) const noexcept {
  const size_t count = (size_t)T * kth;
  return MTL::Size((count + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);
}

std::string ScaledDotProductArgPartitionEnumerateKernel::createSource() const noexcept {
  return R"(
#include <metal_stdlib>
using namespace metal;

constant ushort threadgroup_size = 256;
constant uint T [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint kth [[function_constant(2)]];
constant uint compression_ratio [[function_constant(3)]];
constant bool is_causal [[function_constant(4)]];
constant int query_offset [[function_constant(5)]];

inline uint visible_count_for_token(const uint t) {
  if (!is_causal) {
    return C;
  }
  int visible = (query_offset + int(t) + 1) / int(compression_ratio);
  visible = max(visible, 0);
  visible = min(visible, int(C));
  return uint(visible);
}

kernel void enumerate(
  device int* selected [[buffer(0)]],
  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint index = tgid.x * threadgroup_size + lid;
  if (index >= T * kth) {
    return;
  }
  const uint t = index / kth;
  const uint position = index - t * kth;
  selected[index] = position < visible_count_for_token(t) ? int(position) : -1;
}
)";
}
