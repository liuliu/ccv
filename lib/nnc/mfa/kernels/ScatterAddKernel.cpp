#include "ScatterAddKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

ScatterAddKernel::ScatterAddKernel(ScatterAddKernelDescriptor descriptor, MTL::Device* const device)
{
	source = R"(
#include <metal_stdlib>
using namespace metal;
)";
	if (descriptor.memoryPrecision == GEMMOperandPrecision::FP32)
		source += "typedef float4 real4;\n";
	else
		source += "typedef half4 real4;\n";
	source += R"(

struct ScatterAddParams {
  uint input_rows;
  uint output_rows;
  uint column_vectors;
  uint count_per_output;
  uint strategy;
};

enum ScatterAddStrategy : uint {
  scatter_add_scan = 0,
  scatter_add_atomic_min = 1,
  scatter_add_atomic_sort = 2,
};

kernel void scatter_add_single_output(
  device const real4* source [[buffer(0)]],
  device real4* destination [[buffer(1)]],
  constant ScatterAddParams& params [[buffer(2)]],
  uint column [[thread_position_in_grid]])
{
  if (column >= params.column_vectors)
    return;
  float4 total = 0.0f;
  for (uint row = 0; row < params.input_rows; ++row)
    total += float4(source[(ulong)row * params.column_vectors + column]);
  destination[column] = real4(total);
}

kernel void scatter_add_clear(
  device atomic_uint* values [[buffer(0)]],
  constant ScatterAddParams& params [[buffer(1)]],
  uint index [[thread_position_in_grid]])
{
  const uint count = params.strategy == scatter_add_atomic_min ? params.input_rows : params.output_rows;
  if (index >= count)
    return;
  atomic_store_explicit(values + index, params.strategy == scatter_add_atomic_min ? 0xffffffffu : 0u, memory_order_relaxed);
}

kernel void scatter_add_build_inverse(
  device const int* indices [[buffer(0)]],
  device atomic_uint* counters [[buffer(1)]],
  device uint* inverse_rows [[buffer(2)]],
  constant ScatterAddParams& params [[buffer(3)]],
  uint index [[thread_position_in_grid]])
{
  if (params.strategy == scatter_add_scan) {
    const uint output_row = index;
    if (output_row >= params.output_rows)
      return;
    device const packed_int4* vector_indices = (device const packed_int4*)indices;
    const int target = (int)output_row;
    uint slot = 0;
    uint input_row = 0;
    for (; input_row + 4 <= params.input_rows && slot < params.count_per_output; input_row += 4) {
      const int4 values = int4(vector_indices[input_row / 4]);
      if (values.x == target)
        inverse_rows[output_row * params.count_per_output + slot++] = input_row;
      if (slot < params.count_per_output && values.y == target)
        inverse_rows[output_row * params.count_per_output + slot++] = input_row + 1;
      if (slot < params.count_per_output && values.z == target)
        inverse_rows[output_row * params.count_per_output + slot++] = input_row + 2;
      if (slot < params.count_per_output && values.w == target)
        inverse_rows[output_row * params.count_per_output + slot++] = input_row + 3;
    }
    for (; input_row < params.input_rows && slot < params.count_per_output; ++input_row)
      if (indices[input_row] == target)
        inverse_rows[output_row * params.count_per_output + slot++] = input_row;
    return;
  }
  const uint input_row = index;
  if (input_row >= params.input_rows)
    return;
  const int output_row = indices[input_row];
  if (output_row < 0 || output_row >= (int)params.output_rows)
    return;
  const uint base = (uint)output_row * params.count_per_output;
  if (params.strategy == scatter_add_atomic_min) {
    uint value = input_row;
    for (uint slot = 0; slot < params.count_per_output; ++slot) {
      const uint old = atomic_fetch_min_explicit(counters + base + slot, value, memory_order_relaxed);
      if (old == 0xffffffffu)
        return;
      value = max(old, value);
    }
  } else {
    const uint slot = atomic_fetch_add_explicit(counters + output_row, 1u, memory_order_relaxed);
    if (slot < params.count_per_output)
      inverse_rows[base + slot] = input_row;
  }
}

kernel void scatter_add_sort_inverse(
  device uint* inverse_rows [[buffer(0)]],
  constant ScatterAddParams& params [[buffer(1)]],
  threadgroup uint* rows [[threadgroup(0)]],
  uint output_row [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint thread_count [[threads_per_threadgroup]])
{
  uint sort_count = 1;
  while (sort_count < params.count_per_output)
    sort_count <<= 1;
  for (uint i = tid; i < sort_count; i += thread_count)
    rows[i] = i < params.count_per_output ?
      inverse_rows[output_row * params.count_per_output + i] : 0xffffffffu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= sort_count; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      for (uint i = tid; i < sort_count; i += thread_count) {
        const uint other = i ^ j;
        if (other <= i)
          continue;
        const uint a = rows[i];
        const uint b = rows[other];
        const bool ascending = (i & k) == 0;
        if ((a > b) == ascending) {
          rows[i] = b;
          rows[other] = a;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = tid; i < params.count_per_output; i += thread_count)
    inverse_rows[output_row * params.count_per_output + i] = rows[i];
}

kernel void scatter_add_reduce(
  device const real4* source [[buffer(0)]],
  device const uint* inverse_rows [[buffer(1)]],
  device real4* destination [[buffer(2)]],
  constant ScatterAddParams& params [[buffer(3)]],
  uint index [[thread_position_in_grid]])
{
  const uint output_count = params.output_rows * params.column_vectors;
  if (index >= output_count)
    return;
  const uint output_row = index / params.column_vectors;
  const uint column = index - output_row * params.column_vectors;
  float4 total = 0.0f;
  for (uint slot = 0; slot < params.count_per_output; ++slot) {
    const uint input_row = inverse_rows[output_row * params.count_per_output + slot];
    total += float4(source[(ulong)input_row * params.column_vectors + column]);
  }
  destination[index] = real4(total);
}
)";
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
