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

kernel void scatter_add_clear_counts(
  device atomic_uint* counts [[buffer(0)]],
  constant ScatterAddParams& params [[buffer(1)]],
  uint row [[thread_position_in_grid]])
{
  if (row < params.output_rows)
    atomic_store_explicit(counts + row, 0u, memory_order_relaxed);
}

kernel void scatter_add_build_inverse(
  device const int* indices [[buffer(0)]],
  device atomic_uint* counts [[buffer(1)]],
  device uint* inverse_rows [[buffer(2)]],
  constant ScatterAddParams& params [[buffer(3)]],
  uint input_row [[thread_position_in_grid]])
{
  if (input_row >= params.input_rows)
    return;
  const int output_row = indices[input_row];
  if (output_row < 0 || output_row >= (int)params.output_rows)
    return;
  const uint slot = atomic_fetch_add_explicit(counts + output_row, 1u, memory_order_relaxed);
  if (slot < params.count_per_output)
    inverse_rows[(uint)output_row * params.count_per_output + slot] = input_row;
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
