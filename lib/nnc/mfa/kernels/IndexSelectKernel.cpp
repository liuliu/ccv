#include "IndexSelectKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

IndexSelectKernel::IndexSelectKernel(IndexSelectKernelDescriptor descriptor, MTL::Device* const device)
{
	std::string source = R"(
#include <metal_stdlib>
using namespace metal;
)";
	if (descriptor.dataType == IndexSelectDataType::BF16)
		source += "typedef bfloat real;\n";
	else if (descriptor.dataType == IndexSelectDataType::FP16)
		source += "typedef half real;\n";
	else {
		CCV_NNC_MFA_PRECONDITION(descriptor.dataType == IndexSelectDataType::INT32);
		source += "typedef int real;\n";
	}
	if (descriptor.vectorWidth == 4)
		source += "typedef real realx __attribute__((ext_vector_type(4)));\n";
	else if (descriptor.vectorWidth == 2)
		source += "typedef real realx __attribute__((ext_vector_type(2)));\n";
	else {
		CCV_NNC_MFA_PRECONDITION(descriptor.vectorWidth == 1);
		source += "typedef real realx;\n";
	}
	source += R"(

constant ushort threads_per_row [[function_constant(0)]];
constant ushort rows_per_threadgroup [[function_constant(1)]];
)";
	if (!descriptor.loadM)
		source += "constant uint output_rows [[function_constant(2)]];\n";
	source += R"(

struct IndexSelectParams {
  uint row_units;
  uint output_rows;
};

kernel void index_select(
  device const realx* source [[buffer(0)]],
  device const int* indices [[buffer(1)]],
  device realx* destination [[buffer(2)]],
  constant IndexSelectParams& params [[buffer(3)]],
  uint3 threadgroup_position [[threadgroup_position_in_grid]],
  ushort thread_index [[thread_index_in_threadgroup]])
{
  const uint row_in_threadgroup = thread_index / threads_per_row;
  const uint column_in_row = thread_index - row_in_threadgroup * threads_per_row;
  const uint destination_row = threadgroup_position.x * rows_per_threadgroup + row_in_threadgroup;
)";
	if (descriptor.loadM)
		source += "  const uniform<uint> output_rows = make_uniform(params.output_rows);\n";
	source += R"(  if (destination_row >= output_rows)
    return;
  const int source_row = indices[destination_row];
  const ulong source_base = (ulong)source_row * params.row_units;
  const ulong destination_base = (ulong)destination_row * params.row_units;
  for (uint column = column_in_row; column < params.row_units; column += threads_per_row)
    destination[destination_base + column] = source[source_base + column];
}
)";
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}

MTL::Size IndexSelectKernel::gridSize(uint32_t outputRows, uint16_t threadsPerRow) const noexcept
{
	const uint32_t rows_per_threadgroup = 256 / threadsPerRow;
	return MTL::Size((outputRows + rows_per_threadgroup - 1) / rows_per_threadgroup, 1, 1);
}

MTL::Size IndexSelectKernel::threadgroupSize(uint32_t outputRows, uint16_t threadsPerRow) const noexcept
{
	const uint32_t rows_per_threadgroup = 256 / threadsPerRow;
	const uint32_t active_rows = outputRows < rows_per_threadgroup ? outputRows : rows_per_threadgroup;
	return MTL::Size(active_rows * threadsPerRow, 1, 1);
}
