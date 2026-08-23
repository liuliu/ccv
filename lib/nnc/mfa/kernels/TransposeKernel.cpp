#include "TransposeKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

TransposeKernel::TransposeKernel(TransposeKernelDescriptor descriptor, MTL::Device* const device)
{
	memoryPrecision = descriptor.memoryPrecision;
	threadgroupSize = MTL::Size(32, 8, 1);
	const std::string source = createSource();
	{
		auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
		NS::Error* error = nil;
		library = NS::TransferPtr(device->newLibrary(string, nil, &error));
		CCV_NNC_MFA_CHECK_ERROR(error);
	}
}

MTL::Size TransposeKernel::gridSize(uint32_t batchSize, uint32_t rows, uint32_t cols) const noexcept
{
	return MTL::Size((cols + 31) / 32, (rows + 31) / 32, batchSize);
}

std::string TransposeKernel::createSource() const noexcept
{
	const char* const scalarType = memoryPrecision == GEMMOperandPrecision::FP32 ? "uint" : "ushort";
	return std::string("typedef ") + scalarType + R"( element;

#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
  uint rows;
  uint cols;
  uint source_batch_stride;
  uint source_row_stride;
  uint destination_batch_stride;
  uint destination_row_stride;
};

kernel void transpose(
  device const element *source [[buffer(0)]],
  device element *destination [[buffer(1)]],
  constant TransposeParams& params [[buffer(2)]],
  uint3 tid [[thread_position_in_threadgroup]],
  uint3 group [[threadgroup_position_in_grid]]
) {
  // Pad the logical 32x32 tile so column-wise reads avoid a power-of-two row
  // stride in threadgroup memory, which may reduce bank conflicts.
  threadgroup element tile[32][33];
  const uint source_col = group.x * 32 + tid.x;
  for (uint j = 0; j < 32; j += 8) {
    const uint source_row = group.y * 32 + tid.y + j;
    if (source_row < params.rows && source_col < params.cols) {
      const ulong source_idx = ulong(group.z) * params.source_batch_stride +
        ulong(source_row) * params.source_row_stride + source_col;
      tile[tid.y + j][tid.x] = source[source_idx];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint destination_col = group.y * 32 + tid.x;
  for (uint j = 0; j < 32; j += 8) {
    const uint destination_row = group.x * 32 + tid.y + j;
    if (destination_row < params.cols && destination_col < params.rows) {
      const ulong destination_idx = ulong(group.z) * params.destination_batch_stride +
        ulong(destination_row) * params.destination_row_stride + destination_col;
      destination[destination_idx] = tile[tid.x][tid.y + j];
    }
  }
}
)";
}
