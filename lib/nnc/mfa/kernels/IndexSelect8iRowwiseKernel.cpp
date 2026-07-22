#include "IndexSelect8iRowwiseKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

IndexSelect8iRowwiseKernel::IndexSelect8iRowwiseKernel(IndexSelect8iRowwiseKernelDescriptor descriptor, MTL::Device* const device) {
	vectorized = descriptor.vectorized;
	loadM = descriptor.loadM;
	memoryPrecision = descriptor.memoryPrecision;

	source = createSource();

	threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();
	threadgroupSize = MTL::Size(256, 1, 1);

	{
		auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
		NS::Error* error = nil;
		library = NS::TransferPtr(device->newLibrary(string, nil, &error));
		CCV_NNC_MFA_CHECK_ERROR(error);
	}
}

MTL::Size IndexSelect8iRowwiseKernel::gridSize(uint32_t outputLength) const noexcept {
	const uint32_t elementCount = vectorized ? (outputLength / 4) : outputLength;
	return MTL::Size((elementCount + 255) / 256, 1, 1);
}

unsigned short IndexSelect8iRowwiseKernel::createThreadgroupMemoryAllocation() const noexcept {
	return 0;
}

std::string IndexSelect8iRowwiseKernel::createSource() const noexcept {
	std::string shader = createConstants() + "\n";
	if (vectorized) {
		shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void index_select_8i_rowwise(
  device const char4 *source [[buffer(0)]],
  device const int *indices [[buffer(1)]],
  device real4 *destination [[buffer(2)]],
  device const real *scales [[buffer(3)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  if (x >= element_count)
    return;
  const uint dest_row = x / row_units;
  const uint col = x - dest_row * row_units;
  const int source_row = indices[dest_row];
  const real scale = scales[source_row];
  const char4 q = source[(ulong)source_row * row_units + col];
  destination[x] = real4((real)q.x, (real)q.y, (real)q.z, (real)q.w) * scale;
}
		)";
	} else {
		shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void index_select_8i_rowwise(
  device const char *source [[buffer(0)]],
  device const int *indices [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device const real *scales [[buffer(3)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  if (x >= element_count)
    return;
  const uint dest_row = x / row_units;
  const uint col = x - dest_row * row_units;
  const int source_row = indices[dest_row];
  destination[x] = (real)source[(ulong)source_row * row_units + col] * scales[source_row];
}
		)";
	}
	if (loadM) {
		const std::string::size_type argumentPosition = shader.find("  uint3 tgid [[threadgroup_position_in_grid]]");
		CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
		shader.insert(argumentPosition, "  const device uint *loadM [[buffer(4)]],\n");
		const std::string::size_type elementCountPosition = shader.find("  const uint x = tgid.x * threadgroup_size + lid;");
		CCV_NNC_MFA_PRECONDITION(elementCountPosition != std::string::npos);
		shader.insert(elementCountPosition, "  const uniform<uint> element_count = make_uniform(loadM[0]);\n");
	}
	return shader;
}

std::string IndexSelect8iRowwiseKernel::createConstants() const noexcept {
	std::string defines = "";
	if (memoryPrecision == GEMMOperandPrecision::FP32) {
		defines += "typedef float real;\n";
		if (vectorized)
			defines += "typedef float4 real4;\n";
	} else if (memoryPrecision == GEMMOperandPrecision::BF16) {
		defines += "typedef bfloat real;\n";
		if (vectorized)
			defines += "typedef bfloat4 real4;\n";
	} else {
		defines += "typedef half real;\n";
		if (vectorized)
			defines += "typedef half4 real4;\n";
	}
	defines += "constant ushort threadgroup_size = 256;\n";
	defines += "constant uint row_units [[function_constant(0)]];\n";
	if (!loadM)
		defines += "constant uint element_count [[function_constant(1)]];\n";
	return defines;
}
