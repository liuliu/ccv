#include "Dequantize8iRowwiseKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

Dequantize8iRowwiseKernel::Dequantize8iRowwiseKernel(Dequantize8iRowwiseKernelDescriptor descriptor, MTL::Device* const device) {
	vectorized = descriptor.vectorized;
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

MTL::Size Dequantize8iRowwiseKernel::gridSize(uint32_t length) const noexcept {
	const uint32_t elementCount = vectorized ? (length / 4) : length;
	return MTL::Size((elementCount + 255) / 256, 1, 1);
}

unsigned short Dequantize8iRowwiseKernel::createThreadgroupMemoryAllocation() const noexcept {
	return 0;
}

std::string Dequantize8iRowwiseKernel::createSource() const noexcept {
	std::string shader = createConstants() + "\n";
	if (vectorized) {
		shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void dequantize_8i_rowwise(
  device const char4 *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],
  device const real *scales [[buffer(2)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  if (x >= element_count)
    return;
  const uint row = x / row_units;
  const real scale = scales[row];
  const char4 q = source[x];
  destination[x] = real4((real)q.x, (real)q.y, (real)q.z, (real)q.w) * scale;
}
		)";
	} else {
		shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void dequantize_8i_rowwise(
  device const char *source [[buffer(0)]],
  device real *destination [[buffer(1)]],
  device const real *scales [[buffer(2)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint x = tgid.x * threadgroup_size + lid;
  if (x >= element_count)
    return;
  const uint row = x / row_units;
  destination[x] = (real)source[x] * scales[row];
}
		)";
	}
	return shader;
}

std::string Dequantize8iRowwiseKernel::createConstants() const noexcept {
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
	defines += "constant uint element_count [[function_constant(1)]];\n";
	return defines;
}
