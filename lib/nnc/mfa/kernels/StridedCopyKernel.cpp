#include "StridedCopyKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

StridedCopyKernel::StridedCopyKernel(StridedCopyKernelDescriptor descriptor, MTL::Device* const device)
{
	vectorized = descriptor.vectorized;
	destinationStrided = descriptor.destinationStrided;
	loadM = descriptor.loadM;
	memoryPrecision = descriptor.memoryPrecision;

	const std::string source = createSource();

	threadgroupSize = MTL::Size(256, 1, 1);

	{
		auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
		NS::Error* error = nil;
		library = NS::TransferPtr(device->newLibrary(string, nil, &error));
		CCV_NNC_MFA_CHECK_ERROR(error);
	}
}

MTL::Size StridedCopyKernel::gridSize(uint32_t rows, uint32_t cols) const noexcept
{
	const uint32_t elementCount = vectorized ? (rows * cols / 4) : (rows * cols);
	return MTL::Size((elementCount + 255) / 256, 1, 1);
}

std::string StridedCopyKernel::createSource() const noexcept
{
	std::string shader = createConstants() + "\n";
	if (vectorized) {
		const char* const destinationIndex = destinationStrided ? "row * destination_row_stride_units + col" : "x";
		shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void strided_copy(
  device const real4 *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  if (x >= element_count)
    return;
  const uint row = x / col_units;
  const uint col = x - row * col_units;
)";
		shader += "  destination[";
		shader += destinationIndex;
		shader += R"(] = source[row * source_row_stride_units + col];
}
		)";
	} else {
		const char* const destinationIndex = destinationStrided ? "row * destination_row_stride + col" : "x";
		shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void strided_copy(
  device const real *source [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  if (x >= element_count)
    return;
  const uint row = x / cols;
  const uint col = x - row * cols;
)";
		shader += "  destination[";
		shader += destinationIndex;
		shader += R"(] = source[row * source_row_stride + col];
}
		)";
	}
	if (loadM) {
		const std::string::size_type argumentPosition = shader.find("  uint3 tpig [[thread_position_in_grid]]");
		CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
		shader.insert(argumentPosition, "  const device uint *loadM [[buffer(2)]],\n");
		const std::string::size_type elementCountPosition = shader.find("  const uint x = tpig.x;");
		CCV_NNC_MFA_PRECONDITION(elementCountPosition != std::string::npos);
		shader.insert(elementCountPosition, "  const uniform<uint> element_count = make_uniform(loadM[0]);\n");
	}
	return shader;
}

std::string StridedCopyKernel::createConstants() const noexcept
{
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
	if (vectorized)
	{
		defines += "constant uint col_units [[function_constant(0)]];\n";
		defines += "constant uint source_row_stride_units [[function_constant(1)]];\n";
		if (destinationStrided)
			defines += "constant uint destination_row_stride_units [[function_constant(2)]];\n";
	} else {
		defines += "constant uint cols [[function_constant(0)]];\n";
		defines += "constant uint source_row_stride [[function_constant(1)]];\n";
		if (destinationStrided)
			defines += "constant uint destination_row_stride [[function_constant(2)]];\n";
	}
	if (!loadM)
		defines += destinationStrided ? "constant uint element_count [[function_constant(3)]];\n" : "constant uint element_count [[function_constant(2)]];\n";
	return defines;
}
