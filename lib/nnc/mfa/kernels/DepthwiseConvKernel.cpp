#include "DepthwiseConvKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

DepthwiseConvKernel::DepthwiseConvKernel(DepthwiseConvKernelDescriptor descriptor, MTL::Device* const device)
{
	vectorized = descriptor.vectorized;
	fusedBias = descriptor.fusedBias;
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

MTL::Size DepthwiseConvKernel::gridSize(uint32_t elementCount) const noexcept
{
	return MTL::Size((uint64_t(elementCount) + 255) / 256, 1, 1);
}

std::string DepthwiseConvKernel::createSource() const noexcept
{
	std::string shader = "#include <metal_stdlib>\nusing namespace metal;\n\n";
	if (memoryPrecision == GEMMOperandPrecision::FP32) {
		shader += "typedef float real;\ntypedef float4 real4;\n";
	} else if (memoryPrecision == GEMMOperandPrecision::BF16) {
		shader += "typedef bfloat real;\ntypedef bfloat4 real4;\n";
	} else {
		shader += "typedef half real;\ntypedef half4 real4;\n";
	}
	// One thread per output element (or group of four channels). Each thread walks its
	// kH x kW window with zero padding outside the input and accumulates in FP32.
	const std::string vector = vectorized ? "real4" : "real";
	const std::string accumulator = vectorized ? "float4" : "float";
	shader += R"(
struct depthwise_conv_arguments {
  uint channel_units;
  uint input_height;
  uint input_width;
  uint output_height;
  uint output_width;
  uint kernel_height;
  uint kernel_width;
  uint stride_height;
  uint stride_width;
  uint dilation_height;
  uint dilation_width;
  uint padding_top;
  uint padding_left;
  uint element_count;
};

kernel void depthwise_conv(
  device const )" + vector + R"( *input [[buffer(0)]],
  device const real *weights [[buffer(1)]],
  device )" + vector + R"( *output [[buffer(2)]],
)" + (fusedBias ? "  device const real *bias [[buffer(3)]],\n" : "") + R"(  constant depthwise_conv_arguments &arguments [[buffer(4)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= arguments.element_count)
    return;
  const uint channel_unit = index % arguments.channel_units;
  uint pixel = index / arguments.channel_units;
  const uint output_x = pixel % arguments.output_width;
  pixel /= arguments.output_width;
  const uint output_y = pixel % arguments.output_height;
  const uint batch = pixel / arguments.output_height;
  const uint kernel_size = arguments.kernel_height * arguments.kernel_width;
)";
	if (vectorized) {
		shader += R"(  const uint channel = channel_unit * 4;
  device const real *channel_weights[4] = {
    weights + channel * kernel_size, weights + (channel + 1) * kernel_size,
    weights + (channel + 2) * kernel_size, weights + (channel + 3) * kernel_size
  };
)";
		shader += fusedBias ? "  float4 sum = float4(bias[channel], bias[channel + 1], bias[channel + 2], bias[channel + 3]);\n" : "  float4 sum = 0;\n";
	} else {
		shader += "  device const real *channel_weights = weights + channel_unit * kernel_size;\n";
		shader += fusedBias ? "  float sum = float(bias[channel_unit]);\n" : "  float sum = 0;\n";
	}
	shader += R"(  const device )" + vector + R"( *batch_input = input + batch * arguments.input_height * arguments.input_width * arguments.channel_units;
  for (uint ky = 0; ky < arguments.kernel_height; ky++) {
    const int input_y = int(output_y * arguments.stride_height + ky * arguments.dilation_height) - int(arguments.padding_top);
    if (input_y < 0 || input_y >= int(arguments.input_height))
      continue;
    for (uint kx = 0; kx < arguments.kernel_width; kx++) {
      const int input_x = int(output_x * arguments.stride_width + kx * arguments.dilation_width) - int(arguments.padding_left);
      if (input_x < 0 || input_x >= int(arguments.input_width))
        continue;
      const uint k = ky * arguments.kernel_width + kx;
)";
	if (vectorized) {
		shader += "      const float4 weight = float4(channel_weights[0][k], channel_weights[1][k], channel_weights[2][k], channel_weights[3][k]);\n";
	} else {
		shader += "      const float weight = float(channel_weights[k]);\n";
	}
	shader += "      sum += " + accumulator + "(batch_input[(uint(input_y) * arguments.input_width + uint(input_x)) * arguments.channel_units + channel_unit]) * weight;\n";
	shader += R"(    }
  }
  output[index] = )" + vector + R"((sum);
}
)";
	return shader;
}
