#include "NAConv3DKernel.hpp"
#include "NAConv3DDescriptor.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"
#include <algorithm>

NAConv3DKernel::NAConv3DKernel(NAConv3DKernelDescriptor descriptor, MTL::Device *const device) {
  blockDimensions = descriptor.blockDimensions;
  kernelDimensions = descriptor.kernelDimensions;
  dataType = descriptor.dataType;
  inputChannels = descriptor.inputChannels;
  outputChannels = descriptor.outputChannels;
  useBias = descriptor.useBias;
  executionSIMDGroups = 4;

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t NAConv3DKernel::permutationThreadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept {
  return std::min<uint32_t>(std::max<uint32_t>(pipelineState->threadExecutionWidth(), 1), pipelineState->maxTotalThreadsPerThreadgroup());
}

uint16_t NAConv3DKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState, const NAConv3DDescriptor &descriptor) const noexcept {
  return pipelineState->threadExecutionWidth() * executionSIMDGroups;
}

MTL::Size NAConv3DKernel::threadgroupsPerGrid(const NAConv3DDescriptor &descriptor) const noexcept {
  auto ceilDivide =
  [=](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  return MTL::Size(ceilDivide(descriptor.matrixDimensions[2], blockDimensions[0]), ceilDivide(descriptor.matrixDimensions[1], blockDimensions[1]), descriptor.batchDimension * descriptor.matrixDimensions[0]);
}

std::string NAConv3DKernel::createSource() const noexcept {
  CodeWriter source;
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint DEPTH [[function_constant(0)]];
constant uint HEIGHT [[function_constant(1)]];
constant uint WIDTH [[function_constant(2)]];
)";

  source.SetValue("SCALAR_NAME", dataType == 3 ? "float" : "half");
  source.SetValue("BLOCK_DIMENSIONS_WIDTH", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_HEIGHT", std::to_string(blockDimensions[1]));
  source.SetValue("KERNEL_DEPTH", std::to_string(kernelDimensions[0]));
  source.SetValue("KERNEL_HEIGHT", std::to_string(kernelDimensions[1]));
  source.SetValue("KERNEL_WIDTH", std::to_string(kernelDimensions[2]));
  source.SetValue("INPUT_CHANNELS", std::to_string(inputChannels));
  source.SetValue("OUTPUT_CHANNELS", std::to_string(outputChannels));
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
  source.SetValue("INPUT_TILE_WIDTH", std::to_string(blockDimensions[0] + kernelDimensions[2] - 1));
  source.SetValue("INPUT_TILE_HEIGHT", std::to_string(blockDimensions[1] + kernelDimensions[1] - 1));

  auto createDestinationSetup =
  [&](bool accumulate) {
    if (accumulate) {
      source += R"(
  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), {{SCALAR_NAME}}>();
  cOutput.load(output);
  conv2d_op.run(activation, weights, cOutput);
  cOutput.store(output);
)";
    } else {
      source += R"(
  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), {{SCALAR_NAME}}>();
)";
      if (useBias) {
        source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      auto idx = cOutput.get_multidimensional_index(i);
      cOutput[i] = bias_buf[idx[0]];
    }
  }
)";
      } else {
        source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      cOutput[i] = 0;
    }
  }
)";
      }
      source += R"(
  conv2d_op.run(activation, weights, cOutput);
  cOutput.store(output);
)";
    }
  };

  auto createKernel =
  [&](const char* functionName, bool accumulate) {
    source += std::string("kernel void ") + functionName + R"((device {{SCALAR_NAME}} *activation_buf [[buffer(0)]],
                   device {{SCALAR_NAME}} *weights_buf [[buffer(1)]],
                   device {{SCALAR_NAME}} *output_buf [[buffer(2)]],)";
    if (!accumulate && useBias) {
      source += R"(
                   device const {{SCALAR_NAME}} *bias_buf [[buffer(3)]],)";
    }
    source += R"(
                   uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint batch = tgid.z / DEPTH;
  const uint output_slice = tgid.z % DEPTH;
  activation_buf += (((DEPTH + {{KERNEL_DEPTH}} - 1) * batch) + output_slice) * ((WIDTH + {{KERNEL_WIDTH}} - 1) * (HEIGHT + {{KERNEL_HEIGHT}} - 1) * {{INPUT_CHANNELS}});
  output_buf += ((DEPTH * batch) + output_slice) * (WIDTH * HEIGHT * {{OUTPUT_CHANNELS}});

  const int output_origin_x = int(tgid.x) * {{BLOCK_DIMENSIONS_WIDTH}};
  const int output_origin_y = int(tgid.y) * {{BLOCK_DIMENSIONS_HEIGHT}};
  if (output_origin_x >= int(WIDTH) || output_origin_y >= int(HEIGHT)) {
    return;
  }

  const int input_origin_x = output_origin_x;
  const int input_origin_y = output_origin_y;

  auto activation_base_tensor = tensor<device {{SCALAR_NAME}}, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>({{INPUT_CHANNELS}}, int(WIDTH) + {{KERNEL_WIDTH}} - 1, int(HEIGHT) + {{KERNEL_HEIGHT}} - 1, 1));
  auto output_base = tensor<device {{SCALAR_NAME}}, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>({{OUTPUT_CHANNELS}}, WIDTH, HEIGHT, 1));
  auto weights = tensor<device {{SCALAR_NAME}}, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>({{OUTPUT_CHANNELS}}, {{INPUT_CHANNELS}}, {{KERNEL_WIDTH}}, {{KERNEL_HEIGHT}}));
  constexpr auto descriptor = convolution2d_descriptor(
      int4({{OUTPUT_CHANNELS}}, {{BLOCK_DIMENSIONS_WIDTH}}, {{BLOCK_DIMENSIONS_HEIGHT}}, 1),
      int4({{INPUT_CHANNELS}}, {{INPUT_TILE_WIDTH}}, {{INPUT_TILE_HEIGHT}}, 1),
      int2({{KERNEL_WIDTH}}, {{KERNEL_HEIGHT}}),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(1, 1),
      int2(1, 1),
      1,
      false,
      convolution2d_descriptor::mode::)" + std::string(accumulate ? "multiply_accumulate" : "multiply") + R"();
  convolution2d<descriptor, execution_simdgroups<{{EXECUTION_SIMD_GROUPS}}>> conv2d_op;
  conv2d_op.set_offsets(int2(({{KERNEL_WIDTH}} - 1) / 2, ({{KERNEL_HEIGHT}} - 1) / 2));

  if (output_origin_x + {{BLOCK_DIMENSIONS_WIDTH}} <= int(WIDTH) &&
      output_origin_y + {{BLOCK_DIMENSIONS_HEIGHT}} <= int(HEIGHT)) {
    auto activation = activation_base_tensor.slice<{{INPUT_CHANNELS}}, {{INPUT_TILE_WIDTH}}, {{INPUT_TILE_HEIGHT}}, 1>(
        0,
        input_origin_x,
        input_origin_y,
        0);
    auto output = output_base.slice<{{OUTPUT_CHANNELS}}, {{BLOCK_DIMENSIONS_WIDTH}}, {{BLOCK_DIMENSIONS_HEIGHT}}, 1>(
        0,
        output_origin_x,
        output_origin_y,
        0);
)";
    createDestinationSetup(accumulate);
    source += R"(
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        input_origin_x,
        input_origin_y,
        0);
    auto output = output_base.slice(
        0,
        output_origin_x,
        output_origin_y,
        0);
)";
    createDestinationSetup(accumulate);
    source += R"(
  }
}
)";
  };

  createKernel("conv3d_multiply", false);
  createKernel("conv3d_multiply_accumulate", true);

  source += R"(
kernel void permute_oidhw_to_dhwio(device const {{SCALAR_NAME}} *source [[buffer(0)]],
                                   device {{SCALAR_NAME}} *destination [[buffer(1)]],
                                   constant uint &outputChannels [[buffer(2)]],
                                   constant uint &inputChannels [[buffer(3)]],
                                   constant uint &kernelDepth [[buffer(4)]],
                                   constant uint &kernelHeight [[buffer(5)]],
                                   constant uint &kernelWidth [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]])
{
  const uint elementCount = outputChannels * inputChannels * kernelDepth * kernelHeight * kernelWidth;
  if (gid >= elementCount) {
    return;
  }

  uint linear = gid;
  const uint oc = linear % outputChannels;
  linear /= outputChannels;
  const uint ic = linear % inputChannels;
  linear /= inputChannels;
  const uint kw = linear % kernelWidth;
  linear /= kernelWidth;
  const uint kh = linear % kernelHeight;
  linear /= kernelHeight;
  const uint kd = linear;

  const uint sourceIndex = ((((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth) + kw;
  destination[gid] = source[sourceIndex];
}
)";

  return source.ToString();
}
