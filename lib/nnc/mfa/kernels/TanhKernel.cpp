#include "TanhKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

TanhKernel::TanhKernel(TanhKernelDescriptor descriptor, MTL::Device* const device) {
  gradient = descriptor.gradient;
  value = descriptor.value;
  loadM = descriptor.loadM;
  memoryPrecision = descriptor.memoryPrecision;
  threadgroupSize = MTL::Size(256, 1, 1);
  source = createSource();

  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string TanhKernel::createSource() const noexcept {
  std::string shader = "#include <metal_stdlib>\nusing namespace metal;\n" + createConstants();
  const std::string real = value < 2 ? "real4" : "real";
  const std::string arithmetic = value < 2 ? "float4" : "float";
  shader += "\nkernel void tanh_soft_cap(\n";
  if (gradient)
    shader += "  device const " + real + "* g [[buffer(0)]],\n";
  shader += "  device const " + real + "* src [[buffer(" + std::to_string(gradient ? 1 : 0) + ")]],\n";
  shader += "  device " + real + "* destination [[buffer(" + std::to_string(gradient ? 2 : 1) + ")]],\n";
  if (loadM)
    shader += "  const device uint* loadM [[buffer(" + std::to_string(gradient ? 3 : 2) + ")]],\n";
  shader += "  uint3 tpig [[thread_position_in_grid]])\n{\n";
  if (loadM)
    shader += "  const uniform<uint> count = make_uniform(loadM[0]);\n";
  shader += "  const uint idx = tpig.x;\n";
  if (value != 0)
    shader += "  if (idx >= count)\n    return;\n";
  if (gradient) {
    // The source is the forward output y, d(soft_cap * tanh(x / soft_cap)) / dx = 1 - (y / soft_cap)^2.
    shader += "  const " + arithmetic + " y = (" + arithmetic + ")(src[idx]) / soft_cap;\n";
    shader += "  destination[idx] = (" + real + ")((" + arithmetic + ")g[idx] * (1.0f - y * y));\n";
  } else {
    // The fast tanh loses accuracy for large inputs, use the precise one like MLX does.
    shader += "  const " + arithmetic + " x = (" + arithmetic + ")(src[idx]);\n";
    shader += "  destination[idx] = (" + real + ")(soft_cap * precise::tanh(x / soft_cap));\n";
  }
  shader += "}\n";
  return shader;
}

std::string TanhKernel::createConstants() const noexcept {
  std::string defines = "typedef " + memoryPrecision.name() + (value < 2 ? "4 real4;\n" : " real;\n");
  if (value != 0 && !loadM)
    defines += "constant uint count [[function_constant(0)]];\n";
  defines += "constant float soft_cap [[function_constant(1)]];\n";
  return defines;
}
