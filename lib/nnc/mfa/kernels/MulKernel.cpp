#include "MulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "CodeWriter.hpp"

MulKernel::MulKernel(MulKernelDescriptor descriptor, MTL::Device *const device) {
  value = descriptor.value;
  loadM = descriptor.loadM;
  channel_broadcast = descriptor.channel_broadcast;
  memoryPrecision = descriptor.memoryPrecision;
  source = createSource();
  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

unsigned short MulKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string MulKernel::createSource() const noexcept {
  CodeWriter source;
  source += createConstants() + "\n";
  source.SetValue("REAL", value == 2 ? "real" : "real4");
  for (int i = 0; i < 2; i++) {
    const bool channel_weight = channel_broadcast & (1u << i);
    source.SetValue("REAL" + std::to_string(i), value == 2 || channel_weight ? "real" : "real4");
    source.SetValue("INDEX" + std::to_string(i), !channel_broadcast ? "idx" :
      (channel_weight ? "(idx / channel_length) % channel_count" : "(idx / channel_length / channel_count) * channel_length + idx % channel_length"));
  }
  source.SetValue("LOAD_M", loadM ? "const device uint *loadM [[buffer(3)]]," : "");
  source.SetValue("COUNT", loadM ? "const uniform<uint> count = make_uniform(loadM[0]);" : "");
  source.SetValue("BOUNDS_CHECK", value == 0 ? "" : "if (idx >= count) return;");
  source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void mul(
  device const {{REAL0}} *src0 [[buffer(0)]],
  device const {{REAL1}} *src1 [[buffer(1)]],
  device {{REAL}} *destination [[buffer(2)]],
  {{LOAD_M}}
  uint3 tpig [[thread_position_in_grid]]
) {
  {{COUNT}}
  const uint idx = tpig.x;
  {{BOUNDS_CHECK}}
  destination[idx] = src0[{{INDEX0}}] * src1[{{INDEX1}}];
}
)";
  return source.ToString();
}

std::string MulKernel::createConstants() const noexcept {
  const std::string type = memoryPrecision == GEMMOperandPrecision::FP32 ? "float" :
    (memoryPrecision == GEMMOperandPrecision::BF16 ? "bfloat" : "half");
  std::string defines = "typedef " + type + " real;\n";
  if (value != 2)
    defines += "typedef " + type + "4 real4;\n";
  if (value != 0 && !loadM)
    defines += "constant uint count [[function_constant(0)]];\n";
  if (channel_broadcast) {
    defines += "constant uint channel_count [[function_constant(1)]];\n";
    defines += "constant uint channel_length [[function_constant(2)]];\n";
  }
  return defines;
}
