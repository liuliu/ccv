#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define private public
#include "nnc/mfa/kernels/CodeWriter.hpp"
#include "nnc/mfa/kernels/NAAttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAAttentionKernel.hpp"
#include "nnc/mfa/kernels/NAAttentionKernelDescriptor.hpp"
#undef private

static std::string shader_preamble()
{
  return R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";
}

static void write_text_file(const std::string& path, const std::string& contents)
{
  std::ofstream stream(path);
  if (!stream.good()) {
    std::cerr << "cannot open " << path << " for writing\n";
    std::exit(1);
  }
  stream << contents;
  if (!stream.good()) {
    std::cerr << "cannot write " << path << "\n";
    std::exit(1);
  }
}

enum class BypassMode {
  Auto,
  Disable,
  Enable,
};

static NAAttentionKernel* create_kernel(const NAAttentionKernelDescriptor& kernel_descriptor)
{
  auto* kernel = reinterpret_cast<NAAttentionKernel*>(::operator new(sizeof(NAAttentionKernel)));
  kernel->type = kernel_descriptor.type;
  kernel->scale = kernel_descriptor.scale;
  kernel->memoryPrecisions = kernel_descriptor.memoryPrecisions;
  kernel->blockDimensions = kernel_descriptor.blockDimensions;
  kernel->headDimension = kernel_descriptor.headDimension;
  kernel->Hq = kernel_descriptor.Hq;
  kernel->Hk = kernel_descriptor.Hk;
  kernel->executionSIMDGroups = kernel_descriptor.executionSIMDGroups;
  kernel->bypassThreadgroupMemory = kernel_descriptor.bypassThreadgroupMemory;
  kernel->checkCEdge1 = kernel_descriptor.checkCEdge1;
  kernel->isCausal = kernel_descriptor.isCausal;
  return kernel;
}

static std::string create_compute_d_source(const NAAttentionKernel& kernel)
{
  CodeWriter source;
  source += shader_preamble();
  kernel.createConstants(source);
  source += kernel.createComputeD();
  return source.ToString();
}

int main(int argc, char** argv)
{
  const uint32_t R = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 8192;
  const uint32_t C = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 8192;
  const uint32_t D = (argc > 3) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 128;
  const uint16_t Hq = (argc > 4) ? static_cast<uint16_t>(std::strtoul(argv[4], nullptr, 10)) : 32;
  const uint16_t Hk = (argc > 5) ? static_cast<uint16_t>(std::strtoul(argv[5], nullptr, 10)) : 32;
  const uint32_t batch = (argc > 6) ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 1;
  const char* precision = (argc > 7) ? argv[7] : "fp16";
  const bool low_precision_inputs = std::strcmp(precision, "fp32") != 0;
  const bool is_bf16 = std::strcmp(precision, "bf16") == 0;
  const bool low_precision_intermediates = (argc > 8) ? (std::atoi(argv[8]) != 0) : false;
  const char* bypass_arg = (argc > 9) ? argv[9] : "auto";
  const char* causal_arg = (argc > 10) ? argv[10] : "off";
  const bool is_causal = std::strcmp(causal_arg, "1") == 0 ||
      std::strcmp(causal_arg, "on") == 0 ||
      std::strcmp(causal_arg, "causal") == 0;
  BypassMode bypass_mode = BypassMode::Auto;
  if (std::strcmp(bypass_arg, "0") == 0 || std::strcmp(bypass_arg, "off") == 0 || std::strcmp(bypass_arg, "disable") == 0) {
    bypass_mode = BypassMode::Disable;
  } else if (std::strcmp(bypass_arg, "1") == 0 || std::strcmp(bypass_arg, "on") == 0 || std::strcmp(bypass_arg, "enable") == 0) {
    bypass_mode = BypassMode::Enable;
  }

  DeviceProperties dprops = DeviceProperties();
  NAAttentionDescriptor descriptor = {
    .batchDimension = batch,
    .Hq = Hq,
    .Hk = Hk,
    .lowPrecisionInputs = low_precision_inputs,
    .isBF16 = is_bf16,
    .lowPrecisionIntermediates = low_precision_intermediates,
    .matrixDimensions = simd::uint3 { R, C, D },
    .batchStrides = AttentionOperands<unsigned int>(),
    .type = AttentionKernelType::backwardQuery,
    .scale = 1.0f / std::sqrt(static_cast<float>(D)),
    .isCausal = is_causal,
  };

  descriptor.type = AttentionKernelType::forward;
  auto forward_descriptor = descriptor.kernelDescriptor(nullptr, dprops);
  auto* forward_kernel = create_kernel(forward_descriptor);

  descriptor.type = AttentionKernelType::backwardQuery;
  auto query_descriptor = descriptor.kernelDescriptor(nullptr, dprops);
  if (bypass_mode == BypassMode::Disable)
    query_descriptor.bypassThreadgroupMemory = false;
  else if (bypass_mode == BypassMode::Enable)
    query_descriptor.bypassThreadgroupMemory = true;
  auto* query_kernel = create_kernel(query_descriptor);

  descriptor.type = AttentionKernelType::backwardKeyValue;
  auto keyvalue_descriptor = descriptor.kernelDescriptor(nullptr, dprops);
  if (bypass_mode == BypassMode::Disable)
    keyvalue_descriptor.bypassThreadgroupMemory = false;
  else if (bypass_mode == BypassMode::Enable)
    keyvalue_descriptor.bypassThreadgroupMemory = true;
  auto* keyvalue_kernel = create_kernel(keyvalue_descriptor);

  const std::string suffix =
      (bypass_mode == BypassMode::Enable) ? "bypass" :
      (bypass_mode == BypassMode::Disable) ? "shared" :
      "auto";
  const std::string causal_suffix =
      is_causal ? "_causal" : "";

  const std::string header =
      "// Generated from current NAAttention backward source generator\n"
      "// Config: R=" + std::to_string(R) +
      " C=" + std::to_string(C) +
      " D=" + std::to_string(D) +
      " Hq=" + std::to_string(Hq) +
      " Hk=" + std::to_string(Hk) +
      " batch=" + std::to_string(batch) +
      " ioPrecision=" + std::string(low_precision_inputs ? (is_bf16 ? "BF16" : "FP16") : "FP32") +
      " lowPrecisionIntermediates=" + std::string(low_precision_intermediates ? "1" : "0") +
      " isCausal=" + std::string(is_causal ? "1" : "0") +
      " blockR=" + std::to_string(forward_descriptor.blockDimensions[0]) +
      " blockC=" + std::to_string(forward_descriptor.blockDimensions[1]) +
      " executionSIMDGroups=" + std::to_string(forward_descriptor.executionSIMDGroups) +
      " bypassThreadgroupMemory=" + suffix +
      "\n\n";

  write_text_file("../../na_attention_source_" + suffix + causal_suffix + "_current.metal", header + forward_kernel->createSource());
  write_text_file("../../na_attention_compute_d_source_" + suffix + "_current.metal", header + create_compute_d_source(*query_kernel));
  write_text_file("../../na_attention_backward_query_source_" + suffix + "_current.metal", header + query_kernel->createSource());
  write_text_file("../../na_attention_backward_keyvalue_source_" + suffix + "_current.metal", header + keyvalue_kernel->createSource());

  ::operator delete(forward_kernel);
  ::operator delete(query_kernel);
  ::operator delete(keyvalue_kernel);
  return 0;
}
