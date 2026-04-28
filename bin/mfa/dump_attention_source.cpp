#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define private public
#include "nnc/mfa/kernels/CodeWriter.hpp"
#include "nnc/mfa/kernels/AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernel.hpp"
#include "nnc/mfa/kernels/AttentionKernelDescriptor.hpp"
#undef private

static std::string shader_preamble()
{
  return R"(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_math>
#include <metal_geometric>

using namespace metal;

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

static AttentionKernel* create_kernel(const AttentionKernelDescriptor& kernel_descriptor)
{
  auto* kernel = reinterpret_cast<AttentionKernel*>(::operator new(sizeof(AttentionKernel)));
  kernel->blockDimensions = kernel_descriptor.blockDimensions;
  kernel->cacheState = kernel_descriptor.cacheState;
  kernel->headDimension = kernel_descriptor.headDimension;
  kernel->isCausal = kernel_descriptor.isCausal;
  kernel->masked = kernel_descriptor.masked;
  kernel->isVarlen = kernel_descriptor.isVarlen;
  kernel->memoryPrecisions = kernel_descriptor.memoryPrecisions;
  kernel->preferAsyncCache = kernel_descriptor.preferAsyncCache;
  kernel->preferAsyncLoad = kernel_descriptor.preferAsyncLoad;
  kernel->registerPrecisions = kernel_descriptor.registerPrecisions;
  kernel->transposeState = kernel_descriptor.transposeState;
  kernel->leadingDimensions = kernel_descriptor.leadingDimensions;
  kernel->type = kernel_descriptor.type;
  kernel->disableAsyncCopy = false;
  kernel->threadgroupSize = 32 * (kernel->blockDimensions[0] / 8);
  kernel->threadgroupMemoryAllocation = kernel->createThreadgroupMemoryAllocation();
  return kernel;
}

int main(int argc, char** argv)
{
  const uint32_t R = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 4096;
  const uint32_t C = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 4096;
  const uint32_t D = (argc > 3) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 128;
  const uint16_t Hq = (argc > 4) ? static_cast<uint16_t>(std::strtoul(argv[4], nullptr, 10)) : 32;
  const uint16_t Hk = (argc > 5) ? static_cast<uint16_t>(std::strtoul(argv[5], nullptr, 10)) : 32;
  const uint32_t batch = (argc > 6) ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 1;
  const char* precision = (argc > 7) ? argv[7] : "fp16";
  const bool low_precision_inputs = std::strcmp(precision, "fp32") != 0;
  const bool is_bf16 = std::strcmp(precision, "bf16") == 0;
  const bool low_precision_intermediates = (argc > 8) ? (std::atoi(argv[8]) != 0) : low_precision_inputs;
  const bool use_leading_dimensions = (argc > 9) ? (std::atoi(argv[9]) != 0) : false;
  const char* causal_arg = (argc > 10) ? argv[10] : "off";
  const bool is_causal = std::strcmp(causal_arg, "1") == 0 ||
      std::strcmp(causal_arg, "on") == 0 ||
      std::strcmp(causal_arg, "causal") == 0;
  const char* masked_arg = (argc > 11) ? argv[11] : "off";
  const bool masked = std::strcmp(masked_arg, "1") == 0 ||
      std::strcmp(masked_arg, "on") == 0 ||
      std::strcmp(masked_arg, "masked") == 0;
  const char* varlen_arg = (argc > 12) ? argv[12] : "off";
  const bool is_varlen = std::strcmp(varlen_arg, "1") == 0 ||
      std::strcmp(varlen_arg, "on") == 0 ||
      std::strcmp(varlen_arg, "varlen") == 0;
  if (is_varlen && masked) {
    std::cerr << "generic Attention varlen does not support mask\n";
    return 1;
  }

  DeviceProperties dprops = DeviceProperties();
  AttentionDescriptor descriptor;
  descriptor.batchDimension = batch;
  descriptor.Hq = Hq;
  descriptor.Hk = Hk;
  descriptor.lowPrecisionInputs = low_precision_inputs;
  descriptor.isBF16 = is_bf16;
  descriptor.lowPrecisionIntermediates = low_precision_intermediates;
  descriptor.matrixDimensions = simd::uint3 { R, C, D };
  descriptor.transposeState = simd::uchar4 { 0, 0, 0, 0 };
  descriptor.batchStrides = AttentionOperands<unsigned int>();
  descriptor.scale = 1.0f / std::sqrt(static_cast<float>(D));
  descriptor.isCausal = is_causal;
  descriptor.masked = masked;
  descriptor.isVarlen = is_varlen;
  if (use_leading_dimensions)
    descriptor.leadingDimensions = simd::uint4 { Hq * D, Hk * D, Hk * D, Hq * D };

  descriptor.type = AttentionKernelType::forward;
  auto forward_descriptor = descriptor.kernelDescriptor(nullptr, dprops);
  auto* forward_kernel = create_kernel(forward_descriptor);

  descriptor.type = AttentionKernelType::backwardQuery;
  auto query_descriptor = descriptor.kernelDescriptor(nullptr, dprops);
  auto* query_kernel = create_kernel(query_descriptor);

  descriptor.type = AttentionKernelType::backwardKeyValue;
  auto keyvalue_descriptor = descriptor.kernelDescriptor(nullptr, dprops);
  auto* keyvalue_kernel = create_kernel(keyvalue_descriptor);

  const std::string header =
      "// Generated from current generic Attention source generator\n"
      "// Config: R=" + std::to_string(R) +
      " C=" + std::to_string(C) +
      " D=" + std::to_string(D) +
      " Hq=" + std::to_string(Hq) +
      " Hk=" + std::to_string(Hk) +
      " batch=" + std::to_string(batch) +
      " ioPrecision=" + std::string(low_precision_inputs ? (is_bf16 ? "BF16" : "FP16") : "FP32") +
      " lowPrecisionIntermediates=" + std::string(low_precision_intermediates ? "1" : "0") +
      " useLeadingDimensions=" + std::string(use_leading_dimensions ? "1" : "0") +
      " isCausal=" + std::string(is_causal ? "1" : "0") +
      " masked=" + std::string(masked ? "1" : "0") +
      std::string(is_varlen ? " isVarlen=1" : "") +
      "\n\n";

  const std::string suffix =
      (is_causal ? "_causal" : "") +
      std::string(masked ? "_masked" : "") +
      std::string(is_varlen ? "_varlen" : "");
  write_text_file("../../attention_forward_source" + suffix + "_current.metal", header + shader_preamble() + forward_kernel->createSource());
  write_text_file("../../attention_backward_query_source_current.metal", header + shader_preamble() + query_kernel->createSource());
  write_text_file("../../attention_backward_keyvalue_source_current.metal", header + shader_preamble() + keyvalue_kernel->createSource());

  ::operator delete(forward_kernel);
  ::operator delete(query_kernel);
  ::operator delete(keyvalue_kernel);
  return 0;
}
