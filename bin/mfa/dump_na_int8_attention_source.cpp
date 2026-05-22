#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#define private public
#include "nnc/mfa/kernels/CodeWriter.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernelDescriptor.hpp"
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

static NAInt8AttentionKernel* create_kernel(const NAInt8AttentionKernelDescriptor& kernel_descriptor)
{
  auto* kernel = reinterpret_cast<NAInt8AttentionKernel*>(::operator new(sizeof(NAInt8AttentionKernel)));
  kernel->blockDimensions = kernel_descriptor.blockDimensions;
  kernel->type = kernel_descriptor.type;
  kernel->headDimension = kernel_descriptor.headDimension;
  kernel->Hq = kernel_descriptor.Hq;
  kernel->Hk = kernel_descriptor.Hk;
  kernel->qScaleTileSize = kernel_descriptor.qScaleTileSize;
  kernel->kvScaleTileSize = kernel_descriptor.kvScaleTileSize;
  kernel->executionSIMDGroups = kernel_descriptor.executionSIMDGroups;
  kernel->vMeanThreads = kernel_descriptor.vMeanThreads;
  kernel->hasCRemainder = kernel_descriptor.hasCRemainder;
  kernel->threadBarrierEveryC = kernel_descriptor.threadBarrierEveryC;
  kernel->ioPrecision = kernel_descriptor.ioPrecision;
  kernel->lowPrecisionIntermediates = kernel_descriptor.lowPrecisionIntermediates;
  kernel->scale = kernel_descriptor.scale;
  kernel->isCausal = kernel_descriptor.isCausal;
  kernel->masked = kernel_descriptor.masked;
  kernel->isVarlen = kernel_descriptor.isVarlen;
  kernel->attentionSinks = kernel_descriptor.attentionSinks;
  kernel->hasCausalEmptyRows = kernel_descriptor.hasCausalEmptyRows;
  return kernel;
}

static std::string create_compute_d_source(const NAInt8AttentionKernel& kernel)
{
  CodeWriter source;
  source += shader_preamble();
  source.SetValue("V_MEAN_MEMORY_NAME", "float");
  kernel.createConstants(source);
  source += kernel.createComputeD();
  return source.ToString();
}

int main(int argc, char** argv)
{
  const uint32_t R = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 128;
  const uint32_t C = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 128;
  const uint32_t D = (argc > 3) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 64;
  const uint16_t Hq = (argc > 4) ? static_cast<uint16_t>(std::strtoul(argv[4], nullptr, 10)) : 8;
  const uint16_t Hk = (argc > 5) ? static_cast<uint16_t>(std::strtoul(argv[5], nullptr, 10)) : 8;
  const uint32_t batch = (argc > 6) ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 2;
  const bool bf16 = (argc > 7) ? (std::strtoul(argv[7], nullptr, 10) != 0) : false;
  const bool fp32 = (argc > 8) ? (std::strtoul(argv[8], nullptr, 10) != 0) : false;
  const bool is_causal = (argc > 9) ? (
      std::string(argv[9]) == "1" ||
      std::string(argv[9]) == "on" ||
      std::string(argv[9]) == "causal") : false;
  const bool masked = (argc > 10) ? (
      std::string(argv[10]) == "1" ||
      std::string(argv[10]) == "on" ||
      std::string(argv[10]) == "masked") : false;
  const bool is_varlen = (argc > 11) ? (
      std::string(argv[11]) == "1" ||
      std::string(argv[11]) == "on" ||
      std::string(argv[11]) == "varlen") : false;

  NAInt8AttentionDescriptor descriptor;
  descriptor.batchDimension = batch;
  descriptor.Hq = Hq;
  descriptor.Hk = Hk;
  descriptor.matrixDimensions = simd::uint3 { R, C, D };
  descriptor.batchStrides = AttentionOperands<unsigned int>();
  descriptor.scale = 1.0f / std::sqrt(static_cast<float>(D));
  descriptor.ioPrecision =
      fp32 ? GEMMOperandPrecision::FP32 :
      (bf16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16);
  descriptor.lowPrecisionIntermediates = !fp32;
  descriptor.isCausal = is_causal;
  descriptor.masked = masked;
  descriptor.isVarlen = is_varlen;

  descriptor.type = AttentionKernelType::forward;
  const auto forward_descriptor = descriptor.kernelDescriptor();
  auto* forward_kernel = create_kernel(forward_descriptor);

  descriptor.type = AttentionKernelType::backwardQuery;
  const auto query_descriptor = descriptor.kernelDescriptor();
  auto* query_kernel = create_kernel(query_descriptor);

  descriptor.type = AttentionKernelType::backwardKeyValue;
  const auto keyvalue_descriptor = descriptor.kernelDescriptor();
  auto* keyvalue_kernel = create_kernel(keyvalue_descriptor);

  const std::string precision =
      fp32 ? "FP32" :
      (bf16 ? "BF16" : "FP16");
  const std::string header =
      "// Generated from current NAInt8Attention source generator\n"
      "// Config: R=" + std::to_string(R) +
      " C=" + std::to_string(C) +
      " D=" + std::to_string(D) +
      " Hq=" + std::to_string(Hq) +
      " Hk=" + std::to_string(Hk) +
      " batch=" + std::to_string(batch) +
      " ioPrecision=" + precision +
      " isCausal=" + std::string(is_causal ? "1" : "0") +
      " masked=" + std::string(masked ? "1" : "0") +
      (is_varlen ? " isVarlen=1" : "") +
      " lowPrecisionIntermediates=" + std::string(descriptor.lowPrecisionIntermediates ? "true" : "false") + "\n\n";

  const std::string causal_suffix = is_causal ? "_causal" : "";
  const std::string masked_suffix = masked ? "_masked" : "";
  const std::string varlen_suffix = is_varlen ? "_varlen" : "";
  write_text_file("../../na_int8_attention_source" + causal_suffix + masked_suffix + varlen_suffix + "_current.metal", header + forward_kernel->createSource());
  write_text_file("../../na_int8_attention_compute_d_source_current.metal", header + create_compute_d_source(*query_kernel));
  write_text_file("../../na_int8_attention_backward_query_source_current.metal", header + query_kernel->createSource());
  write_text_file("../../na_int8_attention_backward_keyvalue_source_current.metal", header + keyvalue_kernel->createSource());

  ::operator delete(forward_kernel);
  ::operator delete(query_kernel);
  ::operator delete(keyvalue_kernel);
  return 0;
}
