#include <cstdlib>
#include <iostream>

#define private public
#include "nnc/mfa/kernels/NAMatMulKernel.hpp"
#include "nnc/mfa/kernels/NAMatMulKernelDescriptor.hpp"
#undef private

int main(int argc, char** argv)
{
  const uint32_t M = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 10;
  const uint32_t N = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 64;
  const uint32_t K = (argc > 3) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 128;
  const uint16_t split_k = (argc > 4) ? static_cast<uint16_t>(std::strtoul(argv[4], nullptr, 10)) : 1;
  const bool use_bias = (argc > 5) ? (std::atoi(argv[5]) != 0) : true;
  const bool a_trans = (argc > 6) ? (std::atoi(argv[6]) != 0) : false;
  const bool b_trans = (argc > 7) ? (std::atoi(argv[7]) != 0) : true;
  (void)M;
  (void)N;
  (void)K;

  const GEMMOperandPrecisions memory_precisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  const GEMMOperandPrecisions register_precisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  const NAMatMulKernelDescriptor kernel_descriptor(
      simd::ushort3{128, 64, 64},
      memory_precisions,
      register_precisions,
      split_k,
      4,
      false,
      simd::uchar3{static_cast<uint8_t>(a_trans), static_cast<uint8_t>(b_trans), 0},
      use_bias,
      true,
      0,
      0);
  auto* kernel = reinterpret_cast<NAMatMulKernel*>(::operator new(sizeof(NAMatMulKernel)));
  kernel->blockDimensions = kernel_descriptor.blockDimensions;
  kernel->memoryPrecisions = kernel_descriptor.memoryPrecisions;
  kernel->registerPrecisions = kernel_descriptor.registerPrecisions;
  kernel->splitK = kernel_descriptor.splitK;
  kernel->executionSIMDGroups = kernel_descriptor.executionSIMDGroups;
  kernel->threadBarrierOverK = kernel_descriptor.threadBarrierOverK;
  kernel->transposeState = kernel_descriptor.transposeState;
  kernel->useBias = kernel_descriptor.useBias;
  kernel->loadM = kernel_descriptor.loadM;
  kernel->groupM = kernel_descriptor.groupM;
  kernel->groupN = kernel_descriptor.groupN;
  std::cout << kernel->createSource();
  ::operator delete(kernel);
  return 0;
}
