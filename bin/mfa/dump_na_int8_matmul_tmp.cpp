#include <cstdlib>
#include <iostream>

#define private public
#include "nnc/mfa/kernels/NAInt8MatMulKernel.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.hpp"
#undef private

int main(int argc, char** argv)
{
  const uint32_t M = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 4096;
  const uint32_t N = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 4096;
  const uint32_t K = (argc > 3) ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4096;
  const uint16_t block_m = (argc > 4) ? static_cast<uint16_t>(std::strtoul(argv[4], nullptr, 10)) : 128;
  const uint16_t block_n = (argc > 5) ? static_cast<uint16_t>(std::strtoul(argv[5], nullptr, 10)) : 128;
  const uint16_t block_k = (argc > 6) ? static_cast<uint16_t>(std::strtoul(argv[6], nullptr, 10)) : 128;
  const uint16_t simdgroups = (argc > 7) ? static_cast<uint16_t>(std::strtoul(argv[7], nullptr, 10)) : 8;
  const uint16_t quant_threads = (argc > 8) ? static_cast<uint16_t>(std::strtoul(argv[8], nullptr, 10)) : 256;
  const uint32_t group_m = (argc > 9) ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : ((M >= 4096) ? 4096 : 0);
  const uint32_t group_n = (argc > 10) ? static_cast<uint32_t>(std::strtoul(argv[10], nullptr, 10)) : ((N >= 4096) ? 4096 : 0);

  const NAInt8MatMulKernelDescriptor kernel_descriptor(
      simd::ushort3 { block_m, block_n, block_k },
      simdgroups,
      GEMMOperandPrecision::FP16,
      false,
      quant_threads,
      group_m,
      group_n);

  auto* kernel = reinterpret_cast<NAInt8MatMulKernel*>(::operator new(sizeof(NAInt8MatMulKernel)));
  kernel->blockDimensions = kernel_descriptor.blockDimensions;
  kernel->executionSIMDGroups = kernel_descriptor.executionSIMDGroups;
  kernel->ioPrecision = kernel_descriptor.ioPrecision;
  kernel->useBias = kernel_descriptor.useBias;
  kernel->activationQuantizeThreads = kernel_descriptor.activationQuantizeThreads;
  kernel->groupM = kernel_descriptor.groupM;
  kernel->groupN = kernel_descriptor.groupN;

  std::cout << "// M=" << M
            << " N=" << N
            << " K=" << K
            << " blockM=" << block_m
            << " blockN=" << block_n
            << " blockK=" << block_k
            << " simdgroups=" << simdgroups
            << " quantThreads=" << quant_threads
            << " groupM=" << group_m
            << " groupN=" << group_n
            << '\n';
  std::cout << kernel->createSource();
  ::operator delete(kernel);
  return 0;
}
