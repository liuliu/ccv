#include "NAInt8MatMulKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8MatMulKernelDescriptor::operator==(const NAInt8MatMulKernelDescriptor& rhs) const {
  return
      simd_all(blockDimensions == rhs.blockDimensions) &&
      executionSIMDGroups == rhs.executionSIMDGroups &&
      ioPrecision == rhs.ioPrecision &&
      useBias == rhs.useBias &&
      activationQuantizeThreads == rhs.activationQuantizeThreads &&
      groupM == rhs.groupM &&
      groupN == rhs.groupN;
}

std::size_t std::hash<NAInt8MatMulKernelDescriptor>::operator()(const NAInt8MatMulKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_32(seed, pack_32(simd::ushort2 { hash.executionSIMDGroups, (uint16_t)hash.ioPrecision.value }));
  combine_32(seed, pack_32(simd::ushort2 { hash.activationQuantizeThreads, (uint16_t)hash.useBias }));
  combine_32(seed, hash.groupM);
  combine_32(seed, hash.groupN);
  return seed;
}

NAInt8MatMulKernelDescriptor::NAInt8MatMulKernelDescriptor(
    simd::ushort3 blockDimensions,
    uint16_t executionSIMDGroups,
    GEMMOperandPrecision ioPrecision,
    bool useBias,
    uint16_t activationQuantizeThreads,
    uint32_t groupM,
    uint32_t groupN) noexcept
{
  this->blockDimensions = blockDimensions;
  this->executionSIMDGroups = executionSIMDGroups;
  this->ioPrecision = ioPrecision;
  this->useBias = useBias;
  this->activationQuantizeThreads = activationQuantizeThreads;
  this->groupM = groupM;
  this->groupN = groupN;
}
