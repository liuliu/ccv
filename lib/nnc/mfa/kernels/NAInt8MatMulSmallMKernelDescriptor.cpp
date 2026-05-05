#include "NAInt8MatMulSmallMKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8MatMulSmallMKernelDescriptor::operator==(const NAInt8MatMulSmallMKernelDescriptor& rhs) const {
  return
    simd_all(blockDimensions == rhs.blockDimensions) &&
    pack == rhs.pack &&
    executionSIMDGroups == rhs.executionSIMDGroups &&
    ioPrecision == rhs.ioPrecision &&
    useBias == rhs.useBias;
}

std::size_t std::hash<NAInt8MatMulSmallMKernelDescriptor>::operator()(const NAInt8MatMulSmallMKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_64(seed, pack_64(simd::ushort4 {
      hash.pack,
      hash.executionSIMDGroups,
      hash.ioPrecision.value,
      uint16_t(hash.useBias) }));
  return seed;
}

NAInt8MatMulSmallMKernelDescriptor::NAInt8MatMulSmallMKernelDescriptor(
    simd::ushort3 blockDimensions,
    uint16_t pack,
    uint16_t executionSIMDGroups,
    GEMMOperandPrecision ioPrecision,
    bool useBias) noexcept {
  this->blockDimensions = blockDimensions;
  this->pack = pack;
  this->executionSIMDGroups = executionSIMDGroups;
  this->ioPrecision = ioPrecision;
  this->useBias = useBias;
}
