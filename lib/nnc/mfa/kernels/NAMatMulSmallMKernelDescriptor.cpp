#include "NAMatMulSmallMKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAMatMulSmallMKernelDescriptor::operator==(const NAMatMulSmallMKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  memoryPrecisions == rhs.memoryPrecisions &&
  pack == rhs.pack &&
  executionSIMDGroups == rhs.executionSIMDGroups &&
  useBias == rhs.useBias &&
  loadM == rhs.loadM;
}

std::size_t std::hash<NAMatMulSmallMKernelDescriptor>::operator()(const NAMatMulSmallMKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_64(seed, pack_64(simd::ushort4 {
      hash.memoryPrecisions.A.value,
      hash.memoryPrecisions.B.value,
      hash.memoryPrecisions.C.value,
      hash.memoryPrecisions.bias.value }));
  combine_64(seed, pack_64(simd::ushort4 {
      hash.pack,
      hash.executionSIMDGroups,
      uint16_t(hash.useBias),
      uint16_t(hash.loadM) }));
  return seed;
}

NAMatMulSmallMKernelDescriptor::NAMatMulSmallMKernelDescriptor(
    simd::ushort3 blockDimensions,
    GEMMOperandPrecisions memoryPrecisions,
    uint16_t pack,
    uint16_t executionSIMDGroups,
    bool useBias,
    bool loadM) noexcept {
  this->blockDimensions = blockDimensions;
  this->memoryPrecisions = memoryPrecisions;
  this->pack = pack;
  this->executionSIMDGroups = executionSIMDGroups;
  this->useBias = useBias;
  this->loadM = loadM;
}
