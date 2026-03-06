#include "NAMatMulKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

bool NAMatMulKernelDescriptor::operator==(const NAMatMulKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  memoryPrecisions == rhs.memoryPrecisions &&
  registerPrecisions == rhs.registerPrecisions &&
  (splitK == rhs.splitK) &&
  (executionSIMDGroups == rhs.executionSIMDGroups) &&
  simd_all(transposeState == rhs.transposeState) &&
  (useBias == rhs.useBias) &&
  (loadM == rhs.loadM);
}

std::size_t std::hash<NAMatMulKernelDescriptor>::operator()(const NAMatMulKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_64(seed, pack_64(simd::ushort4 { hash.memoryPrecisions.A.value, hash.memoryPrecisions.B.value, hash.memoryPrecisions.C.value, hash.memoryPrecisions.bias.value }));
  combine_64(seed, pack_64(simd::ushort4 { hash.registerPrecisions.A.value, hash.registerPrecisions.B.value, hash.registerPrecisions.C.value, hash.registerPrecisions.bias.value }));
  combine_32(seed, pack_32(simd::ushort2 { hash.splitK, hash.executionSIMDGroups }));
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], hash.transposeState[2], hash.useBias }));
  combine_32(seed, pack_32(simd::uchar4 { hash.loadM, 0, 0, 0 }));
  return seed;
}

// MARK: - Initializer

NAMatMulKernelDescriptor::NAMatMulKernelDescriptor(simd::ushort3 blockDimensions, GEMMOperandPrecisions memoryPrecisions, GEMMOperandPrecisions registerPrecisions, uint16_t splitK, uint16_t executionSIMDGroups, simd::uchar3 transposeState, bool useBias, bool loadM) noexcept {
  this->blockDimensions = blockDimensions;
  this->memoryPrecisions = memoryPrecisions;
  this->registerPrecisions = registerPrecisions;
  this->splitK = splitK;
  this->executionSIMDGroups = executionSIMDGroups;
  this->transposeState = transposeState;
  this->useBias = useBias;
  this->loadM = loadM;
}

