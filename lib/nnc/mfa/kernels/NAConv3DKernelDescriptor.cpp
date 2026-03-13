#include "NAConv3DKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAConv3DKernelDescriptor::operator==(const NAConv3DKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  simd_all(kernelDimensions == rhs.kernelDimensions) &&
  dataType == rhs.dataType &&
  inputChannels == rhs.inputChannels &&
  outputChannels == rhs.outputChannels &&
  useBias == rhs.useBias;
}

std::size_t std::hash<NAConv3DKernelDescriptor>::operator()(const NAConv3DKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, pack_32(simd::ushort2 { hash.blockDimensions[0], hash.blockDimensions[1] }));
  combine_64(seed, pack_64(simd_make_ushort4(hash.kernelDimensions, 0)));
  combine_64(seed, hash.dataType);
  combine_32(seed, hash.inputChannels);
  combine_32(seed, hash.outputChannels);
  combine_32(seed, pack_32(simd::uchar4 { hash.useBias, 0, 0, 0 }));
  return seed;
}

NAConv3DKernelDescriptor::NAConv3DKernelDescriptor(simd::ushort2 blockDimensions, simd::ushort3 kernelDimensions, uint64_t dataType, uint32_t inputChannels, uint32_t outputChannels, bool useBias) noexcept {
  this->blockDimensions = blockDimensions;
  this->kernelDimensions = kernelDimensions;
  this->dataType = dataType;
  this->inputChannels = inputChannels;
  this->outputChannels = outputChannels;
  this->useBias = useBias;
}
