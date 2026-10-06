#include "Conv3DKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool Conv3DKernelDescriptor::operator==(const Conv3DKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  simd_all(kernelDimensions == rhs.kernelDimensions) &&
  dataType == rhs.dataType &&
  inputChannels == rhs.inputChannels &&
  outputChannels == rhs.outputChannels &&
  paddingLeft == rhs.paddingLeft &&
  paddingRight == rhs.paddingRight &&
  paddingTop == rhs.paddingTop &&
  paddingBottom == rhs.paddingBottom &&
  useBias == rhs.useBias;
}

std::size_t std::hash<Conv3DKernelDescriptor>::operator()(const Conv3DKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  seed = combine_32(seed, pack_32(simd::ushort2 { hash.blockDimensions[0], hash.blockDimensions[1] }));
  seed = combine_64(seed, pack_64(simd_make_ushort4(hash.kernelDimensions, 0)));
  seed = combine_64(seed, hash.dataType);
  seed = combine_32(seed, hash.inputChannels);
  seed = combine_32(seed, hash.outputChannels);
  seed = combine_32(seed, hash.paddingLeft);
  seed = combine_32(seed, hash.paddingRight);
  seed = combine_32(seed, hash.paddingTop);
  seed = combine_32(seed, hash.paddingBottom);
  seed = combine_32(seed, pack_32(simd::uchar4 { hash.useBias, 0, 0, 0 }));
  return seed;
}

Conv3DKernelDescriptor::Conv3DKernelDescriptor(simd::ushort2 blockDimensions, simd::ushort3 kernelDimensions, uint64_t dataType, uint32_t inputChannels, uint32_t outputChannels, uint32_t paddingLeft, uint32_t paddingRight, uint32_t paddingTop, uint32_t paddingBottom, bool useBias) noexcept {
  this->blockDimensions = blockDimensions;
  this->kernelDimensions = kernelDimensions;
  this->dataType = dataType;
  this->inputChannels = inputChannels;
  this->outputChannels = outputChannels;
  this->paddingLeft = paddingLeft;
  this->paddingRight = paddingRight;
  this->paddingTop = paddingTop;
  this->paddingBottom = paddingBottom;
  this->useBias = useBias;
}
