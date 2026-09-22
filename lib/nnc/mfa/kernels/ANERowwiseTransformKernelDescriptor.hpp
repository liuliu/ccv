#ifndef ANERowwiseTransformKernelDescriptor_hpp
#define ANERowwiseTransformKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"

struct ANERowwiseTransformKernelDescriptor {
  GEMMOperandPrecision memoryPrecision;
  bool activationHadamard256 = false;
  bool supportsApple10;

  ANERowwiseTransformKernelDescriptor() = delete;
  ANERowwiseTransformKernelDescriptor(
      GEMMOperandPrecision memoryPrecision,
      bool supportsApple10,
      bool activationHadamard256 = false) noexcept;

  bool operator==(const ANERowwiseTransformKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ANERowwiseTransformKernelDescriptor>
{
  std::size_t operator()(const ANERowwiseTransformKernelDescriptor& hash) const noexcept;
};

#endif
