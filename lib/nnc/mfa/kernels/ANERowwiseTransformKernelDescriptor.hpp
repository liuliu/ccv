#ifndef ANERowwiseTransformKernelDescriptor_hpp
#define ANERowwiseTransformKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"

struct ANERowwiseTransformKernelDescriptor {
  GEMMOperandPrecision memoryPrecision;

  ANERowwiseTransformKernelDescriptor() = delete;
  explicit ANERowwiseTransformKernelDescriptor(GEMMOperandPrecision memoryPrecision) noexcept;

  bool operator==(const ANERowwiseTransformKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ANERowwiseTransformKernelDescriptor>
{
  std::size_t operator()(const ANERowwiseTransformKernelDescriptor& hash) const noexcept;
};

#endif
