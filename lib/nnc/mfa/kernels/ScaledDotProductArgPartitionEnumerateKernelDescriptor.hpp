#ifndef MFA_SCALEDDOTPRODUCTARGPARTITIONENUMERATEKERNELDESCRIPTOR_HPP_
#define MFA_SCALEDDOTPRODUCTARGPARTITIONENUMERATEKERNELDESCRIPTOR_HPP_

#include <cstddef>
#include <functional>

struct ScaledDotProductArgPartitionEnumerateKernelDescriptor {
  bool operator==(const ScaledDotProductArgPartitionEnumerateKernelDescriptor& rhs) const;
};

template<>
struct std::hash<ScaledDotProductArgPartitionEnumerateKernelDescriptor>
{
  std::size_t operator()(const ScaledDotProductArgPartitionEnumerateKernelDescriptor& hash) const noexcept;
};

#endif
