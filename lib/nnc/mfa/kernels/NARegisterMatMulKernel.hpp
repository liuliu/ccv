#ifndef NARegisterMatMulKernel_hpp
#define NARegisterMatMulKernel_hpp

#include "NARegisterMatMulDescriptor.hpp"

struct NARegisterMatMulKernel {
  NS::SharedPtr<MTL::Library> library;

  NARegisterMatMulKernel(NARegisterMatMulKernelDescriptor descriptor, MTL::Device* device);
};

#endif
