#ifndef NARegisterAttentionKernel_hpp
#define NARegisterAttentionKernel_hpp

#include "NARegisterAttentionDescriptor.hpp"

struct NARegisterAttentionKernel {
  NS::SharedPtr<MTL::Library> library;

  NARegisterAttentionKernel(NARegisterAttentionKernelDescriptor descriptor, MTL::Device* device);
};

#endif
