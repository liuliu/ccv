#ifndef MFA_PIPELINE_VALUE_HPP_
#define MFA_PIPELINE_VALUE_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

template<typename T>
struct PipelineValue {
  T* kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

#endif
