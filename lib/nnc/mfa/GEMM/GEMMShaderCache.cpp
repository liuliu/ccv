#include "GEMMShaderCache.hpp"
#include "../ccv_nnc_mfa_error.hpp"

std::unordered_map<GEMMKernelKey, GEMMKernel*> GEMMShaderCache::libraryCache = {};

std::unordered_map<GEMMKey, GEMMPipelineValue*> GEMMShaderCache::pipelineCache = {};

GEMMPipelineValue* GEMMShaderCache::fetchKernel(GEMMDescriptor gemmDesc) {
  // Perform the early return before anything with high latency.
  {
    GEMMKey hash(gemmDesc);
    auto iterator = pipelineCache.find(hash);
    if (iterator != pipelineCache.end()) {
      std::cout << "Pipeline cache hit." << std::endl;
      return iterator->second;
    } else {
      std::cout << "Pipeline cache miss." << std::endl;
    }
  }
  
  // WARNING: Make sure to delete GEMM kernels that aren't referenced in the
  // cache. Perhaps return a flag saying "this kernel must be deleted if not
  // cached".
  //
  // This warning has not been properly addressed yet.
  auto createKernel =
  [=](GEMMKernelDescriptor descriptor) -> GEMMKernel* {
    CCV_NNC_MFA_PRECONDITION(descriptor.preferAsyncStore.has_value());
    
    GEMMKernelKey hash(descriptor);
    auto iterator = libraryCache.find(hash);
    if (iterator != libraryCache.end()) {
      std::cout << "Library cache hit." << std::endl;
      return iterator->second;
    } else {
      std::cout << "Library cache miss." << std::endl;
      return new GEMMKernel(descriptor);
    }
  };
  
  // Create a MTLDevice object, a function call with very high latency.
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  
  // WARNING: 'Transfer' the compute pipeline state, after you receive it.
  //
  // This warning has not been properly addressed yet.
  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t M = gemmDesc.matrixDimensions.value()[0];
    uint32_t N = gemmDesc.matrixDimensions.value()[1];
    uint32_t K = gemmDesc.matrixDimensions.value()[2];
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&N, MTL::DataTypeUInt, 1);
    constants->setConstantValue(&K, MTL::DataTypeUInt, 2);
    
    std::string cppName = "gemm";
    NS::String* swiftName = NS::String::string
    (cppName.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  
  // Set the device and examine the block dimensions.
  GEMMKernelDescriptor kernelDesc(gemmDesc);
  kernelDesc.device = device.get();
  if (device->supportsFamily(MTL::GPUFamily(1009))) {
    kernelDesc.preferAsyncStore = false;
  } else {
    CCV_NNC_MFA_PRECONDITION(kernelDesc.blockDimensions.has_value());
    auto blockDimensions = kernelDesc.blockDimensions.value();
    if (simd_all(blockDimensions == simd::ushort3 { 48, 48, 32 })) {
      kernelDesc.preferAsyncStore = std::make_optional<bool>();
    } else {
      kernelDesc.preferAsyncStore = true;
    }
  }
  
  // Run a combinatorial search to find the correct value for
  // 'preferAsyncStore'.
  return nullptr;
}
