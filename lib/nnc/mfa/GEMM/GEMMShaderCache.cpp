#include "GEMMShaderCache.hpp"
#include "../ccv_nnc_mfa_error.hpp"

std::unordered_map<GEMMKernelKey, GEMMKernel*> GEMMShaderCache::libraryCache = {};

std::unordered_map<GEMMKey, GEMMPipelineValue*> GEMMShaderCache::pipelineCache = {};

GEMMPipelineValue* GEMMShaderCache::fetchKernel(GEMMDescriptor gemmDesc) {
  // Perform the early return before anything with high latency.
  GEMMKey gemmKey(gemmDesc);
  {
    auto iterator = pipelineCache.find(gemmKey);
    if (iterator != pipelineCache.end()) {
      std::cout << "Pipeline cache hit." << std::endl;
      return iterator->second;
    } else {
      std::cout << "Pipeline cache miss." << std::endl;
    }
  }
  
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](GEMMKernelDescriptor descriptor) -> GEMMKernel* {
    CCV_NNC_MFA_PRECONDITION(descriptor.preferAsyncStore.has_value());
    
    GEMMKernelKey gemmKernelKey(descriptor);
    auto iterator = libraryCache.find(gemmKernelKey);
    if (iterator != libraryCache.end()) {
      std::cout << "Library cache hit." << std::endl;
      return iterator->second;
    } else {
      std::cout << "Library cache miss." << std::endl;
      
      GEMMKernel* kernel = new GEMMKernel(descriptor);
      libraryCache[gemmKernelKey] = kernel;
      return kernel;
    }
  };
  
  // Create a MTLDevice object, a function call with very high latency.
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  
  // WARNING: The owner must explicitly retain the compute pipeline.
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
      kernelDesc.preferAsyncStore.reset();
    } else {
      kernelDesc.preferAsyncStore = true;
    }
  }
  
  // Run a combinatorial search to find the correct value for
  // 'preferAsyncStore'.
  if (kernelDesc.preferAsyncStore.has_value()) {
    GEMMKernel* kernel = createKernel(kernelDesc);
    auto pipeline = NS::TransferPtr
    (createPipeline(kernel->library.get()));
    
    // Force the user to retrieve the return value from the cache. We ensure
    // the cache takes ownership, and the pointer doesn't become a zombie
    // object.
    GEMMPipelineValue* output = new GEMMPipelineValue { kernel, pipeline };
    pipelineCache[gemmKey] = output;
  } else {
    struct Candidate {
      GEMMKernelDescriptor kernelDesc;
      GEMMKernel* kernel;
      NS::SharedPtr<MTL::ComputePipelineState> pipeline;
    };
    std::vector<Candidate> candidates;
    
    for (int8_t candidateID = 0; candidateID < 4; ++candidateID) {
      simd::ushort3 blockDimensions;
      if (candidateID % 2 == 0) {
        blockDimensions = simd::ushort3 { 48, 48, 32 };
      } else {
        blockDimensions = simd::ushort3 { 48, 48, 40 };
      }
      
      bool preferAsyncStore;
      if (candidateID / 2 == 0) {
        preferAsyncStore = false;
      } else {
        preferAsyncStore = true;
      }
      
      // Set the data that's unique to this variant.
      auto newKernelDesc = kernelDesc;
      newKernelDesc.blockDimensions = blockDimensions;
      newKernelDesc.preferAsyncStore = preferAsyncStore;
      
      GEMMKernel* kernel = createKernel(newKernelDesc);
      auto pipeline = NS::TransferPtr
      (createPipeline(kernel->library.get()));
      
      Candidate candidate {
        .kernelDesc = newKernelDesc,
        .kernel = kernel,
        .pipeline = pipeline
      };
      candidates.push_back(candidate);
    }
    
    // Find the maximum occupancy.
    int64_t maximumOccupancy = -1;
    for (Candidate candidate : candidates) {
      int64_t occupancy = candidate.pipeline->maxTotalThreadsPerThreadgroup();
      maximumOccupancy = std::max(maximumOccupancy, occupancy);
    }
    
    // Remove all candidates that don't match this occupancy.
    {
      std::vector<Candidate> newCandidates;
      for (Candidate candidate : candidates) {
        int64_t occupancy = candidate.pipeline->maxTotalThreadsPerThreadgroup();
        if (occupancy != maximumOccupancy) {
          continue;
        }
        newCandidates.push_back(candidate);
      }
      candidates = newCandidates;
    }
    
    // Choose the highest-performing candidate.
    Candidate candidate = candidates[candidates.size() - 1];
    kernelDesc = candidate.kernelDesc;
    
    // Force the user to retrieve the return value from the cache. We ensure
    // the cache takes ownership, and the pointer doesn't become a zombie
    // object.
    GEMMPipelineValue* output = new GEMMPipelineValue {
      candidate.kernel, candidate.pipeline
    };
    pipelineCache[gemmKey] = output;
  }
  return pipelineCache[gemmKey];
}
