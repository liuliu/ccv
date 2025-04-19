#include "SegmentedGEMMPrologueDescriptor.hpp"
#include "SegmentedGEMMPrologueKernelDescriptor.hpp"
#include "SegmentedGEMMPrologueKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SegmentedGEMMPrologueDescriptor::operator==(const SegmentedGEMMPrologueDescriptor& rhs) const {
  return
  simd_all(matrixDimensions == rhs.matrixDimensions) &&
  simd_all(blockDimensions == rhs.blockDimensions) &&
  memoryPrecisions == rhs.memoryPrecisions &&
  threadgroupSize == rhs.threadgroupSize &&
  threadgroupMemoryAllocation == rhs.threadgroupMemoryAllocation &&
  useBias == rhs.useBias;
}

std::size_t std::hash<SegmentedGEMMPrologueDescriptor>::operator()(const SegmentedGEMMPrologueDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, hash.blockDimensions[0]);
  combine_32(seed, hash.blockDimensions[1]);
  combine_32(seed, hash.blockDimensions[2]);
  combine_64(seed, pack_64(simd::ushort4 { hash.memoryPrecisions.A.value, hash.memoryPrecisions.B.value, hash.memoryPrecisions.C.value, hash.memoryPrecisions.bias.value }));
  combine_32(seed, pack_32(simd::uchar4 { hash.useBias, 0, 0, 0 }));
  combine_32(seed, hash.threadgroupMemoryAllocation);
  combine_32(seed, hash.threadgroupSize);
  return seed;
}

std::pair<SegmentedGEMMPrologueKernelDescriptor, PipelineValue<SegmentedGEMMPrologueKernel> *> SegmentedGEMMPrologueDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, std::unordered_map<SegmentedGEMMPrologueKernelDescriptor, std::unique_ptr<SegmentedGEMMPrologueKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](SegmentedGEMMPrologueKernelDescriptor descriptor) -> SegmentedGEMMPrologueKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      SegmentedGEMMPrologueKernel* kernel = new SegmentedGEMMPrologueKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<SegmentedGEMMPrologueKernel>(kernel);
      return kernel;
    }
  };

  // WARNING: The owner must explicitly retain the compute pipeline.
  auto createPipeline =
  [=](MTL::Library* library, SegmentedGEMMPrologueKernel* kernel) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t M = this->matrixDimensions[0];
    uint32_t N = this->matrixDimensions[1];
    uint32_t K = this->matrixDimensions[2];
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&N, MTL::DataTypeUInt, 1);
    constants->setConstantValue(&K, MTL::DataTypeUInt, 2);

    uint32_t MBlock = this->blockDimensions[0];
    uint32_t NBlock = this->blockDimensions[1];
    constants->setConstantValue(&MBlock, MTL::DataTypeUInt, 3);
    constants->setConstantValue(&NBlock, MTL::DataTypeUInt, 4);

    uint32_t threadgroupSize = this->threadgroupSize;
    constants->setConstantValue(&threadgroupSize, MTL::DataTypeUInt, 5);

    uint32_t threadgroupMemoryAllocation = this->threadgroupMemoryAllocation;
    constants->setConstantValue(&threadgroupMemoryAllocation, MTL::DataTypeUInt, 6);

    NS::String* swiftName = NS::String::string("segmented_gemm_prologue", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    kernel->function = function;
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  auto kernelDesc = SegmentedGEMMPrologueKernelDescriptor(this->memoryPrecisions, this->useBias);
  SegmentedGEMMPrologueKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), kernel));
    
  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<SegmentedGEMMPrologueKernel>* output = new PipelineValue<SegmentedGEMMPrologueKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
