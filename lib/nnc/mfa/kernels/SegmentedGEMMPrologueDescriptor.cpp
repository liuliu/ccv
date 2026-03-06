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
  dispatchMMajor == rhs.dispatchMMajor &&
  splitK == rhs.splitK &&
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
  combine_32(seed, pack_32(simd::uchar4 { hash.useBias, hash.dispatchMMajor, 0, 0 }));
  combine_32(seed, hash.threadgroupMemoryAllocation);
  combine_32(seed, hash.threadgroupSize);
  return seed;
}

std::pair<SegmentedGEMMPrologueKernelDescriptor, PipelineValue<SegmentedGEMMPrologueKernel> *> SegmentedGEMMPrologueDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SegmentedGEMMPrologueKernelDescriptor, std::unique_ptr<SegmentedGEMMPrologueKernel>> *const libraryCache) const noexcept {
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
  auto createFunctionPipelineIndirect =
  [=](MTL::Library* library, SegmentedGEMMPrologueKernel* kernel) -> std::tuple<NS::SharedPtr<MTL::Function>, NS::SharedPtr<MTL::ComputePipelineState>, NS::SharedPtr<MTL::IndirectCommandBuffer>, NS::SharedPtr<MTL::IndirectCommandBuffer>> {
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

    bool dispatchMMajor = this->dispatchMMajor;
    constants->setConstantValue(&dispatchMMajor, MTL::DataTypeBool, 7);

    NS::String* swiftName = NS::String::string("segmented_gemm_prologue", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto icbDesc = NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
    icbDesc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
    icbDesc->setInheritPipelineState(false);
    icbDesc->setInheritBuffers(false);
    icbDesc->setMaxKernelBufferBindCount(5);
    auto indirectCommandBuffer1 = NS::TransferPtr(device->newIndirectCommandBuffer(icbDesc.get(), M, MTL::ResourceStorageModePrivate));
    auto indirectCommandBuffer2 = this->splitK > 1 ? NS::TransferPtr(device->newIndirectCommandBuffer(icbDesc.get(), M, MTL::ResourceStorageModePrivate)) : NS::SharedPtr<MTL::IndirectCommandBuffer>();
    return std::make_tuple(function, pipeline, indirectCommandBuffer1, indirectCommandBuffer2);
  };
  auto kernelDesc = SegmentedGEMMPrologueKernelDescriptor(this->memoryPrecisions, this->useBias, this->splitK);
  SegmentedGEMMPrologueKernel* kernel = createKernel(kernelDesc);
  auto tuple = createFunctionPipelineIndirect(kernel->library.get(), kernel);
  auto function = std::get<0>(tuple);
  auto pipeline = std::get<1>(tuple);
  auto indirect1 = std::get<2>(tuple);
  auto indirect2 = std::get<3>(tuple);

  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<SegmentedGEMMPrologueKernel>* output = new PipelineValue<SegmentedGEMMPrologueKernel> { kernel, pipeline, indirect1, function, NS::SharedPtr<MTL::ComputePipelineState>(), indirect2 };
  return std::make_pair(kernelDesc, output);
}
