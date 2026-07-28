#include "SegmentedScaledGEMMPrologueDescriptor.hpp"
#include "SegmentedScaledGEMMPrologueKernelDescriptor.hpp"
#include "SegmentedScaledGEMMPrologueKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SegmentedScaledGEMMPrologueDescriptor::operator==(const SegmentedScaledGEMMPrologueDescriptor& rhs) const
{
  return
    simd_all(matrixDimensions == rhs.matrixDimensions) &&
    expertCount == rhs.expertCount &&
    binCount == rhs.binCount &&
    simd_all(blockDimensions == rhs.blockDimensions) &&
    ioPrecision == rhs.ioPrecision &&
    useBias == rhs.useBias &&
    threadgroupSize == rhs.threadgroupSize;
}

std::size_t std::hash<SegmentedScaledGEMMPrologueDescriptor>::operator()(const SegmentedScaledGEMMPrologueDescriptor& hash) const noexcept
{
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.expertCount);
  combine_32(seed, hash.binCount);
  combine_32(seed, hash.blockDimensions[0]);
  combine_32(seed, hash.blockDimensions[1]);
  combine_32(seed, hash.blockDimensions[2]);
  combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.ioPrecision.value, (uint16_t)hash.useBias }));
  combine_32(seed, hash.threadgroupSize);
  return seed;
}

std::pair<SegmentedScaledGEMMPrologueKernelDescriptor, PipelineValue<SegmentedScaledGEMMPrologueKernel> *> SegmentedScaledGEMMPrologueDescriptor::findKernel(
    MTL::Device *const device,
    const DeviceProperties &dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<SegmentedScaledGEMMPrologueKernelDescriptor, std::unique_ptr<SegmentedScaledGEMMPrologueKernel>> *const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  auto createKernel =
  [=](SegmentedScaledGEMMPrologueKernelDescriptor descriptor) -> SegmentedScaledGEMMPrologueKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    SegmentedScaledGEMMPrologueKernel* kernel = new SegmentedScaledGEMMPrologueKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<SegmentedScaledGEMMPrologueKernel>(kernel);
    return kernel;
  };

  auto createFunctionPipelineIndirect =
  [=](MTL::Library* library, SegmentedScaledGEMMPrologueKernel* kernel) -> std::tuple<NS::SharedPtr<MTL::Function>, NS::SharedPtr<MTL::ComputePipelineState>, NS::SharedPtr<MTL::IndirectCommandBuffer>> {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t binCount = this->binCount;
    const uint32_t N = this->matrixDimensions[0];
    const uint32_t K = this->matrixDimensions[1];
    const uint32_t expertCount = this->expertCount;
    const uint32_t MBlock = this->blockDimensions[0];
    const uint32_t NBlock = this->blockDimensions[1];
    const uint32_t threadgroupSize = this->threadgroupSize;
    constants->setConstantValue(&binCount, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&MBlock, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&NBlock, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&threadgroupSize, MTL::DataTypeUInt, NS::UInteger(5));
    constants->setConstantValue(&expertCount, MTL::DataTypeUInt, NS::UInteger(6));

    auto functionName = NS::String::string("segmented_scaled_gemm_prologue", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto icbDesc = NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
    icbDesc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
    icbDesc->setInheritPipelineState(false);
    icbDesc->setInheritBuffers(false);
    icbDesc->setMaxKernelBufferBindCount(this->useBias ? 7 : 6);
    auto indirect = NS::TransferPtr(device->newIndirectCommandBuffer(icbDesc.get(), binCount, MTL::ResourceStorageModePrivate));
    return std::make_tuple(function, pipeline, indirect);
  };

  auto kernelDesc = SegmentedScaledGEMMPrologueKernelDescriptor(ioPrecision, useBias);
  auto kernel = createKernel(kernelDesc);
  auto tuple = createFunctionPipelineIndirect(kernel->library.get(), kernel);
  auto function = std::get<0>(tuple);
  auto pipeline = std::get<1>(tuple);
  auto indirect = std::get<2>(tuple);
  PipelineValue<SegmentedScaledGEMMPrologueKernel>* output = new PipelineValue<SegmentedScaledGEMMPrologueKernel> { kernel, pipeline, indirect, function };
  return std::make_pair(kernelDesc, output);
}
