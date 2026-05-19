#include "SegmentedScaledGEMMDescriptor.hpp"
#include "SegmentedScaledGEMMKernelDescriptor.hpp"
#include "SegmentedScaledGEMMKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

namespace {

static void serializeBinaries(MTL::BinaryArchive* const binaryArchive, const std::string& pathToWrite) noexcept
{
  NS::Error* error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
}

}

bool SegmentedScaledGEMMDescriptor::operator==(const SegmentedScaledGEMMDescriptor& rhs) const
{
  return
    ioPrecision == rhs.ioPrecision &&
    simd_all(matrixDimensions == rhs.matrixDimensions) &&
    useBias == rhs.useBias;
}

std::size_t std::hash<SegmentedScaledGEMMDescriptor>::operator()(const SegmentedScaledGEMMDescriptor& hash) const noexcept
{
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, (uint32_t)hash.ioPrecision.value);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, hash.matrixDimensions[3]);
  combine_32(seed, hash.useBias ? 1 : 0);
  return seed;
}

SegmentedScaledGEMMKernelDescriptor SegmentedScaledGEMMDescriptor::kernelDescriptor() const noexcept
{
  return SegmentedScaledGEMMKernelDescriptor(
      simd::ushort3 { 128, 128, 128 },
      8,
      ioPrecision,
      useBias);
}

std::pair<SegmentedScaledGEMMKernelDescriptor, PipelineValue<SegmentedScaledGEMMKernel>*> SegmentedScaledGEMMDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<SegmentedScaledGEMMKernelDescriptor, std::unique_ptr<SegmentedScaledGEMMKernel>>* const libraryCache) const noexcept
{
  (void)dprops;

  auto createKernel =
  [=](const SegmentedScaledGEMMKernelDescriptor& descriptor) -> SegmentedScaledGEMMKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    SegmentedScaledGEMMKernel* kernel = new SegmentedScaledGEMMKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<SegmentedScaledGEMMKernel>(kernel);
    return kernel;
  };

  auto createPipeline =
  [=](SegmentedScaledGEMMKernel* kernel, const char* functionNameString) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t originalM = matrixDimensions[0];
    const uint32_t N = matrixDimensions[1];
    const uint32_t K = matrixDimensions[2];
    const uint32_t segments = matrixDimensions[3];
    const uint32_t MBlock = kernel->blockDimensions[0];
    const uint32_t NBlock = kernel->blockDimensions[1];
    const uint32_t KBlock = kernel->blockDimensions[2];
    const uint32_t maxRecords = kernel->maxTileRecords(originalM, segments);
    constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&segments, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&MBlock, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&NBlock, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&KBlock, MTL::DataTypeUInt, NS::UInteger(5));
    constants->setConstantValue(&maxRecords, MTL::DataTypeUInt, NS::UInteger(6));
    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDescriptor->setComputeFunction(function.get());
    MTL::ComputePipelineState* pipeline = nullptr;
    if (binaryArchivesToRead) {
      pipelineDescriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      if (binaryArchiveToWrite != nullptr) {
        binaryArchiveToWrite->addComputePipelineFunctions(pipelineDescriptor.get(), &error);
        serializeBinaries(binaryArchiveToWrite, pathToWrite);
      }
    }
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto kernelDesc = kernelDescriptor();
  auto kernel = createKernel(kernelDesc);
  auto matmul = NS::TransferPtr(createPipeline(kernel, "segmented_scaled_gemm"));
  auto plan = NS::TransferPtr(createPipeline(kernel, "segmented_scaled_gemm_plan"));
  PipelineValue<SegmentedScaledGEMMKernel>* output = new PipelineValue<SegmentedScaledGEMMKernel> { kernel, matmul };
  output->second = plan;
  return std::make_pair(kernelDesc, output);
}
