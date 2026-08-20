#include "ReduceMaxDescriptor.hpp"
#include "ReduceMaxKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool ReduceMaxKernelDescriptor::operator==(const ReduceMaxKernelDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision;
}

bool ReduceMaxDescriptor::operator==(const ReduceMaxDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision &&
    columnCount == rhs.columnCount &&
    partitionSize == rhs.partitionSize &&
    partitionCount == rhs.partitionCount &&
    partitioned == rhs.partitioned;
}

std::size_t std::hash<ReduceMaxKernelDescriptor>::operator()(const ReduceMaxKernelDescriptor& value) const noexcept
{
  return std::hash<int>()(value.memoryPrecision.value);
}

std::size_t std::hash<ReduceMaxDescriptor>::operator()(const ReduceMaxDescriptor& value) const noexcept
{
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 {
    static_cast<uint32_t>(value.memoryPrecision.value),
    value.columnCount,
  }));
  combine_64(seed, pack_64(simd::uint2 {
    value.partitionSize,
    value.partitionCount,
  }));
  combine_32(seed, value.partitioned ? 1u : 0u);
  return seed;
}

std::pair<ReduceMaxKernelDescriptor, PipelineValue<ReduceMaxKernel>*> ReduceMaxDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ReduceMaxKernelDescriptor, std::unique_ptr<ReduceMaxKernel>>* const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  const ReduceMaxKernelDescriptor kernelDesc = {
    .memoryPrecision = memoryPrecision,
  };
  ReduceMaxKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    auto newKernel = std::make_unique<ReduceMaxKernel>(kernelDesc, device);
    kernel = newKernel.get();
    (*libraryCache)[kernelDesc] = std::move(newKernel);
  }

  const auto createPipeline =
  [=](const char* const name) -> NS::SharedPtr<MTL::ComputePipelineState> {
    auto functionName = NS::String::string(name, NS::UTF8StringEncoding);
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&columnCount, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&partitionSize, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&partitionCount, MTL::DataTypeUInt, NS::UInteger(2));
    NS::Error* error = nil;
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto* const output = new PipelineValue<ReduceMaxKernel> {
    kernel,
    createPipeline(partitioned ? "reduce_max_partition" : "reduce_max_one_pass"),
  };
  if (partitioned) {
    output->second = createPipeline("reduce_max_merge_partitions");
  }
  return std::make_pair(kernelDesc, output);
}
