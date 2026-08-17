#include "ArgmaxDescriptor.hpp"
#include "ArgmaxKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool ArgmaxKernelDescriptor::operator==(const ArgmaxKernelDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision;
}

bool ArgmaxDescriptor::operator==(const ArgmaxDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision &&
    columnCount == rhs.columnCount &&
    partitionSize == rhs.partitionSize &&
    partitionCount == rhs.partitionCount &&
    (!gumbel || scale == rhs.scale) &&
    gumbel == rhs.gumbel &&
    partitioned == rhs.partitioned;
}

std::size_t std::hash<ArgmaxKernelDescriptor>::operator()(const ArgmaxKernelDescriptor& value) const noexcept
{
  return std::hash<int>()(value.memoryPrecision.value);
}

std::size_t std::hash<ArgmaxDescriptor>::operator()(const ArgmaxDescriptor& value) const noexcept
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
  combine_32(seed, (value.gumbel ? 1u : 0u) | (value.partitioned ? 2u : 0u));
  if (value.gumbel)
    combine_32(seed, *reinterpret_cast<const uint32_t*>(&value.scale));
  return seed;
}

std::pair<ArgmaxKernelDescriptor, PipelineValue<ArgmaxKernel>*> ArgmaxDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ArgmaxKernelDescriptor, std::unique_ptr<ArgmaxKernel>>* const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  const ArgmaxKernelDescriptor kernelDesc = {
    .memoryPrecision = memoryPrecision,
  };
  ArgmaxKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    auto newKernel = std::make_unique<ArgmaxKernel>(kernelDesc, device);
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
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(3));
    NS::Error* error = nil;
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  const char* const entrypoint = gumbel ?
    (partitioned ? "gumbel_argmax_partition" : "gumbel_argmax_one_pass") :
    (partitioned ? "argmax_partition" : "argmax_one_pass");
  auto* const output = new PipelineValue<ArgmaxKernel> {
    kernel,
    createPipeline(entrypoint),
  };
  if (partitioned) {
    output->second = createPipeline("argmax_merge_partitions");
  }
  return std::make_pair(kernelDesc, output);
}
