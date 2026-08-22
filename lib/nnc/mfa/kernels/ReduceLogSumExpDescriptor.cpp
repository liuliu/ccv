#include "ReduceLogSumExpDescriptor.hpp"
#include "ReduceLogSumExpKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool ReduceLogSumExpKernelDescriptor::operator==(const ReduceLogSumExpKernelDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision;
}

bool ReduceLogSumExpDescriptor::operator==(const ReduceLogSumExpDescriptor& rhs) const
{
  return memoryPrecision == rhs.memoryPrecision &&
    columnCount == rhs.columnCount &&
    partitionSize == rhs.partitionSize &&
    partitionCount == rhs.partitionCount &&
    scale == rhs.scale &&
    partitioned == rhs.partitioned;
}

std::size_t std::hash<ReduceLogSumExpKernelDescriptor>::operator()(const ReduceLogSumExpKernelDescriptor& value) const noexcept
{
  return std::hash<int>()(value.memoryPrecision.value);
}

std::size_t std::hash<ReduceLogSumExpDescriptor>::operator()(const ReduceLogSumExpDescriptor& value) const noexcept
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
  combine_64(seed, std::hash<float>()(value.scale));
  combine_32(seed, value.partitioned ? 1u : 0u);
  return seed;
}

std::pair<ReduceLogSumExpKernelDescriptor, PipelineValue<ReduceLogSumExpKernel>*> ReduceLogSumExpDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ReduceLogSumExpKernelDescriptor, std::unique_ptr<ReduceLogSumExpKernel>>* const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  const ReduceLogSumExpKernelDescriptor kernel_desc = {
    .memoryPrecision = memoryPrecision,
  };
  ReduceLogSumExpKernel* kernel;
  const auto iterator = libraryCache->find(kernel_desc);
  if (iterator != libraryCache->end()) {
    kernel = iterator->second.get();
  } else {
    auto new_kernel = std::make_unique<ReduceLogSumExpKernel>(kernel_desc, device);
    kernel = new_kernel.get();
    (*libraryCache)[kernel_desc] = std::move(new_kernel);
  }

  const auto create_pipeline =
  [=](const char* const name) -> NS::SharedPtr<MTL::ComputePipelineState> {
    auto function_name = NS::String::string(name, NS::UTF8StringEncoding);
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&columnCount, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&partitionSize, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&partitionCount, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(3));
    NS::Error* error = nil;
    auto function = NS::TransferPtr(kernel->library->newFunction(function_name, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto* const output = new PipelineValue<ReduceLogSumExpKernel> {
    kernel,
    create_pipeline(partitioned ? "reduce_logsumexp_partition" : "reduce_logsumexp_one_pass"),
  };
  if (partitioned) {
    output->second = create_pipeline("reduce_logsumexp_merge_partitions");
  }
  return std::make_pair(kernel_desc, output);
}
