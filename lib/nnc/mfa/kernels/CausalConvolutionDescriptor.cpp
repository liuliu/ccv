#include "CausalConvolutionDescriptor.hpp"
#include "CausalConvolutionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool CausalConvolutionDescriptor::operator==(const CausalConvolutionDescriptor& rhs) const {
  return silu == rhs.silu && memoryPrecision == rhs.memoryPrecision;
}

std::size_t std::hash<CausalConvolutionDescriptor>::operator()(const CausalConvolutionDescriptor& descriptor) const noexcept {
  const CausalConvolutionKernelDescriptor kernelDesc = { descriptor.silu, descriptor.memoryPrecision };
  return std::hash<CausalConvolutionKernelDescriptor>()(kernelDesc);
}

std::pair<CausalConvolutionKernelDescriptor, PipelineValue<CausalConvolutionKernel>*> CausalConvolutionDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CausalConvolutionKernelDescriptor, std::unique_ptr<CausalConvolutionKernel>>* const libraryCache) const noexcept {
  const CausalConvolutionKernelDescriptor kernelDesc = { silu, memoryPrecision };
  CausalConvolutionKernel* kernel;
  const auto iterator = libraryCache->find(kernelDesc);
  if (iterator != libraryCache->end())
    kernel = iterator->second.get();
  else {
    auto newKernel = std::make_unique<CausalConvolutionKernel>(kernelDesc, device);
    kernel = newKernel.get();
    (*libraryCache)[kernelDesc] = std::move(newKernel);
  }
  NS::Error* error = nil;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("causal_convolution", NS::UTF8StringEncoding)));
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return std::make_pair(kernelDesc, new PipelineValue<CausalConvolutionKernel> { kernel, pipeline });
}
