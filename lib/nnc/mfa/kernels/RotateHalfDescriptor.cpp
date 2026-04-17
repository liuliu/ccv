#include "RotateHalfDescriptor.hpp"
#include "RotateHalfKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool RotateHalfDescriptor::operator==(const RotateHalfDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  value == rhs.value &&
  rowCount == rhs.rowCount &&
  dim == rhs.dim;
}

std::size_t std::hash<RotateHalfDescriptor>::operator()(const RotateHalfDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, pack_64(simd::uint2 { hash.rowCount, hash.dim }));
  return seed;
}

std::pair<RotateHalfKernelDescriptor, PipelineValue<RotateHalfKernel> *> RotateHalfDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<RotateHalfKernelDescriptor, std::unique_ptr<RotateHalfKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](RotateHalfKernelDescriptor descriptor) -> RotateHalfKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      RotateHalfKernel* kernel = new RotateHalfKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<RotateHalfKernel>(kernel);
      return kernel;
    }
  };

  RotateHalfKernelDescriptor kernelDesc;
  kernelDesc.value = value;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    const int vectorized = (value == 0);
    const uint32_t half_dim_units = vectorized ? (dim / 8) : (dim / 2);
    const uint32_t dim_units = vectorized ? (dim / 4) : dim;
    const uint32_t count = rowCount * dim_units;
    constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&dim_units, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&half_dim_units, MTL::DataTypeUInt, NS::UInteger(2));

    NS::String* swiftName = NS::String::string("rotate_half", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  RotateHalfKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<RotateHalfKernel>* output = new PipelineValue<RotateHalfKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
