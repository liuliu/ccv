#include "SwishMulDescriptor.hpp"
#include "SwishMulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool SwishMulDescriptor::operator==(const SwishMulDescriptor& rhs) const {
  return
  gradient == rhs.gradient &&
  outputMask == rhs.outputMask &&
  value == rhs.value &&
  beta == rhs.beta &&
  scale == rhs.scale &&
  gPrecision == rhs.gPrecision &&
  aPrecision == rhs.aPrecision &&
  bPrecision == rhs.bPrecision &&
  daPrecision == rhs.daPrecision &&
  dbPrecision == rhs.dbPrecision &&
  length == rhs.length;
}

std::size_t std::hash<SwishMulDescriptor>::operator()(const SwishMulDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.gradient, (unsigned int)hash.outputMask }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.gPrecision.value, (unsigned int)hash.aPrecision.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.bPrecision.value, (unsigned int)hash.daPrecision.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.dbPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, (uint64_t)hash.length);
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.beta), *reinterpret_cast<const uint32_t*>(&hash.scale) }));
  return seed;
}

std::pair<SwishMulKernelDescriptor, PipelineValue<SwishMulKernel>*> SwishMulDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<SwishMulKernelDescriptor, std::unique_ptr<SwishMulKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](SwishMulKernelDescriptor descriptor) -> SwishMulKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      SwishMulKernel* kernel = new SwishMulKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<SwishMulKernel>(kernel);
      return kernel;
    }
  };

  SwishMulKernelDescriptor kernelDesc;
  kernelDesc.gradient = gradient;
  kernelDesc.outputMask = outputMask;
  kernelDesc.value = value;
  kernelDesc.beta = beta;
  kernelDesc.scale = scale;
  kernelDesc.gPrecision = gPrecision;
  kernelDesc.aPrecision = aPrecision;
  kernelDesc.bPrecision = bPrecision;
  kernelDesc.daPrecision = daPrecision;
  kernelDesc.dbPrecision = dbPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t count;
    if (value == 0) {
    } else if (value == 1) {
      count = length / 4;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    } else {
      count = length;
      constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
    }
    if (beta != 1)
      constants->setConstantValue(&beta, MTL::DataTypeFloat, NS::UInteger(1));
    if (scale != 1)
      constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(2));

    NS::String* swiftName = NS::String::string("swish_mul", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  SwishMulKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<SwishMulKernel>* output = new PipelineValue<SwishMulKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
