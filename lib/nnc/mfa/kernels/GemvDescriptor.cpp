#include "GemvDescriptor.hpp"
#include "GemvKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

#include <cstring>

bool GemvDescriptor::operator==(const GemvDescriptor& rhs) const {
  return
  fusedBias == rhs.fusedBias &&
  mrows == rhs.mrows &&
  memoryPrecision == rhs.memoryPrecision &&
  nrows == rhs.nrows &&
  ncols == rhs.ncols;
}

uint32_t GemvDescriptor::rowsPerThreadgroup(MTL::Device* const device) noexcept {
  uint32_t rows = 4;
  const char* const deviceName = device->name()->utf8String();
  if (deviceName && (strstr(deviceName, "M1") != 0 ||
      strstr(deviceName, "M2") != 0 ||
      strstr(deviceName, "M3") != 0 ||
      strstr(deviceName, "M4") != 0 ||
      strstr(deviceName, "M5") != 0)) {
    if (strstr(deviceName, "Max") != 0 ||
        strstr(deviceName, "Ultra") != 0) {
      rows = 8;
    } else if (strstr(deviceName, "Pro") != 0) {
      rows = 4;
    }
  }
  return rows;
}

std::size_t std::hash<GemvDescriptor>::operator()(const GemvDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.fusedBias | ((unsigned int)hash.mrows << 8) }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.nrows, (unsigned int)hash.ncols }));
  return seed;
}

std::pair<GemvKernelDescriptor, PipelineValue<GemvKernel>*> GemvDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GemvKernelDescriptor, std::unique_ptr<GemvKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](GemvKernelDescriptor descriptor) -> GemvKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      GemvKernel* kernel = new GemvKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<GemvKernel>(kernel);
      return kernel;
    }
  };

  GemvKernelDescriptor kernelDesc;
  kernelDesc.fusedBias = fusedBias;
  kernelDesc.mrows = mrows;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    const uint32_t rows = rowsPerThreadgroup(device);
    constants->setConstantValue(&rows, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&ncols, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&nrows, MTL::DataTypeUInt, NS::UInteger(2));

    NS::String* swiftName = NS::String::string("gemv", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  GemvKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<GemvKernel>* output = new PipelineValue<GemvKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
