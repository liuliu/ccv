#include "CMulDescriptor.hpp"
#include "CMulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool CMulDescriptor::operator==(const CMulDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  conjugate == rhs.conjugate &&
  value == rhs.value &&
  simd_all(stridesA == rhs.stridesA) &&
  simd_all(stridesB == rhs.stridesB) &&
  simd_all(stridesC == rhs.stridesC) &&
  simd_all(dimensions == rhs.dimensions);
}

std::size_t std::hash<CMulDescriptor>::operator()(const CMulDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.stridesA[0], (unsigned int)hash.stridesA[1] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.stridesA[2], (unsigned int)hash.stridesB[0] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.stridesB[1], (unsigned int)hash.stridesB[2] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.stridesC[0], (unsigned int)hash.stridesC[1] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.stridesC[2], (unsigned int)hash.dimensions[0] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.dimensions[1], (unsigned int)hash.dimensions[2] }));
  return seed;
}

std::pair<CMulKernelDescriptor, PipelineValue<CMulKernel> *> CMulDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<CMulKernelDescriptor, std::unique_ptr<CMulKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](CMulKernelDescriptor descriptor) -> CMulKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      CMulKernel* kernel = new CMulKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<CMulKernel>(kernel);
      return kernel;
    }
  };

  CMulKernelDescriptor kernelDesc;
  kernelDesc.conjugate = conjugate;
  kernelDesc.value = value;
  kernelDesc.memoryPrecision = memoryPrecision;

  // WARNING: The owner must explicitly retain the compute pipeline.
  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t dim0 = dimensions[0] / 2;
    constants->setConstantValue(&dim0, MTL::DataTypeUInt, NS::UInteger(0));

    if (value != 0) {
      uint32_t dim1 = dimensions[1];
      uint32_t astride0 = stridesA[0];
      uint32_t bstride0 = stridesB[0];
      uint32_t cstride0 = stridesC[0];
      constants->setConstantValue(&dim1, MTL::DataTypeUInt, 1);
      constants->setConstantValue(&astride0, MTL::DataTypeUInt, 2);
      constants->setConstantValue(&bstride0, MTL::DataTypeUInt, 3);
      constants->setConstantValue(&cstride0, MTL::DataTypeUInt, 4);
	}

    if (value != 0 && value != 1) {
      uint32_t astride1 = stridesA[1];
      uint32_t bstride1 = stridesB[1];
      uint32_t cstride1 = stridesC[1];
      constants->setConstantValue(&astride1, MTL::DataTypeUInt, 5);
      constants->setConstantValue(&bstride1, MTL::DataTypeUInt, 6);
      constants->setConstantValue(&cstride1, MTL::DataTypeUInt, 7);
	}

    if (value != 0 && value != 1 && value != 2) {
      uint32_t dim2 = dimensions[2];
      uint32_t astride2 = stridesA[2];
      uint32_t bstride2 = stridesB[2];
      uint32_t cstride2 = stridesC[2];
      constants->setConstantValue(&dim2, MTL::DataTypeUInt, 8);
      constants->setConstantValue(&astride2, MTL::DataTypeUInt, 9);
      constants->setConstantValue(&bstride2, MTL::DataTypeUInt, 10);
      constants->setConstantValue(&cstride2, MTL::DataTypeUInt, 11);
	}

    NS::String* swiftName = NS::String::string("cmul", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    
    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  CMulKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
    
  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<CMulKernel>* output = new PipelineValue<CMulKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
