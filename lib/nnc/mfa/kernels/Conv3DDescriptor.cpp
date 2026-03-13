#include "Conv3DDescriptor.hpp"
#include "Conv3DKernelDescriptor.hpp"
#include "Conv3DKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

static void serializeBinaries(MTL::BinaryArchive *const binaryArchive, const std::string& pathToWrite) noexcept {
  NS::Error *error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
}

bool Conv3DDescriptor::operator==(const Conv3DDescriptor& rhs) const {
  return
  dataType == rhs.dataType &&
  batchDimension == rhs.batchDimension &&
  inputChannels == rhs.inputChannels &&
  outputChannels == rhs.outputChannels &&
  paddingLeft == rhs.paddingLeft &&
  paddingRight == rhs.paddingRight &&
  paddingTop == rhs.paddingTop &&
  paddingBottom == rhs.paddingBottom &&
  simd_all(matrixDimensions == rhs.matrixDimensions) &&
  simd_all(kernelDimensions == rhs.kernelDimensions) &&
  useBias == rhs.useBias;
}

std::size_t std::hash<Conv3DDescriptor>::operator()(const Conv3DDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, hash.dataType);
  combine_32(seed, hash.batchDimension);
  combine_32(seed, hash.inputChannels);
  combine_32(seed, hash.outputChannels);
  combine_32(seed, hash.paddingLeft);
  combine_32(seed, hash.paddingRight);
  combine_32(seed, hash.paddingTop);
  combine_32(seed, hash.paddingBottom);
  combine_64(seed, pack_64(simd_make_uint2(hash.matrixDimensions[0], hash.matrixDimensions[1])));
  combine_32(seed, hash.matrixDimensions[2]);
  combine_64(seed, pack_64(simd_make_uint2(hash.kernelDimensions[0], hash.kernelDimensions[1])));
  combine_32(seed, hash.kernelDimensions[2]);
  combine_32(seed, pack_32(simd::uchar4 { hash.useBias, 0, 0, 0 }));
  return seed;
}

Conv3DKernelDescriptor Conv3DDescriptor::kernelDescriptor() const noexcept {
  return Conv3DKernelDescriptor(
      simd::ushort2 { 32, 32 },
      simd::ushort3 {
          (unsigned short)kernelDimensions[0],
          (unsigned short)kernelDimensions[1],
          (unsigned short)kernelDimensions[2] },
      dataType,
      inputChannels,
      outputChannels,
      paddingLeft,
      paddingRight,
      paddingTop,
      paddingBottom,
      useBias);
}

std::pair<Conv3DKernelDescriptor, PipelineValue<Conv3DKernel> *> Conv3DDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Conv3DKernelDescriptor, std::unique_ptr<Conv3DKernel>> *const libraryCache) const noexcept {
  auto createKernel =
  [=](Conv3DKernelDescriptor descriptor) -> Conv3DKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      Conv3DKernel* kernel = new Conv3DKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<Conv3DKernel>(kernel);
      return kernel;
    }
  };

  auto createConvPipeline =
  [=](MTL::Library* library, const char* functionName) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    uint32_t depth = matrixDimensions[0];
    uint32_t height = matrixDimensions[1];
    uint32_t width = matrixDimensions[2];
    constants->setConstantValue(&depth, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&height, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&width, MTL::DataTypeUInt, NS::UInteger(2));

    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(NS::String::string(functionName, NS::UTF8StringEncoding), constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    descriptor->setComputeFunction(function.get());

    MTL::ComputePipelineState* pipeline = nullptr;
    if (binaryArchivesToRead) {
      descriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      CCV_NNC_MFA_CHECK_ERROR(error);
      if (binaryArchiveToWrite != nullptr) {
        binaryArchiveToWrite->addComputePipelineFunctions(descriptor.get(), &error);
        serializeBinaries(binaryArchiveToWrite, pathToWrite);
      }
    }
    return pipeline;
  };

  auto createPermutationPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(NS::String::string("permute_oidhw_to_ok", NS::UTF8StringEncoding), constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    descriptor->setComputeFunction(function.get());

    MTL::ComputePipelineState* pipeline = nullptr;
    if (binaryArchivesToRead) {
      descriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      CCV_NNC_MFA_CHECK_ERROR(error);
      if (binaryArchiveToWrite != nullptr) {
        binaryArchiveToWrite->addComputePipelineFunctions(descriptor.get(), &error);
        serializeBinaries(binaryArchiveToWrite, pathToWrite);
      }
    }
    return pipeline;
  };

  auto kernelDesc = kernelDescriptor();
  auto kernel = createKernel(kernelDesc);
  if (kernel->permutationPipeline.get() == nullptr) {
    kernel->permutationPipeline = NS::TransferPtr(createPermutationPipeline(kernel->library.get()));
  }
  auto pipeline = NS::TransferPtr(createConvPipeline(kernel->library.get(), "conv3d"));
  auto output = new PipelineValue<Conv3DKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
