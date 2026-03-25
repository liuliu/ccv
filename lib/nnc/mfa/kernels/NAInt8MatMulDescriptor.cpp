#include "NAInt8MatMulDescriptor.hpp"
#include "NAInt8MatMulKernelDescriptor.hpp"
#include "NAInt8MatMulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

namespace {

static void serializeBinaries(MTL::BinaryArchive *const binaryArchive, const std::string& pathToWrite) noexcept {
  NS::Error *error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
}

static uint32_t groupM(const uint32_t M) noexcept {
  return (M >= 4096) ? 4096 : 0;
}

static uint32_t groupN(const uint32_t N) noexcept {
  return (N >= 4096) ? 4096 : 0;
}

}

bool NAInt8MatMulDescriptor::operator==(const NAInt8MatMulDescriptor& rhs) const {
  return
    ioPrecision == rhs.ioPrecision &&
    useBias == rhs.useBias &&
    simd_all(matrixDimensions == rhs.matrixDimensions);
}

std::size_t std::hash<NAInt8MatMulDescriptor>::operator()(const NAInt8MatMulDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, (uint32_t)hash.ioPrecision.value);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, hash.useBias ? 1 : 0);
  return seed;
}

NAInt8MatMulKernelDescriptor NAInt8MatMulDescriptor::kernelDescriptor() const noexcept {
  return NAInt8MatMulKernelDescriptor(
      simd::ushort3 { 128, 128, 128 },
      8,
      ioPrecision,
      useBias,
      256,
      groupM(matrixDimensions[0]),
      groupN(matrixDimensions[1]));
}

std::pair<NAInt8MatMulKernelDescriptor, PipelineValue<NAInt8MatMulKernel> *> NAInt8MatMulDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties &dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<NAInt8MatMulKernelDescriptor, std::unique_ptr<NAInt8MatMulKernel>> *const libraryCache) const noexcept
{
  (void)dprops;

  auto createKernel =
  [=](const NAInt8MatMulKernelDescriptor& descriptor) -> NAInt8MatMulKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    NAInt8MatMulKernel* kernel = new NAInt8MatMulKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<NAInt8MatMulKernel>(kernel);
    return kernel;
  };

  auto createPipeline =
  [=](NAInt8MatMulKernel* kernel, const char* functionNameString) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t M = matrixDimensions[0];
    const uint32_t N = matrixDimensions[1];
    const uint32_t K = matrixDimensions[2];
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
    NS::Error* error = nil;
    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
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
  auto pipeline = NS::TransferPtr(createPipeline(kernel, "int8_matmul"));
  auto quantize = NS::TransferPtr(createPipeline(kernel, "quantize_activation"));

  PipelineValue<NAInt8MatMulKernel>* output = new PipelineValue<NAInt8MatMulKernel> { kernel, pipeline };
  output->second = quantize;
  return std::make_pair(kernelDesc, output);
}
