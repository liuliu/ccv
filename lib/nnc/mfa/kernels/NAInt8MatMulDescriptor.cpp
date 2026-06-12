#include "NAInt8MatMulDescriptor.hpp"
#include "NAInt8MatMulKernelDescriptor.hpp"
#include "NAInt8MatMulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include <cstring>

namespace {

static void serializeBinaries(MTL::BinaryArchive *const binaryArchive, const std::string& pathToWrite) noexcept {
  NS::Error *error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
}

static uint32_t groupM(const uint32_t M) noexcept {
  return (M >= 4096) ? 4096 : 0;
}

static uint32_t groupN(const uint32_t N) noexcept {
  return (N >= 4096) ? 4096 : 0;
}

}

bool NAInt8MatMulDescriptor::operator==(const NAInt8MatMulDescriptor& rhs) const {
  auto lhsMatrixDimensions = matrixDimensions;
  auto rhsMatrixDimensions = rhs.matrixDimensions;
  if (loadM) {
    lhsMatrixDimensions[0] = groupM(lhsMatrixDimensions[0]);
    rhsMatrixDimensions[0] = groupM(rhsMatrixDimensions[0]);
  }
  return
    batchDimension == rhs.batchDimension &&
    ioPrecision == rhs.ioPrecision &&
    simd_all(batchStrides.value_or(simd::uint4(UINT32_MAX)) == rhs.batchStrides.value_or(simd::uint4(UINT32_MAX))) &&
    packedABatchStride == rhs.packedABatchStride &&
    aScaleBatchStride == rhs.aScaleBatchStride &&
    useBias == rhs.useBias &&
    loadM == rhs.loadM &&
    supportIndirectCommandBuffers == rhs.supportIndirectCommandBuffers &&
    simd_all(lhsMatrixDimensions == rhsMatrixDimensions);
}

std::size_t std::hash<NAInt8MatMulDescriptor>::operator()(const NAInt8MatMulDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, hash.batchDimension);
  combine_32(seed, (uint32_t)hash.ioPrecision.value);
  combine_32(seed, hash.loadM ? groupM(hash.matrixDimensions[0]) : hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  if (hash.batchStrides.has_value()) {
    combine_32(seed, hash.batchStrides.value()[0]);
    combine_32(seed, hash.batchStrides.value()[1]);
    combine_32(seed, hash.batchStrides.value()[2]);
    combine_32(seed, hash.batchStrides.value()[3]);
  }
  combine_32(seed, hash.packedABatchStride.value_or(0));
  combine_32(seed, hash.aScaleBatchStride.value_or(0));
  combine_32(seed, hash.useBias ? 1 : 0);
  combine_32(seed, hash.loadM ? 1 : 0);
  combine_32(seed, hash.supportIndirectCommandBuffers ? 1 : 0);
  return seed;
}

NAInt8MatMulKernelDescriptor NAInt8MatMulDescriptor::kernelDescriptor() const noexcept {
  return NAInt8MatMulKernelDescriptor(
      simd::ushort3 { 128, 128, 128 },
      8,
      ioPrecision,
      useBias,
      loadM,
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
    const bool batched = batchDimension > 1;
    const simd::uint4 batchStrides = this->batchStrides.value_or(simd::uint4(0));
    const uint32_t batchStrideSourceA = batchStrides[0];
    const uint32_t batchStridePackedA = packedABatchStride.value_or(batched ? M * K : 0);
    const uint32_t batchStrideB = batchStrides[1];
    const uint32_t batchStrideC = batchStrides[2];
    const uint32_t batchStrideBias = batchStrides[3];
    const uint32_t batchStrideAScale = aScaleBatchStride.value_or(batched ? M : 0);
    const uint32_t batchStrideBScale = batchStrides[1] > 0 ? N : 0;
    const bool quantizeActivation = (strcmp(functionNameString, "quantize_activation") == 0);
    const uint32_t batchStrideA = quantizeActivation ? batchStrideSourceA : batchStridePackedA;
    if (!this->loadM)
      constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&batched, MTL::DataTypeBool, NS::UInteger(11));
    constants->setConstantValue(&batchStrideA, MTL::DataTypeUInt, NS::UInteger(15));
    constants->setConstantValue(&batchStrideB, MTL::DataTypeUInt, NS::UInteger(16));
    constants->setConstantValue(&batchStrideC, MTL::DataTypeUInt, NS::UInteger(17));
    constants->setConstantValue(&batchStrideBias, MTL::DataTypeUInt, NS::UInteger(18));
    constants->setConstantValue(&batchStrideAScale, MTL::DataTypeUInt, NS::UInteger(19));
    constants->setConstantValue(&batchStrideBScale, MTL::DataTypeUInt, NS::UInteger(20));
    constants->setConstantValue(&batchStridePackedA, MTL::DataTypeUInt, NS::UInteger(21));
    NS::Error* error = nil;
    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDescriptor->setComputeFunction(function.get());
    pipelineDescriptor->setSupportIndirectCommandBuffers(this->supportIndirectCommandBuffers);
    MTL::ComputePipelineState* pipeline = nullptr;
    if (binaryArchivesToRead) {
      pipelineDescriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      if (binaryArchiveToWrite != nullptr) {
        NS::Error* archiveError = nil;
        binaryArchiveToWrite->addComputePipelineFunctions(pipelineDescriptor.get(), &archiveError);
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
