#include "NAInt8MatMulSmallMDescriptor.hpp"
#include "NAInt8MatMulSmallMKernel.hpp"
#include "NAInt8MatMulSmallMKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

static void serializeBinaries(MTL::BinaryArchive* const binaryArchive, const std::string& pathToWrite) noexcept
{
  NS::Error* error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
}

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
  return (value + alignment - 1) / alignment * alignment;
}

bool NAInt8MatMulSmallMDescriptor::operator==(const NAInt8MatMulSmallMDescriptor& rhs) const {
  auto lhsMatrixDimensions = matrixDimensions;
  auto rhsMatrixDimensions = rhs.matrixDimensions;
  if (loadM) {
    lhsMatrixDimensions[0] = 0;
    rhsMatrixDimensions[0] = 0;
  }
  return
    batchDimension == rhs.batchDimension &&
    simd_all(lhsMatrixDimensions == rhsMatrixDimensions) &&
    ioPrecision == rhs.ioPrecision &&
    useBias == rhs.useBias &&
    loadM == rhs.loadM;
}

uint16_t NAInt8MatMulSmallMDescriptor::pack() const noexcept
{
  return 8;
}

simd::ushort3 NAInt8MatMulSmallMDescriptor::blockDimensions() const noexcept
{
  return simd::ushort3 { 128, 64, 128 };
}

uint16_t NAInt8MatMulSmallMDescriptor::executionSIMDGroups() const noexcept
{
  return 8;
}

uint16_t NAInt8MatMulSmallMDescriptor::splitK() const noexcept
{
  if (loadM) {
    return 1;
  }
  if (matrixDimensions[2] < 16384) {
    return 1;
  }
  const uint32_t packValue = pack();
  const simd::ushort3 blockDims = blockDimensions();
  if ((matrixDimensions[0] * packValue) % blockDims[1] != 0 ||
      (matrixDimensions[1] * packValue) % blockDims[0] != 0) {
    return 1;
  }
  const uint32_t kpack = matrixDimensions[2] / packValue;
  if (kpack % (8 * blockDims[2]) == 0) {
    return 8;
  }
  if (kpack % (4 * blockDims[2]) == 0) {
    return 4;
  }
  return 1;
}

NAInt8MatMulSmallMScratchOffsets NAInt8MatMulSmallMDescriptor::scratchOffsets() const noexcept
{
  const uint32_t M = matrixDimensions[0];
  const uint32_t N = matrixDimensions[1];
  const uint32_t packValue = pack();
  const uint32_t splitKValue = splitK();
  size_t offset = 0;
  const size_t partials = align_up(offset, 256);
  offset = partials + (size_t)splitKValue * M * packValue * N * sizeof(int32_t);
  return NAInt8MatMulSmallMScratchOffsets {
    .partials = partials,
    .total = align_up(offset, 256),
  };
}

std::size_t std::hash<NAInt8MatMulSmallMDescriptor>::operator()(const NAInt8MatMulSmallMDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, hash.batchDimension);
  combine_64(seed, pack_64(simd::uint2 { hash.loadM ? 0 : hash.matrixDimensions[0], hash.matrixDimensions[1] }));
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, (uint32_t)hash.ioPrecision.value);
  combine_32(seed, hash.useBias ? 1 : 0);
  combine_32(seed, hash.loadM ? 1 : 0);
  return seed;
}

std::pair<NAInt8MatMulSmallMKernelDescriptor, PipelineValue<NAInt8MatMulSmallMKernel>*> NAInt8MatMulSmallMDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<NAInt8MatMulSmallMKernelDescriptor, std::unique_ptr<NAInt8MatMulSmallMKernel>>* const libraryCache) const noexcept {
  (void)dprops;
  auto createKernel =
  [=](NAInt8MatMulSmallMKernelDescriptor descriptor) -> NAInt8MatMulSmallMKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      NAInt8MatMulSmallMKernel* kernel = new NAInt8MatMulSmallMKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<NAInt8MatMulSmallMKernel>(kernel);
      return kernel;
    }
  };

  auto createConstants =
  [=]() -> NS::SharedPtr<MTL::FunctionConstantValues> {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t N = matrixDimensions[1];
    const uint32_t K = matrixDimensions[2];
    const uint32_t kpack = K / pack();
    const uint32_t splitKValue = splitK();
    const uint32_t splitKPack = kpack / splitKValue;
    if (!loadM) {
      const uint32_t M = matrixDimensions[0];
      constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    }
    constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&kpack, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&splitKValue, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&splitKPack, MTL::DataTypeUInt, NS::UInteger(5));
    return constants;
  };

  auto createPipeline =
  [=](MTL::Library* library, const char* name, MTL::FunctionConstantValues* constants) -> MTL::ComputePipelineState* {
    auto functionName = NS::String::string(name, NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(functionName, constants, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    descriptor->setComputeFunction(function.get());
    MTL::ComputePipelineState* pipeline = nullptr;
    bool binaryArchiveMiss = false;
    if (binaryArchivesToRead) {
      descriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      binaryArchiveMiss = true;
    }
    if (binaryArchiveMiss && binaryArchiveToWrite != nullptr) {
      binaryArchiveToWrite->addComputePipelineFunctions(descriptor.get(), &error);
      serializeBinaries(binaryArchiveToWrite, pathToWrite);
    }
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto kernelDesc = NAInt8MatMulSmallMKernelDescriptor(
      blockDimensions(), pack(), executionSIMDGroups(), ioPrecision, useBias, loadM);
  NAInt8MatMulSmallMKernel* kernel = createKernel(kernelDesc);
  auto constants = createConstants();
  PipelineValue<NAInt8MatMulSmallMKernel>* output = new PipelineValue<NAInt8MatMulSmallMKernel> {
    kernel,
    NS::TransferPtr(createPipeline(kernel->library.get(), "int8_matmul_small_m_block_view", constants.get())),
    NS::SharedPtr<MTL::IndirectCommandBuffer>(),
    NS::SharedPtr<MTL::Function>(),
    NS::TransferPtr(createPipeline(kernel->library.get(), "reduce_diagonal", constants.get())),
  };
  return std::make_pair(kernelDesc, output);
}
