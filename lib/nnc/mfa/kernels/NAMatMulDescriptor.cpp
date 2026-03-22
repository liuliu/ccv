#include "NAMatMulDescriptor.hpp"
#include "NAMatMulKernelDescriptor.hpp"
#include "NAMatMulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

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

bool NAMatMulDescriptor::operator==(const NAMatMulDescriptor& rhs) const {
  auto lhsMatrixDimensions = matrixDimensions;
  auto rhsMatrixDimensions = rhs.matrixDimensions;
  if (loadM) {
    lhsMatrixDimensions[0] = groupM(lhsMatrixDimensions[0]);
    rhsMatrixDimensions[0] = groupM(rhsMatrixDimensions[0]);
  }
  return
  (batchDimension == rhs.batchDimension) &&
  simd_all(lhsMatrixDimensions == rhsMatrixDimensions) &&
  simd_all(batchStrides.value_or(simd::uint4(UINT32_MAX)) == rhs.batchStrides.value_or(simd::uint4(UINT32_MAX))) &&
  memoryPrecisions == rhs.memoryPrecisions &&
  registerPrecisionC == rhs.registerPrecisionC &&
  simd_all(transposeState == rhs.transposeState) &&
  (useBias == rhs.useBias) &&
  (loadM == rhs.loadM) &&
  (supportIndirectCommandBuffers == rhs.supportIndirectCommandBuffers);
}

std::size_t std::hash<NAMatMulDescriptor>::operator()(const NAMatMulDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, hash.batchDimension);
  combine_32(seed, hash.loadM ? groupM(hash.matrixDimensions[0]) : hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  if (hash.batchStrides.has_value()) {
    combine_32(seed, hash.batchStrides.value()[0]);
    combine_32(seed, hash.batchStrides.value()[1]);
    combine_32(seed, hash.batchStrides.value()[2]);
    combine_32(seed, hash.batchStrides.value()[3]);
  }
  combine_64(seed, pack_64(simd::ushort4 { hash.memoryPrecisions.A.value, hash.memoryPrecisions.B.value, hash.memoryPrecisions.C.value, hash.memoryPrecisions.bias.value }));
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], hash.transposeState[2], 0 }));
  combine_32(seed, pack_32(simd::uchar4 { hash.useBias, hash.loadM, hash.supportIndirectCommandBuffers, 0 }));
  if (hash.registerPrecisionC.has_value()) {
    combine_32(seed, pack_32(simd::ushort2 { hash.registerPrecisionC.value().value, 0 }));
  }
  return seed;
}

uint16_t NAMatMulDescriptor::splitK() const noexcept {
  if ((this->matrixDimensions[1] % 64) != 0) { // If cannot divide by 64, we cannot have splitK.
    assert(this->matrixDimensions[2] < 65536); // It seems without split K, MPP have issues with K >= 65536.
    return 1;
  }
  if (this->matrixDimensions[2] > 3072 * 4) {
    // Still multiple of 2, but more than 1.
    return this->matrixDimensions[2] / 3072 / 2 * 2;
  } else if (this->matrixDimensions[2] >= 2048 * 4) {
    return 4; // Use split by 4 if we can end up with >= 2048 per split.
  } else if (this->matrixDimensions[2] >= 2048 * 2) {
    return 2; // Use split by 2 if we can end up with >= 2048 per split.
  }
  return 1;
}

bool NAMatMulDescriptor::threadBarrierOverK(uint32_t K, uint16_t splitKValue) noexcept {
  // Benchmark-guided policy:
  // - splitK == 2 lost on the tested 4096-6144 K range where it is selected.
  // - splitK == 1, 4, and 8 start to benefit once K is very large.
  if (K < 16384) {
    return false;
  }
  if (splitKValue == 2) {
    return false;
  }
  return true;
}

std::pair<NAMatMulKernelDescriptor, PipelineValue<NAMatMulKernel> *> NAMatMulDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NAMatMulKernelDescriptor, std::unique_ptr<NAMatMulKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](NAMatMulKernelDescriptor descriptor) -> NAMatMulKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      NAMatMulKernel* kernel = new NAMatMulKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<NAMatMulKernel>(kernel);
      return kernel;
    }
  };

  // WARNING: The owner must explicitly retain the compute pipeline.
  auto createPipeline =
  [=](MTL::Library* library, uint16_t splitK, bool divisibleBy2) -> std::pair<MTL::ComputePipelineState*, MTL::ComputePipelineState*> {
    // Set the function constants.
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    if (!this->loadM) {
      uint32_t M = this->matrixDimensions[0];
      constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    }
    uint32_t N = this->matrixDimensions[1];
    uint32_t K = this->matrixDimensions[2];
    constants->setConstantValue(&N, MTL::DataTypeUInt, 1);
    constants->setConstantValue(&K, MTL::DataTypeUInt, 2);
    bool batched = this->batchDimension > 1;
    constants->setConstantValue(&batched, MTL::DataTypeBool, 11);
    simd::uint4 batchStrides = this->batchStrides.value_or(simd::uint4(0));
    auto batchStrideA = batchStrides[0];
    auto batchStrideB = batchStrides[1];
    auto batchStrideC = batchStrides[2];
    auto batchStrideBias = batchStrides[3];
    constants->setConstantValue(&batchStrideA, MTL::DataTypeUInt, 15);
    constants->setConstantValue(&batchStrideB, MTL::DataTypeUInt, 16);
    constants->setConstantValue(&batchStrideC, MTL::DataTypeUInt, 17);
    constants->setConstantValue(&batchStrideBias, MTL::DataTypeUInt, 18);

    NS::String* swiftName = NS::String::string("matmul", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    if (splitK > 1) {
      NS::String* reduceSumName = NS::String::string(divisibleBy2 ? "reduce_sum_2" : "reduce_sum", NS::UTF8StringEncoding);
      auto reduceSum = NS::TransferPtr
      (library->newFunction(reduceSumName, constants.get(), &error));
      CCV_NNC_MFA_CHECK_ERROR(error);
      auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
      descriptor->setComputeFunction(function.get());
      descriptor->setSupportIndirectCommandBuffers(this->supportIndirectCommandBuffers);
      MTL::ComputePipelineState* pipeline = nullptr;
      if (binaryArchivesToRead) {
        descriptor->setBinaryArchives(binaryArchivesToRead);
        pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
      }
      bool binaryArchiveMiss = false;
      if (pipeline == nullptr) {
        error = nil;
        pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
        binaryArchiveMiss = true;
      }
      auto secondDesc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
      secondDesc->setComputeFunction(reduceSum.get());
      secondDesc->setSupportIndirectCommandBuffers(this->supportIndirectCommandBuffers);
      MTL::ComputePipelineState* second = nullptr;
      if (binaryArchivesToRead) {
        secondDesc->setBinaryArchives(binaryArchivesToRead);
        second = device->newComputePipelineState(secondDesc.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
      }
      if (second == nullptr) {
        error = nil;
        second = device->newComputePipelineState(secondDesc.get(), MTL::PipelineOptionNone, nullptr, &error);
        binaryArchiveMiss = true;
      }
      if (binaryArchiveMiss && binaryArchiveToWrite != nullptr) {
        binaryArchiveToWrite->addComputePipelineFunctions(descriptor.get(), &error);
        binaryArchiveToWrite->addComputePipelineFunctions(secondDesc.get(), &error);
        serializeBinaries(binaryArchiveToWrite, pathToWrite);
      }
      return std::pair(pipeline, second);
    } else {
      auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
      descriptor->setComputeFunction(function.get());
      descriptor->setSupportIndirectCommandBuffers(this->supportIndirectCommandBuffers);
      MTL::ComputePipelineState* pipeline = nullptr;
      if (binaryArchivesToRead) {
        descriptor->setBinaryArchives(binaryArchivesToRead);
        pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
      }
      if (pipeline == nullptr) {
        error = nil;
        pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
        if (binaryArchiveToWrite != nullptr) {
          binaryArchiveToWrite->addComputePipelineFunctions(descriptor.get(), &error);
          serializeBinaries(binaryArchiveToWrite, pathToWrite);
        }
      }
      return std::pair(pipeline, nullptr);
    }
  };

  GEMMOperandPrecision registerPrecisionA = memoryPrecisions.A;
  GEMMOperandPrecision registerPrecisionB = memoryPrecisions.B;
  GEMMOperandPrecision registerPrecisionBias = memoryPrecisions.bias;
  GEMMOperandPrecision registerPrecisionC = this->registerPrecisionC.value_or(GEMMOperandPrecision::FP32);
  if (!this->registerPrecisionC.has_value()) {
    if (memoryPrecisions.A == GEMMOperandPrecision::FP16 &&
        memoryPrecisions.B == GEMMOperandPrecision::FP16 &&
        memoryPrecisions.C == GEMMOperandPrecision::FP16) {
      registerPrecisionC = GEMMOperandPrecision::FP16;
    } else if (memoryPrecisions.A == GEMMOperandPrecision::BF16 &&
        memoryPrecisions.B == GEMMOperandPrecision::BF16 &&
        memoryPrecisions.C == GEMMOperandPrecision::BF16) {
      registerPrecisionC = GEMMOperandPrecision::BF16;
    }
  }
  const GEMMOperandPrecisions registerPrecisions = {
    .A = registerPrecisionA,
    .B = registerPrecisionB,
    .C = registerPrecisionC,
    .bias = registerPrecisionBias,
  };

  uint16_t splitK = this->splitK();
  const bool threadBarrierOverK = NAMatMulDescriptor::threadBarrierOverK(this->matrixDimensions[2], splitK);
  const uint32_t groupMValue = groupM(this->matrixDimensions[0]);
  const uint32_t groupNValue = this->transposeState[1] ? groupN(this->matrixDimensions[1]) : 0;
  auto kernelDesc = NAMatMulKernelDescriptor(simd::ushort3 { 128, 64, 64 }, this->memoryPrecisions, registerPrecisions, splitK, 4, threadBarrierOverK, this->transposeState, this->useBias, this->loadM, groupMValue, groupNValue);
  NAMatMulKernel* kernel = createKernel(kernelDesc);
  auto pipelines = createPipeline(kernel->library.get(), splitK, (this->matrixDimensions[1] % 2) == 0);

  // Force the user to retrieve the return value from the cache. We ensure
  // the cache takes ownership, and the pointer doesn't become a zombie
  // object.
  PipelineValue<NAMatMulKernel>* output;
  if (pipelines.second != nullptr) {
    output = new PipelineValue<NAMatMulKernel> { kernel, NS::TransferPtr(pipelines.first), NS::SharedPtr<MTL::IndirectCommandBuffer>(), NS::SharedPtr<MTL::Function>(), NS::TransferPtr(pipelines.second) };
  } else {
    output = new PipelineValue<NAMatMulKernel> { kernel, NS::TransferPtr(pipelines.first) };
  }
  return std::make_pair(kernelDesc, output);
}
