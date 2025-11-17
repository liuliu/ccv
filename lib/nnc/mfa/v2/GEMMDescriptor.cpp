#include "GEMMDescriptor.hpp"
#include "GEMMKernelDescriptor.hpp"
#include "GEMMKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool GEMMDescriptor::operator==(const GEMMDescriptor& rhs) const {
  return
  (batchDimension == rhs.batchDimension) &&
  simd_all(matrixDimensions == rhs.matrixDimensions) &&
  simd_all(leadingDimensions.value_or(simd::uint3(UINT32_MAX)) == rhs.leadingDimensions.value_or(simd::uint3(UINT32_MAX))) &&
  simd_all(batchStrides.value_or(simd::uint4(UINT32_MAX)) == rhs.batchStrides.value_or(simd::uint4(UINT32_MAX))) &&
  memoryPrecisions == rhs.memoryPrecisions &&
  registerPrecisionC == rhs.registerPrecisionC &&
  simd_all(transposeState == rhs.transposeState) &&
  (useBias == rhs.useBias) &&
  (loadM == rhs.loadM) &&
  (supportIndirectCommandBuffers == rhs.supportIndirectCommandBuffers);
}

std::size_t std::hash<GEMMDescriptor>::operator()(const GEMMDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, hash.batchDimension);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  if (hash.leadingDimensions.has_value()) {
    combine_32(seed, hash.leadingDimensions.value()[0]);
    combine_32(seed, hash.leadingDimensions.value()[1]);
    combine_32(seed, hash.leadingDimensions.value()[2]);
  }
  if (hash.batchStrides.has_value()) {
    combine_32(seed, hash.batchStrides.value()[0]);
    combine_32(seed, hash.batchStrides.value()[1]);
    combine_32(seed, hash.batchStrides.value()[2]);
    combine_32(seed, hash.batchStrides.value()[3]);
  }
  combine_64(seed, pack_64(simd::ushort4 { hash.memoryPrecisions.A.value, hash.memoryPrecisions.B.value, hash.memoryPrecisions.C.value, hash.memoryPrecisions.bias.value }));
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], hash.transposeState[2], 0 }));
  combine_32(seed, pack_32(simd::uchar4 { hash.loadPreviousC, hash.useBias, hash.loadM, hash.supportIndirectCommandBuffers }));
  if (hash.registerPrecisionC.has_value()) {
    combine_32(seed, pack_32(simd::ushort2 { hash.registerPrecisionC.value().value, 0 }));
  }
  return seed;
}

std::pair<GEMMKernelDescriptor, PipelineValue<GEMMKernel> *> GEMMDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<GEMMKernelDescriptor, std::unique_ptr<GEMMKernel>> *const libraryCache) const noexcept {
  // The caller is not responsible for calling 'delete' on this pointer. The
  // reference is saved in the 'libraryCache'. It will be deallocated whenever
  // the shader cache itself is cleaned up.
  auto createKernel =
  [=](GEMMKernelDescriptor descriptor) -> GEMMKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      GEMMKernel* kernel = new GEMMKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<GEMMKernel>(kernel);
      return kernel;
    }
  };

  // WARNING: The owner must explicitly retain the compute pipeline.
  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
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

    auto chooseLeadingDimension =
    [=](unsigned int specifiedLeading, bool transposeState, unsigned int untransposedRows, unsigned int untransposedColumns) -> unsigned int {
      unsigned int expectedLeading;
      if (transposeState) {
        expectedLeading = untransposedRows;
      } else {
        expectedLeading = untransposedColumns;
      }

      unsigned int actualLeading;
      if (specifiedLeading > 0) {
        if (specifiedLeading < expectedLeading) {
          CCV_NNC_MFA_PRECONDITION(false && "Leading block dimension was too small.");
        }
        actualLeading = specifiedLeading;
      } else {
        actualLeading = expectedLeading;
      }

      return actualLeading;
    };

    auto leadingDimensionA = chooseLeadingDimension(
      leadingDimensions.value_or(simd::uint3())[0], transposeState[0],
      matrixDimensions[0], matrixDimensions[2]);
    auto leadingDimensionB = chooseLeadingDimension(
      leadingDimensions.value_or(simd::uint3())[1], transposeState[1],
      matrixDimensions[2], matrixDimensions[1]);
    auto leadingDimensionC = chooseLeadingDimension(
      leadingDimensions.value_or(simd::uint3())[2], false,
      matrixDimensions[0], matrixDimensions[1]);

    constants->setConstantValue(&leadingDimensionA, MTL::DataTypeUInt, 5);
    constants->setConstantValue(&leadingDimensionB, MTL::DataTypeUInt, 6);
    constants->setConstantValue(&leadingDimensionC, MTL::DataTypeUInt, 7);

    bool loadPreviousC = this->loadPreviousC;
    constants->setConstantValue(&loadPreviousC, MTL::DataTypeBool, 10);

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

    NS::String* swiftName = NS::String::string("gemm", NS::UTF8StringEncoding);
    NS::Error* error = nil;
    
    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    if (this->supportIndirectCommandBuffers) {
      auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
      descriptor->setComputeFunction(function.get());
      descriptor->setSupportIndirectCommandBuffers(true);
      auto pipeline = device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      return pipeline;
    } else {
      auto pipeline = device->newComputePipelineState(function.get(), &error);
      CCV_NNC_MFA_CHECK_ERROR(error);
      return pipeline;
    }
  };

  GEMMOperandPrecision registerPrecisionA = memoryPrecisions.A;
  GEMMOperandPrecision registerPrecisionB = memoryPrecisions.B;
  GEMMOperandPrecision registerPrecisionBias = memoryPrecisions.bias;
  GEMMOperandPrecision registerPrecisionC = this->registerPrecisionC.value_or(GEMMOperandPrecision::FP32);
  if (!this->registerPrecisionC.has_value() &&
      memoryPrecisions.A == GEMMOperandPrecision::FP16 &&
      memoryPrecisions.B == GEMMOperandPrecision::FP16 &&
      memoryPrecisions.C == GEMMOperandPrecision::FP16) {
    // If FP16 is causing accuracy issues, you can change this to FP32. Note
    // that doing so cuts out a very important part of the performance
    // spectrum. It is only FP16xFP16->FP16 that reaches peak performance.
    // This statement applies to both the M1 and M3 architectures.
    //
    // FP16xFP16 into FP16 accumulator triggers this instruction:
    // https://github.com/dougallj/applegpu/blob/aeb81519159246d70c56d3f77adb4bc9cca7aa0d/applegpu.py#L3232-L3244
    //
    // FP16xFP16/BF16xBF16 into FP32 accumulator triggers this instruction:
    // https://github.com/dougallj/applegpu/blob/aeb81519159246d70c56d3f77adb4bc9cca7aa0d/applegpu.py#L3195-L3207
    //
    // No other input/output register types map to a native instruction.
    //
    // I would recommend changing the accumulator precision on a case-by-case
    // (operation-by-operation) basis. Provide some mechanism in the high-level
    // API, to control certain low-level features. Without harming execution
    // latency and without imposing technical debt on the high-level API.
    // Definitely NOT a global flag that forces all matrices to change from
    // FP16 -> FP32.
    registerPrecisionC = GEMMOperandPrecision::FP16;
  }
  
  // Set the device and examine the block dimensions.
  auto blockDimensionsAndPaddedBlockDimensions = GEMMKernelDescriptor::getBlockDimensions(device, dprops.coreCount, this->matrixDimensions, this->batchDimension, this->memoryPrecisions, registerPrecisionC, this->transposeState);
  std::optional<bool> preferAsyncStore = std::nullopt;
  bool preferAsyncLoad;
  simd::ushort2 splits;
  if (device->supportsFamily(MTL::GPUFamily(1009))) {
    preferAsyncLoad = false;
    preferAsyncStore = false;
    splits = { 1, 1 };
  } else {
    // For device without native BF16 support, use register at FP32.
    if (memoryPrecisions.A == GEMMOperandPrecision::BF16) {
      registerPrecisionA = GEMMOperandPrecision::FP32;
    }
    if (memoryPrecisions.B == GEMMOperandPrecision::BF16) {
      registerPrecisionB = GEMMOperandPrecision::FP32;
    }
    preferAsyncLoad = true;
    if (simd_all(blockDimensionsAndPaddedBlockDimensions.first == simd::ushort3 { 48, 48, 32 })) {
      preferAsyncStore.reset();
    } else {
      preferAsyncStore = true;
    }
    splits = { 2, 2 };
  }
  const GEMMOperandPrecisions registerPrecisions = {
    .A = registerPrecisionA,
    .B = registerPrecisionB,
    .C = registerPrecisionC,
    .bias = registerPrecisionBias,
  };
  
  // Run a combinatorial search to find the correct value for
  // 'preferAsyncStore'.
  if (preferAsyncStore.has_value()) {
    auto kernelDesc = GEMMKernelDescriptor(blockDimensionsAndPaddedBlockDimensions.first, this->memoryPrecisions, blockDimensionsAndPaddedBlockDimensions.second, preferAsyncLoad, preferAsyncStore.value(), registerPrecisions, splits, this->transposeState, this->useBias, this->loadM);
    GEMMKernel* kernel = createKernel(kernelDesc);
    auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
    
    // Force the user to retrieve the return value from the cache. We ensure
    // the cache takes ownership, and the pointer doesn't become a zombie
    // object.
    PipelineValue<GEMMKernel>* output = new PipelineValue<GEMMKernel> { kernel, pipeline };
    return std::make_pair(kernelDesc, output);
  } else {
    auto kernelDesc = GEMMKernelDescriptor(blockDimensionsAndPaddedBlockDimensions.first, this->memoryPrecisions, blockDimensionsAndPaddedBlockDimensions.second, preferAsyncLoad, false, registerPrecisions, splits, this->transposeState, this->useBias, this->loadM);
    struct Candidate {
      GEMMKernelDescriptor kernelDesc;
      GEMMKernel* kernel;
      NS::SharedPtr<MTL::ComputePipelineState> pipeline;
    };
    std::vector<Candidate> candidates;
    
    for (int8_t candidateID = 0; candidateID < 4; ++candidateID) {
      simd::ushort3 blockDimensions;
      if (candidateID % 2 == 0) {
        blockDimensions = simd::ushort3 { 48, 48, 32 };
      } else {
        blockDimensions = simd::ushort3 { 48, 48, 40 };
      }
      
      bool preferAsyncStore;
      if (candidateID / 2 == 0) {
        preferAsyncStore = false;
      } else {
        preferAsyncStore = true;
      }
      
      // Set the data that's unique to this variant.
      auto newKernelDesc = kernelDesc;
      newKernelDesc.blockDimensions = blockDimensions;
      newKernelDesc.preferAsyncStore = preferAsyncStore;
      
      GEMMKernel* kernel = createKernel(newKernelDesc);
      auto pipeline = NS::TransferPtr
      (createPipeline(kernel->library.get()));
      
      Candidate candidate {
        .kernelDesc = newKernelDesc,
        .kernel = kernel,
        .pipeline = pipeline
      };
      candidates.push_back(candidate);
    }
    
    // Find the maximum occupancy.
    int64_t maximumOccupancy = -1;
    for (Candidate candidate : candidates) {
      int64_t occupancy = candidate.pipeline->maxTotalThreadsPerThreadgroup();
      maximumOccupancy = std::max(maximumOccupancy, occupancy);
    }
    
    // Remove all candidates that don't match this occupancy.
    {
      std::vector<Candidate> newCandidates;
      for (Candidate candidate : candidates) {
        int64_t occupancy = candidate.pipeline->maxTotalThreadsPerThreadgroup();
        if (occupancy != maximumOccupancy) {
          continue;
        }
        newCandidates.push_back(candidate);
      }
      candidates = newCandidates;
    }
    
    // Choose the highest-performing candidate.
    Candidate candidate = candidates[candidates.size() - 1];
    kernelDesc = candidate.kernelDesc;
    
    // Force the user to retrieve the return value from the cache. We ensure
    // the cache takes ownership, and the pointer doesn't become a zombie
    // object.
    PipelineValue<GEMMKernel>* output = new PipelineValue<GEMMKernel> {
      candidate.kernel, candidate.pipeline
    };
    return std::make_pair(candidate.kernelDesc, output);
  }
}
