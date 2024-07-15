#include "GEMMDescriptor.hpp"
#include "CoreCount.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

GEMMKernelKey::GEMMKernelKey(GEMMKernelDescriptor descriptor) {
  blockDimensions = descriptor.blockDimensions.value_or
  (simd::ushort3(UINT16_MAX));
  
  if (descriptor.memoryPrecisions.has_value()) {
    auto precisions = descriptor.memoryPrecisions.value();
    memoryPrecisions = simd::ushort3 {
      precisions.A.value,
      precisions.B.value,
      precisions.C.value,
    };
  } else {
    memoryPrecisions = simd::ushort3(UINT16_MAX);
  }
  paddedBlockDimensions = simd::ushort8(UINT16_MAX);
  if (descriptor.paddedBlockDimensions.has_value()) {
    auto dimensions = descriptor.paddedBlockDimensions.value();
    for (int8_t laneID = 0; laneID < 6; ++laneID) {
      paddedBlockDimensions[laneID] = dimensions[laneID];
    }
  }
  preferAsyncLoad = descriptor.preferAsyncLoad;
  preferAsyncStore = descriptor.preferAsyncStore.value_or(UINT8_MAX);
  
  if (descriptor.registerPrecisions.has_value()) {
    auto precisions = descriptor.registerPrecisions.value();
    registerPrecisions = simd::ushort3 {
      precisions.A.value,
      precisions.B.value,
      precisions.C.value,
    };
  } else {
    registerPrecisions = simd::ushort3(UINT16_MAX);
  }
  splits = descriptor.splits.value_or
  (simd::ushort2(UINT16_MAX));
  transposeState = descriptor.transposeState.value_or
  (simd::uchar2(UINT8_MAX));
}

bool GEMMKernelKey::operator==(const GEMMKernelKey& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  simd_all(memoryPrecisions == rhs.memoryPrecisions) &&
  simd_all(paddedBlockDimensions == rhs.paddedBlockDimensions) &&
  (preferAsyncLoad == rhs.preferAsyncLoad) &&
  (preferAsyncStore == rhs.preferAsyncStore) &&
  simd_all(registerPrecisions == rhs.registerPrecisions) &&
  simd_all(splits == rhs.splits) &&
  simd_all(transposeState == rhs.transposeState);
}

std::size_t std::hash<GEMMKernelKey>::operator()(const GEMMKernelKey& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_64(seed, pack_64(simd_make_ushort4(hash.memoryPrecisions, 0)));
  combine_64(seed, pack_128(hash.preferAsyncStore)[0]);
  combine_64(seed, pack_128(hash.preferAsyncStore)[1]);
  combine_32(seed, pack_32(simd::uchar4 { hash.preferAsyncLoad, hash.preferAsyncStore, 0, 0 }));
  combine_64(seed, pack_64(simd_make_ushort4(hash.registerPrecisions, 0)));
  combine_32(seed, pack_32(hash.splits));
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], 0, 0 }));
  return 0;
}

// MARK: - Initializer

GEMMKernelDescriptor::GEMMKernelDescriptor(GEMMDescriptor descriptor) {
  CCV_NNC_MFA_PRECONDITION(descriptor.matrixDimensions.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.memoryPrecisions.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.transposeState.has_value());
  auto matrixDimensions = descriptor.matrixDimensions.value();
  auto memoryPrecisions = descriptor.memoryPrecisions.value();
  auto transposeState = descriptor.transposeState.value();
  
  // Select the only GPU on an Apple silicon system.
  //
  // NOTE: To avoid potentially costly API calls, you may wish to cache the
  // MTLDevice object or enter a previously created one. The core count
  // could also be cached on macOS.
  //
  // Typical latency to initiate a Metal device, provided the function has
  // been called numerous times prior:
  // - macOS 14
  //   - Swift debug mode,   Metal API validation on:  ≥33 μs
  //   - Swift release mode, Metal API validation off: ≥38 μs
  // - iOS 17
  //   - Swift debug mode,   Metal API validation on:   ≥0 μs
  //   - Swift release mode, Metal API validation off:  ≥0 μs
  auto mtlDevice = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  
  // Trim the device name to something easier to process.
  //
  // M1 Max: Apple M1 Max -> M1
  // M4:     Apple M4 GPU -> M4
  auto createDeviceName = [=]() -> std::string {
    auto swiftDeviceName = mtlDevice->name();
    std::string deviceName(swiftDeviceName->cString(NS::UTF8StringEncoding));
    std::vector<std::string> splits;
    {
      // Manually specify the algorithms for generating splits. C++ does not
      // have ergonomic list manipulation APIs like Swift.
      int64_t cursor = 0;
      for (int64_t i = 0; i <= deviceName.size(); ++i) {
        int8_t character;
        if (i < deviceName.size()) {
          character = deviceName[i];
        } else {
          // Handles the edge case of words positioned exactly at the right
          // end of the string. Without having to duplicate the code for
          // appending a split to the list.
          character = ' ';
        }
        
        if (character == ' ') {
          std::string split;
          for (int64_t characterID = cursor; characterID < i; ++characterID) {
            int8_t character = deviceName[characterID];
            split.push_back(character);
          }
          splits.push_back(split);
          cursor = i + 1;
        }
      }
    }
    
    // Iterate over the space-separated words.
    std::vector<uint32_t> matchingSplitIDs;
    for (int64_t splitID = 0; splitID < splits.size(); ++splitID) {
      // Screen out obvious non-candidates.
      std::string split = splits[splitID];
      if (split.size() < 1) {
        continue;
      }
      if (split[0] == 'A' || split[0] == 'M') {
        // Jump to the next section of code.
      } else {
        continue;
      }
      
      // Extract the second character.
      if (split.size() < 2) {
        continue;
      }
      int8_t secondCharacter = split[1];
      
      // If the second character is numeric, the candidate passes.
      if (isdigit(secondCharacter)) {
        matchingSplitIDs.push_back(uint32_t(splitID));
      }
    }
    CCV_NNC_MFA_PRECONDITION(matchingSplitIDs.size() == 1);
    
    uint32_t splitID = matchingSplitIDs[0];
    return splits[splitID];
  };
  std::string deviceName = createDeviceName();
  
  // Find the core count.
#if TARGET_OS_MAC
  // Typical latency to query IORegistry, provided the function has been
  // called numerous times prior:
  // - macOS 14
  //   - Swift debug mode,   Metal API validation on:  ≥9 μs
  //   - Swift release mode, Metal API validation off: ≥9 μs
  int64_t coreCount = findCoreCount();
#else
  int64_t coreCount;
  CCV_NNC_MFA_PRECONDITION(deviceName.size() >= 1);
  if (deviceName[0] == 'A') {
    if (mtlDevice->supportsFamily(MTL::GPUFamily(1009))) {
      coreCount = 6;
    } else {
      coreCount = 5;
    }
  } else {
    coreCount = 10;
  }
#endif
  
  // Select the register precisions.
  GEMMOperandPrecision registerPrecisionA = memoryPrecisions.A;
  GEMMOperandPrecision registerPrecisionB = memoryPrecisions.B;
  GEMMOperandPrecision registerPrecisionC = GEMMOperandPrecision::FP32;
  if (memoryPrecisions.A == GEMMOperandPrecision::FP16 &&
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
  if (!mtlDevice->supportsFamily(MTL::GPUFamily(1009))) {
    if (memoryPrecisions.A == GEMMOperandPrecision::BF16) {
      registerPrecisionA = GEMMOperandPrecision::FP32;
    }
    if (memoryPrecisions.B == GEMMOperandPrecision::BF16) {
      registerPrecisionB = GEMMOperandPrecision::FP32;
    }
  }
  
  // Set the properties of the 'GEMMKernelDescriptor' object.
  this->memoryPrecisions = memoryPrecisions;
  if (mtlDevice->supportsFamily(MTL::GPUFamily(1009))) {
    preferAsyncLoad = false;
  } else {
    preferAsyncLoad = true;
  }
  this->registerPrecisions = {
    .A = registerPrecisionA,
    .B = registerPrecisionB,
    .C = registerPrecisionC,
  };
  if (!mtlDevice->supportsFamily(MTL::GPUFamily(1009))) {
    splits = simd::ushort2 { 2, 2 };
  } else {
    splits = simd::ushort2 { 1, 1 };
  }
  this->transposeState = transposeState;
  
  // Set the properties that deal with block size.
  setBlockDimensions
  (mtlDevice.get(), coreCount, matrixDimensions, descriptor.batchDimension);
}

void GEMMKernelDescriptor::setBlockDimensions
(MTL::Device* mtlDevice,
 int64_t coreCount,
 simd::uint3 matrixDimensions,
 int64_t batchDimension)
{
  CCV_NNC_MFA_PRECONDITION(memoryPrecisions.has_value());
  CCV_NNC_MFA_PRECONDITION(transposeState.has_value());
  auto memoryPrecisions = this->memoryPrecisions.value();
  auto transposeState = this->transposeState.value();
  
  if (mtlDevice->supportsFamily(MTL::GPUFamily(1009))) {
    blockDimensions = simd::ushort3 { 32, 32, 8 };
    return;
  }
  
  // Find the actual number of threadgroups, with a large block size.
  auto ceilDivide =
  [=](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + uint32_t(granularity) - 1) / uint32_t(granularity);
  };
  int64_t actualGroups = 1;
  actualGroups *= ceilDivide(matrixDimensions[0], 48);
  actualGroups *= ceilDivide(matrixDimensions[1], 48);
  actualGroups *= batchDimension;
  
  // Does the kernel use 48x48x24xFP32 (9 KB) or 48x48x32xFP16/BF16 (6 KB)?
  bool useLargeAllocation = false;
  if (memoryPrecisions.A == GEMMOperandPrecision::FP32 ||
      memoryPrecisions.B == GEMMOperandPrecision::FP32 ||
      memoryPrecisions.C == GEMMOperandPrecision::FP32) {
    useLargeAllocation = true;
  }
  
  // Branch on whether the allocation is large / target occupancy is low.
  if (useLargeAllocation) {
    auto idealGroups = coreCount * 6;
    if (actualGroups <= idealGroups) {
      blockDimensions = simd::ushort3 { 32, 32, 32 };
    } else {
      blockDimensions = simd::ushort3 { 48, 48, 24 };
    }
    
    // This is verified to be optimal for:
    // - (memA, memB, memC) = (FP32, FP32, FP32)
    // - (memA, memB, memC) = (FP16, FP16, FP32)
    // - (memA, memB, memC) = (FP16, FP32, FP32)
    // - (memA, memB, memC) = (FP16, FP32, FP16)
    if (transposeState[0] == false && transposeState[1] == false) {
      paddedBlockDimensions = simd::ushort8 { 48, 24, 24, 48, 48, 48 };
    } else if (transposeState[0] == false && transposeState[1] == true) {
      if (memoryPrecisions.B == GEMMOperandPrecision::FP32) {
        paddedBlockDimensions = simd::ushort8 { 48, 24, 28, 48, 48, 48 };
      } else {
        paddedBlockDimensions = simd::ushort8 { 48, 24, 24, 48, 48, 48 };
      }
    } else {
      if (memoryPrecisions.A == GEMMOperandPrecision::FP32) {
        paddedBlockDimensions = simd::ushort8 { 52, 24, 24, 48, 48, 48 };
      } else {
        paddedBlockDimensions = simd::ushort8 { 56, 24, 24, 48, 48, 48 };
      }
    }
  } else {
    auto idealGroups = coreCount * 9;
    if (actualGroups <= idealGroups) {
      blockDimensions = simd::ushort3 { 32, 32, 32 };
    } else {
      blockDimensions = simd::ushort3 { 48, 48, 32 };
    }
  }
  
  // Check that the block dimensions property has been initialized.
  CCV_NNC_MFA_PRECONDITION(blockDimensions.has_value());
}
