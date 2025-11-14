#include "AttentionKernel.hpp"
#include <algorithm>
extern "C" {
#include "AttentionKernel+Precompiled.inc"
#include <simd/simd.h>
}

MTL::Library* AttentionKernel::findPrecompiledLibrary(AttentionKernelDescriptor descriptor, MTL::Device *const device, NS::Error **error) const noexcept {
  if (transposeState[AttentionOperand::Q].value_or(true) ||
      transposeState[AttentionOperand::K].value_or(true) ||
      transposeState[AttentionOperand::V].value_or(true) ||
      transposeState[AttentionOperand::O].value_or(true) ||
      transposeState[AttentionOperand::dO].value_or(true) ||
      transposeState[AttentionOperand::dV].value_or(true) ||
      transposeState[AttentionOperand::dK].value_or(true) ||
      transposeState[AttentionOperand::dQ].value_or(true) ||
      !leadingDimensions[AttentionOperand::Q].value_or(true) ||
      !leadingDimensions[AttentionOperand::K].value_or(true) ||
      !leadingDimensions[AttentionOperand::V].value_or(true) ||
      !leadingDimensions[AttentionOperand::O].value_or(true) ||
      !leadingDimensions[AttentionOperand::dO].value_or(true) ||
      !leadingDimensions[AttentionOperand::dV].value_or(true) ||
      !leadingDimensions[AttentionOperand::dK].value_or(true) ||
      !leadingDimensions[AttentionOperand::dQ].value_or(true)) { // Only precompiled versions with transposeState = false and leadingDimensions = true.
    return 0;
  }
  // Not low precision inputs.
  if (memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::FP32 ||
      memoryPrecisions[AttentionOperand::K].value() == GEMMOperandPrecision::FP32 ||
      memoryPrecisions[AttentionOperand::V].value() == GEMMOperandPrecision::FP32 ||
      memoryPrecisions[AttentionOperand::dO].value() == GEMMOperandPrecision::FP32) {
    return 0;
  }
  bool isBF16 = memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16 ||
    memoryPrecisions[AttentionOperand::K].value() == GEMMOperandPrecision::BF16 ||
    memoryPrecisions[AttentionOperand::V].value() == GEMMOperandPrecision::BF16 ||
    memoryPrecisions[AttentionOperand::dO].value() == GEMMOperandPrecision::BF16;
  bool lowPrecisionIntermediates = memoryPrecisions[AttentionOperand::L].value() != GEMMOperandPrecision::FP32 ||
    memoryPrecisions[AttentionOperand::D].value() != GEMMOperandPrecision::FP32;
  if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h40_i1_t1_cqo_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x128x32_h40_i1_t1_cqo_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h40_i1_t1_cqo_b0_c0_l1_macosx_metallib, sizeof(f_b32x128x32_h40_i1_t1_cqo_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h40_i1_t1_cdq_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x32_h40_i1_t1_cdq_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h40_i1_t1_cdq_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x32_h40_i1_t1_cdq_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 32 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x32_h40_i1_t1_cdv_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x32_h40_i1_t1_cdv_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x32_h40_i1_t1_cdv_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x32_h40_i1_t1_cdv_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h64_i1_t1_cqo_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x128x32_h64_i1_t1_cqo_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h64_i1_t1_cqo_b0_c0_l1_macosx_metallib, sizeof(f_b32x128x32_h64_i1_t1_cqo_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h64_i1_t1_cdq_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x32_h64_i1_t1_cdq_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h64_i1_t1_cdq_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x32_h64_i1_t1_cdq_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h64_i1_t1_cdv_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x64x32_h64_i1_t1_cdv_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h64_i1_t1_cdv_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x64x32_h64_i1_t1_cdv_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h80_i1_t1_cqo_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x128x32_h80_i1_t1_cqo_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h80_i1_t1_cqo_b0_c0_l1_macosx_metallib, sizeof(f_b32x128x32_h80_i1_t1_cqo_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h80_i1_t1_cdq_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x32_h80_i1_t1_cdq_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h80_i1_t1_cdq_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x32_h80_i1_t1_cdq_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h80_i1_t1_cdv_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x64x32_h80_i1_t1_cdv_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h80_i1_t1_cdv_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x64x32_h80_i1_t1_cdv_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h128_i1_t1_cq_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x128x32_h128_i1_t1_cq_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h128_i1_t1_cq_b0_c0_l1_macosx_metallib, sizeof(f_b32x128x32_h128_i1_t1_cq_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h128_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x32_h128_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h128_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x32_h128_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h128_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x64x32_h128_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h128_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x64x32_h128_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h160_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x128x32_h160_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h160_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x128x32_h160_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h160_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x32_h160_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h160_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x32_h160_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h160_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x64x32_h160_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h160_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x64x32_h160_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h256_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x128x32_h256_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x128x32_h256_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x128x32_h256_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h256_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x32_h256_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x32_h256_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x32_h256_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h256_i1_t1_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x64x32_h256_i1_t1_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x64x32_h256_i1_t1_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x64x32_h256_i1_t1_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h40_i1_t1_cqo_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x32_h40_i1_t1_cqo_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h40_i1_t1_cqo_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x32_h40_i1_t1_cqo_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 8 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x8_h40_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x8_h40_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x8_h40_i1_t1_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x8_h40_i1_t1_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 8 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x64x8_h40_i1_t1_ckvdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x64x8_h40_i1_t1_ckvdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x64x8_h40_i1_t1_ckvdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x64x8_h40_i1_t1_ckvdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h64_i1_t1_cqo_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x32_h64_i1_t1_cqo_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h64_i1_t1_cqo_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x32_h64_i1_t1_cqo_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 8 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x8_h64_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x8_h64_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x8_h64_i1_t1_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x8_h64_i1_t1_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h64_i1_t1_cvdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x32x16_h64_i1_t1_cvdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h64_i1_t1_cvdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x32x16_h64_i1_t1_cvdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h80_i1_t1_cqo_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x32_h80_i1_t1_cqo_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h80_i1_t1_cqo_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x32_h80_i1_t1_cqo_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 8 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x8_h80_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x8_h80_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x8_h80_i1_t1_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x8_h80_i1_t1_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h80_i1_t1_cvdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x32x16_h80_i1_t1_cvdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h80_i1_t1_cvdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x32x16_h80_i1_t1_cvdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h128_i1_t1_co_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x32_h128_i1_t1_co_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h128_i1_t1_co_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x32_h128_i1_t1_co_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h128_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h128_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h128_i1_t1_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h128_i1_t1_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h128_i1_t1_cdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h128_i1_t1_cdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h128_i1_t1_cdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h128_i1_t1_cdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h160_i1_t1_co_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x32_h160_i1_t1_co_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h160_i1_t1_co_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x32_h160_i1_t1_co_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h160_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h160_i1_t1_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h160_i1_t1_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h160_i1_t1_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h160_i1_t1_cdv_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h160_i1_t1_cdv_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h160_i1_t1_cdv_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h160_i1_t1_cdv_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h256_i1_t1_c_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x32_h256_i1_t1_c_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x32_h256_i1_t1_c_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x32_h256_i1_t1_c_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x128x32_h256_i1_t1_c_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x128x32_h256_i1_t1_c_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x128x32_h256_i1_t1_c_b0_c1_l0_macosx_metallib, sizeof(bq_b16x128x32_h256_i1_t1_c_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 32 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 1 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x32_h256_i1_t1_c_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x32_h256_i1_t1_c_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x32_h256_i1_t1_c_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x32_h256_i1_t1_c_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 40 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x32x40_h40_i1_t0_cq_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x32x40_h40_i1_t0_cq_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x32x40_h40_i1_t0_cq_b0_c0_l1_macosx_metallib, sizeof(f_b32x32x40_h40_i1_t0_cq_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 24 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x24_h40_i1_t0_cdq_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x24_h40_i1_t0_cdq_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x24_h40_i1_t0_cdq_b0_c0_l1_macosx_metallib, sizeof(bq_b32x64x24_h40_i1_t0_cdq_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h40_i1_t0_cdv_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h40_i1_t0_cdv_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h40_i1_t0_cdv_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h40_i1_t0_cdv_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h64_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h64_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h64_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h64_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h64_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h64_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h64_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h64_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h64_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h64_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h64_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h64_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h80_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h80_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h80_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h80_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h80_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h80_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h80_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h80_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h80_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h80_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h80_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h80_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h128_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h128_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h128_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h128_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h128_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h128_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h128_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h128_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h128_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h128_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h128_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h128_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h160_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h160_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h160_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h160_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h160_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h160_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h160_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h160_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h160_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h160_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h160_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h160_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h256_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h256_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h256_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h256_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h256_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h256_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h256_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h256_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h256_i1_t0_c_b0_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h256_i1_t0_c_b0_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h256_i1_t0_c_b0_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h256_i1_t0_c_b0_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 40 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x32x40_h40_i1_t0_cq_b1_c0_l1_iphoneos_metallib, sizeof(f_b32x32x40_h40_i1_t0_cq_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x32x40_h40_i1_t0_cq_b1_c0_l1_macosx_metallib, sizeof(f_b32x32x40_h40_i1_t0_cq_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 64 && blockDimensions[2] == 24 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x64x24_h40_i1_t0_cdq_b1_c0_l1_iphoneos_metallib, sizeof(bq_b32x64x24_h40_i1_t0_cdq_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x64x24_h40_i1_t0_cdq_b1_c0_l1_macosx_metallib, sizeof(bq_b32x64x24_h40_i1_t0_cdq_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h40_i1_t0_cdv_b1_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h40_i1_t0_cdv_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h40_i1_t0_cdv_b1_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h40_i1_t0_cdv_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h64_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h64_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h64_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h64_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h64_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h64_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h64_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h64_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h64_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h64_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h64_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h64_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h80_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h80_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h80_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h80_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h80_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h80_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h80_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h80_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h80_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h80_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h80_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h80_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h128_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h128_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h128_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h128_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h128_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h128_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h128_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h128_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h128_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h128_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h128_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h128_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h160_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h160_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h160_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h160_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h160_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h160_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h160_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h160_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h160_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h160_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h160_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h160_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h256_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(f_b32x80x16_h256_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b32x80x16_h256_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(f_b32x80x16_h256_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h256_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bq_b32x80x16_h256_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b32x80x16_h256_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bq_b32x80x16_h256_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 32 && blockDimensions[1] == 80 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 0 && preferAsyncLoad == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h256_i1_t0_c_b1_c0_l1_iphoneos_metallib, sizeof(bkv_b32x80x16_h256_i1_t0_c_b1_c0_l1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b32x80x16_h256_i1_t0_c_b1_c0_l1_macosx_metallib, sizeof(bkv_b32x80x16_h256_i1_t0_c_b1_c0_l1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x32x8_h40_i1_t0_cqo_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x32x8_h40_i1_t0_cqo_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x32x8_h40_i1_t0_cqo_b0_c1_l0_macosx_metallib, sizeof(f_b16x32x8_h40_i1_t0_cqo_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h40_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h40_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h40_i1_t0_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h40_i1_t0_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 16 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h40_i1_t0_cvdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x32x16_h40_i1_t0_cvdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h40_i1_t0_cvdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x32x16_h40_i1_t0_cvdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h64_i1_t0_co_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h64_i1_t0_co_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h64_i1_t0_co_b0_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h64_i1_t0_co_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h64_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h64_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h64_i1_t0_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h64_i1_t0_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h64_i1_t0_cvdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x32x16_h64_i1_t0_cvdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h64_i1_t0_cvdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x32x16_h64_i1_t0_cvdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h80_i1_t0_co_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h80_i1_t0_co_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h80_i1_t0_co_b0_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h80_i1_t0_co_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h80_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h80_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h80_i1_t0_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h80_i1_t0_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h80_i1_t0_cdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h80_i1_t0_cdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h80_i1_t0_cdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h80_i1_t0_cdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h128_i1_t0_co_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h128_i1_t0_co_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h128_i1_t0_co_b0_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h128_i1_t0_co_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h128_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h128_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h128_i1_t0_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h128_i1_t0_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h128_i1_t0_cdvdk_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h128_i1_t0_cdvdk_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h128_i1_t0_cdvdk_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h128_i1_t0_cdvdk_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h160_i1_t0_co_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h160_i1_t0_co_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h160_i1_t0_co_b0_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h160_i1_t0_co_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h160_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h160_i1_t0_cqdq_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h160_i1_t0_cqdq_b0_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h160_i1_t0_cqdq_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h160_i1_t0_cdv_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h160_i1_t0_cdv_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h160_i1_t0_cdv_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h160_i1_t0_cdv_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x16_h256_i1_t0_c_b0_c1_l0_iphoneos_metallib, sizeof(f_b16x128x16_h256_i1_t0_c_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x16_h256_i1_t0_c_b0_c1_l0_macosx_metallib, sizeof(f_b16x128x16_h256_i1_t0_c_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x128x16_h256_i1_t0_c_b0_c1_l0_iphoneos_metallib, sizeof(bq_b16x128x16_h256_i1_t0_c_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x128x16_h256_i1_t0_c_b0_c1_l0_macosx_metallib, sizeof(bq_b16x128x16_h256_i1_t0_c_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 0 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h256_i1_t0_c_b0_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h256_i1_t0_c_b0_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h256_i1_t0_c_b0_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h256_i1_t0_c_b0_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x32x8_h40_i1_t0_cqo_b1_c1_l0_iphoneos_metallib, sizeof(f_b16x32x8_h40_i1_t0_cqo_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x32x8_h40_i1_t0_cqo_b1_c1_l0_macosx_metallib, sizeof(f_b16x32x8_h40_i1_t0_cqo_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h40_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h40_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h40_i1_t0_cqdq_b1_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h40_i1_t0_cqdq_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 16 &&
    headDimension == 40 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h40_i1_t0_cvdvdk_b1_c1_l0_iphoneos_metallib, sizeof(bkv_b16x32x16_h40_i1_t0_cvdvdk_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h40_i1_t0_cvdvdk_b1_c1_l0_macosx_metallib, sizeof(bkv_b16x32x16_h40_i1_t0_cvdvdk_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h64_i1_t0_co_b1_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h64_i1_t0_co_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h64_i1_t0_co_b1_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h64_i1_t0_co_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h64_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h64_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h64_i1_t0_cqdq_b1_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h64_i1_t0_cqdq_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 32 && blockDimensions[2] == 16 &&
    headDimension == 64 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h64_i1_t0_cvdvdk_b1_c1_l0_iphoneos_metallib, sizeof(bkv_b16x32x16_h64_i1_t0_cvdvdk_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x32x16_h64_i1_t0_cvdvdk_b1_c1_l0_macosx_metallib, sizeof(bkv_b16x32x16_h64_i1_t0_cvdvdk_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h80_i1_t0_co_b1_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h80_i1_t0_co_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h80_i1_t0_co_b1_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h80_i1_t0_co_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h80_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h80_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h80_i1_t0_cqdq_b1_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h80_i1_t0_cqdq_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 80 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h80_i1_t0_cdvdk_b1_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h80_i1_t0_cdvdk_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h80_i1_t0_cdvdk_b1_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h80_i1_t0_cdvdk_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h128_i1_t0_co_b1_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h128_i1_t0_co_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h128_i1_t0_co_b1_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h128_i1_t0_co_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h128_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h128_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h128_i1_t0_cqdq_b1_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h128_i1_t0_cqdq_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 128 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h128_i1_t0_cdvdk_b1_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h128_i1_t0_cdvdk_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h128_i1_t0_cdvdk_b1_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h128_i1_t0_cdvdk_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h160_i1_t0_co_b1_c1_l0_iphoneos_metallib, sizeof(f_b16x64x16_h160_i1_t0_co_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x64x16_h160_i1_t0_co_b1_c1_l0_macosx_metallib, sizeof(f_b16x64x16_h160_i1_t0_co_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 64 && blockDimensions[2] == 32 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h160_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib, sizeof(bq_b16x64x32_h160_i1_t0_cqdq_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x64x32_h160_i1_t0_cqdq_b1_c1_l0_macosx_metallib, sizeof(bq_b16x64x32_h160_i1_t0_cqdq_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 160 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h160_i1_t0_cdv_b1_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h160_i1_t0_cdv_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h160_i1_t0_cdv_b1_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h160_i1_t0_cdv_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::forward &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(f_b16x128x16_h256_i1_t0_c_b1_c1_l0_iphoneos_metallib, sizeof(f_b16x128x16_h256_i1_t0_c_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(f_b16x128x16_h256_i1_t0_c_b1_c1_l0_macosx_metallib, sizeof(f_b16x128x16_h256_i1_t0_c_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardQuery &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bq_b16x128x16_h256_i1_t0_c_b1_c1_l0_iphoneos_metallib, sizeof(bq_b16x128x16_h256_i1_t0_c_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bq_b16x128x16_h256_i1_t0_c_b1_c1_l0_macosx_metallib, sizeof(bq_b16x128x16_h256_i1_t0_c_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (type.value == AttentionKernelType::backwardKeyValue &&
    blockDimensions[0] == 16 && blockDimensions[1] == 128 && blockDimensions[2] == 16 &&
    headDimension == 256 &&
    lowPrecisionIntermediates == 0 && isBF16 == 1 &&
    preferAsyncCache == 1 && preferAsyncLoad == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h256_i1_t0_c_b1_c1_l0_iphoneos_metallib, sizeof(bkv_b16x128x16_h256_i1_t0_c_b1_c1_l0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(bkv_b16x128x16_h256_i1_t0_c_b1_c1_l0_macosx_metallib, sizeof(bkv_b16x128x16_h256_i1_t0_c_b1_c1_l0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
  }
  return 0;
}
