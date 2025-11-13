#include "GEMMKernel.hpp"
#include "GEMMHeaders.hpp"
#include <algorithm>
extern "C" {
#include "GEMMKernel+Precompiled.inc"
#include <simd/simd.h>
}

MTL::Library* GEMMKernel::findPrecompiledLibrary(GEMMKernelDescriptor descriptor, MTL::Device *const device, NS::Error **error) const noexcept {
  if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m0_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m0_macosx_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m1_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m1_macosx_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m0_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m0_macosx_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m1_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m1_macosx_metallib, sizeof(b32x32x8_p32x32x32_half_cfloat_l0_s0_s1x1_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m0_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m0_macosx_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m1_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m1_macosx_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m0_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m0_macosx_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m1_iphoneos_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m1_macosx_metallib, sizeof(b32x32x8_p32x32x32_bfloat_cfloat_l0_s0_s1x1_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 0 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a0_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 0 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b0_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_chalf_l0_s0_s1x1_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP16 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_chalf_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_half_cfloat_l0_s0_s1x1_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::FP16 && memoryPrecisions.B == GEMMOperandPrecision::FP16 && memoryPrecisions.C == GEMMOperandPrecision::FP16 && memoryPrecisions.bias == GEMMOperandPrecision::FP16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP16 && registerPrecisions.B == GEMMOperandPrecision::FP16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_half_ahalf_bhalf_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m0_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m1_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 0 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b0_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m0_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m0_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 0) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m0_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == false && preferAsyncStore == false &&
    splits[0] == 1 && splits[1] == 1 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m1_iphoneos_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m1_macosx_metallib, sizeof(b32x32x8_pnil_bfloat_cfloat_l0_s0_s1x1_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == false &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s0_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x32_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;

  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
    memoryPrecisions.A == GEMMOperandPrecision::BF16 && memoryPrecisions.B == GEMMOperandPrecision::BF16 && memoryPrecisions.C == GEMMOperandPrecision::BF16 && memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
    registerPrecisions.A == GEMMOperandPrecision::FP32 && registerPrecisions.B == GEMMOperandPrecision::FP32 && registerPrecisions.C == GEMMOperandPrecision::FP32 &&
    preferAsyncLoad == true && preferAsyncStore == true &&
    splits[0] == 2 && splits[1] == 2 &&
    transposeState[0] == 1 && transposeState[1] == 1 && transposeState[2] == false &&
    useBias == 1 && loadM == 1) {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_iphoneos_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib, sizeof(b48x48x40_pnil_bfloat_afloat_bfloat_cfloat_l1_s1_s2x2_a1_b1_b1_m1_macosx_metallib), NULL, 0);
#endif
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
  }
  return 0;
}
