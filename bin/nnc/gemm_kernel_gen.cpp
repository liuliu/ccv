extern "C" {
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <sys/time.h>
#include <ctype.h>
}
#include "nnc/mfa/kernels/ShaderCache.hpp"
#include "nnc/mfa/kernels/GEMMDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernelDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernel.hpp"
#include "3rdparty/dsfmt/dSFMT.h"
#include <iostream>

static std::string name(GEMMOperandPrecision value) {
	if (value == GEMMOperandPrecision::FP16) {
		return "GEMMOperandPrecision::FP16";
	} else if (value == GEMMOperandPrecision::BF16) {
		return "GEMMOperandPrecision::BF16";
	}
	return "GEMMOperandPrecision::FP32";
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	{
		NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
		// Loop through precisions, useBias, loadM, transposeStates
		bool transposeStates[] = {
			false, false,
			false, true,
			true, false,
			true, true,
		};
		bool useBias[] = {
			false,
			true,
		};
		bool loadM[] = {
			false,
			true,
		};
		GEMMOperandPrecision memoryPrecisions[] = {
			GEMMOperandPrecision::FP16,
			GEMMOperandPrecision::BF16
		};
		int i, j, k, l, m;
		for (i = 0; i < 4; i++)
			for (j = 0; j < 2; j++)
				for (k = 0; k < 2; k++)
					for (l = 0; l < 2; l++)
						for (m = 0; m < 2; m++)
						{
							GEMMOperandPrecision registerPrecisionC = m == 0 ? memoryPrecisions[j] : GEMMOperandPrecision::FP32;
							if (registerPrecisionC == GEMMOperandPrecision::BF16)
								continue;
							// M3 kernels
							if (!transposeStates[i * 2] && transposeStates[i * 2 + 1] && registerPrecisionC == GEMMOperandPrecision::FP32)
							{
								GEMMKernelDescriptor kernelDesc = GEMMKernelDescriptor(
									simd::ushort3 { 32, 32, 8 }, // Block dimensions
									(GEMMOperandPrecisions){ // Memory precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = memoryPrecisions[j], .bias = memoryPrecisions[j]
									},
									simd::ushort3 { 32, 32, 32 }, // Padded block dimensions
									false, false, // preferAsyncLoad, preferAsyncStore
									(GEMMOperandPrecisions){ // Register precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = registerPrecisionC, .bias = memoryPrecisions[j]
									},
									simd::ushort2 { 1, 1 }, // splits
									simd::uchar3 { transposeStates[i * 2], transposeStates[i * 2 + 1], false }, // transpose states
									useBias[k], loadM[l]); // useBias, loadM
								std::string file = std::string("b32x32x8_") + "p32x32x32_" + memoryPrecisions[j].name() + "_c" + registerPrecisionC.name() + "_l0_s0_" + "s1x1_" + "a" + std::to_string(transposeStates[i * 2]) + "_b" + std::to_string(transposeStates[i * 2 + 1]) + "_b" + std::to_string(useBias[k]) + "_m" + std::to_string(loadM[l]);
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								*/
								std::cout << R"(
  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3 { 32, 32, 32 }) &&
)";
								std::cout << "    memoryPrecisions.A == " << name(memoryPrecisions[j]) << " && memoryPrecisions.B == " << name(memoryPrecisions[j]) << " && memoryPrecisions.C == " << name(memoryPrecisions[j]) << " && memoryPrecisions.bias == " << name(memoryPrecisions[j]) << " &&" << std::endl;
								std::cout << "    registerPrecisions.A == " << name(memoryPrecisions[j]) << " && registerPrecisions.B == " << name(memoryPrecisions[j]) << " && registerPrecisions.C == " << name(registerPrecisionC) << " &&" << std::endl;
								std::cout << "    preferAsyncLoad == false && preferAsyncStore == false &&" << std::endl;
								std::cout << "    splits[0] == 1 && splits[1] == 1 &&" << std::endl;
								std::cout << "    transposeState[0] == " << transposeStates[i * 2] << " && transposeState[1] == " << transposeStates[i * 2 + 1] << " && transposeState[2] == false &&" << std::endl;
								std::cout << "    useBias == " << useBias[k] << " && loadM == " << loadM[l] << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
								auto kernel = new GEMMKernel(kernelDesc, device.get());
								delete kernel;
							} else {
								GEMMKernelDescriptor kernelDesc = GEMMKernelDescriptor(
									simd::ushort3 { 32, 32, 8 }, // Block dimensions
									(GEMMOperandPrecisions){ // Memory precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = memoryPrecisions[j], .bias = memoryPrecisions[j]
									},
									std::nullopt, // Padded block dimensions
									false, false, // preferAsyncLoad, preferAsyncStore
									(GEMMOperandPrecisions){ // Register precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = registerPrecisionC, .bias = memoryPrecisions[j]
									},
									simd::ushort2 { 1, 1 }, // splits
									simd::uchar3 { transposeStates[i * 2], transposeStates[i * 2 + 1], false }, // transpose states
									useBias[k], loadM[l]); // useBias, loadM
								std::string file = std::string("b32x32x8_") + "pnil_" + memoryPrecisions[j].name() + "_c" + registerPrecisionC.name() + "_l0_s0_" + "s1x1_" + "a" + std::to_string(transposeStates[i * 2]) + "_b" + std::to_string(transposeStates[i * 2 + 1]) + "_b" + std::to_string(useBias[k]) + "_m" + std::to_string(loadM[l]);
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								*/
								std::cout << R"(
  } else if (blockDimensions[0] == 32 && blockDimensions[1] == 32 && blockDimensions[2] == 8 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
)";
								std::cout << "    memoryPrecisions.A == " << name(memoryPrecisions[j]) << " && memoryPrecisions.B == " << name(memoryPrecisions[j]) << " && memoryPrecisions.C == " << name(memoryPrecisions[j]) << " && memoryPrecisions.bias == " << name(memoryPrecisions[j]) << " &&" << std::endl;
								std::cout << "    registerPrecisions.A == " << name(memoryPrecisions[j]) << " && registerPrecisions.B == " << name(memoryPrecisions[j]) << " && registerPrecisions.C == " << name(registerPrecisionC) << " &&" << std::endl;
								std::cout << "    preferAsyncLoad == false && preferAsyncStore == false &&" << std::endl;
								std::cout << "    splits[0] == 1 && splits[1] == 1 &&" << std::endl;
								std::cout << "    transposeState[0] == " << transposeStates[i * 2] << " && transposeState[1] == " << transposeStates[i * 2 + 1] << " && transposeState[2] == false &&" << std::endl;
								std::cout << "    useBias == " << useBias[k] << " && loadM == " << loadM[l] << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
								auto kernel = new GEMMKernel(kernelDesc, device.get());
								delete kernel;
							}
							// M1 / M2 kernels.
							GEMMOperandPrecision registerPrecisionA = memoryPrecisions[j];
							GEMMOperandPrecision registerPrecisionB = memoryPrecisions[j];
							if (registerPrecisionA == GEMMOperandPrecision::BF16)
								registerPrecisionA = GEMMOperandPrecision::FP32;
							if (registerPrecisionB == GEMMOperandPrecision::BF16)
								registerPrecisionB = GEMMOperandPrecision::FP32;
							{
								GEMMKernelDescriptor kernelDesc = GEMMKernelDescriptor(
									simd::ushort3 { 48, 48, 32 }, // Block dimensions
									(GEMMOperandPrecisions){ // Memory precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = memoryPrecisions[j], .bias = memoryPrecisions[j]
									},
									std::nullopt, // Padded block dimensions
									true, false, // preferAsyncLoad, preferAsyncStore
									(GEMMOperandPrecisions){ // Register precisions
										.A = registerPrecisionA, .B = registerPrecisionB, .C = registerPrecisionC, .bias = memoryPrecisions[j]
									},
									simd::ushort2 { 2, 2 }, // splits
									simd::uchar3 { transposeStates[i * 2], transposeStates[i * 2 + 1], false }, // transpose states
									useBias[k], loadM[l]); // useBias, loadM
								std::string file = std::string("b48x48x32_") + "pnil_" + memoryPrecisions[j].name() + "_a" + registerPrecisionA.name() + "_b" + registerPrecisionB.name() + "_c" + registerPrecisionC.name() + "_l1_s0_" + "s2x2_" + "a" + std::to_string(transposeStates[i * 2]) + "_b" + std::to_string(transposeStates[i * 2 + 1]) + "_b" + std::to_string(useBias[k]) + "_m" + std::to_string(loadM[l]);
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								*/
								std::cout << R"(
  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
)";
								std::cout << "    memoryPrecisions.A == " << name(memoryPrecisions[j]) << " && memoryPrecisions.B == " << name(memoryPrecisions[j]) << " && memoryPrecisions.C == " << name(memoryPrecisions[j]) << " && memoryPrecisions.bias == " << name(memoryPrecisions[j]) << " &&" << std::endl;
								std::cout << "    registerPrecisions.A == " << name(registerPrecisionA) << " && registerPrecisions.B == " << name(registerPrecisionB) << " && registerPrecisions.C == " << name(registerPrecisionC) << " &&" << std::endl;
								std::cout << "    preferAsyncLoad == true && preferAsyncStore == false &&" << std::endl;
								std::cout << "    splits[0] == 2 && splits[1] == 2 &&" << std::endl;
								std::cout << "    transposeState[0] == " << transposeStates[i * 2] << " && transposeState[1] == " << transposeStates[i * 2 + 1] << " && transposeState[2] == false &&" << std::endl;
								std::cout << "    useBias == " << useBias[k] << " && loadM == " << loadM[l] << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
								auto kernel = new GEMMKernel(kernelDesc, device.get());
								delete kernel;
							}
							{
								GEMMKernelDescriptor kernelDesc = GEMMKernelDescriptor(
									simd::ushort3 { 48, 48, 40 }, // Block dimensions
									(GEMMOperandPrecisions){ // Memory precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = memoryPrecisions[j], .bias = memoryPrecisions[j]
									},
									std::nullopt, // Padded block dimensions
									true, false, // preferAsyncLoad, preferAsyncStore
									(GEMMOperandPrecisions){ // Register precisions
										.A = registerPrecisionA, .B = registerPrecisionB, .C = registerPrecisionC, .bias = memoryPrecisions[j]
									},
									simd::ushort2 { 2, 2 }, // splits
									simd::uchar3 { transposeStates[i * 2], transposeStates[i * 2 + 1], false }, // transpose states
									useBias[k], loadM[l]); // useBias, loadM
								std::string file = std::string("b48x48x40_") + "pnil_" + memoryPrecisions[j].name() + "_a" + registerPrecisionA.name() + "_b" + registerPrecisionB.name() + "_c" + registerPrecisionC.name() + "_l1_s0_" + "s2x2_" + "a" + std::to_string(transposeStates[i * 2]) + "_b" + std::to_string(transposeStates[i * 2 + 1]) + "_b" + std::to_string(useBias[k]) + "_m" + std::to_string(loadM[l]);
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								*/
								std::cout << R"(
  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
)";
								std::cout << "    memoryPrecisions.A == " << name(memoryPrecisions[j]) << " && memoryPrecisions.B == " << name(memoryPrecisions[j]) << " && memoryPrecisions.C == " << name(memoryPrecisions[j]) << " && memoryPrecisions.bias == " << name(memoryPrecisions[j]) << " &&" << std::endl;
								std::cout << "    registerPrecisions.A == " << name(registerPrecisionA) << " && registerPrecisions.B == " << name(registerPrecisionB) << " && registerPrecisions.C == " << name(registerPrecisionC) << " &&" << std::endl;
								std::cout << "    preferAsyncLoad == true && preferAsyncStore == false &&" << std::endl;
								std::cout << "    splits[0] == 2 && splits[1] == 2 &&" << std::endl;
								std::cout << "    transposeState[0] == " << transposeStates[i * 2] << " && transposeState[1] == " << transposeStates[i * 2 + 1] << " && transposeState[2] == false &&" << std::endl;
								std::cout << "    useBias == " << useBias[k] << " && loadM == " << loadM[l] << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
								auto kernel = new GEMMKernel(kernelDesc, device.get());
								delete kernel;
							}
							{
								GEMMKernelDescriptor kernelDesc = GEMMKernelDescriptor(
									simd::ushort3 { 48, 48, 32 }, // Block dimensions
									(GEMMOperandPrecisions){ // Memory precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = memoryPrecisions[j], .bias = memoryPrecisions[j]
									},
									std::nullopt, // Padded block dimensions
									true, true, // preferAsyncLoad, preferAsyncStore
									(GEMMOperandPrecisions){ // Register precisions
										.A = registerPrecisionA, .B = registerPrecisionB, .C = registerPrecisionC, .bias = memoryPrecisions[j]
									},
									simd::ushort2 { 2, 2 }, // splits
									simd::uchar3 { transposeStates[i * 2], transposeStates[i * 2 + 1], false }, // transpose states
									useBias[k], loadM[l]); // useBias, loadM
								std::string file = std::string("b48x48x32_") + "pnil_" + memoryPrecisions[j].name() + "_a" + registerPrecisionA.name() + "_b" + registerPrecisionB.name() + "_c" + registerPrecisionC.name() + "_l1_s1_" + "s2x2_" + "a" + std::to_string(transposeStates[i * 2]) + "_b" + std::to_string(transposeStates[i * 2 + 1]) + "_b" + std::to_string(useBias[k]) + "_m" + std::to_string(loadM[l]);
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								*/
								std::cout << R"(
  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 32 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
)";
								std::cout << "    memoryPrecisions.A == " << name(memoryPrecisions[j]) << " && memoryPrecisions.B == " << name(memoryPrecisions[j]) << " && memoryPrecisions.C == " << name(memoryPrecisions[j]) << " && memoryPrecisions.bias == " << name(memoryPrecisions[j]) << " &&" << std::endl;
								std::cout << "    registerPrecisions.A == " << name(registerPrecisionA) << " && registerPrecisions.B == " << name(registerPrecisionB) << " && registerPrecisions.C == " << name(registerPrecisionC) << " &&" << std::endl;
								std::cout << "    preferAsyncLoad == true && preferAsyncStore == true &&" << std::endl;
								std::cout << "    splits[0] == 2 && splits[1] == 2 &&" << std::endl;
								std::cout << "    transposeState[0] == " << transposeStates[i * 2] << " && transposeState[1] == " << transposeStates[i * 2 + 1] << " && transposeState[2] == false &&" << std::endl;
								std::cout << "    useBias == " << useBias[k] << " && loadM == " << loadM[l] << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
								auto kernel = new GEMMKernel(kernelDesc, device.get());
								delete kernel;
							}
							{
								GEMMKernelDescriptor kernelDesc = GEMMKernelDescriptor(
									simd::ushort3 { 48, 48, 40 }, // Block dimensions
									(GEMMOperandPrecisions){ // Memory precisions
										.A = memoryPrecisions[j], .B = memoryPrecisions[j], .C = memoryPrecisions[j], .bias = memoryPrecisions[j]
									},
									std::nullopt, // Padded block dimensions
									true, true, // preferAsyncLoad, preferAsyncStore
									(GEMMOperandPrecisions){ // Register precisions
										.A = registerPrecisionA, .B = registerPrecisionB, .C = registerPrecisionC, .bias = memoryPrecisions[j]
									},
									simd::ushort2 { 2, 2 }, // splits
									simd::uchar3 { transposeStates[i * 2], transposeStates[i * 2 + 1], false }, // transpose states
									useBias[k], loadM[l]); // useBias, loadM
								std::string file = std::string("b48x48x40_") + "pnil_" + memoryPrecisions[j].name() + "_a" + registerPrecisionA.name() + "_b" + registerPrecisionB.name() + "_c" + registerPrecisionC.name() + "_l1_s1_" + "s2x2_" + "a" + std::to_string(transposeStates[i * 2]) + "_b" + std::to_string(transposeStates[i * 2 + 1]) + "_b" + std::to_string(useBias[k]) + "_m" + std::to_string(loadM[l]);
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								*/
								std::cout << R"(
  } else if (blockDimensions[0] == 48 && blockDimensions[1] == 48 && blockDimensions[2] == 40 &&
    simd_all(descriptor.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == simd::ushort3(UINT16_MAX)) &&
)";
								std::cout << "    memoryPrecisions.A == " << name(memoryPrecisions[j]) << " && memoryPrecisions.B == " << name(memoryPrecisions[j]) << " && memoryPrecisions.C == " << name(memoryPrecisions[j]) << " && memoryPrecisions.bias == " << name(memoryPrecisions[j]) << " &&" << std::endl;
								std::cout << "    registerPrecisions.A == " << name(registerPrecisionA) << " && registerPrecisions.B == " << name(registerPrecisionB) << " && registerPrecisions.C == " << name(registerPrecisionC) << " &&" << std::endl;
								std::cout << "    preferAsyncLoad == true && preferAsyncStore == true &&" << std::endl;
								std::cout << "    splits[0] == 2 && splits[1] == 2 &&" << std::endl;
								std::cout << "    transposeState[0] == " << transposeStates[i * 2] << " && transposeState[1] == " << transposeStates[i * 2 + 1] << " && transposeState[2] == false &&" << std::endl;
								std::cout << "    useBias == " << useBias[k] << " && loadM == " << loadM[l] << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
								auto kernel = new GEMMKernel(kernelDesc, device.get());
								delete kernel;
							}
						}
	}
	return 0;
}
