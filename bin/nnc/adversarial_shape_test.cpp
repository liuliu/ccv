extern "C" {
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <sys/time.h>
#include <ctype.h>
}
#include "nnc/mfa/v2/ShaderCache.hpp"
#include "nnc/mfa/v2/GEMMDescriptor.hpp"
#include "nnc/mfa/v2/GEMMKernelDescriptor.hpp"
#include "nnc/mfa/v2/GEMMKernel.hpp"
#include "3rdparty/dsfmt/dSFMT.h"
#include <iostream>

ShaderCache shaderCache;

std::pair<int, int> profileProblemSize(GEMMDescriptor descriptor)
{
	const int problemSize1 = descriptor.matrixDimensions[0];
	const int problemSize2 = descriptor.matrixDimensions[1];
	const int problemSize3 = descriptor.matrixDimensions[2];

	// Allocate FP32 memory for the operands.
	float* A = (float*)ccmalloc(sizeof(float) * problemSize1 * problemSize3);
	float* B = (float*)ccmalloc(sizeof(float) * problemSize2 * problemSize3);
	float* C = (float*)ccmalloc(sizeof(float) * problemSize1 * problemSize2);
	float* bias = (float*)ccmalloc(sizeof(float) * problemSize2);


	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);

	// Initialize A to random numbers.
	int i, j;
	for (i = 0; i < problemSize3; i++)
		for (j = 0; j < problemSize1; j++)
			A[i * problemSize1 + j] = dsfmt_genrand_open_close(&dsfmt);
	// Initialize B to random numbers.
	for (int rowID = 0; rowID < problemSize2; rowID++)
	{
		for (int columnID = 0; columnID < problemSize3; columnID++)
		{
			const int address = rowID * problemSize3 + columnID;
			B[address] = dsfmt_genrand_open_close(&dsfmt);
		}
	}

	// Initialize C to random numbers.
	for (int rowID = 0; rowID < problemSize2; rowID++)
	{
		bias[rowID] = dsfmt_genrand_open_close(&dsfmt);
	}
	void* A_storage = nullptr;
	if (descriptor.memoryPrecisions.A == GEMMOperandPrecision::FP16)
	{
		A_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize1 * problemSize3);
		ccv_float_to_half_precision(A, (uint16_t*)A_storage, problemSize1 * problemSize3);
		void* t = A_storage;
		A_storage = A;
		A = (float*)t;
	} else if (descriptor.memoryPrecisions.A == GEMMOperandPrecision::BF16) {
		A_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize1 * problemSize3);
		for (int i = 0; i < problemSize1 * problemSize3; i++)
			((uint16_t*)A_storage)[i] = ((uint16_t*)A)[i * 2 + 1];
		void* t = A_storage;
		A_storage = A;
		A = (float*)t;
	}
	void* B_storage = nullptr;
	if (descriptor.memoryPrecisions.B == GEMMOperandPrecision::FP16)
	{
		B_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize2 * problemSize3);
		ccv_float_to_half_precision(B, (uint16_t*)B_storage, problemSize2 * problemSize3);
		void* t = B_storage;
		B_storage = B;
		B = (float*)t;
	} else if (descriptor.memoryPrecisions.B == GEMMOperandPrecision::BF16) {
		B_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize2 * problemSize3);
		for (int i = 0; i < problemSize2 * problemSize3; i++)
			((uint16_t*)B_storage)[i] = ((uint16_t*)B)[i * 2 + 1];
		void* t = B_storage;
		B_storage = B;
		B = (float*)t;
	}
	void* bias_storage = nullptr;
	if (descriptor.memoryPrecisions.bias == GEMMOperandPrecision::FP16)
	{
		bias_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize2);
		ccv_float_to_half_precision(bias, (uint16_t*)bias_storage, problemSize2);
		void* t = bias_storage;
		bias_storage = bias;
		bias = (float*)t;
	} else if (descriptor.memoryPrecisions.bias == GEMMOperandPrecision::BF16) {
		bias_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize2);
		for (int i = 0; i < problemSize2; i++)
			((uint16_t*)bias_storage)[i] = ((uint16_t*)bias)[i * 2 + 1];
		void* t = bias_storage;
		bias_storage = bias;
		bias = (float*)t;
	}

	// Multiply A with B.
	int maxGFLOPS = 0;
	int occupancy = 0;
	DeviceProperties dprops;
	NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	NS::SharedPtr<MTL::CommandQueue> queue = NS::TransferPtr(device->newCommandQueue());
	{
		// Generate the kernel.
		auto pipelineValue = shaderCache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(descriptor, device.get(), dprops);
		occupancy = pipelineValue->pipeline->maxTotalThreadsPerThreadgroup();
		NS::SharedPtr<MTL::Buffer> bufferA = NS::TransferPtr(device->newBuffer(A, descriptor.memoryPrecisions.A.size() * problemSize1 * problemSize3, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferB = NS::TransferPtr(device->newBuffer(B, descriptor.memoryPrecisions.B.size() * problemSize2 * problemSize3, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferC = NS::TransferPtr(device->newBuffer(C, descriptor.memoryPrecisions.C.size() * problemSize1 * problemSize2, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferBias = NS::TransferPtr(device->newBuffer(bias, descriptor.memoryPrecisions.bias.size() * problemSize2, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		for (int i = 0; i < 15; i++)
		{
			const int duplicatedCommandCount = 20;
			NS::SharedPtr<MTL::CommandBuffer> commandBuffer = NS::TransferPtr(queue->commandBuffer());
			NS::SharedPtr<MTL::ComputeCommandEncoder> encoder = NS::TransferPtr(commandBuffer->computeCommandEncoder());
	  		encoder->setComputePipelineState(pipelineValue->pipeline.get());
			encoder->setThreadgroupMemoryLength(pipelineValue->kernel->threadgroupMemoryAllocation, 0);
			encoder->setBuffer(bufferA.get(), 0, 0);
			encoder->setBuffer(bufferB.get(), 0, 1);
			encoder->setBuffer(bufferC.get(), 0, 2);
			encoder->useResource(bufferA.get(), MTL::ResourceUsageRead);
			encoder->useResource(bufferB.get(), MTL::ResourceUsageRead);
			encoder->useResource(bufferC.get(), MTL::ResourceUsageWrite);
			if (descriptor.useBias)
			{
				encoder->setBuffer(bufferBias.get(), 0, 3);
				encoder->useResource(bufferBias.get(), MTL::ResourceUsageRead);
			}
			for (int j = 0; j < duplicatedCommandCount; j++)
			{
				auto ceilDivide =
					[=](int64_t target, uint16_t granularity) -> int64_t {
						return (target + int64_t(granularity) - 1) / int64_t(granularity);
					};
				MTL::Size gridSize = MTL::Size(ceilDivide(problemSize2, pipelineValue->kernel->blockDimensions[1]), ceilDivide(problemSize1, pipelineValue->kernel->blockDimensions[0]), 1);
				MTL::Size groupSize = MTL::Size(pipelineValue->kernel->threadgroupSize, 1, 1);
				encoder->dispatchThreadgroups(gridSize, groupSize);
			}
			encoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			auto start = commandBuffer->GPUStartTime();
			auto end = commandBuffer->GPUEndTime();
			auto latency = end - start;

			// Determine the amount of work done.
			auto operations = (int64_t)2 * problemSize1 * problemSize2 * problemSize3;
			operations = operations * duplicatedCommandCount;
			auto gflops = (int)((double)operations / (double)latency / 1e9);

			// Report the results.
			// let latencyMicroseconds = Int(latency / 1e-6)
			// print(latencyMicroseconds, "μs", gflops, "GFLOPS")
			maxGFLOPS = std::max(maxGFLOPS, gflops);
		}
	}

	ccfree(A);
	ccfree(B);
	ccfree(C);
	ccfree(bias);
	if (A_storage != nullptr)
		ccfree(A_storage);
	if (B_storage != nullptr)
		ccfree(B_storage);
	if (bias_storage != nullptr)
		ccfree(bias_storage);
	return std::make_pair(maxGFLOPS, occupancy);
}

struct TestDescriptor {
	GEMMOperandPrecision precision;
	int problemSize[3];
	bool transposeState[2];
	bool useBias;
};

void runTest(TestDescriptor descriptor)
{
	// Set up the kernel.
	GEMMDescriptor gemmDesc = GEMMDescriptor();
	auto precision = descriptor.precision;
	unsigned int m = (unsigned int)descriptor.problemSize[0];
	unsigned int n = (unsigned int)descriptor.problemSize[1];
	unsigned int k = (unsigned int)descriptor.problemSize[2];
	gemmDesc.matrixDimensions = simd::uint3 { m, n, k };
	gemmDesc.memoryPrecisions = {
		.A = precision, .B = precision, .C = precision, .bias = precision
	};
	gemmDesc.transposeState = simd::uchar3 { descriptor.transposeState[0], descriptor.transposeState[1], descriptor.transposeState[0] };
	gemmDesc.useBias = descriptor.useBias;
	gemmDesc.registerPrecisionC = GEMMOperandPrecision::FP32;

	// Test the kernel.
	auto statistic = profileProblemSize(gemmDesc);

	// Report the results.
	std::cout << "problemSize = " << descriptor.problemSize[0] << "x" << descriptor.problemSize[1] << "x" << descriptor.problemSize[2] << " | ";
	if (descriptor.transposeState[0])
	{
		std::cout << "A^T ";
	} else {
		std::cout << "A   ";
	}
	if (descriptor.transposeState[1])
	{
		std::cout << "B^T ";
	} else {
		std::cout << "B   ";
	}
	if (descriptor.useBias)
	{
		std::cout << "+ BIAS | ";
	} else {
		std::cout << "       | ";
	}

	std::cout << statistic.first << " GFLOPS " << statistic.second << " threads/core | " << std::endl;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	{
		bool transposeStates[] = {
			false, false,
			false, true,
			// true, false,
			// true, true,
			false, false,
			false, true,
			// true, false,
			// true, true,
		};
		bool useBias[] = {
			false,
			false,
			// false,
			// false,
			true,
			true,
			// true,
			// true
		};
		int problemSizes[] = {
			4608 * 2, 3072, 3072 * 4,
			4608 * 2, 3072 * 4, 3072,
			4608 * 2, 3072, 3072,
			// 4608, 3072, 3072 * 3,
			// 4608, 3072 * 3, 3072,
		};

		printf("\nPerformance tests:\n");
		for (int i = 0; i < sizeof(problemSizes) / (sizeof(int) * 3); i++)
		// for (int problemSize = 7936; problemSize <= 3072 * 4; problemSize += 128)
		{
			for (int j = 0; j < sizeof(transposeStates) / (sizeof(bool) * 2); j++)
			{
				TestDescriptor testDescriptor = TestDescriptor();
				testDescriptor.precision = GEMMOperandPrecision::FP16;
				testDescriptor.problemSize[0] = problemSizes[i * 3];
				testDescriptor.problemSize[1] = problemSizes[i * 3 + 1];
				testDescriptor.problemSize[2] = problemSizes[i * 3 + 2];
				testDescriptor.transposeState[0] = transposeStates[j * 2];
				testDescriptor.transposeState[1] = transposeStates[j * 2 + 1];
				testDescriptor.useBias = useBias[j];
				runTest(testDescriptor);
			}
		}
	}
	return 0;
}
