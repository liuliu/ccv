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
	const int problemSize = descriptor.matrixDimensions[0];

	// Allocate FP32 memory for the operands.
	float* A = (float*)ccmalloc(sizeof(float) * problemSize * problemSize);
	float* B = (float*)ccmalloc(sizeof(float) * problemSize * problemSize);
	float* C = (float*)ccmalloc(sizeof(float) * problemSize * problemSize);
	float* bias = (float*)ccmalloc(sizeof(float) * problemSize);

	// Initialize A as the 2nd-order periodic Laplacian.
	int i, j;
	for (i = 0; i < problemSize; i++)
		for (j = 0; j < problemSize; j++)
			A[i * problemSize + j] = 0;
	for (i = 0; i < problemSize; i++)
	{
		const int diagonalAddress = i * problemSize + i;
		A[diagonalAddress] = -2;

		const int leftColumnID = (i + problemSize - 1) % problemSize;
		const int leftSubDiagonalAddress = i * problemSize + leftColumnID;
		A[leftSubDiagonalAddress] = 1;

		const int rightColumnID = (i + problemSize + 1) % problemSize;
		const int rightSubDiagonalAddress = i * problemSize + rightColumnID;
		A[rightSubDiagonalAddress] = 1;
	}

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	// Initialize B to random numbers.
	for (int rowID = 0; rowID < problemSize; rowID++)
	{
		for (int columnID = 0; columnID < problemSize; columnID++)
		{
			const int address = rowID * problemSize + columnID;
			B[address] = dsfmt_genrand_open_close(&dsfmt);
		}
	}

	// Initialize C to random numbers.
	for (int rowID = 0; rowID < problemSize; rowID++)
	{
		bias[rowID] =  dsfmt_genrand_open_close(&dsfmt);
	}
	void* A_storage = nullptr;
	if (descriptor.memoryPrecisions.A == GEMMOperandPrecision::FP16)
	{
		A_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize * problemSize);
		ccv_float_to_half_precision(A, (uint16_t*)A_storage, problemSize * problemSize);
		void* t = A_storage;
		A_storage = A;
		A = (float*)t;
	}
	void* B_storage = nullptr;
	if (descriptor.memoryPrecisions.B == GEMMOperandPrecision::FP16)
	{
		B_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize * problemSize);
		ccv_float_to_half_precision(B, (uint16_t*)B_storage, problemSize * problemSize);
		void* t = B_storage;
		B_storage = B;
		B = (float*)t;
	}
	void* bias_storage = nullptr;
	if (descriptor.memoryPrecisions.bias == GEMMOperandPrecision::FP16)
	{
		bias_storage = (uint16_t*)ccmalloc(sizeof(uint16_t) * problemSize);
		ccv_float_to_half_precision(bias, (uint16_t*)bias_storage, problemSize);
		void* t = bias_storage;
		bias_storage = bias;
		bias = (float*)t;
	}

	// Since the Laplacian is symmetric, we swap roles of the matrices to test
	// transposition of the left-hand side.
	//
	// Note that the test cannot cover correctness of A and B transposition
	// simultaneously. Instead, test the correctness in isolation
	// (AB, AB^T, A^T B). Performance can be tested in all four permutations
	// (AB, AB^T, A^T B, A^T B^T).
	if (descriptor.transposeState[0])
	{
		float* t = A;
		A = B;
		B = t;
	}

	// Multiply A with B.
	int maxGFLOPS = 0;
	int occupancy = 0;
	DeviceProperties dprops;
	dprops.coreCount = 18;
	NS::SharedPtr<MTL::Device> device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	NS::SharedPtr<MTL::CommandQueue> queue = NS::TransferPtr(device->newCommandQueue());
	{
		// Generate the kernel.
		auto pipelineValue = shaderCache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(descriptor, device.get(), dprops);
		occupancy = pipelineValue->pipeline->maxTotalThreadsPerThreadgroup();
		NS::SharedPtr<MTL::Buffer> bufferA = NS::TransferPtr(device->newBuffer(A, descriptor.memoryPrecisions.A.size() * problemSize * problemSize, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferB = NS::TransferPtr(device->newBuffer(B, descriptor.memoryPrecisions.B.size() * problemSize * problemSize, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferC = NS::TransferPtr(device->newBuffer(C, descriptor.memoryPrecisions.C.size() * problemSize * problemSize, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
		NS::SharedPtr<MTL::Buffer> bufferBias = NS::TransferPtr(device->newBuffer(bias, descriptor.memoryPrecisions.bias.size() * problemSize, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

		// load  = directAccessCondition,
		// store = false
		//  problemSize = 1488 | A   B   |  832 threads/core | 8175 GFLOPS
		//  problemSize = 1488 | A   B^T | 1024 threads/core | 8712 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 8818 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 8972 GFLOPS
		//  problemSize = 1489 | A   B   |  768 threads/core | 7888 GFLOPS
		//  problemSize = 1489 | A   B^T |  768 threads/core | 8256 GFLOPS
		//  problemSize = 1489 | A^T B   |  768 threads/core | 8026 GFLOPS
		//  problemSize = 1489 | A^T B^T |  832 threads/core | 8463 GFLOPS
		//
		// load  = directAccessCondition
		// store = directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)
		//  problemSize = 1488 | A   B   |  832 threads/core | 8186 GFLOPS
		//  problemSize = 1488 | A   B^T | 1024 threads/core | 8709 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 8808 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 8984 GFLOPS
		//  problemSize = 1489 | A   B   |  768 threads/core | 7902 GFLOPS
		//  problemSize = 1489 | A   B^T |  768 threads/core | 8249 GFLOPS
		//  problemSize = 1489 | A^T B   |  768 threads/core | 8034 GFLOPS
		//  problemSize = 1489 | A^T B^T |  832 threads/core | 8469 GFLOPS
		//
		// load  = directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)
		// store = directAccessCondition && (gid.y * M_group < M_edge) && (gid.x * N_group < N_edge)
		//  problemSize = 1488 | A   B   |  832 threads/core | 8181 GFLOPS
		//  problemSize = 1488 | A   B^T | 1024 threads/core | 8710 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 8806 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 8979 GFLOPS
		//  problemSize = 1489 | A   B   |  768 threads/core | 7892 GFLOPS
		//  problemSize = 1489 | A   B^T |  768 threads/core | 8242 GFLOPS
		//  problemSize = 1489 | A^T B   |  768 threads/core | 8034 GFLOPS
		//  problemSize = 1489 | A^T B^T |  832 threads/core | 8461 GFLOPS
		//
		// load previous C = false (M1 Max)
		//  problemSize = 1488 | A   B   |  896 threads/core | 8358 GFLOPS
		//  problemSize = 1488 | A   B^T | 1024 threads/core | 8682 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 8803 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 9024 GFLOPS
		//  problemSize = 1489 | A   B   |  768 threads/core | 8039 GFLOPS
		//  problemSize = 1489 | A   B^T |  832 threads/core | 8376 GFLOPS
		//  problemSize = 1489 | A^T B   |  832 threads/core | 8374 GFLOPS
		//  problemSize = 1489 | A^T B^T |  832 threads/core | 8654 GFLOPS
		//
		// load previous C = true (M1 Max)
		//  problemSize = 1488 | A   B   |  896 threads/core | 8352 GFLOPS
		//  problemSize = 1488 | A   B^T |  896 threads/core | 8515 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 8760 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 9007 GFLOPS
		//  problemSize = 1489 | A   B   |  768 threads/core | 7917 GFLOPS
		//  problemSize = 1489 | A   B^T |  768 threads/core | 7992 GFLOPS
		//  problemSize = 1489 | A^T B   |  832 threads/core | 8185 GFLOPS
		//  problemSize = 1489 | A^T B^T |  832 threads/core | 8583 GFLOPS
		//
		// load previous C = false (M4)
		//  problemSize = 1488 | A   B   | 1024 threads/core | 3353 GFLOPS
		//  problemSize = 1488 | A   B^T | 1024 threads/core | 3324 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 3338 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 3289 GFLOPS
		//  problemSize = 1489 | A   B   | 1024 threads/core | 3375 GFLOPS
		//  problemSize = 1489 | A   B^T | 1024 threads/core | 3317 GFLOPS
		//  problemSize = 1489 | A^T B   | 1024 threads/core | 3343 GFLOPS
		//  problemSize = 1489 | A^T B^T | 1024 threads/core | 3298 GFLOPS
		//
		// load previous C = true (M4)
		//  problemSize = 1488 | A   B   | 1024 threads/core | 3374 GFLOPS
		//  problemSize = 1488 | A   B^T | 1024 threads/core | 3312 GFLOPS
		//  problemSize = 1488 | A^T B   | 1024 threads/core | 3321 GFLOPS
		//  problemSize = 1488 | A^T B^T | 1024 threads/core | 3249 GFLOPS
		//  problemSize = 1489 | A   B   | 1024 threads/core | 3323 GFLOPS
		//  problemSize = 1489 | A   B^T | 1024 threads/core | 3280 GFLOPS
		//  problemSize = 1489 | A^T B   | 1024 threads/core | 3308 GFLOPS
		//  problemSize = 1489 | A^T B^T | 1024 threads/core | 3256 GFLOPS

		// Profile the latency of matrix multiplication.
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
				MTL::Size gridSize = MTL::Size(ceilDivide(problemSize, pipelineValue->kernel->blockDimensions[1]), ceilDivide(problemSize, pipelineValue->kernel->blockDimensions[0]), 1);
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
			auto operations = (int64_t)2 * problemSize * problemSize * problemSize;
			operations = operations * duplicatedCommandCount;
			auto gflops = (int)((double)operations / (double)latency / 1e9);

			// Report the results.
			// let latencyMicroseconds = Int(latency / 1e-6)
			// print(latencyMicroseconds, "μs", gflops, "GFLOPS")
			maxGFLOPS = std::max(maxGFLOPS, gflops);
		}
		// Copy the results to C.
		{
			auto precision = descriptor.memoryPrecisions.C;
			auto raw = bufferC->contents();
			for (int rowID = 0; rowID < problemSize; rowID++)
			{
				for (int columnID = 0; columnID < problemSize; columnID++)
				{
					const int address = rowID * problemSize + columnID;
					float entry32;
					switch (precision.value) {
						case GEMMOperandPrecision::FP32:
							entry32 = ((float*)raw)[address];
							break;
						case GEMMOperandPrecision::FP16: {
							uint16_t value = ((uint16_t*)raw)[address];
							ccv_half_precision_to_float(&value, &entry32, 1);
							break;
						}
					}
					C[address] = entry32;
				}
			}
		}
	}

	// Choose an error threshold.
	auto createErrorThreshold =
		[=](GEMMOperandPrecision precision) -> float {
			switch (precision.value) {
				case GEMMOperandPrecision::FP32:
					return 1e-5;
				case GEMMOperandPrecision::FP16:
					return 5e-3;
				case GEMMOperandPrecision::BF16:
					return 5e-2;
			}
		};
	float errorThreshold = 0;
	{
		auto memoryPrecisions = descriptor.memoryPrecisions;
		auto thresholdA = createErrorThreshold(memoryPrecisions.A);
		auto thresholdB = createErrorThreshold(memoryPrecisions.B);
		auto thresholdC = createErrorThreshold(memoryPrecisions.C);
		errorThreshold = std::max(errorThreshold, thresholdA);
		errorThreshold = std::max(errorThreshold, thresholdB);
		errorThreshold = std::max(errorThreshold, thresholdC);
	}
	// Check the results.
	int errorCount = 0;
	if (A_storage != nullptr)
	{
		void* t = A_storage;
		A_storage = A;
		A = (float*)t;
	}
	if (B_storage != nullptr)
	{
		void* t = B_storage;
		B_storage = B;
		B = (float*)t;
	}
	for (int m = 0; m < problemSize; m++)
	{
		for (int n = 0; n < problemSize; n++)
		{
			// Find the source row IDs.
			int leftRowID = (m + problemSize - 1) % problemSize;
			int centerRowID = m;
			int rightRowID = (m + problemSize + 1) % problemSize;

			// Find the source scalars.
			float leftSource;
			float centerSource;
			float rightSource;
			if (descriptor.transposeState[0])
			{
				leftSource = A[leftRowID * problemSize + n];
				centerSource = A[centerRowID * problemSize + n];
				rightSource = A[rightRowID * problemSize + n];
			} else if (descriptor.transposeState[1]) {
				leftSource = B[n * problemSize + leftRowID];
				centerSource = B[n * problemSize + centerRowID];
				rightSource = B[n * problemSize + rightRowID];
			} else {
				leftSource = B[leftRowID * problemSize + n];
				centerSource = B[centerRowID * problemSize + n];
				rightSource = B[rightRowID * problemSize + n];
			}

			// Find the expected result.
			float expected = leftSource - 2 * centerSource + rightSource;

			// Find the actual result.
			float actual;
			if (descriptor.transposeState[0])
			{
				actual = C[n * problemSize + m];
			} else {
				actual = C[m * problemSize + n];
			}

			// Report whether it is correct.
			float error = fabs(expected - actual);
			if (error > errorThreshold)
			{
				if (errorCount < 10)
				{
					printf("error: %f / ~1.000\n", error);
					errorCount += 1;
				}
			}
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
	int problemSize;
	bool transposeState[2];
};

void runTest(TestDescriptor descriptor)
{
	// Set up the kernel.
	GEMMDescriptor gemmDesc = GEMMDescriptor();
	auto precision = descriptor.precision;
	unsigned int n = (unsigned int)descriptor.problemSize;
	gemmDesc.matrixDimensions = simd::uint3 { n, n, n };
	gemmDesc.memoryPrecisions = {
		.A = precision, .B = precision, .C = precision, .bias = precision
	};
	gemmDesc.transposeState = simd::uchar3 { descriptor.transposeState[0], descriptor.transposeState[1] };
	gemmDesc.useBias = false;

	// Test the kernel.
	auto statistic = profileProblemSize(gemmDesc);

	// Report the results.
	std::cout << "problemSize = " << descriptor.problemSize << " | ";
	if (descriptor.transposeState[0])
	{
		std::cout << "A^T ";
	} else {
		std::cout << "A   ";
	}
	if (descriptor.transposeState[1])
	{
		std::cout << "B^T | ";
	} else {
		std::cout << "B   | ";
	}

	std::cout << statistic.first << " GFLOPS " << statistic.second << " threads/core | " << std::endl;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	{
		int problemSizes[] = {
			7, 8, 9, 10,
			15, 16, 17, 18,
			23, 24, 25,
			31, 32, 33,
			47, 48, 49,
			63, 64, 65,
			103, 104, 112,
			126, 127, 128, 129,
			130, 131,
			135, 136, 137,
			143, 144, 145,
			151, 152, 153,
		};
		bool transposeStates[] = {
			false, false,
			false, true,
			true, false,
		};
		printf("Correctness tests:\n");
		for (int i = 0; i < sizeof(problemSizes) / sizeof(int); i++)
		{
			for (int j = 0; j < sizeof(transposeStates) / (sizeof(bool) * 2); j++)
			{
				TestDescriptor testDescriptor = TestDescriptor();
				testDescriptor.precision = GEMMOperandPrecision::FP32;
				testDescriptor.problemSize = problemSizes[i];
				testDescriptor.transposeState[0] = transposeStates[j * 2];
				testDescriptor.transposeState[1] = transposeStates[j * 2 + 1];
				runTest(testDescriptor);
			}
		}
	}
	{
		bool transposeStates[] = {
			false, false,
			false, true,
			true, false,
			true, true,
		};

		printf("\nPerformance tests:\n");
		for (int problemSize = 1488; problemSize <= 1489; problemSize++)
		{
			for (int j = 0; j < sizeof(transposeStates) / (sizeof(bool) * 2); j++)
			{
				TestDescriptor testDescriptor = TestDescriptor();
				testDescriptor.precision = GEMMOperandPrecision::FP16;
				testDescriptor.problemSize = problemSize;
				testDescriptor.transposeState[0] = transposeStates[j * 2];
				testDescriptor.transposeState[1] = transposeStates[j * 2 + 1];
				runTest(testDescriptor);
			}
		}
	}
	return 0;
}
