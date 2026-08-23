#ifndef MFA_TRANSPOSEKERNEL_HPP_
#define MFA_TRANSPOSEKERNEL_HPP_

#include "ShaderCache.hpp"
#include "TransposeDescriptor.hpp"

struct TransposeKernelParams {
	uint32_t rows;
	uint32_t cols;
	uint32_t sourceBatchStride;
	uint32_t sourceRowStride;
	uint32_t destinationBatchStride;
	uint32_t destinationRowStride;
};

struct TransposeKernel {
	NS::SharedPtr<MTL::Library> library;

	GEMMOperandPrecision memoryPrecision;

	MTL::Size threadgroupSize;

	TransposeKernel(TransposeKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t batchSize, uint32_t rows, uint32_t cols) const noexcept;

	std::string createSource() const noexcept;
};

#endif
