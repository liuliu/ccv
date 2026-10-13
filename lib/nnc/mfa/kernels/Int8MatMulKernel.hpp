#ifndef MFA_INT8MATMULKERNEL_HPP_
#define MFA_INT8MATMULKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "Int8MatMulKernelDescriptor.hpp"

class CodeWriter;

struct Int8MatMulKernel {
	NS::SharedPtr<MTL::Library> library;

	std::string source;

	/// These properties are copied from Int8MatMulKernelDescriptor.
	Int8MatMulOperation operation;

	GEMMOperandPrecision ioPrecision;

	simd::ushort3 blockDimensions;

	simd::ushort2 splits;

	uint16_t kInterleave;

	bool useBias;

	bool activationHadamard256;

	/// The dense kernel's per-simdgroup tile.
	uint16_t registerM;

	uint16_t registerN;

	/// The number of threads per group.
	MTL::Size threadgroupSize;

	Int8MatMulKernel(Int8MatMulKernelDescriptor descriptor, MTL::Device* device);

	const char* functionName() const noexcept;

	/// The threadgroups covering one dispatch.
	MTL::Size threadgroupsPerGrid(const Int8MatMulDescriptor& descriptor) const noexcept;

private:
	std::string createSource() const noexcept;
	void createRotate256(CodeWriter* source) const noexcept;
	void createQuantizeActivation(CodeWriter* source) const noexcept;
	void createRotateActivation(CodeWriter* source) const noexcept;
	void createCastWeights(CodeWriter* source) const noexcept;
	void createDequantizeOutput(CodeWriter* source) const noexcept;
	void createDequantizeSegmentedOutput(CodeWriter* source) const noexcept;
	void createSegmented(CodeWriter* source) const noexcept;
	void createDense(CodeWriter* source) const noexcept;
	void createDenseMultiplyIterations(CodeWriter* source) const noexcept;
	void createDenseStoreC(CodeWriter* source) const noexcept;
};

#endif
