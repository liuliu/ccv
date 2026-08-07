#include "RMSNormCmulKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include <sstream>

static const char* _rmsnorm_cmul_type(const GEMMOperandPrecision precision)
{
	if (precision == GEMMOperandPrecision::FP32)
		return "float";
	if (precision == GEMMOperandPrecision::BF16)
		return "bfloat";
	return "half";
}

static const char* _rmsnorm_cmul_type2(const GEMMOperandPrecision precision)
{
	if (precision == GEMMOperandPrecision::FP32)
		return "float2";
	if (precision == GEMMOperandPrecision::BF16)
		return "bfloat2";
	return "half2";
}

RMSNormCmulKernel::RMSNormCmulKernel(RMSNormCmulKernelDescriptor descriptor, MTL::Device* const device)
{
	threadgroupSize = MTL::Size(256, 1, 1);
	std::ostringstream constants;
	constants << "typedef " << _rmsnorm_cmul_type(descriptor.aPrecision) << " realA;\n";
	constants << "typedef " << _rmsnorm_cmul_type2(descriptor.aPrecision) << " realA2;\n";
	constants << "typedef " << _rmsnorm_cmul_type(descriptor.rotationPrecision) << " realRotation;\n";
	constants << "typedef " << _rmsnorm_cmul_type2(descriptor.rotationPrecision) << " realRotation2;\n";
	constants << "typedef " << _rmsnorm_cmul_type(descriptor.scalePrecision) << " realScale;\n";
	constants << "typedef " << _rmsnorm_cmul_type2(descriptor.scalePrecision) << " realScale2;\n";
	constants << "constant uint column_count [[function_constant(0)]];\n";
	constants << "constant uint broadcast_ratio [[function_constant(1)]];\n";
	constants << "constant float epsilon [[function_constant(2)]];\n";
	constants << "constant uint rows_per_threadgroup [[function_constant(3)]];\n";
	constants << "constant bool elementwise_affine = " << (descriptor.elementwiseAffine ? "true" : "false") << ";\n";
	source = constants.str() + R"(
#include <metal_stdlib>
using namespace metal;

inline float threadgroup_inv_rms(float square_sum, threadgroup float* partials, uint tid, uint simd_lane, uint simd_group)
{
	square_sum = simd_sum(square_sum);
	if (simd_lane == 0)
		partials[simd_group] = square_sum;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < 8) {
		square_sum = quad_sum(partials[tid]);
		square_sum += simd_shuffle_xor(square_sum, 4);
		if (tid == 0)
			partials[0] = rsqrt(square_sum / float(column_count) + epsilon);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return partials[0];
}

kernel void rmsnorm_cmul(
	device const realA2* source [[buffer(0)]],
	device const realRotation2* rotation [[buffer(1)]],
	device const realScale2* scale [[buffer(2)]],
	device realA2* destination [[buffer(3)]],
	uint tid [[thread_index_in_threadgroup]],
	uint simd_lane [[thread_index_in_simdgroup]],
	uint simd_group [[simdgroup_index_in_threadgroup]],
	uint2 group [[threadgroup_position_in_grid]])
{
	const uint complex_count = column_count / 2;
	const uint first_head = group.x * rows_per_threadgroup;
	const uint rotation_offset = group.y * complex_count;
	float2 rotate = 0;
	if (tid < complex_count)
		rotate = float2(rotation[rotation_offset + tid]);
	threadgroup float partials[8];
#pragma clang loop unroll(full)
	for (uint head_delta = 0; head_delta < rows_per_threadgroup; head_delta++) {
		const uint head = first_head + head_delta;
		if (head >= broadcast_ratio)
			break;
		const uint row = group.y * broadcast_ratio + head;
		const uint source_offset = row * complex_count;
		float2 value = 0;
		if (tid < complex_count)
			value = float2(source[source_offset + tid]);
		const float inv_rms = threadgroup_inv_rms(dot(value, value), partials, tid, simd_lane, simd_group);
		if (tid < complex_count) {
			value *= inv_rms;
			if (elementwise_affine)
				value *= float2(scale[tid]);
			const float2 result = float2(value.x * rotate.x - value.y * rotate.y, value.x * rotate.y + value.y * rotate.x);
			destination[source_offset + tid] = realA2(result);
		}
	}
}
)";
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
