#include "HyperConnectionKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

HyperConnectionKernel::HyperConnectionKernel(HyperConnectionKernelDescriptor, MTL::Device* const device) {
	const char* const source = R"(
#include <metal_stdlib>
using namespace metal;

constant uint row_count [[function_constant(0)]];
constant uint hc [[function_constant(1)]];
constant uint hidden [[function_constant(2)]];
constant uint sinkhorn_iterations [[function_constant(3)]];
constant float epsilon [[function_constant(4)]];
constant uint operation [[function_constant(5)]];

kernel void hyper_connection(
	device const float* input0 [[buffer(0)]],
	device const float* input1 [[buffer(1)]],
	device const float* input2 [[buffer(2)]],
	device const float* input3 [[buffer(3)]],
	device float* output0 [[buffer(4)]],
	device float* output1 [[buffer(5)]],
	device float* output2 [[buffer(6)]],
	device float* output3 [[buffer(7)]],
	uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]],
	uint ntg [[threads_per_threadgroup]])
{
	if (row >= row_count || hc == 0 || hc > 16)
		return;
	if (operation == 2)
	{
		for (uint index = tid; index < hc * hidden; index += ntg)
		{
			const uint i = index / hidden;
			const uint d = index - i * hidden;
			float value = input0[row * hidden + d] * input2[row * hc + i];
			for (uint j = 0; j < hc; j++)
				value += input3[(row * hc + j) * hc + i] * input1[(row * hc + j) * hidden + d];
			output0[(row * hc + i) * hidden + d] = value;
		}
		return;
	}
	device const float* mix = input0;
	device const float* scale = input1;
	device const float* base = input2;
	device const float* residual = input3;
	device float* pre_out = output0;
	device float* post_out = output1;
	device float* comb_out = output2;
	device float* weighted = output3;
	const bool has_residual = operation == 1;
	const uint mix_dim = 2 * hc + hc * hc;
	device const float* row_mix = mix + row * mix_dim;
	threadgroup float pre[16];
	if (tid == 0)
	{
		for (uint i = 0; i < hc; i++)
		{
			const float v = 1.0f / (1.0f + exp(-(row_mix[i] * scale[0] + base[i]))) + epsilon;
			if (!has_residual)
				pre_out[row * hc + i] = v;
			pre[i] = v;
		}
		for (uint i = 0; i < hc; i++)
			post_out[row * hc + i] = 2.0f / (1.0f + exp(-(row_mix[hc + i] * scale[1] + base[hc + i])));
		float comb[256];
		for (uint i = 0; i < hc; i++)
		{
			float row_max = -INFINITY;
			for (uint j = 0; j < hc; j++)
			{
				const uint k = i * hc + j;
				comb[k] = row_mix[2 * hc + k] * scale[2] + base[2 * hc + k];
				row_max = max(row_max, comb[k]);
			}
			float row_sum = 0;
			for (uint j = 0; j < hc; j++)
			{
				const uint k = i * hc + j;
				comb[k] = exp(comb[k] - row_max);
				row_sum += comb[k];
			}
			for (uint j = 0; j < hc; j++)
				comb[i * hc + j] = comb[i * hc + j] / row_sum + epsilon;
		}
		for (uint j = 0; j < hc; j++)
		{
			float sum = 0;
			for (uint i = 0; i < hc; i++)
				sum += comb[i * hc + j];
			for (uint i = 0; i < hc; i++)
				comb[i * hc + j] /= sum + epsilon;
		}
		for (uint iter = 1; iter < sinkhorn_iterations; iter++)
		{
			for (uint i = 0; i < hc; i++)
			{
				float sum = 0;
				for (uint j = 0; j < hc; j++)
					sum += comb[i * hc + j];
				for (uint j = 0; j < hc; j++)
					comb[i * hc + j] /= sum + epsilon;
			}
			for (uint j = 0; j < hc; j++)
			{
				float sum = 0;
				for (uint i = 0; i < hc; i++)
					sum += comb[i * hc + j];
				for (uint i = 0; i < hc; i++)
					comb[i * hc + j] /= sum + epsilon;
			}
		}
		for (uint i = 0; i < hc * hc; i++)
			comb_out[row * hc * hc + i] = comb[i];
	}
	if (!has_residual)
		return;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint d = tid; d < hidden; d += ntg)
	{
		float sum = 0;
		for (uint i = 0; i < hc; i++)
			sum += residual[(row * hc + i) * hidden + d] * pre[i];
		weighted[row * hidden + d] = sum;
	}
}
)";
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
