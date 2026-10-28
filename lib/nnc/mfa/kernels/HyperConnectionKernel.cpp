#include "HyperConnectionKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

HyperConnectionKernel::HyperConnectionKernel(HyperConnectionKernelDescriptor descriptor, MTL::Device* const device) {
	std::string source = R"(
#include <metal_stdlib>
using namespace metal;

typedef )" + std::string(descriptor.blockFP16 ? "half" : "float") + R"( input0_t;

constant uint row_count [[function_constant(0)]];
constant uint hc [[function_constant(1)]];
constant uint hidden [[function_constant(2)]];
constant uint sinkhorn_iterations [[function_constant(3)]];
constant float epsilon [[function_constant(4)]];
constant uint operation [[function_constant(5)]];
constant float norm_epsilon [[function_constant(6)]];
constant uint dot_splits [[function_constant(7)]];
constant uint collapse_splits [[function_constant(8)]];
constant uint expand_addend [[function_constant(9)]];

kernel void hyper_connection(
	device const input0_t* input0 [[buffer(0)]],
	device const float* input1 [[buffer(1)]],
	device const float* input2 [[buffer(2)]],
	device const float* input3 [[buffer(3)]],
	device float* output0 [[buffer(4)]],
	device float* output1 [[buffer(5)]],
	device float* output2 [[buffer(6)]],
	device float* output3 [[buffer(7)]],
	uint2 tgid [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]],
	uint2 ntg2 [[threads_per_threadgroup]])
{
	const uint row = tgid.x;
	const uint ntg = ntg2.x;
	if (row >= row_count || hc == 0 || hc > 16)
		return;
	if (operation == 2)
	{
		// Decode needs several threadgroups; prefill already has independent rows.
		const uint block_size = row_count == 1 ? 256 : hc * hidden;
		for (uint index = tgid.y * block_size + tid; index < min((tgid.y + 1) * block_size, hc * hidden); index += ntg)
		{
			const uint i = index / hidden;
			const uint d = index - i * hidden;
			// The addend comes in the second output's slot.
			const float block = expand_addend ? (float)input0[row * hidden + d] + output1[row * hidden + d] : (float)input0[row * hidden + d];
			float value = block * input2[row * hc + i];
			for (uint j = 0; j < hc; j++)
				value += input3[(row * hc + j) * hc + i] * input1[(row * hc + j) * hidden + d];
			output0[(row * hc + i) * hidden + d] = value;
		}
		return;
	}
	device const input0_t* mix = input0;
	device const float* scale = input1;
	device const float* base = input2;
	device const float* residual = input3;
	device float* pre_out = output0;
	device float* post_out = output1;
	device float* comb_out = output2;
	device float* weighted = output3;
	const bool has_residual = operation == 1;
	const uint mix_dim = 2 * hc + hc * hc;
	device const input0_t* row_mix = mix + row * mix_dim;
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

// Mix, decode step 1. The first pairs * dot_splits threadgroups stream two projection rows over
// one MIX_CHUNK of the flattened residual, the pair-0 ones also adding its squares. The others
// collapse MIX_THREADS hidden columns with the incoming pre-mix and add the squares of the collapse.
constant uint MIX_THREADS = 256;
constant uint MIX_CHUNK = MIX_THREADS * 4;

kernel void hyper_connection_mix_partials(
	device const float* residual [[buffer(0)]],
	device const float* incoming_pre [[buffer(1)]],
	device const float* projection [[buffer(2)]],
	device float* dots [[buffer(3)]],
	device float* squares [[buffer(4)]],
	device float* collapsed [[buffer(5)]],
	device float* collapsed_squares [[buffer(6)]],
	uint2 tgid [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]],
	ushort sgid [[simdgroup_index_in_threadgroup]],
	ushort lane [[thread_index_in_simdgroup]])
{
	threadgroup float sums[3][MIX_THREADS / 32];
	const uint row = tgid.y;
	const uint mix_dim = 2 * hc + hc * hc;
	const uint flat = hc * hidden;
	const uint pairs = (mix_dim + 1) / 2;
	device const float* x = residual + (ulong)row * flat;
	if (tgid.x < pairs * dot_splits)
	{
		const uint pair = tgid.x / dot_splits;
		const uint split = tgid.x - pair * dot_splits;
		const uint r0 = pair * 2;
		const uint r1 = r0 + 1;
		device const float4* x4 = (device const float4*)x;
		device const float4* w0 = (device const float4*)(projection + (ulong)r0 * flat);
		device const float4* w1 = (device const float4*)(projection + (ulong)min(r1, mix_dim - 1) * flat);
		float d0 = 0, d1 = 0, square = 0;
		for (uint i = split * (MIX_CHUNK / 4) + tid; i < min((split + 1) * (MIX_CHUNK / 4), flat / 4); i += MIX_THREADS)
		{
			const float4 v = x4[i];
			d0 += dot(w0[i], v);
			d1 += dot(w1[i], v);
			square += dot(v, v);
		}
		d0 = simd_sum(d0);
		d1 = simd_sum(d1);
		square = simd_sum(square);
		if (lane == 0)
		{
			sums[0][sgid] = d0;
			sums[1][sgid] = d1;
			sums[2][sgid] = square;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid < 3)
		{
			float total = 0;
			for (uint s = 0; s < MIX_THREADS / 32; s++)
				total += sums[tid][s];
			if (tid == 0)
				dots[(row * dot_splits + split) * mix_dim + r0] = total;
			else if (tid == 1 && r1 < mix_dim)
				dots[(row * dot_splits + split) * mix_dim + r1] = total;
			else if (tid == 2 && pair == 0)
				squares[row * dot_splits + split] = total;
		}
		return;
	}
	const uint q = tgid.x - pairs * dot_splits;
	const uint d = q * MIX_THREADS + tid;
	float value = 0;
	if (d < hidden)
	{
		for (uint i = 0; i < hc; i++)
			value += x[i * hidden + d] * incoming_pre[row * hc + i];
		collapsed[(ulong)row * hidden + d] = value;
	}
	const float square = simd_sum(value * value);
	if (lane == 0)
		sums[0][sgid] = square;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid == 0)
	{
		float total = 0;
		for (uint s = 0; s < MIX_THREADS / 32; s++)
			total += sums[0][s];
		collapsed_squares[row * collapse_splits + q] = total;
	}
}

// Mix split: one SIMD group per row. Lane k holds mix value k, scaled by the residual's RMS
// reciprocal, so the gates and the combination Sinkhorn run in parallel. Row and column sums visit
// the same elements in the same order as the serial split.
inline void hyper_connection_mix_split_row(
	device const float* dots,
	device const float* squares,
	device const float* scale,
	device const float* base,
	device float* pre_out,
	device float* post_out,
	device float* comb_out,
	const uint row,
	const ushort lane)
{
	const uint mix_dim = 2 * hc + hc * hc;
	float sum_squares = 0;
	for (uint s = 0; s < dot_splits; s++)
		sum_squares += squares[row * dot_splits + s];
	const float rstd = rsqrt(sum_squares / float(hc * hidden) + norm_epsilon);
	float mix = 0;
	if (lane < mix_dim)
	{
		for (uint s = 0; s < dot_splits; s++)
			mix += dots[(row * dot_splits + s) * mix_dim + lane];
		mix *= rstd;
	}
	if (lane < hc)
		pre_out[row * hc + lane] = 1.0f / (1.0f + exp(-(mix * scale[0] + base[lane]))) + epsilon;
	else if (lane < 2 * hc)
		post_out[row * hc + lane - hc] = 2.0f / (1.0f + exp(-(mix * scale[1] + base[lane])));
	const bool is_comb = lane >= 2 * hc && lane < mix_dim;
	const uint k = is_comb ? lane - 2 * hc : 0;
	const ushort row_lane = 2 * hc + (k / hc) * hc;
	const ushort column_lane = 2 * hc + k % hc;
	float v = is_comb ? mix * scale[2] + base[lane] : 0;
	float row_max = -INFINITY;
	for (uint t = 0; t < hc; t++)
		row_max = max(row_max, simd_shuffle(v, ushort(row_lane + t)));
	v = exp(v - row_max);
	float row_sum = 0;
	for (uint t = 0; t < hc; t++)
		row_sum += simd_shuffle(v, ushort(row_lane + t));
	v = v / row_sum + epsilon;
	float column_sum = 0;
	for (uint t = 0; t < hc; t++)
		column_sum += simd_shuffle(v, ushort(column_lane + t * hc));
	v /= column_sum + epsilon;
	for (uint iteration = 1; iteration < sinkhorn_iterations; iteration++)
	{
		row_sum = 0;
		for (uint t = 0; t < hc; t++)
			row_sum += simd_shuffle(v, ushort(row_lane + t));
		v /= row_sum + epsilon;
		column_sum = 0;
		for (uint t = 0; t < hc; t++)
			column_sum += simd_shuffle(v, ushort(column_lane + t * hc));
		v /= column_sum + epsilon;
	}
	if (is_comb)
		comb_out[row * hc * hc + k] = v;
}

// Mix with many rows: the split after the projection GEMM.
kernel void hyper_connection_mix_split(
	device const float* dots [[buffer(0)]],
	device const float* squares [[buffer(1)]],
	device const float* scale [[buffer(2)]],
	device const float* base [[buffer(3)]],
	device float* pre_out [[buffer(4)]],
	device float* post_out [[buffer(5)]],
	device float* comb_out [[buffer(6)]],
	uint row [[threadgroup_position_in_grid]],
	ushort lane [[thread_index_in_simdgroup]])
{
	hyper_connection_mix_split_row(dots, squares, scale, base, pre_out, post_out, comb_out, row, lane);
}

// Mix, decode step 2. The normalized collapse does not depend on the split, so the split and the
// normalization share one dispatch: threadgroup 0 splits the row, and each of the others reduces
// the collapse's squares itself and normalizes MIX_THREADS hidden columns.
kernel void hyper_connection_mix_finish(
	device const float* dots [[buffer(0)]],
	device const float* squares [[buffer(1)]],
	device const float* scale [[buffer(2)]],
	device const float* base [[buffer(3)]],
	device float* pre_out [[buffer(4)]],
	device float* post_out [[buffer(5)]],
	device float* comb_out [[buffer(6)]],
	device const float* collapsed [[buffer(7)]],
	device const float* collapsed_squares [[buffer(8)]],
	device const float* norm_scale [[buffer(9)]],
	device float* output [[buffer(10)]],
	uint2 tgid [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]],
	ushort sgid [[simdgroup_index_in_threadgroup]],
	ushort lane [[thread_index_in_simdgroup]])
{
	const uint row = tgid.y;
	if (tgid.x == 0)
	{
		if (sgid == 0)
			hyper_connection_mix_split_row(dots, squares, scale, base, pre_out, post_out, comb_out, row, lane);
		return;
	}
	const uint d = (tgid.x - 1) * MIX_THREADS + tid;
	if (d >= hidden)
		return;
	float sum = 0;
	for (uint q = 0; q < collapse_splits; q++)
		sum += collapsed_squares[row * collapse_splits + q];
	const float rstd = rsqrt(sum / float(hidden) + norm_epsilon);
	output[(ulong)row * hidden + d] = collapsed[(ulong)row * hidden + d] * rstd * norm_scale[d];
}

// Mix with many rows: a threadgroup collapses and normalizes one row, and also returns the sum of
// the residual's squares for the split. Rows hold at most 16 * 1024 hidden columns.
constant uint ROW_THREADS = 1024;

kernel void hyper_connection_collapse_norm_rows(
	device const float* residual [[buffer(0)]],
	device const float* incoming_pre [[buffer(1)]],
	device const float* norm_scale [[buffer(2)]],
	device float* output [[buffer(3)]],
	device float* squares [[buffer(4)]],
	uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]],
	ushort sgid [[simdgroup_index_in_threadgroup]],
	ushort lane [[thread_index_in_simdgroup]])
{
	threadgroup float sums[2][ROW_THREADS / 32];
	device const float* x = residual + (ulong)row * hc * hidden;
	float values[16];
	float square = 0;
	float collapsed_square = 0;
	#pragma clang loop unroll(full)
	for (uint t = 0; t < 16; t++)
	{
		const uint d = t * ROW_THREADS + tid;
		if (d < hidden)
		{
			float value = 0;
			for (uint i = 0; i < hc; i++)
			{
				const float r = x[i * hidden + d];
				value += r * incoming_pre[row * hc + i];
				square += r * r;
			}
			values[t] = value;
			collapsed_square += value * value;
		}
	}
	square = simd_sum(square);
	collapsed_square = simd_sum(collapsed_square);
	if (lane == 0)
	{
		sums[0][sgid] = square;
		sums[1][sgid] = collapsed_square;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (sgid == 0)
	{
		square = simd_sum(sums[0][lane]);
		collapsed_square = simd_sum(sums[1][lane]);
		if (lane == 0)
		{
			squares[row] = square;
			sums[1][0] = rsqrt(collapsed_square / float(hidden) + norm_epsilon);
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const float rstd = sums[1][0];
	#pragma clang loop unroll(full)
	for (uint t = 0; t < 16; t++)
	{
		const uint d = t * ROW_THREADS + tid;
		if (d < hidden)
			output[(ulong)row * hidden + d] = values[t] * rstd * norm_scale[d];
	}
}
)";
	if (descriptor.loadM)
	{
		const std::string rowCountConstant = "constant uint row_count [[function_constant(0)]];\n";
		const std::string::size_type rowCountConstantPosition = source.find(rowCountConstant);
		CCV_NNC_MFA_PRECONDITION(rowCountConstantPosition != std::string::npos);
		source.erase(rowCountConstantPosition, rowCountConstant.size());
		const std::string::size_type argumentPosition = source.find("\tuint2 tgid [[threadgroup_position_in_grid]],");
		CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
		source.insert(argumentPosition, "\tdevice const uint* loadM [[buffer(8)]],\n");
		const std::string::size_type rowCountPosition = source.find("\tif (row >= row_count || hc == 0 || hc > 16)");
		CCV_NNC_MFA_PRECONDITION(rowCountPosition != std::string::npos);
		source.insert(rowCountPosition, "\tconst uniform<uint> row_count = make_uniform(loadM[0]);\n");
	}
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
