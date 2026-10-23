extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

// One warp owns one dv row of a head's state, so both dot products reduce with warp shuffles. A row that fits
// keeps DK_PER_LANE key channels per lane in registers for the whole sequence, DK_PER_LANE == 0 streams wider
// rows through state_out instead.
template<typename NUM, int DK_PER_LANE>
static __global__ void _ccv_nnc_gated_delta_forw_kernel(const int rows, const int T, const int Hk, const int Hv, const int Dk, const int Dv, const int checkpoints, const int log_decay_input, const int channel_decay, const NUM* const q, const NUM* const k, const NUM* const v, const float* const decay, const void* const beta, const int beta_float, const float* const state_in, NUM* const y, float* const state_out)
{
	const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
	if (row >= rows)
		return;
	const int lane = threadIdx.x % 32;
	const int dv = row % Dv;
	const int hv = (row / Dv) % Hv;
	const int b = row / (Dv * Hv);
	const int hk = hv / (Hv / Hk);
	const int dk_per_lane = DK_PER_LANE > 0 ? DK_PER_LANE : (Dk + 31) / 32;
	float* const state = state_out + (((size_t)b * (checkpoints + 1) * Hv + hv) * Dv + dv) * Dk;
	float cached[DK_PER_LANE > 0 ? DK_PER_LANE : 1];
#pragma unroll
	for (int j = 0; j < dk_per_lane; j++)
	{
		const int dk = lane + j * 32;
		const float value = dk < Dk ? state_in[(size_t)row * Dk + dk] : 0;
		if (DK_PER_LANE > 0)
			cached[j] = value;
		else if (dk < Dk)
			state[dk] = value;
	}
	for (int t = 0; t < T; t++)
	{
		const size_t qk_offset = (((size_t)b * T + t) * Hk + hk) * Dk;
		const size_t gate_offset = ((size_t)b * T + t) * Hv + hv;
		const size_t value_offset = gate_offset * Dv + dv;
		// The per head decay is shared by every key channel, evaluate it once per step.
		const float head_decay = channel_decay ? 1 : (log_decay_input ? expf(decay[gate_offset]) : decay[gate_offset]);
		float memory = 0;
#pragma unroll
		for (int j = 0; j < dk_per_lane; j++)
		{
			const int dk = lane + j * 32;
			if (dk < Dk)
			{
				float& s = DK_PER_LANE > 0 ? cached[j] : state[dk];
				const float g = channel_decay ? (log_decay_input ? expf(decay[gate_offset * Dk + dk]) : decay[gate_offset * Dk + dk]) : head_decay;
				s *= g;
				memory += s * (float)k[qk_offset + dk];
			}
		}
		for (int offset = 16; offset > 0; offset /= 2)
			memory += __shfl_xor_sync(0xffffffff, memory, offset);
		const float beta_value = beta_float ? ((const float*)beta)[gate_offset] : (float)((const NUM*)beta)[gate_offset];
		const float delta = ((float)v[value_offset] - memory) * beta_value;
		float out = 0;
#pragma unroll
		for (int j = 0; j < dk_per_lane; j++)
		{
			const int dk = lane + j * 32;
			if (dk < Dk)
			{
				float& s = DK_PER_LANE > 0 ? cached[j] : state[dk];
				s += delta * (float)k[qk_offset + dk];
				out += s * (float)q[qk_offset + dk];
			}
		}
		for (int offset = 16; offset > 0; offset /= 2)
			out += __shfl_xor_sync(0xffffffff, out, offset);
		if (lane == 0)
			y[value_offset] = (NUM)out;
		const int history_index = T - 1 - t;
		if (history_index > 0 && history_index <= checkpoints)
		{
			float* const checkpoint = state + (size_t)history_index * Hv * Dv * Dk;
#pragma unroll
			for (int j = 0; j < dk_per_lane; j++)
			{
				const int dk = lane + j * 32;
				if (dk < Dk)
					checkpoint[dk] = DK_PER_LANE > 0 ? cached[j] : state[dk];
			}
		}
	}
	if (DK_PER_LANE > 0)
	{
#pragma unroll
		for (int j = 0; j < dk_per_lane; j++)
		{
			const int dk = lane + j * 32;
			if (dk < Dk)
				state[dk] = cached[j];
		}
	}
}

template<typename NUM>
static void _ccv_nnc_gated_delta_forw_launch(cudaStream_t stream, const int rows, const int T, const int Hk, const int Hv, const int Dk, const int Dv, const int checkpoints, const int log_decay_input, const int channel_decay, const NUM* const q, const NUM* const k, const NUM* const v, const float* const decay, const void* const beta, const int beta_float, const float* const state_in, NUM* const y, float* const state_out)
{
	// 4 warps per block, consecutive rows of a head share the q / k loads through L1.
	const int blocks = (rows + 3) / 4;
#define GATED_DELTA_LAUNCH(DK_PER_LANE) \
	_ccv_nnc_gated_delta_forw_kernel<NUM, DK_PER_LANE><<<blocks, 128, 0, stream>>>(rows, T, Hk, Hv, Dk, Dv, checkpoints, log_decay_input, channel_decay, q, k, v, decay, beta, beta_float, state_in, y, state_out)
	if (Dk <= 32)
		GATED_DELTA_LAUNCH(1);
	else if (Dk <= 64)
		GATED_DELTA_LAUNCH(2);
	else if (Dk <= 128)
		GATED_DELTA_LAUNCH(4);
	else if (Dk <= 256)
		GATED_DELTA_LAUNCH(8);
	else
		GATED_DELTA_LAUNCH(0);
#undef GATED_DELTA_LAUNCH
}

static int _ccv_nnc_gated_delta_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6);
	assert(output_size == 2);
	if (outputs[0]->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const ccv_nnc_tensor_t* const q = inputs[0];
	const ccv_nnc_tensor_t* const k = inputs[1];
	const ccv_nnc_tensor_t* const v = inputs[2];
	const ccv_nnc_tensor_t* const decay = inputs[3];
	const ccv_nnc_tensor_t* const beta = inputs[4];
	const ccv_nnc_tensor_t* const state_in = inputs[5];
	ccv_nnc_tensor_t* const y = outputs[0];
	ccv_nnc_tensor_t* const state_out = outputs[1];
	const int datatype = q->info.datatype;
	if ((datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF) ||
		k->info.datatype != datatype || v->info.datatype != datatype || y->info.datatype != datatype ||
		decay->info.datatype != CCV_32F || (beta->info.datatype != CCV_32F && beta->info.datatype != datatype) ||
		state_in->info.datatype != CCV_32F || state_out->info.datatype != CCV_32F)
		return CCV_NNC_EXEC_INVALID;
	for (int i = 0; i < input_size; i++)
		if (!CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	if (!CCV_IS_TENSOR_CONTIGUOUS(y) || !CCV_IS_TENSOR_CONTIGUOUS(state_out))
		return CCV_NNC_EXEC_INVALID;
	assert(ccv_nnc_tensor_nd(q->info.dim) == 4);
	assert(ccv_nnc_tensor_nd(k->info.dim) == 4);
	assert(ccv_nnc_tensor_nd(v->info.dim) == 4);
	const int B = q->info.dim[0];
	const int T = q->info.dim[1];
	const int Hk = q->info.dim[2];
	const int Dk = q->info.dim[3];
	const int Hv = v->info.dim[2];
	const int Dv = v->info.dim[3];
	const int decay_nd = ccv_nnc_tensor_nd(decay->info.dim);
	const int checkpoints = cmd.info.gated_delta.state_checkpoint_count;
	assert(B > 0 && T > 0 && Hk > 0 && Dk > 0 && Hv > 0 && Dv > 0);
	assert(Hv % Hk == 0);
	assert(checkpoints >= 0 && checkpoints < T);
	assert(decay_nd == 3 || decay_nd == 4);
	assert(ccv_nnc_tensor_nd(beta->info.dim) == 3);
	assert(ccv_nnc_tensor_nd(state_in->info.dim) == 4);
	assert(ccv_nnc_tensor_nd(y->info.dim) == 4);
	assert(ccv_nnc_tensor_nd(state_out->info.dim) == 4);
	assert(k->info.dim[0] == B && k->info.dim[1] == T && k->info.dim[2] == Hk && k->info.dim[3] == Dk);
	assert(v->info.dim[0] == B && v->info.dim[1] == T);
	assert(decay->info.dim[0] == B && decay->info.dim[1] == T && decay->info.dim[2] == Hv);
	assert(decay_nd == 3 || decay->info.dim[3] == Dk);
	assert(beta->info.dim[0] == B && beta->info.dim[1] == T && beta->info.dim[2] == Hv);
	assert(state_in->info.dim[0] == B && state_in->info.dim[1] == Hv && state_in->info.dim[2] == Dv && state_in->info.dim[3] == Dk);
	assert(y->info.dim[0] == B && y->info.dim[1] == T && y->info.dim[2] == Hv && y->info.dim[3] == Dv);
	assert(state_out->info.dim[0] == B && state_out->info.dim[1] == Hv * (checkpoints + 1) && state_out->info.dim[2] == Dv && state_out->info.dim[3] == Dk);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int rows = B * Hv * Dv;
	const int log_decay_input = cmd.info.gated_delta.log_decay;
	const int beta_float = beta->info.datatype == CCV_32F;
	if (datatype == CCV_32F)
		_ccv_nnc_gated_delta_forw_launch(stream, rows, T, Hk, Hv, Dk, Dv, checkpoints, log_decay_input, decay_nd == 4, q->data.f32, k->data.f32, v->data.f32, decay->data.f32, beta->data.u8, beta_float, state_in->data.f32, y->data.f32, state_out->data.f32);
	else if (datatype == CCV_16F)
		_ccv_nnc_gated_delta_forw_launch(stream, rows, T, Hk, Hv, Dk, Dv, checkpoints, log_decay_input, decay_nd == 4, (const __half*)q->data.f16, (const __half*)k->data.f16, (const __half*)v->data.f16, decay->data.f32, beta->data.u8, beta_float, state_in->data.f32, (__half*)y->data.f16, state_out->data.f32);
	else
		_ccv_nnc_gated_delta_forw_launch(stream, rows, T, Hk, Hv, Dk, Dv, checkpoints, log_decay_input, decay_nd == 4, (const __nv_bfloat16*)q->data.f16, (const __nv_bfloat16*)k->data.f16, (const __nv_bfloat16*)v->data.f16, decay->data.f32, beta->data.u8, beta_float, state_in->data.f32, (__nv_bfloat16*)y->data.f16, state_out->data.f32);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gated_delta_forw;
#endif
}
