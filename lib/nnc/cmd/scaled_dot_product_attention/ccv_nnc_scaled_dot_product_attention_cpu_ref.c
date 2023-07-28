#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_scaled_dot_product_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const w = input_size > 3 ? (ccv_nnc_tensor_view_t*)inputs[3] : 0;
	ccv_nnc_tensor_view_t* const bias = input_size > 4 ? (ccv_nnc_tensor_view_t*)inputs[4] : 0;
	ccv_nnc_tensor_view_t* const attn_mask = input_size > 5 ? (ccv_nnc_tensor_view_t*)inputs[5] : 0;
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const saved_softmax = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	assert(v_nd == 3 || v_nd == 4);
	assert(q_nd == k_nd && k_nd == v_nd);
	// Assuming this is float 32.
	int qdim[CCV_NNC_MAX_DIM_ALLOC];
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int vdim[CCV_NNC_MAX_DIM_ALLOC];
	int cdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(c, cdim);
	assert(qdim[0] == kdim[0] && kdim[0] == vdim[0] && vdim[0] == cdim[0]);
	assert(qdim[1] == kdim[1] && kdim[1] == vdim[1] && vdim[1] == cdim[1]);
	assert(qdim[3] == kdim[3]);
	assert(kdim[2] == vdim[2]);
	assert(cdim[2] == qdim[2]);
	assert(cdim[3] == vdim[3]);
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int qstride[CCV_NNC_MAX_DIM_ALLOC];
	int kstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(q, qstride);
	ccv_nnc_tensor_view_get_stride(k, kstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	ccv_nnc_tensor_view_get_stride(c, cstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* qk = ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * qdim[2] * kdim[2], CCV_TENSOR_CPU_MEMORY);
	const float* const qp = q->data.f32;
	const float* const kp = k->data.f32;
	const float* const vp = v->data.f32;
	float* const cp = c->data.f32;
	const float scale = cmd.info.scaled_dot_product_attention.scale;
	for (i[0] = 0; i[0] < qdim[0]; i[0]++)
	{
		const float* const qp0 = qp + i[0] * qstride[0];
		const float* const kp0 = kp + i[0] * kstride[0];
		const float* const vp0 = vp + i[0] * vstride[0];
		float* const cp0 = cp + i[0] * cstride[0];
		for (i[1] = 0; i[1] < qdim[1]; i[1]++)
		{
			const float* const qp1 = qp0 + i[1] * qstride[1];
			const float* const kp1 = kp0 + i[1] * kstride[1];
			const float* const vp1 = vp0 + i[1] * vstride[1];
			float* const cp1 = cp0 + i[1] * cstride[1];
			// Compute Q @ K^T
			parallel_for(x, qdim[2]) {
				int y, k;
				const float* const qp2 = qp1 + x * qstride[2];
				float* const cp2 = cp1 + x * cstride[2];
				float* const qk0 = qk + x * kdim[2];
				for (y = 0; y < kdim[2]; y++)
				{
					const float* const kp2 = kp1 + y * kstride[2];
					float v = 0;
					for (k = 0; k < qdim[3]; k++)
						v += qp2[k * qstride[3]] * kp2[k * kstride[3]];
					qk0[y] = scale * v;
				}
				// Compute softmax on qk.
				double maxval = qk0[0];
				for (y = 1; y < kdim[2]; y++)
					if (qk0[y] > maxval)
						maxval = qk0[y];
				double sumval = 0;
				for (y = 0; y < kdim[2]; y++)
					sumval += (qk0[y] = expf(qk0[y] - maxval));
				sumval = 1.0 / sumval;
				for (y = 0; y < kdim[2]; y++)
					qk0[y] *= sumval;
				for (k = 0; k < vdim[3]; k++)
					cp2[k * cstride[3]] = 0;
				for (y = 0; y < kdim[2]; y++)
				{
					const float* const vp2 = vp1 + y * vstride[2];
					const float v = qk0[y];
					for (k = 0; k < vdim[3]; k++)
						cp2[k * cstride[3]] += v * vp2[k * vstride[3]];
				}
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
}
