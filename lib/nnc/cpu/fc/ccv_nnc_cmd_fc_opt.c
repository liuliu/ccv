#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#if defined(HAVE_SSE2)
#include <xmmintrin.h>
#elif defined(HAVE_NEON)
#include <arm_neon.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#include "ccv_nnc_cmd_fc_opt.h"

int ccv_nnc_fc_forw_opt(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, const ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* b)
{
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
	int batch_size = ccv_max(1, b->info.dim[1]);
	int i;
	for (i = 0; i < batch_size; i++)
	{
		const float* const ap = a->data.f32 + i * ainc[0];
		float* const bp = b->data.f32 + i * binc[0];
		parallel_for(j, b->info.dim[0]) {
			float v = bias->data.f32[j];
			const float* const wp = w->data.f32 + j * winc[0];
			int k;
			for (k = 0; k < a->info.dim[0]; k++)
				v += wp[k] * ap[k];
			bp[j] = v;
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_fc_back_opt(const ccv_nnc_tensor_view_t* g, const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, ccv_nnc_tensor_view_t* dw, ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* h, int flags)
{
	const int* dwinc = CCV_IS_TENSOR_VIEW(dw) ? dw->inc : dw->info.dim;
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(dw->data.u8, 0, sizeof(float) * dwinc[0] * dw->info.dim[1]);
		memset(bias->data.u8, 0, sizeof(float) * bias->info.dim[0]);
	}
	int batch_size = ccv_max(1, g->info.dim[1]);
	int i, j;
	float* gp = g->data.f32;
	float* bp = bias->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < g->info.dim[0]; j++)
			bp[j] += gp[j];
		gp += ginc[0];
	}
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		const float* const gp = g->data.f32 + i * ginc[0];
		const float* const ap = a->data.f32 + i * ainc[0];
		parallel_for(j, g->info.dim[0]) {
			float* const dwp = dw->data.f32 + j * dwinc[0];
			const float v = gp[j];
			int k;
			for (k = 0; k < a->info.dim[0]; k++)
				dwp[k] += ap[k] * v;
		} parallel_endfor
	}
	if (h && w)
	{
		const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
		const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
		for (i = 0; i < batch_size; i++)
		{
			const float* const gp = g->data.f32 + i * ginc[0];
			float* const hp = h->data.f32 + i * hinc[0];
			parallel_for(j, h->info.dim[0]) {
				const float* const wp = w->data.f32 + j;
				float v = 0;
				int k;
				for (k = 0; k < g->info.dim[0]; k++)
					v += wp[k * winc[0]] * gp[k];
				hp[j] = v;
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}
