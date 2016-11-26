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
#include "../_ccv_nnc_gemm_cpu_opt.h"

#ifdef HAVE_SSE2
static int _ccv_nnc_gemm_forw_sse2(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, const ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* b)
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
			const float* const wp = w->data.f32 + j * winc[0];
			int k;
			__m128 v40 = _mm_set_ss(bias->data.f32[j]);
			__m128 v41 = _mm_setzero_ps();
			for (k = 0; k < a->info.dim[0] - 7; k += 8)
			{
				__m128 ap40 = _mm_load_ps(ap + k);
				__m128 ap41 = _mm_load_ps(ap + k + 4);
				__m128 w40 = _mm_load_ps(wp + k);
				__m128 w41 = _mm_load_ps(wp + k + 4);
				v40 =_mm_add_ps(_mm_mul_ps(w40, ap40), v40);
				v41 =_mm_add_ps(_mm_mul_ps(w41, ap41), v41);
			}
			v40 = _mm_add_ps(v40, v41);
			v41 = _mm_add_ps(v40, _mm_movehl_ps(v40, v40));
			v40 = _mm_add_ss(v41, _mm_shuffle_ps(v41, v41, 1));
			_mm_store_ss(bp + j, v40);
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back_sse2(const ccv_nnc_tensor_view_t* g, const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, ccv_nnc_tensor_view_t* dw, ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* h, int flags)
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
		for (j = 0; j < g->info.dim[0] - 3; j += 4)
		{
			__m128 g4 = _mm_load_ps(gp + j);
			__m128 b4 = _mm_load_ps(bp + j);
			_mm_stream_ps(bp + j, _mm_add_ps(b4, g4));
		}
		gp += ginc[0];
	}
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		const float* const gp = g->data.f32 + i * ginc[0];
		const float* const ap = a->data.f32 + i * ainc[0];
		parallel_for(j, g->info.dim[0]) {
			float* const dwp = dw->data.f32 + j * dwinc[0];
			__m128 g4 = _mm_set1_ps(gp[j]);
			int k;
			for (k = 0; k < a->info.dim[0] - 3; k+= 4)
			{
				__m128 a4 = _mm_load_ps(ap + k);
				__m128 dw4 = _mm_load_ps(dwp + k);
				_mm_stream_ps(dwp + k, _mm_add_ps(dw4, _mm_mul_ps(a4, g4)));
			}
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
			parallel_for(y, h->info.dim[0] / 4) {
				const int j = y * 4;
				const float* const wp = w->data.f32 + j;
				__m128 v40 = _mm_setzero_ps();
				__m128 v41 = _mm_setzero_ps();
				__m128 v42 = _mm_setzero_ps();
				__m128 v43 = _mm_setzero_ps();
				int k;
				for (k = 0; k < g->info.dim[0]; k += 4)
				{
					__m128 g4 = _mm_load_ps(gp + k);
					__m128 w40 = _mm_load_ps(wp + k * winc[0]);
					__m128 w41 = _mm_load_ps(wp + (k + 1) * winc[0]);
					__m128 w42 = _mm_load_ps(wp + (k + 2) * winc[0]);
					__m128 w43 = _mm_load_ps(wp + (k + 3) * winc[0]);
					__m128 g40 = _mm_shuffle_ps(g4, g4, 0x00);
					__m128 g41 = _mm_shuffle_ps(g4, g4, 0x55);
					__m128 g42 = _mm_shuffle_ps(g4, g4, 0xAA);
					__m128 g43 = _mm_shuffle_ps(g4, g4, 0xFF);
					v40 = _mm_add_ps(_mm_mul_ps(g40, w40), v40);
					v41 = _mm_add_ps(_mm_mul_ps(g41, w41), v41);
					v42 = _mm_add_ps(_mm_mul_ps(g42, w42), v42);
					v43 = _mm_add_ps(_mm_mul_ps(g43, w43), v43);
				}
				v40 = _mm_add_ps(v40, v41);
				v42 = _mm_add_ps(v42, v43);
				_mm_stream_ps(hp + j, _mm_add_ps(v40, v42));
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}
#endif

#ifdef HAVE_NEON
static int _ccv_nnc_gemm_forw_neon(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, const ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* b)
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
			const float* const wp = w->data.f32 + j * winc[0];
			int k;
			float32x4_t v41 = vmovq_n_f32(0);
			float32x4_t v40 = vld1q_lane_f32(bias->data.f32 + j, v41, 0);
			for (k = 0; k < a->info.dim[0] - 7; k += 8)
			{
				float32x4_t ap40 = vld1q_f32(ap + k);
				float32x4_t ap41 = vld1q_f32(ap + k + 4);
				float32x4_t w40 = vld1q_f32(wp + k);
				float32x4_t w41 = vld1q_f32(wp + k + 4);
				v40 = vmlaq_f32(v40, w40, ap40);
				v41 = vmlaq_f32(v41, w41, ap41);
			}
			v40 = vaddq_f32(v40, v41);
			float32x2_t v2 = vpadd_f32(vget_high_f32(v40), vget_low_f32(v40));
			bp[j] = vget_lane_f32(vpadd_f32(v2, v2), 0);
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back_neon(const ccv_nnc_tensor_view_t* g, const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, ccv_nnc_tensor_view_t* dw, ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* h, int flags)
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
		for (j = 0; j < g->info.dim[0] - 3; j += 4)
		{
			float32x4_t g4 = vld1q_f32(gp + j);
			float32x4_t b4 = vld1q_f32(bp + j);
			vst1q_f32(bp + j, vaddq_f32(b4, g4));
		}
		gp += ginc[0];
	}
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		const float* const gp = g->data.f32 + i * ginc[0];
		const float* const ap = a->data.f32 + i * ainc[0];
		parallel_for(j, g->info.dim[0]) {
			float* const dwp = dw->data.f32 + j * dwinc[0];
			float32x4_t g4 = vld1q_dup_f32(gp + j);
			int k;
			for (k = 0; k < a->info.dim[0] - 3; k+= 4)
			{
				float32x4_t a4 = vld1q_f32(ap + k);
				float32x4_t dw4 = vld1q_f32(dwp + k);
				vst1q_f32(dwp + k, vmlaq_f32(dw4, a4, g4));
			}
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
			parallel_for(y, h->info.dim[0] / 4) {
				const int j = y * 4;
				const float* const wp = w->data.f32 + j;
				float32x4_t v40 = vmovq_n_f32(0);
				float32x4_t v41 = vmovq_n_f32(0);
				float32x4_t v42 = vmovq_n_f32(0);
				float32x4_t v43 = vmovq_n_f32(0);
				int k;
				for (k = 0; k < g->info.dim[0]; k += 4)
				{
					float32x2x2_t g4 = vld2_f32(gp + k);
					float32x4_t w40 = vld1q_f32(wp + k * winc[0]);
					float32x4_t w41 = vld1q_f32(wp + (k + 1) * winc[0]);
					float32x4_t w42 = vld1q_f32(wp + (k + 2) * winc[0]);
					float32x4_t w43 = vld1q_f32(wp + (k + 3) * winc[0]);
					float32x4_t g40 = vdupq_lane_f32(g4.val[0], 0);
					float32x4_t g41 = vdupq_lane_f32(g4.val[1], 0);
					float32x4_t g42 = vdupq_lane_f32(g4.val[0], 1);
					float32x4_t g43 = vdupq_lane_f32(g4.val[1], 1);
					v40 = vmlaq_f32(v40, g40, w40);
					v41 = vmlaq_f32(v41, g41, w41);
					v42 = vmlaq_f32(v42, g42, w42);
					v43 = vmlaq_f32(v43, g43, w43);
				}
				v40 = vaddq_f32(v40, v41);
				v42 = vaddq_f32(v42, v43);
				vst1q_f32(hp + j, vaddq_f32(v40, v42));
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}
#endif

int _ccv_nnc_gemm_forw_cpu_opt(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, const ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* b)
{
#if defined(HAVE_SSE2)
	if (a->info.dim[0] % 8 == 0)
		return _ccv_nnc_gemm_forw_sse2(a, w, bias, b);
#elif defined(HAVE_NEON)
	if (a->info.dim[0] % 8 == 0)
		return _ccv_nnc_gemm_forw_neon(a, w, bias, b);
#endif
	return CCV_NNC_EXEC_INVALID;
}

int _ccv_nnc_gemm_back_cpu_opt(const ccv_nnc_tensor_view_t* g, const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, ccv_nnc_tensor_view_t* dw, ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* h, int flags)
{
#if defined(HAVE_SSE2)
	if (g->info.dim[0] % 4 == 0 && a->info.dim[0] % 4 == 0 && (!h || h->info.dim[0] % 4 == 0))
		return _ccv_nnc_gemm_back_sse2(g, a, w, dw, bias, h, flags);
#elif defined(HAVE_NEON)
	if (g->info.dim[0] % 4 == 0 && a->info.dim[0] % 4 == 0 && (!h || h->info.dim[0] % 4 == 0))
		return _ccv_nnc_gemm_back_neon(g, a, w, dw, bias, h, flags);
#endif
	return CCV_NNC_EXEC_INVALID;
}
