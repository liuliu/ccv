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

typedef struct {
	float v[5];
} float5;
#define less_than(a, b, aux) ((a).v[0] > (b).v[0])
#define swap_func(a, b, array, aux, t) do { \
	(t) = (a); \
	(a) = (b); \
	(b) = (t); \
	int _t = aux[&(a) - array]; \
	aux[&(a) - array] = aux[&(b) - array]; \
	aux[&(b) - array] = _t; \
} while (0)
CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_nms_sortby_f5_32f, float5, less_than, swap_func, int*)
#undef less_than
#undef swap_func

static int _ccv_nnc_nms_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 2);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[1];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(a_nd == b_nd);
	int i;
	for (i = 0; i < a_nd; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? c->inc : c->info.dim;
	const int n = a_nd >= 3 ? a->info.dim[0] : 1;
	const int aninc = a_nd >= 3 ? ainc[1] * ainc[2] : 0;
	const int bninc = b_nd >= 3 ? binc[1] * binc[2] : 0;
	const int cninc = c_nd >= 2 ? cinc[1] : 0;
	const int m = a_nd >= 3 ? a->info.dim[1] : a->info.dim[0];
	if (c_nd == 1)
		{ assert(m == c->info.dim[0]); }
	else
		{ assert(c_nd == 2 && n == c->info.dim[0] && m == c->info.dim[1]); }
	const int aminc = ainc[a_nd - 1];
	const int bminc = binc[b_nd - 1];
	const int d = a_nd <= 1 ? 1 : a->info.dim[a_nd - 1];
	const float iou_threshold = cmd.info.nms.iou_threshold;
	if (d == 5 && aminc == 5 && aminc == bminc) // If it is 5, we can use our quick sort implementation.
	{
		parallel_for(i, n)
		{
			int x, y;
			const float* const ap = a->data.f32 + i * aninc;
			float* const bp = b->data.f32 + i * bninc;
			int* const cp = c->data.i32 + i * cninc;
			for (x = 0; x < m; x++)
				cp[x] = x;
			for (x = 0; x < m * d; x++)
				bp[x] = ap[x];
			_ccv_nnc_nms_sortby_f5_32f((float5*)bp, m, cp);
			for (x = 0; x < m; x++)
			{
				float v = bp[x * 5];
				if (v == -FLT_MAX) // Suppressed.
					continue;
				const float area1 = bp[x * 5 + 3] * bp[x * 5 + 4];
				for (y = x + 1; y < m; y++)
				{
					const float u = bp[y * 5];
					if (u == -FLT_MAX) // Suppressed.
						continue;
					const float area2 = bp[y * 5 + 3] * bp[y * 5 + 4];
					const float xdiff = ccv_max(0, ccv_min(bp[x * 5 + 1] + bp[x * 5 + 3], bp[y * 5 + 1] + bp[y * 5 + 3]) - ccv_max(bp[x * 5 + 1], bp[y * 5 + 1]));
					const float ydiff = ccv_max(0, ccv_min(bp[x * 5 + 2] + bp[x * 5 + 4], bp[y * 5 + 2] + bp[y * 5 + 4]) - ccv_max(bp[x * 5 + 2], bp[y * 5 + 2]));
					const float intersection = xdiff * ydiff;
					const float iou = intersection / (area1 + area2 - intersection);
					if (iou >= iou_threshold)
						bp[y * 5] = -FLT_MAX;
				}
			}
			// Move these values up and move suppressed to the end.
			for (x = 0, y = 0; x < m; x++)
				if (bp[x * 5] != -FLT_MAX)
				{
					int j;
					if (x != y)
					{
						for (j = 0; j < 5; j++)
							bp[y * 5 + j] = bp[x * 5 + j];
						cp[y] = cp[x];
					}
					++y;
				}
			for (x = y; x < m; x++)
				cp[x] = -1, bp[x * 5] = -FLT_MAX;
		} parallel_endfor
	} else {
		// Otherwise, fall to use selection sort.
		parallel_for(i, n)
		{
			int x, y;
			const float* const ap = a->data.f32 + i * aninc;
			float* const bp = b->data.f32 + i * bninc;
			int* const cp = c->data.i32 + i * cninc;
			for (x = 0; x < m; x++)
				cp[x] = x;
			for (x = 0; x < m; x++)
				for (y = 0; y < d; y++)
					bp[x * bminc + y] = ap[x * aminc + y];
			for (x = 0; x < m; x++)
			{
				float v = bp[x * bminc];
				int k = x;
				for (y = x + 1; y < m; y++)
				{
					const float u = bp[y * bminc];
					if (u > v)
						k = y, v = u;
				}
				for (y = 0; y < d; y++)
				{
					const float t = bp[k * bminc + y];
					bp[k * bminc + y] = bp[x * bminc + y];
					bp[x * bminc + y] = t;
					const int u = cp[k];
					cp[k] = cp[x];
					cp[x] = u;
				}
			}
			for (x = 0; x < m; x++)
			{
				float v = bp[x * bminc];
				if (v == -FLT_MAX) // Suppressed.
					continue;
				const float area1 = bp[x * bminc + 3] * bp[x * bminc + 4];
				for (y = x + 1; y < m; y++)
				{
					const float u = bp[y * bminc];
					if (u == -FLT_MAX) // Suppressed.
						continue;
					const float area2 = bp[y * bminc + 3] * bp[y * bminc + 4];
					const float xdiff = ccv_max(0, ccv_min(bp[x * bminc + 1] + bp[x * bminc + 3], bp[y * bminc + 1] + bp[y * bminc + 3]) - ccv_max(bp[x * bminc + 1], bp[y * bminc + 1]));
					const float ydiff = ccv_max(0, ccv_min(bp[x * bminc + 2] + bp[x * bminc + 4], bp[y * bminc + 2] + bp[y * bminc + 4]) - ccv_max(bp[x * bminc + 2], bp[y * bminc + 2]));
					const float intersection = xdiff * ydiff;
					const float iou = intersection / (area1 + area2 - intersection);
					if (iou >= iou_threshold)
						bp[y * bminc] = -FLT_MAX;
				}
			}
			for (x = 0, y = 0; x < m; x++)
				if (bp[x * bminc] != -FLT_MAX)
				{
					int j;
					if (x != y)
					{
						for (j = 0; j < 5; j++)
							bp[y * bminc + j] = bp[x * bminc + j];
						cp[y] = cp[x];
					}
					++y;
				}
			for (x = y; x < m; x++)
				cp[x] = -1, bp[x * bminc] = -FLT_MAX;
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_nms_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 5);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)inputs[4];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
	assert(a_nd == b_nd);
	int i;
	for (i = 0; i < a_nd; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int* cinc = CCV_IS_TENSOR_VIEW(c) ? c->inc : c->info.dim;
	const int n = a_nd >= 3 ? a->info.dim[0] : 1;
	const int aninc = a_nd >= 3 ? ainc[1] * ainc[2] : 0;
	const int bninc = b_nd >= 3 ? binc[1] * binc[2] : 0;
	const int cninc = c_nd >= 2 ? cinc[1] : 0;
	const int m = a_nd >= 3 ? a->info.dim[1] : a->info.dim[0];
	if (c_nd == 1)
		{ assert(m == c->info.dim[0]); }
	else
		{ assert(c_nd == 2 && n == c->info.dim[0] && m == c->info.dim[1]); }
	const int aminc = ainc[a_nd - 1];
	const int bminc = binc[b_nd - 1];
	const int d = a_nd <= 1 ? 1 : a->info.dim[a_nd - 1];
	parallel_for(i, n)
	{
		int x, y;
		const float* const ap = a->data.f32 + i * aninc;
		const int* const cp = c->data.i32 + i * cninc;
		float* const bp = b->data.f32 + i * bninc;
		for (x = 0; x < m; x++)
			for (y = 0; y < d; y++)
				bp[x * bminc + y] = 0;
		for (x = 0; x < m; x++)
		{
			const int k = cp[x];
			if (k < 0)
				break;
			for (y = 0; y < d; y++)
				bp[k * bminc + y] = ap[x * aminc + y];
		}
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_NMS_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_nms_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_NMS_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_nms_back;
}
