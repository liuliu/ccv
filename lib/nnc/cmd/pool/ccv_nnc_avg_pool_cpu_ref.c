#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_avg_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ?  a->inc : a->inc + 1) : adim;
	float* bp = b->data.f32;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ?  b->inc : b->inc + 1) : bdim;
	for (i[0] = 0; i[0] < bdim[0]; i[0]++)
	{
		SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, dim, adim, n, m);
		for (i[1] = 0; i[1] < bdim[1]; i[1]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, dim, adim, n, m);
			for (c = 0; c < bdim[CCV_NNC_MAX_DIM]; c++)
			{
				float* apz = ap + ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[CCV_NNC_MAX_DIM] + c;
				float v = 0;
				for (j[0] = 0; j[0] < m[0]; j[0]++)
				{
					for (j[1] = 0; j[1] < m[1]; j[1]++)
						v += apz[j[1] * ainc[CCV_NNC_MAX_DIM]];
					apz += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM];
				}
				bp[i[1] * binc[CCV_NNC_MAX_DIM] + c] = v / (m[0] * m[1]);
			}
		}
		bp += binc[CCV_NNC_MAX_DIM - 1] * binc[CCV_NNC_MAX_DIM];
		ap += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_avg_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == CCV_NNC_MAX_DIM + 1 || g_nd == CCV_NNC_MAX_DIM + 2);
	const int* gdim = (g_nd == CCV_NNC_MAX_DIM + 1) ? g->info.dim : g->info.dim + 1;
	const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
	assert(h_nd == CCV_NNC_MAX_DIM + 1 || h_nd == CCV_NNC_MAX_DIM + 2);
	const int* hdim = (h_nd == CCV_NNC_MAX_DIM + 1) ? h->info.dim : h->info.dim + 1;
	float* gp = g->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? ((g_nd == CCV_NNC_MAX_DIM + 1) ? g->inc : g->inc + 1) : gdim;
	float* hp = h->data.f32;
	const int* hinc = CCV_IS_TENSOR_VIEW(h) ? ((h_nd == CCV_NNC_MAX_DIM + 1) ? h->inc : h->inc + 1) : hdim;
	ccv_nnc_tensor_zero(h);
	for (i[0] = 0; i[0] < gdim[0]; i[0]++)
	{
		SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, dim, hdim, n, m);
		for (i[1] = 0; i[1] < gdim[1]; i[1]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, dim, hdim, n, m);
			for (c = 0; c < gdim[CCV_NNC_MAX_DIM]; c++)
			{
				float* hpz = hp + ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) * hinc[CCV_NNC_MAX_DIM] + c;
				float u = gp[i[1] * ginc[CCV_NNC_MAX_DIM] + c] / (m[0] * m[1]);
				for (j[0] = 0; j[0] < m[0]; j[0]++)
				{
					for (j[1] = 0; j[1] < m[1]; j[1]++)
						hpz[j[1] * hinc[CCV_NNC_MAX_DIM]] += u;
					hpz += hinc[CCV_NNC_MAX_DIM - 1] * hinc[CCV_NNC_MAX_DIM];
				}
			}
		}
		gp += ginc[CCV_NNC_MAX_DIM - 1] * ginc[CCV_NNC_MAX_DIM];
		hp += hinc[CCV_NNC_MAX_DIM - 1] * hinc[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_AVERAGE_POOL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_avg_pool_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_AVERAGE_POOL_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_avg_pool_back;
}
