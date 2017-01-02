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

static int _ccv_nnc_max_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
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
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	float* bp = b->data.f32;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, dim, a->info.dim, n, m);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, dim, a->info.dim, n, m);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0] + c;
				float v = apz[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						if (apz[j[0] * ainc[0]] > v)
							v = apz[j[0] * ainc[0]];
					apz += ainc[1] * ainc[0];
				}
				bp[i[0] * binc[0] + c] = v;
			}
		}
		bp += binc[1] * binc[0];
		ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_max_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	float* bp = b->data.f32;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	float* gp = g->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	float* hp = h->data.f32;
	const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
	for (c = 0; c < CCV_NNC_MAX_DIM_ALLOC; c++)
	{
		assert(a->info.dim[c] == h->info.dim[c]);
		if (a->info.dim[c] == 0 || h->info.dim[c] == 0)
			break;
	}
	for (c = 0; c < CCV_NNC_MAX_DIM_ALLOC; c++)
	{
		assert(b->info.dim[c] == g->info.dim[c]);
		if (b->info.dim[c] == 0 || g->info.dim[c] == 0)
			break;
	}
	ccv_nnc_tensor_zero(h);
	// Using b->info.dim and a->info.dim directly because they equal to g->info.dim and h->info.dim
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, dim, a->info.dim, n, m);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, dim, a->info.dim, n, m);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0] + c;
				float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * hinc[0] + c;
				float v = bp[i[0] * binc[0] + c];
				float u = gp[i[0] * ginc[0] + c];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						if (apz[j[0] * ainc[0]] == v)
							hpz[j[0] * hinc[0]] += u;
					apz += ainc[1] * ainc[0];
					hpz += hinc[1] * hinc[0];
				}
			}
		}
		gp += ginc[1] * ginc[0];
		bp += binc[1] * binc[0];
		ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		hp += hinc[1] * hinc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MAX_POOL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_max_pool_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MAX_POOL_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_max_pool_back;
}
