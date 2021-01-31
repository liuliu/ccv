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

static int _ccv_nnc_smooth_l1_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int cinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(c, cinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int cstep = ccv_nnc_tensor_nd(c->info.dim) == 1 ? 1 : cinc[CCV_NNC_MAX_DIM + 1];
	const float beta = cmd.info.smooth_l1.beta;
	const float beta_inv_2 = 0.5 / beta;
	const float beta_2 = 0.5 * beta;
	parallel_for(i, batch_size) {
		int j;
		const float* const ap = a->data.f32 + i * astep;
		const float* const bp = b->data.f32 + i * bstep;
		float cp = 0;
		for (j = 0; j < count; j++)
			cp += fabs(bp[j] - ap[j]);
		if (cp < beta)
		{
			cp = 0;
			for (j = 0; j < count; j++)
				cp += (bp[j] - ap[j]) * (bp[j] - ap[j]);
			cp *= beta_inv_2;
		} else
			cp -= beta_2;
		c->data.f32[i * cstep] = cp;
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_smooth_l1_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int cinc[CCV_NNC_MAX_DIM_ALLOC];
	int hinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(h, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(c, cinc);
	ccv_nnc_tensor_view_get_inc(h, hinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int hstep = hinc[CCV_NNC_MAX_DIM + 1];
	const int cstep = ccv_nnc_tensor_nd(c->info.dim) == 1 ? 1 : cinc[CCV_NNC_MAX_DIM + 1];
	const float beta = cmd.info.smooth_l1.beta;
	const float beta_2 = 0.5 * beta;
	const float inv_beta = 1.0 / beta;
	if (g)
	{
		int ginc[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_inc(g, ginc);
		assert(ccv_nnc_tensor_count(g->info) == batch_size);
		const int gstep = ccv_nnc_tensor_nd(g->info.dim) == 1 ? 1 : ginc[CCV_NNC_MAX_DIM + 1];
		parallel_for(i, batch_size) {
			int j;
			const float cp = c->data.f32[i * cstep];
			const float* const ap = a->data.f32 + i * astep;
			const float* const bp = b->data.f32 + i * bstep;
			float* const hp = h->data.f32 + i * hstep;
			if (cp < beta_2)
			{
				const float gp = inv_beta * g->data.f32[i * gstep];
				for (j = 0; j < count; j++)
					hp[j] = gp * (ap[j] - bp[j]);
			} else {
				const float gp = g->data.f32[i * gstep];
				for (j = 0; j < count; j++)
					hp[j] = ((ap[j] - bp[j]) > 0 ? 1 : -1) * gp;
			}
		} parallel_endfor
	} else {
		parallel_for(i, batch_size) {
			int j;
			const float cp = c->data.f32[i * cstep];
			const float* const ap = a->data.f32 + i * astep;
			const float* const bp = b->data.f32 + i * bstep;
			float* const hp = h->data.f32 + i * hstep;
			if (cp < beta_2)
				for (j = 0; j < count; j++)
					hp[j] = inv_beta * (ap[j] - bp[j]);
			else
				for (j = 0; j < count; j++)
					hp[j] = (ap[j] - bp[j]) > 0 ? 1 : -1;
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SMOOTH_L1_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_smooth_l1_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SMOOTH_L1_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_smooth_l1_back;
}
