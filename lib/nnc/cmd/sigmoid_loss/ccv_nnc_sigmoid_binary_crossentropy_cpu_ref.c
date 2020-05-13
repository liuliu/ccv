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

static int _ccv_nnc_sigmoid_binary_crossentropy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 2);
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const d = (ccv_nnc_tensor_view_t*)outputs[1];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int dinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(d, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(d, dinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int dstep = dinc[CCV_NNC_MAX_DIM + 1];
	if (c)
	{
		int cinc[CCV_NNC_MAX_DIM_ALLOC];
		assert(ccv_nnc_tensor_count(c->info) == batch_size);
		ccv_nnc_tensor_view_get_inc(c, cinc);
		const int cstep = cinc[CCV_NNC_MAX_DIM + 1];
		parallel_for(i, batch_size) {
			int j;
			const float* const ap = a->data.f32 + i * astep;
			const float* const bp = b->data.f32 + i * bstep;
			float* const dp = d->data.f32 + i * dstep;
			float cp = 0;
			for (j = 0; j < count; j++)
			{
				cp += (1 - bp[j]) * ap[j] + log(1. + exp(-ap[j]));
				dp[j] = 1. / (1. + exp(-ap[j]));
			}
			c->data.f32[i * cstep] = cp;
		} parallel_endfor
	} else {
		parallel_for(i, batch_size) {
			int j;
			const float* const ap = a->data.f32 + i * astep;
			float* const dp = d->data.f32 + i * dstep;
			for (j = 0; j < count; j++)
				dp[j] = 1. / (1. + exp(-ap[j]));
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sigmoid_binary_crossentropy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 6);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[5];
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int hinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(h, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(h, hinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int hstep = hinc[CCV_NNC_MAX_DIM + 1];
	if (g)
	{
		int ginc[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_inc(g, ginc);
		assert(ccv_nnc_tensor_count(g->info) == batch_size);
		const int gstep = ginc[CCV_NNC_MAX_DIM + 1];
		parallel_for(i, batch_size) {
			int j;
			const float gp = g->data.f32[i * gstep];
			const float* const ap = a->data.f32 + i * astep;
			const float* const bp = b->data.f32 + i * bstep;
			float* const hp = h->data.f32 + i * hstep;
			for (j = 0; j < count; j++)
				hp[j] = gp * (ap[j] - bp[j]);
		} parallel_endfor
	} else {
		parallel_for(i, batch_size) {
			int j;
			const float* const ap = a->data.f32 + i * astep;
			const float* const bp = b->data.f32 + i * bstep;
			float* const hp = h->data.f32 + i * hstep;
			for (j = 0; j < count; j++)
				hp[j] = ap[j] - bp[j];
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sigmoid_binary_crossentropy_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sigmoid_binary_crossentropy_back;
}
