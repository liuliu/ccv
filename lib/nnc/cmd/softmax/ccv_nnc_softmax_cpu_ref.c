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

static int _ccv_nnc_softmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(b));
	const int axis_count = ccv_nnc_tensor_nd(a->info.dim);
	const int batch_size = axis_count < 2 ? 1 : a->info.dim[0];
	const int count = ccv_nnc_tensor_count(a->info) / batch_size;
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }

	parallel_for(i, batch_size) {
		int j;
		float* const ap = a->data.f32 + i * count;
		float* const bp = b->data.f32 + i * count;
		double maxval = ap[0];
		for (j = 1; j < count; j++)
			if (ap[j] > maxval)
				maxval = ap[j];
		double sumval = 0;
		for (j = 0; j < count; j++)
			sumval += (bp[j] = expf(ap[j] - maxval));
		sumval = 1.0 / sumval;
		for (j = 0; j < count; j++)
			bp[j] *= sumval;
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_softmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* g = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_t* b = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(b));
	ccv_nnc_tensor_t* h = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(h));
	const int axis_count = ccv_nnc_tensor_nd(g->info.dim);
	const int batch_size = axis_count < 2 ? 1 : g->info.dim[0];
	const int count = ccv_nnc_tensor_count(g->info) / batch_size;
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
		{ assert(g->info.dim[i] == h->info.dim[i] && h->info.dim[i] == b->info.dim[i]); }
	parallel_for(i, batch_size) {
		int j;
		float* const gp = g->data.f32 + i * count;
		float* const bp = b->data.f32 + i * count;
		float* const hp = h->data.f32 + i * count;
		float sumval = 0;
		for (j = 0; j < count; j++)
			sumval += gp[j] * bp[j];
		for (j = 0; j < count; j++)
			hp[j] = (gp[j] - sumval) * bp[j];
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SOFTMAX_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_softmax_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SOFTMAX_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_softmax_back;
}
