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

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = (const ccv_nnc_tensor_view_t*)inputs[2];
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(b->info.dim[2] == 0); // It is a 2-d array.
	assert(w->info.dim[2] == 0); // It is a 2-d array
	assert(bias->info.dim[1] == 0); // It is a 1-d array
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == 1 || a_nd == 2);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == 1 || b_nd == 2);
	const int* bdim = (b_nd == 1) ? b->info.dim : b->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	assert(batch_size == (b_nd == 1) ? 1 : ccv_max(1, b->info.dim[0]));
	assert(bdim[0] == bias->info.dim[0]);
	assert(bdim[0] == w->info.dim[0]);
	assert(adim[0] == w->info.dim[1]);
	const int a_batch_inc = CCV_IS_TENSOR_VIEW(a) ? (a_nd == 1 ? 1 : a->inc[0]) : batch_size;
	const int b_batch_inc = CCV_IS_TENSOR_VIEW(b) ? (b_nd == 1 ? 1 : b->inc[0]) : batch_size;
	const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
	const float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	parallel_for(i, bdim[0]) {
		const float* const wp = w->data.f32 + i * winc[1];
		int j, k;
		for (k = 0; k < batch_size; k++)
			bp[i * b_batch_inc + k] = bias->data.f32[i];
		for (j = 0; j < adim[0]; j++)
			for (k = 0; k < batch_size; k++)
				bp[i * b_batch_inc + k] += wp[j] * ap[j * a_batch_inc + k];
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(input_size >= 2 && output_size == 3);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(dw->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* bias = (ccv_nnc_tensor_view_t*)outputs[2];
	assert(bias->info.dim[1] == 0); // It is a 1-d array.
	const int* dwinc = CCV_IS_TENSOR_VIEW(dw) ? dw->inc : dw->info.dim;
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(dw->data.u8, 0, sizeof(float) * dwinc[1] * dw->info.dim[0]);
		memset(bias->data.u8, 0, sizeof(float) * bias->info.dim[0]);
	}
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == 1 || a_nd == 2);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == 1 || g_nd == 2);
	const int* gdim = (g_nd == 1) ? g->info.dim : g->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	assert(batch_size == (g_nd == 1) ? 1 : ccv_max(1, g->info.dim[0]));
	assert(bias->info.dim[0] == gdim[0]);
	int i;
	float* gp = g->data.f32;
	float* bp = bias->data.f32;
	const int g_batch_inc = CCV_IS_TENSOR_VIEW(g) ? ((g_nd == 1) ? 1 : g->inc[0]) : batch_size;
	for (i = 0; i < gdim[0]; i++)
	{
		float v = bp[i];
		int j;
		for (j = 0; j < batch_size; j++)
			v += gp[j];
		bp[i] = v;
		gp += g_batch_inc;
	}
	assert(gdim[0] == dw->info.dim[0]);
	assert(adim[0] == dw->info.dim[1]);
	const int a_batch_inc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == 1) ? 1 : a->inc[0]) : batch_size;
	gp = g->data.f32;
	const float* const ap = a->data.f32;
	parallel_for(i, gdim[0]) {
		float* const dwp = dw->data.f32 + i * dwinc[1];
		int j, k;
		for (j = 0; j < adim[0]; j++)
			for (k = 0; k < batch_size; k++)
				dwp[j] += ap[j * a_batch_inc + k] * gp[i * g_batch_inc + k];
	} parallel_endfor
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (h)
	{
		const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2];
		assert(h->info.dim[2] == 0); // It is a 2-d array.
		const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
		assert(h_nd == 1 || h_nd == 2);
		const int* hdim = (h_nd == 1) ? h->info.dim : h->info.dim + 1;
		assert(hdim[0] == adim[0]);
		assert(batch_size == (h_nd == 1) ? 1 : ccv_max(1, h->info.dim[0]));
		const int h_batch_inc = CCV_IS_TENSOR_VIEW(h) ? ((h_nd == 1) ? 1 : h->inc[0]) : batch_size;
		const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
		gp = g->data.f32;
		float* const hp = h->data.f32;
		parallel_for(i, hdim[0]) {
			const float* const wp = w->data.f32 + i;
			int j, k;
			for (k = 0; k < batch_size; k++)
				hp[i * h_batch_inc + k] = 0;
			for (j = 0; j < gdim[0]; j++)
				for (k = 0; k < batch_size; k++)
					hp[i * h_batch_inc + k] += wp[j * winc[1]] * gp[j * g_batch_inc + k];
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
}
