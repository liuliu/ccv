#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

#include "_ccv_nnc_gemm_cpu_opt.h"

FIND_FILE(cpu_opt/_ccv_nnc_gemm_cpu_opt.c, cpu_sys/_ccv_nnc_gemm_cpu_sys.c)

enum {
	CCV_NNC_CMD_OPT_GEMM_ALGO_DIRECT, // Direct multiplication
	CCV_NNC_CMD_OPT_GEMM_ALGO_SYSTEM, // Use system GEMM
	CCV_NNC_CMD_OPT_GEMM_ALGO_COUNT
};

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(b->info.dim[2] == 0); // It is a 2-d array.
	assert(w->info.dim[2] == 0); // It is a 2-d array
	assert(!bias || bias->info.dim[1] == 0); // It is a 1-d array
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == 1 || a_nd == 2);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == 1 || b_nd == 2);
	const int* bdim = (b_nd == 1) ? b->info.dim : b->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	assert(batch_size == (b_nd == 1) ? 1 : ccv_max(1, b->info.dim[0]));
	assert(!bias || bdim[0] == bias->info.dim[0]);
	assert(bdim[0] == w->info.dim[0]);
	assert(adim[0] == w->info.dim[1]);
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_GEMM_ALGO_DIRECT:
			return _ccv_nnc_gemm_forw_cpu_opt(a, w, bias, b);
		case CCV_NNC_CMD_OPT_GEMM_ALGO_SYSTEM:
			if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(w) && (!bias || !CCV_IS_TENSOR_VIEW(bias)) && !CCV_IS_TENSOR_VIEW(b))
				return _ccv_nnc_gemm_forw_cpu_sys(a, w, bias, b);
			return CCV_NNC_EXEC_INVALID;
		case -1:
			// Pass-through
			break;
	}
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(w) && (!bias || !CCV_IS_TENSOR_VIEW(bias)) && !CCV_IS_TENSOR_VIEW(b))
		return _ccv_nnc_gemm_forw_cpu_sys(a, w, bias, b);
#endif
	return _ccv_nnc_gemm_forw_cpu_opt(a, w, bias, b);
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert((input_size == 2 && output_size >= 2) || (input_size == 3 && output_size >= 2));
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(dw->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* bias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	assert(!bias || bias->info.dim[1] == 0); // It is a 1-d array.
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == 1 || a_nd == 2);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == 1 || g_nd == 2);
	const int* gdim = (g_nd == 1) ? g->info.dim : g->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	assert(batch_size == (g_nd == 1) ? 1 : ccv_max(1, g->info.dim[0]));
	assert(!bias || bias->info.dim[0] == gdim[0]);
	assert(gdim[0] == dw->info.dim[0]);
	assert(adim[0] == dw->info.dim[1]);
	const ccv_nnc_tensor_view_t* w = (input_size > 2) ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (h)
	{
		assert(h->info.dim[2] == 0); // It is a 2-d array.
		const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
		assert(h_nd == 1 || h_nd == 2);
		const int* hdim = (h_nd == 1) ? h->info.dim : h->info.dim + 1;
		assert(hdim[0] == adim[0]);
	}
	if (w)
	{
		assert(w->info.dim[2] == 0); // It is a 2-d array.
		assert(w->info.dim[0] == dw->info.dim[0]);
		assert(w->info.dim[1] == dw->info.dim[1]);
	}
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_GEMM_ALGO_DIRECT:
			return _ccv_nnc_gemm_back_cpu_opt(g, a, w, dw, bias, h, flags);
		case CCV_NNC_CMD_OPT_GEMM_ALGO_SYSTEM:
			if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(dw) && (!bias || !CCV_IS_TENSOR_VIEW(bias)) &&
				(!w || !CCV_IS_TENSOR_VIEW(w)) && (!h || !CCV_IS_TENSOR_VIEW(h)))
				return _ccv_nnc_gemm_back_cpu_sys(g, a, w, dw, bias, h, flags);
			return CCV_NNC_EXEC_INVALID;
		case -1:
			// Pass-through
			break;
	}
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(dw) && (!bias || !CCV_IS_TENSOR_VIEW(bias)) &&
		(!w || !CCV_IS_TENSOR_VIEW(w)) && (!h || !CCV_IS_TENSOR_VIEW(h)))
		return _ccv_nnc_gemm_back_cpu_sys(g, a, w, dw, bias, h, flags);
#endif
	return _ccv_nnc_gemm_back_cpu_opt(g, a, w, dw, bias, h, flags);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_CPU_OPT)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = CCV_NNC_CMD_OPT_GEMM_ALGO_COUNT;
	registry->exec = _ccv_nnc_gemm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_CPU_OPT)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = CCV_NNC_CMD_OPT_GEMM_ALGO_COUNT;
	registry->exec = _ccv_nnc_gemm_back;
}
