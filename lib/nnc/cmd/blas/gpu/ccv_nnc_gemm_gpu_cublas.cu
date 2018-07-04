extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
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
	const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
	const int a_batch_inc = CCV_IS_TENSOR_VIEW(a) ? (a_nd == 1 ? a->inc[0] : a->inc[1]) : adim[0];
	const int b_batch_inc = CCV_IS_TENSOR_VIEW(b) ? (b_nd == 1 ? b->inc[0] : b->inc[1]) : bdim[0];

	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	static const float one = 1;
	static const float zero = 0;
	const float* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, batch_size);
	if (bias)
	{
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bdim[0], batch_size, 1, &one, bias->data.f32, bdim[0], device_ones, 1, &zero, b->data.f32, b_batch_inc);
		cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, bdim[0], batch_size, adim[0], &one, w->data.f32, winc[1], a->data.f32, a_batch_inc, &one, b->data.f32, b_batch_inc);
	} else
		cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, bdim[0], batch_size, adim[0], &one, w->data.f32, winc[1], a->data.f32, a_batch_inc, &zero, b->data.f32, b_batch_inc);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(input_size >= 2 && output_size >= 2);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(dw->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* bias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	assert(!bias || bias->info.dim[1] == 0); // It is a 1-d array.
	const int* dwinc = CCV_IS_TENSOR_VIEW(dw) ? dw->inc : dw->info.dim;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == 1 || a_nd == 2);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == 1 || g_nd == 2);
	const int* gdim = (g_nd == 1) ? g->info.dim : g->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	assert(batch_size == (g_nd == 1) ? 1 : ccv_max(1, g->info.dim[0]));
	assert(!bias || bias->info.dim[0] == gdim[0]);
	static const float one = 1;
	static const float zero = 0;
	const int g_batch_inc = CCV_IS_TENSOR_VIEW(g) ? ((g_nd == 1) ? g->inc[0] : g->inc[1]) : gdim[0];

	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const float * const device_ones = ccv_nnc_stream_context_get_ones(stream_context, batch_size);
	if (bias)
	{
		if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
			cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, gdim[0], 1, batch_size, &one, g->data.f32, g_batch_inc, device_ones, batch_size, &zero, bias->data.f32, gdim[0]);
		else
			cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, gdim[0], 1, batch_size, &one, g->data.f32, g_batch_inc, device_ones, batch_size, &one, bias->data.f32, gdim[0]);
	}
	assert(gdim[0] == dw->info.dim[0]);
	assert(adim[0] == dw->info.dim[1]);
	const int a_batch_inc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == 1) ? a->inc[0] : a->inc[1]) : adim[0];
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, adim[0], gdim[0], batch_size, &one, a->data.f32, a_batch_inc, g->data.f32, g_batch_inc, &zero, dw->data.f32, dwinc[1]);
	else
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, adim[0], gdim[0], batch_size, &one, a->data.f32, a_batch_inc, g->data.f32, g_batch_inc, &one, dw->data.f32, dwinc[1]);
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
		const int h_batch_inc = CCV_IS_TENSOR_VIEW(h) ? ((h_nd == 1) ? h->inc[0] : h->inc[1]) : hdim[0];
		const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, adim[0], batch_size, gdim[0], &one, w->data.f32, winc[1], g->data.f32, g_batch_inc, &zero, h->data.f32, h_batch_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
#endif
}
