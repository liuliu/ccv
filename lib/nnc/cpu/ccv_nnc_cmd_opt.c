#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

#include "conv/ccv_nnc_cmd_conv_opt.h"
#include "fc/ccv_nnc_cmd_fc_opt.h"

enum {
	CCV_NNC_CMD_OPT_CONV_ALGO_DC, // Direct convolution
	CCV_NNC_CMD_OPT_CONV_ALGO_GEMM, // GEMM (for 1x1)
	CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD, // Winograd algorithm
	CCV_NNC_CMD_OPT_CONV_ALGO_FFT, // Fast Fourier transform
	CCV_NNC_CMD_OPT_CONV_ALGO_COUNT
};

enum {
	CCV_NNC_CMD_OPT_FC_ALGO_DIRECT, // Direct multiplication
	CCV_NNC_CMD_OPT_FC_ALGO_GEMM, // Use system GEMM
	CCV_NNC_CMD_OPT_FC_ALGO_COUNT
};

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_t* w = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(w));
	const ccv_nnc_tensor_t* bias = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(bias));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(w->info.dim[0] == cmd.info.size.dim[0]);
	assert(w->info.dim[0] == a->info.dim[0]);
	assert(b->info.dim[0] == cmd.info.convolution.count);
	int i;
	// Make sure the weights dimension matches the network dimension
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (w->info.dim[i] == 0 || cmd.info.size.dim[i] == 0)
			break;
		assert(w->info.dim[i] == cmd.info.size.dim[i]);
	}
	// Make sure the weights output dimension matches the network convolution kernels
	for (i = CCV_NNC_MAX_DIM_ALLOC - 1; i > 0; i--)
		if (w->info.dim[i] == 0 && w->info.dim[i])
		{
			assert(w->info.dim[i] == cmd.info.convolution.count);
			break;
		}
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_CONV_ALGO_DC:
			return ccv_nnc_conv_forw_opt(a, w, bias, hint, b);
		case CCV_NNC_CMD_OPT_CONV_ALGO_GEMM:
			if (w->info.dim[1] == 1 && w->info.dim[1] == 1 && hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1 &&
				hint.border.begin[1] == 0 && hint.border.begin[2] == 0 && hint.border.end[1] == 0 && hint.border.end[2] == 0 &&
				!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(w) && !CCV_IS_TENSOR_VIEW(bias))
				return ccv_nnc_conv_forw_gemm(a, w, bias, hint, b);
			return CCV_NNC_EXEC_INVALID;
		case CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD:
			if (w->info.dim[1] == 3 && w->info.dim[2] == 3 && hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1)
				return ccv_nnc_conv_forw_4x4_3x3_winograd(a, w, bias, hint, b);
			return CCV_NNC_EXEC_INVALID;
		case CCV_NNC_CMD_OPT_CONV_ALGO_FFT:
			return CCV_NNC_EXEC_INVALID; // Placeholder, for fft.
		case -1:
			// Pass-through
			break;
	}
	// If the size is 3x3, and no stride, choose Winograd kernel
	if (w->info.dim[1] == 3 && w->info.dim[2] == 3 && hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1)
		return ccv_nnc_conv_forw_4x4_3x3_winograd(a, w, bias, hint, b);
	// If the size is 1x1, and no stride, and not a tensor view object, no padding, choose GEMM kernel
	if (w->info.dim[1] == 1 && w->info.dim[1] == 1 && hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1 &&
		hint.border.begin[1] == 0 && hint.border.begin[2] == 0 && hint.border.end[1] == 0 && hint.border.end[2] == 0 &&
		!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(w) && !CCV_IS_TENSOR_VIEW(bias))
		return ccv_nnc_conv_forw_gemm(a, w, bias, hint, b);
	// Otherwise, use direct convolution kernel
	return ccv_nnc_conv_forw_opt(a, w, bias, hint, b);
}

static int _ccv_nnc_full_connect_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = (const ccv_nnc_tensor_view_t*)inputs[2];
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(b->info.dim[0] == bias->info.dim[0]);
	assert(bias->info.dim[1] == 0); // It is a 1-d array
	assert(b->info.dim[2] == 0); // It is a 2-d array.
	assert(ccv_max(1, b->info.dim[1]) == ccv_max(1, a->info.dim[1]));
	assert(a->info.dim[0] == w->info.dim[0]);
	assert(b->info.dim[0] == w->info.dim[1]);
	assert(w->info.dim[2] == 0); // It is a 2-d array
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_FC_ALGO_DIRECT:
			return ccv_nnc_fc_forw_opt(a, w, bias, b);
		case CCV_NNC_CMD_OPT_FC_ALGO_GEMM:
			if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(w) && !CCV_IS_TENSOR_VIEW(bias) && !CCV_IS_TENSOR_VIEW(b))
				return ccv_nnc_fc_forw_gemm(a, w, bias, b);
			return CCV_NNC_EXEC_INVALID;
		case -1:
			// Pass-through
			break;
	}
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(w) && !CCV_IS_TENSOR_VIEW(bias) && !CCV_IS_TENSOR_VIEW(b))
		return ccv_nnc_fc_forw_gemm(a, w, bias, b);
#endif
	return ccv_nnc_fc_forw_opt(a, w, bias, b);
}

static int _ccv_nnc_full_connect_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: weight updates, bias updates, [output gradient]
	assert((input_size == 2 && output_size == 2) || (input_size == 3 && output_size == 3));
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(dw->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* bias = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(bias->info.dim[1] == 0); // It is a 1-d array.
	assert(ccv_max(1, a->info.dim[1]) == ccv_max(1, g->info.dim[1]));
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	assert(bias->info.dim[0] == g->info.dim[0]);
	assert(a->info.dim[0] == dw->info.dim[0]);
	assert(g->info.dim[0] == dw->info.dim[1]);
	const ccv_nnc_tensor_view_t* w = (input_size == 3) ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* h = (output_size == 3) ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	if (output_size == 3)
	{
		assert(h->info.dim[0] == a->info.dim[0]);
		assert(ccv_max(1, h->info.dim[1]) == ccv_max(1, a->info.dim[1]));
		assert(h->info.dim[2] == 0); // It is a 2-d array.
	}
	if (input_size == 3)
	{
		assert(w->info.dim[2] == 0); // It is a 2-d array.
		assert(w->info.dim[0] == dw->info.dim[0]);
		assert(w->info.dim[1] == dw->info.dim[1]);
	}
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_FC_ALGO_DIRECT:
			return ccv_nnc_fc_back_opt(g, a, w, dw, bias, h, flags);
		case CCV_NNC_CMD_OPT_FC_ALGO_GEMM:
			if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(dw) && !CCV_IS_TENSOR_VIEW(bias) &&
				(input_size == 2 || !CCV_IS_TENSOR_VIEW(w)) && (output_size == 2 || !CCV_IS_TENSOR_VIEW(h)))
				return ccv_nnc_fc_back_gemm(g, a, w, dw, bias, h, flags);
			return CCV_NNC_EXEC_INVALID;
		case -1:
			// Pass-through
			break;
	}
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(dw) && !CCV_IS_TENSOR_VIEW(bias) &&
		(input_size == 2 || !CCV_IS_TENSOR_VIEW(w)) && (output_size == 2 || !CCV_IS_TENSOR_VIEW(h)))
		return ccv_nnc_fc_back_gemm(g, a, w, dw, bias, h, flags);
#endif
	return ccv_nnc_fc_back_opt(g, a, w, dw, bias, h, flags);
}

//@ccv_nnc_init CCV_NNC_BACKEND_CPU_OPT
void ccv_nnc_cpu_opt_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/* These are optimized kernels for the task, specifically,
	 * a set of kernels for Winograd, FFT, SIMD-optimized convolutions. */
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].algorithms = CCV_NNC_CMD_OPT_CONV_ALGO_COUNT;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].exec = _ccv_nnc_conv_forw;
	/* Full connect layer */
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].algorithms = CCV_NNC_CMD_OPT_FC_ALGO_COUNT;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].exec = _ccv_nnc_full_connect_forw;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].algorithms = CCV_NNC_CMD_OPT_FC_ALGO_COUNT;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].exec = _ccv_nnc_full_connect_back;
	/* Max pool layer */
	/* Average pool layer */
	/* Softmax layer */
	/* ReLU activation */
}
