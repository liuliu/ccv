#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

#include "_ccv_nnc_conv_cpu_opt.h"

FIND_FILE(cpu_opt/_ccv_nnc_conv_cpu_4x4_3x3_winograd.c, cpu_opt/_ccv_nnc_conv_cpu_fft.c, cpu_opt/_ccv_nnc_conv_cpu_gemm.c, cpu_opt/_ccv_nnc_conv_cpu_opt.c)

enum {
	CCV_NNC_CMD_OPT_CONV_ALGO_DC, // Direct convolution
	CCV_NNC_CMD_OPT_CONV_ALGO_GEMM, // GEMM (for 1x1)
	CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD, // Winograd algorithm
	CCV_NNC_CMD_OPT_CONV_ALGO_FFT, // Fast Fourier transform
	CCV_NNC_CMD_OPT_CONV_ALGO_COUNT
};

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_t* w = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(w));
	const ccv_nnc_tensor_t* bias = input_size > 2 ? inputs[2] : 0;
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	assert(w->info.dim[CCV_NNC_MAX_DIM + 1] == adim[CCV_NNC_MAX_DIM]);
	assert(bdim[CCV_NNC_MAX_DIM] == cmd.info.convolution.count);
	if (cmd.info.convolution.groups != 1)
		return CCV_NNC_EXEC_INVALID;
	int i;
	// Make sure the weights dimension matches the network dimension
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (w->info.dim[i] == 0 || cmd.info.size.dim[i - 1] == 0)
			break;
		assert(w->info.dim[i] == cmd.info.size.dim[i - 1]);
	}
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_CONV_ALGO_DC:
			return _ccv_nnc_conv_forw_cpu_opt(a, w, bias, hint, b);
		case CCV_NNC_CMD_OPT_CONV_ALGO_GEMM:
			if (w->info.dim[1] == 1 && w->info.dim[2] == 1 && hint.stride.dim[0] <= 1 && hint.stride.dim[1] <= 1 &&
				hint.border.begin[0] == 0 && hint.border.begin[1] == 0 && hint.border.end[0] == 0 && hint.border.end[1] == 0 &&
				!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(w) && (!bias || !CCV_IS_TENSOR_VIEW(bias)))
				return _ccv_nnc_conv_forw_gemm_cpu_opt(a, w, bias, hint, b);
			return CCV_NNC_EXEC_INVALID;
		case CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD:
			if (w->info.dim[1] == 3 && w->info.dim[2] == 3 && hint.stride.dim[0] <= 1 && hint.stride.dim[1] <= 1)
				return _ccv_nnc_conv_forw_4x4_3x3_winograd_cpu_opt(a, w, bias, hint, b);
			return CCV_NNC_EXEC_INVALID;
		case CCV_NNC_CMD_OPT_CONV_ALGO_FFT:
			return CCV_NNC_EXEC_INVALID; // Placeholder, for fft.
		case -1:
			// Pass-through
			break;
	}
	// If the size is 3x3, and no stride, choose Winograd kernel
	if (w->info.dim[1] == 3 && w->info.dim[2] == 3 && hint.stride.dim[0] <= 1 && hint.stride.dim[1] <= 1)
		return _ccv_nnc_conv_forw_4x4_3x3_winograd_cpu_opt(a, w, bias, hint, b);
	// If the size is 1x1, and no stride, and not a tensor view object, no padding, choose GEMM kernel
	if (w->info.dim[1] == 1 && w->info.dim[2] == 1 && hint.stride.dim[0] <= 1 && hint.stride.dim[1] <= 1 &&
		hint.border.begin[0] == 0 && hint.border.begin[1] == 0 && hint.border.end[0] == 0 && hint.border.end[1] == 0 &&
		!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(w) && (!bias || !CCV_IS_TENSOR_VIEW(bias)))
		return _ccv_nnc_conv_forw_gemm_cpu_opt(a, w, bias, hint, b);
	// Otherwise, use direct convolution kernel
	return _ccv_nnc_conv_forw_cpu_opt(a, w, bias, hint, b);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_CPU_OPT)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = CCV_NNC_CMD_OPT_CONV_ALGO_COUNT;
	registry->exec = _ccv_nnc_conv_forw;
}
