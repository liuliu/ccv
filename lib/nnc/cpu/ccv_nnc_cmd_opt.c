#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

#include "conv/ccv_nnc_cmd_conv_opt.h"

enum {
	CCV_NNC_CMD_OPT_CONV_ALGO_DC, // Direct convolution
	CCV_NNC_CMD_OPT_CONV_ALGO_GEMM, // GEMM (for 1x1)
	CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD, // Winograd algorithm
	CCV_NNC_CMD_OPT_CONV_ALGO_FFT, // Fast Fourier transform
	CCV_NNC_CMD_OPT_CONV_ALGO_COUNT
};

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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
	assert(b->info.dim[0] == cmd.info.convolutional.count);
	int i;
	// Make sure the weights dimension matches the network dimension
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (w->info.dim[i] == 0 || cmd.info.size.dim[i] == 0)
			break;
		assert(w->info.dim[i] == cmd.info.size.dim[i]);
	}
	// Make sure the weights output dimension matches the network convolutional kernels
	for (i = CCV_NNC_MAX_DIM_ALLOC - 1; i > 0; i--)
		if (w->info.dim[i] == 0 && w->info.dim[i])
		{
			assert(w->info.dim[i] == cmd.info.convolutional.count);
			break;
		}
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_OPT_CONV_ALGO_DC:
			return ccv_nnc_conv_forw_opt(a, w, bias, hint, b);
		case CCV_NNC_CMD_OPT_CONV_ALGO_GEMM:
			if (w->info.dim[1] == 1 && w->info.dim[1] == 1 && hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1)
				return CCV_NNC_EXEC_INVALID; // Placeholder, for gemm call.
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
	if (w->info.dim[1] == 3 && w->info.dim[2] == 3 && hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1)
		return ccv_nnc_conv_forw_4x4_3x3_winograd(a, w, bias, hint, b);
	return ccv_nnc_conv_forw_opt(a, w, bias, hint, b);
}

static int _ccv_nnc_full_connect_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_t* w = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(w));
	const ccv_nnc_tensor_t* bias = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(bias));
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	ccv_nnc_tensor_param_t a_params = inputs[0]->info;
	assert(!CCV_IS_TENSOR_VIEW(inputs[0]));
	assert(a_params.dim[2] == 0); // It is a 2-d array.
	ccv_dense_matrix_t a = ccv_dense_matrix(ccv_max(1, a_params.dim[1]), a_params.dim[0], CCV_32F | CCV_C1, inputs[0]->data.u8, 0);
	assert(output_size == 1);
	ccv_nnc_tensor_param_t b_params = outputs[0]->info;
	assert(!CCV_IS_TENSOR_VIEW(outputs[0]));
	int bias_count = ccv_nnc_tensor_count(bias->info);
	assert(b_params.dim[0] == bias_count);
	assert(b_params.dim[2] == 0); // It is a 2-d array.
	assert(ccv_max(1, b_params.dim[1]) == ccv_max(1, a_params.dim[1]));
	ccv_dense_matrix_t b = ccv_dense_matrix(ccv_max(1, b_params.dim[1]), b_params.dim[0], CCV_32F | CCV_C1, outputs[0]->data.u8, 0);
	// copy bias into each row.
	int i;
	for (i = 0; i < ccv_max(1, b_params.dim[1]); i++)
		memcpy(b.data.f32 + i * b_params.dim[0], bias->data.f32, sizeof(float) * b_params.dim[0]);
	assert(a_params.dim[0] == w->info.dim[0]);
	assert(b_params.dim[0] == w->info.dim[1]);
	ccv_dense_matrix_t dw = ccv_dense_matrix(b_params.dim[0], a_params.dim[0], CCV_32F | CCV_C1, w->data.u8, 0);
	ccv_dense_matrix_t* db = &b;
	ccv_gemm(&a, &dw, 1, db, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&db, 0); // supply b as matrix C is allowed
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_full_connect_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: weight updates, bias updates, [output gradient]
	assert((input_size == 2 && output_size == 2) || (input_size == 3 && output_size == 3));
	const ccv_nnc_tensor_param_t g_params = inputs[0]->info;
	assert(!CCV_IS_TENSOR_VIEW(inputs[0]));
	const ccv_nnc_tensor_param_t a_params = inputs[1]->info;
	assert(!CCV_IS_TENSOR_VIEW(inputs[1]));
	ccv_nnc_tensor_t* w = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(w));
	ccv_nnc_tensor_t* bias = outputs[1];
	assert(!CCV_IS_TENSOR_VIEW(bias));
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(w->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	assert(ccv_max(1, a_params.dim[1]) == ccv_max(1, g_params.dim[1]));
	assert(a_params.dim[2] == 0); // It is a 2-d array.
	assert(g_params.dim[2] == 0); // It is a 2-d array.
	ccv_dense_matrix_t g = ccv_dense_matrix(ccv_max(1, g_params.dim[1]), g_params.dim[0], CCV_32F | CCV_C1, inputs[0]->data.u8, 0);
	assert(bias->info.dim[0] == g_params.dim[0]);
	ccv_dense_matrix_t a = ccv_dense_matrix(ccv_max(1, a_params.dim[1]), a_params.dim[0], CCV_32F | CCV_C1, inputs[1]->data.u8, 0);
	int i, j;
	float* gp = inputs[0]->data.f32;
	float* bp = bias->data.f32;
	for (i = 0; i < ccv_max(1, g_params.dim[1]); i++)
	{
		for (j = 0; j < g_params.dim[0]; j++)
			bp[j] += gp[j];
		gp += g_params.dim[0];
	}
	assert(a_params.dim[0] == w->info.dim[0]);
	assert(g_params.dim[0] == w->info.dim[1]);
	ccv_dense_matrix_t dw = ccv_dense_matrix(g_params.dim[0], a_params.dim[0], CCV_32F | CCV_C1, w->data.u8, 0);
	ccv_dense_matrix_t* ddw = &dw;
	ccv_gemm(&g, &a, 1, ddw, 1, CCV_A_TRANSPOSE, (ccv_matrix_t**)&ddw, 0);
	if (output_size == 3)
	{
		const ccv_nnc_tensor_param_t h_params = outputs[2]->info;
		assert(!CCV_IS_TENSOR_VIEW(outputs[2]));
		w = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(w));
		assert(h_params.dim[0] == a_params.dim[0]);
		assert(ccv_max(1, h_params.dim[1]) == ccv_max(1, a_params.dim[1]));
		assert(h_params.dim[2] == 0); // It is a 2-d array.
		ccv_dense_matrix_t dw = ccv_dense_matrix(g_params.dim[0], h_params.dim[0], CCV_32F | CCV_C1, w->data.u8, 0);
		ccv_dense_matrix_t h = ccv_dense_matrix(ccv_max(1, h_params.dim[1]), h_params.dim[0], CCV_32F | CCV_C1, outputs[2]->data.u8, 0);
		ccv_dense_matrix_t* dh = &h;
		ccv_gemm(&g, &dw, 1, 0, 0, 0 /* No transpose */, (ccv_matrix_t**)&dh, 0);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

//@ccv_nnc_init CCV_NNC_BACKEND_CPU_OPT
void ccv_nnc_cpu_opt_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/* These are optimized kernels for the task, specifically,
	 * a set of kernels for Winograd, FFT, SIMD-optimized convolutions. */
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].algorithms = CCV_NNC_CMD_OPT_CONV_ALGO_COUNT;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_conv_forw;
	/* Full connect layer */
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].exec = _ccv_nnc_full_connect_forw;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].exec = _ccv_nnc_full_connect_back;
	/* Max pool layer */
	/* Average pool layer */
	/* Softmax layer */
	/* ReLU activation */
}
