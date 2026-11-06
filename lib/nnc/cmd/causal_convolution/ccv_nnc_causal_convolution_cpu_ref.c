#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

static int _ccv_nnc_causal_convolution_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 2);
	const ccv_nnc_tensor_t* const x = inputs[0];
	const ccv_nnc_tensor_t* const previous = inputs[1];
	const ccv_nnc_tensor_t* const weight = inputs[2];
	ccv_nnc_tensor_t* const y = outputs[0];
	ccv_nnc_tensor_t* const next = outputs[1];
	int i;
	for (i = 0; i < 3; i++)
		assert(inputs[i]->info.datatype == CCV_32F && CCV_IS_TENSOR_CONTIGUOUS(inputs[i]));
	for (i = 0; i < 2; i++)
		assert(outputs[i]->info.datatype == CCV_32F && CCV_IS_TENSOR_CONTIGUOUS(outputs[i]));
	// Shapes only matter through their counts: C is x's last axis and the previous rows hold B batches.
	const int channels = x->info.dim[ccv_nnc_tensor_nd(x->info.dim) - 1];
	const int kernel_size = ccv_nnc_tensor_count(weight->info) / channels;
	// One tap has no previous rows to carry.
	assert(kernel_size >= 2 && ccv_nnc_tensor_count(weight->info) == (size_t)kernel_size * channels);
	const int B = ccv_nnc_tensor_count(previous->info) / ((size_t)(kernel_size - 1) * channels);
	const int T = ccv_nnc_tensor_count(x->info) / ((size_t)B * channels);
	const int rows = ccv_nnc_tensor_count(next->info) / ((size_t)B * channels);
	assert(ccv_nnc_tensor_count(y->info) == (size_t)B * T * channels);
	assert(rows <= kernel_size - 1 + T);
	const int history = kernel_size - 1;
	// Context row r of batch b is previous row r, then x row r - history.
#define CONTEXT(b, r, c) ((r) < history ? previous->data.f32[((size_t)(b) * history + (r)) * channels + (c)] : x->data.f32[((size_t)(b) * T + (r) - history) * channels + (c)])
	const int silu = cmd.info.causal_convolution.silu;
	int b, t, c, j;
	for (b = 0; b < B; b++)
		for (c = 0; c < channels; c++)
		{
			const float* const w = weight->data.f32 + (size_t)c * kernel_size;
			for (t = 0; t < T; t++)
			{
				float sum = 0;
				for (j = 0; j < kernel_size; j++)
					sum += w[j] * CONTEXT(b, t + j, c);
				y->data.f32[((size_t)b * T + t) * channels + c] = silu ? sum / (1 + expf(-sum)) : sum;
			}
			// The next rows are the context's last rows.
			const int first = history + T - rows;
			for (j = 0; j < rows; j++)
				next->data.f32[((size_t)b * rows + j) * channels + c] = CONTEXT(b, first + j, c);
		}
#undef CONTEXT
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_causal_convolution_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_causal_convolution_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CAUSAL_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_causal_convolution_back;
}
