#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// Each output is, for one query, a softmax mixture of the RMS-normalized states: the block
// states and then the partial state. A state's score is the query's dot product with it, scaled.
static int _ccv_nnc_attention_residual_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_t* const blocks = inputs[0];
	const ccv_nnc_tensor_t* const partial = inputs[1];
	const ccv_nnc_tensor_t* const queries = inputs[2];
	int i;
	for (i = 0; i < 3; i++)
		{ assert(inputs[i]->info.datatype == CCV_32F && CCV_IS_TENSOR_CONTIGUOUS(inputs[i])); }
	for (i = 0; i < output_size; i++)
		{ assert(outputs[i]->info.datatype == CCV_32F && CCV_IS_TENSOR_CONTIGUOUS(outputs[i])); }
	// Shapes only matter through their counts: C is the partial state's last axis, every block
	// state has as many values as the partial state, and every query has C.
	const int channels = partial->info.dim[ccv_nnc_tensor_nd(partial->info.dim) - 1];
	const size_t state_size = ccv_nnc_tensor_count(partial->info);
	const int rows = state_size / channels;
	const int block_count = ccv_nnc_tensor_count(blocks->info) / state_size;
	assert(ccv_nnc_tensor_count(blocks->info) == (size_t)block_count * state_size);
	assert(ccv_nnc_tensor_count(queries->info) == (size_t)output_size * channels);
	for (i = 0; i < output_size; i++)
		{ assert(ccv_nnc_tensor_count(outputs[i]->info) == state_size); }
	const float epsilon = cmd.info.attention_residual.epsilon;
	const float scale = cmd.info.attention_residual.scale;
	const int state_count = block_count + 1;
	parallel_for(t, rows) {
		// A state's reciprocal RMS, then its weight for the query at hand.
		float* const reciprocal = (float*)ccmalloc(sizeof(float) * state_count * 2);
		float* const weight = reciprocal + state_count;
		int s, c, q;
#define STATE(s) ((s) < block_count ? blocks->data.f32 + ((size_t)(s) * rows + t) * channels : partial->data.f32 + (size_t)t * channels)
		for (s = 0; s < state_count; s++)
		{
			const float* const x = STATE(s);
			// The sums of thousands of channels are in double.
			double sum = 0;
			for (c = 0; c < channels; c++)
				sum += (double)x[c] * x[c];
			reciprocal[s] = 1 / sqrt(sum / channels + epsilon);
		}
		for (q = 0; q < output_size; q++)
		{
			const float* const query = queries->data.f32 + (size_t)q * channels;
			float maximum = 0;
			for (s = 0; s < state_count; s++)
			{
				const float* const x = STATE(s);
				double dot = 0;
				for (c = 0; c < channels; c++)
					dot += (double)query[c] * x[c];
				weight[s] = dot * reciprocal[s] * scale;
				if (s == 0 || weight[s] > maximum)
					maximum = weight[s];
			}
			float sum = 0;
			for (s = 0; s < state_count; s++)
				sum += (weight[s] = expf(weight[s] - maximum));
			float* const y = outputs[q]->data.f32 + (size_t)t * channels;
			for (c = 0; c < channels; c++)
				y[c] = 0;
			for (s = 0; s < state_count; s++)
			{
				const float* const x = STATE(s);
				// The mixture is of the normalized states.
				const float w = weight[s] / sum * reciprocal[s];
				for (c = 0; c < channels; c++)
					y[c] += w * x[c];
			}
		}
#undef STATE
		ccfree(reciprocal);
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_attention_residual_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ATTENTION_RESIDUAL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_attention_residual_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ATTENTION_RESIDUAL_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_attention_residual_back;
}
