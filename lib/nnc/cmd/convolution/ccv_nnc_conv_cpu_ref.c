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
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	assert(bias->info.dim[0] == cmd.info.convolution.count);
	parallel_for(k, cmd.info.convolution.count) {
		int c;
		float* ap = a->data.f32;
		float* bp = b->data.f32 + k;
		// kernel weight for one dim.
		float* wp = w->data.f32 + k * w->info.dim[0] * w->info.dim[1] * w->info.dim[2];
		float biasval = bias->data.f32[k];
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		int i[CCV_NNC_MAX_DIM];
		int n[CCV_NNC_MAX_DIM];
		int m[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, w->info.dim, a->info.dim, n, m);
			float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
			for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, w->info.dim, a->info.dim, n, m);
				float p = biasval;
				float* wpz = wpu + n[0] * w->info.dim[0];
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						for (c = 0; c < a->info.dim[0]; c++)
							p += wpz[j[0] * w->info.dim[0] + c] * apz[j[0] * ainc[0] + c];
					wpz += w->info.dim[1] * w->info.dim[0];
					apz += ainc[1] * ainc[0];
				}
				bp[i[0] * binc[0]] = p;
			}
			bp += binc[1] * binc[0];
			ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		}
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert((input_size == 2 && output_size == 3) || (input_size == 3 && output_size == 3));
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_t* w = outputs[1];
	assert(!CCV_IS_TENSOR_VIEW(w));
	ccv_nnc_tensor_t* bias = outputs[2];
	assert(!CCV_IS_TENSOR_VIEW(bias));
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0]; // output gradients
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(w->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	parallel_for(k, cmd.info.convolution.count) {
		int c;
		float* ap = a->data.f32;
		float* gp = g->data.f32 + k;
		// kernel weight for one dim.
		float* wp = w->data.f32 + k * w->info.dim[0] * w->info.dim[1] * w->info.dim[2];
		float biasval = 0;
		int i[CCV_NNC_MAX_DIM];
		int n[CCV_NNC_MAX_DIM];
		int m[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		for (i[1] = 0; i[1] < g->info.dim[2]; i[1]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, w->info.dim, a->info.dim, n, m);
			float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
			for (i[0] = 0; i[0] < g->info.dim[1]; i[0]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, w->info.dim, a->info.dim, n, m);
				const float v = gp[i[0] * g->info.dim[0]];
				if (v == 0) // shortcut if v is zero
					continue;
				biasval += v;
				float* wpz = wpu + n[0] * w->info.dim[0];
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						for (c = 0; c < a->info.dim[0]; c++)
							wpz[j[0] * w->info.dim[0] + c] += v * apz[j[0] * ainc[0] + c];
					wpz += w->info.dim[1] * w->info.dim[0];
					apz += ainc[1] * ainc[0];
				}
			}
			gp += ginc[1] * ginc[0];
			ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		}
		bias->data.f32[k] = biasval;
	} parallel_endfor
	// If h is available, therefore, we need to propagate the gradients back
	if (h)
	{
		assert(h);
		const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
		// reset it to 0.
		ccv_nnc_tensor_zero(h);
		w = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(w));
		int k;
		for (k = 0; k < cmd.info.convolution.count; k++)
		{
			int c;
			float* hp = h->data.f32;
			float* gp = g->data.f32 + k;
			// kernel weight for one dim.
			float* wp = w->data.f32 + k * w->info.dim[0] * w->info.dim[1] * w->info.dim[2];
			// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
			int i[CCV_NNC_MAX_DIM];
			int n[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int j[CCV_NNC_MAX_DIM];
			for (i[1] = 0; i[1] < g->info.dim[2]; i[1]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, w->info.dim, h->info.dim, n, m);
				float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
				for (i[0] = 0; i[0] < g->info.dim[1]; i[0]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, w->info.dim, h->info.dim, n, m);
					const float v = gp[i[0] * ginc[0]];
					if (v == 0) // shortcut if v is zero
						continue;
					float* wpz = wpu + n[0] * w->info.dim[0];
					float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * hinc[0];
					for (j[1] = 0; j[1] < m[1]; j[1]++)
					{
						for (j[0] = 0; j[0] < m[0]; j[0]++)
							for (c = 0; c < h->info.dim[0]; c++)
								hpz[j[0] * hinc[0] + c] += v * wpz[j[0] * w->info.dim[0] + c];
						wpz += w->info.dim[1] * w->info.dim[0];
						hpz += hinc[1] * hinc[0];
					}
				}
				gp += ginc[1] * ginc[0];
				hp += hinc[1] * hinc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_back;
}

