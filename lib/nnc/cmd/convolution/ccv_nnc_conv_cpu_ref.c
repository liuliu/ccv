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
	assert(bdim[CCV_NNC_MAX_DIM] == cmd.info.convolution.count);
	int i;
	// Make sure the weights dimension matches the network dimension
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (w->info.dim[i] == 0 || cmd.info.size.dim[i - 1] == 0)
			break;
		assert(w->info.dim[i] == cmd.info.size.dim[i - 1]);
	}
	const int groups = cmd.info.convolution.groups;
	assert(w->info.dim[CCV_NNC_MAX_DIM + 1] * groups == adim[CCV_NNC_MAX_DIM]);
	assert(cmd.info.convolution.count % groups == 0);
	const int group_size = cmd.info.convolution.count / groups;
	// Make sure the weights output dimension matches the network convolution kernels
	assert(w->info.dim[0] == cmd.info.convolution.count);
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ? a->inc : a->inc + 1) : adim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? ((b_nd == CCV_NNC_MAX_DIM + 1) ? b->inc : b->inc + 1) : bdim;
	assert(!bias || bias->info.dim[0] == cmd.info.convolution.count);
	const int channel_size = w->info.dim[CCV_NNC_MAX_DIM + 1];
	parallel_for(k, cmd.info.convolution.count) {
		int c;
		const int gidx = k / group_size;
		float* ap = a->data.f32;
		float* bp = b->data.f32 + k;
		// kernel weight for one dim.
		float* wp = w->data.f32 + k * w->info.dim[1] * w->info.dim[2] * channel_size;
		float biasval = bias ? bias->data.f32[k] : 0;
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		int i[CCV_NNC_MAX_DIM];
		int n[CCV_NNC_MAX_DIM];
		int m[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		for (i[0] = 0; i[0] < bdim[0]; i[0]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, w->info.dim + 1, adim, n, m);
			float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
			for (i[1] = 0; i[1] < bdim[1]; i[1]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, w->info.dim + 1, adim, n, m);
				float p = biasval;
				float* wpz = wpu + n[1] * channel_size;
				float* apz = ap + ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[CCV_NNC_MAX_DIM] + gidx * channel_size;
				for (j[0] = 0; j[0] < m[0]; j[0]++)
				{
					for (j[1] = 0; j[1] < m[1]; j[1]++)
						for (c = 0; c < channel_size; c++)
							p += wpz[j[1] * channel_size + c] * apz[j[1] * ainc[CCV_NNC_MAX_DIM] + c];
					wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
					apz += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM];
				}
				bp[i[1] * binc[CCV_NNC_MAX_DIM]] = p;
			}
			bp += binc[CCV_NNC_MAX_DIM - 1] * binc[CCV_NNC_MAX_DIM];
			ap += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
		}
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(input_size >= 2 && output_size >= 2);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_t* w = outputs[1];
	assert(!CCV_IS_TENSOR_VIEW(w));
	ccv_nnc_tensor_t* bias = output_size > 2 ? outputs[2] : 0;
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0]; // output gradients
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(w->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		if (bias)
			memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == CCV_NNC_MAX_DIM + 1 || g_nd == CCV_NNC_MAX_DIM + 2);
	const int* gdim = (g_nd == CCV_NNC_MAX_DIM + 1) ? g->info.dim : g->info.dim + 1;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? ((a_nd == CCV_NNC_MAX_DIM + 1) ? a->inc : a->inc + 1) : adim;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? ((g_nd == CCV_NNC_MAX_DIM + 1) ? g->inc : g->inc + 1) : gdim;
	const int groups = cmd.info.convolution.groups;
	assert(w->info.dim[CCV_NNC_MAX_DIM + 1] * groups == adim[CCV_NNC_MAX_DIM]);
	assert(cmd.info.convolution.count % groups == 0);
	const int group_size = cmd.info.convolution.count / groups;
	const int channel_size = w->info.dim[CCV_NNC_MAX_DIM + 1];
	parallel_for(k, cmd.info.convolution.count) {
		int c;
		const int gidx = k / group_size;
		float* ap = a->data.f32;
		float* gp = g->data.f32 + k;
		// kernel weight for one dim.
		float* wp = w->data.f32 + k * w->info.dim[1] * w->info.dim[2] * w->info.dim[3];
		float biasval = 0;
		int i[CCV_NNC_MAX_DIM];
		int n[CCV_NNC_MAX_DIM];
		int m[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		for (i[0] = 0; i[0] < gdim[0]; i[0]++)
		{
			SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, w->info.dim + 1, adim, n, m);
			float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
			for (i[1] = 0; i[1] < gdim[1]; i[1]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, w->info.dim + 1, adim, n, m);
				const float v = gp[i[1] * gdim[CCV_NNC_MAX_DIM]];
				if (v == 0) // shortcut if v is zero
					continue;
				biasval += v;
				float* wpz = wpu + n[1] * channel_size;
				float* apz = ap + ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[CCV_NNC_MAX_DIM] + gidx * channel_size;
				for (j[0] = 0; j[0] < m[0]; j[0]++)
				{
					for (j[1] = 0; j[1] < m[1]; j[1]++)
						for (c = 0; c < channel_size; c++)
							wpz[j[1] * channel_size + c] += v * apz[j[1] * ainc[CCV_NNC_MAX_DIM] + c];
					wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
					apz += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM];
				}
			}
			gp += ginc[CCV_NNC_MAX_DIM - 1] * ginc[CCV_NNC_MAX_DIM];
			ap += ainc[CCV_NNC_MAX_DIM - 1] * ainc[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
		}
		if (bias)
			bias->data.f32[k] = biasval;
	} parallel_endfor
	// If h is available, therefore, we need to propagate the gradients back
	if (h)
	{
		assert(h);
		const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
		assert(h_nd == CCV_NNC_MAX_DIM + 1 || h_nd == CCV_NNC_MAX_DIM + 2);
		const int* hdim = (h_nd == CCV_NNC_MAX_DIM + 1) ? h->info.dim : h->info.dim + 1;
		const int* hinc = CCV_IS_TENSOR_VIEW(h) ? ((h_nd == CCV_NNC_MAX_DIM + 1) ? h->inc : h->inc + 1) : hdim;
		// reset it to 0.
		ccv_nnc_tensor_zero(h);
		w = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(w));
		int k, gidx;
		for (gidx = 0; gidx < groups; gidx++)
			for (k = gidx * group_size; k < (gidx + 1) * group_size; k++)
			{
				int c;
				float* hp = h->data.f32;
				float* gp = g->data.f32 + k;
				// kernel weight for one dim.
				float* wp = w->data.f32 + k * w->info.dim[1] * w->info.dim[2] * channel_size;
				// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
				int i[CCV_NNC_MAX_DIM];
				int n[CCV_NNC_MAX_DIM];
				int m[CCV_NNC_MAX_DIM];
				int j[CCV_NNC_MAX_DIM];
				for (i[0] = 0; i[0] < gdim[0]; i[0]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, w->info.dim + 1, h->info.dim, n, m);
					float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
					for (i[1] = 0; i[1] < gdim[1]; i[1]++)
					{
						SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, w->info.dim + 1, h->info.dim, n, m);
						const float v = gp[i[1] * ginc[CCV_NNC_MAX_DIM]];
						if (v == 0) // shortcut if v is zero
							continue;
						float* wpz = wpu + n[1] * channel_size;
						float* hpz = hp + ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) * hinc[CCV_NNC_MAX_DIM] + gidx * channel_size;
						for (j[0] = 0; j[0] < m[0]; j[0]++)
						{
							for (j[1] = 0; j[1] < m[1]; j[1]++)
								for (c = 0; c < channel_size; c++)
									hpz[j[1] * hinc[CCV_NNC_MAX_DIM] + c] += v * wpz[j[1] * channel_size + c];
							wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
							hpz += hinc[CCV_NNC_MAX_DIM - 1] * hinc[CCV_NNC_MAX_DIM];
						}
					}
					gp += ginc[CCV_NNC_MAX_DIM - 1] * ginc[CCV_NNC_MAX_DIM];
					hp += hinc[CCV_NNC_MAX_DIM - 1] * hinc[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
				}
			}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_back;
}

