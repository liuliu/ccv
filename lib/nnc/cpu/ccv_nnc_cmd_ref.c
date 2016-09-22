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

static void _ccv_nnc_tensor_transfer(const ccv_nnc_tensor_view_t* a, ccv_nnc_tensor_view_t* b)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int x;
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
		binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do memcpy.
		memcpy(b->data.f32, a->data.f32, ccv_nnc_tensor_count(a->info) * sizeof(float));
		return;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	if (ainc[0] == dim[0] && binc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0] (do memcpy for the last two dim)
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				memcpy(bp, ap, dim[1] * dim[0] * sizeof(float));
				ap += ainc[1] * ainc[0];
				bp += binc[1] * binc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
		}
		return;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				memcpy(bp, ap, dim[0] * sizeof(float));
				ap += ainc[0];
				bp += binc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
			bp += (binc[1] - dim[1]) * binc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		bp += (binc[2] - dim[2]) * binc[1] * binc[0];
	}
}

static int _ccv_nnc_data_transfer(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(output_size == input_size);
	int i;
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		if (a != b) // Only do transfer if these are two different tensors.
			_ccv_nnc_tensor_transfer(a, b);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_tensor_nhwc_nchw(const ccv_nnc_tensor_view_t* a, ccv_nnc_tensor_view_t* b)
{
	// Assuming this is float 32.
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int k;
	// In case it is Toll-free bridged matrix object (NHWC format is possible).
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0 || a->info.dim[CCV_NNC_MAX_DIM + 1] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	for (k = 0; k < CCV_NNC_MAX_DIM + 2; k++)
	{
		ainc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[k] : a->info.dim[k]);
		binc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[k] : b->info.dim[k]);
	}
	// Comparing N
	assert(ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]) == ccv_max(1, b->info.dim[CCV_NNC_MAX_DIM + 1]));
	const int n = ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]);
	// Comparing C
	assert(a->info.dim[0] == b->info.dim[CCV_NNC_MAX_DIM]);
	const int c = a->info.dim[0];
	// Comparing HW
	int hw[CCV_NNC_MAX_DIM];
	for (k = 0; k < CCV_NNC_MAX_DIM; k++)
	{
		assert(a->info.dim[k + 1] == b->info.dim[k]);
		hw[k] = a->info.dim[k + 1];
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < n; i[3]++)
	{
		for (i[0] = 0; i[0] < c; i[0]++)
		{
			float* apu = ap + i[0];
			for (i[2] = 0; i[2] < hw[1]; i[2]++)
			{
				for (i[1] = 0; i[1] < hw[0]; i[1]++)
					bp[i[1]] = apu[i[1] * ainc[0]];
				apu += ainc[0] * ainc[1];
				bp += binc[0];
			}
			bp += (binc[1] - hw[1]) * binc[0];
		}
		ap += ainc[2] * ainc[1] * ainc[0];
		bp += (binc[2] - c) * binc[1] * binc[0];
	}
}

static void _ccv_nnc_tensor_nchw_nhwc(const ccv_nnc_tensor_view_t* a, ccv_nnc_tensor_view_t* b)
{
	// Assuming this is float 32.
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int k;
	// In case it is Toll-free bridged matrix object (NHWC format is possible).
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0 || b->info.dim[CCV_NNC_MAX_DIM + 1] == 0);
	for (k = 0; k < CCV_NNC_MAX_DIM + 2; k++)
	{
		ainc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[k] : a->info.dim[k]);
		binc[k] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[k] : b->info.dim[k]);
	}
	// Comparing N
	assert(ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]) == ccv_max(1, b->info.dim[CCV_NNC_MAX_DIM + 1]));
	const int n = ccv_max(1, a->info.dim[CCV_NNC_MAX_DIM + 1]);
	// Comparing C
	assert(a->info.dim[CCV_NNC_MAX_DIM] == b->info.dim[0]);
	const int c = a->info.dim[CCV_NNC_MAX_DIM];
	// Comparing HW
	int hw[CCV_NNC_MAX_DIM];
	for (k = 0; k < CCV_NNC_MAX_DIM; k++)
	{
		assert(a->info.dim[k] == b->info.dim[k + 1]);
		hw[k] = a->info.dim[k];
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < n; i[3]++)
	{
		for (i[0] = 0; i[0] < c; i[0]++)
		{
			float* bpu = bp + i[0];
			for (i[2] = 0; i[2] < hw[1]; i[2]++)
			{
				for (i[1] = 0; i[1] < hw[0]; i[1]++)
					bpu[i[1] * binc[0]] = ap[i[1]];
				ap += ainc[0];
				bpu += binc[0] * binc[1];
			}
			ap += (ainc[1] - hw[1]) * ainc[0];
		}
		ap += (ainc[2] - c) * ainc[1] * ainc[0];
		bp += binc[2] * binc[1] * binc[0];
	}
}

static int _ccv_nnc_format_transform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(output_size == input_size);
	int i;
	for (i = 0; i < input_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		assert(a != b); // Cannot do inplace transform.
		if (a->info.format == b->info.format) {
			// If it is the same, just do a normal data transfer.
			_ccv_nnc_tensor_transfer(a, b);
		} else if (a->info.format == CCV_TENSOR_FORMAT_NHWC && b->info.format == CCV_TENSOR_FORMAT_NCHW) {
			_ccv_nnc_tensor_nhwc_nchw(a, b);
		} else if (a->info.format == CCV_TENSOR_FORMAT_NHWC && b->info.format == CCV_TENSOR_FORMAT_CHWN) {
		} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW && b->info.format == CCV_TENSOR_FORMAT_NHWC) {
			_ccv_nnc_tensor_nchw_nhwc(a, b);
		} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW && b->info.format == CCV_TENSOR_FORMAT_CHWN) {
		} else if (a->info.format == CCV_TENSOR_FORMAT_CHWN && b->info.format == CCV_TENSOR_FORMAT_NHWC) {
		} else if (a->info.format == CCV_TENSOR_FORMAT_CHWN && b->info.format == CCV_TENSOR_FORMAT_NCHW) {
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define set_n_m_dim(x, wd, ad) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1], 0) - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1]); \
		m[x] = wd[x + 1] - n[x] - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1] + wd[x + 1] - ccv_min(ad[x + 1], i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1] + wd[x + 1])); \
	} while (0)

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
			set_n_m_dim(1, w->info.dim, a->info.dim);
			float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
			for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
			{
				set_n_m_dim(0, w->info.dim, a->info.dim);
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
			set_n_m_dim(1, w->info.dim, a->info.dim);
			float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
			for (i[0] = 0; i[0] < g->info.dim[1]; i[0]++)
			{
				set_n_m_dim(0, w->info.dim, a->info.dim);
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
				set_n_m_dim(1, w->info.dim, h->info.dim);
				float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
				for (i[0] = 0; i[0] < g->info.dim[1]; i[0]++)
				{
					set_n_m_dim(0, w->info.dim, h->info.dim);
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

static int _ccv_nnc_max_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	float* bp = b->data.f32;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, a->info.dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, a->info.dim);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0] + c;
				float v = apz[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						if (apz[j[0] * ainc[0]] > v)
							v = apz[j[0] * ainc[0]];
					apz += ainc[1] * ainc[0];
				}
				bp[i[0] * binc[0] + c] = v;
			}
		}
		bp += binc[1] * binc[0];
		ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_max_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	float* bp = b->data.f32;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	float* gp = g->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	float* hp = h->data.f32;
	const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
	for (c = 0; c < CCV_NNC_MAX_DIM_ALLOC; c++)
	{
		assert(a->info.dim[c] == h->info.dim[c]);
		if (a->info.dim[c] == 0 || h->info.dim[c] == 0)
			break;
	}
	for (c = 0; c < CCV_NNC_MAX_DIM_ALLOC; c++)
	{
		assert(b->info.dim[c] == g->info.dim[c]);
		if (b->info.dim[c] == 0 || g->info.dim[c] == 0)
			break;
	}
	ccv_nnc_tensor_zero(h);
	// Using b->info.dim and a->info.dim directly because they equal to g->info.dim and h->info.dim
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, a->info.dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, a->info.dim);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0] + c;
				float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * hinc[0] + c;
				float v = bp[i[0] * binc[0] + c];
				float u = gp[i[0] * ginc[0] + c];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						if (apz[j[0] * ainc[0]] == v)
							hpz[j[0] * hinc[0]] += u;
					apz += ainc[1] * ainc[0];
					hpz += hinc[1] * hinc[0];
				}
			}
		}
		gp += ginc[1] * ginc[0];
		bp += binc[1] * binc[0];
		ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		hp += hinc[1] * hinc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_avg_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	float* bp = b->data.f32;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, a->info.dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, a->info.dim);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * ainc[0] + c;
				float v = 0;
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						v += apz[j[0] * ainc[0]];
					apz += ainc[1] * ainc[0];
				}
				bp[i[0] * binc[0] + c] = v / (m[0] * m[1]);
			}
		}
		bp += binc[1] * binc[0];
		ap += ainc[1] * ainc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_avg_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* gp = g->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	float* hp = h->data.f32;
	const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
	ccv_nnc_tensor_zero(h);
	for (i[1] = 0; i[1] < g->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, h->info.dim);
		for (i[0] = 0; i[0] < g->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, h->info.dim);
			for (c = 0; c < g->info.dim[0]; c++)
			{
				float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * hinc[0] + c;
				float u = gp[i[0] * ginc[0] + c] / (m[0] * m[1]);
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						hpz[j[0] * hinc[0]] += u;
					hpz += hinc[1] * hinc[0];
				}
			}
		}
		gp += ginc[1] * ginc[0];
		hp += hinc[1] * hinc[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = (const ccv_nnc_tensor_view_t*)inputs[2];
	// Copy the most of parameters, but reshape the dimension of a to a vector.
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
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
	int batch_size = ccv_max(1, b->info.dim[1]);
	int i;
	for (i = 0; i < batch_size; i++)
	{
		const float* const ap = a->data.f32 + i * ainc[0];
		float* const bp = b->data.f32 + i * binc[0];
		parallel_for(j, b->info.dim[0]) {
			float v = bias->data.f32[j];
			const float* const wp = w->data.f32 + j * winc[0];
			int k;
			for (k = 0; k < a->info.dim[0]; k++)
				v += wp[k] * ap[k];
			bp[j] = v;
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert((input_size == 2 && output_size == 3) || (input_size == 3 && output_size == 3));
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(dw->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* bias = (ccv_nnc_tensor_view_t*)outputs[2];
	assert(bias->info.dim[1] == 0); // It is a 1-d array.
	const int* dwinc = CCV_IS_TENSOR_VIEW(dw) ? dw->inc : dw->info.dim;
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(dw->data.u8, 0, sizeof(float) * dwinc[0] * dw->info.dim[1]);
		memset(bias->data.u8, 0, sizeof(float) * bias->info.dim[0]);
	}
	assert(ccv_max(1, a->info.dim[1]) == ccv_max(1, g->info.dim[1]));
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	assert(bias->info.dim[0] == g->info.dim[0]);
	int batch_size = ccv_max(1, g->info.dim[1]);
	int i, j;
	float* gp = g->data.f32;
	float* bp = bias->data.f32;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < g->info.dim[0]; j++)
			bp[j] += gp[j];
		gp += ginc[0];
	}
	assert(a->info.dim[0] == dw->info.dim[0]);
	assert(g->info.dim[0] == dw->info.dim[1]);
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		const float* const gp = g->data.f32 + i * ginc[0];
		const float* const ap = a->data.f32 + i * ainc[0];
		parallel_for(j, g->info.dim[0]) {
			float* const dwp = dw->data.f32 + j * dwinc[0];
			const float v = gp[j];
			int k;
			for (k = 0; k < a->info.dim[0]; k++)
				dwp[k] += ap[k] * v;
		} parallel_endfor
	}
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (h)
	{
		const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2];
		assert(h->info.dim[0] == a->info.dim[0]);
		assert(ccv_max(1, h->info.dim[1]) == batch_size);
		assert(h->info.dim[2] == 0); // It is a 2-d array.
		const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
		const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
		for (i = 0; i < batch_size; i++)
		{
			const float* const gp = g->data.f32 + i * ginc[0];
			float* const hp = h->data.f32 + i * hinc[0];
			parallel_for(j, h->info.dim[0]) {
				const float* const wp = w->data.f32 + j;
				float v = 0;
				int k;
				for (k = 0; k < g->info.dim[0]; k++)
					v += wp[k * winc[0]] * gp[k];
				hp[j] = v;
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_softmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	int i, count = ccv_nnc_tensor_count(a->info);
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
	}
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	double maxval = ap[0];
	for (i = 1; i < count; i++)
		if (ap[i] > maxval)
			maxval = ap[i];
	double sumval = 0;
	for (i = 0; i < count; i++)
		sumval += (bp[i] = expf(ap[i] - maxval));
	sumval = 1.0 / sumval;
	for (i = 0; i < count; i++)
		bp[i] *= sumval;
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_softmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(0 && "This should never be called.");
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_relu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(b));
	int i, count = ccv_nnc_tensor_count(a->info);
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
	}
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	for (i = 0; i < count; i++)
		bp[i] = ccv_max(ap[i], 0);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_relu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* g = inputs[0]; // gradient
	assert(!CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_t* b = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* h = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(h));
	int i, count = ccv_nnc_tensor_count(g->info);
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
	{
		assert(b->info.dim[i] == g->info.dim[i]);
		assert(g->info.dim[i] == h->info.dim[i]);
	}
	float* bp = b->data.f32;
	float* gp = g->data.f32;
	float* hp = h->data.f32;
	for (i = 0; i < count; i++)
		hp[i] = (bp[i] >= 0) ? gp[i] : 0;
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewsum_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer((const ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	int x, z;
	int k = 0;
	// Bad, I promised this can be inplace operation. Need to first find out if there are share the same pointer first.
	for (z = 1; z < input_size; z++)
	{
		ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
		ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z];
		if (c->data.f32 == a->data.f32)
		{
			k = z;
			break;
		}
	}
	for (z = 0; z < input_size - 1; z++)
	{
		ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
		ccv_nnc_tensor_view_t* a = z > 0 ? c : (ccv_nnc_tensor_view_t*)inputs[k];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)(z >= k ? inputs[z + 1] : inputs[z]);
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
			assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
			dim[x] = ccv_max(1, a->info.dim[x]);
			ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
			cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = a->data.f32[x] + b->data.f32[x];
			continue;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[1] * dim[0];
		if (ainc[0] == dim[0] && binc[0] == dim[0] && cinc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0] (do memcpy for the last two dim)
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = ap[x] + bp[x];
					ap += ainc[1] * ainc[0];
					bp += binc[1] * binc[0];
					cp += cinc[1] * cinc[0];
				}
				ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
						cp[x] = ap[x] + bp[x];
					ap += ainc[0];
					bp += binc[0];
					cp += cinc[0];
				}
				ap += (ainc[1] - dim[1]) * ainc[0];
				bp += (binc[1] - dim[1]) * binc[0];
				cp += (cinc[1] - dim[1]) * cinc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static void _ccv_nnc_tensor_set(ccv_nnc_tensor_view_t* a, float b)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	int x;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			a->data.f32[x] = b;
		return;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	const int count = dim[1] * dim[0];
	if (ainc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0]
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < count; x++)
					ap[x] = b;
				ap += ainc[1] * ainc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		}
		return;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < dim[0]; x++)
					ap[x] = b;
				ap += ainc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
	}
}

static int _ccv_nnc_ewsum_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// D[x + y + z, x] = 1
	int i;
	if (inputs[0] == 0)
		// Set them to 1.
		for (i = 0; i < output_size; i++)
			_ccv_nnc_tensor_set((ccv_nnc_tensor_view_t*)outputs[i], 1);
	else
		// Copy over the gradient
		for (i = 0; i < output_size; i++)
			_ccv_nnc_tensor_transfer((ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[i]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_axpy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	if (input_size == 1 || inputs[1] == 0)
	{
		// It cannot be set otherwise we have trouble.
		assert(cmd.info.blas.a[1] == 0);
		if (cmd.info.blas.a[0] == 1)
		{
			_ccv_nnc_tensor_transfer((ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[0]);
			return CCV_NNC_EXEC_SUCCESS;
		} else if (cmd.info.blas.a[0] == 0) {
			ccv_nnc_tensor_zero(outputs[0]);
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Assuming this is float 32.
		int dim[CCV_NNC_MAX_DIM + 2];
		int ainc[CCV_NNC_MAX_DIM + 2];
		int binc[CCV_NNC_MAX_DIM + 2];
		ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		const float p = cmd.info.blas.a[0];
		int x;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
			dim[x] = ccv_max(1, a->info.dim[x]);
			ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				b->data.f32[x] = p * a->data.f32[x];
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		const int count = dim[1] * dim[0];
		if (ainc[0] == dim[0] && binc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0]
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
						bp[x] = p * ap[x];
					ap += ainc[1] * ainc[0];
					bp += binc[1] * binc[0];
				}
				ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
						bp[x] = p * ap[x];
					ap += ainc[0];
					bp += binc[0];
				}
				ap += (ainc[1] - dim[1]) * ainc[0];
				bp += (binc[1] - dim[1]) * binc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	if (cmd.info.blas.a[0] == 1 && cmd.info.blas.a[1] == 1)
	{
		ccv_nnc_cmd_t forw_cmd = cmd;
		forw_cmd.compute = CCV_NNC_COMPUTE_EWSUM_FORWARD;
		return _ccv_nnc_ewsum_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	} else if (cmd.info.blas.a[0] == 1 && cmd.info.blas.a[1] == 0) {
		_ccv_nnc_tensor_transfer((const ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	} else if (cmd.info.blas.a[0] == 0 && cmd.info.blas.a[1] == 1) {
		_ccv_nnc_tensor_transfer((const ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	} else if (cmd.info.blas.a[0] == 0 && cmd.info.blas.a[1] == 0) {
		ccv_nnc_tensor_zero(outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
	int x;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
		assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
		binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
		cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			c->data.f32[x] = p * a->data.f32[x] + q * b->data.f32[x];
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	float* cp = c->data.f32;
	const int count = dim[1] * dim[0];
	if (ainc[0] == dim[0] && binc[0] == dim[0] && cinc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0]
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < count; x++)
					cp[x] = p * ap[x] + q * bp[x];
				ap += ainc[1] * ainc[0];
				bp += binc[1] * binc[0];
				cp += cinc[1] * cinc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < dim[0]; x++)
					cp[x] = p * ap[x] + q * bp[x];
				ap += ainc[0];
				bp += binc[0];
				cp += cinc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
			bp += (binc[1] - dim[1]) * binc[0];
			cp += (cinc[1] - dim[1]) * cinc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		bp += (binc[2] - dim[2]) * binc[1] * binc[0];
		cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_axpy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	if (inputs[0] == 0)
	{
		if (outputs[0])
			_ccv_nnc_tensor_set((ccv_nnc_tensor_view_t*)outputs[0], cmd.info.blas.a[0]);
		if (output_size > 1 && outputs[1])
			_ccv_nnc_tensor_set((ccv_nnc_tensor_view_t*)outputs[1], cmd.info.blas.a[1]);
	} else {
		ccv_nnc_cmd_t forw_cmd = cmd;
		forw_cmd.compute = CCV_NNC_COMPUTE_AXPY_FORWARD;
		memset(forw_cmd.info.blas.a, 0, sizeof(forw_cmd.info.blas.a));
		if (outputs[0])
		{
			forw_cmd.info.blas.a[0] = cmd.info.blas.a[0];
			_ccv_nnc_axpy_forw(cmd, hint, flags, inputs, 1, outputs, 1, stream_context);
		}
		if (output_size > 1 && outputs[1])
		{
			forw_cmd.info.blas.a[0] = cmd.info.blas.a[1];
			_ccv_nnc_axpy_forw(cmd, hint, flags, inputs, 1, outputs + 1, 1, stream_context);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewprod_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	if (input_size == 1 && output_size == 1)
	{
		_ccv_nnc_tensor_transfer((const ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	int x, z;
	int k = 0;
	// Bad, I promised this can be inplace operation. Need to first find out if there are share the same pointer first.
	for (z = 1; z < input_size; z++)
	{
		ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
		ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z];
		if (c->data.f32 == a->data.f32)
		{
			k = z;
			break;
		}
	}
	for (z = 0; z < input_size - 1; z++)
	{
		ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
		ccv_nnc_tensor_view_t* a = z > 0 ? c : (ccv_nnc_tensor_view_t*)inputs[k];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)(z >= k ? inputs[z + 1] : inputs[z]);
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
			assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
			dim[x] = ccv_max(1, a->info.dim[x]);
			ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
			cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = a->data.f32[x] * b->data.f32[x];
			continue;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[1] * dim[0];
		if (ainc[0] == dim[0] && binc[0] == dim[0] && cinc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0]
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = ap[x] * bp[x];
					ap += ainc[1] * ainc[0];
					bp += binc[1] * binc[0];
					cp += cinc[1] * cinc[0];
				}
				ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
			}
			continue;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
						cp[x] = ap[x] * bp[x];
					ap += ainc[0];
					bp += binc[0];
					cp += cinc[0];
				}
				ap += (ainc[1] - dim[1]) * ainc[0];
				bp += (binc[1] - dim[1]) * binc[0];
				cp += (cinc[1] - dim[1]) * cinc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewprod_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// D[x * y * z, x] = y * z
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int hinc[CCV_NNC_MAX_DIM + 2];
	int x, z;
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[output_size + 1];
	if (g == 0)
	{
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			dim[x] = ccv_max(1, b->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
		}
		for (z = 0; z < output_size; z++)
		{
			ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z + 1];
			ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[z];
			assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			{
				assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, h->info.dim[x]));
				assert(ccv_max(1, h->info.dim[x]) == ccv_max(1, b->info.dim[x]));
				ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
				hinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(h) ? h->inc[x] : h->info.dim[x]);
			}
			if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(h))
			{
				// Super optimal case, just do one for-loop for sum.
				const int tensor_count = ccv_nnc_tensor_count(b->info);
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = b->data.f32[x] / a->data.f32[x];
				continue;
			}
			assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
			int i[CCV_NNC_MAX_DIM + 2];
			float* ap = a->data.f32;
			float* bp = b->data.f32;
			float* hp = h->data.f32;
			const int count = dim[1] * dim[0];
			if (ainc[0] == dim[0] && binc[0] == dim[0] && hinc[0] == dim[0])
			{
				// Special casing if the ainc[0] is the same as dim[0]
				for (i[3] = 0; i[3] < dim[3]; i[3]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < count; x++)
							hp[x] = bp[x] / ap[x];
						ap += ainc[1] * ainc[0];
						bp += binc[1] * binc[0];
						hp += hinc[1] * hinc[0];
					}
					ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
					bp += (binc[2] - dim[2]) * binc[1] * binc[0];
					hp += (hinc[2] - dim[2]) * hinc[1] * hinc[0];
				}
				continue;
			}
			// Non-optimal case, need to do skip copy.
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < dim[0]; x++)
							hp[x] = bp[x] / ap[x];
						ap += ainc[0];
						bp += binc[0];
						hp += hinc[0];
					}
					ap += (ainc[1] - dim[1]) * ainc[0];
					bp += (binc[1] - dim[1]) * binc[0];
					hp += (hinc[1] - dim[1]) * hinc[0];
				}
				ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				hp += (hinc[2] - dim[2]) * hinc[1] * hinc[0];
			}
		}
	} else {
		assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			dim[x] = ccv_max(1, b->info.dim[x]);
			ginc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(g) ? g->inc[x] : g->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
		}
		for (z = 0; z < output_size; z++)
		{
			ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[z + 1];
			ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[z];
			assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			assert(h->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
			for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
			{
				assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, h->info.dim[x]));
				assert(ccv_max(1, h->info.dim[x]) == ccv_max(1, g->info.dim[x]));
				assert(ccv_max(1, g->info.dim[x]) == ccv_max(1, b->info.dim[x]));
				ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
				hinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(h) ? h->inc[x] : h->info.dim[x]);
			}
			if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(h))
			{
				// Super optimal case, just do one for-loop for sum.
				const int tensor_count = ccv_nnc_tensor_count(g->info);
				for (x = 0; x < tensor_count; x++)
					h->data.f32[x] = g->data.f32[x] * b->data.f32[x] / a->data.f32[x];
				continue;
			}
			assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
			int i[CCV_NNC_MAX_DIM + 2];
			float* gp = g->data.f32;
			float* ap = a->data.f32;
			float* bp = b->data.f32;
			float* hp = h->data.f32;
			const int count = dim[1] * dim[0];
			if (ginc[0] == dim[0] && ainc[0] == dim[0] && binc[0] == dim[0] && hinc[0] == dim[0])
			{
				// Special casing if the ainc[0] is the same as dim[0]
				for (i[3] = 0; i[3] < dim[3]; i[3]++)
				{
					for (i[2] = 0; i[2] < dim[2]; i[2]++)
					{
						for (x = 0; x < count; x++)
							hp[x] = gp[x] * bp[x] / ap[x];
						gp += ginc[1] * ginc[0];
						ap += ainc[1] * ainc[0];
						bp += binc[1] * binc[0];
						hp += hinc[1] * hinc[0];
					}
					gp += (ginc[2] - dim[2]) * ginc[1] * ginc[0];
					ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
					bp += (binc[2] - dim[2]) * binc[1] * binc[0];
					hp += (hinc[2] - dim[2]) * hinc[1] * hinc[0];
				}
				continue;
			}
			// Non-optimal case, need to do skip copy.
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (i[1] = 0; i[1] < dim[1]; i[1]++)
					{
						for (x = 0; x < dim[0]; x++)
							hp[x] = gp[x] * bp[x] / ap[x];
						gp += ginc[0];
						ap += ainc[0];
						bp += binc[0];
						hp += hinc[0];
					}
					gp += (ginc[1] - dim[1]) * ginc[0];
					ap += (ainc[1] - dim[1]) * ainc[0];
					bp += (binc[1] - dim[1]) * binc[0];
					hp += (hinc[1] - dim[1]) * hinc[0];
				}
				gp += (ginc[2] - dim[2]) * ginc[1] * ginc[0];
				ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				hp += (hinc[2] - dim[2]) * hinc[1] * hinc[0];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewdiv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	if (a == 0) // Take 0 as all ones tensor.
	{
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		int x;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
			dim[x] = ccv_max(1, b->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
			cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(b->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = 1 / b->data.f32[x];
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[1] * dim[0];
		if (binc[0] == dim[0] && cinc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0]
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = 1 / bp[x];
					bp += binc[1] * binc[0];
					cp += cinc[1] * cinc[0];
				}
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
						cp[x] = 1 / bp[x];
					bp += binc[0];
					cp += cinc[0];
				}
				bp += (binc[1] - dim[1]) * binc[0];
				cp += (cinc[1] - dim[1]) * cinc[0];
			}
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
		}
	} else {
		assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		int x;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
			assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
			dim[x] = ccv_max(1, a->info.dim[x]);
			ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
			cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(a->info);
			for (x = 0; x < tensor_count; x++)
				c->data.f32[x] = a->data.f32[x] / b->data.f32[x];
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* ap = a->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		const int count = dim[1] * dim[0];
		if (ainc[0] == dim[0] && binc[0] == dim[0] && cinc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0]
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
						cp[x] = ap[x] / bp[x];
					ap += ainc[1] * ainc[0];
					bp += binc[1] * binc[0];
					cp += cinc[1] * cinc[0];
				}
				ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
						cp[x] = ap[x] / bp[x];
					ap += ainc[0];
					bp += binc[0];
					cp += cinc[0];
				}
				ap += (ainc[1] - dim[1]) * ainc[0];
				bp += (binc[1] - dim[1]) * binc[0];
				cp += (cinc[1] - dim[1]) * cinc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewdiv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// D[x / y, x] = 1 / y, D[x / y, y] = -x / y^2
	if (output_size == 1 || outputs[1] == 0)
	{
		// When we only need D[x / y, y]
		ccv_nnc_cmd_t forw_cmd = cmd;
		forw_cmd.compute = CCV_NNC_COMPUTE_EWDIV_FORWARD;
		return _ccv_nnc_ewdiv_forw(cmd, ccv_nnc_no_hint, flags, TENSOR_LIST(inputs[0], inputs[2]), &outputs[0], 1, stream_context);
	}
	int dim[CCV_NNC_MAX_DIM + 2];
	int ginc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int cinc[CCV_NNC_MAX_DIM + 2];
	int hainc[CCV_NNC_MAX_DIM + 2];
	int hbinc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* ha = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* hb = (ccv_nnc_tensor_view_t*)outputs[1];
	if (g == 0)
	{
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(ha->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(hb->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		int x;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
			assert(ccv_max(1, c->info.dim[x]) == ccv_max(1, ha->info.dim[x]));
			assert(ccv_max(1, ha->info.dim[x]) == ccv_max(1, hb->info.dim[x]));
			dim[x] = ccv_max(1, b->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
			cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
			hainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? ha->inc[x] : ha->info.dim[x]);
			hbinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(hb) ? hb->inc[x] : hb->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c) && !CCV_IS_TENSOR_VIEW(ha) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(b->info);
			for (x = 0; x < tensor_count; x++)
			{
				const float v = 1 / b->data.f32[x];
				ha->data.f32[x] = v;
				hb->data.f32[x] = -c->data.f32[x] * v;
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		float* hap = ha->data.f32;
		float* hbp = hb->data.f32;
		const int count = dim[1] * dim[0];
		if (binc[0] == dim[0] && cinc[0] == dim[0] && hainc[0] == dim[0] && hbinc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0]
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
					{
						const float v = 1 / bp[x];
						hap[x] = v;
						hbp[x] = -cp[x] * v;
					}
					bp += binc[1] * binc[0];
					cp += cinc[1] * cinc[0];
					hap += hainc[1] * hainc[0];
					hbp += hbinc[1] * hbinc[0];
				}
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
				hap += (hainc[2] - dim[2]) * hainc[1] * hainc[0];
				hbp += (hbinc[2] - dim[2]) * hbinc[1] * hbinc[0];
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
					{
						const float v = 1 / bp[x];
						hap[x] = v;
						hbp[x] = -cp[x] * v;
					}
					bp += binc[0];
					cp += cinc[0];
					hap += hainc[0];
					hbp += hbinc[0];
				}
				bp += (binc[1] - dim[1]) * binc[0];
				cp += (cinc[1] - dim[1]) * cinc[0];
				hap += (hainc[1] - dim[1]) * hainc[0];
				hbp += (hbinc[1] - dim[1]) * hbinc[0];
			}
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
			hap += (hainc[2] - dim[2]) * hainc[1] * hainc[0];
			hbp += (hbinc[2] - dim[2]) * hbinc[1] * hbinc[0];
		}
	} else {
		assert(g->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(c->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(ha->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		assert(hb->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
		int x;
		for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		{
			assert(ccv_max(1, g->info.dim[x]) == ccv_max(1, g->info.dim[x]));
			assert(ccv_max(1, b->info.dim[x]) == ccv_max(1, c->info.dim[x]));
			assert(ccv_max(1, c->info.dim[x]) == ccv_max(1, ha->info.dim[x]));
			assert(ccv_max(1, ha->info.dim[x]) == ccv_max(1, hb->info.dim[x]));
			dim[x] = ccv_max(1, b->info.dim[x]);
			ginc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(g) ? g->inc[x] : g->info.dim[x]);
			binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
			cinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(c) ? c->inc[x] : c->info.dim[x]);
			hainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? ha->inc[x] : ha->info.dim[x]);
			hbinc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(hb) ? hb->inc[x] : hb->info.dim[x]);
		}
		if (!CCV_IS_TENSOR_VIEW(g) && !CCV_IS_TENSOR_VIEW(b) && !CCV_IS_TENSOR_VIEW(c) && !CCV_IS_TENSOR_VIEW(ha) && !CCV_IS_TENSOR_VIEW(hb))
		{
			// Super optimal case, just do one for-loop for sum.
			const int tensor_count = ccv_nnc_tensor_count(g->info);
			for (x = 0; x < tensor_count; x++)
			{
				const float v = g->data.f32[x] / b->data.f32[x];
				ha->data.f32[x] = v;
				hb->data.f32[x] = -c->data.f32[x] * v;
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
		int i[CCV_NNC_MAX_DIM + 2];
		float* gp = g->data.f32;
		float* bp = b->data.f32;
		float* cp = c->data.f32;
		float* hap = ha->data.f32;
		float* hbp = hb->data.f32;
		const int count = dim[1] * dim[0];
		if (ginc[0] == dim[0] && binc[0] == dim[0] && cinc[0] == dim[0] && hainc[0] == dim[0] && hbinc[0] == dim[0])
		{
			// Special casing if the ainc[0] is the same as dim[0]
			for (i[3] = 0; i[3] < dim[3]; i[3]++)
			{
				for (i[2] = 0; i[2] < dim[2]; i[2]++)
				{
					for (x = 0; x < count; x++)
					{
						const float v = gp[x] / bp[x];
						hap[x] = v;
						hbp[x] = -cp[x] * v;
					}
					gp += ginc[1] * ginc[0];
					bp += binc[1] * binc[0];
					cp += cinc[1] * cinc[0];
					hap += hainc[1] * hainc[0];
					hbp += hbinc[1] * hbinc[0];
				}
				gp += (ginc[2] - dim[2]) * ginc[1] * ginc[0];
				bp += (binc[2] - dim[2]) * binc[1] * binc[0];
				cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
				hap += (hainc[2] - dim[2]) * hainc[1] * hainc[0];
				hbp += (hbinc[2] - dim[2]) * hbinc[1] * hbinc[0];
			}
			return CCV_NNC_EXEC_SUCCESS;
		}
		// Non-optimal case, need to do skip copy.
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (i[1] = 0; i[1] < dim[1]; i[1]++)
				{
					for (x = 0; x < dim[0]; x++)
					{
						const float v = gp[x] / bp[x];
						hap[x] = v;
						hbp[x] = -cp[x] * v;
					}
					gp += ginc[0];
					bp += binc[0];
					cp += cinc[0];
					hap += hainc[0];
					hbp += hbinc[0];
				}
				gp += (ginc[1] - dim[1]) * ginc[0];
				bp += (binc[1] - dim[1]) * binc[0];
				cp += (cinc[1] - dim[1]) * cinc[0];
				hap += (hainc[1] - dim[1]) * hainc[0];
				hbp += (hbinc[1] - dim[1]) * hbinc[0];
			}
			gp += (ginc[2] - dim[2]) * ginc[1] * ginc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
			cp += (cinc[2] - dim[2]) * cinc[1] * cinc[0];
			hap += (hainc[2] - dim[2]) * hainc[1] * hainc[0];
			hbp += (hbinc[2] - dim[2]) * hbinc[1] * hbinc[0];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewexp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	int x;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
		binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			b->data.f32[x] = exp(a->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	const int count = dim[1] * dim[0];
	if (ainc[0] == dim[0] && binc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0]
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < count; x++)
					bp[x] = exp(ap[x]);
				ap += ainc[1] * ainc[0];
				bp += binc[1] * binc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < dim[0]; x++)
					bp[x] = exp(ap[x]);
				ap += ainc[0];
				bp += binc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
			bp += (binc[1] - dim[1]) * binc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		bp += (binc[2] - dim[2]) * binc[1] * binc[0];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewexp_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// D[Exp[x], x] = Exp[x]
	if (inputs[0] == 0)
	{
		_ccv_nnc_tensor_transfer((ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
		return CCV_NNC_EXEC_SUCCESS;
	} else {
		ccv_nnc_cmd_t forw_cmd = cmd;
		forw_cmd.compute = CCV_NNC_COMPUTE_EWPROD_FORWARD;
		return _ccv_nnc_ewprod_forw(cmd, ccv_nnc_no_hint, flags, TENSOR_LIST(inputs[0], inputs[2]), outputs, output_size, stream_context);
	}
}

static int _ccv_nnc_ewlog_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	// Assuming this is float 32.
	int dim[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	int x;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		assert(ccv_max(1, a->info.dim[x]) == ccv_max(1, b->info.dim[x]));
		dim[x] = ccv_max(1, a->info.dim[x]);
		ainc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(a) ? a->inc[x] : a->info.dim[x]);
		binc[x] = ccv_max(1, CCV_IS_TENSOR_VIEW(b) ? b->inc[x] : b->info.dim[x]);
	}
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		// Super optimal case, just do one for-loop for sum.
		const int tensor_count = ccv_nnc_tensor_count(a->info);
		for (x = 0; x < tensor_count; x++)
			b->data.f32[x] = log(a->data.f32[x]);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int i[CCV_NNC_MAX_DIM + 2];
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	const int count = dim[1] * dim[0];
	if (ainc[0] == dim[0] && binc[0] == dim[0])
	{
		// Special casing if the ainc[0] is the same as dim[0]
		for (i[3] = 0; i[3] < dim[3]; i[3]++)
		{
			for (i[2] = 0; i[2] < dim[2]; i[2]++)
			{
				for (x = 0; x < count; x++)
					bp[x] = log(ap[x]);
				ap += ainc[1] * ainc[0];
				bp += binc[1] * binc[0];
			}
			ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
			bp += (binc[2] - dim[2]) * binc[1] * binc[0];
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	// Non-optimal case, need to do skip copy.
	for (i[3] = 0; i[3] < dim[3]; i[3]++)
	{
		for (i[2] = 0; i[2] < dim[2]; i[2]++)
		{
			for (i[1] = 0; i[1] < dim[1]; i[1]++)
			{
				for (x = 0; x < dim[0]; x++)
					bp[x] = log(ap[x]);
				ap += ainc[0];
				bp += binc[0];
			}
			ap += (ainc[1] - dim[1]) * ainc[0];
			bp += (binc[1] - dim[1]) * binc[0];
		}
		ap += (ainc[2] - dim[2]) * ainc[1] * ainc[0];
		bp += (binc[2] - dim[2]) * binc[1] * binc[0];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_ewlog_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_cmd_t forw_cmd = cmd;
	forw_cmd.compute = CCV_NNC_COMPUTE_EWDIV_FORWARD;
	// D[Log[x], x] = 1 / x
	return _ccv_nnc_ewdiv_forw(forw_cmd, ccv_nnc_no_hint, flags, TENSOR_LIST(inputs[0], inputs[1]), outputs, output_size, stream_context);
	// Otherwise, need to add them together.
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_set(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	int i;
	if (cmd.info.blas.a[0] == 0)
		for (i = 0; i < output_size; i++)
			ccv_nnc_tensor_zero(outputs[i]);
	else
		for (i = 0; i < output_size; i++)
			_ccv_nnc_tensor_set((ccv_nnc_tensor_view_t*)outputs[i], cmd.info.blas.a[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

//@ccv_nnc_init CCV_NNC_BACKEND_CPU_REF
void ccv_nnc_cpu_ref_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/*TODO: I don't think any of these methods handles batch input, and I better to handle CHWN as well. */
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_FORWARD].exec = _ccv_nnc_conv_forw;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTION_BACKWARD].exec = _ccv_nnc_conv_back;
	/* Max pool layer */
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].exec = _ccv_nnc_max_pool_forw;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].exec = _ccv_nnc_max_pool_back;
	/* Average pool layer */
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].exec = _ccv_nnc_avg_pool_forw;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].exec = _ccv_nnc_avg_pool_back;
	/* Softmax layer */
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].exec = _ccv_nnc_softmax_forw;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].exec = _ccv_nnc_softmax_back;
	/* ReLU activation */
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].exec = _ccv_nnc_relu_forw;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].exec = _ccv_nnc_relu_back;
	/* GEMM layer */
	cmd_api[CCV_NNC_COMPUTE_GEMM_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_GEMM_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_GEMM_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_GEMM_FORWARD].exec = _ccv_nnc_gemm_forw;
	cmd_api[CCV_NNC_COMPUTE_GEMM_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_GEMM_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_GEMM_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_GEMM_BACKWARD].exec = _ccv_nnc_gemm_back;
	/* axpy layer */
	cmd_api[CCV_NNC_COMPUTE_AXPY_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_AXPY_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_AXPY_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_AXPY_FORWARD].exec = _ccv_nnc_axpy_forw;
	cmd_api[CCV_NNC_COMPUTE_AXPY_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_AXPY_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_AXPY_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_AXPY_BACKWARD].exec = _ccv_nnc_axpy_back;
	// Element-wise computation
	cmd_api[CCV_NNC_COMPUTE_EWSUM_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_FORWARD].exec = _ccv_nnc_ewsum_forw;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWSUM_BACKWARD].exec = _ccv_nnc_ewsum_back;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_FORWARD].exec = _ccv_nnc_ewprod_forw;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWPROD_BACKWARD].exec = _ccv_nnc_ewprod_back;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_FORWARD].exec = _ccv_nnc_ewdiv_forw;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWDIV_BACKWARD].exec = _ccv_nnc_ewdiv_back;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_FORWARD].exec = _ccv_nnc_ewexp_forw;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWEXP_BACKWARD].exec = _ccv_nnc_ewexp_back;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_FORWARD].exec = _ccv_nnc_ewlog_forw;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_EWLOG_BACKWARD].exec = _ccv_nnc_ewlog_back;
	/* Set to a scalar */
	cmd_api[CCV_NNC_COMPUTE_SET_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_SET_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_SET_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_SET_FORWARD].exec = _ccv_nnc_set;
	cmd_api[CCV_NNC_COMPUTE_SET_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_SET_BACKWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_SET_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_SET_BACKWARD].exec = _ccv_nnc_set;
	/* Data transfer */
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].algorithms = -1;
	cmd_api[CCV_NNC_COMPUTE_DATA_TRANSFER_FORWARD].exec = _ccv_nnc_data_transfer;
	/* Format transform */
	cmd_api[CCV_NNC_COMPUTE_FORMAT_TRANSFORM_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	cmd_api[CCV_NNC_COMPUTE_FORMAT_TRANSFORM_FORWARD].tensor_memory = CCV_TENSOR_CPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_FORMAT_TRANSFORM_FORWARD].algorithms = -1;
	cmd_api[CCV_NNC_COMPUTE_FORMAT_TRANSFORM_FORWARD].exec = _ccv_nnc_format_transform;
}
