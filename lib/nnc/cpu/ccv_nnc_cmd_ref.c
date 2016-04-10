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

// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define set_n_m_dim(x, wd, ad) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1], 0) - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1]); \
		m[x] = wd[x + 1] - n[x] - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1] + wd[x + 1] - ccv_min(ad[x + 1], i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1] + wd[x + 1])); \
	} while (0)

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
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* binc = CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim;
	assert(bias->info.dim[0] == cmd.info.convolutional.count);
	parallel_for(k, cmd.info.convolutional.count) {
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

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	// inputs: gradient, forw prop input / forw prop output, [w]
	// outputs: weight updates, bias updates, [output gradient]
	assert((input_size == 2 && output_size == 2) || (input_size == 3 && output_size == 3));
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
	ccv_nnc_tensor_t* w = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(w));
	ccv_nnc_tensor_t* bias = outputs[1];
	assert(!CCV_IS_TENSOR_VIEW(bias));
	ccv_nnc_tensor_view_t* h = output_size == 3 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0; // output gradients
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(w->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	const int* ginc = CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim;
	parallel_for(k, cmd.info.convolutional.count) {
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
		const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
		// reset it to 0.
		ccv_nnc_tensor_zero(h);
		w = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(w));
		int k;
		for (k = 0; k < cmd.info.convolutional.count; k++)
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

static int _ccv_nnc_max_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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

static int _ccv_nnc_max_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
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

static int _ccv_nnc_avg_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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

static int _ccv_nnc_avg_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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

static int _ccv_nnc_full_connect_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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

static int _ccv_nnc_full_connect_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: weight updates, bias updates, [output gradient]
	assert((input_size == 2 && output_size == 2) || (input_size == 3 && output_size == 3));
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* w = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(w->info.dim[2] == 0); // It is a 2-d array.
	ccv_nnc_tensor_view_t* bias = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(bias->info.dim[1] == 0); // It is a 1-d array.
	const int* winc = CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim;
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(w->data.u8, 0, sizeof(float) * winc[0] * w->info.dim[1]);
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
	assert(a->info.dim[0] == w->info.dim[0]);
	assert(g->info.dim[0] == w->info.dim[1]);
	const int* ainc = CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim;
	for (i = 0; i < batch_size; i++)
	{
		const float* const gp = g->data.f32 + i * ginc[0];
		const float* const ap = a->data.f32 + i * ainc[0];
		parallel_for(j, g->info.dim[0]) {
			float* const wp = w->data.f32 + j * winc[0];
			const float v = gp[j];
			int k;
			for (k = 0; k < a->info.dim[0]; k++)
				wp[k] += ap[k] * v;
		} parallel_endfor
	}
	if (output_size == 3)
	{
		ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[2];
		const ccv_nnc_tensor_view_t* dw = (const ccv_nnc_tensor_view_t*)inputs[2];
		assert(h->info.dim[0] == a->info.dim[0]);
		assert(ccv_max(1, h->info.dim[1]) == batch_size);
		assert(h->info.dim[2] == 0); // It is a 2-d array.
		const int* hinc = CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim;
		const int* dwinc = CCV_IS_TENSOR_VIEW(dw) ? dw->inc : dw->info.dim;
		for (i = 0; i < batch_size; i++)
		{
			const float* const gp = g->data.f32 + i * ginc[0];
			float* const hp = h->data.f32 + i * hinc[0];
			parallel_for(j, h->info.dim[0]) {
				const float* const dwp = dw->data.f32 + j;
				float v = 0;
				int k;
				for (k = 0; k < g->info.dim[0]; k++)
					v += dwp[k * dwinc[0]] * gp[k];
				hp[j] = v;
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_softmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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

static int _ccv_nnc_softmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(0 && "This should never be called.");
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_relu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
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

static int _ccv_nnc_relu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* b = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(b));
	const ccv_nnc_tensor_t* g = inputs[0]; // gradient
	assert(!CCV_IS_TENSOR_VIEW(g));
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

//@ccv_nnc_init CCV_NNC_BACKEND_CPU_REF
void ccv_nnc_cpu_ref_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/*TODO: I don't think any of these methods handles batch input, and I better to handle CHWN as well. */
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_conv_forw;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].exec = _ccv_nnc_conv_back;
	/* Full connect layer */
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].exec = _ccv_nnc_full_connect_forw;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].exec = _ccv_nnc_full_connect_back;
	/* Max pool layer */
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].exec = _ccv_nnc_max_pool_forw;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].exec = _ccv_nnc_max_pool_back;
	/* Average pool layer */
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].exec = _ccv_nnc_avg_pool_forw;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].exec = _ccv_nnc_avg_pool_back;
	/* Softmax layer */
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].exec = _ccv_nnc_softmax_forw;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].exec = _ccv_nnc_softmax_back;
	/* ReLU activation */
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].exec = _ccv_nnc_relu_forw;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].algorithms = 1;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].exec = _ccv_nnc_relu_back;
}
