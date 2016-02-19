#include <nnc/ccv_nnc.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc_internal.h>
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define set_n_m_dim(x, wd, ad) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1], 0) - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1]); \
		m[x] = wd[x + 1] - n[x] - (i[x] * hint.stride.dim[x + 1] + wd[x + 1] - ccv_min(ad[x + 1] + hint.border.end[x + 1], i[x] * hint.stride.dim[x + 1] + wd[x + 1])); \
	} while (0)

static void _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_t* a = inputs[0];
	const ccv_nnc_tensor_t* w = inputs[1];
	const ccv_nnc_tensor_t* bias = inputs[2];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(w->info.dim[0] == cmd.info.size.dim[0]);
	assert(w->info.dim[0] == a->info.dim[0]);
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
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * a->info.dim[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						for (c = 0; c < a->info.dim[0]; c++)
							p += wpz[j[0] * w->info.dim[0] + c] * apz[j[0] * a->info.dim[0] + c];
					wpz += w->info.dim[1] * w->info.dim[0];
					apz += a->info.dim[1] * a->info.dim[0];
				}
				bp[i[0] * b->info.dim[0]] = p;
			}
			bp += b->info.dim[1] * b->info.dim[0];
			ap += a->info.dim[1] * a->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		}
	} parallel_endfor
}

static void _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	// inputs: gradient, forw prop input / forw prop output, [w]
	// outputs: weight updates, bias updates, [output gradient]
	assert((input_size == 2 && output_size == 2) || (input_size == 3 && output_size == 3));
	ccv_nnc_tensor_t* a = inputs[1];
	ccv_nnc_tensor_t* g = inputs[0]; // gradients
	ccv_nnc_tensor_t* w = outputs[0];
	ccv_nnc_tensor_t* bias = outputs[1];
	ccv_nnc_tensor_t* h = output_size == 3 ? outputs[2] : 0; // output gradients
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		int i, count = 1;
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && w->info.dim[i] > 0; i++)
			count *= w->info.dim[i];
		memset(w->data.u8, 0, sizeof(float) * count);
		count = 1;
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && bias->info.dim[i] > 0; i++)
			count *= bias->info.dim[i];
		memset(bias->data.u8, 0, sizeof(float) * count);
	}
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
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * a->info.dim[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						for (c = 0; c < a->info.dim[0]; c++)
							wpz[j[0] * w->info.dim[0] + c] += v * apz[j[0] * a->info.dim[0] + c];
					wpz += w->info.dim[1] * w->info.dim[0];
					apz += a->info.dim[1] * a->info.dim[0];
				}
			}
			gp += g->info.dim[1] * g->info.dim[0];
			ap += a->info.dim[1] * a->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		}
		bias->data.f32[k] = biasval;
	} parallel_endfor
	// If h is available, therefore, we need to propagate the gradients back
	if (h)
	{
		// reset it to 0.
		int i, count = 1;
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && h->info.dim[i] > 0; i++)
			count *= h->info.dim[i];
		memset(h->data.u8, 0, sizeof(float) * count);
		w = inputs[2];
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
					const float v = gp[i[0] * g->info.dim[0]];
					if (v == 0) // shortcut if v is zero
						continue;
					float* wpz = wpu + n[0] * w->info.dim[0];
					float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * h->info.dim[0];
					for (j[1] = 0; j[1] < m[1]; j[1]++)
					{
						for (j[0] = 0; j[0] < m[0]; j[0]++)
							for (c = 0; c < h->info.dim[0]; c++)
								hpz[j[0] * h->info.dim[0] + c] += v * wpz[j[0] * w->info.dim[0] + c];
						wpz += w->info.dim[1] * w->info.dim[0];
						hpz += h->info.dim[1] * h->info.dim[0];
					}
				}
				gp += g->info.dim[1] * g->info.dim[0];
				hp += h->info.dim[1] * h->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
			}
		}
	}
}

static void _ccv_nnc_max_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, a->info.dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, a->info.dim);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * a->info.dim[0];
				float v = apz[0];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						if (apz[j[0] * a->info.dim[0]] > v)
							v = apz[j[1] * a->info.dim[0]];
					apz += a->info.dim[1] * a->info.dim[0];
				}
				bp[i[0] * b->info.dim[0] + c] = v;
			}
		}
		bp += b->info.dim[1] * b->info.dim[0];
		ap += a->info.dim[1] * a->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
}

static void _ccv_nnc_max_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	ccv_nnc_tensor_t* a = inputs[1];
	ccv_nnc_tensor_t* b = inputs[2];
	ccv_nnc_tensor_t* g = inputs[0]; // gradients
	assert(output_size == 1);
	ccv_nnc_tensor_t* h = outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	float* gp = g->data.f32;
	float* hp = h->data.f32;
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
	int count = 1;
	for (c = 0; c < CCV_NNC_MAX_DIM_ALLOC && h->info.dim[c] > 0; c++)
		count *= h->info.dim[c];
	memset(h->data.u8, 0, sizeof(float) * count);
	// Using b->info.dim and a->info.dim directly because they equal to g->info.dim and h->info.dim
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, a->info.dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, a->info.dim);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * a->info.dim[0];
				float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * a->info.dim[0];
				float v = bp[i[0] * b->info.dim[0] + c];
				float u = gp[i[0] * b->info.dim[0] + c];
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						if (apz[j[0] * a->info.dim[0]] == v)
							hpz[j[1] * a->info.dim[0]] += u;
					apz += a->info.dim[1] * a->info.dim[0];
					hpz += a->info.dim[1] * a->info.dim[0];
				}
			}
		}
		gp += b->info.dim[1] * b->info.dim[0];
		bp += b->info.dim[1] * b->info.dim[0];
		ap += a->info.dim[1] * a->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
		hp += a->info.dim[1] * a->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
}

static void _ccv_nnc_avg_pool_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, a->info.dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, a->info.dim);
			for (c = 0; c < b->info.dim[0]; c++)
			{
				float* apz = ap + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * a->info.dim[0];
				float v = 0;
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						v += apz[j[0] * a->info.dim[0]];
					apz += a->info.dim[1] * a->info.dim[0];
				}
				bp[i[0] * b->info.dim[0] + c] = v / (m[0] * m[1]);
			}
		}
		bp += b->info.dim[1] * b->info.dim[0];
		ap += a->info.dim[1] * a->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
}

static void _ccv_nnc_avg_pool_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* g = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* h = outputs[0];
	const int *dim = cmd.info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* gp = g->data.f32;
	float* hp = h->data.f32;
	int count = 1;
	for (c = 0; c < CCV_NNC_MAX_DIM_ALLOC && h->info.dim[c] > 0; c++)
		count *= h->info.dim[c];
	memset(h->data.u8, 0, sizeof(float) * count);
	for (i[1] = 0; i[1] < g->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim, h->info.dim);
		for (i[0] = 0; i[0] < g->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim, h->info.dim);
			for (c = 0; c < g->info.dim[0]; c++)
			{
				float* hpz = hp + ccv_max(i[0] * hint.stride.dim[1] - hint.border.begin[1], 0) * h->info.dim[0];
				float u = gp[i[0] * g->info.dim[0] + c] / (m[0] * m[1]);
				for (j[1] = 0; j[1] < m[1]; j[1]++)
				{
					for (j[0] = 0; j[0] < m[0]; j[0]++)
						hpz[j[0] * h->info.dim[0]] += u;
					hpz += h->info.dim[1] * h->info.dim[0];
				}
				;
			}
		}
		gp += g->info.dim[1] * g->info.dim[0];
		hp += h->info.dim[1] * h->info.dim[0] * (ccv_max((i[1] + 1) * hint.stride.dim[2] - hint.border.begin[2], 0) - ccv_max(i[1] * hint.stride.dim[2] - hint.border.begin[2], 0));
	}
}

static void _ccv_nnc_full_connect_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	ccv_nnc_tensor_t* w = inputs[1];
	ccv_nnc_tensor_t* bias = inputs[2];
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	ccv_nnc_tensor_param_t a_params = inputs[0]->info;
	int i, count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a_params.dim[i] > 0; i++)
		count *= a_params.dim[i];
	ccv_dense_matrix_t a = ccv_dense_matrix(1, count, CCV_32F | CCV_C1, inputs[0]->data.u8, 0);
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(b->info.dim[0] == 1);
	assert(b->info.dim[1] == bias->info.dim[1]);
	assert(bias->info.dim[0] == 1);
	// copy bias into each row.
	memcpy(b->data.f32, bias->data.f32, sizeof(float) * b->info.dim[1]);
	assert(a.info.dim[1] == w->info.dim[1]);
	assert(b->info.dim[1] == w->info.dim[2]);
	ccv_gemm(&a, &w, 1, b, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&b, 0); // supply b as matrix C is allowed
}

static void _ccv_nnc_full_connect_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	// inputs: gradient, forw prop input / forw prop output, [w]
	// outputs: weight updates, bias updates, [output gradient]
	assert((input_size == 2 && output_size == 2) || (input_size == 3 && output_size == 3));
	ccv_nnc_tensor_param_t g_params = inputs[0]->info;
	ccv_nnc_tensor_param_t a_params = inputs[1]->info;
	ccv_nnc_tensor_t* w = outputs[0];
	ccv_nnc_tensor_t* bias = outputs[1];
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		int i, count = 1;
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && w->info.dim[i] > 0; i++)
			count *= w->info.dim[i];
		memset(w->data.u8, 0, sizeof(float) * count);
		count = 1;
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && bias->info.dim[i] > 0; i++)
			count *= bias->info.dim[i];
		memset(bias->data.u8, 0, sizeof(float) * count);
	}
	int i, count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g_params.dim[i] > 0; i++)
		count *= g_params.dim[i];
	ccv_dense_matrix_t g = ccv_dense_matrix(count, 1, CCV_32F | CCV_C1, inputs[0]->data.u8, 0);
	ccv_dense_matrix_t bv = ccv_dense_matrix(count, 1, CCV_32F | CCV_C1, bias->data.u8, 0);
	ccv_dense_matrix_t* dbv = &bv;
	count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a_params.dim[i] > 0; i++)
		count *= a_params.dim[i];
	ccv_dense_matrix_t a = ccv_dense_matrix(count, 1, CCV_32F | CCV_C1, inputs[1]->data.u8, 0);
	ccv_add(&g, dbv, (ccv_matrix_t**)&dbv, 0);
	ccv_gemm(&g, &a, 1, w, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&w, 0);
	if (output_size == 3)
	{
		ccv_nnc_tensor_param_t h_params = outputs[2]->info;
		count = 1;
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && h_params.dim[i] > 0; i++)
			count *= h_params.dim[i];
		w = inputs[2];
		ccv_dense_matrix_t h = ccv_dense_matrix(count, 1, CCV_32F | CCV_C1, outputs[2]->data.u8, 0);
		ccv_dense_matrix_t* dh = &h;
		ccv_gemm(&w, &g, 1, 0, 0, CCV_A_TRANSPOSE, (ccv_matrix_t**)&dh, 0);
	}
}

static void _ccv_nnc_softmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	ccv_nnc_tensor_t* a = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	int i, count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
		count *= a->info.dim[i];
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
}

static void _ccv_nnc_softmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(0 && "This should never be called.");
}

static void _ccv_nnc_relu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	ccv_nnc_tensor_t* a = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	int i, count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
		count *= a->info.dim[i];
	}
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	for (i = 0; i < count; i++)
		bp[i] = ccv_max(ap[i], 0);
}

static void _ccv_nnc_relu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 2);
	ccv_nnc_tensor_t* b = inputs[1];
	ccv_nnc_tensor_t* g = inputs[0]; // gradient
	assert(output_size == 1);
	ccv_nnc_tensor_t* h = outputs[0];
	int i, count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
	{
		assert(b->info.dim[i] == g->info.dim[i]);
		assert(g->info.dim[i] == h->info.dim[i]);
		count *= g->info.dim[i];
	}
	float* bp = b->data.f32;
	float* gp = g->data.f32;
	float* hp = h->data.f32;
	for (i = 0; i < count; i++)
		hp[i] = (bp[i] >= 0) ? gp[i] : 0;
}

//@ccv_nnc_init
void ccv_nnc_cpu_ref_init(ccv_nnc_cmd_api_t cmd_api[])
{
	/*TODO: I don't think any of these methods handles batch input. */
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_conv_forw;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].exec = _ccv_nnc_conv_back;
	/* Full connect layer */
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD].exec = _ccv_nnc_full_connect_forw;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD].exec = _ccv_nnc_full_connect_back;
	/* Max pool layer */
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].exec = _ccv_nnc_max_pool_forw;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].exec = _ccv_nnc_max_pool_back;
	/* Average pool layer */
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].exec = _ccv_nnc_avg_pool_forw;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].exec = _ccv_nnc_avg_pool_back;
	/* Softmax layer */
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_FORWARD].exec = _ccv_nnc_softmax_forw;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_SOFTMAX_BACKWARD].exec = _ccv_nnc_softmax_back;
	/* ReLU activation */
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_RELU_FORWARD].exec = _ccv_nnc_relu_forw;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_RELU_BACKWARD].exec = _ccv_nnc_relu_back;
}
