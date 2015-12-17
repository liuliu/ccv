#include <nnc/ccv_nnc.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc_internal.h>
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define set_n_m_dim(x, wd) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1], 0) - (i[x] * hint.stride.dim[x + 1] - hint.border.begin[x + 1]); \
		m[x] = wd[x + 1] - n[x] - (i[x] * hint.stride.dim[x + 1] + wd[x + 1] - ccv_min(a->info.dim[x + 1] + hint.border.end[x + 1], i[x] * hint.stride.dim[x + 1] + wd[x + 1])); \
	} while (0)

static void _ccv_nnc_net_conv_forw(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_t* a = inputs[0];
	const ccv_nnc_tensor_t* w = inputs[1];
	const ccv_nnc_tensor_t* bias = inputs[2];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(w->info.dim[0] == net->info.size.dim[0]);
	assert(w->info.dim[0] == a->info.dim[0]);
	int i;
	// Make sure the weights dimension matches the network dimension
	for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC; i++)
	{
		if (w->info.dim[i] == 0 || net->info.size.dim[i] == 0)
			break;
		assert(w->info.dim[i] == net->info.size.dim[i]);
	}
	// Make sure the weights output dimension matches the network convolutional kernels
	for (i = CCV_NNC_MAX_DIM_ALLOC - 1; i > 0; i--)
		if (w->info.dim[i] == 0 && w->info.dim[i])
		{
			assert(w->info.dim[i] == net->info.convolutional.count);
			break;
		}
	assert(bias->info.dim[0] == net->info.convolutional.count);
	// kernel weight for one dim.
	int kw_dim = w->info.dim[0] * w->info.dim[1] * w->info.dim[2];
	parallel_for(k, net->info.convolutional.count) {
		int c;
		float* ap = a->data.f32;
		float* bp = b->data.f32 + k;
		float* wp = w->data.f32 + k * kw_dim;
		float biasval = bias->data.f32[k];
		// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
		int i[CCV_NNC_MAX_DIM];
		int n[CCV_NNC_MAX_DIM];
		int m[CCV_NNC_MAX_DIM];
		int j[CCV_NNC_MAX_DIM];
		for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
		{
			set_n_m_dim(1, w->info.dim);
			float* wpu = wp + n[1] * w->info.dim[1] * w->info.dim[0];
			for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
			{
				set_n_m_dim(0, w->info.dim);
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

static void _ccv_nnc_net_conv_back(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
}

static void _ccv_nnc_net_max_pool_forw(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	const int *dim = net->info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim);
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

static void _ccv_nnc_net_max_pool_back(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
}

static void _ccv_nnc_net_avg_pool_forw(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	const int *dim = net->info.size.dim;
	int i[CCV_NNC_MAX_DIM];
	int n[CCV_NNC_MAX_DIM];
	int m[CCV_NNC_MAX_DIM];
	int j[CCV_NNC_MAX_DIM];
	int c;
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	for (i[1] = 0; i[1] < b->info.dim[2]; i[1]++)
	{
		set_n_m_dim(1, dim);
		for (i[0] = 0; i[0] < b->info.dim[1]; i[0]++)
		{
			set_n_m_dim(0, dim);
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

static void _ccv_nnc_net_avg_pool_back(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
}

//@ccv_nnc_init
void ccv_nnc_cpu_ref_init(ccv_nnc_api_t api[])
{
	/* Convolutional layer */
	api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_net_conv_forw;
	api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD].exec = _ccv_nnc_net_conv_back;
	/* Max pool layer */
	api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_COMPUTE_MAX_POOL_FORWARD].exec = _ccv_nnc_net_max_pool_forw;
	api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_COMPUTE_MAX_POOL_BACKWARD].exec = _ccv_nnc_net_max_pool_back;
	/* Average pool layer */
	api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD].exec = _ccv_nnc_net_avg_pool_forw;
	api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD].exec = _ccv_nnc_net_avg_pool_back;
}
