#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>

static void _ccv_nnc_net_conv_inference(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias)
{
	assert(w->meta.dim[0] == net->meta.convolutional.dim[0]);
	assert(w->meta.dim[1] == net->meta.convolutional.dim[1]);
	parallel_for(k, net->meta.convolutional.count) {
		/*
		int i, j, x, y, c;
		float* ap = a->data.f32;
		float* bp = b->data.f32 + k;
		float* wp = w->data.f32 + k * w->meta.dim[0] * w->meta.dim[1];
		for (i = 0; i < a->meta.dim[1]; i++)
		{
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border);
			int maxy = net->meta.convolutional.dim[1] - comy - (i * strides + net->meta.convolutional.dim[1] - ccv_min(a->meta.dim[1] + border, i * strides + net->meta.convolutional.dim[1]));
			for (j = 0; j < a->meta.dim[0]; j++)
			{
				for (y = 0; y < maxy; y++)
				{
					for (x = 0; x < maxx; x++)
					{
						for (c = 0; c < a->meta.channels; c++)
						{
						}
					}
				}
			}
		}
		*/
	} parallel_endfor
}

static void _ccv_nnc_net_conv_backprop(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* c, const ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias)
{
}

static void _ccv_nnc_net_max_pool_inference(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias)
{
}

static void _ccv_nnc_net_max_pool_backprop(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* c, const ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias)
{
}

static void _ccv_nnc_net_avg_pool_inference(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias)
{
}

static void _ccv_nnc_net_avg_pool_backprop(const ccv_nnc_net_t* net, const ccv_nnc_net_hint_t hint, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* c, const ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias)
{
}

void ccv_nnc_cpu_ref_init(ccv_nnc_api_t api[])
{
	api[CCV_NNC_TYPE_CONVOLUTIONAL].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_TYPE_CONVOLUTIONAL].inference = _ccv_nnc_net_conv_inference;
	api[CCV_NNC_TYPE_CONVOLUTIONAL].backprop = _ccv_nnc_net_conv_backprop;
	api[CCV_NNC_TYPE_MAX_POOL].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_TYPE_MAX_POOL].inference = _ccv_nnc_net_max_pool_inference;
	api[CCV_NNC_TYPE_MAX_POOL].backprop = _ccv_nnc_net_max_pool_backprop;
	api[CCV_NNC_TYPE_AVERAGE_POOL].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	api[CCV_NNC_TYPE_AVERAGE_POOL].inference = _ccv_nnc_net_avg_pool_inference;
	api[CCV_NNC_TYPE_AVERAGE_POOL].backprop = _ccv_nnc_net_avg_pool_backprop;
}
