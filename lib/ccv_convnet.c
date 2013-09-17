#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

ccv_convnet_t* ccv_convnet_new(ccv_convnet_param_t params[], int count)
{
	ccv_convnet_t* convnet = (ccv_convnet_t*)ccmalloc(sizeof(ccv_convnet_t) + sizeof(ccv_convnet_layer_t) * count);
	convnet->layers = (ccv_convnet_layer_t*)(convnet + 1);
	convnet->rows = params[0].input.matrix.rows;
	convnet->cols = params[0].input.matrix.cols;
	convnet->channels = params[0].input.matrix.channels;
	ccv_convnet_layer_t* layers = convnet->layers;
	int i;
	for (i = 0; i < count; i++)
	{
		layers[i].type = params[i].type;
		layers[i].net = params[i].output;
		size_t size = 0;
		switch (params[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				size = params[i].output.convolutional.rows * params[i].output.convolutional.cols * params[i].output.convolutional.channels * params[i].output.convolutional.count;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				size = params[i].input.node.count * params[i].output.full_connect.count;
				break;
			case CCV_CONVNET_SOFTMAX:
				size = params[i].input.node.count * params[i].output.softmax.count;
				break;
		}
		switch (params[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
			case CCV_CONVNET_FULL_CONNECT:
			case CCV_CONVNET_SOFTMAX:
				layers[i].w = (float*)ccmalloc(sizeof(float) * size);
				layers[i].b = layers[i].w + size;
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				layers[i].w = layers[i].b = 0;
				break;
		}
	}
	return convnet;
}

static void _ccv_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows = (a->rows + layer->net.convolutional.border * 2 + 1 - layer->net.convolutional.rows) / layer->net.convolutional.strides;
	int cols = (a->cols + layer->net.convolutional.border * 2 + 1 - layer->net.convolutional.cols) / layer->net.convolutional.strides;
	int type = CCV_32F | layer->net.convolutional.count;
	assert(CCV_GET_CHANNEL(a->type) == layer->net.convolutional.channels);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	// ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
}

int ccv_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t* a)
{
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		ccv_dense_matrix_t* b = 0;
		_ccv_convnet_convolutional_forward_propagate(layer, a, &b);
	}
	return 0;
}

void ccv_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* supervised_items)
{
	int i;
	for (i = 0; i < supervised_items->rnum; i++)
	{
		// ccv_train_supervised_item_t* item = (ccv_train_supervised_item_t*)ccv_array_get(supervised_items, i);
	}
}

void ccv_convnet_free(ccv_convnet_t* convnet)
{
	ccfree(convnet);
}
