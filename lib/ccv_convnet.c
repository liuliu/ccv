#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

ccv_convnet_t* ccv_convnet_new(ccv_convnet_param_t params[], int count)
{
	ccv_convnet_t* convnet = (ccv_convnet_t*)ccmalloc(sizeof(ccv_convnet_t) + sizeof(ccv_convnet_layer_t) * count + sizeof(ccv_dense_matrix_t*) * count);
	convnet->layers = (ccv_convnet_layer_t*)(convnet + 1);
	convnet->neurons = (ccv_dense_matrix_t**)(convnet->layers + count);
	memset(convnet->neurons, 0, sizeof(ccv_dense_matrix_t*) * count);
	convnet->count = count;
	convnet->rows = params[0].input.matrix.rows;
	convnet->cols = params[0].input.matrix.cols;
	convnet->channels = params[0].input.matrix.channels;
	ccv_convnet_layer_t* layers = convnet->layers;
	int i;
	for (i = 0; i < count; i++)
	{
		layers[i].type = params[i].type;
		layers[i].net = params[i].output;
		layers[i].b = 0;
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
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				layers[i].w = 0;
				break;
		}
	}
	return convnet;
}

static void _ccv_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int ch = layer->net.convolutional.channels;
	int count = layer->net.convolutional.count;
	int strides = layer->net.convolutional.strides;
	int border = layer->net.convolutional.border;
	int kernel_rows = layer->net.convolutional.rows;
	int kernel_cols = layer->net.convolutional.cols;
	assert(kernel_rows % 2); // as of now, don't support even number of kernel size
	assert(kernel_cols % 2);
	assert((a->rows + border * 2 - kernel_rows) % strides == 0);
	assert((a->cols + border * 2 - kernel_cols) % strides == 0);
	int rows = (a->rows + border * 2 - kernel_rows) / strides + 1;
	int cols = (a->cols + border * 2 - kernel_cols) / strides + 1;
	int type = CCV_32F | count;
	assert(CCV_GET_CHANNEL(a->type) == ch);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, x, y, k;
	for (k = 0; k < count; k++)
	{
		float* ap = a->data.f32;
		float* bp = db->data.f32 + k;
		for (i = 0; i < db->rows; i++)
		{
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border);
			int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(a->rows + border, i * strides + kernel_rows));
			comy *= ch * kernel_cols;
			for (j = 0; j < db->cols; j++)
			{
				float v = 0;
				int comx = (ccv_max(j * strides - border, 0) - (j * strides - border)) * ch;
				int maxx = kernel_cols * ch - comx - (j * strides + kernel_cols - ccv_min(a->cols + border, j * strides + kernel_cols)) * ch;
				float* w = layer->w + comx + comy;
				float* apz = ap + ccv_max(j * strides - border, 0) * ch;
				// when we have border, we simply do zero padding
				for (y = 0; y < maxy; y++)
				{
					for (x = 0; x < maxx; x++)
						v += w[x] * apz[x];
					w += kernel_cols * ch;
					apz += a->cols * ch;
				}
				bp[j * count] = v;
			}
			bp += db->cols * count;
			ap += a->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0));
		}
	}
}

static void _ccv_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, layer->net.full_connect.count, 1, CCV_32F | CCV_C1, CCV_32F | CCV_C1, 0);
	int ch = CCV_GET_CHANNEL(a->type);
	int rows = a->rows;
	int cols = a->cols;
	// reshape a for gemm
	assert(a->step == a->cols * CCV_GET_DATA_TYPE_SIZE(a->type) * ch);
	a->rows = rows * cols * ch;
	a->cols = 1;
	a->type = (a->type - ch) | CCV_C1;
	ccv_dense_matrix_t dw = ccv_dense_matrix(db->rows, a->rows, CCV_32F | CCV_C1, layer->w, 0);
	ccv_gemm(&dw, a, 1, 0, 0, 0, (ccv_matrix_t**)&db, 0);
	a->type = (a->type - CCV_GET_CHANNEL(a->type)) | ch;
	a->rows = rows;
	a->cols = cols;
}

static void _ccv_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int size = layer->net.pool.size;
	int strides = layer->net.pool.strides;
	assert((a->rows - size) % strides == 0);
	assert((a->cols - size) % strides == 0);
	int rows = (a->rows - size) / strides + 1;
	int cols = (a->cols - size) / strides + 1;
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int ch = CCV_GET_CHANNEL(a->type);
	int type = CCV_32F | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, k, x, y;
	float* ap = a->data.f32;
	float* bp = db->data.f32;
	for (i = 0; i < db->rows; i++)
	{
		for (j = 0; j < db->cols; j++)
			for (k = 0; k < ch; k++)
			{
				float v = ap[j* strides * ch + k];
				for (x = 1; x < size; x++)
					if (ap[(j * strides + x) * ch + k] > v)
						v = ap[(j * strides + x) * ch + k];
				for (y = 1; y < size; y++)
					for (x = 0; x < size; x++)
						if (ap[(j * strides + x + y * a->cols) * ch + k] > v)
							v = ap[(j * strides + x + y * a->cols) * ch + k];
				bp[j * ch + k] = v;
			}
		ap += a->cols * ch * strides;
		bp += db->cols * ch;
	}
}

static void _ccv_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int size = layer->net.pool.size;
	int strides = layer->net.pool.strides;
	assert((a->rows - size) % strides == 0);
	assert((a->cols - size) % strides == 0);
	int rows = (a->rows - size) / strides + 1;
	int cols = (a->cols - size) / strides + 1;
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int ch = CCV_GET_CHANNEL(a->type);
	int type = CCV_32F | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, k, x, y;
	float* ap = a->data.f32;
	float* bp = db->data.f32;
	float inv_size = 1.0 / (size * size);
	for (i = 0; i < db->rows; i++)
	{
		for (j = 0; j < db->cols; j++)
			for (k = 0; k < ch; k++)
			{
				float v = 0;
				for (y = 0; y < size; y++)
					for (x = 0; x < size; x++)
						v += ap[(j * strides + x + y * a->cols) * ch + k];
				bp[j * ch + k] = v * inv_size;
			}
		ap += a->cols * ch * strides;
		bp += db->cols * ch;
	}
}

void ccv_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	assert(CCV_GET_CHANNEL(a->type) == convnet->channels);
	assert(a->rows == convnet->rows);
	assert(a->cols == convnet->cols);
	int i;
	// save the last layer of neuron cache in case that we encode to a different matrix
	ccv_dense_matrix_t* out_neuron = convnet->neurons[convnet->count - 1];
	convnet->neurons[convnet->count - 1] = *b;
	switch(convnet->layers->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			_ccv_convnet_convolutional_forward_propagate(convnet->layers, a, convnet->neurons);
			break;
		case CCV_CONVNET_FULL_CONNECT:
			_ccv_convnet_full_connect_forward_propagate(convnet->layers, a, convnet->neurons);
			break;
		case CCV_CONVNET_MAX_POOL:
			_ccv_convnet_max_pool_forward_propagate(convnet->layers, a, convnet->neurons);
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			_ccv_convnet_average_pool_forward_propagate(convnet->layers, a, convnet->neurons);
			break;
	}
	assert(type == 0 || CCV_GET_DATA_TYPE(type) == CCV_32F);
	for (i = 1; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		switch(layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_ccv_convnet_convolutional_forward_propagate(layer, convnet->neurons[i - 1], convnet->neurons + i);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_ccv_convnet_full_connect_forward_propagate(layer, convnet->neurons[i - 1], convnet->neurons + i);
				break;
			case CCV_CONVNET_MAX_POOL:
				_ccv_convnet_max_pool_forward_propagate(layer, convnet->neurons[i - 1], convnet->neurons + i);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_ccv_convnet_average_pool_forward_propagate(layer, convnet->neurons[i - 1], convnet->neurons + i);
				break;
		}
	}
	*b = convnet->neurons[convnet->count - 1];
	// restore the last layer of neuron cache
	convnet->neurons[convnet->count - 1] = out_neuron;
}

int ccv_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t* a)
{
	ccv_convnet_encode(convnet, a, convnet->neurons + convnet->count - 1, 0);
	int i, c = 0;
	ccv_dense_matrix_t* b = convnet->neurons[convnet->count - 1];
	int maxc = b->data.f32[0];
	for (i = 1; i < b->cols; i++)
		if (b->data.f32[i] > maxc)
			maxc = b->data.f32[i], c = i;
	return c;
}

void ccv_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds)
{
	int i;
	for (i = 0; i < categorizeds->rnum; i++)
	{
		// ccv_train_supervised_item_t* item = (ccv_train_supervised_item_t*)ccv_array_get(supervised_items, i);
	}
}

void ccv_convnet_free(ccv_convnet_t* convnet)
{
	ccfree(convnet);
}
