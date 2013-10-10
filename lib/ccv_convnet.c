#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

#ifndef CASE_TESTS

ccv_convnet_t* ccv_convnet_new(ccv_convnet_param_t params[], int count)
{
	ccv_convnet_t* convnet = (ccv_convnet_t*)ccmalloc(sizeof(ccv_convnet_t) + sizeof(ccv_convnet_layer_t) * count + sizeof(ccv_dense_matrix_t*) * count + sizeof(ccv_dense_matrix_t*) * (count - 1));
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	convnet->layers = (ccv_convnet_layer_t*)(convnet + 1);
	convnet->acts = (ccv_dense_matrix_t**)(convnet->layers + count);
	memset(convnet->acts, 0, sizeof(ccv_dense_matrix_t*) * count);
	if (count > 1) 
	{
		convnet->dropouts = (ccv_dense_matrix_t**)(convnet->acts + count);
		memset(convnet->dropouts, 0, sizeof(ccv_dense_matrix_t*) * (count - 1));
	} else {
		convnet->dropouts = 0;
	}
	convnet->count = count;
	convnet->rows = params[0].input.matrix.rows;
	convnet->cols = params[0].input.matrix.cols;
	convnet->channels = params[0].input.matrix.channels;
	ccv_convnet_layer_t* layers = convnet->layers;
	int i, j;
	for (i = 0; i < count; i++)
	{
		layers[i].type = params[i].type;
		layers[i].net = params[i].output;
		layers[i].dropout_rate = params[i].dropout_rate;
		switch (params[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				layers[i].wnum = params[i].output.convolutional.rows * params[i].output.convolutional.cols * params[i].output.convolutional.channels * params[i].output.convolutional.count;
				layers[i].w = (float*)ccmalloc(sizeof(float) * (layers[i].wnum + 1));
				layers[i].bias = layers[i].w + layers[i].wnum;
				for (j = 0; j < layers[i].wnum; j++)
					layers[i].w[j] = gsl_ran_gaussian(rng, params[i].sigma);
				layers[i].bias[0] = params[i].bias;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				layers[i].wnum = params[i].input.node.count * params[i].output.full_connect.count;
				layers[i].w = (float*)ccmalloc(sizeof(float) * (layers[i].wnum + params[i].output.full_connect.count));
				layers[i].bias = layers[i].w + layers[i].wnum;
				for (j = 0; j < layers[i].wnum; j++)
					layers[i].w[j] = gsl_ran_gaussian(rng, params[i].sigma);
				for (j = 0; j < params[i].output.full_connect.count; j++)
					layers[i].bias[j] = params[i].bias;
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				layers[i].wnum = 0;
				layers[i].w = 0;
				layers[i].bias = 0;
				break;
		}
	}
	gsl_rng_free(rng);
	return convnet;
}

#endif

static inline void _ccv_convnet_compute_output_scale(int a_rows, int a_cols, ccv_convnet_layer_t* layer, int* rows, int* cols)
{
	assert(rows != 0 && cols != 0);
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			assert(layer->net.convolutional.rows % 2); // as of now, don't support even number of kernel size
			assert(layer->net.convolutional.cols % 2);
			assert((a_rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) % layer->net.convolutional.strides == 0);
			assert((a_cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) % layer->net.convolutional.strides == 0);
			*rows = (a_rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) / layer->net.convolutional.strides + 1;
			*cols = (a_cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) / layer->net.convolutional.strides + 1;
			break;
		case CCV_CONVNET_FULL_CONNECT:
			*rows = layer->net.full_connect.count;
			*cols = 1;
			break;
		case CCV_CONVNET_MAX_POOL:
		case CCV_CONVNET_AVERAGE_POOL:
			assert((a_rows - layer->net.pool.size) % layer->net.pool.strides == 0);
			assert((a_cols - layer->net.pool.size) % layer->net.pool.strides == 0);
			*rows = (a_rows - layer->net.pool.size) / layer->net.pool.strides + 1;
			*cols = (a_cols - layer->net.pool.size) / layer->net.pool.strides + 1;
			break;
	}
}

static void _ccv_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t** b)
{
	int rows, cols;
	_ccv_convnet_compute_output_scale(a->rows, a->cols, layer, &rows, &cols);
	int ch = layer->net.convolutional.channels;
	int count = layer->net.convolutional.count;
	int strides = layer->net.convolutional.strides;
	int border = layer->net.convolutional.border;
	int kernel_rows = layer->net.convolutional.rows;
	int kernel_cols = layer->net.convolutional.cols;
	float bias = layer->bias[0];
	int type = CCV_32F | count;
	assert(CCV_GET_CHANNEL(a->type) == ch);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, x, y, k;
#define for_block(act_block_setup, act_block_begin, act_block_end) \
	for (k = 0; k < count; k++) \
	{ \
		float* ap = a->data.f32; \
		float* bp = db->data.f32 + k; \
		act_block_setup; \
		for (i = 0; i < db->rows; i++) \
		{ \
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border); \
			int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(a->rows + border, i * strides + kernel_rows)); \
			comy *= ch * kernel_cols; \
			for (j = 0; j < db->cols; j++) \
			{ \
				act_block_begin; \
				float v = bias; \
				int comx = (ccv_max(j * strides - border, 0) - (j * strides - border)) * ch; \
				int maxx = kernel_cols * ch - comx - (j * strides + kernel_cols - ccv_min(a->cols + border, j * strides + kernel_cols)) * ch; \
				float* w = layer->w + comx + comy; \
				float* apz = ap + ccv_max(j * strides - border, 0) * ch; \
				/* when we have border, we simply do zero padding */ \
				for (y = 0; y < maxy; y++) \
				{ \
					for (x = 0; x < maxx; x++) \
						v += w[x] * apz[x]; \
					w += kernel_cols * ch; \
					apz += a->cols * ch; \
				} \
				bp[j * count] = ccv_max(0, v) /* ReLU */; \
				act_block_end; \
			} \
			bp += db->cols * count; \
			ap += a->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0)); \
		} \
	}
	if (d)
	{
#define act_block_setup \
		int* dp = d->data.i32 + k;
#define act_block_begin \
		if (!*dp) \
		{
#define act_block_end \
		} else \
			bp[j * count] = 0; \
		dp += count;
		for_block(act_block_setup, act_block_begin, act_block_end);
#undef act_block_setup
#undef act_block_begin
#undef act_block_end
	} else {
		for_block(/* empty act block setup */, /* empty act block begin */, /* empty act block end */);
	}
#undef for_block
}

static void _ccv_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t** b)
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
	int i;
	float* bptr = db->data.f32;
	if (d)
	{
		int j;
		float* aptr = a->data.f32;
		float* wptr = layer->w;
		int* dptr = d->data.i32;
		for (i = 0; i < db->rows; i++)
		{
			if (!dptr[i])
			{
				float v = layer->bias[i];
				for (j = 0; j < a->rows; j++)
					v += aptr[j] * wptr[j];
				wptr += a->rows;
				bptr[i] = v;
			} else
				bptr[i] = 0;
		}
	} else {
		for (i = 0; i < db->rows; i++)
			bptr[i] = layer->bias[i];
		ccv_dense_matrix_t dw = ccv_dense_matrix(db->rows, a->rows, CCV_32F | CCV_C1, layer->w, 0);
		ccv_gemm(&dw, a, 1, db, 0, 0, (ccv_matrix_t**)&db, 0); // supply db as matrix C is allowed
	}
	a->type = (a->type - CCV_GET_CHANNEL(a->type)) | ch;
	a->rows = rows;
	a->cols = cols;
}

static void _ccv_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows, cols;
	_ccv_convnet_compute_output_scale(a->rows, a->cols, layer, &rows, &cols);
	int size = layer->net.pool.size;
	int strides = layer->net.pool.strides;
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

#ifndef CASE_TESTS

void ccv_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	assert(CCV_GET_CHANNEL(a->type) == convnet->channels);
	assert(a->rows == convnet->rows);
	assert(a->cols == convnet->cols);
	int i;
	// save the last layer of neuron cache in case that we encode to a different matrix
	ccv_dense_matrix_t* out_neuron = convnet->acts[convnet->count - 1];
	convnet->acts[convnet->count - 1] = *b;
	switch(convnet->layers->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			_ccv_convnet_convolutional_forward_propagate(convnet->layers, a, convnet->count > 1 ? convnet->dropouts[0] : 0, convnet->acts);
			break;
		case CCV_CONVNET_FULL_CONNECT:
			_ccv_convnet_full_connect_forward_propagate(convnet->layers, a, convnet->count > 1 ? convnet->dropouts[0] : 0, convnet->acts);
			break;
		case CCV_CONVNET_MAX_POOL:
			_ccv_convnet_max_pool_forward_propagate(convnet->layers, a, convnet->acts);
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			_ccv_convnet_average_pool_forward_propagate(convnet->layers, a, convnet->acts);
			break;
	}
	assert(type == 0 || CCV_GET_DATA_TYPE(type) == CCV_32F);
	for (i = 1; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		ccv_dense_matrix_t* d = i < convnet->count - 1 ? convnet->dropouts[i] : 0;
		switch(layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_ccv_convnet_convolutional_forward_propagate(layer, convnet->acts[i - 1], d, convnet->acts + i);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_ccv_convnet_full_connect_forward_propagate(layer, convnet->acts[i - 1], d, convnet->acts + i);
				break;
			case CCV_CONVNET_MAX_POOL:
				_ccv_convnet_max_pool_forward_propagate(layer, convnet->acts[i - 1], convnet->acts + i);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_ccv_convnet_average_pool_forward_propagate(layer, convnet->acts[i - 1], convnet->acts + i);
				break;
		}
	}
	*b = convnet->acts[convnet->count - 1];
	// restore the last layer of neuron cache
	convnet->acts[convnet->count - 1] = out_neuron;
}

int ccv_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t* a)
{
	ccv_convnet_encode(convnet, a, convnet->acts + convnet->count - 1, 0);
	int i, c = 0;
	ccv_dense_matrix_t* b = convnet->acts[convnet->count - 1];
	int maxc = b->data.f32[0];
	for (i = 1; i < b->cols; i++)
		if (b->data.f32[i] > maxc)
			maxc = b->data.f32[i], c = i;
	return c;
}

#endif

static void _ccv_convnet_randomize_dropouts(gsl_rng* rng, int a_rows, int a_cols, ccv_convnet_layer_t* layer, ccv_dense_matrix_t** dropouts)
{
	int rows, cols;
	_ccv_convnet_compute_output_scale(a_rows, a_cols, layer, &rows, &cols);
	ccv_dense_matrix_t* db = *dropouts = ccv_dense_matrix_renew(*dropouts, rows, cols, CCV_32S | CCV_C1, CCV_32S | CCV_C1, 0);
	int* bptr = db->data.i32;
	memset(bptr, 0, sizeof(int) * cols * rows);
	// dropout half of the neurons
	int i;
	for (i = 0; i < (int)(rows * cols * (1 - layer->dropout_rate) + 0.5); i++)
		bptr[i] = 1;
	// shuffle the dropout
	gsl_ran_shuffle(rng, bptr, rows * cols, sizeof(int));
}

static void _ccv_convnet_compute_softmax(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_32F | CCV_C1, CCV_32F | CCV_C1, 0);
	int i;
	float* aptr = a->data.f32;
	float* bptr = db->data.f32;
	float mv = aptr[0];
	for (i = 1; i < a->rows * a->cols; i++)
		if (aptr[i] > mv)
			mv = aptr[i];
	double tt = 0;
	for (i = 0; i < a->rows * a->cols; i++)
		tt += (bptr[i] = expf(aptr[i] - mv));
	tt = 1.0 / tt;
	for (i = 0; i < a->rows * a->cols; i++)
		bptr[i] *= tt;
}

// compute back propagated gradient & weight update delta
static void _ccv_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t* x, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), d is the dropout,
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// note that y (the output from forward prop) is not included because the full connect net is simple enough that we don't need it
}

static void _ccv_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t* x, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), d is the dropout,
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// note that y (the output from forward prop) is not included because the full connect net is simple enough that we don't need it
	// ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, x->rows, x->cols, CCV_32F | CCV_GET_CHANNEL(x->type), CCV_32F | CCV_GET_CHANNEL(x->type), 0);
	if (d)
	{
	} else {
	}
}

static void _ccv_convnet_max_pool_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* y, ccv_dense_matrix_t* x, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), y is the output (from forward prop),
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// pooling layer doesn't need the dropout
}

static void _ccv_convnet_average_pool_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* y, ccv_dense_matrix_t* x, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), y is the output (from forward prop),
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// pooling layer doesn't need the dropout
}

static void _ccv_convnet_propagate_loss(ccv_convnet_t* convnet, ccv_dense_matrix_t* dloss, ccv_convnet_t* update_params)
{
	int i;
	ccv_convnet_layer_t* layer = convnet->layers + convnet->count - 1;
	assert(layer->type == CCV_CONVNET_FULL_CONNECT); // the last layer has too be a full connect one to generate softmax result
	_ccv_convnet_full_connect_backward_propagate(layer, dloss, 0, convnet->acts[convnet->count - 2], update_params->acts + convnet->count - 2, update_params->layers + convnet->count - 1);
	for (i = convnet->count - 2; i >= 0; i--)
	{
		layer = convnet->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_ccv_convnet_convolutional_backward_propagate(layer, update_params->acts[i], convnet->dropouts[i], i > 0 ? convnet->acts[i - 1] : 0, i > 0 ? update_params->acts + i - 1 : 0, update_params->layers + i);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_ccv_convnet_full_connect_backward_propagate(layer, update_params->acts[i], convnet->dropouts[i], i > 0 ? convnet->acts[i - 1] : 0, i > 0 ? update_params->acts + i - 1 : 0, update_params->layers + i);
				break;
			case CCV_CONVNET_MAX_POOL:
				_ccv_convnet_max_pool_backward_propagate(layer, update_params->acts[i], convnet->acts[i], i > 0 ? convnet->acts[i - 1] : 0, i > 0 ? update_params->acts + i - 1 : 0, update_params->layers + i);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_ccv_convnet_average_pool_backward_propagate(layer, update_params->acts[i], convnet->acts[i], i > 0 ? convnet->acts[i - 1] : 0, i > 0 ? update_params->acts + i - 1 : 0, update_params->layers + i);
				break;
		}
	}
}

static ccv_convnet_t* _ccv_convnet_update_params(ccv_convnet_t* convnet)
{
	ccv_convnet_t* update_params = (ccv_convnet_t*)ccmalloc(sizeof(ccv_convnet_t) + sizeof(ccv_convnet_layer_t) * convnet->count + sizeof(ccv_dense_matrix_t*) * (convnet->count - 1));
	update_params->layers = (ccv_convnet_layer_t*)(update_params + 1);
	update_params->acts = (ccv_dense_matrix_t**)(update_params->layers + convnet->count);
	// the update params doesn't need the neuron layers (acts) for the input image, and the loss layer, therefore, convnet->count - 1
	memset(update_params->acts, 0, sizeof(ccv_dense_matrix_t*) * (convnet->count - 1));
	update_params->dropouts = 0;
	update_params->rows = convnet->rows;
	update_params->cols = convnet->cols;
	update_params->count = convnet->count;
	update_params->channels = convnet->channels;
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		update_params->layers[i].dropout_rate = convnet->layers[i].dropout_rate;
		update_params->layers[i].type = convnet->layers[i].type;
		update_params->layers[i].net = convnet->layers[i].net;
		update_params->layers[i].wnum = convnet->layers[i].wnum;
		switch (update_params->layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				update_params->layers[i].w = (float*)cccalloc(sizeof(float), update_params->layers[i].wnum + 1);
				update_params->layers[i].bias = update_params->layers[i].w + update_params->layers[i].wnum;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(update_params->layers[i].wnum % update_params->layers[i].net.full_connect.count == 0);
				update_params->layers[i].w = (float*)cccalloc(sizeof(float), update_params->layers[i].wnum + update_params->layers[i].net.full_connect.count);
				update_params->layers[i].bias = update_params->layers[i].w + update_params->layers[i].wnum;
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				update_params->layers[i].w = 0;
				update_params->layers[i].bias = 0;
				break;
		}
	}
	return update_params;
}

#ifndef CASE_TESTS

void ccv_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params)
{
	int i, j, t;
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	int aligned_padding = categorizeds->rnum % params.mini_batch;
	int aligned_rnum = categorizeds->rnum - aligned_padding;
	int* idx = (int*)ccmalloc(sizeof(int) * (categorizeds->rnum + aligned_padding));
	for (i = 0; i < categorizeds->rnum; i++)
		idx[i] = i;
	gsl_ran_shuffle(rng, idx, categorizeds->rnum, sizeof(int));
	// the last layer has to be full connect, thus we can use it as softmax layer
	assert(convnet->layers[convnet->count - 1].type == CCV_CONVNET_FULL_CONNECT);
	int category_count = convnet->layers[convnet->count - 1].net.full_connect.count;
	ccv_convnet_t* update_params = _ccv_convnet_update_params(convnet);
	for (t = 0; t < params.max_epoch; t++)
	{
		for (i = 0; i < aligned_rnum; i++)
		{
			// dropout the first hidden layer
			int a_rows = convnet->rows, a_cols = convnet->cols;
			for (j = 0; j < convnet->count - 1; j++)
			{
				switch (convnet->layers[j].type)
				{
					case CCV_CONVNET_CONVOLUTIONAL:
					case CCV_CONVNET_FULL_CONNECT:
						_ccv_convnet_randomize_dropouts(rng, a_rows, a_cols, convnet->layers + j, convnet->dropouts + j);
						break;
					case CCV_CONVNET_MAX_POOL:
					case CCV_CONVNET_AVERAGE_POOL:
						_ccv_convnet_compute_output_scale(a_rows, a_cols, convnet->layers + j, &a_rows, &a_cols);
						break;
				}
			}
			ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx[i]);
			ccv_convnet_encode(convnet, categorized->matrix, convnet->acts + convnet->count - 1, 0);
			ccv_dense_matrix_t* softmax = convnet->acts[convnet->count - 1];
			_ccv_convnet_compute_softmax(softmax, &softmax, 0);
			assert(softmax->rows == category_count && softmax->cols == 1);
			float* dloss = softmax->data.f32;
			// this mashes softmax and logistic regression together
			// also, it gives you -D[loss w.r.t. to x_i] (notice the negative sign), which is handy when updating weights
			for (j = 0; j < category_count; j++)
				dloss[j] = (j == categorized->c) - dloss[j];
			_ccv_convnet_propagate_loss(convnet, softmax, update_params);
			if ((i + 1) % params.mini_batch == 0)
			{
				// update weights
			}
		}
		if (t + 1 < params.max_epoch)
		{
			// reshuffle the parts we visited and move the rest to the beginning
			memcpy(idx + categorizeds->rnum, idx + aligned_rnum, sizeof(int) * aligned_padding);
			memmove(idx + aligned_padding, idx, sizeof(int) * aligned_rnum);
			memcpy(idx, idx + categorizeds->rnum, sizeof(int) * aligned_padding);
			gsl_ran_shuffle(rng, idx + aligned_padding, aligned_rnum, sizeof(int));
		}
	}
	ccfree(idx);
	ccv_convnet_free(update_params);
	gsl_rng_free(rng);
}

void ccv_convnet_free(ccv_convnet_t* convnet)
{
	ccfree(convnet);
}

#endif
