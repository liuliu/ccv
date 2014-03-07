#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#ifdef HAVE_CUDA
#include "cuda/cwc.h"
#endif
#include "3rdparty/sqlite3/sqlite3.h"

inline static void _ccv_convnet_layer_deduce_output_format(ccv_convnet_layer_t* layer, int* rows, int* cols, int* partition)
{
	assert(rows != 0 && cols != 0);
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			assert(layer->net.convolutional.rows % 2); // as of now, don't support even number of kernel size
			assert(layer->net.convolutional.cols % 2);
			assert((layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows) % layer->net.convolutional.strides == 0);
			assert((layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols) % layer->net.convolutional.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.convolutional.border * 2 - layer->net.convolutional.rows + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.convolutional.border * 2 - layer->net.convolutional.cols + layer->net.convolutional.strides - 1) / layer->net.convolutional.strides + 1;
			*partition = layer->input.matrix.partition;
			break;
		case CCV_CONVNET_FULL_CONNECT:
			*rows = layer->net.full_connect.count;
			*cols = 1;
			*partition = 1;
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			*rows = layer->input.matrix.rows;
			*cols = layer->input.matrix.cols;
			*partition = layer->input.matrix.partition;
			break;
		case CCV_CONVNET_MAX_POOL:
		case CCV_CONVNET_AVERAGE_POOL:
			assert((layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			assert((layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size) % layer->net.pool.strides == 0);
			*rows = (layer->input.matrix.rows + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*cols = (layer->input.matrix.cols + layer->net.pool.border * 2 - layer->net.pool.size + layer->net.pool.strides - 1) / layer->net.pool.strides + 1;
			*partition = layer->input.matrix.partition;
			break;
		default:
			assert(0);
			break;
	}
}

#ifndef CASE_TESTS

ccv_convnet_t* ccv_convnet_new(int use_cwc_accel, ccv_size_t input, ccv_convnet_layer_param_t params[], int count)
{
	ccv_convnet_t* convnet = (ccv_convnet_t*)ccmalloc(sizeof(ccv_convnet_t) + sizeof(ccv_convnet_layer_t) * count + sizeof(ccv_dense_matrix_t*) * count * 2 + sizeof(ccv_dense_matrix_t*) * (count - 1));
	convnet->use_cwc_accel = use_cwc_accel;
#ifdef HAVE_GSL
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(rng, (unsigned long int)convnet);
#endif
	convnet->reserved = 0;
	convnet->layers = (ccv_convnet_layer_t*)(convnet + 1);
	convnet->acts = (ccv_dense_matrix_t**)(convnet->layers + count);
	memset(convnet->acts, 0, sizeof(ccv_dense_matrix_t*) * count);
	convnet->denoms = (ccv_dense_matrix_t**)(convnet->acts + count);
	memset(convnet->denoms, 0, sizeof(ccv_dense_matrix_t*) * count);
	convnet->count = count;
	convnet->input = input;
	convnet->rows = params[0].input.matrix.rows;
	convnet->cols = params[0].input.matrix.cols;
	convnet->channels = params[0].input.matrix.channels;
	convnet->mean_activity = ccv_dense_matrix_new(convnet->input.height, convnet->input.width, convnet->channels | CCV_32F, 0, 0);
	ccv_zero(convnet->mean_activity);
	ccv_convnet_layer_t* layers = convnet->layers;
	int i, j;
	for (i = 0; i < count; i++)
	{
		layers[i].type = params[i].type;
		layers[i].input = params[i].input;
		layers[i].net = params[i].output;
		switch (params[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				assert(params[i].input.matrix.channels % params[i].input.matrix.partition == 0);
				assert(params[i].output.convolutional.count % params[i].output.convolutional.partition == 0);
				assert(params[i].output.convolutional.partition % params[i].input.matrix.partition == 0);
				assert(params[i].output.convolutional.partition >= params[i].input.matrix.partition);
				layers[i].wnum = params[i].output.convolutional.rows * params[i].output.convolutional.cols * params[i].output.convolutional.channels / params[i].input.matrix.partition * params[i].output.convolutional.count;
				layers[i].w = (float*)ccmalloc(sizeof(float) * (layers[i].wnum + params[i].output.convolutional.count));
				layers[i].bias = layers[i].w + layers[i].wnum;
#ifdef HAVE_GSL
				for (j = 0; j < layers[i].wnum; j++)
					layers[i].w[j] = gsl_ran_gaussian(rng, params[i].sigma);
#else
				for (j = 0; j < layers[i].wnum; j++)
					layers[i].w[j] = 0;
#endif
				for (j = 0; j < params[i].output.convolutional.count; j++)
					layers[i].bias[j] = params[i].bias;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				layers[i].wnum = params[i].input.node.count * params[i].output.full_connect.count;
				layers[i].w = (float*)ccmalloc(sizeof(float) * (layers[i].wnum + params[i].output.full_connect.count));
				layers[i].bias = layers[i].w + layers[i].wnum;
#ifdef HAVE_GSL
				for (j = 0; j < layers[i].wnum; j++)
					layers[i].w[j] = gsl_ran_gaussian(rng, params[i].sigma);
#else
				for (j = 0; j < layers[i].wnum; j++)
					layers[i].w[j] = 0;
#endif
				for (j = 0; j < params[i].output.full_connect.count; j++)
					layers[i].bias[j] = params[i].bias;
				break;
			default:
				layers[i].wnum = 0;
				layers[i].w = 0;
				layers[i].bias = 0;
				break;
		}
	}
#ifdef HAVE_GSL
	gsl_rng_free(rng);
#endif
	return convnet;
}

int ccv_convnet_verify(ccv_convnet_t* convnet, int output)
{
	int i, out_rows, out_cols, out_partition;
	if (convnet->count < 1)
		return -1;
	for (i = 0; i < convnet->count; i++)
	{
		ccv_convnet_layer_t* layer = convnet->layers + i;
		if (i > 0 && (out_rows != layer->input.matrix.rows || out_cols != layer->input.matrix.cols))
			return -1;
		_ccv_convnet_layer_deduce_output_format(layer, &out_rows, &out_cols, &out_partition);
	}
	if (out_rows * out_cols != output)
		return -1;
	return 0;
}

#endif

static void _ccv_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows, cols, partition;
	_ccv_convnet_layer_deduce_output_format(layer, &rows, &cols, &partition);
	int ch = layer->net.convolutional.channels;
	int count = layer->net.convolutional.count;
	int strides = layer->net.convolutional.strides;
	int border = layer->net.convolutional.border;
	int kernel_rows = layer->net.convolutional.rows;
	int kernel_cols = layer->net.convolutional.cols;
	int type = CCV_32F | count;
	assert(CCV_GET_CHANNEL(a->type) == ch);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int count_per_partition = count / partition;
	int ch_per_partition = ch / partition;
#ifdef USE_DISPATCH
	dispatch_apply(count, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t k) {
#else
	int k;
	for (k = 0; k < count; k++)
	{
#endif
		int i, j, x, y, c;
		int p = k / count_per_partition;
		float* ap = a->data.f32 + p * ch_per_partition;
		float* bp = db->data.f32 + k;
		float* layer_w = layer->w + k * kernel_rows * kernel_cols * ch_per_partition;
		float bias = layer->bias[k];
		for (i = 0; i < db->rows; i++)
		{
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border);
			int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(a->rows + border, i * strides + kernel_rows));
			comy *= ch_per_partition * kernel_cols;
			for (j = 0; j < db->cols; j++)
			{
				float v = bias;
				int comx = ccv_max(j * strides - border, 0) - (j * strides - border);
				int maxx = kernel_cols - comx - (j * strides + kernel_cols - ccv_min(a->cols + border, j * strides + kernel_cols));
				float* w = layer_w + comx * ch_per_partition + comy;
				float* apz = ap + ccv_max(j * strides - border, 0) * ch;
				// when we have border, we simply do zero padding
				for (y = 0; y < maxy; y++)
				{
					for (x = 0; x < maxx; x++)
						for (c = 0; c < ch_per_partition; c++)
							v += w[x * ch_per_partition + c] * apz[x * ch + c];
					w += kernel_cols * ch_per_partition;
					apz += a->cols * ch;
				}
				bp[j * count] = ccv_max(0, v); // ReLU
			}
			bp += db->cols * count;
			ap += a->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0));
		}
#ifdef USE_DISPATCH
	});
#else
	}
#endif
}

static void _ccv_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, layer->net.full_connect.count, 1, CCV_32F | CCV_C1, CCV_32F | CCV_C1, 0);
	int ch = CCV_GET_CHANNEL(a->type);
	int rows = a->rows, cols = a->cols;
	// reshape a for gemm
	assert(a->step == a->cols * CCV_GET_DATA_TYPE_SIZE(a->type) * ch);
	a->rows = rows * cols * ch, a->cols = 1, a->type = (a->type - ch) | CCV_C1;
	assert(a->rows * db->rows == layer->wnum);
	a->step = a->cols * CCV_GET_DATA_TYPE_SIZE(a->type);
	int i;
	float* bptr = db->data.f32;
	for (i = 0; i < db->rows; i++)
		bptr[i] = layer->bias[i];
	ccv_dense_matrix_t dw = ccv_dense_matrix(db->rows, a->rows, CCV_32F | CCV_C1, layer->w, 0);
	ccv_gemm(&dw, a, 1, db, 1, 0, (ccv_matrix_t**)&db, 0); // supply db as matrix C is allowed
	a->rows = rows, a->cols = cols, a->type = (a->type - CCV_GET_CHANNEL(a->type)) | ch;
	a->step = a->cols * CCV_GET_DATA_TYPE_SIZE(a->type) * CCV_GET_CHANNEL(a->type);
}

static void _ccv_convnet_rnorm_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, ccv_dense_matrix_t** denoms)
{
	int rows, cols, partition;
	_ccv_convnet_layer_deduce_output_format(layer, &rows, &cols, &partition);
	int size = layer->net.rnorm.size;
	float kappa = layer->net.rnorm.kappa;
	float alpha = layer->net.rnorm.alpha;
	float beta = layer->net.rnorm.beta;
	int way = size / 2;
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int ch = CCV_GET_CHANNEL(a->type);
	int type = CCV_32F | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, k, x, p;
	float* ap = a->data.f32;
	float* bp = db->data.f32;
	int ch_per_partition = ch / partition;
	if (denoms)
	{
		ccv_dense_matrix_t* ddenoms = *denoms = ccv_dense_matrix_renew(*denoms, rows, cols, type, type, 0);
		float* dp = ddenoms->data.f32;
		for (i = 0; i < db->rows; i++)
		{
			for (j = 0; j < db->cols; j++)
				for (p = 0; p < partition; p++)
					for (k = 0; k < ch_per_partition; k++)
					{
						float v = ap[j * ch + p * ch_per_partition + k];
						float denom = 0;
						for (x = ccv_max(k - way, 0); x <= ccv_min(k + way, ch_per_partition - 1); x++)
							denom += ap[j * ch + p * ch_per_partition + x] * ap[j * ch + p * ch_per_partition + x];
						denom = kappa + alpha * denom;
						dp[j * ch + p * ch_per_partition + k] = denom;
						bp[j * ch + p * ch_per_partition + k] = v * powf(denom, -beta);
					}
			ap += a->cols * ch;
			dp += ddenoms->cols * ch;
			bp += db->cols * ch;
		}
	} else {
		for (i = 0; i < db->rows; i++)
		{
			for (j = 0; j < db->cols; j++)
				for (p = 0; p < partition; p++)
					for (k = 0; k < ch_per_partition; k++)
					{
						float v = ap[j * ch + p * ch_per_partition + k];
						float denom = 0;
						for (x = ccv_max(k - way, 0); x <= ccv_min(k + way, ch_per_partition - 1); x++)
							denom += ap[j * ch + p * ch_per_partition + x] * ap[j * ch + p * ch_per_partition + x];
						denom = kappa + alpha * denom;
						bp[j * ch + p * ch_per_partition + k] = v * powf(denom, -beta);
					}
			ap += a->cols * ch;
			bp += db->cols * ch;
		}
	}
}

static void _ccv_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows, cols, partition;
	_ccv_convnet_layer_deduce_output_format(layer, &rows, &cols, &partition);
	int size = layer->net.pool.size;
	int strides = layer->net.pool.strides;
	int border = layer->net.pool.border;
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int ch = CCV_GET_CHANNEL(a->type);
	int type = CCV_32F | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, k, x, y;
	float* ap = a->data.f32;
	float* bp = db->data.f32;
	for (i = 0; i < db->rows; i++)
	{
		const int start_y = ccv_max(i * strides - border, 0) - (i * strides - border);
		const int end_y = size + ccv_min(i * strides + size - border, a->rows) - (i * strides + size - border);
		for (j = 0; j < db->cols; j++)
		{
			const int start_x = ccv_max(j * strides - border, 0) - (j * strides - border);
			const int end_x = size + ccv_min(j * strides + size - border, a->cols) - (j * strides + size - border);
			for (k = 0; k < ch; k++)
			{
				float v = 0;
				for (y = start_y; y < end_y; y++)
					for (x = start_x; x < end_x; x++)
						if (x == start_x && y == start_y)
							v = ap[(j * strides - border + x + (y - border) * a->cols) * ch + k];
						else if (ap[(j * strides - border + x + (y - border) * a->cols) * ch + k] > v)
							v = ap[(j * strides - border + x + (y - border) * a->cols) * ch + k];
				bp[j * ch + k] = v;
			}
		}
		ap += a->cols * ch * strides;
		bp += db->cols * ch;
	}
}

static void _ccv_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows, cols, partition;
	_ccv_convnet_layer_deduce_output_format(layer, &rows, &cols, &partition);
	int size = layer->net.pool.size;
	int strides = layer->net.pool.strides;
	int border = layer->net.pool.border;
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int ch = CCV_GET_CHANNEL(a->type);
	int type = CCV_32F | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, k, x, y;
	float* ap = a->data.f32;
	float* bp = db->data.f32;
	for (i = 0; i < db->rows; i++)
	{
		const int start_y = ccv_max(i * strides - border, 0) - (i * strides - border);
		const int end_y = size + ccv_min(i * strides + size - border, a->rows) - (i * strides + size - border);
		for (j = 0; j < db->cols; j++)
		{
			const int start_x = ccv_max(j * strides - border, 0) - (j * strides - border);
			const int end_x = size + ccv_min(j * strides + size - border, a->cols) - (j * strides + size - border);
			for (k = 0; k < ch; k++)
			{
				float v = 0;
				for (y = start_y; y < end_y; y++)
					for (x = start_x; x < end_x; x++)
						v += ap[(j * strides - border + x + (y - border) * a->cols) * ch + k];
				bp[j * ch + k] = v / ((end_x - start_x) * (end_y - start_y));
			}
		}
		ap += a->cols * ch * strides;
		bp += db->cols * ch;
	}
}

static void _ccv_convnet_compute_softmax(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	int ch = CCV_GET_CHANNEL(a->type);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_32F | ch, CCV_32F | ch, 0);
	int i;
	float* aptr = a->data.f32;
	float* bptr = db->data.f32;
	double max = aptr[0];
	for (i = 1; i < a->rows * a->cols * ch; i++)
		if (aptr[i] > max)
			max = aptr[i];
	double tt = 0;
	for (i = 0; i < a->rows * a->cols * ch; i++)
		tt += (bptr[i] = expf(aptr[i] - max));
	tt = 1.0 / tt;
	for (i = 0; i < a->rows * a->cols * ch; i++)
		bptr[i] *= tt;
}

static void _ccv_convnet_layer_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, ccv_dense_matrix_t** denoms)
{
	switch(layer->type)
	{
		case CCV_CONVNET_CONVOLUTIONAL:
			_ccv_convnet_convolutional_forward_propagate(layer, a, b);
			break;
		case CCV_CONVNET_FULL_CONNECT:
			_ccv_convnet_full_connect_forward_propagate(layer, a, b);
			break;
		case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			_ccv_convnet_rnorm_forward_propagate(layer, a, b, denoms);
			break;
		case CCV_CONVNET_MAX_POOL:
			_ccv_convnet_max_pool_forward_propagate(layer, a, b);
			break;
		case CCV_CONVNET_AVERAGE_POOL:
			_ccv_convnet_average_pool_forward_propagate(layer, a, b);
			break;
	}
}

static void _ccv_convnet_full_connect_forward_propagate_parallel(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, layer->net.full_connect.count, CCV_32F | CCV_C1, CCV_32F | CCV_C1, 0);
	// reshape a for gemm
	int i, j;
	float* bptr = db->data.f32;
	for (i = 0; i < db->rows; i++)
	{
		for (j = 0; j < db->cols; j++)
			bptr[j] = layer->bias[j];
		bptr += db->cols;
	}
	ccv_dense_matrix_t dw = ccv_dense_matrix(db->cols, a->cols, CCV_32F | CCV_C1, layer->w, 0);
	ccv_gemm(a, &dw, 1, db, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&db, 0); // supply db as matrix C is allowed
}

static void _ccv_convnet_compute_softmax_parallel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, 1, a->cols, CCV_32F | CCV_C1, CCV_32F | CCV_C1, 0);
	ccv_zero(db);
	int i, j;
	float* aptr = a->data.f32;
	float* bptr = db->data.f32;
	float* cptr = (float*)ccmalloc(sizeof(float) * a->cols);
	for (i = 0; i < a->rows; i++)
	{
		double max = aptr[0];
		for (j = 1; j < a->cols; j++)
			if (aptr[j] > max)
				max = aptr[j];
		double tt = 0;
		for (j = 0; j < a->cols; j++)
			tt += (cptr[j] = expf(aptr[j] - max));
		tt = 1.0 / tt;
		for (j = 0; j < a->cols; j++)
			bptr[j] += cptr[j] * tt;
		aptr += a->cols;
	}
	ccfree(cptr);
}

#ifndef CASE_TESTS

void ccv_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch)
{
#ifdef HAVE_CUDA
	if (convnet->use_cwc_accel)
		cwc_convnet_encode(convnet, a, b, batch);
	else {
#endif
	assert(batch == 1);
	assert(CCV_GET_CHANNEL((*a)->type) == convnet->channels);
	assert((*a)->rows == convnet->rows);
	assert((*a)->cols == convnet->cols);
	int i;
	// save the last layer of neuron cache in case that we encode to a different matrix
	ccv_dense_matrix_t* out_neuron = convnet->acts[convnet->count - 1];
	convnet->acts[convnet->count - 1] = *b;
	_ccv_convnet_layer_forward_propagate(convnet->layers, *a, convnet->acts, convnet->denoms);
	for (i = 1; i < convnet->count; i++)
		_ccv_convnet_layer_forward_propagate(convnet->layers + i, convnet->acts[i - 1], convnet->acts + i, convnet->denoms + i);
	if (convnet->acts + convnet->count - 1 != b)
	{
		*b = convnet->acts[convnet->count - 1];
		// restore the last layer of neuron cache
		convnet->acts[convnet->count - 1] = out_neuron;
	}
#ifdef HAVE_CUDA
	}
#endif
}

void ccv_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int symmetric, ccv_array_t** ranks, int tops, int batch)
{
#ifdef HAVE_CUDA
	if (convnet->use_cwc_accel)
		cwc_convnet_classify(convnet, a, symmetric, ranks, tops, batch);
	else {
#endif
	int i, j, k;
	ccv_dense_matrix_t** b = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (convnet->count + 1));
	memset(b, 0, sizeof(ccv_dense_matrix_t*) * (convnet->count + 1));
	int last = -1;
	for (i = 0; i < convnet->count; i++)
		// find the first full connect layer
		if (convnet->layers[i].type == CCV_CONVNET_FULL_CONNECT)
		{
			last = i;
			break;
		}
	int second = last;
	assert(last >= 0);
	for (i = last - 1; i >= 0; i--)
		// find the last convolutional layer
		if (convnet->layers[i].type == CCV_CONVNET_CONVOLUTIONAL)
		{
			last = i + 1;
			break;
		}
	assert(last >= 0 && last < convnet->count);
	for (i = 0; i < batch; i++)
	{
		assert(CCV_GET_CHANNEL(a[i]->type) == convnet->channels);
		assert(a[i]->rows == convnet->input.height);
		assert(a[i]->cols == convnet->input.width);
		ccv_subtract(a[i], convnet->mean_activity, (ccv_matrix_t**)b, CCV_32F);
		// doing the first few layers until the first full connect layer
		int rows, cols;
		int previous_rows = convnet->input.height;
		int previous_cols = convnet->input.width;
		for (j = 0; j < last; j++)
		{
			ccv_convnet_layer_t* layer = convnet->layers + j;
			rows = layer->input.matrix.rows;
			cols = layer->input.matrix.cols;
			layer->input.matrix.rows = previous_rows;
			layer->input.matrix.cols = previous_cols;
			_ccv_convnet_layer_forward_propagate(layer, b[j], b + j + 1, 0);
			int partition;
			_ccv_convnet_layer_deduce_output_format(layer, &previous_rows, &previous_cols, &partition);
			layer->input.matrix.rows = rows;
			layer->input.matrix.cols = cols;
			ccv_matrix_free(b[j]);
			b[j] = 0;
		}
		int c = (!!symmetric + 1) * 5;
		ccv_convnet_layer_t* start_layer = convnet->layers + last;
		int d[5][2] = {
			{(b[last]->cols - start_layer->input.matrix.cols) / 2, (b[last]->rows - start_layer->input.matrix.rows) / 2}, // center
			{0, 0}, // left top corner
			{b[last]->cols - start_layer->input.matrix.cols, 0}, // right top corner
			{0, b[last]->rows - start_layer->input.matrix.rows}, // left bottom corner
			{b[last]->cols - start_layer->input.matrix.cols, b[last]->rows - start_layer->input.matrix.rows}, // right bottom corner
		};
		ccv_dense_matrix_t* multi = ccv_dense_matrix_new(c, convnet->layers[second].input.node.count, CCV_32F | CCV_C1, 0, 0);
		// for the last convolutional layer, we sample the layer at different locations (and horizontal mirrors), and average all of them
		for (k = 0; k < 5; k++)
		{
			int x = d[k][0], y = d[k][1];
			ccv_dense_matrix_t* input = 0;
			ccv_slice(b[last], (ccv_matrix_t**)&input, CCV_32F, y, x, start_layer->input.matrix.rows, start_layer->input.matrix.cols);
			ccv_dense_matrix_t* full = b[last];
			b[last] = input;
			b[second] = ccv_dense_matrix_new(convnet->layers[second].input.matrix.rows, convnet->layers[second].input.matrix.cols, CCV_NO_DATA_ALLOC | CCV_32F | convnet->layers[second].input.matrix.channels, multi->data.f32 + k * convnet->layers[second].input.node.count, 0);
			for (j = last; j < second; j++)
			{
				ccv_convnet_layer_t* layer = convnet->layers + j;
				_ccv_convnet_layer_forward_propagate(layer, b[j], b + j + 1, 0);
				if (j > last)
				{
					ccv_matrix_free(b[j]);
					b[j] = 0;
				}
			}
			ccv_matrix_free(b[second]);
			b[second] = 0;
			if (symmetric)
			{
				ccv_flip(input, &input, 0, CCV_FLIP_X);
				b[second] = ccv_dense_matrix_new(convnet->layers[second].input.matrix.rows, convnet->layers[second].input.matrix.cols, CCV_NO_DATA_ALLOC | CCV_32F | convnet->layers[second].input.matrix.channels, multi->data.f32 + (k + 5) * convnet->layers[second].input.node.count, 0);
				// horizontal mirroring
				for (j = last; j < second; j++)
				{
					ccv_convnet_layer_t* layer = convnet->layers + j;
					_ccv_convnet_layer_forward_propagate(layer, b[j], b + j + 1, 0);
					if (j > last)
					{
						ccv_matrix_free(b[j]);
						b[j] = 0;
					}
				}
				ccv_matrix_free(b[second]);
				b[second] = 0;
			}
			ccv_matrix_free(input);
			b[last] = full;
		}
		ccv_matrix_free(b[last]);
		b[last] = 0;
		b[second] = multi;
		for (j = second; j < convnet->count; j++)
		{
			ccv_convnet_layer_t* layer = convnet->layers + j;
			assert(layer->type == CCV_CONVNET_FULL_CONNECT);
			_ccv_convnet_full_connect_forward_propagate_parallel(layer, b[j], b + j + 1);
			ccv_matrix_free(b[j]);
			b[j] = 0;
		}
		ccv_dense_matrix_t* softmax = 0;
		_ccv_convnet_compute_softmax_parallel(b[convnet->count], &softmax, 0);
		ccv_matrix_free(b[convnet->count]);
		b[convnet->count] = 0;
		ranks[i] = ccv_array_new(sizeof(ccv_classification_t), tops, 0);
		float* r = softmax->data.f32;
		assert(tops <= softmax->cols);
		for (j = 0; j < tops; j++)
		{
			float maxr = -1;
			int id = -1;
			for (k = 0; k < softmax->cols; k++)
				if (r[k] >= 0 && r[k] > maxr)
					maxr = r[k], id = k;
			assert(id >= 0);
			r[id] = -1;
			ccv_classification_t classification = {
				.id = id,
				.confidence = maxr / c,
			};
			ccv_array_push(ranks[i], &classification);
		}
		ccv_matrix_free(softmax);
	}
#ifdef HAVE_CUDA
	}
#endif
}

#endif

#ifdef HAVE_GSL

// compute back propagated gradient & weight update delta
static void _ccv_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* n, ccv_dense_matrix_t* m, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop).
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// note that y (the output from forward prop) is not included because the full connect net is simple enough that we don't need it
	int rows, cols, partition;
	_ccv_convnet_layer_deduce_output_format(layer, &rows, &cols, &partition);
	int ch = layer->net.convolutional.channels;
	int count = layer->net.convolutional.count;
	int strides = layer->net.convolutional.strides;
	int border = layer->net.convolutional.border;
	int kernel_rows = layer->net.convolutional.rows;
	int kernel_cols = layer->net.convolutional.cols;
	assert(a->rows == rows);
	assert(a->cols == cols);
	assert(CCV_GET_CHANNEL(a->type) == count);
	int a_rows = a->rows, a_cols = a->cols, a_ch = CCV_GET_CHANNEL(a->type);
	a->rows = rows, a->cols = cols, a->type = (a->type - a_ch) | count;
	assert(CCV_GET_CHANNEL(m->type) == ch);
	assert(CCV_GET_DATA_TYPE(m->type) == CCV_32F);
	int count_per_partition = count / partition;
	int ch_per_partition = ch / partition;
	// update weight gradient
#ifdef USE_DISPATCH
	dispatch_apply(count, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t k) {
#else
	int k;
	for (k = 0; k < count; k++)
	{
#endif
		int i, j, x, y, c;
		int p = k / count_per_partition;
		float* mp = m->data.f32 + p * ch_per_partition;
		float* ap = a->data.f32 + k;
		float* np = n->data.f32 + k;
		float* update_w = update_params->w + k * kernel_rows * kernel_cols * ch_per_partition;
		float bias = 0;
		for (i = 0; i < rows; i++)
		{
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border);
			int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(m->rows + border, i * strides + kernel_rows));
			comy *= ch_per_partition * kernel_cols;
			for (j = 0; j < cols; j++)
			{
				if (np[j * count] > 0)
				{ /* when np is bigger than 0, relu continues to update the weight, otherwise it stops */
					float v = ap[j * count];
					bias += v;
					int comx = ccv_max(j * strides - border, 0) - (j * strides - border);
					int maxx = kernel_cols - comx - (j * strides + kernel_cols - ccv_min(m->cols + border, j * strides + kernel_cols));
					float* w = update_w + comx * ch_per_partition + comy;
					float* mpz = mp + ccv_max(j * strides - border, 0) * ch;
					/* when we have border, we simply do zero padding */
					for (y = 0; y < maxy; y++)
					{
						for (x = 0; x < maxx; x++)
							for (c = 0; c < ch_per_partition; c++)
								w[x * ch_per_partition + c] += v * mpz[x * ch + c];
						w += kernel_cols * ch_per_partition;
						mpz += m->cols * ch;
					}
				}
			}
			ap += a->cols * count;
			np += n->cols * count;
			mp += m->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0));
		}
		update_params->bias[k] = bias;
#ifdef USE_DISPATCH
	});
#else
	}
#endif
	if (b)
	{
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, m->rows, m->cols, CCV_32F | CCV_GET_CHANNEL(m->type), CCV_32F | CCV_GET_CHANNEL(m->type), 0);
		// clear it up before propagate result
		ccv_zero(db);
		int k;
		for (k = 0; k < count; k++)
		{
			int i, j, x, y, c;
			int p = k / count_per_partition;
			float* bp = db->data.f32 + p * ch_per_partition;
			float* ap = a->data.f32 + k;
			float* np = n->data.f32 + k;
			float* layer_w = layer->w + k * kernel_rows * kernel_cols * ch_per_partition;
			for (i = 0; i < rows; i++)
			{
				int comy = ccv_max(i * strides - border, 0) - (i * strides - border);
				int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(db->rows + border, i * strides + kernel_rows));
				comy *= ch_per_partition * kernel_cols;
				for (j = 0; j < cols; j++)
				{
					if (np[j * count] > 0)
					{ /* when np is bigger than 0, relu continues to update the weight, otherwise it stops */
						float v = ap[j * count];
						int comx = ccv_max(j * strides - border, 0) - (j * strides - border);
						int maxx = kernel_cols - comx - (j * strides + kernel_cols - ccv_min(db->cols + border, j * strides + kernel_cols));
						float* w = layer_w + comx * ch_per_partition + comy;
						float* bpz = bp + ccv_max(j * strides - border, 0) * ch;
						/* when we have border, we simply do zero padding */
						for (y = 0; y < maxy; y++)
						{
							for (x = 0; x < maxx; x++)
								for (c = 0; c < ch_per_partition; c++)
									bpz[x * ch + c] += v * w[x * ch_per_partition + c];
							w += kernel_cols * ch_per_partition;
							bpz += db->cols * ch;
						}
					}
				}
				ap += a->cols * count;
				np += n->cols * count;
				bp += db->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0));
			}
		}
	}
	a->rows = a_rows, a->cols = a_cols, a->type = (a->type - CCV_GET_CHANNEL(a->type)) | a_ch;
}

static void _ccv_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* x, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), d is the dropout,
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// note that y (the output from forward prop) is not included because the full connect net is simple enough that we don't need it
	ccv_dense_matrix_t* db = 0;
	if (b)
		db = *b = ccv_dense_matrix_renew(*b, x->rows, x->cols, CCV_32F | CCV_GET_CHANNEL(x->type), CCV_32F | CCV_GET_CHANNEL(x->type), 0);
	int x_rows = x->rows, x_cols = x->cols, x_ch = CCV_GET_CHANNEL(x->type);
	x->rows = x_rows * x_cols * x_ch, x->cols = 1, x->type = (x->type - x_ch) | CCV_C1;
	x->step = x->cols * CCV_GET_DATA_TYPE_SIZE(x->type);
	ccv_dense_matrix_t w = ccv_dense_matrix(a->rows, x->rows, CCV_32F | CCV_C1, update_params->w, 0);
	ccv_dense_matrix_t* dw = &w;
	// compute bias gradient
	ccv_dense_matrix_t bias = ccv_dense_matrix(a->rows, 1, CCV_32F | CCV_C1, update_params->bias, 0);
	ccv_dense_matrix_t* dbias = &bias;
	ccv_add(a, dbias, (ccv_matrix_t**)&dbias, 0);
	// compute weight gradient
	ccv_gemm(a, x, 1, dw, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&dw, 0);
	w = ccv_dense_matrix(a->rows, x->rows, CCV_32F | CCV_C1, layer->w, 0);
	// propagate error
	if (db)
	{
		db->rows = x->rows, db->cols = x->cols, db->type = (db->type - x_ch) | CCV_C1;
		db->step = db->cols * CCV_GET_DATA_TYPE_SIZE(db->type);
		ccv_gemm(&w, a, 1, 0, 0, CCV_A_TRANSPOSE, (ccv_matrix_t**)&db, 0);
		db->rows = x_rows, db->cols = x_cols, db->type = (db->type - CCV_GET_CHANNEL(db->type)) | x_ch;
		db->step = db->cols * CCV_GET_DATA_TYPE_SIZE(db->type) * CCV_GET_CHANNEL(db->type);
	}
	x->rows = x_rows, x->cols = x_cols, x->type = (x->type - CCV_GET_CHANNEL(x->type)) | x_ch;
	x->step = x->cols * CCV_GET_DATA_TYPE_SIZE(x->type) * CCV_GET_CHANNEL(x->type);
}

static void _ccv_convnet_rnorm_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* n, ccv_dense_matrix_t* m, ccv_dense_matrix_t* denoms, ccv_dense_matrix_t** b)
{
	int rows, cols, partition;
	_ccv_convnet_layer_deduce_output_format(layer, &rows, &cols, &partition);
	int size = layer->net.rnorm.size;
	float alpha = layer->net.rnorm.alpha;
	float beta = layer->net.rnorm.beta;
	int way = size / 2;
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	int ch = CCV_GET_CHANNEL(a->type);
	int type = CCV_32F | ch;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, type, type, 0);
	int i, j, k, x, p;
	float* ap = a->data.f32;
	float* np = n->data.f32;
	float* mp = m->data.f32;
	float* dp = denoms->data.f32;
	float* bp = db->data.f32;
	int ch_per_partition = ch / partition;
	for (i = 0; i < db->rows; i++)
	{
		for (j = 0; j < db->cols; j++)
			for (p = 0; p < partition; p++)
				for (k = 0; k < ch_per_partition; k++)
				{
					float nom = 0;
					for (x = ccv_max(k - way, 0); x <= ccv_min(k + way, ch_per_partition - 1); x++)
						nom += -2 * alpha * beta * ap[j * ch + x + p * ch_per_partition] * np[j * ch + x + p * ch_per_partition] / dp[j * ch + x + p * ch_per_partition];
					bp[j * ch + k + p * ch_per_partition] = mp[j * ch + k + p * ch_per_partition] * nom + ap[j * ch + k + p * ch_per_partition] * powf(dp[j * ch + k + p * ch_per_partition], -beta);
				}
		ap += a->cols * ch;
		np += n->cols * ch;
		mp += m->cols * ch;
		dp += denoms->cols * ch;
		bp += db->cols * ch;
	}
}

static void _ccv_convnet_max_pool_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* n, ccv_dense_matrix_t* m, ccv_dense_matrix_t** b)
{
	// a is the input gradient (for back prop), y is the output (from forward prop),
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// pooling layer doesn't need the dropout
	if (b)
	{
		assert(CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(n->type));
		assert(CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(m->type));
		int ch = CCV_GET_CHANNEL(a->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, m->rows, m->cols, CCV_32F | ch, CCV_32F | ch, 0);
		ccv_zero(db);
		int size = layer->net.pool.size;
		int strides = layer->net.pool.strides;
		int border = layer->net.pool.border;
		int i, j, k, x, y;
		float* ap = a->data.f32;
		float* bp = db->data.f32;
		float* np = n->data.f32;
		float* mp = m->data.f32;
		for (i = 0; i < a->rows; i++)
		{
			const int start_y = ccv_max(i * strides - border, 0) - (i * strides - border);
			const int end_y = size + ccv_min(i * strides + size - border, db->rows) - (i * strides + size - border);
			for (j = 0; j < a->cols; j++)
			{
				const int start_x = ccv_max(j * strides - border, 0) - (j * strides - border);
				const int end_x = size + ccv_min(j * strides + size - border, db->cols) - (j * strides + size - border);
				for (k = 0; k < ch; k++)
				{
					float v = np[j * ch + k];
					float u = ap[j * ch + k];
					for (y = start_y; y < end_y; y++)
						for (x = start_x; x < end_x; x++)
							// we have to do direct comparison otherwise it will contribute to too many cells
							// and the propagation won't work. But CPU will have different result comparing with GPU
							if (mp[(j * strides - border + x + (y - border) * m->cols) * ch + k] == v)
								bp[(j * strides - border + x + (y - border) * db->cols) * ch + k] += u;
				}
			}
			ap += a->cols * ch;
			np += n->cols * ch;
			bp += db->cols * ch * strides;
			mp += m->cols * ch * strides;
		}
	}
}

static void _ccv_convnet_average_pool_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* m, ccv_dense_matrix_t** b)
{
	// a is the input gradient (for back prop), y is the output (from forward prop),
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// pooling layer doesn't need the dropout
	if (b)
	{
		assert(CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(m->type));
		int ch = CCV_GET_CHANNEL(a->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, m->rows, m->cols, CCV_32F | ch, CCV_32F | ch, 0);
		ccv_zero(db);
		int size = layer->net.pool.size;
		int strides = layer->net.pool.strides;
		int border = layer->net.pool.border;
		int i, j, k, x, y;
		float* ap = a->data.f32;
		float* bp = db->data.f32;
		for (i = 0; i < a->rows; i++)
		{
			const int start_y = ccv_max(i * strides - border, 0) - (i * strides - border);
			const int end_y = size + ccv_min(i * strides + size - border, db->rows) - (i * strides + size - border);
			for (j = 0; j < a->cols; j++)
			{
				const int start_x = ccv_max(j * strides - border, 0) - (j * strides - border);
				const int end_x = size + ccv_min(j * strides + size - border, db->cols) - (j * strides + size - border);
				for (k = 0; k < ch; k++)
				{
					float u = ap[j * ch + k] / ((end_x - start_x) * (end_y - start_y));
					for (y = start_y; y < end_y; y++)
						for (x = start_x; x < end_x; x++)
							bp[(j * strides - border + x + (y - border) * db->cols) * ch + k] += u;
				}
			}
			ap += a->cols * ch;
			bp += db->cols * ch * strides;
		}
	}
}

static void _ccv_convnet_propagate_loss(ccv_convnet_t* convnet, ccv_dense_matrix_t* a, ccv_dense_matrix_t* dloss, ccv_convnet_t* update_params)
{
	int i;
	ccv_convnet_layer_t* layer = convnet->layers + convnet->count - 1;
	assert(layer->type == CCV_CONVNET_FULL_CONNECT); // the last layer has too be a full connect one to generate softmax result
	_ccv_convnet_full_connect_backward_propagate(layer, dloss, convnet->acts[convnet->count - 2], convnet->count - 1 > 0 ? update_params->acts + convnet->count - 2 : 0, update_params->layers + convnet->count - 1);
	for (i = convnet->count - 2; i >= 0; i--)
	{
		layer = convnet->layers + i;
		switch (layer->type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				_ccv_convnet_convolutional_backward_propagate(layer, update_params->acts[i], convnet->acts[i], i > 0 ? convnet->acts[i - 1] : a, i > 0 ? update_params->acts + i - 1 : 0, update_params->layers + i);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				_ccv_convnet_full_connect_backward_propagate(layer, update_params->acts[i], i > 0 ? convnet->acts[i - 1] : a, i > 0 ? update_params->acts + i - 1 : 0, update_params->layers + i);
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
				_ccv_convnet_rnorm_backward_propagate(layer, update_params->acts[i], convnet->acts[i], i > 0 ? convnet->acts[i - 1] : a, convnet->denoms[i], i > 0 ? update_params->acts + i - 1 : 0);
				break;
			case CCV_CONVNET_MAX_POOL:
				_ccv_convnet_max_pool_backward_propagate(layer, update_params->acts[i], convnet->acts[i], i > 0 ? convnet->acts[i - 1] : a, i > 0 ? update_params->acts + i - 1 : 0);
				break;
			case CCV_CONVNET_AVERAGE_POOL:
				_ccv_convnet_average_pool_backward_propagate(layer, update_params->acts[i], i > 0 ? convnet->acts[i - 1] : a, i > 0 ? update_params->acts + i - 1 : 0);
				break;
		}
	}
}

static void _ccv_convnet_update(ccv_convnet_t* convnet, ccv_convnet_t* momentum, ccv_convnet_t* update_params, ccv_convnet_layer_train_param_t* layer_params)
{
	int i, j;
	for (i = 0; i < convnet->count; i++)
		switch (update_params->layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
			{
				float* w = convnet->layers[i].w;
				float* vw = momentum->layers[i].w;
				float* dw = update_params->layers[i].w;
				for (j = 0; j < convnet->layers[i].wnum; j++)
				{
					vw[j] = layer_params[i].w.momentum * vw[j] - layer_params[i].w.decay * layer_params[i].w.learn_rate * w[j] + layer_params[i].w.learn_rate * dw[j];
					w[j] += vw[j];
				}
				float* bias = convnet->layers[i].bias;
				float* vbias = momentum->layers[i].bias;
				float* dbias = update_params->layers[i].bias;
				for (j = 0; j < convnet->layers[i].net.convolutional.count; j++)
				{
					vbias[j] = layer_params[i].bias.momentum * vbias[j] - layer_params[i].bias.decay * layer_params[i].bias.learn_rate * bias[j] + layer_params[i].bias.learn_rate * dbias[j];
					bias[j] += vbias[j];
				}
				break;
			}
			case CCV_CONVNET_FULL_CONNECT:
			{
				float* w = convnet->layers[i].w;
				float* vw = momentum->layers[i].w;
				float* dw = update_params->layers[i].w;
				for (j = 0; j < convnet->layers[i].wnum; j++)
				{
					vw[j] = layer_params[i].w.momentum * vw[j] - layer_params[i].w.decay * layer_params[i].w.learn_rate * w[j] + layer_params[i].w.learn_rate * dw[j];
					w[j] += vw[j];
				}
				float* bias = convnet->layers[i].bias;
				float* vbias = momentum->layers[i].bias;
				float* dbias = update_params->layers[i].bias;
				for (j = 0; j < convnet->layers[i].net.full_connect.count; j++)
				{
					vbias[j] = layer_params[i].bias.momentum * vbias[j] - layer_params[i].bias.decay * layer_params[i].bias.learn_rate * bias[j] + layer_params[i].bias.learn_rate * dbias[j];
					bias[j] += vbias[j];
				}
				break;
			}
		}
}

static void _ccv_convnet_update_zero(ccv_convnet_t* update_params)
{
	int i;
	for (i = 0; i < update_params->count; i++)
		switch (update_params->layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				memset(update_params->layers[i].w, 0, sizeof(float) * update_params->layers[i].wnum);
				memset(update_params->layers[i].bias, 0, sizeof(float) * update_params->layers[i].net.convolutional.count);
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(update_params->layers[i].wnum % update_params->layers[i].net.full_connect.count == 0);
				memset(update_params->layers[i].w, 0, sizeof(float) * update_params->layers[i].wnum);
				memset(update_params->layers[i].bias, 0, sizeof(float) * update_params->layers[i].net.full_connect.count);
				break;
		}
}

static ccv_convnet_t* _ccv_convnet_update_new(ccv_convnet_t* convnet)
{
	ccv_convnet_t* update_params = (ccv_convnet_t*)ccmalloc(sizeof(ccv_convnet_t) + sizeof(ccv_convnet_layer_t) * convnet->count + sizeof(ccv_dense_matrix_t*) * convnet->count);
	update_params->reserved = 0;
	update_params->layers = (ccv_convnet_layer_t*)(update_params + 1);
	update_params->acts = (ccv_dense_matrix_t**)(update_params->layers + convnet->count);
	memset(update_params->acts, 0, sizeof(ccv_dense_matrix_t*) * convnet->count);
	update_params->denoms = 0;
	update_params->input = convnet->input;
	update_params->rows = convnet->rows;
	update_params->cols = convnet->cols;
	update_params->count = convnet->count;
	update_params->channels = convnet->channels;
	update_params->mean_activity = 0;
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		update_params->layers[i].type = convnet->layers[i].type;
		update_params->layers[i].net = convnet->layers[i].net;
		update_params->layers[i].wnum = convnet->layers[i].wnum;
		switch (update_params->layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				update_params->layers[i].w = (float*)cccalloc(sizeof(float), update_params->layers[i].wnum + update_params->layers[i].net.convolutional.count);
				update_params->layers[i].bias = update_params->layers[i].w + update_params->layers[i].wnum;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(update_params->layers[i].wnum % update_params->layers[i].net.full_connect.count == 0);
				update_params->layers[i].w = (float*)cccalloc(sizeof(float), update_params->layers[i].wnum + update_params->layers[i].net.full_connect.count);
				update_params->layers[i].bias = update_params->layers[i].w + update_params->layers[i].wnum;
				break;
			case CCV_CONVNET_LOCAL_RESPONSE_NORM:
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				update_params->layers[i].w = 0;
				update_params->layers[i].bias = 0;
				break;
		}
	}
	return update_params;
}

static void _ccv_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch)
{
	assert(batch == 1);
	ccv_convnet_encode(convnet, a, convnet->acts + convnet->count - 1, 1);
	int i, c = 0;
	ccv_dense_matrix_t* b = convnet->acts[convnet->count - 1];
	int maxc = b->data.f32[0];
	for (i = 1; i < b->rows; i++)
		if (b->data.f32[i] > maxc)
			maxc = b->data.f32[i], c = i;
	labels[0] = c;
}

#endif

#ifndef CASE_TESTS

void ccv_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, const char* filename, ccv_convnet_train_param_t params)
{
#ifdef HAVE_GSL
#ifdef HAVE_CUDA
	if (convnet->use_cwc_accel)
		cwc_convnet_supervised_train(convnet, categorizeds, tests, filename, params);
	else {
#endif
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
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	ccv_convnet_t* momentum = _ccv_convnet_update_new(convnet);
	for (t = 0; t < params.max_epoch; t++)
	{
		for (i = 0; i < aligned_rnum; i++)
		{
			// dropout the first hidden layer
			ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx[i]);
			ccv_convnet_encode(convnet, &categorized->matrix, convnet->acts + convnet->count - 1, 1);
			ccv_dense_matrix_t* softmax = convnet->acts[convnet->count - 1];
			float* dloss = softmax->data.f32;
			_ccv_convnet_compute_softmax(softmax, &softmax, 0);
			assert(softmax->rows == category_count && softmax->cols == 1);
			// this mashes softmax and logistic regression together
			// also, it gives you -D[loss w.r.t. to x_i] (note the negative sign)
			for (j = 0; j < category_count; j++)
				dloss[j] = (j == categorized->c) - dloss[j];
			_ccv_convnet_propagate_loss(convnet, categorized->matrix, softmax, update_params);
			if ((i + 1) % params.mini_batch == 0)
			{
				FLUSH(" - at epoch %03d / %d => stochastic gradient descent at %d / %d", t + 1, params.max_epoch, (i + 1) / params.mini_batch, aligned_rnum / params.mini_batch);
				// update weights
				_ccv_convnet_update(convnet, momentum, update_params, params.layer_params);
				_ccv_convnet_update_zero(update_params);
			}
		}
		int miss = 0;
		for (i = 0; i < tests->rnum; i++)
		{
			FLUSH(" - at epoch %03d / %d => going through %d / %d for tests", t + 1, params.max_epoch, i + 1, tests->rnum);
			ccv_categorized_t* test = (ccv_categorized_t*)ccv_array_get(tests, i);
			int c = 0;
			_ccv_convnet_classify(convnet, &test->matrix, &c, 1);
			if (c != test->c)
				++miss;
		}
		FLUSH(" - at epoch %03d / %d => with miss rate %.2f%%\n", t + 1, params.max_epoch, miss * 100.0f / tests->rnum);
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
	ccv_convnet_free(momentum);
	ccv_convnet_free(update_params);
	gsl_rng_free(rng);
#ifdef HAVE_CUDA
	}
#endif
#else
	assert(0 && "ccv_convnet_supervised_train requires GSL library support");
#endif
}

void ccv_convnet_compact(ccv_convnet_t* convnet)
{
#ifdef HAVE_CUDA
	cwc_convnet_compact(convnet);
#endif
	int i;
	for (i = 0; i < convnet->count; i++)
	{
		if (convnet->acts[i])
			ccv_matrix_free(convnet->acts[i]);
		convnet->acts[i] = 0;
		if (convnet->denoms)
		{
			if (convnet->denoms[i])
				ccv_matrix_free(convnet->denoms[i]);
			convnet->denoms[i] = 0;
		}
	}
}

void ccv_convnet_write(ccv_convnet_t* convnet, const char* filename, ccv_convnet_write_param_t params)
{
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char layer_create_table_qs[] =
			"CREATE TABLE IF NOT EXISTS layer_params "
			"(layer INTEGER PRIMARY KEY ASC, type INTEGER, "
			"input_matrix_rows INTEGER, input_matrix_cols INTEGER, input_matrix_channels INTEGER, input_matrix_partition INTEGER, input_node_count INTEGER, "
			"output_rows INTEGER, output_cols INTEGER, output_channels INTEGER, output_partition INTEGER, output_count INTEGER, output_strides INTEGER, output_border INTEGER, "
			"output_size INTEGER, output_kappa REAL, output_alpha REAL, output_beta REAL);"
			"CREATE TABLE IF NOT EXISTS convnet_params "
			"(convnet INTEGER PRIMARY KEY ASC, input_height INTEGER, input_width INTEGER, mean_activity BLOB);"
			"CREATE TABLE IF NOT EXISTS layer_data "
			"(layer INTEGER PRIMARY KEY ASC, weight BLOB, bias BLOB, half_precision INTEGER);";
		assert(SQLITE_OK == sqlite3_exec(db, layer_create_table_qs, 0, 0, 0));
		const char layer_params_insert_qs[] = 
			"REPLACE INTO layer_params "
			"(layer, type, "
			"input_matrix_rows, input_matrix_cols, input_matrix_channels, input_matrix_partition, input_node_count, "
			"output_rows, output_cols, output_channels, output_partition, output_count, output_strides, output_border, "
			"output_size, output_kappa, output_alpha, output_beta) VALUES "
			"($layer, $type, " // 1
			"$input_matrix_rows, $input_matrix_cols, $input_matrix_channels, $input_matrix_partition, $input_node_count, " // 6
			"$output_rows, $output_cols, $output_channels, $output_partition, $output_count, $output_strides, $output_border, " // 13
			"$output_size, $output_kappa, $output_alpha, $output_beta);"; // 17
		sqlite3_stmt* layer_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, layer_params_insert_qs, sizeof(layer_params_insert_qs), &layer_params_insert_stmt, 0));
		const char layer_data_insert_qs[] =
			"REPLACE INTO layer_data "
			"(layer, weight, bias, half_precision) VALUES ($layer, $weight, $bias, $half_precision);";
		sqlite3_stmt* layer_data_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, layer_data_insert_qs, sizeof(layer_data_insert_qs), &layer_data_insert_stmt, 0));
		int i;
		for (i = 0; i < convnet->count; i++)
		{
			ccv_convnet_layer_t* layer = convnet->layers + i;
			// insert layer params
			sqlite3_bind_int(layer_params_insert_stmt, 1, i);
			sqlite3_bind_int(layer_params_insert_stmt, 2, layer->type);
			sqlite3_bind_int(layer_params_insert_stmt, 3, layer->input.matrix.rows);
			sqlite3_bind_int(layer_params_insert_stmt, 4, layer->input.matrix.cols);
			sqlite3_bind_int(layer_params_insert_stmt, 5, layer->input.matrix.channels);
			sqlite3_bind_int(layer_params_insert_stmt, 6, layer->input.matrix.partition);
			sqlite3_bind_int(layer_params_insert_stmt, 7, layer->input.node.count);
			switch (layer->type)
			{
				case CCV_CONVNET_CONVOLUTIONAL:
					sqlite3_bind_int(layer_params_insert_stmt, 8, layer->net.convolutional.rows);
					sqlite3_bind_int(layer_params_insert_stmt, 9, layer->net.convolutional.cols);
					sqlite3_bind_int(layer_params_insert_stmt, 10, layer->net.convolutional.channels);
					sqlite3_bind_int(layer_params_insert_stmt, 11, layer->net.convolutional.partition);
					sqlite3_bind_int(layer_params_insert_stmt, 12, layer->net.convolutional.count);
					sqlite3_bind_int(layer_params_insert_stmt, 13, layer->net.convolutional.strides);
					sqlite3_bind_int(layer_params_insert_stmt, 14, layer->net.convolutional.border);
					break;
				case CCV_CONVNET_FULL_CONNECT:
					sqlite3_bind_int(layer_params_insert_stmt, 12, layer->net.full_connect.count);
					break;
				case CCV_CONVNET_MAX_POOL:
				case CCV_CONVNET_AVERAGE_POOL:
					sqlite3_bind_int(layer_params_insert_stmt, 13, layer->net.pool.strides);
					sqlite3_bind_int(layer_params_insert_stmt, 14, layer->net.pool.border);
					sqlite3_bind_int(layer_params_insert_stmt, 15, layer->net.pool.size);
					break;
				case CCV_CONVNET_LOCAL_RESPONSE_NORM:
					sqlite3_bind_int(layer_params_insert_stmt, 15, layer->net.rnorm.size);
					sqlite3_bind_double(layer_params_insert_stmt, 16, layer->net.rnorm.kappa);
					sqlite3_bind_double(layer_params_insert_stmt, 17, layer->net.rnorm.alpha);
					sqlite3_bind_double(layer_params_insert_stmt, 18, layer->net.rnorm.beta);
					break;
			}
			assert(SQLITE_DONE == sqlite3_step(layer_params_insert_stmt));
			sqlite3_reset(layer_params_insert_stmt);
			sqlite3_clear_bindings(layer_params_insert_stmt);
			// insert layer data
			if (layer->type == CCV_CONVNET_CONVOLUTIONAL || layer->type == CCV_CONVNET_FULL_CONNECT)
			{
				sqlite3_bind_int(layer_data_insert_stmt, 1, i);
				if (params.half_precision)
				{
					uint16_t* w = (uint16_t*)ccmalloc(sizeof(uint16_t) * layer->wnum);
					ccv_float_to_half_precision(layer->w, w, layer->wnum);
					uint16_t* bias = (uint16_t*)ccmalloc(sizeof(uint16_t) * (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count));
					ccv_float_to_half_precision(layer->bias, bias, layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count);
					sqlite3_bind_blob(layer_data_insert_stmt, 2, w, sizeof(uint16_t) * layer->wnum, ccfree);
					sqlite3_bind_blob(layer_data_insert_stmt, 3, bias, sizeof(uint16_t) * (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count), ccfree);
				} else {
					sqlite3_bind_blob(layer_data_insert_stmt, 2, layer->w, sizeof(float) * layer->wnum, SQLITE_STATIC);
					sqlite3_bind_blob(layer_data_insert_stmt, 3, layer->bias, sizeof(float) * (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count), SQLITE_STATIC);
				}
				sqlite3_bind_int(layer_data_insert_stmt, 4, params.half_precision);
				assert(SQLITE_DONE == sqlite3_step(layer_data_insert_stmt));
				sqlite3_reset(layer_data_insert_stmt);
				sqlite3_clear_bindings(layer_data_insert_stmt);
			}
		}
		// insert convnet related params
		const char convnet_params_insert_qs[] =
			"REPLACE INTO convnet_params "
			"(convnet, mean_activity, input_height, input_width) VALUES (0, $mean_activity, $input_height, $input_width);";
		sqlite3_stmt* convnet_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, convnet_params_insert_qs, sizeof(convnet_params_insert_qs), &convnet_params_insert_stmt, 0));
		assert(convnet->mean_activity->rows == convnet->input.height);
		assert(convnet->mean_activity->cols == convnet->input.width);
		assert(CCV_GET_CHANNEL(convnet->mean_activity->type) == convnet->channels);
		assert(CCV_GET_DATA_TYPE(convnet->mean_activity->type) == CCV_32F);
		sqlite3_bind_blob(convnet_params_insert_stmt, 1, convnet->mean_activity->data.f32, sizeof(float) * convnet->input.height * convnet->input.width * convnet->channels, SQLITE_STATIC);
		sqlite3_bind_int(convnet_params_insert_stmt, 2, convnet->input.height);
		sqlite3_bind_int(convnet_params_insert_stmt, 3, convnet->input.width);
		assert(SQLITE_DONE == sqlite3_step(convnet_params_insert_stmt));
		sqlite3_reset(convnet_params_insert_stmt);
		sqlite3_clear_bindings(convnet_params_insert_stmt);

		sqlite3_finalize(layer_params_insert_stmt);
		sqlite3_finalize(layer_data_insert_stmt);
		sqlite3_finalize(convnet_params_insert_stmt);
		sqlite3_close(db);
	}
}

ccv_convnet_t* ccv_convnet_read(int use_cwc_accel, const char* filename)
{
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		ccv_convnet_t* convnet = 0;
		sqlite3_stmt* layer_params_stmt = 0;
		// load layer params
		const char layer_params_qs[] =
			"SELECT type, " // 1
			"input_matrix_rows, input_matrix_cols, input_matrix_channels, input_matrix_partition, input_node_count, " // 6
			"output_rows, output_cols, output_channels, output_partition, output_count, output_strides, output_border, " // 13
			"output_size, output_kappa, output_alpha, output_beta FROM layer_params ORDER BY layer ASC;"; // 17
		if (SQLITE_OK == sqlite3_prepare_v2(db, layer_params_qs, sizeof(layer_params_qs), &layer_params_stmt, 0))
		{
			ccv_array_t* layer_params = ccv_array_new(sizeof(ccv_convnet_layer_param_t), 3, 0);
			while (sqlite3_step(layer_params_stmt) == SQLITE_ROW)
			{
				ccv_convnet_layer_param_t layer_param;
				layer_param.type = sqlite3_column_int(layer_params_stmt, 0);
				layer_param.input.matrix.rows = sqlite3_column_int(layer_params_stmt, 1);
				layer_param.input.matrix.cols = sqlite3_column_int(layer_params_stmt, 2);
				layer_param.input.matrix.channels = sqlite3_column_int(layer_params_stmt, 3);
				layer_param.input.matrix.partition = sqlite3_column_int(layer_params_stmt, 4);
				layer_param.input.node.count = sqlite3_column_int(layer_params_stmt, 5);
				layer_param.bias = layer_param.sigma = 0; // this is irrelevant to read convnet
				switch (layer_param.type)
				{
					case CCV_CONVNET_CONVOLUTIONAL:
						layer_param.output.convolutional.rows = sqlite3_column_int(layer_params_stmt, 6);
						layer_param.output.convolutional.cols = sqlite3_column_int(layer_params_stmt, 7);
						layer_param.output.convolutional.channels = sqlite3_column_int(layer_params_stmt, 8);
						layer_param.output.convolutional.partition = sqlite3_column_int(layer_params_stmt, 9);
						layer_param.output.convolutional.count = sqlite3_column_int(layer_params_stmt, 10);
						layer_param.output.convolutional.strides = sqlite3_column_int(layer_params_stmt, 11);
						layer_param.output.convolutional.border = sqlite3_column_int(layer_params_stmt, 12);
						break;
					case CCV_CONVNET_FULL_CONNECT:
						layer_param.output.full_connect.count = sqlite3_column_int(layer_params_stmt, 10);
						break;
					case CCV_CONVNET_MAX_POOL:
					case CCV_CONVNET_AVERAGE_POOL:
						layer_param.output.pool.strides = sqlite3_column_int(layer_params_stmt, 11);
						layer_param.output.pool.border = sqlite3_column_int(layer_params_stmt, 12);
						layer_param.output.pool.size = sqlite3_column_int(layer_params_stmt, 13);
						break;
					case CCV_CONVNET_LOCAL_RESPONSE_NORM:
						layer_param.output.rnorm.size = sqlite3_column_int(layer_params_stmt, 13);
						layer_param.output.rnorm.kappa = sqlite3_column_double(layer_params_stmt, 14);
						layer_param.output.rnorm.alpha = sqlite3_column_double(layer_params_stmt, 15);
						layer_param.output.rnorm.beta = sqlite3_column_double(layer_params_stmt, 16);
						break;
				}
				ccv_array_push(layer_params, &layer_param);
			}
			sqlite3_finalize(layer_params_stmt);
			sqlite3_stmt* convnet_params_input_stmt = 0;
			// load convnet params for input
			const char convnet_params_input_qs[] =
				"SELECT input_height, input_width FROM convnet_params WHERE convnet = 0;";
			ccv_size_t input = ccv_size(0, 0);
			if (SQLITE_OK == sqlite3_prepare_v2(db, convnet_params_input_qs, sizeof(convnet_params_input_qs), &convnet_params_input_stmt, 0))
			{
				if (sqlite3_step(convnet_params_input_stmt) == SQLITE_ROW)
				{
					input.height = sqlite3_column_int(convnet_params_input_stmt, 0);
					input.width = sqlite3_column_int(convnet_params_input_stmt, 1);
				}
				sqlite3_finalize(convnet_params_input_stmt);
			}
			assert(input.height != 0 && input.width != 0);
			convnet = ccv_convnet_new(use_cwc_accel, input, (ccv_convnet_layer_param_t*)ccv_array_get(layer_params, 0), layer_params->rnum);
			ccv_array_free(layer_params);
			// load layer data
			sqlite3_stmt* layer_data_stmt = 0;
			const char layer_data_qs[] =
				"SELECT layer, weight, bias, half_precision FROM layer_data;";
			if (SQLITE_OK == sqlite3_prepare_v2(db, layer_data_qs, sizeof(layer_data_qs), &layer_data_stmt, 0))
			{
				while (sqlite3_step(layer_data_stmt) == SQLITE_ROW)
				{
					ccv_convnet_layer_t* layer = convnet->layers + sqlite3_column_int(layer_data_stmt, 0);
					int half_precision = sqlite3_column_int(layer_data_stmt, 3);
					int wnum = sqlite3_column_bytes(layer_data_stmt, 1) / (half_precision ? sizeof(uint16_t) : sizeof(float));
					int bnum = sqlite3_column_bytes(layer_data_stmt, 2) / (half_precision ? sizeof(uint16_t) : sizeof(float));
					if (wnum != layer->wnum)
						continue;
					const void* w = sqlite3_column_blob(layer_data_stmt, 1);
					const void* bias = sqlite3_column_blob(layer_data_stmt, 2);
					if (half_precision)
					{
						float* f = (float*)ccmalloc(sizeof(float) * (layer->wnum + (layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count)));
						ccv_half_precision_to_float((uint16_t*)w, f, layer->wnum);
						ccv_half_precision_to_float((uint16_t*)bias, f + layer->wnum, layer->type == CCV_CONVNET_CONVOLUTIONAL ? layer->net.convolutional.count : layer->net.full_connect.count);
						w = f;
						bias = f + layer->wnum;
					}
					switch (layer->type)
					{
						case CCV_CONVNET_CONVOLUTIONAL:
							if (bnum != layer->net.convolutional.count)
								continue;
							memcpy(layer->w, w, sizeof(float) * layer->wnum);
							memcpy(layer->bias, bias, sizeof(float) * layer->net.convolutional.count);
							break;
						case CCV_CONVNET_FULL_CONNECT:
							if (bnum != layer->net.full_connect.count)
								continue;
							memcpy(layer->w, w, sizeof(float) * layer->wnum);
							memcpy(layer->bias, bias, sizeof(float) * layer->net.full_connect.count);
							break;
					}
					if (half_precision)
						ccfree((void*)w);
				}
				sqlite3_finalize(layer_data_stmt);
			}
			sqlite3_stmt* convnet_params_mean_activity_stmt = 0;
			// load convnet params for mean activity
			const char convnet_params_mean_activity_qs[] =
				"SELECT mean_activity FROM convnet_params WHERE convnet = 0;";
			if (SQLITE_OK == sqlite3_prepare_v2(db, convnet_params_mean_activity_qs, sizeof(convnet_params_mean_activity_qs), &convnet_params_mean_activity_stmt, 0))
			{
				if (sqlite3_step(convnet_params_mean_activity_stmt) == SQLITE_ROW)
				{
					int elems = sqlite3_column_bytes(convnet_params_mean_activity_stmt, 0) / sizeof(float);
					if (elems == convnet->input.height * convnet->input.width * convnet->channels)
						memcpy(convnet->mean_activity->data.f32, sqlite3_column_blob(convnet_params_mean_activity_stmt, 0), sizeof(float) * elems);
				}
				sqlite3_finalize(convnet_params_mean_activity_stmt);
			}
		}
		sqlite3_close(db);
		return convnet;
	}
	return 0;
}

void ccv_convnet_free(ccv_convnet_t* convnet)
{
	ccv_convnet_compact(convnet);
	int i;
	for (i = 0; i < convnet->count; i++)
		if (convnet->layers[i].w)
			ccfree(convnet->layers[i].w);
	if (convnet->mean_activity)
		ccv_matrix_free(convnet->mean_activity);
	ccfree(convnet);
}

#endif
