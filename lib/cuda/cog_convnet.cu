extern "C" {
#include "cog.h"
}

// this structure holds intermediate on-device memory representation of convnet
typedef struct {
	ccv_convnet_layer_t* layers;
	ccv_convnet_layer_t* updates;
} cog_convnet_t;

#define GPU(x) ((cog_convnet_t*)((x)->reserved))

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

template <int input_per_thread, int filter_per_thread>
__global__ void _cog_kern_convolutional_forward_propagate(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols,
		float* filter, const int filter_rows, const int filter_cols, const int count,
		float* const biases)
{
	// gridDim.x == out_rows
	// gridDim.y == out_cols
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	assert(batch == input_per_thread * blockDim.x);
	float* shared_weights = &shared[batch];
	assert(count == filter_per_thread * blockDim.y);
	float* shared_bias = &shared[batch + count];
	float prod[filter_per_thread][input_per_thread];
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	const int input_loads = (batch + thcnt - 1) / thcnt;
	const int filter_loads = (count + thcnt - 1) / thcnt;
	int c, i, j, x, y;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			prod[i][j] = 0;
	input += (blockIdx.x * strides * cols + blockIdx.y * strides) * batch;
	#pragma unroll
	for (i = 0; i < filter_loads; i++)
		if (i * thcnt + thidx < count)
			shared_bias[i * thcnt + thidx] = biases[i * thcnt + thidx];
	for (c = 0; c < channels; c++)
	{
		for (y = 0; y < filter_rows; y++)
		{
			const int iy = y + blockIdx.x * strides - border;
			for (x = 0; x < filter_cols; x++)
			{
				const int ix = x + blockIdx.y * strides - border;
				if (iy >= 0 && iy < rows && ix >= 0 && ix < cols)
				{
					#pragma unroll
					for (i = 0; i < input_loads; i++)
						if (i * thcnt + thidx < batch)
							shared_block[i * thcnt + thidx] = input[((y - border) * cols + x - border) * batch + i * thcnt + thidx];
					#pragma unroll
					for (i = 0; i < filter_loads; i++)
						if (i * thcnt + thidx < count)
							shared_weights[i * thcnt + thidx] = filter[(y * filter_cols + x) * count + i * thcnt + thidx];
					__syncthreads();
					#pragma unroll
					for (i = 0; i < filter_per_thread; i++)
						#pragma unroll
						for (j = 0; j < input_per_thread; j++)
							prod[i][j] += shared_block[j + threadIdx.x * input_per_thread] * shared_weights[i + threadIdx.y * filter_per_thread];
					__syncthreads();
				}
			}
		}
		input += rows * cols * batch;
		filter += filter_rows * filter_cols * count;
	}
	const int outcnt = out_rows * out_cols * batch;
	out += (blockIdx.x * out_cols + blockIdx.y) * batch;
	#pragma unroll
	for (i = 0; i < filter_per_thread; i++)
	{
		const float bias = shared_bias[i + threadIdx.y * filter_per_thread];
		#pragma unroll
		for (j = 0; j < input_per_thread; j++)
			out[(i + threadIdx.y * filter_per_thread) * outcnt + j + threadIdx.x * input_per_thread] = max(0.0, prod[i][j] + bias);
	}
}

template <int channel_per_thread, int filter_per_thread, int batch_per_thread>
__global__ void _cog_kern_convolutional_backward_propagate_delta(const int strides, const int border, const int batch,
		float* input, const int rows, const int cols, const int channels,
		float* out, const int out_rows, const int out_cols,
		float* out_grad, float* delta,
		float* filter, const int filter_rows, const int filter_cols, const int count)
{
	// gridDim.x == filter_rows
	// gridDim.y == filter_cols
	assert(gridDim.z == batch / batch_per_thread);
	extern __shared__ float shared[];
	float* shared_block = &shared[0];
	float* shared_out = &shared[batch_per_thread * channels];
	float* shared_grad = &shared[batch_per_thread * (channels + count)];
	float prod[channel_per_thread][filter_per_thread];
	// channel_per_thread * blockDim.x == channels
	// filter_per_thread * blockDim.y == filter_count
	assert(channel_per_thread * blockDim.x == channels);
	assert(filter_per_thread * blockDim.y == count);
	const int thidx = threadIdx.x + threadIdx.y * blockDim.x;
	const int thcnt = blockDim.x * blockDim.y;
	assert(batch % batch_per_thread == 0);
	assert(thcnt % batch_per_thread == 0);
	int i, j, k, x, y;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			prod[i][j] = 0;
	const int bxidx = thidx % batch_per_thread;
	const int byidx = thidx / batch_per_thread;
	const int incnt = rows * cols * batch;
	input += (blockIdx.x * cols + blockIdx.y) * batch + blockIdx.z * batch_per_thread + byidx * incnt + bxidx;
	const int outcnt = out_rows * out_cols * batch;
	out += blockIdx.z * batch_per_thread + byidx * outcnt + bxidx;
	out_grad += blockIdx.z * batch_per_thread + byidx * outcnt + bxidx;
	const int block_loads = (batch_per_thread * channels + thcnt - 1) / thcnt;
	const int out_loads = (batch_per_thread * count + thcnt - 1) / thcnt;
	const int block_loads_factor = (thcnt / batch_per_thread) * incnt;
	const int out_loads_factor = (thcnt / batch_per_thread) * outcnt;
	const int filter_idx = threadIdx.y * filter_per_thread;
	const int channel_idx = threadIdx.x * channel_per_thread;
	for (y = 0; y < out_rows; y++)
	{
		const int iy = blockIdx.x + y * strides - border;
		for (x = 0; x < out_cols; x++)
		{
			const int ix = blockIdx.y + x * strides - border;
			if (iy >= 0 && iy < rows && ix >= 0 && ix < cols)
			{
				#pragma unroll
				for (i = 0; i < block_loads; i++)
					if (thidx + i * thcnt < batch_per_thread * channels)
						shared_block[thidx + i * thcnt] = input[((y * strides - border) * cols + x * strides - border) * batch + i * block_loads_factor];
				#pragma unroll
				for (i = 0; i < out_loads; i++)
					if (thidx + i * thcnt < batch_per_thread * count)
						shared_out[thidx + i * thcnt] = out[x * batch + i * out_loads_factor],
						shared_grad[thidx + i * thcnt] = out_grad[x * batch + i * out_loads_factor];
				__syncthreads();
				#pragma unroll
				for (k = 0; k < batch_per_thread; k++)
					#pragma unroll
					for (i = 0; i < filter_per_thread; i++)
						if (shared_out[(i + filter_idx) * batch_per_thread + k] > 0)
							#pragma unroll
							for (j = 0; j < channel_per_thread; j++)
								prod[j][i] += shared_block[(j + channel_idx) * batch_per_thread + k] * shared_grad[(i + filter_idx) * batch_per_thread + k];
				__syncthreads();
			}
		}
		out += out_cols * batch;
		out_grad += out_cols * batch;
	}
	delta += (blockIdx.x * filter_cols + blockIdx.y) * count + blockIdx.z * filter_rows * filter_cols * count * channels;
	const int deltacnt = filter_rows * filter_cols * count;
	#pragma unroll
	for (i = 0; i < channel_per_thread; i++)
		#pragma unroll
		for (j = 0; j < filter_per_thread; j++)
			delta[(i + channel_idx) * deltacnt + j + filter_idx] = prod[i][j];
}

#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static void _cog_convolutional_forward_propagate(ccv_convnet_layer_t* layer, int batch, int rows, int cols, int ch, float* a, float* d, float** b)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int out_rows, out_cols;
	_ccv_convnet_compute_output_scale(rows, cols, layer, &out_rows, &out_cols);
	assert(b);
	float* db = *b;
	if (!db)
		cudaMalloc(&db, sizeof(float) * out_rows * out_cols * layer->net.convolutional.count * batch);
	*b = db;
	dim3 threads_per_block(batch / 8, layer->net.convolutional.count / 4);
	dim3 num_blocks(out_rows, out_cols);
	int shared_memory_size = sizeof(float) * (batch + layer->net.convolutional.count);
	_cog_kern_convolutional_forward_propagate
		<8, 4>
		<<<num_blocks, threads_per_block, shared_memory_size + /* need extra space for bias */ sizeof(float) * layer->net.convolutional.count>>>
		(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		 a, rows, cols, ch,
		 db, out_rows, out_cols,
		 layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count,
		 layer->bias);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cuda elapsed time forward propagate: %.2lf\n", elapsed_time);
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
		float* layer_w = layer->w + k * kernel_rows * kernel_cols * ch; \
		float bias = layer->bias[k]; \
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
				float* w = layer_w + comx + comy; \
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

static void _cog_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, int batch, int rows, int cols, int ch, float* a, float* n, float* d, float* m, float** b, ccv_convnet_layer_t* update)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	assert(layer->net.convolutional.count % 4 == 0);
	int out_rows, out_cols;
	_ccv_convnet_compute_output_scale(rows, cols, layer, &out_rows, &out_cols);
	dim3 threads_per_block(ch, layer->net.convolutional.count);
	assert(batch % 8 == 0);
	dim3 num_blocks(layer->net.convolutional.rows, layer->net.convolutional.cols, batch / 8);
	int shared_memory_size = sizeof(float) * (8 * (ch + layer->net.convolutional.count * 2));
	_cog_kern_convolutional_backward_propagate_delta
	<1, 1, 8>
	<<<num_blocks, threads_per_block, shared_memory_size>>>
	(layer->net.convolutional.strides, layer->net.convolutional.border, batch,
		m, rows, cols, ch,
		n, out_rows, out_cols,
		a, update->w,
		layer->w, layer->net.convolutional.rows, layer->net.convolutional.cols, layer->net.convolutional.count);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cuda elapsed time of backward propagate: %.2lf\n", elapsed_time);
}

// compute back propagated gradient & weight update delta
static void _ccv_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* n, ccv_dense_matrix_t* d, ccv_dense_matrix_t* m, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), d is the dropout,
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// note that y (the output from forward prop) is not included because the full connect net is simple enough that we don't need it
	int rows, cols;
	_ccv_convnet_compute_output_scale(m->rows, m->cols, layer, &rows, &cols);
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
	int i, j, x, y, k;
	// update weight gradient
#define for_block_w(act_block_setup, act_block_begin, act_block_end) \
	for (k = 0; k < count; k++) \
	{ \
		float* mp = m->data.f32; \
		float* ap = a->data.f32 + k; \
		float* np = n->data.f32 + k; \
		float* update_w = update_params->w + k * kernel_rows * kernel_cols * ch; \
		float bias = 0; \
		act_block_setup; \
		for (i = 0; i < rows; i++) \
		{ \
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border); \
			int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(m->rows + border, i * strides + kernel_rows)); \
			comy *= ch * kernel_cols; \
			for (j = 0; j < cols; j++) \
			{ \
				act_block_begin; \
				if (np[j * count] > 0) \
				{ /* when np is bigger than 0, relu continues to update the weight, otherwise it stops */ \
					float v = ap[j * count]; \
					bias += v; \
					int comx = (ccv_max(j * strides - border, 0) - (j * strides - border)) * ch; \
					int maxx = kernel_cols * ch - comx - (j * strides + kernel_cols - ccv_min(m->cols + border, j * strides + kernel_cols)) * ch; \
					float* w = update_w + comx + comy; \
					float* mpz = mp + ccv_max(j * strides - border, 0) * ch; \
					/* when we have border, we simply do zero padding */ \
					for (y = 0; y < maxy; y++) \
					{ \
						for (x = 0; x < maxx; x++) \
							w[x] += v * mpz[x]; \
						w += kernel_cols * ch; \
						mpz += m->cols * ch; \
					} \
				} \
				act_block_end; \
			} \
			ap += a->cols * count; \
			np += n->cols * count; \
			mp += m->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0)); \
		} \
		update_params->bias[k] = bias; \
	}
	ccv_dense_matrix_t* db = 0;
	if (b)
	{
		db = *b = ccv_dense_matrix_renew(*b, m->rows, m->cols, CCV_32F | CCV_GET_CHANNEL(m->type), CCV_32F | CCV_GET_CHANNEL(m->type), 0);
		// clear it up before propagate result
		ccv_zero(db);
	}
#define for_block_b(act_block_setup, act_block_begin, act_block_end) \
	for (k = 0; k < count; k++) \
	{ \
		float* bp = db->data.f32; \
		float* ap = a->data.f32 + k; \
		float* np = n->data.f32 + k; \
		float* layer_w = layer->w + k * kernel_rows * kernel_cols * ch; \
		act_block_setup; \
		for (i = 0; i < rows; i++) \
		{ \
			int comy = ccv_max(i * strides - border, 0) - (i * strides - border); \
			int maxy = kernel_rows - comy - (i * strides + kernel_rows - ccv_min(db->rows + border, i * strides + kernel_rows)); \
			comy *= ch * kernel_cols; \
			for (j = 0; j < cols; j++) \
			{ \
				act_block_begin; \
				if (np[j * count] > 0) \
				{ /* when np is bigger than 0, relu continues to update the weight, otherwise it stops */ \
					float v = ap[j * count]; \
					int comx = (ccv_max(j * strides - border, 0) - (j * strides - border)) * ch; \
					int maxx = kernel_cols * ch - comx - (j * strides + kernel_cols - ccv_min(db->cols + border, j * strides + kernel_cols)) * ch; \
					float* w = layer_w + comx + comy; \
					float* bpz = bp + ccv_max(j * strides - border, 0) * ch; \
					/* when we have border, we simply do zero padding */ \
					for (y = 0; y < maxy; y++) \
					{ \
						for (x = 0; x < maxx; x++) \
							bpz[x] += v * w[x]; \
						w += kernel_cols * ch; \
						bpz += db->cols * ch; \
					} \
				} \
				act_block_end; \
			} \
			ap += a->cols * count; \
			np += n->cols * count; \
			bp += db->cols * ch * (ccv_max((i + 1) * strides - border, 0) - ccv_max(i * strides - border, 0)); \
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
		} \
		dp += count;
		for_block_w(act_block_setup, act_block_begin, act_block_end);
		if (db)
			for_block_b(act_block_setup, act_block_begin, act_block_end);
#undef act_block_setup
#undef act_block_begin
#undef act_block_end
	} else {
		for_block_w(/* empty act block setup */, /* empty act block begin */, /* empty act block end */);
		if (db)
			for_block_b(/* empty act block setup */, /* empty act block begin */, /* empty act block end */);
	}
#undef for_block_w
#undef for_block_b
	a->rows = a_rows, a->cols = a_cols, a->type = (a->type - CCV_GET_CHANNEL(a->type)) | a_ch;
}

static void _cog_convnet_reserve_on_device(ccv_convnet_t* convnet)
{
	assert(GPU(convnet) == 0);
	convnet->reserved = (cog_convnet_t*)ccmalloc(sizeof(cog_convnet_t) + sizeof(ccv_convnet_layer_t) * convnet->count * 2);
	GPU(convnet)->layers = (ccv_convnet_layer_t*)(GPU(convnet) + 1);
	GPU(convnet)->updates = GPU(convnet)->layers + convnet->count;
	memcpy(GPU(convnet)->layers, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	memcpy(GPU(convnet)->updates, convnet->layers, sizeof(ccv_convnet_layer_t) * convnet->count);
	ccv_convnet_layer_t* layers = GPU(convnet)->layers;
	ccv_convnet_layer_t* updates = GPU(convnet)->updates;
	int i;
	for (i = 0; i < convnet->count; i++)
		switch (layers[i].type)
		{
			case CCV_CONVNET_CONVOLUTIONAL:
				assert(updates[i].type == CCV_CONVNET_CONVOLUTIONAL);
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count));
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				// this is wrong, I need to rewind w
				cudaMemcpy(layers[i].w, convnet->layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.convolutional.count), cudaMemcpyHostToDevice);
				updates[i].w = 0;
				cudaMalloc(&updates[i].w, sizeof(float) * (updates[i].wnum * 16 + updates[i].net.convolutional.count));
				assert(updates[i].w);
				updates[i].bias = updates[i].w + updates[i].wnum;
				break;
			case CCV_CONVNET_FULL_CONNECT:
				assert(updates[i].type == CCV_CONVNET_FULL_CONNECT);
				layers[i].w = 0;
				cudaMalloc(&layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count));
				assert(layers[i].w);
				layers[i].bias = layers[i].w + layers[i].wnum;
				cudaMemcpy(layers[i].w, convnet->layers[i].w, sizeof(float) * (layers[i].wnum + layers[i].net.full_connect.count), cudaMemcpyHostToDevice);
				updates[i].w = 0;
				cudaMalloc(&updates[i].w, sizeof(float) * (updates[i].wnum * 16 + updates[i].net.full_connect.count));
				updates[i].bias = updates[i].w + updates[i].wnum;
				break;
			case CCV_CONVNET_MAX_POOL:
			case CCV_CONVNET_AVERAGE_POOL:
				assert(updates[i].type == layers[i].type);
				updates[i].w = updates[i].bias = 0;
				layers[i].w = layers[i].bias = 0;
				break;
		}
}

void cog_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch)
{
	int ch = CCV_GET_CHANNEL(a[0]->type);
	int rows = a[0]->rows, cols = a[0]->cols;
	float* vec = 0;
	cudaMallocHost(&vec, sizeof(float) * batch * rows * cols * ch);
	int i, j, k, c, z;
	for (i = 0; i < batch; i++)
		for (k = 0; k < ch; k++)
			for (j = 0; j < rows * cols; j++)
				vec[i + (k * rows * cols + j) * batch] = a[i]->data.f32[j * ch + k];
	float* od_vec = 0;
	cudaMalloc(&od_vec, sizeof(float) * batch * rows * cols * ch);
	cudaMemcpy(od_vec, vec, sizeof(float) * batch * rows * cols * ch, cudaMemcpyHostToDevice);
	float* od_out = 0;
	_cog_convolutional_forward_propagate(GPU(convnet)->layers, batch, rows, cols, ch, od_vec, 0, &od_out);
	_cog_convnet_convolutional_backward_propagate(GPU(convnet)->layers, batch, rows, cols, ch, od_out, od_out, 0, od_vec, 0, GPU(convnet)->updates);
	float* out_weights = 0;
	cudaMallocHost(&out_weights, sizeof(float) * convnet->layers->wnum * 16);
	assert(out_weights);
	cudaMemcpy(out_weights, GPU(convnet)->updates->w, sizeof(float) * convnet->layers->wnum * 16, cudaMemcpyDeviceToHost);
	int out_rows, out_cols;
	_ccv_convnet_compute_output_scale(rows, cols, convnet->layers, &out_rows, &out_cols);
	ccv_convnet_layer_t updates;
	updates.w = (float*)ccmalloc(sizeof(float) * (convnet->layers->wnum + convnet->layers->net.convolutional.count));
	memset(updates.w, 0, sizeof(float) * (convnet->layers->wnum + convnet->layers->net.convolutional.count));
	updates.bias = updates.w + convnet->layers->wnum;
	unsigned int elapsed_time = get_current_time();
	for (i = 0; i < batch; i++)
	{
		ccv_dense_matrix_t* b = 0;
		_ccv_convnet_convolutional_forward_propagate(convnet->layers, a[i], 0, &b);
		ccv_dense_matrix_t* out = 0;
		_ccv_convnet_convolutional_backward_propagate(convnet->layers, b, b, 0, a[i], &out, &updates);
		ccv_matrix_free(b);
		ccv_matrix_free(out);
	}
	elapsed_time = get_current_time() - elapsed_time;
	printf("cpu elapsed time of backward propagate: %u\n", elapsed_time);
	int filter_rows = convnet->layers->net.convolutional.rows;
	int filter_cols = convnet->layers->net.convolutional.cols;
	int filter_count = convnet->layers->net.convolutional.count;
	for (i = 0; i < filter_rows; i++)
		for (j = 0; j < filter_cols; j++)
			for (k = 0; k < filter_count; k++)
				for (c = 0; c < ch; c++)
				{
					float w = updates.w[(i * filter_cols + j) * ch + k * filter_cols * filter_rows * ch + c];
					float ow = out_weights[(i * filter_cols + j) * filter_count + k + c * filter_cols * filter_rows * filter_count];
					for (z = 1; z < 16; z++)
						ow += out_weights[z * filter_rows * filter_cols * filter_count * ch + (i * filter_cols + j) * filter_count + k + c * filter_cols * filter_rows * filter_count];
					float delta = fabsf(ow - w) / w;
					if (delta > 0.0001)
						printf("%d,%d,%d,%d: %f, %f\n", i, j, k, c, w, ow);
				}
}

void cog_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch)
{
}

void cog_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, ccv_convnet_train_param_t params)
{
	assert(categorizeds->rnum >= 128);
	if (!GPU(convnet))
		_cog_convnet_reserve_on_device(convnet);
	int i;
	ccv_dense_matrix_t* a[128];
	for (i = 0; i < 128; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
		ccv_dense_matrix_t* b = 0;
		if (image->rows > 251 && image->cols > 251)
			ccv_resample(image, &b, 0, ccv_max(251, (int)(image->rows * 251.0 / image->cols + 0.5)), ccv_max(251, (int)(image->cols * 251.0 / image->rows + 0.5)), CCV_INTER_AREA);
		else if (image->rows < 251 || image->cols < 251)
			ccv_resample(image, &b, 0, ccv_max(251, (int)(image->rows * 251.0 / image->cols + 0.5)), ccv_max(251, (int)(image->cols * 251.0 / image->rows + 0.5)), CCV_INTER_CUBIC);
		else
			b = image;
		if (b != image)
			ccv_matrix_free(image);
		ccv_dense_matrix_t* c = 0;
		ccv_slice(b, (ccv_matrix_t**)&c, CCV_32F, 0, 0, 225, 225);
		int j, ch = CCV_GET_CHANNEL(c->type);
		for (j = 0; j < c->rows * c->cols * ch; j++)
			c->data.f32[j] = c->data.f32[j] / 255.0 * 2 - 1;
		a[i] = c;
		ccv_matrix_free(b);
	}
	cog_convnet_encode(convnet, a, 0, 128);
}

void cog_convnet_free(ccv_convnet_t* convnet)
{
	int i;
	ccv_convnet_layer_t* layers = GPU(convnet)->layers;
	for (i = 0; i < convnet->count; i++)
		cudaFree(layers[i].w);
	ccfree(convnet);
}
