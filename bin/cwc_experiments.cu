// ===================================== TEST CODE ==========================================

static void _ccv_convnet_convolutional_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t** b)
{
	int rows, cols;
	_ccv_convnet_layer_deduce_output_format(a->rows, a->cols, layer, &rows, &cols);
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

// compute back propagated gradient & weight update delta
static void _ccv_convnet_convolutional_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* n, ccv_dense_matrix_t* d, ccv_dense_matrix_t* m, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
{
	// a is the input gradient (for back prop), d is the dropout,
	// x is the input (for forward prop), b is the output gradient (gradient, or known as propagated error)
	// note that y (the output from forward prop) is not included because the full connect net is simple enough that we don't need it
	int rows, cols;
	_ccv_convnet_layer_deduce_output_format(m->rows, m->cols, layer, &rows, &cols);
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
		update_params->bias[k] += bias; \
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

static void _ccv_convnet_max_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows, cols;
	_ccv_convnet_layer_deduce_output_format(a->rows, a->cols, layer, &rows, &cols);
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
						{
							float mv = mp[(j * strides - border + x + (y - border) * m->cols) * ch + k];
							float delta = fabsf(mv - v) / ccv_max(ccv_max(fabsf(mv), fabsf(v)), 1e-5);
							if (delta < 1e-5) // we cannot do direct comparison because CPU have different result comparing with GPU
								bp[(j * strides - border + x + (y - border) * db->cols) * ch + k] += u;
						}
				}
			}
			ap += a->cols * ch;
			np += n->cols * ch;
			bp += db->cols * ch * strides;
			mp += m->cols * ch * strides;
		}
	}
}

static void _ccv_convnet_average_pool_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
	int rows, cols;
	_ccv_convnet_layer_deduce_output_format(a->rows, a->cols, layer, &rows, &cols);
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

static void _ccv_convnet_full_connect_forward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t** b)
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
		ccv_gemm(&dw, a, 1, db, 1, 0, (ccv_matrix_t**)&db, 0); // supply db as matrix C is allowed
	}
	a->rows = rows, a->cols = cols, a->type = (a->type - CCV_GET_CHANNEL(a->type)) | ch;
	a->step = a->cols * CCV_GET_DATA_TYPE_SIZE(a->type) * CCV_GET_CHANNEL(a->type);
}

static void _ccv_convnet_full_connect_backward_propagate(ccv_convnet_layer_t* layer, ccv_dense_matrix_t* a, ccv_dense_matrix_t* d, ccv_dense_matrix_t* x, ccv_dense_matrix_t** b, ccv_convnet_layer_t* update_params)
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
	if (d)
	{
		int* dptr = d->data.i32;
		float* aptr = a->data.f32;
		float* bptr = update_params->bias;
		int i, j;
		// bias gradient
		for (i = 0; i < a->rows; i++)
			if (dptr[i])
				bptr[i] += aptr[i];
		// weight gradient
		float* dwptr = update_params->w;
		for (i = 0; i < a->rows; i++)
		{
			if (dptr[i])
			{
				float* xptr = x->data.f32;
				for (j = 0; j < x->rows; j++)
					dwptr[j] += aptr[i] * xptr[j];
			}
			dwptr += x->rows;
		}
		// propagate error
		if (db)
		{
			ccv_zero(db);
			float* wptr = layer->w;
			for (i = 0; i < a->rows; i++)
			{
				if (dptr[i])
				{
					float* bptr = db->data.f32;
					for (j = 0; j < db->rows; j++)
						bptr[j] += wptr[j] * aptr[i];
				}
				wptr += x->rows;
			}
		}
	} else {
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
	}
	x->rows = x_rows, x->cols = x_cols, x->type = (x->type - CCV_GET_CHANNEL(x->type)) | x_ch;
	x->step = x->cols * CCV_GET_DATA_TYPE_SIZE(x->type) * CCV_GET_CHANNEL(x->type);
}

#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

inline static float _ccv_relative_delta(float a, float b)
{
	return fabsf(a - b) / ccv_max(ccv_max(fabsf(a), fabsf(b)), 1);
}

void cwc_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch)
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
	int out_rows, out_cols;
	_ccv_convnet_layer_deduce_output_format(rows, cols, convnet->layers, &out_rows, &out_cols);
	cudaMemcpy(od_vec, vec, sizeof(float) * batch * rows * cols * ch, cudaMemcpyHostToDevice);
	float* od_out = 0;
	cudaStream_t streams[2];
	for (i = 0; i < 2; i++)
		cudaStreamCreate(&streams[i]);

	// convolutional forward propagate
	unsigned int elapsed_time = get_current_time();
	_cwc_convolutional_forward_propagate(GPU(convnet)->layers, batch, rows, cols, ch, od_vec, &od_out, streams[0]);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time convolutional forward propagate: %u\n", elapsed_time);
	float* out = 0;
	cudaMallocHost(&out, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch);
	assert(out);
	cudaMemcpy(out, od_out, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToHost);

	// max pool forward propagate
	float* od_max = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_max_pool_forward_propagate(GPU(convnet)->layers + 1, batch, out_rows, out_cols, convnet->layers->net.convolutional.count, od_out, &od_max, streams[0]);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time max pool forward propagate: %u\n", elapsed_time);
	assert(od_max);
	float* max_pooled = 0;
	int max_rows, max_cols;
	_ccv_convnet_layer_deduce_output_format(out_rows, out_cols, convnet->layers + 1, &max_rows, &max_cols);
	cudaMallocHost(&max_pooled, sizeof(float) * max_rows * max_cols * convnet->layers->net.convolutional.count * batch);
	assert(max_pooled);
	cudaMemcpy(max_pooled, od_max, sizeof(float) * max_rows * max_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToHost);

	// average pool forward propagate
	float* od_average = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_average_pool_forward_propagate(GPU(convnet)->layers + 2, batch, out_rows, out_cols, convnet->layers->net.convolutional.count, od_out, &od_average, streams[0]);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time average pool forward propagate: %u\n", elapsed_time);
	assert(od_average);
	float* average_pooled = 0;
	int average_rows, average_cols;
	_ccv_convnet_layer_deduce_output_format(out_rows, out_cols, convnet->layers + 2, &average_rows, &average_cols);
	cudaMallocHost(&average_pooled, sizeof(float) * average_rows * average_cols * convnet->layers->net.convolutional.count * batch);
	assert(average_pooled);
	cudaMemcpy(average_pooled, od_average, sizeof(float) * average_rows * average_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToHost);

	// full connect forward propagate
	float* batch_unit = 0;
	cudaMalloc(&batch_unit, sizeof(float) * batch);
	float* host_batch_unit = 0;
	cudaMallocHost(&host_batch_unit, sizeof(float) * batch);
	for (i = 0; i < batch; i++)
		host_batch_unit[i] = 1;
	cudaMemcpy(batch_unit, host_batch_unit, sizeof(float) * batch, cudaMemcpyHostToDevice);
	cudaFreeHost(host_batch_unit);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetStream(handle, streams[0]);
	float* od_full_connect = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_full_connect_forward_propagate(GPU(convnet)->layers + 3, batch, average_rows, average_cols, 5, od_average, &od_full_connect, batch_unit, handle);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time full connect forward propagate: %u\n", elapsed_time);
	assert(od_full_connect);
	float* full_connected = 0;
	cudaMallocHost(&full_connected, sizeof(float) * batch * convnet->layers[3].net.full_connect.count);
	assert(full_connected);
	cudaMemcpy(full_connected, od_full_connect, sizeof(float) * batch * convnet->layers[3].net.full_connect.count, cudaMemcpyDeviceToHost);

	// convolutional backward propagate
	float* out_grad = 0;
	cudaMalloc(&out_grad, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch);
	cudaMemcpy(out_grad, od_out, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToDevice);
	float* input_grad = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_convolutional_backward_propagate(GPU(convnet)->layers, batch, rows, cols, ch, out_grad, od_out, od_vec, &input_grad, GPU(convnet)->updates, streams[0]);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time convolutional backward propagate: %u\n", elapsed_time);
	float* out_weights = 0;
	cudaMallocHost(&out_weights, sizeof(float) * convnet->layers->wnum * 8 * out_rows);
	assert(out_weights);
	cudaMemcpy(out_weights, GPU(convnet)->updates->w, sizeof(float) * convnet->layers->wnum * 8 * out_rows, cudaMemcpyDeviceToHost);
	float* out_bias = 0;
	cudaMallocHost(&out_bias, sizeof(float) * convnet->layers->net.convolutional.count);
	assert(out_bias);
	cudaMemcpy(out_bias, GPU(convnet)->updates->bias, sizeof(float) * convnet->layers->net.convolutional.count, cudaMemcpyDeviceToHost);
	float* out_input_grad = 0;
	cudaMallocHost(&out_input_grad, sizeof(float) * rows * cols * batch * ch);
	assert(out_input_grad);
	cudaMemcpy(out_input_grad, input_grad, sizeof(float) * rows * cols * batch * ch, cudaMemcpyDeviceToHost);

	// max pool backward propagate
	float* max_pooled_grad = 0;
	cudaMalloc(&max_pooled_grad, sizeof(float) * max_rows * max_cols * convnet->layers->net.convolutional.count * batch);
	cudaMemcpy(max_pooled_grad, max_pooled, sizeof(float) * max_rows * max_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToDevice);
	float* max_pooled_input_grad = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_max_pool_backward_propagate(GPU(convnet)->layers + 1, batch, out_rows, out_cols, convnet->layers->net.convolutional.count, max_pooled_grad, max_pooled, od_out, &max_pooled_input_grad, streams[0]);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time max pool backward propagate: %u\n", elapsed_time);
	float* max_pooled_out_input_grad = 0;
	cudaMallocHost(&max_pooled_out_input_grad, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch);
	cudaMemcpy(max_pooled_out_input_grad, max_pooled_input_grad, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToHost);

	// average pool backward propagate
	float* average_pooled_input_grad = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_average_pool_backward_propagate(GPU(convnet)->layers + 1, batch, out_rows, out_cols, convnet->layers->net.convolutional.count, average_pooled, &average_pooled_input_grad, streams[0]);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time average pool backward propagate: %u\n", elapsed_time);
	float* average_pooled_out_input_grad = 0;
	cudaMallocHost(&average_pooled_out_input_grad, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch);
	cudaMemcpy(average_pooled_out_input_grad, average_pooled_input_grad, sizeof(float) * out_rows * out_cols * convnet->layers->net.convolutional.count * batch, cudaMemcpyDeviceToHost);

	// full connect backward propagate
	float* full_connect_grad = 0;
	elapsed_time = get_current_time();
	_cwc_convnet_full_connect_backward_propagate(GPU(convnet)->layers + 3, batch, average_rows, average_cols, 5, od_full_connect, od_average, &full_connect_grad, batch_unit, GPU(convnet)->updates + 3, handle);
	cudaDeviceSynchronize();
	elapsed_time = get_current_time() - elapsed_time;
	printf("cuda elapsed time full connect backward propagate: %u\n", elapsed_time);
	float* full_connected_grad = 0;
	cudaMallocHost(&full_connected_grad, sizeof(float) * average_rows * average_cols * 5 * batch);
	assert(full_connect_grad);
	cudaMemcpy(full_connected_grad, full_connect_grad, sizeof(float) * average_rows * average_cols * 5 * batch, cudaMemcpyDeviceToHost);
	float* out_fcbias = 0;
	cudaMallocHost(&out_fcbias, sizeof(float) * convnet->layers[3].net.full_connect.count);
	cudaMemcpy(out_fcbias, GPU(convnet)->updates[3].bias, sizeof(float) * convnet->layers[3].net.full_connect.count, cudaMemcpyDeviceToHost);
	float* out_fcw = 0;
	cudaMallocHost(&out_fcw, sizeof(float) * average_rows * average_cols * 5 * convnet->layers[3].net.full_connect.count);
	cudaMemcpy(out_fcw, GPU(convnet)->updates[3].w, sizeof(float) * average_rows * average_cols * 5 * convnet->layers[3].net.full_connect.count, cudaMemcpyDeviceToHost);

	ccv_convnet_layer_t updates;
	updates.w = (float*)ccmalloc(sizeof(float) * (convnet->layers->wnum + convnet->layers->net.convolutional.count));
	memset(updates.w, 0, sizeof(float) * (convnet->layers->wnum + convnet->layers->net.convolutional.count));
	updates.bias = updates.w + convnet->layers->wnum;
	ccv_convnet_layer_t fcupdates;
	fcupdates.w = (float*)ccmalloc(sizeof(float) * (convnet->layers[3].wnum + convnet->layers[3].net.full_connect.count));
	memset(fcupdates.w, 0, sizeof(float) * (convnet->layers[3].wnum + convnet->layers[3].net.full_connect.count));
	fcupdates.bias = fcupdates.w + convnet->layers[3].wnum;
	elapsed_time = get_current_time();
	for (i = 0; i < batch; i++)
	{
		// check convolutional forward propagate
		ccv_dense_matrix_t* b = 0;
		_ccv_convnet_convolutional_forward_propagate(convnet->layers, a[i], 0, &b);
		for (k = 0; k < convnet->layers->net.convolutional.count; k++)
			for (j = 0; j < out_rows * out_cols; j++)
			{
				float o = b->data.f32[j * convnet->layers->net.convolutional.count + k];
				float oo = out[j * batch + i + k * out_rows * out_cols * batch];
				float delta = _ccv_relative_delta(o, oo);
				assert(!isnan(delta) && !isinf(delta));
				if (delta > 0.001)
					printf("forwprop: %d %d %f %f %f\n", k, j, delta, o, oo);
			}

		// check max pool forward propagate
		ccv_dense_matrix_t* c = 0;
		_ccv_convnet_max_pool_forward_propagate(convnet->layers + 1, b, &c);
		assert(CCV_GET_CHANNEL(c->type) == convnet->layers->net.convolutional.count);
		for (k = 0; k < convnet->layers->net.convolutional.count; k++)
			for (j = 0; j < max_rows * max_cols; j++)
			{
				float m = c->data.f32[j * convnet->layers->net.convolutional.count + k];
				float om = max_pooled[j * batch + i + k * max_rows * max_cols * batch];
				float delta = _ccv_relative_delta(m, om);
				assert(!isnan(delta) && !isinf(delta));
				if (delta > 0.001)
					printf("maxpool: %d %d %f %f %f\n", k, j, delta, m, om);
			}

		// check average pool forward propagate
		ccv_dense_matrix_t* d = 0;
		_ccv_convnet_average_pool_forward_propagate(convnet->layers + 2, b, &d);
		assert(CCV_GET_CHANNEL(d->type) == convnet->layers->net.convolutional.count);
		for (k = 0; k < convnet->layers->net.convolutional.count; k++)
			for (j = 0; j < average_rows * average_cols; j++)
			{
				float a = d->data.f32[j * convnet->layers->net.convolutional.count + k];
				float oa = average_pooled[j * batch + i + k * max_rows * max_cols * batch];
				float delta = _ccv_relative_delta(a, oa);
				assert(!isnan(delta) && !isinf(delta));
				if (delta > 0.001)
					printf("avgpool: %d %d %f %f %f\n", k, j, delta, a, oa);
			}

		// check full connect forward propagate
		ccv_dense_matrix_t* g = ccv_dense_matrix_new(27, 27, CCV_32F | 5, 0, 0);
		for (k = 0; k < 5; k++)
			for (j = 0; j < average_rows * average_cols; j++)
				g->data.f32[j * 5 + k] = d->data.f32[j * convnet->layers->net.convolutional.count + k];
		ccv_dense_matrix_t* h = 0;
		_ccv_convnet_full_connect_forward_propagate(convnet->layers + 3, g, 0, &h);
		for (k = 0; k < convnet->layers[3].net.full_connect.count; k++)
		{
			float f = h->data.f32[k];
			float of = full_connected[k * batch + i];
			float delta = _ccv_relative_delta(f, of);
			assert(!isnan(delta) && !isinf(delta));
			if (delta > 0.001)
				printf("fc: %d %f %f %f\n", k, delta, f, of);
		}

		// check convolutional backward propagate
		ccv_dense_matrix_t* backprop = 0;
		_ccv_convnet_convolutional_backward_propagate(convnet->layers, b, b, 0, a[i], &backprop, &updates);
		for (k = 0; k < ch; k++)
			for (j = 0; j < rows * cols; j++)
			{
				float g = backprop->data.f32[j * ch + k];
				float og = out_input_grad[j * batch + i + k * rows * cols * batch];
				float delta = _ccv_relative_delta(g, og);
				assert(!isnan(delta) && !isinf(delta));
				if (delta > 0.01)
					printf("backprop: %d %d %f %f %f\n", k, j, delta, g, og);
			}

		// check max pool backward propagate
		ccv_dense_matrix_t* e = 0;
		_ccv_convnet_max_pool_backward_propagate(convnet->layers + 1, c, c, b, &e);
		assert(e->rows == out_rows && e->cols == out_cols);
		for (k = 0; k < convnet->layers->net.convolutional.count; k++)
			for (j = 0; j < out_rows * out_cols; j++)
			{
				float m = e->data.f32[j * convnet->layers->net.convolutional.count + k];
				float om = max_pooled_out_input_grad[j * batch + i + k * out_rows * out_cols * batch];
				float delta = _ccv_relative_delta(m, om);
				if (delta > 0.001)
					printf("maxpool backprop: %d %d %f %f %f\n", k, j, delta, m, om);
			}

		// check average pool backward propagate
		ccv_dense_matrix_t* f = 0;
		_ccv_convnet_average_pool_backward_propagate(convnet->layers + 1, d, b, &f);
		assert(f->rows == out_rows && f->cols == out_cols);
		for (k = 0; k < convnet->layers->net.convolutional.count; k++)
			for (j = 0; j < out_rows * out_cols; j++)
			{
				float a = f->data.f32[j * convnet->layers->net.convolutional.count + k];
				float oa = average_pooled_out_input_grad[j * batch + i + k * out_rows * out_cols * batch];
				float delta = _ccv_relative_delta(a, oa);
				if (delta > 0.001)
					printf("avgpool backprop: %d %d %f %f %f\n", k, j, delta, a, oa);
			}

		// check full connect backward propagate
		ccv_dense_matrix_t* p = 0;
		_ccv_convnet_full_connect_backward_propagate(convnet->layers + 3, h, 0, g, &p, &fcupdates);
		for (k = 0; k < 5; k++)
			for (j = 0; j < average_rows * average_cols; j++)
			{
				float f = p->data.f32[j * 5 + k];
				float of = full_connected_grad[(k * average_rows * average_cols + j) * batch + i];
				float delta = _ccv_relative_delta(f, of);
				if (delta > 0.0001)
					printf("fc backprop: %d %f %f %f\n", j, delta, f, of);
			}

		ccv_matrix_free(b);
		ccv_matrix_free(c);
		ccv_matrix_free(d);
		ccv_matrix_free(e);
		ccv_matrix_free(f);
		ccv_matrix_free(g);
		ccv_matrix_free(h);
		ccv_matrix_free(p);
		ccv_matrix_free(backprop);
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
					for (z = 1; z < 8 * out_rows; z++)
						ow += out_weights[z * filter_rows * filter_cols * filter_count * ch + (i * filter_cols + j) * filter_count + k + c * filter_cols * filter_rows * filter_count];
					float delta = _ccv_relative_delta(w, ow);
					if (delta > 0.0001)
						printf("convw: %d,%d,%d,%d: %f, %f\n", i, j, k, c, w, ow);
				}
	for (i = 0; i < filter_count; i++)
	{
		float b = updates.bias[i];
		float ob = out_bias[i];
		float delta = _ccv_relative_delta(b, ob);
		if (delta > 0.0001)
			printf("convb: %d: %f, %f\n", i, b, ob);
	}
	for (i = 0; i < convnet->layers[3].net.full_connect.count; i++)
		for (j = 0; j < average_rows * average_cols; j++)
			for (k = 0; k < 5; k++)
			{
				float w = fcupdates.w[i * average_rows * average_cols * 5 + j * 5 + k];
				float ow = out_fcw[i * average_rows * average_cols * 5 + k * average_rows * average_cols + j];
				float delta = _ccv_relative_delta(w, ow);
				if (delta > 0.01)
					printf("fcw: %d: %f %f,%f\n", i, delta, w, ow);
			}
	for (i = 0; i < convnet->layers[3].net.full_connect.count; i++)
	{
		float b = fcupdates.bias[i];
		float ob = out_fcbias[i];
		float delta = _ccv_relative_delta(b, ob);
		if (delta > 0.0001)
			printf("fcb: %d: %f %f,%f\n", i, delta, b, ob);
	}
}

void cwc_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch)
{
}

void cwc_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, ccv_convnet_train_param_t params)
{
	assert(categorizeds->rnum >= 128);
	if (!GPU(convnet))
		_cwc_convnet_reserve_onto_device(convnet, 128);
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
	cwc_convnet_encode(convnet, a, 0, 128);
}
