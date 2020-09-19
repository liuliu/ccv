#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "3rdparty/khash/khash.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#else
#include "3rdparty/sfmt/SFMT.h"
#endif
#ifdef CCV_BLOCK_SUPPORT
#include <Block.h>
#endif

KHASH_MAP_INIT_INT64(ctx, ccv_array_t*)

struct ccv_cnnp_dataframe_s {
	int row_count;
	int column_size;
	int* shuffled_idx;
#ifdef HAVE_GSL
	gsl_rng* rng;
#else
	sfmt_t sfmt;
#endif
	khash_t(ctx)* data_ctx; // The stream context based cache for data entity of columns. This helps us to avoid allocations when iterate through data.
	ccv_array_t* derived_column_data;
	ccv_cnnp_column_data_t column_data[1];
};

typedef struct {
	int stream_type;
	int column_idx_size;
	int enum_block_type;
	int map_block_type;
	int* column_idxs;
	union {
		ccv_cnnp_column_data_enum_f data_enum;
#ifdef CCV_BLOCK_SUPPORT
		ccv_cnnp_column_data_enum_d data_enum_d;
#endif
	};
	ccv_cnnp_column_data_deinit_f data_deinit;
	void* context;
	ccv_cnnp_column_data_context_deinit_f context_deinit;
	union {
		ccv_cnnp_column_data_map_f map;
#ifdef CCV_BLOCK_SUPPORT
		ccv_cnnp_column_data_map_d map_d;
#endif
	};
} ccv_cnnp_derived_column_data_t;

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_new(const ccv_cnnp_column_data_t* const column_data, const int column_size, const int row_count)
{
	assert(column_size >= 0);
	ccv_cnnp_dataframe_t* const dataframe = (ccv_cnnp_dataframe_t*)cccalloc(1, sizeof(ccv_cnnp_dataframe_t) + sizeof(ccv_cnnp_column_data_t) * (column_size - 1));
	dataframe->row_count = row_count;
	dataframe->column_size = column_size;
	dataframe->data_ctx = kh_init(ctx);
	if (column_size > 0)
		memcpy(dataframe->column_data, column_data, sizeof(ccv_cnnp_column_data_t) * column_size);
	return dataframe;
}

void ccv_cnnp_dataframe_shuffle(ccv_cnnp_dataframe_t* const dataframe)
{
	assert(dataframe->row_count);
	int i;
	if (!dataframe->shuffled_idx)
	{
		dataframe->shuffled_idx = (int*)ccmalloc(sizeof(int) * dataframe->row_count);
		for (i = 0; i < dataframe->row_count; i++)
			dataframe->shuffled_idx[i] = i;
#ifdef HAVE_GSL
		assert(!dataframe->rng);
		gsl_rng_env_setup();
		dataframe->rng = gsl_rng_alloc(gsl_rng_default);
		gsl_rng_set(dataframe->rng, (unsigned long int)(uintptr_t)dataframe);
#else
		sfmt_init_gen_rand(&dataframe->sfmt, (uint32_t)(uintptr_t)dataframe);
#endif
	}
#ifdef HAVE_GSL
	gsl_ran_shuffle(dataframe->rng, dataframe->shuffled_idx, dataframe->row_count, sizeof(int));
#else
	sfmt_genrand_shuffle(&dataframe->sfmt, dataframe->shuffled_idx, dataframe->row_count, sizeof(int));
#endif
}

int ccv_cnnp_dataframe_row_count(ccv_cnnp_dataframe_t* const dataframe)
{
	return dataframe->row_count;
}

int ccv_cnnp_dataframe_add(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_enum_f data_enum, const int stream_type, ccv_cnnp_column_data_deinit_f data_deinit, void* const context, ccv_cnnp_column_data_context_deinit_f context_deinit)
{
	if (!dataframe->derived_column_data)
		dataframe->derived_column_data = ccv_array_new(sizeof(ccv_cnnp_derived_column_data_t), 1, 0);
	ccv_cnnp_derived_column_data_t column_data = {
		.stream_type = stream_type,
		.enum_block_type = CCV_FUNCTION_POINTER,
		.data_enum = data_enum,
		.data_deinit = data_deinit,
		.context = context,
		.context_deinit = context_deinit,
	};
	ccv_array_push(dataframe->derived_column_data, &column_data);
	return dataframe->column_size + dataframe->derived_column_data->rnum - 1;
}

#ifdef CCV_BLOCK_SUPPORT
int ccv_cnnp_dataframe_add_d(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_enum_d data_enum, const int stream_type, ccv_cnnp_column_data_deinit_f data_deinit, void* const context, ccv_cnnp_column_data_context_deinit_f context_deinit)
{
	if (!dataframe->derived_column_data)
		dataframe->derived_column_data = ccv_array_new(sizeof(ccv_cnnp_derived_column_data_t), 1, 0);
	ccv_cnnp_derived_column_data_t column_data = {
		.stream_type = stream_type,
		.enum_block_type = CCV_FUNCTION_BLOCK,
		.data_enum_d = Block_copy(data_enum),
		.data_deinit = data_deinit,
		.context = context,
		.context_deinit = context_deinit,
	};
	ccv_array_push(dataframe->derived_column_data, &column_data);
	return dataframe->column_size + dataframe->derived_column_data->rnum - 1;
}
#endif

int ccv_cnnp_dataframe_map(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_map_f map, const int stream_type, ccv_cnnp_column_data_deinit_f data_deinit, const int* const column_idxs, const int column_idx_size, void* const context, ccv_cnnp_column_data_context_deinit_f context_deinit)
{
	assert(column_idx_size > 0);
	if (!dataframe->derived_column_data)
		dataframe->derived_column_data = ccv_array_new(sizeof(ccv_cnnp_derived_column_data_t), 1, 0);
	const int column_size = dataframe->column_size + dataframe->derived_column_data->rnum;
	int i;
	for (i = 0; i < column_idx_size; i++)
		{ assert(column_idxs[i] < column_size); }
	ccv_cnnp_derived_column_data_t column_data = {
		.stream_type = stream_type,
		.column_idx_size = column_idx_size,
		.map_block_type = CCV_FUNCTION_POINTER,
		.column_idxs = (int*)ccmalloc(sizeof(int) * column_idx_size),
		.map = map,
		.data_deinit = data_deinit,
		.context = context,
		.context_deinit = context_deinit,
	};
	memcpy(column_data.column_idxs, column_idxs, sizeof(int) * column_idx_size);
	ccv_array_push(dataframe->derived_column_data, &column_data);
	return dataframe->column_size + dataframe->derived_column_data->rnum - 1;
}

#ifdef CCV_BLOCK_SUPPORT
int ccv_cnnp_dataframe_map_d(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_map_d map, const int stream_type, ccv_cnnp_column_data_deinit_f data_deinit, const int* const column_idxs, const int column_idx_size, void* const context, ccv_cnnp_column_data_context_deinit_f context_deinit)
{
	assert(column_idx_size > 0);
	if (!dataframe->derived_column_data)
		dataframe->derived_column_data = ccv_array_new(sizeof(ccv_cnnp_derived_column_data_t), 1, 0);
	const int column_size = dataframe->column_size + dataframe->derived_column_data->rnum;
	int i;
	for (i = 0; i < column_idx_size; i++)
		{ assert(column_idxs[i] < column_size); }
	ccv_cnnp_derived_column_data_t column_data = {
		.stream_type = stream_type,
		.column_idx_size = column_idx_size,
		.map_block_type = CCV_FUNCTION_BLOCK,
		.column_idxs = (int*)ccmalloc(sizeof(int) * column_idx_size),
		.map_d = Block_copy(map),
		.data_deinit = data_deinit,
		.context = context,
		.context_deinit = context_deinit,
	};
	memcpy(column_data.column_idxs, column_idxs, sizeof(int) * column_idx_size);
	ccv_array_push(dataframe->derived_column_data, &column_data);
	return dataframe->column_size + dataframe->derived_column_data->rnum - 1;
}
#endif

void* ccv_cnnp_dataframe_column_context(const ccv_cnnp_dataframe_t* const dataframe, const int column_idx)
{
	assert(column_idx >= 0);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	assert(column_idx < column_size);
	if (column_idx < dataframe->column_size)
		return dataframe->column_data[column_idx].context;
	assert(dataframe->derived_column_data);
	ccv_cnnp_derived_column_data_t* const derived_column_data = (ccv_cnnp_derived_column_data_t*)ccv_array_get(dataframe->derived_column_data, column_idx - dataframe->column_size);
	return derived_column_data->context;
}

typedef struct {
	int flag; // Mark this as cached or not.
	uint64_t ctx; // The stream context.
	void* data;
} ccv_cnnp_dataframe_data_item_t;

typedef struct {
	ccv_nnc_stream_context_t* stream_context;
	ccv_nnc_stream_signal_t* signal;
} ccv_cnnp_dataframe_column_ctx_t;

KHASH_MAP_INIT_INT64(iter_ctx, ccv_cnnp_dataframe_column_ctx_t*)

struct ccv_cnnp_dataframe_iter_s {
	int flag; // Whether we called next or not.
	int idx;
	int prefetch_head;
	int prefetch_tail;
	int column_idx_size;
	int fetched_size; // The size of fetched data.
	ccv_cnnp_dataframe_t* dataframe;
	void**** derived_data; // This is ridiculous, but it is true.
	void** fetched_data; // The cache to store fetched data.
	khash_t(iter_ctx)* column_ctx; // Column context specific to a stream context. The key will be a parent stream context and value will be child stream context + signal.
	ccv_array_t* prefetches; // The prefetch contents.
	int* column_idxs;
	ccv_cnnp_dataframe_data_item_t cached_data[1]; // The data cached when deriving data.
};

#define INDEX_DATA(iter) ((int*)((iter)->fetched_data))
#define FETCHED_DATA(iter, idx) ((iter)->fetched_data + ((idx) + 1) * (iter)->fetched_size)

ccv_cnnp_dataframe_iter_t* ccv_cnnp_dataframe_iter_new(ccv_cnnp_dataframe_t* const dataframe, const int* const column_idxs, const int column_idx_size)
{
	assert(column_idx_size > 0);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	int i;
	for (i = 0; i < column_idx_size; i++)
		{ assert(column_idxs[i] < column_size); }
	ccv_cnnp_dataframe_iter_t* const iter = (ccv_cnnp_dataframe_iter_t*)cccalloc(1, sizeof(ccv_cnnp_dataframe_iter_t) + sizeof(ccv_cnnp_dataframe_data_item_t) * column_size + sizeof(void*) * (column_idx_size - 1) + sizeof(int) * column_idx_size);
	iter->dataframe = dataframe;
	iter->prefetch_tail = -1;
	iter->column_idx_size = column_idx_size;
	iter->column_idxs = (int*)(iter->cached_data + column_size);
	memcpy(iter->column_idxs, column_idxs, sizeof(int) * column_idx_size);
	// Preallocate fetched data.
	iter->fetched_size = 1;
	iter->fetched_data = (void**)ccmalloc(sizeof(void*) * (column_size + 1));
	return iter;
}

static void _ccv_cnnp_dataframe_enqueue_data(ccv_cnnp_dataframe_t* const dataframe, void* const data, const int column_idx, const uint64_t ctx)
{
	if (!data)
		return;
	khash_t(ctx)* const data_ctx = dataframe->data_ctx;
	int ret = 0;
	khiter_t k = kh_put(ctx, data_ctx, ctx, &ret);
	assert(ret >= 0);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	assert(column_idx < column_size);
	// If ret == 0, the key already exist, we can get the columns directly, otherwise, create and assign back.
	ccv_array_t* const columns = (ret == 0) ? kh_val(data_ctx, k) : ccv_array_new(sizeof(ccv_array_t*), column_size, 0);
	if (ret != 0)
		kh_val(data_ctx, k) = columns;
	if (columns->rnum < column_size)
		ccv_array_resize(columns, column_size);
	ccv_array_t* column = *(ccv_array_t**)ccv_array_get(columns, column_idx);
	if (!column)
	{
		column = ccv_array_new(sizeof(void*), 1, 0);
		*(ccv_array_t**)ccv_array_get(columns, column_idx) = column;
	}
	ccv_array_push(column, &data);
}

static void* _ccv_cnnp_dataframe_dequeue_data(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, ccv_nnc_stream_context_t* const stream_context)
{
	const uint64_t ctx = (uint64_t)(uintptr_t)stream_context;
	khash_t(ctx)* const data_ctx = dataframe->data_ctx;
	khiter_t k = kh_get(ctx, data_ctx, ctx);
	if (k == kh_end(data_ctx))
		return 0;
	ccv_array_t* const columns = kh_val(data_ctx, k);
	if (column_idx >= columns->rnum)
		return 0;
	ccv_array_t* const column = *(ccv_array_t**)ccv_array_get(columns, column_idx);
	if (!column || column->rnum == 0)
		return 0;
	void* const data = *(void**)ccv_array_get(column, column->rnum - 1);
	--column->rnum;
	return data;
}

static ccv_cnnp_dataframe_column_ctx_t _ccv_cnnp_child_column_ctx_for_stream_type(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_dataframe_iter_t* const iter, const int column_idx, ccv_nnc_stream_context_t* const stream_context, const int stream_type)
{
	ccv_cnnp_dataframe_column_ctx_t child_ctx = {
		.stream_context = stream_context,
	};
	if (stream_context && ccv_nnc_stream_context_type(stream_context) != stream_type && stream_type != 0)
	{
		if (!iter->column_ctx)
			iter->column_ctx = kh_init(iter_ctx);
		khash_t(iter_ctx)* const column_ctx = iter->column_ctx;
		int ret = 0;
		khiter_t k = kh_put(iter_ctx, column_ctx, (uint64_t)(uintptr_t)stream_context, &ret);
		assert(ret >= 0);
		const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
		ccv_cnnp_dataframe_column_ctx_t* const ctx = (ret == 0) ? kh_val(column_ctx, k) : cccalloc(column_size, sizeof(ccv_cnnp_dataframe_column_ctx_t));
		if (ret != 0)
			kh_val(column_ctx, k) = ctx;
		if (!ctx[column_idx].stream_context)
			ctx[column_idx].stream_context = ccv_nnc_stream_context_new(stream_type);
		if (!ctx[column_idx].signal)
			ctx[column_idx].signal = ccv_nnc_stream_signal_new(stream_type);
		child_ctx = ctx[column_idx];
	}
	return child_ctx;
}

static void _ccv_cnnp_dataframe_column_data(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_dataframe_iter_t* const iter, ccv_cnnp_dataframe_data_item_t* const cached_data, void** const fetched_data, const int* const row_idxs, const int row_size, const int column_idx, const int cached_step, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	if (cached_data[column_idx * cached_step].flag)
	{
		for (i = 1; i < row_size; i++)
			{ assert(cached_data[i + column_idx * cached_step].flag); }
		for (i = 0; i < row_size; i++)
			fetched_data[i] = cached_data[i + column_idx * cached_step].data;
		return;
	} else {
		for (i = 1; i < row_size; i++)
			{ assert(!cached_data[i + column_idx * cached_step].flag); }
		for (i = 0; i < row_size; i++)
			fetched_data[i] = _ccv_cnnp_dataframe_dequeue_data(dataframe, column_idx, stream_context);
	}
	if (column_idx >= dataframe->column_size)
	{
		assert(dataframe->derived_column_data);
		const int derived_column_idx = column_idx - dataframe->column_size;
		const ccv_cnnp_derived_column_data_t* const derived_column_data = (ccv_cnnp_derived_column_data_t*)ccv_array_get(dataframe->derived_column_data, derived_column_idx);
		ccv_cnnp_dataframe_column_ctx_t child_ctx = _ccv_cnnp_child_column_ctx_for_stream_type(dataframe, iter, column_idx, stream_context, derived_column_data->stream_type);
		const int column_idx_size = derived_column_data->column_idx_size;
		if (derived_column_data->map)
		{
			int i;
			if (!iter->derived_data)
				iter->derived_data = (void****)cccalloc(dataframe->derived_column_data->rnum, sizeof(void***));
			if (!iter->derived_data[derived_column_idx])
				iter->derived_data[derived_column_idx] = (void***)cccalloc(derived_column_data->column_idx_size, sizeof(void**));
			void*** const derived_data = iter->derived_data[derived_column_idx];
			for (i = 0; i < column_idx_size; i++)
			{
				derived_data[i] = FETCHED_DATA(iter, derived_column_data->column_idxs[i]);
				_ccv_cnnp_dataframe_column_data(dataframe, iter, cached_data, derived_data[i], row_idxs, row_size, derived_column_data->column_idxs[i], cached_step, stream_context);
			}
			// Mark it as const.
			if (derived_column_data->map_block_type == CCV_FUNCTION_POINTER)
				derived_column_data->map((void *const *const *)derived_data, derived_column_data->column_idx_size, row_size, fetched_data, derived_column_data->context, child_ctx.stream_context);
#ifdef CCV_BLOCK_SUPPORT
			else if (derived_column_data->map_block_type == CCV_FUNCTION_BLOCK)
				derived_column_data->map_d((void *const *const *)derived_data, derived_column_data->column_idx_size, row_size, fetched_data, derived_column_data->context, child_ctx.stream_context);
#endif
		} else
			if (derived_column_data->enum_block_type == CCV_FUNCTION_POINTER)
				derived_column_data->data_enum(column_idx, row_idxs, row_size, fetched_data, derived_column_data->context, child_ctx.stream_context);
#ifdef CCV_BLOCK_SUPPORT
			else if (derived_column_data->enum_block_type == CCV_FUNCTION_BLOCK)
				derived_column_data->data_enum_d(column_idx, row_idxs, row_size, fetched_data, derived_column_data->context, child_ctx.stream_context);
#endif
		if (child_ctx.stream_context != stream_context)
		{
			ccv_nnc_stream_context_emit_signal(child_ctx.stream_context, child_ctx.signal);
			ccv_nnc_stream_context_wait_signal(stream_context, child_ctx.signal);
		}
	} else {
		const ccv_cnnp_column_data_t* const column_data = dataframe->column_data + column_idx;
		ccv_cnnp_dataframe_column_ctx_t child_ctx = _ccv_cnnp_child_column_ctx_for_stream_type(dataframe, iter, column_idx, stream_context, column_data->stream_type);
		if (column_data->block_type == CCV_FUNCTION_POINTER)
			column_data->data_enum(column_idx, row_idxs, row_size, fetched_data, column_data->context, child_ctx.stream_context);
#ifdef CCV_BLOCK_SUPPORT
		else if (column_data->block_type == CCV_FUNCTION_BLOCK)
			column_data->data_enum_d(column_idx, row_idxs, row_size, fetched_data, column_data->context, child_ctx.stream_context);
#endif
		if (child_ctx.stream_context != stream_context)
		{
			ccv_nnc_stream_context_emit_signal(child_ctx.stream_context, child_ctx.signal);
			ccv_nnc_stream_context_wait_signal(stream_context, child_ctx.signal);
		}
	}
	for (i = 0; i < row_size; i++)
	{
		cached_data[i + column_idx * cached_step].flag = 1;
		cached_data[i + column_idx * cached_step].ctx = (uint64_t)(uintptr_t)stream_context;
		cached_data[i + column_idx * cached_step].data = fetched_data[i];
	}
}

int ccv_cnnp_dataframe_iter_next(ccv_cnnp_dataframe_iter_t* const iter, void** const data_ref, const int column_idx_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(column_idx_size <= iter->column_idx_size);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	int i;
	// Push existing data back to reusable state (note, these may not be reused immediately because they may be on a different stream context).
	for (i = 0; i < column_size; i++)
		if (iter->cached_data[i].flag)
		{
			_ccv_cnnp_dataframe_enqueue_data(dataframe, iter->cached_data[i].data, i, iter->cached_data[i].ctx);
			iter->cached_data[i].flag = 0;
			iter->cached_data[i].data = 0;
			iter->cached_data[i].ctx = 0;
		}
	const int idx = iter->idx;
	iter->flag = 1; // Mark it as we called next already.
	if (idx > dataframe->row_count) // If we exceed row count, return -2.
		return -2;
	if (idx == dataframe->row_count) // Otherwise, no more row, return -1.
	{
		++iter->idx;
		return -1;
	}
	if (iter->prefetch_tail != -1) // If there is something in prefetch log.
	{
		ccv_array_t* const prefetches = iter->prefetches;
		assert(prefetches);
		const int lines = prefetches->rnum / column_size;
		if (iter->prefetch_head == iter->prefetch_tail) // Only one item.
			iter->prefetch_tail = -1;
		ccv_cnnp_dataframe_data_item_t* const cached_data = (ccv_cnnp_dataframe_data_item_t*)ccv_array_get(iter->prefetches, iter->prefetch_head);
		for (i = 0; i < column_size; i++)
		{
			if (!cached_data[i * lines].flag)
				continue;
			if (cached_data[i * lines].ctx == (uint64_t)(uintptr_t)stream_context) // If match existing stream context.
				iter->cached_data[i] = cached_data[i * lines];
			else // Recycle
				_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[i * lines].data, i, cached_data[i * lines].ctx);
		}
		++iter->prefetch_head;
		assert(prefetches->rnum % column_size == 0);
		if (iter->prefetch_head >= lines)
			iter->prefetch_head = 0;
	}
	for (i = 0; i < column_idx_size; i++)
	{
		void* fetched_data[1]; // This guards better than just give away data_ref + i.
		_ccv_cnnp_dataframe_column_data(dataframe, iter, iter->cached_data, fetched_data, dataframe->shuffled_idx ? dataframe->shuffled_idx + idx : &idx, 1, iter->column_idxs[i], 1, stream_context);
		data_ref[i] = fetched_data[0];
	}
	++iter->idx;
	return 0;
}

void ccv_cnnp_dataframe_iter_peek(ccv_cnnp_dataframe_iter_t* const iter, void** const data_ref, const int offset, const int data_ref_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(iter->flag);
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(offset + data_ref_size <= iter->column_idx_size);
	const int idx = iter->idx - 1; // next is called, therefore, index is already incremented.
	assert(idx >= 0);
	assert(idx < dataframe->row_count);
	int i;
	for (i = 0; i < data_ref_size; i++)
		_ccv_cnnp_dataframe_column_data(dataframe, iter, iter->cached_data, data_ref + i, dataframe->shuffled_idx ? dataframe->shuffled_idx + idx : &idx, 1, iter->column_idxs[i + offset], 1, stream_context);
}

static void _ccv_cnnp_null_prefetches(ccv_cnnp_dataframe_iter_t* const iter)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	int i, j;
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	if (iter->prefetch_head <= iter->prefetch_tail)
	{
		assert(iter->prefetches);
		const int lines = iter->prefetches->rnum / column_size;
		for (i = iter->prefetch_head; i <= iter->prefetch_tail; i++)
		{
			ccv_cnnp_dataframe_data_item_t* const cached_data = ccv_array_get(iter->prefetches, i);
			for (j = 0; j < column_size; j++)
				if (cached_data[j * lines].flag)
					_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[j * lines].data, j, cached_data[j * lines].ctx);
		}
	} else if (iter->prefetch_tail >= 0) { // -1 means no item.
		assert(iter->prefetches);
		const int lines = iter->prefetches->rnum / column_size;
		for (i = iter->prefetch_head; i < lines; i++)
		{
			ccv_cnnp_dataframe_data_item_t* const cached_data = ccv_array_get(iter->prefetches, i);
			for (j = 0; j < column_size; j++)
				if (cached_data[j * lines].flag)
					_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[j * lines].data, j, cached_data[j * lines].ctx);
		}
		for (i = 0; i <= iter->prefetch_tail; i++)
		{
			ccv_cnnp_dataframe_data_item_t* const cached_data = ccv_array_get(iter->prefetches, i);
			for (j = 0; j < column_size; j++)
				if (cached_data[j * lines].flag)
					_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[j * lines].data, j, cached_data[j * lines].ctx);
		}
	}
	iter->prefetch_head = 0;
	iter->prefetch_tail = -1;
}

static void _ccv_cnnp_prefetch_cached_data(ccv_cnnp_dataframe_iter_t* const iter, ccv_cnnp_dataframe_data_item_t* const cached_data, const int idx, const int max_to_prefetch, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	assert(iter->prefetches);
	const int lines = iter->prefetches->rnum / column_size;
	int i, j;
	// Reset
	for (i = 0; i < column_size; i++)
		for (j = 0; j < max_to_prefetch; j++)
		{
			cached_data[j + i * lines].flag = 0;
			cached_data[j + i * lines].data = 0;
			cached_data[j + i * lines].ctx = 0;
		}
	if (iter->fetched_size < max_to_prefetch)
	{
		iter->fetched_data = ccrealloc(iter->fetched_data, sizeof(void*) * max_to_prefetch * (column_size + 1));
		iter->fetched_size = max_to_prefetch;
	}
	if (dataframe->shuffled_idx)
		for (i = 0; i < iter->column_idx_size; i++)
			_ccv_cnnp_dataframe_column_data(dataframe, iter, cached_data, FETCHED_DATA(iter, iter->column_idxs[i]), dataframe->shuffled_idx + idx, max_to_prefetch, iter->column_idxs[i], lines, stream_context);
	else {
		for (i = 0; i < max_to_prefetch; i++)
			INDEX_DATA(iter)[i] = idx + i;
		for (i = 0; i < iter->column_idx_size; i++)
			_ccv_cnnp_dataframe_column_data(dataframe, iter, cached_data, FETCHED_DATA(iter, iter->column_idxs[i]), INDEX_DATA(iter), max_to_prefetch, iter->column_idxs[i], lines, stream_context);
	}
}

int ccv_cnnp_dataframe_iter_prefetch(ccv_cnnp_dataframe_iter_t* const iter, const int prefetch_count, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	int i, j;
	assert(iter->idx <= dataframe->row_count);
	int lines, next, max_to_prefetch;
	if (iter->prefetch_tail == -1)
	{
		if (iter->idx == dataframe->row_count)
			return -1; // Cannot be done.
		max_to_prefetch = ccv_min(dataframe->row_count - iter->idx, prefetch_count);
		if (!iter->prefetches)
		{
			iter->prefetches = ccv_array_new(sizeof(ccv_cnnp_dataframe_data_item_t), max_to_prefetch * column_size, 0);
			ccv_array_resize(iter->prefetches, max_to_prefetch * column_size);
		}
		iter->prefetch_tail = iter->prefetch_head = 0; // Advance!
		next = iter->idx;
		lines = iter->prefetches->rnum / column_size;
		// Reset to enough space.
		if (lines < max_to_prefetch)
		{
			ccv_array_resize(iter->prefetches, max_to_prefetch * column_size);
			lines = max_to_prefetch;
		}
	} else {
		assert(iter->prefetches);
		ccv_array_t* const prefetches = iter->prefetches;
		assert(prefetches->rnum % column_size == 0);
		lines = prefetches->rnum / column_size;
		const int prefetched = iter->prefetch_tail >= iter->prefetch_head ? iter->prefetch_tail - iter->prefetch_head + 1: lines - iter->prefetch_head + iter->prefetch_tail + 1;
		if (iter->idx + prefetched == dataframe->row_count) // Nothing to prefetch.
			return -1;
		max_to_prefetch = ccv_min(dataframe->row_count - (iter->idx + prefetched), prefetch_count);
		// Not enough space, need to resize.
		if (prefetched + max_to_prefetch > lines)
		{
			const int new_lines = prefetched + max_to_prefetch;
			ccv_array_resize(prefetches, new_lines * column_size);
			// These are overlap moves, have to make sure start from the end and move it up to the beginning.
			if (iter->prefetch_head > iter->prefetch_tail)
			{
				const int offset = new_lines - lines;
				for (i = column_size - 1; i >= 0; i--)
				{
					for (j = lines - 1; j >= iter->prefetch_head; j--)
						*(ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, j + offset + i * new_lines) = *(ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, j + i * lines);
					for (j = iter->prefetch_tail; j >= 0; j--)
						*(ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, j + i * new_lines) = *(ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, j + i * lines);
				}
				iter->prefetch_head += offset;
			} else {
				for (i = column_size - 1; i >= 0; i--)
					for (j = iter->prefetch_tail; j >= iter->prefetch_head; j--)
						*(ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, j + i * new_lines) = *(ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, j + i * lines);
			}
			lines = new_lines;
		}
		++iter->prefetch_tail; // Move to the next ready tail.
		if (iter->prefetch_tail >= lines)
			iter->prefetch_tail = 0;
		next = iter->idx + prefetched;
	}
	ccv_array_t* const prefetches = iter->prefetches;
	ccv_cnnp_dataframe_data_item_t* const cached_data = (ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, iter->prefetch_tail);
	// If the tail is before the head, we must have enough space for the max_to_prefetch
	if (iter->prefetch_tail < iter->prefetch_head)
	{
		assert(iter->prefetch_tail + max_to_prefetch - 1 < iter->prefetch_head);
		_ccv_cnnp_prefetch_cached_data(iter, cached_data, next, max_to_prefetch, stream_context);
		iter->prefetch_tail += max_to_prefetch - 1;
	} else {
		// First, fetch to the end.
		const int fetch_to_end = ccv_min(max_to_prefetch, lines - iter->prefetch_tail);
		_ccv_cnnp_prefetch_cached_data(iter, cached_data, next, fetch_to_end, stream_context);
		if (fetch_to_end == max_to_prefetch)
			iter->prefetch_tail += fetch_to_end - 1;
		else {
			// Need to fetch more.
			ccv_cnnp_dataframe_data_item_t* const more_data = (ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, 0);
			assert(max_to_prefetch > fetch_to_end);
			_ccv_cnnp_prefetch_cached_data(iter, more_data, next + fetch_to_end, max_to_prefetch - fetch_to_end, stream_context);
			iter->prefetch_tail = max_to_prefetch - fetch_to_end - 1;
		}
	}
	return 0;
}

int ccv_cnnp_dataframe_iter_set_cursor(ccv_cnnp_dataframe_iter_t* const iter, const int idx)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	if (idx >= dataframe->row_count)
		return -1;
	if (idx == iter->idx)
		return 0;
	iter->idx = idx;
	iter->flag = 0;
	_ccv_cnnp_null_prefetches(iter);
	return 0;
}

void ccv_cnnp_dataframe_iter_free(ccv_cnnp_dataframe_iter_t* const iter)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	int i;
	// Push existing data back to reusable state (note, these may not be reused immediately because they may be on a different stream context).
	for (i = 0; i < column_size; i++)
		if (iter->cached_data[i].flag)
			_ccv_cnnp_dataframe_enqueue_data(dataframe, iter->cached_data[i].data, i, iter->cached_data[i].ctx);
	// Push prefetches back to reusable state.
	_ccv_cnnp_null_prefetches(iter);
	if (iter->prefetches)
		ccv_array_free(iter->prefetches);
	if (iter->derived_data)
	{
		assert(dataframe->derived_column_data);
		for (i = 0; i < dataframe->derived_column_data->rnum; i++)
			if (iter->derived_data[i])
				ccfree(iter->derived_data[i]);
		ccfree(iter->derived_data);
	}
	ccfree(iter->fetched_data);
	if (iter->column_ctx)
	{
		khash_t(iter_ctx)* const column_ctx = iter->column_ctx;
		khiter_t k;
		for (k = kh_begin(column_ctx); k != kh_end(column_ctx); ++k)
		{
			if (!kh_exist(column_ctx, k))
				continue;
			ccv_cnnp_dataframe_column_ctx_t* const ctx = kh_val(column_ctx, k);
			for (i = 0; i < column_size; i++)
			{
				if (ctx[i].stream_context)
					ccv_nnc_stream_context_free(ctx[i].stream_context);
				if (ctx[i].signal)
					ccv_nnc_stream_signal_free(ctx[i].signal);
			}
		}
		kh_destroy(iter_ctx, column_ctx);
	}
	ccfree(iter);
}

void ccv_cnnp_dataframe_free(ccv_cnnp_dataframe_t* const dataframe)
{
	int i, j;
	khash_t(ctx)* const data_ctx = dataframe->data_ctx;
	khiter_t k;
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	for (k = kh_begin(data_ctx); k != kh_end(data_ctx); ++k)
	{
		if (!kh_exist(data_ctx, k))
			continue;
		ccv_array_t* const columns = kh_val(data_ctx, k);
		assert(columns->rnum <= column_size);
		for (i = 0; i < columns->rnum; i++)
		{
			ccv_array_t* const column = *(ccv_array_t**)ccv_array_get(columns, i);
			if (!column)
				continue;
			void* context;
			ccv_cnnp_column_data_deinit_f data_deinit;
			if (i < dataframe->column_size)
			{
				data_deinit = dataframe->column_data[i].data_deinit;
				context = dataframe->column_data[i].context;
			} else {
				assert(dataframe->derived_column_data);
				ccv_cnnp_derived_column_data_t* const derived_column_data = (ccv_cnnp_derived_column_data_t*)ccv_array_get(dataframe->derived_column_data, i - dataframe->column_size);
				data_deinit = derived_column_data->data_deinit;
				context = derived_column_data->context;
			}
			if (data_deinit)
				for (j = 0; j < column->rnum; j++)
					data_deinit(*(void**)ccv_array_get(column, j), context);
			ccv_array_free(column);
		}
		ccv_array_free(columns);
	}
	kh_destroy(ctx, data_ctx);
	if (dataframe->derived_column_data)
	{
		for (i = 0; i < dataframe->derived_column_data->rnum; i++)
		{
			ccv_cnnp_derived_column_data_t* const derived_column_data = (ccv_cnnp_derived_column_data_t*)ccv_array_get(dataframe->derived_column_data, i);
			if (derived_column_data->context_deinit)
				derived_column_data->context_deinit(derived_column_data->context);
#ifdef CCV_BLOCK_SUPPORT
			if (derived_column_data->enum_block_type == CCV_FUNCTION_BLOCK)
				Block_release(derived_column_data->data_enum_d);
			if (derived_column_data->map_block_type == CCV_FUNCTION_BLOCK)
				Block_release(derived_column_data->map_d);
#endif
			ccfree(derived_column_data->column_idxs);
		}
		ccv_array_free(dataframe->derived_column_data);
	}
	for (i = 0; i < dataframe->column_size; i++)
	{
		if (dataframe->column_data[i].context_deinit)
			dataframe->column_data[i].context_deinit(dataframe->column_data[i].context);
#ifdef CCV_BLOCK_SUPPORT
		if (dataframe->column_data[i].block_type == CCV_FUNCTION_BLOCK)
			Block_release(dataframe->column_data[i].data_enum_d);
#endif
	}
	if (dataframe->shuffled_idx)
		ccfree(dataframe->shuffled_idx);
#ifdef HAVE_GSL
	if (dataframe->rng)
		gsl_rng_free(dataframe->rng);
#endif
	ccfree(dataframe);
}
