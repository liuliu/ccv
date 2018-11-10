#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "3rdparty/khash/khash.h"

KHASH_MAP_INIT_INT64(ctx, ccv_array_t*)

struct ccv_cnnp_dataframe_s {
	int row_size;
	int column_size;
	khash_t(ctx)* data_ctx; // The stream context based cache for data entity of columns. This helps us to avoid allocations when iterate through data.
	ccv_array_t* derived_column_data;
	ccv_cnnp_column_data_t column_data[1];
};

typedef struct {
	int column_idx_size;
	int* column_idxs;
	void** data;
	void* context;
	ccv_cnnp_column_data_deinit_f deinit;
	ccv_cnnp_column_data_map_f map;
} ccv_cnnp_derived_column_data_t;

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_new(const ccv_cnnp_column_data_t* const column_data, const int column_size, const int row_size)
{
	assert(column_size > 0);
	ccv_cnnp_dataframe_t* const dataframe = (ccv_cnnp_dataframe_t*)cccalloc(1, sizeof(ccv_cnnp_dataframe_t) + sizeof(ccv_cnnp_column_data_t) * (column_size - 1));
	dataframe->row_size = row_size;
	dataframe->column_size = column_size;
	dataframe->data_ctx = kh_init(ctx);
	memcpy(dataframe->column_data, column_data, sizeof(ccv_cnnp_column_data_t) * column_size);
	return dataframe;
}

void ccv_cnnp_dataframe_shuffle(ccv_cnnp_dataframe_t* const dataframe)
{
}

int ccv_cnnp_dataframe_map(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_map_f map, ccv_cnnp_column_data_deinit_f deinit, const int* const column_idxs, const int column_idx_size, void* const context)
{
	assert(column_idx_size > 0);
	if (!dataframe->derived_column_data)
		dataframe->derived_column_data = ccv_array_new(sizeof(ccv_cnnp_derived_column_data_t), 1, 0);
	const int column_size = dataframe->column_size + dataframe->derived_column_data->rnum;
	int i;
	for (i = 0; i < column_idx_size; i++)
		{ assert(column_idxs[i] < column_size); }
	ccv_cnnp_derived_column_data_t column_data = {
		.column_idx_size = column_idx_size,
		.data = (void*)ccmalloc(sizeof(void*) * column_idx_size + sizeof(int) * column_idx_size),
		.context = context,
		.map = map,
		.deinit = deinit,
	};
	memset(column_data.data, 0, sizeof(void*) * column_idx_size);
	column_data.column_idxs = (int*)(column_data.data + column_idx_size);
	memcpy(column_data.column_idxs, column_idxs, sizeof(int) * column_idx_size);
	ccv_array_push(dataframe->derived_column_data, &column_data);
	return dataframe->column_size + dataframe->derived_column_data->rnum - 1;
}

typedef struct {
	int flag; // Mark this as cached or not.
	uint64_t ctx; // The stream context.
	void* data;
} ccv_cnnp_dataframe_data_item_t;

struct ccv_cnnp_dataframe_iter_s {
	int idx;
	int prefetch_head;
	int prefetch_tail;
	int column_idx_size;
	ccv_array_t* prefetches; // The prefetch contents.
	ccv_cnnp_dataframe_t* dataframe;
	int* column_idxs;
	ccv_cnnp_dataframe_data_item_t cached_data[1]; // The data cached when deriving data.
};

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

static void* _ccv_cnnp_dataframe_column_data(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_dataframe_data_item_t* const cached_data, const int row_idx, const int column_idx, ccv_nnc_stream_context_t* const stream_context)
{
	if (cached_data[column_idx].flag)
		return cached_data[column_idx].data;
	void* data = _ccv_cnnp_dataframe_dequeue_data(dataframe, column_idx, stream_context);
	if (column_idx >= dataframe->column_size)
	{
		const ccv_cnnp_derived_column_data_t* const derived_column_data = (ccv_cnnp_derived_column_data_t*)ccv_array_get(dataframe->derived_column_data, column_idx - dataframe->column_size);
		const int column_idx_size = derived_column_data->column_idx_size;
		int i;
		for (i = 0; i < column_idx_size; i++)
			derived_column_data->data[i] = _ccv_cnnp_dataframe_column_data(dataframe, cached_data, row_idx, derived_column_data->column_idxs[i], stream_context);
		derived_column_data->map(derived_column_data->data, derived_column_data->column_idx_size, &data, derived_column_data->context, stream_context);
	} else {
		const ccv_cnnp_column_data_t* const column_data = dataframe->column_data + column_idx;
		column_data->data_enum(column_idx, row_idx, 1, &data, column_data->context, stream_context);
	}
	cached_data[column_idx].flag = 1;
	cached_data[column_idx].ctx = (uint64_t)(uintptr_t)stream_context;
	cached_data[column_idx].data = data;
	return data;
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
	if (idx == dataframe->row_size)
		return -1;
	if (iter->prefetch_tail != -1) // If there is something in prefetch log.
	{
		if (iter->prefetch_head == iter->prefetch_tail) // Only one item.
			iter->prefetch_tail = -1;
		ccv_array_t* const prefetches = iter->prefetches;
		assert(prefetches);
		ccv_cnnp_dataframe_data_item_t* const cached_data = (ccv_cnnp_dataframe_data_item_t*)ccv_array_get(iter->prefetches, iter->prefetch_head * column_size);
		for (i = 0; i < column_size; i++)
		{
			if (cached_data[i].ctx == (uint64_t)(uintptr_t)stream_context) // If match existing stream context.
				iter->cached_data[i] = cached_data[i];
			else // Recycle
				_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[i].data, i, cached_data[i].ctx);
		}
		++iter->prefetch_head;
		assert(prefetches->rnum % column_size == 0);
		int lines = prefetches->rnum / column_size;
		if (iter->prefetch_head >= lines)
			iter->prefetch_head = 0;
	}
	for (i = 0; i < column_idx_size; i++)
	{
		const int column_idx = iter->column_idxs[i];
		data_ref[i] = _ccv_cnnp_dataframe_column_data(dataframe, iter->cached_data, idx, column_idx, stream_context);
	}
	++iter->idx;
	return 0;
}

static void _ccv_cnnp_null_prefetches(ccv_cnnp_dataframe_iter_t* const iter)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	int i, j;
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	if (iter->prefetch_head <= iter->prefetch_tail)
	{
		for (i = iter->prefetch_head; i <= iter->prefetch_tail; i++)
		{
			ccv_cnnp_dataframe_data_item_t* const cached_data = ccv_array_get(iter->prefetches, i * column_size);
			for (j = 0; j < column_size; j++)
				_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[j].data, j, cached_data[j].ctx);
		}
	} else if (iter->prefetch_tail >= 0) { // -1 means no item.
		assert(iter->prefetches);
		for (i = iter->prefetch_head; i < iter->prefetches->rnum; i++)
		{
			ccv_cnnp_dataframe_data_item_t* const cached_data = ccv_array_get(iter->prefetches, i * column_size);
			for (j = 0; j < column_size; j++)
				_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[j].data, j, cached_data[j].ctx);
		}
		for (i = 0; i <= iter->prefetch_tail; i++)
		{
			ccv_cnnp_dataframe_data_item_t* const cached_data = ccv_array_get(iter->prefetches, i * column_size);
			for (j = 0; j < column_size; j++)
				_ccv_cnnp_dataframe_enqueue_data(dataframe, cached_data[j].data, j, cached_data[j].ctx);
		}
	}
	iter->prefetch_head = 0;
	iter->prefetch_tail = -1;
}

int ccv_cnnp_dataframe_iter_prefetch(ccv_cnnp_dataframe_iter_t* const iter, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	const int column_size = dataframe->column_size + (dataframe->derived_column_data ? dataframe->derived_column_data->rnum : 0);
	int next;
	if (iter->prefetch_tail == -1)
	{
		if (iter->idx == dataframe->row_size)
			return -1; // Cannot be done.
		if (!iter->prefetches)
		{
			iter->prefetches = ccv_array_new(sizeof(ccv_cnnp_dataframe_data_item_t), column_size, 0);
			ccv_array_resize(iter->prefetches, column_size);
		}
		iter->prefetch_tail = iter->prefetch_head; // Advance!
		next = iter->idx;
	} else {
		assert(iter->prefetches);
		ccv_array_t* const prefetches = iter->prefetches;
		assert(prefetches->rnum % column_size == 0);
		int lines = prefetches->rnum / column_size;
		const int prefetched = iter->prefetch_tail >= iter->prefetch_head ? iter->prefetch_tail - iter->prefetch_head + 1: lines - iter->prefetch_head + iter->prefetch_tail + 1;
		if (iter->idx + prefetched == dataframe->row_size)
			return -1; // Cannot be done.
		// This is full, because tail is next to the head. Make room by resize the prefetches to be 1 line longer, and move everything to make space for prefetch_tail.
		if ((iter->prefetch_head + lines - 1) % lines == iter->prefetch_tail)
		{
			ccv_array_resize(prefetches, prefetches->rnum + column_size);
			if (iter->prefetch_head > iter->prefetch_tail)
			{
				assert(iter->prefetch_head == iter->prefetch_tail + 1);
				memmove(ccv_array_get(prefetches, (iter->prefetch_head + 1) * column_size), ccv_array_get(prefetches, iter->prefetch_head * column_size), sizeof(ccv_cnnp_dataframe_data_item_t) * column_size);
				++iter->prefetch_head;
			}
			++lines;
		}
		++iter->prefetch_tail;
		if (iter->prefetch_tail >= lines)
			iter->prefetch_tail = 0;
		next = iter->idx + prefetched;
	}
	int i;
	ccv_array_t* const prefetches = iter->prefetches;
	ccv_cnnp_dataframe_data_item_t* const cached_data = (ccv_cnnp_dataframe_data_item_t*)ccv_array_get(prefetches, iter->prefetch_tail * column_size);
	memset(cached_data, 0, sizeof(ccv_cnnp_dataframe_data_item_t) * column_size);
	for (i = 0; i < iter->column_idx_size; i++)
		_ccv_cnnp_dataframe_column_data(dataframe, cached_data, next, iter->column_idxs[i], stream_context);
	return 0;
}

int ccv_cnnp_dataframe_iter_set_cursor(ccv_cnnp_dataframe_iter_t* const iter, const int idx)
{
	ccv_cnnp_dataframe_t* const dataframe = iter->dataframe;
	assert(dataframe);
	if (idx >= dataframe->row_size)
		return -1;
	iter->idx = idx;
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
			// Get the property deinit function.
			ccv_cnnp_column_data_deinit_f deinit = (i < dataframe->column_size) ? dataframe->column_data[i].deinit : ((ccv_cnnp_derived_column_data_t*)ccv_array_get(dataframe->derived_column_data, i - dataframe->column_size))->deinit;
			if (deinit)
				for (j = 0; j < column->rnum; j++)
					deinit(*(void**)ccv_array_get(column, j));
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
			ccfree(derived_column_data->data);
		}
		ccv_array_free(dataframe->derived_column_data);
	}
	ccfree(dataframe);
}
