#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_dataframe.h"

#pragma mark - Reducer

typedef struct {
	int column_idx;
	int batch_size;
	int iter_idx;
	ccv_cnnp_column_data_reduce_f reduce;
	ccv_cnnp_dataframe_t* dataframe;
	ccv_cnnp_dataframe_iter_t* iter;
	ccv_cnnp_column_data_deinit_f data_deinit;
	void* context;
	ccv_cnnp_column_data_context_deinit_f context_deinit;
	void* batch_data[1];
} ccv_cnnp_dataframe_reducer_t;

static void _ccv_cnnp_reducer_enum(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_dataframe_reducer_t* const reducer = (ccv_cnnp_dataframe_reducer_t*)context;
	if (!reducer->iter)
	{
		reducer->iter = ccv_cnnp_dataframe_iter_new(reducer->dataframe, &reducer->column_idx, 1);
		reducer->iter_idx = -1;
	}
	ccv_cnnp_dataframe_iter_t* const iter = reducer->iter;
	int i, j;
	for (i = 0; i < row_size; i++)
	{
		if (reducer->iter_idx + 1 != row_idxs[i])
			ccv_cnnp_dataframe_iter_set_cursor(iter, row_idxs[i] * reducer->batch_size);
		reducer->iter_idx = row_idxs[i];
		ccv_cnnp_dataframe_iter_prefetch(iter, reducer->batch_size, stream_context);
		for (j = 0; j < reducer->batch_size; j++)
			if (0 != ccv_cnnp_dataframe_iter_next(iter, reducer->batch_data + j, 1, stream_context))
				break;
		reducer->reduce(reducer->batch_data, j, data + i, reducer->context, stream_context);
	}
}

static void _ccv_cnnp_reducer_data_deinit(void* const data, void* const context)
{
	ccv_cnnp_dataframe_reducer_t* const reducer = (ccv_cnnp_dataframe_reducer_t*)context;
	assert(reducer->data_deinit);
	reducer->data_deinit(data, reducer->context);
}

static void _ccv_cnnp_reducer_deinit(void* const context)
{
	ccv_cnnp_dataframe_reducer_t* const reducer = (ccv_cnnp_dataframe_reducer_t*)context;
	if (reducer->iter)
		ccv_cnnp_dataframe_iter_free(reducer->iter);
	if (reducer->context_deinit)
		reducer->context_deinit(reducer->context);
	ccfree(reducer);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_reduce_new(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_reduce_f reduce, ccv_cnnp_column_data_deinit_f data_deinit, const int column_idx, const int batch_size, void* const context, ccv_cnnp_column_data_context_deinit_f context_deinit)
{
	assert(batch_size > 0);
	ccv_cnnp_dataframe_reducer_t* const reducer = (ccv_cnnp_dataframe_reducer_t*)ccmalloc(sizeof(ccv_cnnp_dataframe_reducer_t) + sizeof(void*) * (batch_size - 1));
	reducer->column_idx = column_idx;
	reducer->batch_size = batch_size;
	reducer->reduce = reduce;
	reducer->dataframe = dataframe;
	reducer->iter = 0;
	reducer->data_deinit = data_deinit;
	reducer->context = context;
	reducer->context_deinit = context_deinit;
	ccv_cnnp_column_data_t reduce_column = {
		.data_enum = _ccv_cnnp_reducer_enum,
		.data_deinit = data_deinit ? _ccv_cnnp_reducer_data_deinit : 0, // Redirect to our data deinit method.
		.context = reducer,
		.context_deinit = _ccv_cnnp_reducer_deinit,
	};
	return ccv_cnnp_dataframe_new(&reduce_column, 1, (ccv_cnnp_dataframe_row_count(dataframe) + batch_size - 1) / batch_size);
}

#pragma mark - Make Tuple

static void _ccv_cnnp_tuple_deinit(void* const data, void* const context)
{
	ccfree(data);
}

static void _ccv_cnnp_make_tuple(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_cnnp_dataframe_tuple_t* const tuple = (ccv_cnnp_dataframe_tuple_t*)context;
	int i, j;
	for (i = 0; i < batch_size; i++)
	{
		if (!data[i])
			data[i] = ccmalloc(sizeof(void*) * tuple->size);
		void** tuple_data = (void**)data[i];
		for (j = 0; j < column_size; j++)
			tuple_data[j] = column_data[j][i];
	}
}

int ccv_cnnp_dataframe_make_tuple(ccv_cnnp_dataframe_t* const dataframe, const int* const column_idxs, const int column_idx_size)
{
	ccv_cnnp_dataframe_tuple_t* const tuple = (ccv_cnnp_dataframe_tuple_t*)ccmalloc(sizeof(ccv_cnnp_dataframe_tuple_t));
	tuple->size = column_idx_size;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_make_tuple, 0, _ccv_cnnp_tuple_deinit, column_idxs, column_idx_size, tuple, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}

int ccv_cnnp_dataframe_tuple_size(const ccv_cnnp_dataframe_t* const dataframe, const int column_idx)
{
	const ccv_cnnp_dataframe_tuple_t* const tuple = (ccv_cnnp_dataframe_tuple_t*)ccv_cnnp_dataframe_column_context(dataframe, column_idx);
	return tuple->size;
}

typedef struct {
	int index;
} ccv_cnnp_dataframe_extract_tuple_t;

static void _ccv_cnnp_extract_tuple(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_cnnp_dataframe_extract_tuple_t* const extract_tuple = (ccv_cnnp_dataframe_extract_tuple_t*)context;
	const int index = extract_tuple->index;
	int i;
	for (i = 0; i < batch_size; i++)
	{
		void** tuple_data = (void**)column_data[0][i];
		data[i] = tuple_data[index];
	}
}

int ccv_cnnp_dataframe_extract_tuple(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int index)
{
	ccv_cnnp_dataframe_extract_tuple_t* const extract_tuple = (ccv_cnnp_dataframe_extract_tuple_t*)ccmalloc(sizeof(ccv_cnnp_dataframe_extract_tuple_t));
	extract_tuple->index = index;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_extract_tuple, 0, 0, &column_idx, 1, extract_tuple, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}
