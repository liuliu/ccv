#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

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
