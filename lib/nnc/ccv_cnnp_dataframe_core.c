#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_dataframe.h"

typedef struct {
	int column_idx;
	int batch_size;
	int iter_idx;
	ccv_cnnp_column_data_reduce_f reduce;
	ccv_cnnp_dataframe_t* dataframe;
	ccv_cnnp_dataframe_iter_t* iter;
	void* context;
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

static void _ccv_cnnp_reducer_deinit(ccv_cnnp_dataframe_t* const self)
{
	assert(self->column_size >= 1);
	ccv_cnnp_dataframe_reducer_t* const reducer = (ccv_cnnp_dataframe_reducer_t*)self->column_data[0].context;
	if (reducer->iter)
		ccv_cnnp_dataframe_iter_free(reducer->iter);
	ccfree(reducer);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_reduce_new(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_reduce_f reduce, ccv_cnnp_column_data_deinit_f deinit, const int column_idx, const int batch_size, void* const context)
{
	assert(batch_size > 0);
	ccv_cnnp_dataframe_reducer_t* const reducer = (ccv_cnnp_dataframe_reducer_t*)ccmalloc(sizeof(ccv_cnnp_dataframe_reducer_t) + sizeof(void*) * (batch_size - 1));
	reducer->column_idx = column_idx;
	reducer->batch_size = batch_size;
	reducer->reduce = reduce;
	reducer->dataframe = dataframe;
	reducer->iter = 0;
	reducer->context = context;
	ccv_cnnp_column_data_t reduce_column = {
		.data_enum = _ccv_cnnp_reducer_enum,
		.deinit = deinit,
		.context = reducer,
	};
	ccv_cnnp_dataframe_t* const reduce_dataframe = ccv_cnnp_dataframe_new(&reduce_column, 1, (dataframe->row_count + batch_size - 1) / batch_size);
	reduce_dataframe->isa.deinit = _ccv_cnnp_reducer_deinit;
	return reduce_dataframe;
}
