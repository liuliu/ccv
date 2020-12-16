#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_dataframe.h"
#ifdef CCV_BLOCK_SUPPORT
#include <Block.h>
#endif

// MARK - Reducer

typedef struct {
	int column_idx;
	int batch_size;
	int iter_idx;
	ccv_cnnp_column_data_sample_f sample;
	ccv_cnnp_dataframe_t* dataframe;
	ccv_cnnp_dataframe_iter_t* iter;
	ccv_cnnp_column_data_deinit_f data_deinit;
	void* context;
	ccv_cnnp_column_data_context_deinit_f context_deinit;
	void* batch_data[1];
} ccv_cnnp_dataframe_sampler_t;

static void _ccv_cnnp_sampler_enum(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_dataframe_sampler_t* const sampler = (ccv_cnnp_dataframe_sampler_t*)context;
	if (!sampler->iter)
	{
		sampler->iter = ccv_cnnp_dataframe_iter_new(sampler->dataframe, &sampler->column_idx, 1);
		sampler->iter_idx = -1;
	}
	ccv_cnnp_dataframe_iter_t* const iter = sampler->iter;
	int i, j;
	for (i = 0; i < row_size; i++)
	{
		if (sampler->iter_idx + 1 != row_idxs[i])
			ccv_cnnp_dataframe_iter_set_cursor(iter, row_idxs[i] * sampler->batch_size);
		sampler->iter_idx = row_idxs[i];
		ccv_cnnp_dataframe_iter_prefetch(iter, sampler->batch_size, stream_context);
		for (j = 0; j < sampler->batch_size; j++)
			if (0 != ccv_cnnp_dataframe_iter_next(iter, sampler->batch_data + j, 1, stream_context))
				break;
		sampler->sample(sampler->batch_data, j, data + i, sampler->context, stream_context);
	}
}

static void _ccv_cnnp_sampler_data_deinit(void* const data, void* const context)
{
	ccv_cnnp_dataframe_sampler_t* const sampler = (ccv_cnnp_dataframe_sampler_t*)context;
	assert(sampler->data_deinit);
	sampler->data_deinit(data, sampler->context);
}

static void _ccv_cnnp_sampler_deinit(void* const context)
{
	ccv_cnnp_dataframe_sampler_t* const sampler = (ccv_cnnp_dataframe_sampler_t*)context;
	if (sampler->iter)
		ccv_cnnp_dataframe_iter_free(sampler->iter);
	if (sampler->context_deinit)
		sampler->context_deinit(sampler->context);
	ccfree(sampler);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_sample_new(ccv_cnnp_dataframe_t* const dataframe, ccv_cnnp_column_data_sample_f sample, ccv_cnnp_column_data_deinit_f data_deinit, const int column_idx, const int batch_size, void* const context, ccv_cnnp_column_data_context_deinit_f context_deinit)
{
	assert(batch_size > 0);
	ccv_cnnp_dataframe_sampler_t* const sampler = (ccv_cnnp_dataframe_sampler_t*)ccmalloc(sizeof(ccv_cnnp_dataframe_sampler_t) + sizeof(void*) * (batch_size - 1));
	sampler->column_idx = column_idx;
	sampler->batch_size = batch_size;
	sampler->sample = sample;
	sampler->dataframe = dataframe;
	sampler->iter = 0;
	sampler->data_deinit = data_deinit;
	sampler->context = context;
	sampler->context_deinit = context_deinit;
	ccv_cnnp_column_data_t sample_column = {
		.data_enum = _ccv_cnnp_sampler_enum,
		.data_deinit = data_deinit ? _ccv_cnnp_sampler_data_deinit : 0, // Redirect to our data deinit method.
		.context = sampler,
		.context_deinit = _ccv_cnnp_sampler_deinit,
	};
	return ccv_cnnp_dataframe_new(&sample_column, 1, (ccv_cnnp_dataframe_row_count(dataframe) + batch_size - 1) / batch_size);
}

// MARK - Extract

static void _ccv_cnnp_extract_value(void* const* const* const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	const off_t offset = (off_t)(uintptr_t)context;
	int i;
	for (i = 0; i < batch_size; i++)
	{
		char* const values = (char*)column_data[0][i];
		data[i] = *(void**)(values + offset);
	}
}

int ccv_cnnp_dataframe_extract_value(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const off_t offset, const char* name)
{
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_extract_value, 0, 0, &column_idx, 1, (void*)(uintptr_t)offset, 0, name);
}

// MARK - Make Tuple

static void _ccv_cnnp_tuple_deinit(void* const data, void* const context)
{
	ccfree(data);
}

static void _ccv_cnnp_make_tuple(void* const* const* const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
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

int ccv_cnnp_dataframe_make_tuple(ccv_cnnp_dataframe_t* const dataframe, const int* const column_idxs, const int column_idx_size, const char* name)
{
	ccv_cnnp_dataframe_tuple_t* const tuple = (ccv_cnnp_dataframe_tuple_t*)ccmalloc(sizeof(ccv_cnnp_dataframe_tuple_t));
	tuple->size = column_idx_size;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_make_tuple, 0, _ccv_cnnp_tuple_deinit, column_idxs, column_idx_size, tuple, (ccv_cnnp_column_data_context_deinit_f)ccfree, name);
}

int ccv_cnnp_dataframe_tuple_size(const ccv_cnnp_dataframe_t* const dataframe, const int column_idx)
{
	const ccv_cnnp_dataframe_tuple_t* const tuple = (ccv_cnnp_dataframe_tuple_t*)ccv_cnnp_dataframe_column_context(dataframe, column_idx);
	return tuple->size;
}

int ccv_cnnp_dataframe_extract_tuple(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int index, const char* name)
{
	return ccv_cnnp_dataframe_extract_value(dataframe, column_idx, index * sizeof(void*), name);
}
