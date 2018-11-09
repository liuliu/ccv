#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _ccv_iter_int(const int column_idx, const int row_idx, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int* const array = (int*)context;
	int i;
	for (i = 0; i < row_size; i++)
		*data = (void*)(intptr_t)array[row_idx + i];
}

TEST_CASE("iterate through a simple dataframe")
{
	int int_array[8] = {
		2, 3, 4, 5, 6, 7, 8, 9
	};
	ccv_cnnp_column_data_t columns[] = {
		{
			.data_enum = _ccv_iter_int,
			.context = int_array,
		}
	};
	ccv_cnnp_dataframe_t* const dataframe = ccv_cnnp_dataframe_new(columns, sizeof(columns) / sizeof(columns[0]), 8);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(dataframe, COLUMN_ID_LIST(0));
	int result[8];
	int i = 0;
	void* data;
	while (0 == ccv_cnnp_dataframe_iter_next(iter, &data, 1, 0))
		result[i++] = (int)(intptr_t)data;
	ccv_cnnp_dataframe_iter_free(iter);
	REQUIRE_ARRAY_EQ(int, int_array, result, 8, "iterated result and actual result should be the same");
	ccv_cnnp_dataframe_free(dataframe);
}

static void _ccv_int_plus_1(void** const column_data, const int column_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int i = (int)(intptr_t)column_data[0];
	*data = (void*)(intptr_t)(i + 1);
}

TEST_CASE("iterate through derived column")
{
	int int_array[8] = {
		2, 3, 4, 5, 6, 7, 8, 9
	};
	ccv_cnnp_column_data_t columns[] = {
		{
			.data_enum = _ccv_iter_int,
			.context = int_array,
		}
	};
	ccv_cnnp_dataframe_t* const dataframe = ccv_cnnp_dataframe_new(columns, sizeof(columns) / sizeof(columns[0]), 8);
	const int derived = ccv_cnnp_dataframe_map(dataframe, _ccv_int_plus_1, 0, COLUMN_ID_LIST(0), 0);
	assert(derived > 0);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(dataframe, COLUMN_ID_LIST(derived));
	int result[8];
	int i = 0;
	void* data;
	while (0 == ccv_cnnp_dataframe_iter_next(iter, &data, 1, 0))
		result[i++] = (int)(intptr_t)data;
	ccv_cnnp_dataframe_iter_free(iter);
	for (i = 0; i < 8; i++)
		++int_array[i];
	REQUIRE_ARRAY_EQ(int, int_array, result, 8, "iterated result and actual result should be the same");
	ccv_cnnp_dataframe_free(dataframe);
}

#include "case_main.h"
