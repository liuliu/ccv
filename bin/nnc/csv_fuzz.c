#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerInitialize(int* argc, char*** argv)
{
	ccv_nnc_init();
	return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
	if (size == 0)
		return 0;
	int column_size = 0;
	ccv_cnnp_dataframe_t* dataframe = ccv_cnnp_dataframe_from_csv_new((void*)data, CCV_CNNP_DATAFRAME_CSV_MEMORY, size, ',', '"', 0, &column_size);
	if (dataframe)
		ccv_cnnp_dataframe_free(dataframe);
	return 0;
}
