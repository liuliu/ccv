#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	FILE* f = fopen(argv[1], "r");
	int column_size = 0;
	unsigned int elapsed_time = get_current_time();
	ccv_cnnp_dataframe_t* const dataframe = ccv_cnnp_dataframe_from_csv_new(f, CCV_CNNP_DATAFRAME_CSV_FILE, 0, ',', '"', 0, &column_size);
	fclose(f);
	printf("ccv_cnnp_dataframe_from_csv_new %u ms\n", get_current_time() - elapsed_time);
	if (!dataframe)
	{
		printf("invalid csv file\n");
		return 0;
	}
	int* columns = (int*)ccmalloc(sizeof(int) * column_size);
	int i;
	for (i = 0; i < column_size; i++)
		columns[i] = i;
	ccv_cnnp_dataframe_iter_t* iter = ccv_cnnp_dataframe_iter_new(dataframe, columns, column_size);
	void** data = (void**)cccalloc(column_size, sizeof(void*));
	elapsed_time = get_current_time();
	while (0 == ccv_cnnp_dataframe_iter_next(iter, data, column_size, 0))
	{
		// Do nothing.
	}
	printf("ccv_cnnp_dataframe_iter_next %u ms\n", get_current_time() - elapsed_time);
	ccfree(data);
	ccfree(columns);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(dataframe);
	return 0;
}
