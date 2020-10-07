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
	ccv_cnnp_dataframe_iter_t* iter = ccv_cnnp_dataframe_iter_new(dataframe, COLUMN_ID_LIST(0));
	int i;
	for (i = 0; i < 100; i++)
	{
		void* data = 0;
		ccv_cnnp_dataframe_iter_next(iter, &data, 1, 0);
		printf("%s\n", data);
	}
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(dataframe);
	return 0;
}
