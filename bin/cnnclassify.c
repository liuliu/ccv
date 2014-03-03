#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	assert(argc >= 3);
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	assert(image != 0);
	ccv_convnet_t* convnet = ccv_convnet_read(0, argv[2]);
	ccv_dense_matrix_t* norm = 0;
	if (image->rows > convnet->input.height && image->cols > convnet->input.width)
		ccv_resample(image, &norm, 0, ccv_max(convnet->input.height, (int)(image->rows * (float)convnet->input.height / image->cols + 0.5)), ccv_max(convnet->input.width, (int)(image->cols * (float)convnet->input.width / image->rows + 0.5)), CCV_INTER_AREA);
	else if (image->rows < convnet->input.height || image->cols < convnet->input.width)
		ccv_resample(image, &norm, 0, ccv_max(convnet->input.height, (int)(image->rows * (float)convnet->input.height / image->cols + 0.5)), ccv_max(convnet->input.width, (int)(image->cols * (float)convnet->input.width / image->rows + 0.5)), CCV_INTER_CUBIC);
	else
		norm = image;
	if (norm != image)
		ccv_matrix_free(image);
	ccv_dense_matrix_t* input = 0;
	if (norm->cols != convnet->input.width || norm->rows != convnet->input.height)
	{
		int x = (norm->cols - convnet->input.width + 1) / 2;
		int y =  (norm->rows - convnet->input.height + 1) / 2;
		assert(x == 0 || y == 0);
		ccv_slice(norm, (ccv_matrix_t**)&input, CCV_32F, y, x, convnet->input.height, convnet->input.width);
	} else
		ccv_shift(norm, (ccv_matrix_t**)&input, CCV_32F, 0, 0); // converting to 32f
	unsigned int elapsed_time = get_current_time();
	ccv_array_t* rank = 0;
	ccv_convnet_classify(convnet, &input, 1, &rank, 5, 1);
	elapsed_time = get_current_time() - elapsed_time;
	int i;
	for (i = 0; i < rank->rnum; i++)
	{
		ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(rank, i);
		printf("%d %f\n", classification->id, classification->confidence);
	}
	printf("elapsed time %dms\n", elapsed_time);
	ccv_matrix_free(norm);
	ccv_matrix_free(input);
	ccv_convnet_free(convnet);
	return 0;
}
