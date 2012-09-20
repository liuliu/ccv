#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	ccv_swt_param_t params = {
		.interval = 2,
		.same_word_thresh = { 0.5, 0.9 },
		.min_neighbors = 0,
		.scale_invariant = 1,
		.size = 3,
		.low_thresh = 75,
		.high_thresh = 250,
		.max_height = 300,
		.min_height = 14,
		.min_area = 75,
		.letter_occlude_thresh = 3,
		.aspect_ratio = 10,
		.std_ratio = 0.75,
		.thickness_ratio = 2.0,
		.height_ratio = 2.0,
		.intensity_thresh = 35,
		.distance_ratio = 3.0,
		.intersect_ratio = 1.5,
		.letter_thresh = 3,
		.elongate_ratio = 2.0,
		.breakdown = 1,
		.breakdown_ratio = 1.0,
	};
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_dense_matrix_t* up2x = 0;
		ccv_sample_up(image, &up2x, 0, 0, 0);
		ccv_array_t* words = ccv_swt_detect_words(up2x, params);
		ccv_matrix_free(up2x);
		elapsed_time = get_current_time() - elapsed_time;
		if (words)
		{
			int i;
			for (i = 0; i < words->rnum; i++)
			{
				ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
				printf("%d %d %d %d\n", rect->x / 2, rect->y / 2, rect->width / 2, rect->height / 2);
			}
			printf("total : %d in time %dms\n", words->rnum, elapsed_time);
			ccv_array_free(words);
		}
		ccv_matrix_free(image);
	} else {
		FILE* r = fopen(argv[1], "rt");
		if (argc == 3)
			chdir(argv[2]);
		if(r)
		{
			size_t len = 1024;
			char* file = (char*)malloc(len);
			ssize_t read;
			while((read = getline(&file, &len, r)) != -1)
			{
				while(read > 1 && isspace(file[read - 1]))
					read--;
				file[read] = 0;
				image = 0;
				ccv_read(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				if (image->rows < 500 || image->cols < 500)
				{
					ccv_dense_matrix_t* up2x = 0;
					ccv_sample_up(image, &up2x, 0, 0, 0);
					ccv_array_t* words = ccv_swt_detect_words(up2x, params);
					ccv_matrix_free(up2x);
					int i;
					printf("%s\n", file);
					for (i = 0; i < words->rnum; i++)
					{
						ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
						printf("%d %d %d %d\n", rect->x / 2, rect->y / 2, rect->width / 2, rect->height / 2);
					}
					ccv_array_free(words);
				} else {
					ccv_array_t* words = ccv_swt_detect_words(image, params);
					int i;
					printf("%s\n", file);
					for (i = 0; i < words->rnum; i++)
					{
						ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
						printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
					}
					ccv_array_free(words);
				}
				ccv_matrix_free(image);
			}
			free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	return 0;
}

