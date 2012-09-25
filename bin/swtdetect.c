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
		.interval = 1,
		.same_word_thresh = { 0.1, 0.8 },
		.min_neighbors = 1,
		.scale_invariant = 0,
		.size = 3,
		.low_thresh = 124,
		.high_thresh = 204,
		.max_height = 300,
		.min_height = 8,
		.min_area = 38,
		.letter_occlude_thresh = 3,
		.aspect_ratio = 8,
		.std_ratio = 0.83,
		.thickness_ratio = 1.5,
		.height_ratio = 1.7,
		.intensity_thresh = 31,
		.distance_ratio = 2.9,
		.intersect_ratio = 1.3,
		.letter_thresh = 3,
		.elongate_ratio = 1.9,
		.breakdown = 1,
		.breakdown_ratio = 1.0,
	};
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* words = ccv_swt_detect_words(image, params);
		elapsed_time = get_current_time() - elapsed_time;
		if (words)
		{
			int i;
			for (i = 0; i < words->rnum; i++)
			{
				ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
				printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
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
				printf("%s\n", file);
				ccv_read(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				ccv_array_t* words = ccv_swt_detect_words(image, params);
				int i;
				for (i = 0; i < words->rnum; i++)
				{
					ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
					printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
				}
				ccv_array_free(words);
				ccv_matrix_free(image);
			}
			free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	return 0;
}

