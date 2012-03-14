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
	ccv_swt_param_t params = { .size = 3, .low_thresh = 175, .high_thresh = 320, .max_height = 300, .min_height = 10, .aspect_ratio = 10, .variance_ratio = 0.5, .thickness_ratio = 1.5, .height_ratio = 2, .intensity_thresh = 30, .distance_ratio = 3, .intersect_ratio = 2, .letter_thresh = 3, .elongate_ratio = 1.3, .breakdown = 1, .breakdown_ratio = 12.8 };
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* words = ccv_swt_detect_words(image, params);
		elapsed_time = get_current_time() - elapsed_time;
		int i;
		for (i = 0; i < words->rnum; i++)
		{
			ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
			printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
		}
		printf("total : %d in time %dms\n", words->rnum, elapsed_time);
		ccv_array_free(words);
		ccv_matrix_free(image);
	} else {
		FILE* r = fopen(argv[1], "rt");
		if (argc == 3)
			chdir(argv[2]);
		if(r)
		{
			char file[1000 + 1];
			while(fgets(file, 1000, r))
			{
				int len = (int)strlen(file);
				while(len > 0 && isspace(file[len - 1]))
					len--;
				file[len] = '\0';
				image = 0;
				ccv_read(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				ccv_array_t* words = ccv_swt_detect_words(image, params);
				int i;
				printf("%s\n%d\n", file, words->rnum);
				for (i = 0; i < words->rnum; i++)
				{
					ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
					printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
				}
				ccv_array_free(words);
				ccv_matrix_free(image);
			}
			fclose(r);
		}
	}
	ccv_disable_cache();
	return 0;
}

