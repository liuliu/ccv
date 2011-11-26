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
	int i;
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_bbf_classifier_cascade_t* cascade = ccv_load_bbf_classifier_cascade(argv[2]);
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_bbf_param_t params = { .interval = 5, .min_neighbors = 2, .flags = 0, .size = ccv_size(24, 24) };
		ccv_array_t* seq = ccv_bbf_detect_objects(image, &cascade, 1, params);
		elapsed_time = get_current_time() - elapsed_time;
		for (i = 0; i < seq->rnum; i++)
		{
			ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
			printf("%d %d %d %d %f\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->confidence);
		}
		printf("total : %d in time %dms\n", seq->rnum, elapsed_time);
		ccv_array_free(seq);
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
				ccv_unserialize(file, &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
				ccv_bbf_param_t params = { .interval = 5, .min_neighbors = 2, .flags = 0, .size = ccv_size(24, 24) };
				ccv_array_t* seq = ccv_bbf_detect_objects(image, &cascade, 1, params);
				for (i = 0; i < seq->rnum; i++)
				{
					ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
					printf("%s %d %d %d %d %f\n", file, comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->confidence);
				}
				ccv_array_free(seq);
				ccv_matrix_free(image);
			}
			fclose(r);
		}
	}
	ccv_bbf_classifier_cascade_free(cascade);
	ccv_disable_cache();
	return 0;
}
