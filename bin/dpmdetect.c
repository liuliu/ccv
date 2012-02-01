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
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dpm_root_classifier_t* root_classifier = ccv_load_dpm_root_classifier(argv[2]);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_dpm_param_t params = { .interval = 5, .min_neighbors = 2, .flags = 0, .size = ccv_size(root_classifier->root.size.width * 8, root_classifier->root.size.height * 8) };
		ccv_array_t* seq = ccv_dpm_detect_objects(image, &root_classifier, 1, params);
		/*
		elapsed_time = get_current_time() - elapsed_time;
		for (i = 0; i < seq->rnum; i++)
		{
			ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
			printf("%d %d %d %d %f\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->confidence);
		}
		printf("total : %d in time %dms\n", seq->rnum, elapsed_time);
		ccv_array_free(seq);
		*/
		ccv_matrix_free(image);
	}
	ccv_drain_cache();
	return 0;
}
