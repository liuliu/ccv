#include "ccv.h"
#include <sys/time.h>

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
	ccv_dense_matrix_t* image = NULL;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_bbf_classifier_cascade_t* cascade = ccv_load_bbf_classifier_cascade(argv[2]);
	unsigned int elapsed_time = get_current_time();
	ccv_array_t* seq = ccv_bbf_detect_objects(image, &cascade, 1, 0, 0, ccv_size(24, 24));
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	for (i = 0; i < seq->rnum; i++)
	{
		ccv_bbf_comp_t* comp = (ccv_bbf_comp_t*)ccv_array_get(seq, i);
		printf("%d %d %d %d %f\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->confidence);
	}
	printf("total : %d\n", seq->rnum);
	ccv_bbf_classifier_cascade_free(cascade);
	ccv_array_free(seq);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}
