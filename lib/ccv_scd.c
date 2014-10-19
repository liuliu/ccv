#include "ccv.h"
#include "ccv_internal.h"

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_new(ccv_array_t* categorizeds, ccv_array_t* hard_mine, const char* filename, ccv_scd_train_param_t params)
{
	return 0;
}

void ccv_scd_classifier_cascade_write(ccv_scd_classifier_cascade_t* cascade, const char* filename)
{
}

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_read(const char* filename)
{
	return 0;
}

void ccv_scd_classifier_cascade_free(ccv_scd_classifier_cascade_t* cascade)
{
}

ccv_array_t* ccv_scd_detect_objects(ccv_dense_matrix_t* a, ccv_scd_classifier_cascade_t** cascades, int count, ccv_scd_param_t params)
{
	return 0;
}
