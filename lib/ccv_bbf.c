#include "ccv.h"

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_param_t params)
{
}

ccv_array_t* ccv_bbf_detect_objects(ccv_dense_matrix_t* a, ccv_bbf_classifier_cascade_t** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size)
{
}

ccv_bbf_classifier_cascade_t* ccv_load_bbf_classifier_cascade(const char* directory)
{
}

ccv_bbf_classifier_cascade_t* ccv_bbf_classifier_cascade_read_binary(char* s)
{
}

int ccv_bbf_classifier_cascade_write_binary(ccv_bbf_classifier_cascade_t* cascade, char* s, int slen)
{
}

void ccv_bbf_classifier_cascade_free(ccv_bbf_classifier_cascade_t* cascade)
{
}
