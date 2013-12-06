#include "../ccv.h"

/**********************************************************
 * C-based/Cached/Core Computer Vision Library on GPU (COG)
 * Liu Liu, 2013-12-01
 **********************************************************/

#ifndef GUARD_cog_h
#define GUARD_cog_h

void cog_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch);
void cog_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch);
void cog_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, ccv_convnet_train_param_t params);

#endif
