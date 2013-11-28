#include "../ccv.h"

void ccv_cu_convnet_encode(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, ccv_dense_matrix_t** b, int batch);
void ccv_cu_convnet_classify(ccv_convnet_t* convnet, ccv_dense_matrix_t** a, int* labels, int batch);
void ccv_cu_convnet_supervised_train(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_array_t* tests, ccv_convnet_train_param_t params);
