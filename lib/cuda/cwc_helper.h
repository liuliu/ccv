/*************************************************************
 * C-based/Cached/Core Computer Vision Library with CUDA (CWC)
 * Liu Liu, 2013-12-01
 *************************************************************/

#ifndef GUARD_cwc_helper_h
#define GUARD_cwc_helper_h

// helper functions

#ifdef HAVE_GSL
void cwc_convnet_batch_formation(gsl_rng* rng, ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_dense_matrix_t* eigenvectors, ccv_dense_matrix_t* eigenvalues, float image_manipulation, float color_gain, int* idx, ccv_size_t dim, int min_dim, int max_dim, int rows, int cols, int channels, int category_count, int symmetric, int batch, int offset, int size, float* b, int* c);
#endif
void cwc_convnet_mean_formation(ccv_array_t* categorizeds, ccv_size_t dim, int channels, int symmetric, ccv_dense_matrix_t** b);
void cwc_convnet_channel_eigen(ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_size_t dim, int channels, ccv_dense_matrix_t** eigenvectors, ccv_dense_matrix_t** eigenvalues);

#endif
