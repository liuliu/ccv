#include "ccv.h"

ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_DENSE)
		return (ccv_dense_matrix_t*)mat;
	return NULL;
}

ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_SPARSE)
		return (ccv_sparse_matrix_t*)mat;
	return NULL;
}

ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int idx)
{

}

ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col)
{
}

int ccv_matrix_assert(ccv_matrix_t* mat, int type, int rows_lt, int rows_gt, int cols_lt, int cols_gt)
{
}
