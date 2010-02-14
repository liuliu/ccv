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

ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index)
{
	if (mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)].index != -1)
	{
		ccv_dense_vector_t* vector = &mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)];
		while (vector != NULL && vector.index != index)
			vector = vector.next;
		return vector;
	}
	return NULL;
}

ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col)
{
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row);
	ccv_matrix_cell_t cell = NULL;
	if (vector != NULL && vector->length > 0)
	{
		if (mat->major == CCV_SPARSE_FULL)
		{
			int h = (col * 33) % vector->length, i = 0;
			while (vector->indice[(h + i * i) % vector->length] != col && vector->indice[(h + i * i) % vector->length] != -1)
				i++;
			if (vector->indice[(h + i * i) % vector->length] != -1)
				cell.ptr = vector->data.ptr + CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL_NUM(mat->type) * ((h + i * i) % vector->length);
		} else {
			cell.ptr = vector->data.ptr + CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL_NUM(mat->type) * ((mat->major == CCV_SPARSE_COL_MAJOR) ? row : col);
		}
	}
	return cell;
}

int ccv_matrix_assert(ccv_matrix_t* mat, int type, int rows_lt, int rows_gt, int cols_lt, int cols_gt)
{
}
