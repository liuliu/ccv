#include "ccv.h"

ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
		return (ccv_dense_matrix_t*)mat;
	return NULL;
}

ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_SPARSE)
		return (ccv_sparse_matrix_t*)mat;
	return NULL;
}

ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index)
{
	if (mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)].index != -1)
	{
		ccv_dense_vector_t* vector = &mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)];
		while (vector != NULL && vector->index != index)
			vector = vector->next;
		return vector;
	}
	return NULL;
}

ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col)
{
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row);
	ccv_matrix_cell_t cell;
	cell.ptr = NULL;
	if (vector != NULL && vector->length > 0)
	{
		int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL_NUM(mat->type);
		int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
		if (mat->type & CCV_DENSE_VECTOR)
		{
			cell.ptr = vector->data.ptr + cell_width * vidx;
		} else {
			int h = (vidx * 33) % vector->length, i = 0;
			while (vector->indice[(h + i * i) % vector->length] != vidx && vector->indice[(h + i * i) % vector->length] != -1)
				i++;
			i = (h + i * i) % vector->length;
			if (vector->indice[i] != -1)
				cell.ptr = vector->data.ptr + i * cell_width;
		}
	}
	return cell;
}

static void __ccv_dense_vector_expand(ccv_sparse_matrix_t* mat, ccv_dense_vector_t* vector)
{
	if (vector->prime == -1)
		return;
	vector->prime++;
	int new_length = CCV_GET_SPARSE_PRIME(vector->prime);
	int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL_NUM(mat->type);
	int new_step = (new_length * cell_width + 3) & -4;
	ccv_matrix_cell_t new_data;
	new_data.ptr = (unsigned char*)malloc(new_step + sizeof(int) * new_length);
	int* new_indice = (int*)(new_data.ptr + new_step);
	int i;
	for (i = 0; i < new_length; i++)
		new_indice[i] = -1;
	for (i = 0; i < vector->length; i++)
		if (vector->indice[i] != -1)
		{
			int index = vector->indice[i];
			int h = (index * 33) % new_length, j = 0;
			while (new_indice[(h + j * j) % new_length] != index && new_indice[(h + j * j) % new_length] != -1)
				j++;
			j = (h + j * j) % new_length;
			new_indice[j] = index;
			memcpy(new_data.ptr + j * cell_width, vector->data.ptr + i * cell_width, cell_width);
		}
	vector->length = new_length;
	free(vector->data.ptr);
	vector->data = new_data;
	vector->indice = new_indice;
}

static void __ccv_sparse_matrix_expand(ccv_sparse_matrix_t* mat)
{
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	mat->prime++;
	int new_length = CCV_GET_SPARSE_PRIME(mat->prime);
	ccv_dense_vector_t* new_vector = (ccv_dense_vector_t*)malloc(new_length * sizeof(ccv_dense_vector_t));
	int i;
	for (i = 0; i < new_length; i++)
	{
		new_vector[i].index = -1;
		new_vector[i].length = 0;
		new_vector[i].next = NULL;
	}
	for (i = 0; i < length; i++)
		if (mat->vector[i].index != -1)
		{
			int h = (mat->vector[i].index * 33) % new_length;
			if (new_vector[h].length == 0)
			{
				memcpy(new_vector + h, mat->vector + i, sizeof(ccv_dense_vector_t));
				new_vector[h].next = NULL;
			} else {
				ccv_dense_vector_t* t = (ccv_dense_vector_t*)malloc(sizeof(ccv_dense_vector_t));
				memcpy(t, mat->vector + i, sizeof(ccv_dense_vector_t));
				t->next = new_vector[h].next;
				new_vector[h].next = t;
			}
			ccv_dense_vector_t* iter = mat->vector[i].next;
			while (iter != NULL)
			{
				ccv_dense_vector_t* iter_next = iter->next;
				h = (iter->index * 33) % new_length;
				if (new_vector[h].length == 0)
				{
					memcpy(new_vector + h, iter, sizeof(ccv_dense_vector_t));
					new_vector[h].next = NULL;
					free(iter);
				} else {
					iter->next = new_vector[h].next;
					new_vector[h].next = iter;
				}
				iter = iter_next;
			}
		}
	free(mat->vector);
	mat->vector = new_vector;
}

void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data)
{
	int i;
	int index = (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row;
	int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, index);
	if (vector == NULL)
	{
		mat->load_factor++;
		if (mat->load_factor * 4 > CCV_GET_SPARSE_PRIME(mat->prime) * 3)
		{
			__ccv_sparse_matrix_expand(mat);
			length = CCV_GET_SPARSE_PRIME(mat->prime);
		}
		vector = &mat->vector[(index * 33) % length];
		if (vector->index != -1)
		{
			vector = (ccv_dense_vector_t*)malloc(sizeof(ccv_dense_vector_t));
			vector->index = -1;
			vector->length = 0;
			vector->next = mat->vector[(index * 33) % length].next;
			mat->vector[(index * 33) % length].next = vector;
		}
	}
	int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL_NUM(mat->type);
	if (mat->type & CCV_DENSE_VECTOR)
	{
		if (vector->index == -1)
		{
			vector->prime = -1;
			vector->length = (mat->major == CCV_SPARSE_COL_MAJOR) ? mat->rows : mat->cols;
			vector->index = index;
			vector->step = (vector->length * cell_width + 3) & -4;
			vector->data.ptr = (unsigned char*)malloc(vector->step);
		}
		if (data != NULL)
			memcpy(vector->data.ptr + vidx * cell_width, data, cell_width);
	} else {
		if (vector->index == -1)
		{
			vector->prime = 0;
			vector->load_factor = 0;
			vector->length = CCV_GET_SPARSE_PRIME(vector->prime);
			vector->index = index;
			vector->step  = (vector->length * cell_width + 3) & -4;
			vector->data.ptr = (unsigned char*)malloc(vector->step + sizeof(int) * vector->length);
			vector->indice = (int*)(vector->data.ptr + vector->step);
			for (i = 0; i < vector->length; i++)
				vector->indice[i] = -1;
		}
		vector->load_factor++;
		if (vector->load_factor * 2 > vector->length)
		{
			__ccv_dense_vector_expand(mat, vector);
		}
		i = 0;
		int h = (vidx * 33) % vector->length;
		while (vector->indice[(h + i * i) % vector->length] != vidx && vector->indice[(h + i * i) % vector->length] != -1)
			i++;
		i = (h + i * i) % vector->length;
		vector->indice[i] = vidx;
		if (data != NULL)
			memcpy(vector->data.ptr + i * cell_width, data, cell_width);
	}
}

#define __ccv_indice_less_than(i1, i2, aux) ((i1) < (i2))
#define __ccv_swap_indice_and_float_data(i1, i2, array, aux, t) {  \
	float td = (aux)[(int)(&(i1) - (array))];                      \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;                            \
	CCV_SWAP(i1, i2, t); }
#define __ccv_swap_indice_and_double_data(i1, i2, array, aux, t) { \
	double td = (aux)[(int)(&(i1) - (array))];                     \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;                            \
	CCV_SWAP(i1, i2, t); }

CCV_IMPLEMENT_QSORT_EX(__ccv_indice_float_sort, int, __ccv_indice_less_than, __ccv_swap_indice_and_float_data, float*);
CCV_IMPLEMENT_QSORT_EX(__ccv_indice_double_sort, int, __ccv_indice_less_than, __ccv_swap_indice_and_double_data, double*);

void ccv_compress_sparse_matrix(ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm)
{
	int i, j;
	int nnz = 0;
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	for (i = 0; i < length; i++)
	{
		ccv_dense_vector_t* vector = &mat->vector[i];
		while (vector != NULL)
		{
			if (vector->index != -1)
				nnz += vector->load_factor;
			vector = vector->next;
		}
	}
	ccv_compressed_sparse_matrix_t* cm = *csm = (ccv_compressed_sparse_matrix_t*)malloc(sizeof(ccv_compressed_sparse_matrix_t) + nnz * sizeof(int) + nnz * CCV_GET_DATA_TYPE_SIZE(mat->type) + (mat->rows + 1) * sizeof(int));
	cm->type = (mat->type & ~CCV_MATRIX_SPARSE) | CCV_MATRIX_CSR;
	cm->nnz = nnz;
	cm->rows = mat->rows;
	cm->cols = mat->cols;
	cm->index = (int*)(cm + 1);
	cm->offset = cm->index + nnz;
	cm->data.i = cm->offset + mat->rows + 1;
	unsigned char* m_ptr = cm->data.ptr;
	int* idx = cm->index;
	cm->offset[0] = 0;
	for (i = 0; i < mat->rows; i++)
	{
		ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, i);
		if (vector == NULL)
			cm->offset[i + 1] = cm->offset[i];
		else {
			int k = 0;
			for (j = 0; j < vector->length; j++)
				if (vector->indice[j] != -1)
				{
					ccv_set_value(mat->type, m_ptr, k, ccv_get_value(mat->type, vector->data.ptr, j));
					idx[k] = vector->indice[j];
					k++;
				}
			switch (CCV_GET_DATA_TYPE(mat->type))
			{
				case CCV_32F:
					__ccv_indice_float_sort(idx, vector->load_factor, (float*)m_ptr);
					break;
				case CCV_64F:
					__ccv_indice_double_sort(idx, vector->load_factor, (double*)m_ptr);
					break;
			}
			cm->offset[i + 1] = cm->offset[i] + vector->load_factor;
			idx += vector->load_factor;
			m_ptr += vector->load_factor * CCV_GET_DATA_TYPE_SIZE(mat->type);
		}
	}
}
