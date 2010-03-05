#include "ccv.h"

int main(int argc, char** argv)
{
	ccv_sparse_matrix_t* mat = ccv_sparse_matrix_new(3, 3, CCV_32F | CCV_C1, CCV_SPARSE_ROW_MAJOR, NULL);
	float cell;
	cell = 1.0;
	ccv_set_sparse_matrix_cell(mat, 0, 0, &cell);
	cell = 2.0;
	ccv_set_sparse_matrix_cell(mat, 0, 2, &cell);
	cell = 3.0;
	ccv_set_sparse_matrix_cell(mat, 1, 2, &cell);
	cell = 4.0;
	ccv_set_sparse_matrix_cell(mat, 2, 0, &cell);
	cell = 5.0;
	ccv_set_sparse_matrix_cell(mat, 2, 1, &cell);
	cell = 6.0;
	ccv_set_sparse_matrix_cell(mat, 2, 2, &cell);
	ccv_compressed_sparse_matrix_t* csm = NULL;
	ccv_compress_sparse_matrix(mat, &csm);
	int i, j;
	for (i = 0; i < csm->nnz; i++)
		printf("%f ", csm->data.fl[i]);
	printf("\n");
	for (i = 0; i < csm->nnz; i++)
		printf("%d ", csm->index[i]);
	printf("\n");
	for (i = 0; i < csm->rows + 1; i++)
		printf("%d ", csm->offset[i]);
	printf("\n");
	ccv_sparse_matrix_t* smt = NULL;
	ccv_decompress_sparse_matrix(csm, &smt);
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			ccv_matrix_cell_t cell = ccv_get_sparse_matrix_cell(smt, i, j);
			printf("%f ", (cell.ptr != NULL) ? cell.fl[0] : 0);
		}
		printf("\n");
	}
	ccv_matrix_free(smt);
	ccv_matrix_free(mat);
	ccv_matrix_free(csm);
	return 0;
}
