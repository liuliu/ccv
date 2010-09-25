#include "ccv.h"
#include <assert.h>

int main(int argc, char** argv)
{
	ccv_sparse_matrix_t* mat = ccv_sparse_matrix_new(1000, 1000, CCV_32S | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
	int i, j, k, n;
	k = 0;
	for (i = 0; i < 1000; i++)
		for (j = 0; j < 1000; j++)
		{
			ccv_set_sparse_matrix_cell(mat, i, j, &k);
			k++;
		}
	printf("prime : %d\n", CCV_GET_SPARSE_PRIME(mat->prime));
	for (n = 0; n < 100; n++)
	{
		k = 0;
		for (i = 0; i < 1000; i++)
			for (j = 0; j < 1000; j++)
			{
				ccv_matrix_cell_t cell = ccv_get_sparse_matrix_cell(mat, i, j);
				if (cell.ptr != 0)
					assert(cell.i[0] == k);
				else {
					printf("ERROR: %d %d\n", i, j);
					exit(0);
				}
				k++;
			}
	}
	ccv_matrix_free(mat);
/*	similar OpenCV code for sparse matrix, little slower than my version
	int sizes[] = {1000, 1000};
	CvSparseMat* mat = cvCreateSparseMat(2, sizes, CV_32SC1);
	int k = 0;
	for (int i = 0; i < 1000; i++)
		for (int j = 0; j < 1000; j++)
		{
			cvSetReal2D(mat, i, j, k);
			k++;
		}
	for (int n = 0; n < 100; n++)
	{
		k = 0;
		for (int i = 0; i < 1000; i++)
			for (int j = 0; j < 1000; j++)
			{
				double u = cvGetReal2D(mat, i, j);
				if (fabs(k - u) > 1e-6)
					printf("ERROR: %d %d %d\n", i, j, n);
				k++;
			}
	}
*/
	return 0;
}
