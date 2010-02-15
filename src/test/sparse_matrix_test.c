#include "../ccv.h"
#include <assert.h>

int main(int argc, char** argv)
{
	ccv_sparse_matrix_t* mat = ccv_sparse_matrix_new(1000, 1000, CCV_32S | CCV_C1, CCV_SPARSE_FULL, NULL);
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
				if (cell.ptr != NULL)
					assert(cell.i[0] == k);
				else {
					printf("ERROR: %d %d\n", i, j);
					exit(0);
				}
				k++;
			}
	}
	return 0;
}
