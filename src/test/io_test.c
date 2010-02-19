#include "../ccv.h"

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* x = NULL;
	ccv_unserialize(argv[1], &x, CCV_SERIAL_PNG_FILE);
	int i, j;
	printf("P3\n%d %d\n255\n", x->cols, x->rows);
	for (i = 0; i < x->rows; i++)
	{
		for (j = 0; j < x->cols; j++)
			printf("%d %d %d ", x->data.ptr[i * x->step + j * 3], x->data.ptr[i * x->step + j * 3 + 1], x->data.ptr[i * x->step + j * 3 + 2]);
		printf("\n");
	}
	ccv_matrix_free(x);
	ccv_garbage_collect();
	return 0;
}
