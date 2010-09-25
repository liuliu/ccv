#include "ccv.h"
#include <assert.h>

int main(int argc, char** argv)
{
	int i;
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 10, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 10; i++)
		dmt->data.fl[i] = i;
	ccv_normalize(dmt, (ccv_matrix_t**)&dmt, 0, CCV_L2_NORM);
	for (i = 0; i < 10; i++)
		printf("%f ", dmt->data.fl[i]);
	printf("\n");
	ccv_matrix_free(dmt);
	ccv_garbage_collect();
	return 0;
}
