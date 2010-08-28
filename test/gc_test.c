#include "ccv.h"
#include <assert.h>

int main(int argc, char** argv)
{
	int i;
	for (i = 0; i < 50000; i++)
	{
		ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32S | CCV_C1, 0, 0);
		dmt->data.i[0] = i;
		dmt->sig = ccv_matrix_generate_signature((const char*)&i, 4, 0);
		dmt->type |= CCV_REUSABLE;
		ccv_matrix_free(dmt);
	}
	// ccv_garbage_collect();
	for (i = 0; i < 50000; i++)
	{
		uint64_t sig = ccv_matrix_generate_signature((const char*)&i, 4, 0);
		ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32S | CCV_C1, 0, sig);
		assert(i == dmt->data.i[0]);
		ccv_matrix_free(dmt);
	}
	ccv_garbage_collect();
	return 0;
}
