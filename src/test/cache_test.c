#include "../ccv.h"
#include <assert.h>

int main(int argc, char** argv)
{
	int i;
	for (i = 0; i < 50000; i++)
	{
		ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32S | CCV_C1, NULL, NULL);
		dmt->data.i[0] = i;
		ccv_matrix_generate_signature((const char*)&i, 4, dmt->sig, NULL, NULL, NULL, NULL);
		ccv_matrix_free(dmt);
	}
	// ccv_garbage_collect();
	for (i = 0; i < 50000; i++)
	{
		int sig[5];
		ccv_matrix_generate_signature((const char*)&i, 4, sig, NULL, NULL, NULL, NULL);
		ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 1, CCV_32S | CCV_C1, NULL, sig);
		assert(i == dmt->data.i[0]);
	}
	return 0;
}
