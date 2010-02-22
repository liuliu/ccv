#include "../src/ccv.h"

void copy_data_to_float(ccv_dense_matrix_t* x, float* out)
{
	float* out_ptr = out;
	unsigned char* m_ptr = x->data.ptr;
	int i, j;
	for (i = 0; i < x->rows; i++)
	{
		for (j = 0; j < x->cols; j++)
			out_ptr[j] = ccv_get_value(x->type, m_ptr, i);
		out_ptr += x->cols;
		m_ptr += x->step;
	}
}

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(100, 100, CCV_8U | CCV_C1, NULL, NULL);
	float* out = (float*)malloc(sizeof(float) * 100 * 100);
	copy_data_to_float(x, out);
	free(out);
	ccv_matrix_free(x);
	ccv_gabarge_collect();
	return 0;
}
