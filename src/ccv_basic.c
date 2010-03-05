#include "ccv.h"

void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int dx, int dy)
{
	int sig[5];
	char identifier[64];
	memset(identifier, 0, 64);
	sprintf(identifier, "ccv_sobel(%d,%d)", dx, dy);
	ccv_matrix_generate_signature(identifier, 64, sig, a->sig, NULL);
	ccv_dense_matrix_t* db;
	if (*b == NULL)
	{
		*b = db = ccv_dense_matrix_new(a->rows, a->cols, CCV_32S | CCV_C1, NULL, sig);
		if (db->type & CCV_GARBAGE)
		{
			db->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		db = *b;
		memcpy(db->sig, sig, 20);
	}
	int i, j;
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(a->rows, a->cols, CCV_32S | CCV_C1, NULL, NULL);
	int* c_ptr = c->data.i;
	if (dx > dy)
	{
		for (i = 0; i < a->rows; i++)
		{
			c_ptr[0] = a_ptr[1] - a_ptr[0];
			for (j = 1; j < a->cols - 1; j++)
				c_ptr[j] = a_ptr[j + 1] - a_ptr[j - 1];
			c_ptr[c->cols - 1] = a_ptr[a->cols - 1] - a_ptr[a->cols - 2];
			a_ptr += a->step;
			c_ptr += c->cols;
		}
		c_ptr = c->data.i;
		for (j = 0; j < c->cols; j++)
			ccv_set_value(db->type, b_ptr, j, c_ptr[j + c->cols] + 2 * c_ptr[j]);
		b_ptr += db->step;
		c_ptr += c->cols;
		for (i = 1; i < c->rows - 1; i++)
		{
			for (j = 0; j < c->cols; j++)
				ccv_set_value(db->type, b_ptr, j, c_ptr[j + c->cols] + 2 * c_ptr[j] + c_ptr[j - c->cols]);
			b_ptr += db->step;
			c_ptr += c->cols;
		}
		for (j = 0; j < c->cols; j++)
			ccv_set_value(db->type, b_ptr, j, 2 * c_ptr[j] + c_ptr[j - c->cols]);
	} else {
		for (j = 0; j < a->cols; j++)
			c_ptr[j] = a_ptr[j + a->step] - a_ptr[j];
		a_ptr += a->step;
		c_ptr += c->cols;
		for (i = 1; i < a->rows - 1; i++)
		{
			for (j = 0; j < a->cols; j++)
				c_ptr[j] = a_ptr[j + a->step] - a_ptr[j - a->step];
			a_ptr += a->step;
			c_ptr += c->cols;
		}
		for (j = 0; j < a->cols; j++)
			c_ptr[j] = a_ptr[j] - a_ptr[j - a->step];
		c_ptr = c->data.i;
		for (i = 0; i < c->rows; i++)
		{
			ccv_set_value(db->type, b_ptr, 0, c_ptr[1] + 2 * c_ptr[0]);
			for (j = 1; j < c->cols - 1; j++)
				ccv_set_value(db->type, b_ptr, j, c_ptr[j + 1] + 2 * c_ptr[j] + c_ptr[j - 1]);
			ccv_set_value(db->type, b_ptr, c->cols - 1, c_ptr[c->cols - 1] + 2 * c_ptr[c->cols - 2]);
			b_ptr += db->step;
			c_ptr += c->cols;
		}
	}
	ccv_matrix_free(c);
}

void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int size, int drive)
{
	int sig[5];
	ccv_matrix_generate_signature("ccv_hog", 7, sig, a->sig, NULL);
	ccv_dense_matrix_t* db;
	if (*b == NULL)
	{
		*b = db = ccv_dense_matrix_new(a->rows, a->cols * 8, CCV_32F | CCV_C1, NULL, sig);
		if (db->type & CCV_GARBAGE)
		{
			db->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		db = *b;
		memcpy(db->sig, sig, 20);
	}
	ccv_dense_matrix_t* dx = NULL;
	ccv_dense_matrix_t* dy = NULL;
	ccv_sobel(a, &dx, 1, 0);
	ccv_sobel(a, &dy, 0, 1);
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
}

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int rows, int cols, int type)
{
}

void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
}

void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
}
