#ifndef _GUARD_ccv_assert_h_
#define _GUARD_ccv_assert_h_

#define CCV_ASSERT_MATRIX_EQUAL(x,y) \
	(ccv_matrix_equal(x,y) == 0) ? void : printf("%s %d %s\n", __FILE__, __LINE__, __func__);

#define CCV_ASSERT_MATRIX_EQUAL_FILE(x,f) { \
	ccv_dense_matrix_t* y = 0; \
	ccv_unserialize(f,&y,CCV_SERIAL_ANY_FILE); \
	(y != 0 && ccv_matrix_equal(x,y) == 0) ? void : printf("%s %d %s\n", __FILE__, __LINE__, __func__); \
	ccv_matrix_free(y); \
}

#endif
