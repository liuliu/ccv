#ifndef _GUARD_ccv_case_h_
#define _GUARD_ccv_case_h_

#define REQUIRE_MATRIX_EQ(a, b, err, ...) { \
if (ccv_matrix_eq(a, b) != 0) \
{ \
	printf("\n\t\033[0;31mREQUIRE_MATRIX_EQ\033[0;30m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #b, ##__VA_ARGS__); \
	ABORT_CASE; \
} }

#define REQUIRE_MATRIX_FILE_EQ(a, f, err, ...) { \
ccv_dense_matrix_t* __case_b__ = 0; \
ccv_unserialize(f, &__case_b__, CCV_SERIAL_ANY_FILE); \
if (ccv_matrix_eq(a, __case_b__) != 0) \
{ \
	printf("\n\t\033[0;31mREQUIRE_MATRIX_FILE_EQ\033[0;30m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #f, ##__VA_ARGS__); \
	ccv_matrix_free(__case_b__); \
	ABORT_CASE; \
} \
ccv_matrix_free(__case_b__); }

#endif
