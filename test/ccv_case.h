#ifndef _GUARD_ccv_case_h_
#define _GUARD_ccv_case_h_

#include <math.h>

#define REQUIRE_MATRIX_EQ(a, b, err, ...) { \
if (ccv_matrix_eq(a, b) != 0) \
{ \
	printf("\n\t\033[0;31mREQUIRE_MATRIX_EQ\033[0;0m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #b, ##__VA_ARGS__); \
	ABORT_CASE; \
} }

#define REQUIRE_MATRIX_FILE_EQ(a, f, err, ...) { \
ccv_dense_matrix_t* __case_b__ = 0; \
ccv_read(f, &__case_b__, CCV_IO_ANY_FILE); \
if (ccv_matrix_eq(a, __case_b__) != 0) \
{ \
	printf("\n\t\033[0;31mREQUIRE_MATRIX_FILE_EQ\033[0;0m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #f, ##__VA_ARGS__); \
	ccv_matrix_free(__case_b__); \
	ABORT_CASE; \
} \
ccv_matrix_free(__case_b__); }

#define REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(type, a, b, len, angle, magnitude, err, ...) { \
int __case_i__; \
double __dot_prod__ = 0, __mag_a__ = 0, __mag_b__ = 0; \
for (__case_i__ = 0; __case_i__ < (len); __case_i__++) \
{ \
	__dot_prod__ += (double)(((type*)(a))[__case_i__] * ((type*)(b))[__case_i__]); \
	__mag_a__ += (double)(((type*)(a))[__case_i__] * ((type*)(a))[__case_i__]); \
	__mag_b__ += (double)(((type*)(b))[__case_i__] * ((type*)(b))[__case_i__]); \
} \
__mag_a__ = sqrt(__mag_a__), __mag_b__ = sqrt(__mag_b__); \
if (acos(__dot_prod__ / (__mag_a__ * __mag_b__)) * 180 / 3.141592653 > angle || fabs(__mag_a__ - __mag_b__) / ccv_max(ccv_max(__mag_a__, __mag_b__), 1) > magnitude) \
{ \
	printf("\n\t\033[0;31mREQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE\033[0;0m: %s:%d: angle: %lg | %lg, magnitude: %lg != %lg | +-%lg, " err, __FILE__, __LINE__, (double)(acos(__dot_prod__ / (__mag_a__ * __mag_b__)) * 180 / 3.141592653), (double)angle, __mag_a__, __mag_b__, (double)(magnitude), ##__VA_ARGS__); \
	ABORT_CASE; \
} }

#endif
