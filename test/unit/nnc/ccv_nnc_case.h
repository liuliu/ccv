#ifndef _GUARD_ccv_nnc_case_h_
#define _GUARD_ccv_nnc_case_h_

#include <math.h>

#define REQUIRE_TENSOR_EQ(a, b, err, ...) { \
if (ccv_nnc_tensor_eq(a, b) != 0) \
{ \
	printf("\n\t\033[0;31mREQUIRE_TENSOR_EQ\033[0;0m: %s:%d: %s != %s, " err, __FILE__, __LINE__, #a, #b, ##__VA_ARGS__); \
	ABORT_CASE; \
} }

#endif
