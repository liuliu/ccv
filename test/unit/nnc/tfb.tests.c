#include <stddef.h>

#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "nnc/ccv_nnc.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

#if CCV_NNC_TENSOR_TFB
TEST_CASE("toll-free bridging between ccv_nnc_tensor_t and ccv_dense_matrix_t")
{
	REQUIRE(offsetof(ccv_nnc_tensor_t, type) == offsetof(ccv_dense_matrix_t, type), "type offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, sig) == offsetof(ccv_dense_matrix_t, sig), "sig offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, refcount) == offsetof(ccv_dense_matrix_t, refcount), "refcount offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, data) == offsetof(ccv_dense_matrix_t, data), "data offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, info) + offsetof(ccv_nnc_tensor_param_t, type) == offsetof(ccv_dense_matrix_t, resides), "info.type and resides offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, info) + offsetof(ccv_nnc_tensor_param_t, format) == offsetof(ccv_dense_matrix_t, format), "info.format and format offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, info) + offsetof(ccv_nnc_tensor_param_t, dim) == offsetof(ccv_dense_matrix_t, rows), "info.dim[0] and rows offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, info) + offsetof(ccv_nnc_tensor_param_t, dim) + sizeof(int) * 3 == offsetof(ccv_dense_matrix_t, reserved1), "info.dim[3] and reserved offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, info) + offsetof(ccv_nnc_tensor_param_t, dim) + sizeof(int) * 4 == offsetof(ccv_dense_matrix_t, step), "info.dim[4] and step offset should be the same");
}
#endif

TEST_CASE("toll-free bridging between ccv_nnc_tensor_t and ccv_nnc_tensor_view_t")
{
	REQUIRE(offsetof(ccv_nnc_tensor_t, type) == offsetof(ccv_nnc_tensor_view_t, type), "type offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, sig) == offsetof(ccv_nnc_tensor_view_t, sig), "sig offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, refcount) == offsetof(ccv_nnc_tensor_view_t, refcount), "refcount offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, data) == offsetof(ccv_nnc_tensor_view_t, data), "data offset should be the same");
	REQUIRE(offsetof(ccv_nnc_tensor_t, info) == offsetof(ccv_nnc_tensor_view_t, info), "info offset should be the same");
}

#include "case_main.h"
