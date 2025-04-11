#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("unique consecutive a 1d tensor")
{
	float ap[] = {
		2, 2, 2, 0, 0, 2, 2, 1, 1, 2, 3
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 11), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 7), 0);
	ccv_nnc_cmd_exec(CMD_UNIQUE_CONSECUTIVE_FORWARD(7), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		2, 0, 2, 1, 2, 3, -1
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 7), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		3, 2, 2, 2, 1, 1, 0
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 7), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("unique consecutive a 1d tensor, smaller container")
{
	float ap[] = {
		2, 2, 2, 0, 0, 2, 2, 1, 1, 2, 3
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 11), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_cmd_exec(CMD_UNIQUE_CONSECUTIVE_FORWARD(7), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		2, 0, 2, 1, 2
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 5), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		3, 2, 2, 2, 1
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("unique consecutive a 1d tensor, int")
{
	int ap[] = {
		2, 2, 2, 0, 0, 2, 2, 1, 1, 2, 3
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 11), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 7), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 7), 0);
	ccv_nnc_cmd_exec(CMD_UNIQUE_CONSECUTIVE_FORWARD(7), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		2, 0, 2, 1, 2, 3, -1
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 7), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		3, 2, 2, 2, 1, 1, 0
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 7), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("unique consecutive a 1d tensor, int, smaller container")
{
	int ap[] = {
		2, 2, 2, 0, 0, 2, 2, 1, 1, 2, 3
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 11), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_cmd_exec(CMD_UNIQUE_CONSECUTIVE_FORWARD(7), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		2, 0, 2, 1, 2
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		3, 2, 2, 2, 1
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

#include "case_main.h"
