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

TEST_CASE("argsort a tensor by first axis")
{
	float ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		0, 0,
		1, 2,
		2, 1,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by last axis")
{
	float ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		0, 1,
		1, 0,
		0, 1,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("sort a 1d tensor")
{
	float ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		0, 2, 1, 3, 4
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by first axis in int")
{
	int ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		0, 0,
		1, 2,
		2, 1,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by last axis in int")
{
	int ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		0, 1,
		1, 0,
		0, 1,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("sort a 1d tensor in int")
{
	int ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		0, 2, 1, 3, 4
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by first axis, descending")
{
	float ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		2, 1,
		1, 2,
		0, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by last axis, descending")
{
	float ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		1, 0,
		0, 1,
		1, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("sort a 1d tensor, descending")
{
	float ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		4, 3, 1, 2, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by first axis in int, descending")
{
	int ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		2, 1,
		1, 2,
		0, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("argsort a tensor by last axis in int, descending")
{
	int ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		1, 0,
		0, 1,
		1, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("sort a 1d tensor in int, descending")
{
	int ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_cmd_exec(CMD_ARGSORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(indices), 0);
	int btp[] = {
		4, 3, 1, 2, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 5), 0);
	REQUIRE_TENSOR_EQ(indices, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
}

#include "case_main.h"
