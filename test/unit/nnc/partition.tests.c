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

TEST_CASE("partition a tensor by first axis")
{
	float ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(2, 0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		1, 3,
		2, 4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		0, 0,
		1, 2,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by last axis")
{
	float ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(1, 1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		1,
		1,
		3,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		0,
		1,
		0,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a 1d tensor")
{
	float ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(4, 0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		1, 2, 3, 4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		0, 2, 1, 3
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 4), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by first axis in int")
{
	int ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(2, 0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		1, 3,
		2, 4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		0, 0,
		1, 2,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by last axis in int")
{
	int ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(1, 1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		1,
		1,
		3,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		0,
		1,
		0,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a 1d tensor in int")
{
	int ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(4, 0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		1, 2, 3, 4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		0, 2, 1, 3
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 4), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by first axis, descending")
{
	float ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(2, 0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		3, 5,
		2, 4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		2, 1,
		1, 2,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by last axis, descending")
{
	float ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(1, 1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		3,
		2,
		4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		1,
		0,
		1,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a 1d tensor, descending")
{
	float ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(4, 0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	float btp[] = {
		5, 4, 3, 2,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		4, 3, 1, 2
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 4), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by first axis in int, descending")
{
	int ap[] = {
		1, 3,
		2, 5,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(2, 0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		3, 5,
		2, 4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		2, 1,
		1, 2,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 2, 2), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by last axis in int, descending")
{
	int ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(1, 1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		3,
		2,
		4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		1,
		0,
		1,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a 1d tensor in int, descending")
{
	int ap[] = {
		1, 3, 2, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4), 0);
	ccv_nnc_cmd_exec(CMD_PARTITION_FORWARD(4, 0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	int btp[] = {
		5, 4, 3, 2,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		4, 3, 1, 2
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 4), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
}

TEST_CASE("partition a tensor by last axis in int, descending with model")
{
	int ap[] = {
		1, 3,
		2, 1,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32S, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	ccv_cnnp_model_t* const partition = ccv_cnnp_partition(1, 1, 1, "partition");
	ccv_cnnp_model_compile(partition, TENSOR_PARAM_LIST(a->info), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(partition, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
	}, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0, 0);
	int btp[] = {
		3,
		2,
		4,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	int indicestp[] = {
		1,
		0,
		1,
	};
	ccv_nnc_tensor_t const indicest = ccv_nnc_tensor(indicestp, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	REQUIRE_TENSOR_EQ(indices, &indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
	ccv_cnnp_model_free(partition);
}

#include "case_main.h"
