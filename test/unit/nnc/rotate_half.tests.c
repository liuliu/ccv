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

TEST_CASE("rotate half of a tensor")
{
	float ap[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		3, 4, 1, 2,
		7, 8, 5, 6,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should rotate the last dim by half");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("rotate half backward of a tensor")
{
	float gp[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(h), 0);
	float htp[] = {
		3, 4, 1, 2,
		7, 8, 5, 6,
	};
	ccv_nnc_tensor_t const ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "should rotate gradient by half");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
}

TEST_CASE("rotate half of a tensor view")
{
	float ap[] = {
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 6), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 4), DIM_ALLOC(0, 1), DIM_ALLOC(6, 1));
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 6), 0);
	memset(b->data.f32, 0, sizeof(float) * 12);
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 4), DIM_ALLOC(0, 1), DIM_ALLOC(6, 1));
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	float btp[] = {
		0, 4, 5, 2, 3, 0,
		0, 10, 11, 8, 9, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 6), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should rotate the last dim by half for tensor views");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
}

TEST_CASE("ccv_cnnp rotate half")
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_io_t y = ccv_cnnp_model_apply(ccv_cnnp_rotate_half("rotate_half"), MODEL_IO_LIST(x));
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(y), 0, "rotate_half");
	const ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 2, 4);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	float ap[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	float btp[] = {
		3, 4, 1, 2,
		7, 8, 5, 6,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "ccv_cnnp rotate half should rotate the last dim by half");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_cnnp_model_free(model);
}

#include "case_main.h"
