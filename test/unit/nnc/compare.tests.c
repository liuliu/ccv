#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("min of two tensors")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i - 1;
	ccv_nnc_cmd_exec(CMD_MIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "c should be the minimal of the two");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("min of two tensor views")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 4 * 4 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 4 * 4 * 3; i++)
		b->data.f32[i] = i - 1;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_cmd_exec(CMD_MIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)bv), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const bvt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)bv), TENSOR_LIST(bvt), 0);
	REQUIRE_TENSOR_EQ(bvt, c, "c should be the minimal of the two");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bvt);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
}

TEST_CASE("min of two tensors backward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i - 1;
	for (i = 0; i < 2 * 2 * 3; i++)
		g->data.f32[i] = -1;
	ccv_nnc_cmd_exec(CMD_MIN_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? -1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = -1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("min of two tensors backward with null")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i - 1;
	ccv_nnc_cmd_exec(CMD_MIN_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a, b), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? 1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = 1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("min of two tensor views backward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 4 * 4 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 4 * 4 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i - 1;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	for (i = 0; i < 2 * 2 * 3; i++)
		g->data.f32[i] = -1;
	ccv_nnc_cmd_exec(CMD_MIN_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)bv), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? -1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = -1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("min of two tensor views backward with null")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 4 * 4 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 4 * 4 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i - 1;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_cmd_exec(CMD_MIN_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)bv), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? 1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = 1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("min of two tensors with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i - 1;
	ccv_cnnp_model_t* const min = ccv_cnnp_min(0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 2, 3);
	const ccv_nnc_tensor_param_t b_params = CPU_TENSOR_NHWC(32F, 2, 2, 3);
	ccv_cnnp_model_compile(min, TENSOR_PARAM_LIST(a_params, b_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(min, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a, b), TENSOR_LIST(c), 0, 0);
	REQUIRE_TENSOR_EQ(b, c, "c should be the minimal of the two");
	ccv_cnnp_model_free(min);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("max of two tensors")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i + 1;
	ccv_nnc_cmd_exec(CMD_MAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "c should be the maximal of the two");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("max of two tensor views")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 4 * 4 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 4 * 4 * 3; i++)
		b->data.f32[i] = i + 1;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_cmd_exec(CMD_MAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)bv), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const bvt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)bv), TENSOR_LIST(bvt), 0);
	REQUIRE_TENSOR_EQ(bvt, c, "c should be the maximal of the two");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bvt);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
}

TEST_CASE("max of two tensors backward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i + 1;
	for (i = 0; i < 2 * 2 * 3; i++)
		g->data.f32[i] = -1;
	ccv_nnc_cmd_exec(CMD_MAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? -1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = -1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("max of two tensors backward with null")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i + 1;
	ccv_nnc_cmd_exec(CMD_MAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a, b), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? 1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = 1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("max of two tensor views backward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 4 * 4 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 4 * 4 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i + 1;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	for (i = 0; i < 2 * 2 * 3; i++)
		g->data.f32[i] = -1;
	ccv_nnc_cmd_exec(CMD_MAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)bv), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? -1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = -1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("max of two tensor views backward with null")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 4 * 4 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 4 * 4 * 3; i++)
		b->data.f32[i] = i < 3 ? i : i + 1;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2, 3), DIM_ALLOC(), DIM_ALLOC(4, 4, 3));
	ccv_nnc_cmd_exec(CMD_MAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)bv), TENSOR_LIST(ha, hb), 0);
	for (i = 0; i < 2 * 2 * 3; i++)
		hat->data.f32[i] = i < 3 ? 1 : 0;
	for (i = 0; i < 2 * 2 * 3; i++)
		hbt->data.f32[i] = 1;
	REQUIRE_TENSOR_EQ(ha, hat, "ha should only carry gradients for the first 3");
	REQUIRE_TENSOR_EQ(hb, hbt, "hb should only carry all gradients");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("max of two tensors with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i + 1;
	ccv_cnnp_model_t* const max = ccv_cnnp_max(0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 2, 3);
	const ccv_nnc_tensor_param_t b_params = CPU_TENSOR_NHWC(32F, 2, 2, 3);
	ccv_cnnp_model_compile(max, TENSOR_PARAM_LIST(a_params, b_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(max, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a, b), TENSOR_LIST(c), 0, 0);
	REQUIRE_TENSOR_EQ(b, c, "c should be the maximal of the two");
	ccv_cnnp_model_free(max);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

#include "case_main.h"
