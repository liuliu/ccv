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

TEST_CASE("dropout 40% of a 20x50 matrix")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	int i;
	for (i = 0; i < 20 * 50; i++)
		a->data.f32[i] = (i + 1) * 0.01;
	ccv_nnc_tensor_param_t output_info[2];
	ccv_nnc_hint_tensor_auto(CMD_DROPOUT_FORWARD(0.4), &a->info, 1, ccv_nnc_no_hint, output_info, 2);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, output_info[1], 0);
	ccv_nnc_cmd_exec(CMD_DROPOUT_FORWARD(0.4), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, c), 0);
	int zero_count = 0;
	for (i = 0; i < 20 * 50; i++)
		if (b->data.f32[i] == 0)
			++zero_count;
		else {
			REQUIRE_EQ_WITH_TOLERANCE(a->data.f32[i] / 0.6, b->data.f32[i], 1e-5, "should be scaled up by 1 / 0.6");
		}
	REQUIRE_EQ_WITH_TOLERANCE((float)zero_count / (20 * 50), 0.4, 2 * 1e-2, "should be within 2%% of error");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("dropout gradient for 40% of a 20x30 matrix")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	int i;
	for (i = 0; i < 20 * 50; i++)
		a->data.f32[i] = (i + 1) * 0.01;
	ccv_nnc_tensor_param_t output_info[2];
	ccv_nnc_hint_tensor_auto(CMD_DROPOUT_FORWARD(0.4), &a->info, 1, ccv_nnc_no_hint, output_info, 2);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, output_info[1], 0);
	ccv_nnc_cmd_exec(CMD_DROPOUT_FORWARD(0.4), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, c), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	for (i = 0; i < 20 * 50; i++)
		g->data.f32[i] = i + 1;
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	ccv_nnc_cmd_exec(CMD_DROPOUT_BACKWARD(0.4), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, 0, 0, c), TENSOR_LIST(h), 0);
	int zero_count = 0;
	for (i = 0; i < 20 * 50; i++)
		if (h->data.f32[i] == 0)
			++zero_count;
	REQUIRE_EQ_WITH_TOLERANCE((float)zero_count / (20 * 50), 0.4, 2 * 1e-2, "should be within 2%% of error");
	ccv_nnc_tensor_t* const ht = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(20, 50), 0);
	ccv_nnc_tensor_zero(ht);
	for (i = 0; i < 20 * 50; i++)
		if (b->data.f32[i] != 0)
			ht->data.f32[i] = (i + 1) / 0.6;
	REQUIRE_TENSOR_EQ(h, ht, "propagated gradient should simply match the dropout");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ht);
}

#include "case_main.h"
