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

TEST_CASE("dropout half of [[1, 2, 3], [4, 5, 6]]")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_tensor_param_t output_info[2];
	ccv_nnc_hint_tensor_auto(CMD_DROPOUT_FORWARD(0.5), &a->info, 1, ccv_nnc_no_hint, output_info, 2);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, output_info[1], 0);
	ccv_nnc_cmd_exec(CMD_DROPOUT_FORWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, c), 0);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

#include "case_main.h"
