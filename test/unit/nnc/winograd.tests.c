#include "case.h"
#include "ccv_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>

TEST_CASE("convolutional network of 3x3 on 56x56 with non-uniform weights")
{
	/*
	ccv_nnc_init();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(512, 56, 56), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(512, 54, 54), 0);
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, 0, CMD_CONVOLUTIONAL(512, 512, 3, 3), 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_guess(cmd.info, &a->info, 1, &b->info, 1);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(512, 3, 3, 512), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(512), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < 512 * 3 * 3 * 512; i++)
		w->data.f32[i] = (float)i / 512;
	for (i = 0; i < 56 * 56 * 512; i++)
		a->data.f32[i] = (float)i / 1024;
	for (i = 0; i < 512; i++)
		bias->data.f32[i] = i;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(512, 54, 54), 0);
	cmd.backend = 0; // CCV_NNC_BACKEND_CPU_OPT = 0
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c));
	REQUIRE_MATRIX_EQ(b, c, "54x54 matrix should be exactly the same from reference implementation and winograd.");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	*/
}

#include "case_main.h"
