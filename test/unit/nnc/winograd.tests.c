#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("convolutional network of 3x3 on 56x56 with non-uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 128);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128, 3, 3, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 128 * 3 * 3 * 128; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (3 * 3 * 128);
	for (i = 0; i < 56 * 56 * 128; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		bias->data.f32[i] = (float)i / 128;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_OPT;
	cmd.algorithm = 2; // CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "56x56 matrix should be exactly the same from reference implementation and winograd.");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 3x3 on 55x55 with non-uniform weights")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(55, 55, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(55, 55, 128), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 128);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128, 3, 3, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 128 * 3 * 3 * 128; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (3 * 3 * 128);
	for (i = 0; i < 55 * 55 * 128; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		bias->data.f32[i] = (float)i / 128;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(55, 55, 128), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_OPT;
	cmd.algorithm = 2; // CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "55x55 matrix should be exactly the same from reference implementation and winograd.");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 3x3 on 224x224 with non-uniform weights and RGB channels")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(224, 224, 3), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(224, 224, 128), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128, 3, 3, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 3 * 3 * 3 * 128; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (3 * 3 * 3);
	for (i = 0; i < 224 * 224 * 3; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		bias->data.f32[i] = (float)i / 128;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(224, 224, 128), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_OPT;
	cmd.algorithm = 2; // CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "224x224 matrix should be exactly the same from reference implementation and winograd.");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("convolutional network of 3x3 on 56x56 with no bias")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 128, 3, 3, 128);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128, 3, 3, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 128 * 3 * 3 * 128; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (3 * 3 * 128);
	for (i = 0; i < 56 * 56 * 128; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_OPT;
	cmd.algorithm = 2; // CCV_NNC_CMD_OPT_CONV_ALGO_WINOGRAD
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "56x56 matrix should be exactly the same from reference implementation and winograd.");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

#include "case_main.h"
