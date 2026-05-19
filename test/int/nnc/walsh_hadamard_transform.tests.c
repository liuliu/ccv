#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("walsh hadamard transform matches known values")
{
	float ap[] = {
		1, 2, 3, 4,
		-1, 0, 1, 2,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float bp[] = {
		5, -1, -2, 0,
		1, -1, -2, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(bp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "Walsh-Hadamard transform should match the CPU reference values");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("walsh hadamard transform is self-inverse with normalized scale")
{
	const int row_count = 17;
	const int dim = 128;
	const int count = row_count * dim;
	const float scale = 1.0f / sqrtf((float)dim);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 128), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < count; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_BACKWARD(scale), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, a->data.f32, c->data.f32, count, 1e-5, "normalized Walsh-Hadamard transform should be self-inverse");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("compare walsh hadamard transform with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int row_count = 576;
	const int dim = 128;
	const int count = row_count * dim;
	const float scale = 0.08838834764831845f;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 576, 128), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 576, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 576, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 576, 128), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 576, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	int i;
	for (i = 0; i < count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t wht = CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale);
	wht.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_exec(wht, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb->data.f32, count, 2e-5, "mps should match CPU reference implementation");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("compare walsh hadamard transform in half precision with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int row_count = 13 * 17;
	const int dim = 64;
	const int count = row_count * dim;
	const float scale = 1.0f / sqrtf((float)dim);
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const ha_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const hb32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 13, 17, 64), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	int i;
	for (i = 0; i < count; i++)
		ha32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_half_precision(ha32->data.f32, (uint16_t*)ha->data.f16, count);
	ccv_half_precision_to_float((uint16_t*)ha->data.f16, ha_rounded->data.f32, count);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t wht = CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale);
	wht.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_exec(wht, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_half_precision_to_float((uint16_t*)hb->data.f16, hb32->data.f32, count);
	ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_rounded), TENSOR_LIST(hbt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb32->data.f32, count, 2e-3, "mps should match CPU reference implementation for half precision");
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha_rounded);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hb32);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("walsh hadamard transform supports inplace execution with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int row_count = 23;
	const int dim = 128;
	const int count = row_count * dim;
	const float scale = 1.0f / sqrtf((float)dim);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 23, 128), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 23, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 23, 128), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 23, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4);
	int i;
	for (i = 0; i < count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t wht = CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale);
	wht.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_exec(wht, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(scale), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb->data.f32, count, 2e-5, "mps should support inplace Walsh-Hadamard transform");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

#include "case_main.h"
