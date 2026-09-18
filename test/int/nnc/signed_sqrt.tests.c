#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("signed sqrt MPS forward and backward across precisions shapes and floors")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGNED_SQRT_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SIGNED_SQRT_BACKWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	const int counts[] = {4, 257, 8193, 1, 1024, 8192, 1028, 513, 4};
	const float floors[] = {1e-6f, 0.0625f, 1e-8f, 0.25f, 1e-6f, 0.0625f, 0.25f, 1e-8f, 0.0625f};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 31);
	int mode, dtype, layout, step, i;
	// Specialized MFA, runtime-length MFA, and MPSGraph all compare with CPU.
	for (mode = 0; mode < 3; mode++)
	{
		if (mode == 2)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
		else
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		if (mode == 1)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
		else
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
		for (dtype = 0; dtype < 3; dtype++)
			for (layout = 0; layout < 2; layout++)
				for (step = 0; step < sizeof(counts) / sizeof(counts[0]); step++)
				{
					const int count = counts[step];
					const float minimum = floors[step];
					ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, count);
					params.format = layout ? CCV_TENSOR_FORMAT_NCHW : CCV_TENSOR_FORMAT_NHWC;
					ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const reference_b = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const reference_h = ccv_nnc_tensor_new(0, params, 0);
					params.datatype = datatypes[dtype];
					ccv_nnc_tensor_t* const typed_a = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const typed_g = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const typed_b = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const typed_h = ccv_nnc_tensor_new(0, params, 0);
					params.type = CCV_TENSOR_GPU_MEMORY;
					ccv_nnc_tensor_t* const gpu_a = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const gpu_g = ccv_nnc_tensor_new(0, params, 0);
					ccv_nnc_tensor_t* const gpu_b = ccv_nnc_tensor_new(0, params, 0);
					for (i = 0; i < count; i++)
					{
						a->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 32;
						g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
					}
					const float edges[] = {0, -0.0f, minimum, -minimum, minimum / 2, -minimum / 2, minimum * 2, -minimum * 2, 1, -1};
					memcpy(a->data.f32, edges, sizeof(float) * ccv_min(count, sizeof(edges) / sizeof(edges[0])));
					ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, g), TENSOR_LIST(typed_a, typed_g), 0);
					// CPU reference sees exactly the same rounded inputs as the GPU.
					ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(typed_a, typed_g), TENSOR_LIST(a, g), 0);
					ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(minimum), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(reference_b), 0);
					ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(minimum), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, 0), TENSOR_LIST(reference_h), 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(typed_a, typed_g), TENSOR_LIST(gpu_a, gpu_g), 0);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(minimum), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_a), TENSOR_LIST(gpu_b), 0), CCV_NNC_EXEC_SUCCESS, "MPS forward should run");
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(minimum), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_g, gpu_a, 0), TENSOR_LIST(gpu_g), 0), CCV_NNC_EXEC_SUCCESS, "MPS backward should run in place");
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_b, gpu_g), TENSOR_LIST(typed_b, typed_h), 0);
					ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(typed_b, typed_h), TENSOR_LIST(b, h), 0);
					const float tolerance = dtype == 0 ? 2e-6f : (dtype == 1 ? 1e-3f : 8e-3f);
					for (i = 0; i < count; i++)
					{
						REQUIRE(isfinite(b->data.f32[i]) && isfinite(h->data.f32[i]), "all outputs must be finite");
						REQUIRE_EQ_WITH_TOLERANCE(b->data.f32[i], reference_b->data.f32[i], tolerance * ccv_max(fabsf(reference_b->data.f32[i]), 1e-4f), "forward matches CPU for dtype=%d mode=%d step=%d index=%d", dtype, mode, step, i);
						REQUIRE_EQ_WITH_TOLERANCE(h->data.f32[i], reference_h->data.f32[i], tolerance * ccv_max(fabsf(reference_h->data.f32[i]), 1), "backward matches CPU for dtype=%d mode=%d step=%d index=%d", dtype, mode, step, i);
					}
					REQUIRE_EQ(!!signbit(b->data.f32[0]), 0, "positive zero selects the positive root");
					if (count > 1)
					{
						REQUIRE_EQ(!!signbit(b->data.f32[1]), 1, "negative zero selects the negative root");
					}
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(minimum), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_a), TENSOR_LIST(gpu_a), 0), CCV_NNC_EXEC_SUCCESS, "MPS forward should run in place");
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_a), TENSOR_LIST(typed_a), 0);
					REQUIRE(memcmp(typed_a->data.u8, typed_b->data.u8, ccv_nnc_tensor_data_size_without_padding(typed_a->info)) == 0, "in-place and separate outputs must agree");
					ccv_nnc_tensor_free(a);
					ccv_nnc_tensor_free(g);
					ccv_nnc_tensor_free(b);
					ccv_nnc_tensor_free(h);
					ccv_nnc_tensor_free(reference_b);
					ccv_nnc_tensor_free(reference_h);
					ccv_nnc_tensor_free(typed_a);
					ccv_nnc_tensor_free(typed_g);
					ccv_nnc_tensor_free(typed_b);
					ccv_nnc_tensor_free(typed_h);
					ccv_nnc_tensor_free(gpu_a);
					ccv_nnc_tensor_free(gpu_g);
					ccv_nnc_tensor_free(gpu_b);
				}
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
}

TEST_CASE("signed sqrt MPS tensor views preserve offsets strides and destination guards")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGNED_SQRT_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	int variant, i;
	for (variant = 0; variant < 4; variant++)
	{
		const int strided = variant == 3;
		const ccv_nnc_tensor_param_t a_params = strided ? CPU_TENSOR_NHWC(32F, 35, 21) : CPU_TENSOR_NHWC(32F, 268);
		const ccv_nnc_tensor_param_t b_params = strided ? CPU_TENSOR_NHWC(32F, 36, 23) : CPU_TENSOR_NHWC(32F, 273);
		const ccv_nnc_tensor_param_t view_params = strided ? CPU_TENSOR_NHWC(32F, 17, 9) : CPU_TENSOR_NHWC(32F, variant == 0 ? 257 : 260);
		ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
		ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, a_params, 0);
		ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
		ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, b_params, 0);
		ccv_nnc_tensor_t* const reference_b = ccv_nnc_tensor_new(0, b_params, 0);
		ccv_nnc_tensor_t* const reference_h = ccv_nnc_tensor_new(0, b_params, 0);
		for (i = 0; i < ccv_nnc_tensor_count(a_params); i++)
		{
			a->data.f32[i] = sinf(i * 0.7f) * 8;
			g->data.f32[i] = cosf(i * 0.3f);
		}
		for (i = 0; i < ccv_nnc_tensor_count(b_params); i++)
			b->data.f32[i] = h->data.f32[i] = reference_b->data.f32[i] = reference_h->data.f32[i] = 37;
		ccv_nnc_tensor_param_t gpu_params = a_params;
		gpu_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gpu_a = ccv_nnc_tensor_new(0, gpu_params, 0);
		ccv_nnc_tensor_t* const gpu_g = ccv_nnc_tensor_new(0, gpu_params, 0);
		gpu_params = b_params;
		gpu_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gpu_b = ccv_nnc_tensor_new(0, gpu_params, 0);
		ccv_nnc_tensor_t* const gpu_h = ccv_nnc_tensor_new(0, gpu_params, 0);
		const int* const aofs = strided ? DIM_ALLOC(0, 1) : DIM_ALLOC(variant == 2 ? 4 : 1);
		const int* const bofs = strided ? DIM_ALLOC(0, 1) : DIM_ALLOC(variant == 2 ? 8 : 3);
		const int* const astride = strided ? DIM_ALLOC(42, 2) : DIM_ALLOC(1);
		const int* const bstride = strided ? DIM_ALLOC(46, 2) : DIM_ALLOC(1);
		ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, view_params, aofs, astride);
		ccv_nnc_tensor_view_t* const gv = ccv_nnc_tensor_view_new(g, view_params, aofs, astride);
		ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(reference_b, view_params, bofs, bstride);
		ccv_nnc_tensor_view_t* const hv = ccv_nnc_tensor_view_new(reference_h, view_params, bofs, bstride);
		gpu_params = view_params;
		gpu_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_view_t* const gpu_av = ccv_nnc_tensor_view_new(gpu_a, gpu_params, aofs, astride);
		ccv_nnc_tensor_view_t* const gpu_gv = ccv_nnc_tensor_view_new(gpu_g, gpu_params, aofs, astride);
		ccv_nnc_tensor_view_t* const gpu_bv = ccv_nnc_tensor_view_new(gpu_b, gpu_params, bofs, bstride);
		ccv_nnc_tensor_view_t* const gpu_hv = ccv_nnc_tensor_view_new(gpu_h, gpu_params, bofs, bstride);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, g, b, h), TENSOR_LIST(gpu_a, gpu_g, gpu_b, gpu_h), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(0.0625), ccv_nnc_no_hint, 0, TENSOR_LIST(av), TENSOR_LIST(bv), 0), CCV_NNC_EXEC_SUCCESS, "CPU view forward should run");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(0.0625), ccv_nnc_no_hint, 0, TENSOR_LIST(gv, av, 0), TENSOR_LIST(hv), 0), CCV_NNC_EXEC_SUCCESS, "CPU view backward should run");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(0.0625), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_av), TENSOR_LIST(gpu_bv), 0), CCV_NNC_EXEC_SUCCESS, "MPS view forward should run");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(0.0625), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_gv, gpu_av, 0), TENSOR_LIST(gpu_hv), 0), CCV_NNC_EXEC_SUCCESS, "MPS view backward should run");
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_b, gpu_h), TENSOR_LIST(b, h), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, reference_b->data.f32, ccv_nnc_tensor_count(b_params), 1e-6, "forward view results and all guard elements should match CPU");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, h->data.f32, reference_h->data.f32, ccv_nnc_tensor_count(b_params), 1e-6, "backward view results and all guard elements should match CPU");
		ccv_nnc_tensor_view_free(av);
		ccv_nnc_tensor_view_free(gv);
		ccv_nnc_tensor_view_free(bv);
		ccv_nnc_tensor_view_free(hv);
		ccv_nnc_tensor_view_free(gpu_av);
		ccv_nnc_tensor_view_free(gpu_gv);
		ccv_nnc_tensor_view_free(gpu_bv);
		ccv_nnc_tensor_view_free(gpu_hv);
		ccv_nnc_tensor_free(a);
		ccv_nnc_tensor_free(g);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(h);
		ccv_nnc_tensor_free(reference_b);
		ccv_nnc_tensor_free(reference_h);
		ccv_nnc_tensor_free(gpu_a);
		ccv_nnc_tensor_free(gpu_g);
		ccv_nnc_tensor_free(gpu_b);
		ccv_nnc_tensor_free(gpu_h);
	}
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
}

TEST_CASE("signed sqrt MPS handles nonfinite inputs and empty tensors")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGNED_SQRT_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, a->info, 0);
	ccv_nnc_tensor_t* const gpu_a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const gpu_b = ccv_nnc_tensor_new(0, gpu_a->info, 0);
	const float values[] = {INFINITY, -INFINITY, NAN, -NAN};
	memcpy(a->data.f32, values, sizeof(values));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(gpu_a), 0);
	int fallback;
	for (fallback = 0; fallback < 2; fallback++)
	{
		if (fallback)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
		else
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(0.25), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_a), TENSOR_LIST(gpu_b), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_b), TENSOR_LIST(b), 0);
		REQUIRE(isinf(b->data.f32[0]) && !signbit(b->data.f32[0]), "positive infinity is preserved");
		REQUIRE(isinf(b->data.f32[1]) && signbit(b->data.f32[1]), "negative infinity is preserved");
		REQUIRE_EQ(b->data.f32[2], 0.5f, "positive NaN uses the floor as in fmax");
		REQUIRE_EQ(b->data.f32[3], -0.5f, "negative NaN keeps its sign");
	}
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	int dtype;
	for (dtype = 0; dtype < 3; dtype++)
	{
		ccv_nnc_tensor_param_t params = GPU_TENSOR_NHWC(000, 32F, 0);
		params.datatype = datatypes[dtype];
		ccv_nnc_tensor_t empty = ccv_nnc_tensor(0, params, 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(&empty), TENSOR_LIST(&empty), 0), CCV_NNC_EXEC_SUCCESS, "empty MPS forward should not dispatch");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(&empty, &empty, 0), TENSOR_LIST(&empty), 0), CCV_NNC_EXEC_SUCCESS, "empty MPS backward should not dispatch");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(&empty), TENSOR_LIST(&empty), 0), CCV_NNC_EXEC_INVALID, "MPS rejects a zero floor");
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(gpu_a);
	ccv_nnc_tensor_free(gpu_b);
}

#include "case_main.h"
