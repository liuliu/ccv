#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("index select a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		2, 3,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		2, 3,
		2, 3,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("index select a tensor with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		2, 3,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	float ip[] = {1.5, 0.4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		2.5, 3.5,
		1.4, 2.4,
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, btp, 4, 1e-5, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("index select a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2, 3, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	int ip[] = {3, 2, 4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		4, 3, 5
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("index select a tensor view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	memset(b->data.f32, 0, 2 * 4 * sizeof(float));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const gav = ccv_nnc_tensor_view_new(ga, GPU_TENSOR_NHWC(000, 32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const gbv = ccv_nnc_tensor_view_new(gb, GPU_TENSOR_NHWC(000, 32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, b), TENSOR_LIST(ga, gindices, gb), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)gav, gindices), TENSOR_LIST((ccv_nnc_tensor_t*)gbv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		0, 3, 4, 0,
		0, 3, 4, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_view_free(gav);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_view_free(gbv);
}

TEST_CASE("backward index select a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float bp[] = {
		1, 2,
		2, 3,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(gb, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, 0, gindices), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0,
		3, 5,
		0, 0,
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("backward index select a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float bp[] = {
		4, 3, 5,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	int ip[] = {3, 2, 4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(gb, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, 0, gindices), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0, 3, 4, 5
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 5), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("backward index select a tensor view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float bp[] = {
		0, 3, 4, 0,
		0, 1, 5, 0,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	int i;
	for (i = 0; i < 3 * 4; i++)
		a->data.f32[i] = i;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const gav = ccv_nnc_tensor_view_new(ga, GPU_TENSOR_NHWC(000, 32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const gbv = ccv_nnc_tensor_view_new(gb, GPU_TENSOR_NHWC(000, 32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, b), TENSOR_LIST(ga, gindices, gb), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)gbv, 0, gindices), TENSOR_LIST((ccv_nnc_tensor_t*)gav), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0, 0, 3,
		4, 4, 9, 7,
		8, 0, 0, 11,
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_view_free(gav);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_view_free(gbv);
}

TEST_CASE("index select forward with half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100, 10), 0);
	int i;
	for (i = 0; i < 100 * 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10), 0);
	for (i = 0; i < 10; i++)
		indices->data.i32[i] = i * 9 + 1;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100, 10), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_t* const b32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* const bt32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(b32, bt32), 0);
	REQUIRE_TENSOR_EQ(b32, bt32, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(b32);
	ccv_nnc_tensor_free(bt32);
}

TEST_CASE("mps mfa index select forward with int, half and bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int datatypes[] = { CCV_32S, CCV_16F, CCV_16BF };
	const int widths[] = { 1, 2, 3, 4, 6, 10, 127, 128, 129, 512 };
	const int rows = 97;
	const int selected = 31;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	int datatype_index;
	for (datatype_index = 0; datatype_index < sizeof(datatypes) / sizeof(datatypes[0]); datatype_index++)
	{
		int width_index;
		for (width_index = 0; width_index < sizeof(widths) / sizeof(widths[0]); width_index++)
		{
			const int datatype = datatypes[datatype_index];
			const size_t element_size = datatype == CCV_32S ? sizeof(int32_t) : sizeof(uint16_t);
			const int cols = widths[width_index];
			ccv_nnc_tensor_param_t host_params = {
				.type = CCV_TENSOR_CPU_MEMORY,
				.format = CCV_TENSOR_FORMAT_NHWC,
				.datatype = datatype,
				.dim = { rows, cols, 0 },
			};
			ccv_nnc_tensor_param_t host_output_params = host_params;
			host_output_params.dim[0] = selected;
			ccv_nnc_tensor_param_t gpu_params = host_params;
			gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_param_t gpu_output_params = host_output_params;
			gpu_output_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, host_params, 0);
			ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected), 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, host_output_params, 0);
			ccv_nnc_tensor_t* const gpu_source = ccv_nnc_tensor_new(0, gpu_params, 0);
			ccv_nnc_tensor_t* const gpu_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
			ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, gpu_output_params, 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host_output_params, 0);
			int i;
			if (datatype == CCV_32S)
			{
				for (i = 0; i < rows * cols; i++)
					((uint32_t*)source->data.i32)[i] = (uint32_t)i * 1000003U + (uint32_t)cols * 193U + 4144967295U;
			} else {
				for (i = 0; i < rows * cols; i++)
					((uint16_t*)source->data.f16)[i] = (uint16_t)((i * 73 + cols * 19 + datatype_index * 997) & 0xffff);
			}
			for (i = 0; i < selected; i++)
			{
				const int source_row = (i * 29 + 7) % rows;
				indices->data.i32[i] = source_row;
				memcpy(expected->data.u8 + (size_t)i * cols * element_size, source->data.u8 + (size_t)source_row * cols * element_size, (size_t)cols * element_size);
			}
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(source, indices), TENSOR_LIST(gpu_source, gpu_indices), 0);
			ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source, gpu_indices), TENSOR_LIST(gpu_output), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(actual), 0);
			REQUIRE_ARRAY_EQ(uint8_t, expected->data.u8, actual->data.u8, selected * cols * element_size, "MFA index select should copy datatype=%d rows with width=%d exactly", datatype, cols);
			ccv_nnc_tensor_free(actual);
			ccv_nnc_tensor_free(gpu_output);
			ccv_nnc_tensor_free(gpu_indices);
			ccv_nnc_tensor_free(gpu_source);
			ccv_nnc_tensor_free(expected);
			ccv_nnc_tensor_free(indices);
			ccv_nnc_tensor_free(source);
		}
	}
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
}

TEST_CASE("mps mfa index select forward on a 1d half tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 101;
	const int selected = 37;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, selected), 0);
	ccv_nnc_tensor_t* const gpu_source = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows), 0);
	ccv_nnc_tensor_t* const gpu_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, selected), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, selected), 0);
	uint16_t* const source_data = (uint16_t*)source->data.f16;
	int i;
	for (i = 0; i < rows; i++)
		source_data[i] = (uint16_t)(i * 193 + 17);
	for (i = 0; i < selected; i++)
	{
		indices->data.i32[i] = (i * 41 + 13) % rows;
		((uint16_t*)expected->data.f16)[i] = source_data[indices->data.i32[i]];
	}
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(source, indices), TENSOR_LIST(gpu_source, gpu_indices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source, gpu_indices), TENSOR_LIST(gpu_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(actual), 0);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_ARRAY_EQ(uint16_t, expected->data.f16, actual->data.f16, selected, "MFA index select should copy 1d half values exactly");
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gpu_output);
	ccv_nnc_tensor_free(gpu_indices);
	ccv_nnc_tensor_free(gpu_source);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("mps mfa index select forward with dynamic M including decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int datatypes[] = { CCV_32S, CCV_16F, CCV_16BF };
	const int selected_rows[] = { 1, 33 };
	const int rows = 127;
	const int cols = 512;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	int datatype_index;
	for (datatype_index = 0; datatype_index < sizeof(datatypes) / sizeof(datatypes[0]); datatype_index++)
	{
		const int datatype = datatypes[datatype_index];
		const size_t element_size = datatype == CCV_32S ? sizeof(int32_t) : sizeof(uint16_t);
		ccv_nnc_tensor_param_t host_source_params = {
			.type = CCV_TENSOR_CPU_MEMORY,
			.format = CCV_TENSOR_FORMAT_NHWC,
			.datatype = datatype,
			.dim = { rows, cols, 0 },
		};
		ccv_nnc_tensor_param_t gpu_source_params = host_source_params;
		gpu_source_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, host_source_params, 0);
		ccv_nnc_tensor_t* const gpu_source = ccv_nnc_tensor_new(0, gpu_source_params, 0);
		int i;
		if (datatype == CCV_32S)
		{
			for (i = 0; i < rows * cols; i++)
				((uint32_t*)source->data.i32)[i] = (uint32_t)i * 1000033U + (uint32_t)datatype_index * 1237U + 4124967293U;
		} else {
			for (i = 0; i < rows * cols; i++)
				((uint16_t*)source->data.f16)[i] = (uint16_t)((i * 89 + datatype_index * 1237 + 31) & 0xffff);
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(source), TENSOR_LIST(gpu_source), 0);
		int selected_index;
		for (selected_index = 0; selected_index < sizeof(selected_rows) / sizeof(selected_rows[0]); selected_index++)
		{
			const int selected = selected_rows[selected_index];
			ccv_nnc_tensor_param_t host_output_params = host_source_params;
			host_output_params.dim[0] = selected;
			ccv_nnc_tensor_param_t gpu_output_params = host_output_params;
			gpu_output_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected), 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, host_output_params, 0);
			ccv_nnc_tensor_t* const gpu_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
			ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, gpu_output_params, 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host_output_params, 0);
			for (i = 0; i < selected; i++)
			{
				const int source_row = (i * 43 + 17) % rows;
				indices->data.i32[i] = source_row;
				memcpy(expected->data.u8 + (size_t)i * cols * element_size, source->data.u8 + (size_t)source_row * cols * element_size, (size_t)cols * element_size);
			}
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(indices), TENSOR_LIST(gpu_indices), 0);
			ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_source, gpu_indices), TENSOR_LIST(gpu_output), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(actual), 0);
			REQUIRE_ARRAY_EQ(uint8_t, expected->data.u8, actual->data.u8, selected * cols * element_size, "dynamic-M MFA index select should copy datatype=%d rows with M=%d exactly", datatype, selected);
			ccv_nnc_tensor_free(actual);
			ccv_nnc_tensor_free(gpu_output);
			ccv_nnc_tensor_free(gpu_indices);
			ccv_nnc_tensor_free(expected);
			ccv_nnc_tensor_free(indices);
		}
		ccv_nnc_tensor_free(gpu_source);
		ccv_nnc_tensor_free(source);
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
}

TEST_CASE("gpu index select forward with 32s tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 97;
	const int cols = 13;
	const int selected = 31;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, cols), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
		a->data.i32[i] = ((i * 37) % 1009) - 503;
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected), 0);
	for (i = 0; i < selected; i++)
		indices->data.i32[i] = (i * 17 + 11) % rows;
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, cols), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(actual), 0);
	REQUIRE_ARRAY_EQ(int, expected->data.i32, actual->data.i32, selected * cols, "GPU index select should support 32s input and output tensors");
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
}

static int _mps_index_select_8i_rowwise_x_compare(const int datatype, const int format, const int rows, const int cols, const int selected, const double tolerance, double* const max_abs_ref)
{
	ccv_nnc_tensor_param_t host_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { rows, cols, 0 },
	};
	ccv_nnc_tensor_param_t output_params = host_params;
	output_params.dim[0] = selected;
	ccv_nnc_tensor_param_t gpu_params = host_params;
	gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_param_t gpu_output_params = output_params;
	gpu_output_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, host_params, 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(host_params, format), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, host_params, 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, output_params, 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gpu_params, format), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, gpu_output_params, 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, output_params, 0);
	const size_t count = (size_t)rows * cols;
	float* const values = ccmalloc(sizeof(float) * count);
	size_t i;
	for (i = 0; i < count; i++)
		values[i] = ((int)((i * 31 + format * 7) % 257) - 128) / 67.0f;
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)source->data.f16, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)source->data.f16, count);
	else
		memcpy(source->data.f32, values, sizeof(float) * count);
	int j;
	for (j = 0; j < selected; j++)
		indices->data.i32[j] = (j * 17 + 11) % rows;
	const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	int status = 0;
	if (qsize != ccv_nnc_tensor_data_size_without_padding(q->info))
	{
		status = -1;
		goto cleanup;
	}
	ccv_nnc_dequantize_8i_rowwise_x(q->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, cols, format, dequantized->data.u8, count);
	const size_t row_bytes = (size_t)cols * CCV_GET_DATA_TYPE_SIZE(datatype);
	for (j = 0; j < selected; j++)
		memcpy(expected->data.u8 + (size_t)j * row_bytes, dequantized->data.u8 + (size_t)indices->data.i32[j] * row_bytes, row_bytes);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	const size_t output_count = (size_t)selected * cols;
	float* const expected_f32 = ccmalloc(sizeof(float) * output_count);
	float* const actual_f32 = ccmalloc(sizeof(float) * output_count);
	if (datatype == CCV_16F)
	{
		ccv_half_precision_to_float((uint16_t*)expected->data.f16, expected_f32, output_count);
		ccv_half_precision_to_float((uint16_t*)actual->data.f16, actual_f32, output_count);
	} else if (datatype == CCV_16BF) {
		ccv_bfloat_to_float((uint16_t*)expected->data.f16, expected_f32, output_count);
		ccv_bfloat_to_float((uint16_t*)actual->data.f16, actual_f32, output_count);
	} else {
		memcpy(expected_f32, expected->data.f32, sizeof(float) * output_count);
		memcpy(actual_f32, actual->data.f32, sizeof(float) * output_count);
	}
	double max_abs = 0;
	for (i = 0; i < output_count; i++)
		max_abs = ccv_max(max_abs, fabs((double)expected_f32[i] - (double)actual_f32[i]));
	if (max_abs > tolerance)
		status = 1;
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	ccfree(actual_f32);
	ccfree(expected_f32);
cleanup:
	ccfree(values);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	return status;
}

TEST_CASE("mps index select row-wise 8i quantized tensor with float output")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 23;
	const int cols = 33;
	const int selected = 9;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = ((i * 19) % 67 - 33) / 48.0f;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	int ip[] = {3, 5, 3, 22, 0, 11, 5, 18, 7};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, cols, dequantized->data.f32, rows * cols);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dequantized, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, rows, cols)), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, selected * cols, 1e-5, "MPS index select should dequantize only selected row-wise 8i rows");
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("mps index select row-wise 8i quantized tensor with half output")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 17;
	const int cols = 32;
	const int selected = 8;
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = ((i * 23) % 59 - 29) / 40.0f;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_float_to_half_precision(values, (uint16_t*)source->data.f16, rows * cols);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	int ip[] = {1, 12, 1, 16, 3, 0, 12, 8};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, qsize, cols, dequantized->data.f16, rows * cols);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dequantized, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16F, rows, cols)), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = ccmalloc(sizeof(float) * selected * cols);
	float* const actual_f32 = ccmalloc(sizeof(float) * selected * cols);
	ccv_half_precision_to_float((uint16_t*)expected->data.f16, expected_f32, selected * cols);
	ccv_half_precision_to_float((uint16_t*)actual->data.f16, actual_f32, selected * cols);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, selected * cols, 1e-3, "MPS index select should dequantize only selected half row-wise 8i rows");
	ccfree(actual_f32);
	ccfree(expected_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	ccfree(values);
}

TEST_CASE("mps index select row-wise 8i quantized tensor with bfloat output")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 19;
	const int cols = 36;
	const int selected = 7;
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = ((i * 29) % 71 - 35) / 44.0f;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_float_to_bfloat(values, (uint16_t*)source->data.f16, rows * cols);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16BF, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16BF, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	int ip[] = {2, 18, 2, 6, 14, 0, 18};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16BF, CCV_TENSOR_CPU_MEMORY, qsize, cols, dequantized->data.f16, rows * cols);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, selected, cols), 0);
	for (i = 0; i < selected; i++)
		memcpy(expected->data.u8 + i * cols * sizeof(uint16_t), dequantized->data.u8 + ip[i] * cols * sizeof(uint16_t), cols * sizeof(uint16_t));
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16BF, rows, cols)), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = ccmalloc(sizeof(float) * selected * cols);
	float* const actual_f32 = ccmalloc(sizeof(float) * selected * cols);
	ccv_bfloat_to_float((uint16_t*)expected->data.f16, expected_f32, selected * cols);
	ccv_bfloat_to_float((uint16_t*)actual->data.f16, actual_f32, selected * cols);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, selected * cols, 2e-2, "MPS index select should dequantize only selected bfloat row-wise 8i rows");
	ccfree(actual_f32);
	ccfree(expected_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	ccfree(values);
}

TEST_CASE("mps index select packed row-wise 8i-x half precision all formats")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
	};
	int i;
	for (i = 0; i < (int)(sizeof(formats) / sizeof(formats[0])); i++)
	{
		double max_abs = 0;
		const int status = _mps_index_select_8i_rowwise_x_compare(CCV_16F, formats[i], 17, 130, 11, 1e-3, &max_abs);
		REQUIRE_EQ(status, 0, "MPS packed row-wise 8i-x fp16 index select should match CPU for format=%d, max_abs=%g", formats[i], max_abs);
	}
}

TEST_CASE("mps index select packed row-wise 8i-x float and bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
	};
	int i;
	for (i = 0; i < (int)(sizeof(formats) / sizeof(formats[0])); i++)
	{
		double max_abs = 0;
		int status = _mps_index_select_8i_rowwise_x_compare(CCV_32F, formats[i], 13, 131, 7, 1e-5, &max_abs);
		REQUIRE_EQ(status, 0, "MPS packed row-wise 8i-x fp32 index select should match CPU for format=%d, max_abs=%g", formats[i], max_abs);
		max_abs = 0;
		status = _mps_index_select_8i_rowwise_x_compare(CCV_16BF, formats[i], 13, 131, 7, 1e-2, &max_abs);
		REQUIRE_EQ(status, 0, "MPS packed row-wise 8i-x bf16 index select should match CPU for format=%d, max_abs=%g", formats[i], max_abs);
	}
}

TEST_CASE("index select backward with half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	int i;
	for (i = 0; i < 10 * 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10), 0);
	for (i = 0; i < 10; i++)
		indices->data.i32[i] = i * 9 + 1;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 10), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, 0, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, 0, indices), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_t* const b32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100, 10), 0);
	ccv_nnc_tensor_t* const bt32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(b32, bt32), 0);
	REQUIRE_TENSOR_EQ(b32, bt32, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(b32);
	ccv_nnc_tensor_free(bt32);
}

#include "case_main.h"
