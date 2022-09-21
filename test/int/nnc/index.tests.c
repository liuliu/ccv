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

TEST_CASE("index select a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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

TEST_CASE("index select a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
