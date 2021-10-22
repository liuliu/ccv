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

TEST_CASE("lamb in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAMB_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
}

TEST_CASE("lamb in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAMB_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const m16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const v16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const n16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const u16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(g16, a16, m16, v16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a16, m16, v16), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gbt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gnt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gut16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt16, gnt16, gut16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16, gut16), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, m16, v16), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt16, gnt16, gut16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16, gut16), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(m16);
	ccv_nnc_tensor_free(v16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(n16);
	ccv_nnc_tensor_free(u16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
	ccv_nnc_tensor_free(gbt16);
	ccv_nnc_tensor_free(gnt16);
	ccv_nnc_tensor_free(gut16);
}

TEST_CASE("lamb in mixed precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAMB_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(g16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a, m, v), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_LAMB_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
}

#include "case_main.h"
