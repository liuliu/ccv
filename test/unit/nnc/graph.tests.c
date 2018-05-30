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

static int _ccv_nnc_custom_24_loss_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	assert(input_size == 1);
	const ccv_nnc_tensor_t* m = inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_t* g = outputs[0];
	for (i = 0; i < 21 * 31 * 4; i++)
		g->data.f32[i] = m->data.f32[i] - (i == 24);
	return CCV_NNC_EXEC_SUCCESS;
}

TEST_CASE("run simple graph network")
{
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 4), 0);
	ccv_nnc_cmd_t forw_cmd = CMD_CONVOLUTION_FORWARD(1, 4, 5, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(forw_cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 5, 3, 2), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < 2 * 3 * 5 * 4; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 1.41421356237 / sqrtf(21 * 31 * 2 + 21 * 31 * 4);
	float denom = (21 * 31 * 2 - 1) * 21 * 31 * 2;
	for (i = 0; i < 21 * 31 * 2; i++)
		a->data.f32[i] = (float)(i - 21 * 31) / denom;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_graph_exec_t forw_node = ccv_nnc_graph_exec_new(graph, forw_cmd, hint, TENSOR_LIST(a, w, bias), TENSOR_LIST(b));
	ccv_nnc_cmd_t softmax_cmd = CMD_SOFTMAX_FORWARD();
	ccv_nnc_tensor_t* m = ccv_nnc_tensor_new(0, b->info, 0);
	ccv_nnc_graph_exec_t softmax_node = ccv_nnc_graph_exec_new(graph, softmax_cmd, hint, TENSOR_LIST(b), TENSOR_LIST(m));
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, b->info, 0);
	ccv_nnc_cmd_t loss_cmd = CMD_CUSTOM_FORWARD(_ccv_nnc_custom_24_loss_exec);
	ccv_nnc_graph_exec_t loss_node = ccv_nnc_graph_exec_new(graph, loss_cmd, hint, TENSOR_LIST(m), TENSOR_LIST(g));
	ccv_nnc_cmd_t back_cmd = CMD_CONVOLUTION_BACKWARD(1, 4, 2, 3, 5);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, w->info, 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, bias->info, 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, a->info, 0);
	ccv_nnc_graph_exec_t back_node = ccv_nnc_graph_exec_new(graph, back_cmd, hint, TENSOR_LIST(g, a, w), TENSOR_LIST(h, gw, gbias));
	// All nodes are created, now to concat the graph.
	ccv_nnc_graph_exec_concat(graph, forw_node, softmax_node);
	ccv_nnc_graph_exec_concat(graph, softmax_node, loss_node);
	ccv_nnc_graph_exec_concat(graph, loss_node, back_node);
	ccv_nnc_graph_exec_t source_nodes[] = {
		forw_node,
	};
	ccv_nnc_graph_exec_t destination_nodes[] = {
		back_node,
	};
	ccv_nnc_graph_run(graph, 0, 0, source_nodes, 1, destination_nodes, 1);
	ccv_nnc_graph_free(graph);
	/* At this point, do the computation with a different set of tensors and then compare */
	ccv_nnc_tensor_t* vb = ccv_nnc_tensor_new(0, b->info, 0);
	ccv_nnc_cmd_exec(forw_cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(vb), 0);
	REQUIRE_TENSOR_EQ(b, vb, "Graph computed forward pass result should be the same.");
	ccv_nnc_tensor_t* vm = ccv_nnc_tensor_new(0, b->info, 0);
	ccv_nnc_cmd_exec(softmax_cmd, hint, 0, TENSOR_LIST(vb), TENSOR_LIST(vm), 0);
	REQUIRE_TENSOR_EQ(m, vm, "Graph computed softmax pass result should be the same.");
	ccv_nnc_tensor_t* vg = ccv_nnc_tensor_new(0, g->info, 0);
	for (i = 0; i < 21 * 31 * 4; i++)
		vg->data.f32[i] = vm->data.f32[i] - (i == 24);
	REQUIRE_TENSOR_EQ(g, vg, "Graph computed custom loss result should be the same.");
	ccv_nnc_tensor_t* vgw = ccv_nnc_tensor_new(0, w->info, 0);
	ccv_nnc_tensor_t* vgbias = ccv_nnc_tensor_new(0, bias->info, 0);
	ccv_nnc_tensor_t* vh = ccv_nnc_tensor_new(0, h->info, 0);
	ccv_nnc_cmd_exec(back_cmd, hint, 0, TENSOR_LIST(vg, a, w), TENSOR_LIST(vh, vgw, vgbias), 0);
	REQUIRE_TENSOR_EQ(gbias, vgbias, "Graph computed backward pass weight delta should be the same.");
	REQUIRE_TENSOR_EQ(gw, vgw, "Graph computed backward pass bias delta should be the same.");
	REQUIRE_TENSOR_EQ(h, vh, "Graph computed backward pass result should be the same.");
	// free all the tensor data.
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(vb);
	ccv_nnc_tensor_free(vm);
	ccv_nnc_tensor_free(vg);
	ccv_nnc_tensor_free(vh);
	ccv_nnc_tensor_free(vgw);
	ccv_nnc_tensor_free(vgbias);
}

#include "case_main.h"
