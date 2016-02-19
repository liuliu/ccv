#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "nnc/ccv_nnc.h"
#include "3rdparty/dsfmt/dSFMT.h"

TEST_CASE("run simple graph network")
{
	ccv_nnc_init();
	ccv_nnc_graph_t* graph = ccv_nnc_graph_new();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			2, 21, 31,
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4, 21, 31,
		},
	};
	ccv_nnc_tensor_param_t h_params = a_params;
	ccv_nnc_tensor_param_t g_params = b_params;
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			2, 3, 5, 4,
		},
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4,
		},
	};
	ccv_nnc_cmd_param_t cmd_params = {
		.size = {
			.dim = {
				2, 3, 5,
			},
		},
		.convolutional = {
			.count = 4,
		},
	};
	ccv_nnc_hint_t hint = ccv_nnc_hint_guess(cmd_params, &a_params, 1, &b_params, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_cmd_t forw_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, cmd_params, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, bias_params, 0);
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
	ccv_nnc_tensor_t* forw_inlets[] = {
		a,
		w,
		bias,
	};
	ccv_nnc_tensor_t* forw_outlets[] = {
		b,
	};
	ccv_nnc_graph_node_t forw_node = ccv_nnc_graph_node(graph, forw_cmd, hint, 0, forw_inlets, 3, forw_outlets, 1);
	ccv_nnc_cmd_t softmax_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_SOFTMAX_FORWARD, cmd_params, 0);
	ccv_nnc_tensor_t* m = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* max_inlets[] = {
		b,
	};
	ccv_nnc_tensor_t* max_outlets[] = {
		m,
	};
	ccv_nnc_graph_node_t softmax_node = ccv_nnc_graph_node(graph, softmax_cmd, hint, 0, max_inlets, 1, max_outlets, 1);
	ccv_nnc_cmd_t back_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD, cmd_params, 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, g_params, 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, h_params, 0);
	for (i = 0; i < 21 * 31 * 4; i++)
		g->data.f32[i] = m->data.f32[i] - (i == 24);
	ccv_nnc_tensor_t* back_inlets[] = {
		g,
		a,
		w,
	};
	ccv_nnc_tensor_t* back_outlets[] = {
		gw,
		gbias,
		h,
	};
	ccv_nnc_graph_node_t back_node = ccv_nnc_graph_node(graph, back_cmd, hint, 0, back_inlets, 3, back_outlets, 3);
	ccv_nnc_graph_node_concat(graph, forw_node, softmax_node);
	ccv_nnc_graph_node_concat(graph, softmax_node, back_node);
	ccv_nnc_graph_node_t source_nodes[] = {
		forw_node,
	};
	ccv_nnc_graph_node_t destination_nodes[] = {
		back_node,
	};
	ccv_nnc_graph_run(graph, 0, source_nodes, 1, destination_nodes, 1);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

#include "case_main.h"
