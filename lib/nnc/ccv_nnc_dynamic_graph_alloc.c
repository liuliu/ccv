#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "ccv_internal.h"
#include "_ccv_nnc_dynamic_graph.h"

void ccv_nnc_dynamic_graph_gc(ccv_nnc_dynamic_graph_t* const graph)
{
	ccv_nnc_xpu_gc(-1, &graph->xpu_alloc);
}

ccv_nnc_compilation_artifact_t* ccv_nnc_compilation_artifact_new(ccv_nnc_graph_t* const graph, ccv_nnc_tensor_arena_t* const tensor_arena, ccv_nnc_graph_exec_arena_t* const exec_arena)
{
	ccv_nnc_compilation_artifact_t* const artifact = (ccv_nnc_compilation_artifact_t*)ccmalloc(sizeof(ccv_nnc_compilation_artifact_t));
	artifact->graph = graph;
	artifact->tensor_arena = tensor_arena;
	artifact->exec_arena = exec_arena;
	return artifact;
}

void ccv_nnc_compilation_artifact_free(ccv_nnc_compilation_artifact_t* const artifact)
{
	ccv_nnc_graph_free(artifact->graph);
	ccv_nnc_tensor_arena_free(artifact->tensor_arena);
	ccv_nnc_graph_exec_arena_free(artifact->exec_arena);
	ccfree(artifact);
}
