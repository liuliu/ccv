#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif

struct ccv_nnc_stream_context_s {
	int type;
	// Left for implementation yet, the CPU support for stream context.
	size_t workspace_size;
	void* workspace;
};

ccv_nnc_stream_context_t* ccv_nnc_stream_context_new(const int type)
{
	ccv_nnc_stream_context_t* stream_context = (ccv_nnc_stream_context_t*)ccmalloc(sizeof(ccv_nnc_stream_context_t));
	stream_context->type = type;
	stream_context->workspace_size = 0;
	stream_context->workspace = 0;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		stream_context = ccv_nnc_init_stream_context(stream_context);
#endif
	return stream_context;
}

#ifndef HAVE_CUDA
static __thread ccv_nnc_stream_context_t ccv_nnc_per_thread_cpu_stream_context = {
	.type = CCV_STREAM_CONTEXT_CPU,
};
#endif

void* ccv_nnc_stream_context_get_workspace(ccv_nnc_stream_context_t* stream_context, const size_t workspace_size, const int mem)
{
#ifdef HAVE_CUDA
	return ccv_nnc_stream_compat_get_workspace(stream_context, workspace_size, mem);
#else
	if (!stream_context)
		stream_context = &ccv_nnc_per_thread_cpu_stream_context;
	assert(mem == CCV_TENSOR_CPU_MEMORY);
	if (stream_context->workspace_size >= workspace_size)
		return stream_context->workspace;
	stream_context->workspace_size = workspace_size;
	if (stream_context->workspace)
		ccfree(stream_context->workspace);
	stream_context->workspace = 0;
	ccmemalign(&stream_context->workspace, 16, workspace_size);
	return stream_context->workspace;
#endif
}

void ccv_nnc_stream_context_drain(ccv_nnc_stream_context_t* stream_context)
{
#ifdef HAVE_CUDA
	ccv_nnc_stream_compat_drain(stream_context);
#else
	if (!stream_context)
		stream_context = &ccv_nnc_per_thread_cpu_stream_context;
	if (stream_context->workspace)
	{
		ccfree(stream_context->workspace);
		stream_context->workspace = 0;
		stream_context->workspace_size = 0;
	}
#endif
}

void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_synchronize_stream_context(stream_context);
#endif
}

void ccv_nnc_stream_context_free(ccv_nnc_stream_context_t* const stream_context)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_deinit_stream_context(stream_context);
#else
	if (stream_context->workspace)
		ccfree(stream_context->workspace);
#endif
	ccfree(stream_context);
}
