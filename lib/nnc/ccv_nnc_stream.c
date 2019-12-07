#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "co.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_stream.h"

typedef struct {
	ccv_nnc_stream_context_t super;
	// Left for implementation yet, the CPU support for stream context.
	size_t workspace_size;
	void* workspace;
} ccv_nnc_stream_cpu_t;

ccv_nnc_stream_context_t* ccv_nnc_stream_context_new(const int type)
{
	ccv_nnc_stream_cpu_t* const stream_cpu = (ccv_nnc_stream_cpu_t*)cccalloc(1, sizeof(ccv_nnc_stream_cpu_t));
	stream_cpu->super.type = type;
	stream_cpu->super.signal_container = kh_init(signal_container);
	stream_cpu->workspace_size = 0;
	stream_cpu->workspace = 0;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		return ccv_nnc_init_stream_context((ccv_nnc_stream_context_t*)stream_cpu);
#endif
	return (ccv_nnc_stream_context_t*)stream_cpu;
}

CCV_WARN_UNUSED(int) ccv_nnc_stream_context_type(const ccv_nnc_stream_context_t* const stream_context)
{
	return stream_context->type;
}

#ifndef HAVE_CUDA
static __thread ccv_nnc_stream_cpu_t ccv_nnc_per_thread_stream_cpu = {
	.super = {
		.type = CCV_STREAM_CONTEXT_CPU,
	},
};
#endif

void* ccv_nnc_stream_context_get_workspace(ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem)
{
#ifdef HAVE_CUDA
	return ccv_nnc_stream_compat_get_workspace(stream_context, workspace_size, mem);
#else
	ccv_nnc_stream_cpu_t* stream_cpu = (ccv_nnc_stream_cpu_t*)stream_context;
	if (!stream_cpu)
		stream_cpu = &ccv_nnc_per_thread_stream_cpu;
	assert(mem == CCV_TENSOR_CPU_MEMORY);
	if (stream_cpu->workspace_size >= workspace_size)
		return stream_cpu->workspace;
	stream_cpu->workspace_size = workspace_size;
	if (stream_cpu->workspace)
		ccfree(stream_cpu->workspace);
	stream_cpu->workspace = 0;
	ccmemalign(&stream_cpu->workspace, 16, workspace_size);
	return stream_cpu->workspace;
#endif
}

void ccv_nnc_stream_context_drain(ccv_nnc_stream_context_t* const stream_context)
{
#ifdef HAVE_CUDA
	ccv_nnc_stream_compat_drain(stream_context);
#else
	ccv_nnc_stream_cpu_t* stream_cpu = (ccv_nnc_stream_cpu_t*)stream_context;
	if (!stream_cpu)
		stream_cpu = &ccv_nnc_per_thread_stream_cpu;
	if (stream_cpu->workspace)
	{
		ccfree(stream_cpu->workspace);
		stream_cpu->workspace = 0;
		stream_cpu->workspace_size = 0;
	}
#endif
}

void ccv_nnc_stream_context_add_callback(ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_stream_context_callback_f callback, void* const callback_context)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_stream_compat_add_callback(stream_context, callback, callback_context);
	else
		callback(stream_context, callback_context);
#else
	callback(stream_context, callback_context);
#endif
}

int ccv_nnc_stream_context_try_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return 0;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_synchronize_stream_context(stream_context);
#endif
	co_scheduler_t* const scheduler = stream_context->scheduler;
	return scheduler ? -1 : 0;
}

void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return;
	co_scheduler_t* const scheduler = stream_context->scheduler;
	if (scheduler) // First wait the scheduler to finish.
	{
		pthread_mutex_lock(&scheduler->mutex);
		while (scheduler->active)
			pthread_cond_wait(&scheduler->notify, &scheduler->mutex);
		pthread_mutex_unlock(&scheduler->mutex);
	}
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
	ccv_nnc_stream_cpu_t* stream_cpu = (ccv_nnc_stream_cpu_t*)stream_context;
	if (stream_cpu->workspace)
		ccfree(stream_cpu->workspace);
#endif
	if (stream_context->scheduler)
	{
		co_scheduler_t* const scheduler = stream_context->scheduler;
		co_scheduler_free(scheduler);
	}
	khash_t(signal_container)* const signal_container = stream_context->signal_container;
	khiter_t k;
	for (k = kh_begin(signal_container); k != kh_end(signal_container); ++k)
	{
		if (!kh_exist(signal_container, k))
			continue;
		ccv_nnc_stream_signal_t* const signal = kh_val(signal_container, k);
		ccv_nnc_stream_signal_free(signal);
	}
	kh_destroy(signal_container, signal_container);
	ccfree(stream_context);
}

void ccv_nnc_stream_context_set_neighbor_discovery(ccv_nnc_stream_context_t* const stream_context, ccv_nnc_stream_context_neighbor_discovery_f discovery, void* const context)
{
	stream_context->neighbor_discovery = discovery;
	stream_context->neighbor_discovery_context = context;
}

ccv_nnc_stream_context_t* ccv_nnc_stream_context_find_neighbor(ccv_nnc_stream_context_t* const stream_context, const int device_id)
{
	if (stream_context->neighbor_discovery)
		return stream_context->neighbor_discovery(device_id, stream_context->neighbor_discovery_context);
	return 0;
}

ccv_nnc_stream_signal_t* ccv_nnc_stream_signal_new(const int type)
{
	ccv_nnc_stream_signal_t* const signal = (ccv_nnc_stream_signal_t*)ccmalloc(sizeof(ccv_nnc_stream_signal_t));
	signal->type = type;
	signal->emit_context = 0;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		return ccv_nnc_init_stream_signal(signal);
#endif
	return signal;
}

CCV_WARN_UNUSED(int) ccv_nnc_stream_signal_type(const ccv_nnc_stream_signal_t* const signal)
{
	return signal->type;
}

void ccv_nnc_stream_context_emit_signal(ccv_nnc_stream_context_t* const stream, ccv_nnc_stream_signal_t* const signal)
{
	signal->emit_context = stream;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(signal->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_stream_compat_emit_signal(stream, signal);
#endif
}

ccv_nnc_stream_context_t* ccv_nnc_stream_signal_get_emitter(const ccv_nnc_stream_signal_t* const signal)
{
	return signal->emit_context;
}

void ccv_nnc_stream_context_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(signal->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_stream_compat_wait_signal(stream, signal);
#endif
}

ccv_nnc_stream_signal_t* ccv_nnc_stream_context_get_signal(ccv_nnc_stream_context_t* const stream, const int64_t identifier)
{
	khash_t(signal_container)* const signal_container = stream->signal_container;
	int ret = 0;
	khiter_t k = kh_put(signal_container, signal_container, identifier, &ret);
	assert(ret >= 0);
	// If ret == 0, the key already exist, we can get the columns directly, otherwise, create and assign back.
	ccv_nnc_stream_signal_t* const signal = (ret == 0) ? kh_val(signal_container, k) : ccv_nnc_stream_signal_new(stream->type);
	if (ret != 0)
		kh_val(signal_container, k) = signal;
	return signal;
}

void ccv_nnc_stream_signal_free(ccv_nnc_stream_signal_t* const signal)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(signal->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_deinit_stream_signal(signal);
#endif
	ccfree(signal);
}

int ccv_nnc_device_count(const int type)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		return ccv_nnc_gpu_device_count();
#endif
	return 1; // I don't get core count for CPU yet.
}

co_scheduler_t* ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context)
{
	co_scheduler_t* scheduler = stream_context->scheduler;
	if (!scheduler)
		stream_context->scheduler = scheduler = co_scheduler_new();
	return scheduler;
}

int _co_stream_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream)
{
	if (!stream)
		return 1;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream->type) == CCV_STREAM_CONTEXT_GPU)
		return co_stream_compat_await(self, stream);
#endif
	return 1;
}
