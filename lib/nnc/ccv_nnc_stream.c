#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include "co.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#include "_ccv_nnc_stream.h"

typedef struct {
	ccv_nnc_stream_context_t super;
	// Left for implementation yet, the CPU support for stream context.
	size_t workspace_size;
	void* workspace;
} ccv_nnc_stream_cpu_t;

typedef struct {
	ccv_nnc_stream_context_destructor_f destructor_hook;
	void* context;
} ccv_nnc_stream_destructor_hook_t;

ccv_nnc_stream_context_t* ccv_nnc_stream_context_new(const int type)
{
	ccv_nnc_stream_cpu_t* const stream_cpu = (ccv_nnc_stream_cpu_t*)cccalloc(1, sizeof(ccv_nnc_stream_cpu_t));
	stream_cpu->super.type = type;
	stream_cpu->super.reuse_destructor_hook = -1;
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

static void _ccv_nnc_stream_context_add_callback(ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_stream_compat_add_callback(stream_context, callback, async_callback, callback_context);
	else
		callback(callback_context);
#else
	callback(callback_context);
#endif
}

static void _ccv_nnc_sync_dispatch(ccv_nnc_async_callback_t* const async)
{
	async->fn(async->callback_context);
	ccfree(async);
}

#ifndef USE_DISPATCH
static void* _ccv_nnc_pthread_dispatch(void* const userdata)
{
	_ccv_nnc_sync_dispatch((ccv_nnc_async_callback_t*)userdata);
	return 0;
}
#endif

static void _ccv_nnc_async_dispatch(ccv_nnc_async_callback_t* const async)
{
	// This method dispatches to a different thread because the CUDA callback thread cannot operate CUDA objects.
#ifdef USE_DISPATCH
	dispatch_async_f(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), async, (dispatch_function_t)_ccv_nnc_sync_dispatch);
#else
	pthread_t thread;
	pthread_create(&thread, 0, _ccv_nnc_pthread_dispatch, async);
#endif
}

static co_decl_task(_ccv_nnc_stream_context_add_callback_async, (ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_callback_f callback, void* const callback_context), private())
{
	co_stream_await(CO_P(stream_context));
	_ccv_nnc_stream_context_add_callback(CO_P(stream_context), CO_P(callback), _ccv_nnc_async_dispatch, CO_P(callback_context));
} co_end()

void ccv_nnc_stream_context_add_callback(ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_callback_f callback, void* const callback_context)
{
	if (!stream_context)
	{
		callback(callback_context);
		return;
	}
	co_scheduler_t* const scheduler = stream_context->scheduler;
	if (scheduler && co_scheduler_is_active(scheduler))
	{
		co_routine_t* const task = co_new(_ccv_nnc_stream_context_add_callback_async, (stream_context, callback, callback_context));
		co_schedule(scheduler, task);
	} else
		_ccv_nnc_stream_context_add_callback(stream_context, callback, _ccv_nnc_async_dispatch, callback_context);
}

void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return;
	co_scheduler_t* const scheduler = stream_context->scheduler;
	if (scheduler && !co_is_on_scheduler(scheduler)) // First wait the scheduler to finish if I am not currently on that scheduler.
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

int ccv_nnc_stream_context_add_destructor_hook(ccv_nnc_stream_context_t* const stream, ccv_nnc_stream_context_destructor_f destructor, void* const context)
{
	ccv_nnc_stream_destructor_hook_t hook = {
		.destructor_hook = destructor,
		.context = context
	};
	if (stream->reuse_destructor_hook >= 0)
	{
		assert(stream->destructor_hooks);
		const int reuse_destructor_hook = stream->reuse_destructor_hook;
		assert(reuse_destructor_hook < stream->destructor_hooks->rnum);
		*(ccv_nnc_stream_destructor_hook_t*)ccv_array_get(stream->destructor_hooks, reuse_destructor_hook) = hook;
		int i;
		stream->reuse_destructor_hook = -1;
		for (i = reuse_destructor_hook + 1; i < stream->destructor_hooks->rnum && stream->reuse_destructor_hook < 0; i++)
			if (!((ccv_nnc_stream_destructor_hook_t*)ccv_array_get(stream->destructor_hooks, i))->destructor_hook)
				stream->reuse_destructor_hook = i;
		return reuse_destructor_hook;
	} else {
		if (!stream->destructor_hooks)
			stream->destructor_hooks = ccv_array_new(sizeof(ccv_nnc_stream_destructor_hook_t), 1, 0);
		ccv_array_push(stream->destructor_hooks, &hook);
		return stream->destructor_hooks->rnum - 1;
	}
}

void ccv_nnc_stream_context_remove_destructor_hook(ccv_nnc_stream_context_t* const stream, const int hook_id)
{
	assert(hook_id >= 0);
	assert(hook_id < stream->destructor_hooks->rnum);
	ccv_nnc_stream_destructor_hook_t* const hook = (ccv_nnc_stream_destructor_hook_t*)ccv_array_get(stream->destructor_hooks, hook_id);
	hook->destructor_hook = 0;
	hook->context = 0;
	int i;
	for (i = stream->destructor_hooks->rnum - 1; i >= 0; i--)
		if (((ccv_nnc_stream_destructor_hook_t*)ccv_array_get(stream->destructor_hooks, i))->destructor_hook)
		{
			stream->destructor_hooks->rnum = i + 1;
			break;
		}
	if (hook_id < stream->destructor_hooks->rnum &&
		(hook_id < stream->reuse_destructor_hook || stream->reuse_destructor_hook < 0))
		stream->reuse_destructor_hook = hook_id;
	else if (stream->reuse_destructor_hook >= stream->destructor_hooks->rnum)
		stream->reuse_destructor_hook = -1;
}

void ccv_nnc_stream_context_free(ccv_nnc_stream_context_t* const stream_context)
{
	if (stream_context->destructor_hooks)
	{
		int i;
		for (i = 0; i < stream_context->destructor_hooks->rnum; i++)
		{
			ccv_nnc_stream_destructor_hook_t* const hook = (ccv_nnc_stream_destructor_hook_t*)ccv_array_get(stream_context->destructor_hooks, i);
			if (hook->destructor_hook)
				hook->destructor_hook(stream_context, hook->context);
		}
		ccv_array_free(stream_context->destructor_hooks);
	}
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_deinit_stream_context(stream_context);
	else {
#endif
	ccv_nnc_stream_cpu_t* stream_cpu = (ccv_nnc_stream_cpu_t*)stream_context;
	if (stream_cpu->workspace)
		ccfree(stream_cpu->workspace);
#ifdef HAVE_CUDA
	}
#endif
	if (stream_context->scheduler)
	{
		co_scheduler_t* const scheduler = stream_context->scheduler;
		co_scheduler_free(scheduler);
	}
	if (stream_context->event)
		ccv_nnc_stream_signal_free(stream_context->event);
	if (stream_context->sfmt)
		ccfree(stream_context->sfmt);
	ccfree(stream_context);
}

static ccv_nnc_stream_cpu_t* _ccv_nnc_default_stream_cpu()
{
	static __thread ccv_nnc_stream_cpu_t ccv_nnc_per_thread_cpu_stream_context = {
		.super = {
			.type = CCV_STREAM_CONTEXT_CPU | CCV_COMPUTE_DEVICE_ANY,
			.reuse_destructor_hook = -1,
		},
	};
	return &ccv_nnc_per_thread_cpu_stream_context;
}

void ccv_nnc_stream_context_set_seed(ccv_nnc_stream_context_t* const stream_context, uint32_t seed)
{
	if (!stream_context)
	{
		ccv_nnc_stream_cpu_t* const stream_cpu = _ccv_nnc_default_stream_cpu();
		if (!stream_cpu->super.sfmt)
			stream_cpu->super.sfmt = ccmalloc(sizeof(sfmt_t));
		sfmt_init_gen_rand(stream_cpu->super.sfmt, seed);
		return;
	}
	if (!stream_context->sfmt)
		stream_context->sfmt = ccmalloc(sizeof(sfmt_t));
	sfmt_init_gen_rand(stream_context->sfmt, seed);
}

uint32_t ccv_nnc_stream_context_genrand_uint32(ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
	{
		ccv_nnc_stream_cpu_t* const stream_cpu = _ccv_nnc_default_stream_cpu();
		if (!stream_cpu->super.sfmt)
		{
			stream_cpu->super.sfmt = ccmalloc(sizeof(sfmt_t));
			sfmt_init_gen_rand(stream_cpu->super.sfmt, (uint32_t)(uintptr_t)stream_cpu);
		}
		return sfmt_genrand_uint32(stream_cpu->super.sfmt);
	}
	if (!stream_context->sfmt)
	{
		stream_context->sfmt = ccmalloc(sizeof(sfmt_t));
		// Init with seed from thread-local context.
		sfmt_init_gen_rand(stream_context->sfmt, ccv_nnc_stream_context_genrand_uint32(0));
	}
	return sfmt_genrand_uint32(stream_context->sfmt);
}

uint64_t ccv_nnc_stream_context_genrand_uint64(ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
	{
		ccv_nnc_stream_cpu_t* const stream_cpu = _ccv_nnc_default_stream_cpu();
		if (!stream_cpu->super.sfmt)
		{
			stream_cpu->super.sfmt = ccmalloc(sizeof(sfmt_t));
			sfmt_init_gen_rand(stream_cpu->super.sfmt, (uint32_t)(uintptr_t)stream_cpu);
		}
		return sfmt_genrand_uint64(stream_cpu->super.sfmt);
	}
	if (!stream_context->sfmt)
	{
		stream_context->sfmt = ccmalloc(sizeof(sfmt_t));
		// Init with seed from thread-local context.
		sfmt_init_gen_rand(stream_context->sfmt, ccv_nnc_stream_context_genrand_uint32(0));
	}
	return sfmt_genrand_uint64(stream_context->sfmt);
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
#else
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		return 0;
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

// MARK - Signal Container

ccv_nnc_stream_signal_t* ccv_nnc_stream_context_emit_signal_new(ccv_nnc_stream_context_t* const stream)
{
	/**
	 * We don't need complex containers for this. Based on CUDA documentation, Record will record the
	 * most recent ones, and capture will use the most recent ones. Thus, even if we reuse the same event
	 * again and again and again, as long as we emit and immediate wait, we won't have any problems.
	 */
	if (!stream->event)
		stream->event = ccv_nnc_stream_signal_new(ccv_nnc_stream_context_type(stream));
	ccv_nnc_stream_context_emit_signal(stream, stream->event);
	return stream->event;
}
