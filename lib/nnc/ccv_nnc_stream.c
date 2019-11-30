#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_stream.h"
#include "3rdparty/valgrind/valgrind.h"

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

int ccv_nnc_stream_context_try_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return 0;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_synchronize_stream_context(stream_context);
#endif
	ccv_nnc_stream_scheduler_t* const scheduler = stream_context->scheduler;
	return scheduler ? -1 : 0;
}

void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return;
	ccv_nnc_stream_scheduler_t* const scheduler = stream_context->scheduler;
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
		ccv_nnc_stream_scheduler_t* const scheduler = stream_context->scheduler;
		pthread_mutex_destroy(&scheduler->mutex);
		pthread_cond_destroy(&scheduler->notify);
		pthread_cond_destroy(&scheduler->wait);
		if (scheduler->empty_tasks)
		{
			int i;
			for (i = 0; i < scheduler->empty_tasks->rnum; i++)
			{
				ccv_nnc_stream_task_t* const task = *(ccv_nnc_stream_task_t**)ccv_array_get(scheduler->empty_tasks, i);
				ccfree(task->stack);
				ccfree(task);
			}
			ccv_array_free(scheduler->empty_tasks);
		}
		ccfree(scheduler);
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

ccv_nnc_stream_scheduler_t* ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_scheduler_t* scheduler = stream_context->scheduler;
	if (!scheduler)
	{
		stream_context->scheduler = scheduler = (ccv_nnc_stream_scheduler_t*)cccalloc(1, sizeof(ccv_nnc_stream_scheduler_t));
		pthread_mutex_init(&scheduler->mutex, 0);
		pthread_cond_init(&scheduler->notify, 0);
		pthread_cond_init(&scheduler->wait, 0);
	}
	return scheduler;
}

void ccv_nnc_stream_scheduler_prepend_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task)
{
	if (scheduler->head)
	{
		scheduler->head->prev = task;
		task->next = scheduler->head;
	} else {
		scheduler->tail = task;
		task->next = 0;
	}
	scheduler->head = task;
	task->prev = 0;
}

void ccv_nnc_stream_scheduler_append_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task)
{
	if (scheduler->tail)
	{
		scheduler->tail->next = task;
		task->prev = scheduler->tail;
	} else {
		scheduler->head = task;
		task->prev = 0;
	}
	scheduler->tail = task;
	task->next = 0;
}

static void _ccv_nnc_stream_scheduler_delete_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task)
{
	if (task->prev)
		task->prev->next = task->next;
	else
		scheduler->head = task->next;
	if (task->next)
		task->next->prev = task->prev;
	else
		scheduler->tail = task->prev;
}

static void _ccv_nnc_stream_task_done(ccv_nnc_stream_task_t* const task)
{
	if (task->notify)
	{
		ccv_nnc_stream_task_t* const notify = task->notify;
		task->notify = 0;
		ccv_nnc_stream_scheduler_prepend_task(task->super, notify);
		int i;
		const int other_size = notify->other_size;
		notify->other_size = 0;
		ccv_nnc_stream_task_t* const* const others = notify->others;
		notify->others = 0;
		for (i = 0; i < other_size; i++)
			if (others[i] != task)
			{
				assert(others[i]->notify == notify);
				others[i]->notify = 0;
			}
	}
	ccv_nnc_stream_scheduler_t* const scheduler = task->super;
	if (!scheduler->empty_tasks)
		scheduler->empty_tasks = ccv_array_new(sizeof(ccv_nnc_stream_task_t*), 1, 0);
	ccv_array_push(scheduler->empty_tasks, &task);
}

// Second will invoke this blocking variant to schedule task on a newly created thread.
static void* _ccv_nnc_stream_schedule_main(void* userdata)
{
	ccv_nnc_stream_scheduler_t* const scheduler = (ccv_nnc_stream_scheduler_t*)userdata;
	pthread_mutex_lock(&scheduler->mutex);
	for (;;)
	{
		if (scheduler->head == 0 && scheduler->stream_wait_task_count == 0)
		{
			scheduler->active = 0;
			pthread_cond_broadcast(&scheduler->notify);
			pthread_mutex_unlock(&scheduler->mutex);
			break;
		}
		if (scheduler->head == 0)
		{
			pthread_cond_wait(&scheduler->wait, &scheduler->mutex);
			pthread_mutex_unlock(&scheduler->mutex);
		}
		ccv_nnc_stream_task_t* const task = scheduler->head;
		_ccv_nnc_stream_scheduler_delete_task(scheduler, task);
		pthread_mutex_unlock(&scheduler->mutex);
		swapcontext(&scheduler->caller, &task->context);
		task->context = scheduler->callee;
		pthread_mutex_lock(&scheduler->mutex);
		if (task->done)
			_ccv_nnc_stream_task_done(task);
	}
	return 0;
}

// First will invoke this non-blocking variant to schedule task.
static void _ccv_nnc_stream_schedule_try(ccv_nnc_stream_scheduler_t* const scheduler)
{
	pthread_mutex_lock(&scheduler->mutex);
	if (scheduler->active)
	{
		pthread_mutex_unlock(&scheduler->mutex);
		return;
	}
	scheduler->active = 1;
	for (;;)
	{
		if (scheduler->head == 0 && scheduler->stream_wait_task_count == 0)
		{
			scheduler->active = 0;
			pthread_mutex_unlock(&scheduler->mutex);
			break;
		}
		if (scheduler->head == 0)
		{
			// Launch a thread to continue the execution.
			pthread_create(&scheduler->thread, 0, _ccv_nnc_stream_schedule_main, scheduler);
			pthread_mutex_unlock(&scheduler->mutex);
			break;
		}
		ccv_nnc_stream_task_t* const task = scheduler->head;
		_ccv_nnc_stream_scheduler_delete_task(scheduler, task);
		pthread_mutex_unlock(&scheduler->mutex);
		swapcontext(&scheduler->caller, &task->context);
		task->context = scheduler->callee;
		pthread_mutex_lock(&scheduler->mutex);
		if (task->done)
			_ccv_nnc_stream_task_done(task);
	}
}

void ccv_nnc_stream_schedule_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task)
{
	int activate_scheduler = 0;
	pthread_mutex_lock(&scheduler->mutex);
	// Append to the end, for swap tasks, they all prepend. Thus, this ensures all tasks scheduled this way will be executed later.
	ccv_nnc_stream_scheduler_append_task(scheduler, task);
	if (!scheduler->active)
		activate_scheduler = 1;
	pthread_mutex_unlock(&scheduler->mutex);
	if (activate_scheduler)
		_ccv_nnc_stream_schedule_try(scheduler);
}

typedef union {
	void* ptr;
	uint32_t part[2];
} ccv_nnc_ptr_splitter_u;

static void _ccv_nnc_stream_task_entry_point(uint32_t part0, uint32_t part1)
{
	const ccv_nnc_ptr_splitter_u p = {
		.part = {
			part0, part1
		}
	};
	ccv_nnc_stream_task_t* const task = (ccv_nnc_stream_task_t*)p.ptr;
	task->func(task, task->userdata);
	ccv_nnc_stream_scheduler_t* const scheduler = task->super;
	task->done = 1;
	swapcontext(&scheduler->callee, &scheduler->caller);
}

ccv_nnc_stream_task_t* ccv_nnc_stream_task_new(ccv_nnc_stream_scheduler_t* const scheduler, const ccv_nnc_stream_task_f func, void* const userdata, const size_t userdata_size)
{
	ccv_nnc_stream_task_t* task;
	pthread_mutex_lock(&scheduler->mutex);
	if (scheduler->empty_tasks && scheduler->empty_tasks->rnum)
	{
		task = *(ccv_nnc_stream_task_t**)ccv_array_get(scheduler->empty_tasks, scheduler->empty_tasks->rnum - 1);
		--scheduler->empty_tasks->rnum;
		pthread_mutex_unlock(&scheduler->mutex);
		if (userdata_size)
			task->stack = (char*)ccrealloc(task->stack, CCV_NNC_TASK_STACK_SIZE + userdata_size);
	} else {
		pthread_mutex_unlock(&scheduler->mutex);
		task = (ccv_nnc_stream_task_t*)cccalloc(1, sizeof(ccv_nnc_stream_task_t));
		task->stack = (char*)cccalloc(CCV_NNC_TASK_STACK_SIZE + userdata_size, 1);
		task->super = scheduler;
	}
	task->done = 0;
	task->func = func;
	if (userdata_size)
	{
		// If the size is available, we copy the userdata over.
		task->userdata = task->stack + CCV_NNC_TASK_STACK_SIZE;
		memcpy(task->userdata, userdata, userdata_size);
	} else
		task->userdata = userdata;
	getcontext(&task->context);
	task->context.uc_stack.ss_sp = task->stack;
	task->context.uc_stack.ss_size = CCV_NNC_TASK_STACK_SIZE;
	VALGRIND_STACK_REGISTER(task->stack, task->stack + CCV_NNC_TASK_STACK_SIZE);
	task->context.uc_link = 0;
	const ccv_nnc_ptr_splitter_u p = {
		.ptr = task,
	};
	makecontext(&task->context, (void (*)(void))_ccv_nnc_stream_task_entry_point, 2, p.part[0], p.part[1]);;
	return task;
}

void ccv_nnc_stream_task_resume(ccv_nnc_stream_task_t* const task)
{
	ccv_nnc_stream_scheduler_t* const scheduler = task->super;
	ucontext_t old_context = scheduler->caller;
	swapcontext(&scheduler->caller, &task->context);
	task->context = scheduler->callee;
	scheduler->caller = old_context;
	if (task->done)
	{
		pthread_mutex_lock(&scheduler->mutex);
		_ccv_nnc_stream_task_done(task);
		pthread_mutex_unlock(&scheduler->mutex);
	}
}

void ccv_nnc_stream_task_synchronize(ccv_nnc_stream_task_t* const self, ccv_nnc_stream_context_t* const stream)
{
	if (!stream)
		return;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_stream_compat_task_synchronize(self, stream);
#endif
}

void ccv_nnc_stream_task_wait_any(ccv_nnc_stream_task_t* const self, ccv_nnc_stream_task_t* const* const others, const int other_size)
{
	self->other_size = other_size;
	self->others = others;
	int i;
	for (i = 0; i < other_size; i++)
	{
		assert(others[i]->notify == 0);
		others[i]->notify = self;
	}
	ccv_nnc_stream_scheduler_t* const scheduler = self->super;
	swapcontext(&scheduler->callee, &scheduler->caller);
}
