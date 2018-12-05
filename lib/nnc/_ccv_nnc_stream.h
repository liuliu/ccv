/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_stream_internal_h
#define GUARD_ccv_nnc_stream_internal_h

#include "ccv_nnc.h"

#include <ucontext.h>
#include <pthread.h>

struct ccv_nnc_stream_signal_s {
	int type;
	ccv_nnc_stream_context_t* emit_context;
};

typedef struct ccv_nnc_stream_task_s ccv_nnc_stream_task_t;

typedef void (*ccv_nnc_stream_task_f)(ccv_nnc_stream_task_t* const self, void* const userdata);

#define CCV_NNC_TASK_STACK_SIZE (262144) // 256KiB

// A stream can associate at most one scheduler.
// It manages how all the tasks get scheduled.
typedef struct {
	int active;
	int stream_wait_task_count;
	ccv_array_t* empty_tasks;
	ccv_nnc_stream_task_t* head;
	ccv_nnc_stream_task_t* tail;
	pthread_t thread;
	pthread_cond_t notify;
	pthread_cond_t wait;
	pthread_mutex_t mutex;
	ucontext_t caller, callee;
} ccv_nnc_stream_scheduler_t;

// A stream can have multiple tasks.
struct ccv_nnc_stream_task_s {
	int done;
	int other_size;
	ccv_nnc_stream_task_t* prev;
	ccv_nnc_stream_task_t* next;
	ccv_nnc_stream_scheduler_t* super;
	ccv_nnc_stream_task_t* notify;
	ccv_nnc_stream_task_t* const* others;
	char* stack;
	ucontext_t context;
	void* userdata;
	ccv_nnc_stream_task_f func;
};

struct ccv_nnc_stream_context_s {
	int type;
	// For scheduler
	ccv_nnc_stream_task_t* main; // main task.
	ccv_nnc_stream_scheduler_t* scheduler;
	// For neighbor discovery
	ccv_nnc_stream_context_neighbor_discovery_f neighbor_discovery;
	void* neighbor_discovery_context;
};

// Return the scheduler from a stream (if not created, create one).
CCV_WARN_UNUSED(ccv_nnc_stream_scheduler_t*) ccv_nnc_stream_context_get_scheduler(ccv_nnc_stream_context_t* const stream_context);
// This method activates the scheduler (if necessary), and runs the given task.
void ccv_nnc_stream_schedule_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task);
// Add a task to the beginning of the list of tasks scheduler going to execute.
void ccv_nnc_stream_scheduler_prepend_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task);
// Add a task to the end of the list of tasks scheduler going to execute.
void ccv_nnc_stream_scheduler_append_task(ccv_nnc_stream_scheduler_t* const scheduler, ccv_nnc_stream_task_t* const task);
// Create a task off a stream. If userdata_size is non-zero, we copied it over.
CCV_WARN_UNUSED(ccv_nnc_stream_task_t*) ccv_nnc_stream_task_new(ccv_nnc_stream_scheduler_t* const scheduler, const ccv_nnc_stream_task_f func, void* const userdata, const size_t userdata_size);
// Run a given task immediately from within an existing task.
void ccv_nnc_stream_task_resume(ccv_nnc_stream_task_t* const task);
// Set a point on the stream, and wait until that point is reached to continue execution.
void ccv_nnc_stream_task_synchronize(ccv_nnc_stream_task_t* const self, ccv_nnc_stream_context_t* const stream);
// Wait any other tasks to finish. Since we don't have yield, this means wait until these tasks to finish.
void ccv_nnc_stream_task_wait_any(ccv_nnc_stream_task_t* const self, ccv_nnc_stream_task_t* const* const others, const int other_size);

#endif
